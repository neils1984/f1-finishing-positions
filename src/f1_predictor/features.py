"""Stage 3: engineer features per race from the Stage 2 sessionised table.

Pure, deterministic transforms producing raw (unscaled) human-readable values.
Scaling happens in Stage 4. Cross-race priors live in priors.py.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import yaml

from f1_predictor.priors import compute_priors, build_driver_races

FEATURE_COLUMNS = [
    "position", "positions_gained_from_grid", "num_active_drivers",
    "distance_remaining_km", "gap_to_leader", "interval_to_ahead",
    "rolling_lap_time_3_norm", "rolling_lap_time_3_delta_leader",
    "last_lap_pace_delta_to_ahead", "last_lap_pace_delta_to_behind",
    "mean_gap_cars_ahead", "stdev_gap_cars_ahead", "max_speed_kmh",
    "tyre_soft", "tyre_medium", "tyre_hard", "tyre_inter", "tyre_wet",
    "tyre_age_laps", "stint_number", "stops_vs_median",
    "sc_active", "vsc_active", "red_flag_active", "laps_since_sc_end",
    "is_street_circuit",
    "driver_circuit_finish_rate", "driver_championship_standing",
    "team_circuit_finish_rate", "team_championship_standing",
]

_KEY_COLUMNS = ["session_key", "driver_number", "lap_number", "final_position"]

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


def load_circuits(path: Path | None = None) -> dict:
    """Load the circuit reference (lengths + street-circuit list)."""
    path = path or (_CONFIG_DIR / "circuits.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def circuit_length_km(circuit_short_name: str, circuits: dict) -> float | None:
    """Track length in km for a circuit_short_name, or None if unknown."""
    return circuits.get("lengths_km", {}).get(circuit_short_name)


def is_street_circuit(circuit_short_name: str, circuits: dict) -> bool:
    """True if the circuit is a street circuit (Baku/Singapore/Las Vegas/Miami)."""
    return circuit_short_name in set(circuits.get("street", []))


_GAP_COLUMNS = ["gap_to_leader", "interval_to_ahead"]


def _parse_gap_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Coerce gap columns to Float64; non-numeric markers (e.g. '+1 LAP') -> null.

    pl.col(...).cast(Float64, strict=False) turns any value that doesn't parse as
    a number into null, which is exactly the desired behaviour for '+N LAP(S)',
    '' and existing nulls. Already-Float64 columns pass through unchanged.
    """
    exprs = []
    for col in _GAP_COLUMNS:
        if col in df.columns and df.schema[col] != pl.Float64:
            exprs.append(pl.col(col).cast(pl.Float64, strict=False).alias(col))
    return df.with_columns(exprs) if exprs else df


def _add_active_and_distance(
    df: pl.DataFrame, circuit_length: float | None
) -> pl.DataFrame:
    """Add num_active_drivers (per lap) and distance_remaining_km (raw km).

    A driver is active at lap L if not retired, or retired with retirement_lap >= L.
    total_laps is the maximum lap_number in the race (the winner's lap count).
    distance_remaining_km is null when the circuit length is unknown.
    """
    total_laps = df["lap_number"].max()

    active = (
        pl.col("retirement_lap").is_null() | (pl.col("retirement_lap") >= pl.col("lap_number"))
    )
    df = df.with_columns(active.alias("_active"))

    per_lap = (
        df.group_by("lap_number")
        .agg(pl.col("_active").sum().cast(pl.Int64).alias("num_active_drivers"))
    )

    dist_expr = (
        pl.lit(None, dtype=pl.Float64)
        if circuit_length is None
        else (pl.lit(float(circuit_length)) * (pl.lit(total_laps) - pl.col("lap_number")))
    )

    return (
        df.join(per_lap, on="lap_number", how="left")
        .with_columns(dist_expr.alias("distance_remaining_km"))
        .drop("_active")
    )


def _add_positions_gained(df: pl.DataFrame, grid: dict[int, int]) -> pl.DataFrame:
    """positions_gained_from_grid = grid_position - current position (>0 = gained)."""
    grid_df = pl.DataFrame(
        {"driver_number": list(grid.keys()), "_grid": list(grid.values())},
        schema={"driver_number": df.schema["driver_number"], "_grid": pl.Int64},
    )
    return (
        df.join(grid_df, on="driver_number", how="left")
        .with_columns((pl.col("_grid") - pl.col("position")).alias("positions_gained_from_grid"))
        .drop("_grid")
    )


def _add_pace_deltas(df: pl.DataFrame) -> pl.DataFrame:
    """last_lap_pace_delta_to_ahead/behind: lap_time minus the P-1 / P+1 car's lap_time.

    Stage 2's position asof-join can momentarily tie two drivers at the same
    position on a lap. The position lookup is deduplicated to one lap_time per
    (lap_number, position) — lowest driver_number wins, deterministically — so
    the self-join cannot fan out rows.
    """
    pace = (
        df.select(["lap_number", "position", "driver_number", "lap_time"])
        .sort(["lap_number", "position", "driver_number"])
        .unique(subset=["lap_number", "position"], keep="first", maintain_order=True)
        .select(["lap_number", "position", "lap_time"])
    )
    ahead = pace.rename({"position": "_pos_join", "lap_time": "_lt_ahead"})
    behind = pace.rename({"position": "_pos_join", "lap_time": "_lt_behind"})

    return (
        df.with_columns([
            (pl.col("position") - 1).alias("_ahead_pos"),
            (pl.col("position") + 1).alias("_behind_pos"),
        ])
        .join(ahead, left_on=["lap_number", "_ahead_pos"], right_on=["lap_number", "_pos_join"], how="left")
        .join(behind, left_on=["lap_number", "_behind_pos"], right_on=["lap_number", "_pos_join"], how="left")
        .with_columns([
            (pl.col("lap_time") - pl.col("_lt_ahead")).alias("last_lap_pace_delta_to_ahead"),
            (pl.col("lap_time") - pl.col("_lt_behind")).alias("last_lap_pace_delta_to_behind"),
        ])
        .drop(["_ahead_pos", "_behind_pos", "_lt_ahead", "_lt_behind"])
    )


def _gaps_ahead_for_lap(positions: list[int], gaps: list[float]) -> dict[int, tuple[float, float]]:
    """For one lap, return {position: (mean, stdev)} of inter-car gaps among cars ahead.

    Inter-car gap at position k (k>=2) = gap_to_leader[k] - gap_to_leader[k-1].
    For a driver at position P, aggregate the inter-car gaps of positions 2..P-1.
    Leader and P2 have <2 cars ahead -> (0, 0). Population stdev (ddof=0).
    """
    order = sorted(range(len(positions)), key=lambda i: positions[i])
    sorted_pos = [positions[i] for i in order]
    sorted_gap = [gaps[i] for i in order]

    inter = [sorted_gap[k] - sorted_gap[k - 1] for k in range(1, len(sorted_gap))]  # len n-1
    result: dict[int, tuple[float, float]] = {}
    for idx, pos in enumerate(sorted_pos):
        ahead_inter = inter[: max(idx - 1, 0)]  # gaps among positions strictly ahead
        if len(ahead_inter) == 0:
            result[pos] = (0.0, 0.0)
        else:
            arr = np.array(ahead_inter, dtype=float)
            result[pos] = (float(arr.mean()), float(arr.std(ddof=0)))
    return result


def _add_gaps_ahead(df: pl.DataFrame) -> pl.DataFrame:
    """Add mean_gap_cars_ahead / stdev_gap_cars_ahead (traffic density ahead).

    Rows with null position or null gap_to_leader at a lap are excluded from that
    lap's gap computation and receive null features.
    """
    means: list[float | None] = []
    stdevs: list[float | None] = []
    keys: list[tuple] = []

    for (lap,), grp in df.group_by(["lap_number"], maintain_order=True):
        valid = grp.filter(pl.col("position").is_not_null() & pl.col("gap_to_leader").is_not_null())
        lookup = _gaps_ahead_for_lap(
            valid["position"].to_list(), valid["gap_to_leader"].to_list()
        )
        for d, p in zip(grp["driver_number"].to_list(), grp["position"].to_list()):
            keys.append((lap, d))
            m, s = lookup.get(p, (None, None))
            means.append(m)
            stdevs.append(s)

    feat = pl.DataFrame({
        "lap_number": [k[0] for k in keys],
        "driver_number": [k[1] for k in keys],
        "mean_gap_cars_ahead": means,
        "stdev_gap_cars_ahead": stdevs,
    }, schema_overrides={
        "lap_number": df.schema["lap_number"],
        "driver_number": df.schema["driver_number"],
    })
    return df.join(feat, on=["lap_number", "driver_number"], how="left")


_TYRE_ONEHOT = {
    "SOFT": "tyre_soft",
    "MEDIUM": "tyre_medium",
    "HARD": "tyre_hard",
    "INTER": "tyre_inter",
    "WET": "tyre_wet",
}


def _add_tyre_onehot(df: pl.DataFrame) -> pl.DataFrame:
    """Five binary tyre columns. Null/unknown compound -> all zeros."""
    return df.with_columns([
        (pl.col("tyre_compound") == compound).fill_null(False).cast(pl.Int8).alias(col)
        for compound, col in _TYRE_ONEHOT.items()
    ])


def _add_stops_vs_median(df: pl.DataFrame) -> pl.DataFrame:
    """stops_completed minus the per-lap median stops across the field."""
    med = df.group_by("lap_number").agg(
        pl.col("stops_completed").median().alias("_med_stops")
    )
    return (
        df.join(med, on="lap_number", how="left")
        .with_columns(
            (pl.col("stops_completed").cast(pl.Float64) - pl.col("_med_stops")).alias("stops_vs_median")
        )
        .drop("_med_stops")
    )


def _add_rolling_pace(df: pl.DataFrame) -> pl.DataFrame:
    """Add rolling_lap_time_3_norm and rolling_lap_time_3_delta_leader.

    rolling3 = mean lap_time over the last 3 laps per driver (min_periods=1).
    _norm divides by the field median of rolling3 at that lap; _delta_leader
    subtracts the rolling3 of the car in position 1 at that lap. Pit/SC laps are
    included as-is (they inflate rolling3); this is acceptable for v1.
    """
    df = df.sort(["driver_number", "lap_number"]).with_columns(
        pl.col("lap_time").rolling_mean(window_size=3, min_samples=1).over("driver_number").alias("_roll3")
    )

    field = (
        df.group_by("lap_number").agg(pl.col("_roll3").median().alias("_field_med"))
    )
    leader = (
        df.filter(pl.col("position") == 1)
        .select(["lap_number", pl.col("_roll3").alias("_leader_roll3")])
        .unique("lap_number")
    )

    return (
        df.join(field, on="lap_number", how="left")
        .join(leader, on="lap_number", how="left")
        .with_columns([
            (pl.col("_roll3") / pl.col("_field_med")).alias("rolling_lap_time_3_norm"),
            (pl.col("_roll3") - pl.col("_leader_roll3")).alias("rolling_lap_time_3_delta_leader"),
        ])
        .drop(["_roll3", "_field_med", "_leader_roll3"])
    )


def _grid_from_position(pos_df: pl.DataFrame) -> dict[int, int]:
    """Grid position per driver = position at the earliest reading (pre-race)."""
    earliest = (
        pos_df.sort("date")
        .group_by("driver_number", maintain_order=True)
        .agg(pl.col("position").first().alias("grid"))
    )
    return dict(zip(earliest["driver_number"].to_list(), earliest["grid"].to_list()))


def build_features(
    session_key: int,
    sessions_dir: Path,
    raw_dir: Path,
    features_dir: Path,
    priors: pl.DataFrame,
    circuits: dict | None = None,
) -> pl.DataFrame:
    """Engineer all Stage 3 features for one race and write the parquet.

    `priors` is the cross-race prior table from compute_priors() (one row per
    (session_key, driver_number)); it is computed once for all races by the CLI.
    Returns the feature DataFrame.
    """
    circuits = circuits or load_circuits()
    df = pl.read_parquet(sessions_dir / f"{session_key}.parquet")

    ses = pl.read_parquet(raw_dir / str(session_key) / "sessions.parquet").row(0, named=True)
    circuit = ses["circuit_short_name"]
    pos_raw = pl.read_parquet(raw_dir / str(session_key) / "position.parquet")
    grid = _grid_from_position(pos_raw)

    df = _parse_gap_columns(df)
    df = _add_active_and_distance(df, circuit_length_km(circuit, circuits))
    df = _add_positions_gained(df, grid)
    df = _add_pace_deltas(df)
    df = _add_gaps_ahead(df)
    df = _add_rolling_pace(df)
    df = _add_tyre_onehot(df)
    df = _add_stops_vs_median(df)
    df = df.with_columns(pl.lit(is_street_circuit(circuit, circuits)).alias("is_street_circuit"))

    race_priors = priors.filter(pl.col("session_key") == session_key).drop("session_key")
    df = df.join(race_priors, on="driver_number", how="left")

    out = df.select(_KEY_COLUMNS + FEATURE_COLUMNS)

    features_dir.mkdir(parents=True, exist_ok=True)
    out.write_parquet(features_dir / f"{session_key}.parquet")
    return out
