"""Stage 3: engineer features per race from the Stage 2 sessionised table.

Pure, deterministic transforms producing raw (unscaled) human-readable values.
Scaling happens in Stage 4. Cross-race priors live in priors.py.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import yaml

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
    """last_lap_pace_delta_to_ahead/behind: lap_time minus the P-1 / P+1 car's lap_time."""
    pace = df.select(["lap_number", "position", "lap_time"])
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
