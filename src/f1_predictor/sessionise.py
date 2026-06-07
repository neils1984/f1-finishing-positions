"""Stage 2: collapse per-endpoint raw Parquet into one (driver_number, lap_number) table."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl


# ---------------------------------------------------------------------------
# Internal helpers — each takes the current lap_table and raw data, returns
# the lap_table with new columns appended.
# ---------------------------------------------------------------------------


def _read_raw(session_dir: Path) -> dict[str, pl.DataFrame]:
    """Read all endpoint Parquet files for a session into a dict."""
    endpoints = [
        "laps", "position", "intervals", "stints",
        "race_control", "session_result", "car_data", "drivers",
    ]
    raw: dict[str, pl.DataFrame] = {}
    for ep in endpoints:
        path = session_dir / f"{ep}.parquet"
        raw[ep] = pl.read_parquet(path) if path.exists() else pl.DataFrame()
    return raw


def _build_lap_table(laps: pl.DataFrame) -> pl.DataFrame:
    """Select and rename columns from the laps endpoint to form the base table."""
    return (
        laps.select(["session_key", "driver_number", "lap_number", "date_start", "lap_duration"])
        .rename({"lap_duration": "lap_time"})
        .filter(pl.col("lap_number") > 0)  # exclude formation/pit-out lap 0
    )


# %.f makes fractional seconds optional: real OpenF1 timestamps carry them
# (e.g. ...:38.500000+00:00) while some values / synthetic fixtures do not.
_DT_FMT = "%Y-%m-%dT%H:%M:%S%.f%:z"
_FAR_FUTURE = "2099-01-01T00:00:00+00:00"


def _ensure_utc(df: pl.DataFrame, col: str) -> pl.DataFrame:
    """Parse a string timestamp column to UTC datetime; pass through if already datetime.

    Idempotent: _lap_end_times is called by both _join_positions and
    _join_intervals on the same evolving lap_table, so date_start may already
    have been converted by an earlier join.
    """
    if df.schema[col] == pl.String:
        return df.with_columns(
            pl.col(col).str.to_datetime(format=_DT_FMT, time_unit="us").cast(pl.Datetime("us", "UTC"))
        )
    return df


def _lap_end_times(lap_table: pl.DataFrame) -> pl.DataFrame:
    """Add lap_end_time column: start of the next lap per driver (far future for last lap)."""
    return (
        _ensure_utc(lap_table.sort(["driver_number", "lap_number"]), "date_start")
        .with_columns(
            pl.col("date_start")
            .shift(-1)
            .over("driver_number")
            .fill_null(
                pl.lit(_FAR_FUTURE).str.to_datetime(format=_DT_FMT, time_unit="us").cast(pl.Datetime("us", "UTC"))
            )
            .alias("lap_end_time")
        )
    )


def _join_positions(lap_table: pl.DataFrame, pos_df: pl.DataFrame) -> pl.DataFrame:
    """Add `position` (race position at end of each lap) via backward asof join."""
    if pos_df.is_empty():
        return lap_table.with_columns(pl.lit(None).cast(pl.Int64).alias("position"))

    laps = _lap_end_times(lap_table).sort(["driver_number", "lap_end_time"])

    pos = (
        pos_df.with_columns(
            pl.col("date")
            .str.to_datetime(format=_DT_FMT, time_unit="us")
            .cast(pl.Datetime("us", "UTC"))
            .alias("lap_end_time")
        )
        .sort(["driver_number", "lap_end_time"])
        .select(["driver_number", "lap_end_time", "position"])
    )

    return (
        laps.join_asof(pos, on="lap_end_time", by="driver_number", strategy="backward")
        .drop("lap_end_time")
    )


def _join_intervals(lap_table: pl.DataFrame, intervals_df: pl.DataFrame) -> pl.DataFrame:
    """Add `gap_to_leader` and `interval_to_ahead` via backward asof join."""
    if intervals_df.is_empty():
        return lap_table.with_columns([
            pl.lit(None).cast(pl.Float64).alias("gap_to_leader"),
            pl.lit(None).cast(pl.Float64).alias("interval_to_ahead"),
        ])

    laps = _lap_end_times(lap_table).sort(["driver_number", "lap_end_time"])

    ivl = (
        intervals_df.with_columns(
            pl.col("date")
            .str.to_datetime(format=_DT_FMT, time_unit="us")
            .cast(pl.Datetime("us", "UTC"))
            .alias("lap_end_time")
        )
        .sort(["driver_number", "lap_end_time"])
        .select(["driver_number", "lap_end_time", "gap_to_leader", "interval"])
        .rename({"interval": "interval_to_ahead"})
    )

    return (
        laps.join_asof(ivl, on="lap_end_time", by="driver_number", strategy="backward")
        .drop("lap_end_time")
    )


def _join_stints(lap_table: pl.DataFrame, stints_df: pl.DataFrame) -> pl.DataFrame:
    """Add tyre_compound, tyre_age_laps, stint_number from stints endpoint."""
    if stints_df.is_empty():
        return lap_table.with_columns([
            pl.lit(None).cast(pl.Utf8).alias("tyre_compound"),
            pl.lit(None).cast(pl.Int64).alias("tyre_age_laps"),
            pl.lit(None).cast(pl.Int64).alias("stint_number"),
        ])

    stints = stints_df.select([
        "driver_number", "stint_number", "lap_start", "lap_end",
        "compound", "tyre_age_at_start",
    ])

    # Cross-join on driver then filter to the matching stint range
    result = (
        lap_table
        .join(stints, on="driver_number", how="left")
        .filter(
            (pl.col("lap_number") >= pl.col("lap_start")) &
            (pl.col("lap_number") <= pl.col("lap_end"))
        )
        .with_columns([
            pl.col("compound").alias("tyre_compound"),
            (
                pl.col("lap_number") - pl.col("lap_start")
                + pl.col("tyre_age_at_start").fill_null(0)
            ).alias("tyre_age_laps"),
        ])
        .drop(["lap_start", "lap_end", "compound", "tyre_age_at_start"])
    )
    return result


def _add_pit_from_stints(lap_table: pl.DataFrame) -> pl.DataFrame:
    """Derive pit_this_lap and stops_completed from the stint_number column.

    OpenF1 has no /pit data for 2023 (the training season) while 2024 does, so
    pit info is derived from /stints (present for all seasons) to avoid a
    train/test skew. A pit stop is the first lap of each stint after the first
    (stint_number increments); stops_completed = stint_number - 1.

    Requires `stint_number` to already be present (i.e. run after _join_stints).
    """
    return (
        lap_table.sort(["driver_number", "lap_number"])
        .with_columns(
            (pl.col("stint_number").fill_null(1) - 1).cast(pl.Int32).alias("stops_completed")
        )
        .with_columns(
            (
                (pl.col("stint_number") > 1)
                & (pl.col("stint_number") != pl.col("stint_number").shift(1).over("driver_number"))
            )
            .fill_null(False)
            .alias("pit_this_lap")
        )
    )


def _add_car_data(lap_table: pl.DataFrame, car_data_df: pl.DataFrame) -> pl.DataFrame:
    """Aggregate max speed from car telemetry into max_speed_kmh per driver-lap."""
    if car_data_df.is_empty():
        return lap_table.with_columns(pl.lit(None).cast(pl.Float64).alias("max_speed_kmh"))

    # Assign each telemetry sample to a lap via backward asof join on lap start time
    laps_sorted = (
        _ensure_utc(lap_table, "date_start")
        .sort(["driver_number", "date_start"])
        .select(["driver_number", "lap_number", "date_start"])
    )

    car = (
        car_data_df.with_columns(
            pl.col("date")
            .str.to_datetime(format=_DT_FMT, time_unit="us")
            .cast(pl.Datetime("us", "UTC"))
            .alias("date_start")
        )
        .sort(["driver_number", "date_start"])
        .select(["driver_number", "date_start", "speed"])
    )

    # For each telemetry sample, find which lap it belongs to (last lap started before it)
    car_with_lap = car.join_asof(
        laps_sorted,
        on="date_start",
        by="driver_number",
        strategy="backward",
    )

    max_speed = (
        car_with_lap.filter(pl.col("lap_number").is_not_null())
        .group_by(["driver_number", "lap_number"])
        .agg(pl.col("speed").max().alias("max_speed_kmh"))
    )

    return lap_table.join(max_speed, on=["driver_number", "lap_number"], how="left")


def _parse_flag_windows(rc: pl.DataFrame, deploy_substr: str, end_substr: str) -> list[tuple[int, int]]:
    """Return list of (start_lap, end_lap) where a flag regime was active."""
    deploy_laps = sorted(
        rc.filter(pl.col("message").str.contains(deploy_substr))["lap_number"]
        .drop_nulls()
        .to_list()
    )
    end_laps = sorted(
        rc.filter(pl.col("message").str.contains(end_substr))["lap_number"]
        .drop_nulls()
        .to_list()
    )
    # Pair each deployment with the next ending
    windows = list(zip(deploy_laps, end_laps))
    return windows


def _add_race_control_flags(lap_table: pl.DataFrame, rc_df: pl.DataFrame) -> pl.DataFrame:
    """Add sc_active, vsc_active, red_flag_active, laps_since_sc_end."""
    if rc_df.is_empty():
        return lap_table.with_columns([
            pl.lit(False).alias("sc_active"),
            pl.lit(False).alias("vsc_active"),
            pl.lit(False).alias("red_flag_active"),
            pl.lit(0).cast(pl.Int64).alias("laps_since_sc_end"),
        ])

    sc_windows = _parse_flag_windows(rc_df, "SAFETY CAR DEPLOYED", "SAFETY CAR IN THIS LAP")
    vsc_windows = _parse_flag_windows(rc_df, "VIRTUAL SAFETY CAR DEPLOYED", "VIRTUAL SAFETY CAR ENDING")

    red_flag_laps = set(
        rc_df.filter(pl.col("flag") == "RED")["lap_number"].drop_nulls().to_list()
    )

    def in_any_window(lap: int, windows: list[tuple[int, int]]) -> bool:
        return any(start <= lap <= end for start, end in windows)

    # SC end laps drive laps_since_sc_end. The window is inclusive, so the
    # "SAFETY CAR IN THIS LAP" lap is itself SC-active (laps_since == 0); laps
    # after it count up from that end lap.
    sc_end_laps = sorted(lap for _, lap in sc_windows)

    all_laps = sorted(lap_table["lap_number"].unique().to_list())

    sc_map: dict[int, bool] = {}
    vsc_map: dict[int, bool] = {}
    red_map: dict[int, bool] = {}
    lssce_map: dict[int, int] = {}

    for lap in all_laps:
        sc_now = in_any_window(lap, sc_windows)
        sc_map[lap] = sc_now
        vsc_map[lap] = in_any_window(lap, vsc_windows)
        red_map[lap] = lap in red_flag_laps

        # 0 while SC is active; after it ends, count laps since the end lap.
        if sc_now:
            lssce_map[lap] = 0
        else:
            prior_ends = [e for e in sc_end_laps if e <= lap]
            lssce_map[lap] = (lap - max(prior_ends)) if prior_ends else 0

    return lap_table.with_columns([
        pl.col("lap_number").replace(sc_map).alias("sc_active").cast(pl.Boolean),
        pl.col("lap_number").replace(vsc_map).alias("vsc_active").cast(pl.Boolean),
        pl.col("lap_number").replace(red_map).alias("red_flag_active").cast(pl.Boolean),
        pl.col("lap_number").replace(lssce_map).alias("laps_since_sc_end").cast(pl.Int64),
    ])


def _add_retirements(
    lap_table: pl.DataFrame,
    session_result: pl.DataFrame,
) -> pl.DataFrame:
    """Add is_retired, retirement_lap, final_position from session_result.

    OpenF1's session_result flags non-finishers with boolean dnf/dns/dsq columns
    and leaves their `position` null. The spec requires every driver to keep an
    official classification position in laps-completed order, so unclassified
    drivers are ranked after the finishers by number_of_laps (descending).
    """
    sr = session_result.with_columns(
        (
            pl.col("dnf").fill_null(False)
            | pl.col("dns").fill_null(False)
            | pl.col("dsq").fill_null(False)
        ).alias("is_retired")
    )

    classified = sr.filter(pl.col("position").is_not_null())
    max_pos = classified.select(pl.col("position").max()).item()
    max_pos = max_pos if max_pos is not None else 0

    # Unclassified (null position): order by laps completed, place after finishers.
    unclassified = (
        sr.filter(pl.col("position").is_null())
        .sort("number_of_laps", descending=True, nulls_last=True)
        .with_columns((max_pos + pl.int_range(1, pl.len() + 1)).cast(pl.Int64).alias("position"))
    )

    full = pl.concat([classified, unclassified], how="vertical")
    final_info = full.select(
        "driver_number",
        pl.col("position").alias("final_position"),
        "is_retired",
    )

    # retirement_lap = last lap a driver actually completed (only for retirees).
    last_laps = (
        lap_table.group_by("driver_number")
        .agg(pl.col("lap_number").max().alias("retirement_lap"))
    )

    return (
        lap_table
        .join(final_info, on="driver_number", how="left")
        .join(last_laps, on="driver_number", how="left")
        .with_columns(pl.col("is_retired").fill_null(False))
        .with_columns(
            pl.when(pl.col("is_retired"))
            .then(pl.col("retirement_lap"))
            .otherwise(None)
            .alias("retirement_lap")
        )
    )


def _build_masks(lap_table: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Build attention_mask [n_drivers, n_laps] and target_mask [n_drivers]."""
    drivers = sorted(lap_table["driver_number"].unique().to_list())
    laps = sorted(lap_table["lap_number"].unique().to_list())
    d_idx = {d: i for i, d in enumerate(drivers)}
    l_idx = {l: i for i, l in enumerate(laps)}

    attention_mask = np.ones((len(drivers), len(laps)), dtype=np.int8)

    retired = lap_table.filter(pl.col("is_retired")).select(
        ["driver_number", "retirement_lap"]
    ).unique("driver_number")

    for row in retired.iter_rows(named=True):
        d_i = d_idx[row["driver_number"]]
        ret_lap = row["retirement_lap"]
        if ret_lap is not None:
            for lap in laps:
                if lap > ret_lap:
                    attention_mask[d_i, l_idx[lap]] = 0

    # target_mask: 0 only for DNS (never appeared in laps table) — rare
    target_mask = np.ones(len(drivers), dtype=np.int8)

    return attention_mask, target_mask


def sessionise(session_key: int, raw_dir: Path, sessions_dir: Path) -> pl.DataFrame:
    """Run all Stage 2 joins for one race. Save result and masks; return DataFrame."""
    session_dir = raw_dir / str(session_key)
    raw = _read_raw(session_dir)

    df = _build_lap_table(raw["laps"])
    df = _join_positions(df, raw["position"])
    df = _join_intervals(df, raw["intervals"])
    df = _join_stints(df, raw["stints"])
    df = _add_pit_from_stints(df)
    df = _add_car_data(df, raw["car_data"])
    df = _add_race_control_flags(df, raw["race_control"])
    df = _add_retirements(df, raw["session_result"])

    sessions_dir.mkdir(parents=True, exist_ok=True)
    out_path = sessions_dir / f"{session_key}.parquet"
    df.write_parquet(out_path)

    attention_mask, target_mask = _build_masks(df)
    np.savez(
        sessions_dir / f"{session_key}_masks.npz",
        attention_mask=attention_mask,
        target_mask=target_mask,
    )

    return df
