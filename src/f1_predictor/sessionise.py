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
        "laps", "position", "intervals", "stints", "pit",
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


_DT_FMT = "%Y-%m-%dT%H:%M:%S%:z"
_FAR_FUTURE = "2099-01-01T00:00:00+00:00"


def _lap_end_times(lap_table: pl.DataFrame) -> pl.DataFrame:
    """Add lap_end_time column: start of the next lap per driver (far future for last lap)."""
    return (
        lap_table.sort(["driver_number", "lap_number"])
        .with_columns(
            pl.col("date_start")
            .str.to_datetime(format=_DT_FMT, time_unit="us")
            .cast(pl.Datetime("us", "UTC"))
        )
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


def _join_pit(lap_table: pl.DataFrame, pit_df: pl.DataFrame) -> pl.DataFrame:
    """Add pit_this_lap (bool) and stops_completed (cumulative count) from pit endpoint."""
    pit_flags = (
        pit_df.select(["driver_number", "lap_number"])
        .unique()
        .with_columns(pl.lit(True).alias("pit_this_lap"))
    )

    return (
        lap_table
        .join(pit_flags, on=["driver_number", "lap_number"], how="left")
        .with_columns(pl.col("pit_this_lap").fill_null(False))
        .sort(["driver_number", "lap_number"])
        .with_columns(
            pl.col("pit_this_lap")
            .cast(pl.Int32)
            .cum_sum()
            .over("driver_number")
            .alias("stops_completed")
        )
    )


def _add_car_data(lap_table: pl.DataFrame, car_data_df: pl.DataFrame) -> pl.DataFrame:
    """Aggregate max speed from car telemetry into max_speed_kmh per driver-lap."""
    if car_data_df.is_empty():
        return lap_table.with_columns(pl.lit(None).cast(pl.Float64).alias("max_speed_kmh"))

    # Assign each telemetry sample to a lap via backward asof join on lap start time
    laps_sorted = (
        lap_table.with_columns(
            pl.col("date_start")
            .str.to_datetime(format=_DT_FMT, time_unit="us")
            .cast(pl.Datetime("us", "UTC"))
        )
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
