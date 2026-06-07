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
