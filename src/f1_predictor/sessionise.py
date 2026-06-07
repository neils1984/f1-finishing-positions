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
