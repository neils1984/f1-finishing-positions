"""Stage 4: build chronologically split, scaled snapshot training tensors.

Snapshots are extracted at fixed laps from the Stage 3 feature tables. The
StandardScaler is fitted on the train split only; nulls are imputed to 0.0
before scaling. Output: data/snapshots/{train,val,test}.parquet + metadata.json.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import polars as pl
from sklearn.preprocessing import StandardScaler

# The validation season; anything earlier is train.
_VAL_YEAR = 2024

RELEVANCE_BASE = 21  # relevance = RELEVANCE_BASE - final_position (higher = better)

_META_COLUMNS = ["session_key", "snapshot_lap", "driver_number", "final_position", "relevance"]


def assign_split(date_start: str, val_cutoff: str) -> str:
    """Classify a race into 'train' | 'val' | 'test' by its start date.

    train: any race before the validation season (2024).
    val:   a 2024 race strictly before val_cutoff.
    test:  a 2024 race on or after val_cutoff.
    """
    dt = datetime.fromisoformat(date_start)
    cutoff = datetime.fromisoformat(val_cutoff).date()
    if dt.year < _VAL_YEAR:
        return "train"
    return "val" if dt.date() < cutoff else "test"


def extract_snapshots(
    features: pl.DataFrame,
    snapshot_laps: list[int],
    feature_columns: list[str],
) -> pl.DataFrame:
    """One row per (snapshot_lap, active driver) with relevance + feature columns.

    A driver is "active" at a snapshot lap if it has a feature row at that exact
    lap_number. relevance = RELEVANCE_BASE - final_position.
    """
    snaps = (
        features.filter(pl.col("lap_number").is_in(snapshot_laps))
        .with_columns([
            pl.col("lap_number").alias("snapshot_lap"),
            (RELEVANCE_BASE - pl.col("final_position")).alias("relevance"),
        ])
        .select(_META_COLUMNS + feature_columns)
    )
    return snaps


def _impute(df: pl.DataFrame, feature_columns: list[str]) -> pl.DataFrame:
    """Bool->Int, then fill nulls with 0.0 and cast features to Float64."""
    return df.with_columns([
        pl.col(c).cast(pl.Float64, strict=False).fill_null(0.0).alias(c)
        for c in feature_columns
    ])


def fit_scaler(train: pl.DataFrame, feature_columns: list[str]) -> dict:
    """Fit a StandardScaler on imputed train features; return params as dicts.

    Zero-variance columns get scale 1.0 (sklearn behaviour), so all-null-in-train
    features map to 0 in train and pass real values through unchanged elsewhere.
    """
    x = _impute(train, feature_columns).select(feature_columns).to_numpy()
    scaler = StandardScaler().fit(x)
    scale = np.where(scaler.scale_ == 0.0, 1.0, scaler.scale_)
    return {
        "mean": {c: float(m) for c, m in zip(feature_columns, scaler.mean_)},
        "scale": {c: float(s) for c, s in zip(feature_columns, scale)},
    }


def apply_scaler(df: pl.DataFrame, params: dict, feature_columns: list[str]) -> pl.DataFrame:
    """Impute nulls to 0.0 then standardise each feature with the fitted params."""
    df = _impute(df, feature_columns)
    return df.with_columns([
        ((pl.col(c) - params["mean"][c]) / params["scale"][c]).alias(c)
        for c in feature_columns
    ])
