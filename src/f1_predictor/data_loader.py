"""Load snapshot parquets into padded per-group tensors for the Transformer.

The snapshot splits are long-format (one row per active driver per
(session_key, snapshot_lap)). This module maps driver numbers to embedding
indices, augments each split with the delta-regression target, and packs each
race-lap group into fixed-width [num_slots, ...] tensors with a validity mask.

Target convention matches the LightGBM baseline exactly (so the two models are
directly comparable): delta = current_rank - final_position, score reconstructed
downstream as predicted_delta - current_rank.
"""
from __future__ import annotations

import polars as pl

from f1_predictor.models.baseline_gbm import add_current_rank

UNKNOWN_DRIVER_INDEX = 0  # reserved for padding and unseen drivers


def build_driver_index(train: pl.DataFrame, max_drivers: int) -> dict[int, int]:
    """Map driver_number -> embedding index 1..(max_drivers-1), 0 reserved.

    Built from the train split (sorted driver_number for determinism). Drivers
    beyond the capacity, and any driver unseen in train, map to UNKNOWN at
    lookup time (they are simply absent from this dict).
    """
    drivers = sorted(train["driver_number"].unique().to_list())
    capacity = max_drivers - 1  # index 0 reserved
    return {d: i + 1 for i, d in enumerate(drivers[:capacity])}


def prepare_split(df: pl.DataFrame) -> pl.DataFrame:
    """Add current_rank (reused from the GBM) and the delta regression target.

    delta = current_rank - final_position (positive = places gained), identical
    to the LightGBM baseline so the two models are directly comparable.
    """
    df = add_current_rank(df)
    return df.with_columns(
        (pl.col("current_rank") - pl.col("final_position")).cast(pl.Float64).alias("delta")
    )
