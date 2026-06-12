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
import torch

from f1_predictor.models.baseline_gbm import add_current_rank

UNKNOWN_DRIVER_INDEX = 0  # reserved for padding and unseen drivers
PAD_FINAL_POSITION = 99  # sentinel final_position for padded slots (never scored)


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
        (pl.col("current_rank") - pl.col("final_position"))
        .cast(pl.Float64)
        .alias("delta")
    )


def snapshot_to_tensors(
    group: pl.DataFrame,
    feature_columns: list[str],
    driver_index: dict[int, int],
    num_slots: int,
) -> dict:
    """Pad one (session_key, snapshot_lap) group to num_slots driver slots.

    `group` must already carry `current_rank` and `delta` (see prepare_split).
    Active drivers fill slots 0..n-1; the rest are padding. The returned dict is
    the tensor contract consumed by SnapshotDataset / the model / the loss.
    """
    n = group.height
    if n > num_slots:  # safety: keep the best-classified drivers
        group = group.sort("final_position").head(num_slots)
        n = num_slots

    feats = torch.zeros((num_slots, len(feature_columns)), dtype=torch.float32)
    feats[:n] = torch.tensor(
        group.select(feature_columns).to_numpy(), dtype=torch.float32
    )

    driver_numbers = group["driver_number"].to_list()
    driver_idx = torch.zeros(num_slots, dtype=torch.long)
    driver_number = torch.zeros(num_slots, dtype=torch.long)
    for i, d in enumerate(driver_numbers):
        driver_idx[i] = driver_index.get(d, UNKNOWN_DRIVER_INDEX)
        driver_number[i] = d

    valid = torch.zeros(num_slots, dtype=torch.bool)
    valid[:n] = True

    current_rank = torch.zeros(num_slots, dtype=torch.float32)
    current_rank[:n] = torch.tensor(
        group["current_rank"].to_numpy(), dtype=torch.float32
    )

    delta = torch.zeros(num_slots, dtype=torch.float32)
    delta[:n] = torch.tensor(group["delta"].to_numpy(), dtype=torch.float32)

    final_position = torch.full((num_slots,), PAD_FINAL_POSITION, dtype=torch.long)
    final_position[:n] = torch.tensor(
        group["final_position"].to_numpy(), dtype=torch.long
    )

    return {
        "features": feats,
        "driver_idx": driver_idx,
        "driver_number": driver_number,
        "valid": valid,
        "current_rank": current_rank,
        "delta": delta,
        "final_position": final_position,
        "session_key": int(group["session_key"][0]),
        "snapshot_lap": int(group["snapshot_lap"][0]),
    }
