"""Stage 4: build chronologically split, scaled snapshot training tensors.

Snapshots are extracted at fixed laps from the Stage 3 feature tables. The
StandardScaler is fitted on the train split only; nulls are imputed to 0.0
before scaling. Output: data/snapshots/{train,val,test}.parquet + metadata.json.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.preprocessing import StandardScaler

RELEVANCE_BASE = 21  # relevance = RELEVANCE_BASE - final_position (higher = better)

_META_COLUMNS = ["session_key", "snapshot_lap", "driver_number", "final_position", "relevance"]


def assign_split(date_start: str, val_start: str, test_start: str) -> str:
    """Classify a race into 'train' | 'val' | 'test' by two date boundaries.

    train: before val_start.  val: [val_start, test_start).  test: >= test_start.
    """
    d = datetime.fromisoformat(date_start).date()
    if d < datetime.fromisoformat(val_start).date():
        return "train"
    if d < datetime.fromisoformat(test_start).date():
        return "val"
    return "test"


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


def _race_date(raw_dir: Path, session_key: int) -> str:
    ses = pl.read_parquet(raw_dir / str(session_key) / "sessions.parquet").row(0, named=True)
    return ses["date_start"]


def _data_version(feature_columns: list[str], scaler: dict, git_sha: str) -> str:
    payload = json.dumps({"f": feature_columns, "s": scaler, "g": git_sha}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def build_snapshots(
    features_dir: Path,
    raw_dir: Path,
    out_dir: Path,
    feature_columns: list[str],
    snapshot_laps: list[int],
    val_start: str,
    test_start: str,
    git_sha: str = "unknown",
) -> dict:
    """Build train/val/test snapshot parquets + metadata.json. Returns metadata.

    Races are split by two date boundaries: train < val_start <= val < test_start <= test.
    Scaler is fit on the train split only and applied to all splits.
    """
    keys = sorted(int(p.stem) for p in features_dir.glob("*.parquet"))

    # Group races by split.
    split_keys: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    raw_by_split: dict[str, list[pl.DataFrame]] = {"train": [], "val": [], "test": []}
    for key in keys:
        split = assign_split(_race_date(raw_dir, key), val_start, test_start)
        feats = pl.read_parquet(features_dir / f"{key}.parquet")
        snaps = extract_snapshots(feats, snapshot_laps, feature_columns)
        # Guard: each driver must appear at most once per (race, snapshot_lap).
        # A duplicate here means corrupted upstream data and would silently
        # distort a LightGBM ranking group — fail loudly instead.
        dup = snaps.select(["session_key", "snapshot_lap", "driver_number"]).is_duplicated().sum()
        if dup:
            raise ValueError(f"Session {key}: {dup} duplicate driver-lap snapshot rows")
        snaps = snaps.with_columns(pl.lit(split).alias("split"))
        split_keys[split].append(key)
        raw_by_split[split].append(snaps)

    train_df = pl.concat(raw_by_split["train"], how="vertical") if raw_by_split["train"] else pl.DataFrame()
    if train_df.is_empty():
        raise ValueError("No train races found — cannot fit scaler.")

    scaler = fit_scaler(train_df, feature_columns)

    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        frames = raw_by_split[split]
        if not frames:
            pl.DataFrame().write_parquet(out_dir / f"{split}.parquet")
            continue
        df = pl.concat(frames, how="vertical")
        scaled = apply_scaler(df, scaler, feature_columns)
        scaled.write_parquet(out_dir / f"{split}.parquet")

    metadata = {
        "feature_columns": feature_columns,
        "scaler": scaler,
        "snapshot_laps": snapshot_laps,
        "splits": split_keys,
        "data_version": _data_version(feature_columns, scaler, git_sha),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata
