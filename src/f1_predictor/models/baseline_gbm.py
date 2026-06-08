"""LightGBM LambdaRank baseline for snapshot ranking.

One ranking group per (session_key, snapshot_lap); label = relevance
(21 - final_position). Higher predicted score = better (lower) final position.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import polars as pl

_DEFAULT_PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "min_data_in_leaf": 20,
    "n_estimators": 300,
    "verbose": -1,
}


def group_sizes(df: pl.DataFrame) -> list[int]:
    """Row counts per (session_key, snapshot_lap) group, in row order.

    The DataFrame MUST already be sorted by (session_key, snapshot_lap) so the
    returned sizes line up with LightGBM's contiguous-group expectation.
    """
    return (
        df.group_by(["session_key", "snapshot_lap"], maintain_order=True)
        .len()["len"]
        .to_list()
    )


def _sorted(df: pl.DataFrame) -> pl.DataFrame:
    return df.sort(["session_key", "snapshot_lap"])


def train_baseline(
    train: pl.DataFrame,
    feature_columns: list[str],
    params: dict | None = None,
    valid: pl.DataFrame | None = None,
) -> lgb.Booster:
    """Train a LambdaRank booster. Returns the fitted Booster."""
    train = _sorted(train)
    p = {**_DEFAULT_PARAMS, **(params or {})}
    n_estimators = p.pop("n_estimators")

    dtrain = lgb.Dataset(
        train.select(feature_columns).to_numpy(),
        label=train["relevance"].to_numpy(),
        group=group_sizes(train),
        feature_name=list(feature_columns),
    )
    valid_sets = [dtrain]
    if valid is not None and not valid.is_empty():
        valid = _sorted(valid)
        dvalid = lgb.Dataset(
            valid.select(feature_columns).to_numpy(),
            label=valid["relevance"].to_numpy(),
            group=group_sizes(valid),
            reference=dtrain,
        )
        valid_sets.append(dvalid)

    return lgb.train(p, dtrain, num_boost_round=n_estimators, valid_sets=valid_sets)


def predict(model: lgb.Booster, df: pl.DataFrame, feature_columns: list[str]) -> np.ndarray:
    """Predicted ranking scores aligned to df's current row order."""
    return model.predict(df.select(feature_columns).to_numpy())
