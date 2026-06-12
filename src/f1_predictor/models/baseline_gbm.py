"""LightGBM delta-regression baseline for snapshot ranking (Option B).

Rather than ranking on absolute relevance, the model predicts each driver's
*places gained* relative to their current race position::

    delta = current_rank - final_position        (positive = gained places)

and the ranking score is reconstructed as::

    score = predicted_delta - current_rank        (== -predicted_final_position)

A predicted delta of 0 reproduces the naive persistence baseline
(score = -current_rank), so the model only has to learn the residual movement.
A robust L1 objective is used because the delta distribution is heavy-tailed
(a back-marker recovering to the points has a huge positive delta) and squared
loss over-corrects the front of the grid. Empirically this is the first
formulation that beats naive persistence at every snapshot lap.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from f1_predictor.evaluate import ranking_metrics

_DEFAULT_PARAMS = {
    "objective": "regression_l1",
    "metric": "l1",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "min_data_in_leaf": 20,
    "n_estimators": 300,
    "verbose": -1,
}

_GROUP = ["session_key", "snapshot_lap"]


def add_current_rank(df: pl.DataFrame) -> pl.DataFrame:
    """Add `current_rank`: the 1..N race order within each (race, lap) group.

    `position` is stored standardised in snapshots but stays monotonic within a
    group, so an ordinal rank recovers the integer current race position. Ties
    (rare, momentary asof-join collisions) break by row order.
    """
    return df.with_columns(
        pl.col("position").rank(method="ordinal").over(_GROUP).cast(pl.Int64).alias("current_rank")
    )


def naive_predict(df: pl.DataFrame) -> np.ndarray:
    """Naive persistence baseline: score = -current_rank (predict no movement)."""
    df = add_current_rank(df)
    return -df["current_rank"].to_numpy().astype(float)


def _delta_target(df: pl.DataFrame) -> pl.Series:
    """Places gained = current_rank - final_position (requires `current_rank`)."""
    return (df["current_rank"] - df["final_position"]).cast(pl.Float64)


def season_weights(df: pl.DataFrame, upweight_2026: float) -> np.ndarray:
    """Per-row training weights: 2026 rows get upweight_2026, others 1.0.

    Requires a `season` column on df. The simplest regime-recency lever — sweep
    upweight_2026 in the backtest to trade old-data volume against 2026 focus.
    """
    return np.where(df["season"].to_numpy() >= 2026, float(upweight_2026), 1.0)


def blend_scores(naive: np.ndarray, model: np.ndarray, alpha: float) -> np.ndarray:
    """Convex blend of naive and model ranking scores (both ~ -finish position).

    alpha=0 is pure naive (the strong 2026 baseline); alpha=1 is pure model.
    """
    return (1.0 - alpha) * naive + alpha * model


def train_baseline(
    train: pl.DataFrame,
    feature_columns: list[str],
    params: dict | None = None,
    valid: pl.DataFrame | None = None,
    sample_weight: np.ndarray | None = None,
) -> lgb.Booster:
    """Train an L1 delta-regression booster. Returns the fitted Booster.

    `sample_weight` (per training row, aligned to `train`'s order) is passed
    through to LightGBM; None means uniform weighting.
    """
    train = add_current_rank(train)
    p = {**_DEFAULT_PARAMS, **(params or {})}
    n_estimators = p.pop("n_estimators")

    dtrain = lgb.Dataset(
        train.select(feature_columns).to_numpy(),
        label=_delta_target(train).to_numpy(),
        weight=sample_weight,
        feature_name=list(feature_columns),
    )
    valid_sets = [dtrain]
    if valid is not None and not valid.is_empty():
        valid = add_current_rank(valid)
        dvalid = lgb.Dataset(
            valid.select(feature_columns).to_numpy(),
            label=_delta_target(valid).to_numpy(),
            reference=dtrain,
        )
        valid_sets.append(dvalid)

    return lgb.train(p, dtrain, num_boost_round=n_estimators, valid_sets=valid_sets)


def predict(model: lgb.Booster, df: pl.DataFrame, feature_columns: list[str]) -> np.ndarray:
    """Reconstructed ranking scores aligned to df's current row order.

    score = predicted_delta - current_rank == -predicted_final_position, so a
    higher score means a better (lower) predicted finishing position.
    """
    df = add_current_rank(df)
    delta_hat = model.predict(df.select(feature_columns).to_numpy())
    return delta_hat - df["current_rank"].to_numpy()


def run_baseline(
    snapshots_dir: Path,
    runs_dir: Path,
    params: dict | None = None,
    use_mlflow: bool = True,
) -> dict:
    """Train on train.parquet, evaluate on test.parquet, persist a run directory.

    Returns {"run_dir": str, "metrics": {...}}.
    """
    meta = json.loads((snapshots_dir / "metadata.json").read_text())
    feature_columns = meta["feature_columns"]

    train = pl.read_parquet(snapshots_dir / "train.parquet")
    test = pl.read_parquet(snapshots_dir / "test.parquet")
    val_path = snapshots_dir / "val.parquet"
    valid = pl.read_parquet(val_path) if val_path.exists() else None
    if valid is not None and valid.is_empty():
        valid = None

    model = train_baseline(train, feature_columns, params=params, valid=valid)

    scores = predict(model, test, feature_columns)
    preds = test.select(["session_key", "snapshot_lap", "driver_number", "final_position"]).with_columns(
        pl.Series("score", scores)
    )
    metrics = ranking_metrics(preds)

    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(run_dir / "model.lgb"))
    preds.write_parquet(run_dir / "predictions_test.parquet")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (run_dir / "config.json").write_text(json.dumps({
        "model": "lightgbm_delta_l1",
        "params": {**_DEFAULT_PARAMS, **(params or {})},
        "feature_columns": feature_columns,
        "data_version": meta.get("data_version"),
    }, indent=2))

    if use_mlflow:
        import mlflow
        mlflow.set_experiment("f1-baseline")
        with mlflow.start_run(run_name=run_id):
            mlflow.log_params({"model": "lightgbm_delta_l1", **(params or {})})
            mlflow.log_metrics(metrics)
            mlflow.log_artifacts(str(run_dir))

    return {"run_dir": str(run_dir), "metrics": metrics}
