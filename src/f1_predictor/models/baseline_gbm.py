"""LightGBM LambdaRank baseline for snapshot ranking.

One ranking group per (session_key, snapshot_lap); label = relevance
(21 - final_position). Higher predicted score = better (lower) final position.
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
    valid = pl.read_parquet(val_path) if val_path.exists() and pl.read_parquet(val_path).height else None

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
    (run_dir / "config.yaml").write_text(json.dumps({
        "model": "lightgbm_lambdarank",
        "params": {**_DEFAULT_PARAMS, **(params or {})},
        "feature_columns": feature_columns,
        "data_version": meta.get("data_version"),
    }, indent=2))

    if use_mlflow:
        import mlflow
        mlflow.set_experiment("f1-baseline")
        with mlflow.start_run(run_name=run_id):
            mlflow.log_params({"model": "lightgbm_lambdarank", **(params or {})})
            mlflow.log_metrics(metrics)
            mlflow.log_artifacts(str(run_dir))

    return {"run_dir": str(run_dir), "metrics": metrics}
