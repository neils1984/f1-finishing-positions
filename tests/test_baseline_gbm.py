import json
from pathlib import Path

import polars as pl
import numpy as np
import pytest
from f1_predictor.models.baseline_gbm import group_sizes, train_baseline, predict, run_baseline


def _toy_snapshots(n_races=6):
    # Each race has a clean signal: lower 'position' -> better relevance.
    rows = []
    for r in range(n_races):
        for d in range(1, 5):
            rows.append({
                "session_key": r, "snapshot_lap": 30, "driver_number": d,
                "final_position": d, "relevance": 21 - d, "position": float(d),
                "gap_to_leader": float(d - 1),
            })
    return pl.DataFrame(rows)


def test_group_sizes_counts_rows_per_group():
    df = _toy_snapshots(2)
    sizes = group_sizes(df)
    assert sizes == [4, 4]   # 4 drivers per (race, lap) group


def test_train_and_predict_learns_obvious_signal():
    df = _toy_snapshots(8)
    model = train_baseline(df, feature_columns=["position", "gap_to_leader"],
                           params={"num_leaves": 7, "min_data_in_leaf": 1, "n_estimators": 20})
    scores = predict(model, df, ["position", "gap_to_leader"])
    # Within a group, the lowest 'position' (driver 1) should get the highest score.
    df = df.with_columns(pl.Series("score", scores))
    g0 = df.filter(pl.col("session_key") == 0).sort("score", descending=True)
    assert g0["driver_number"][0] == 1


def test_run_baseline_end_to_end(tmp_path):
    # Synthetic snapshots with a learnable signal; verify the run dir + metrics.
    def write(split, n_races, start):
        rows = []
        for r in range(start, start + n_races):
            for d in range(1, 6):
                rows.append({"session_key": r, "snapshot_lap": 30, "driver_number": d,
                             "final_position": d, "relevance": 21 - d,
                             "position": float(d), "gap_to_leader": float(d - 1)})
        pl.DataFrame(rows).write_parquet(snap_dir / f"{split}.parquet")

    snap_dir = tmp_path / "snapshots"; snap_dir.mkdir()
    write("train", 20, 0); write("val", 4, 100); write("test", 4, 200)
    (snap_dir / "metadata.json").write_text(json.dumps({
        "feature_columns": ["position", "gap_to_leader"], "data_version": "test"
    }))

    runs_dir = tmp_path / "runs"
    result = run_baseline(snap_dir, runs_dir, params={"num_leaves": 7, "min_data_in_leaf": 1, "n_estimators": 30}, use_mlflow=False)

    run_dir = Path(result["run_dir"])
    assert (run_dir / "model.lgb").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "predictions_test.parquet").exists()
    # The signal is perfectly learnable -> strong positive test Spearman.
    assert result["metrics"]["spearman"] > 0.9
