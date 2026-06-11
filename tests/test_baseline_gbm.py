import json
from pathlib import Path

import polars as pl
import numpy as np
import pytest
from f1_predictor.models.baseline_gbm import (
    add_current_rank,
    train_baseline,
    predict,
    run_baseline,
)


def test_add_current_rank_recovers_order_from_scaled_position():
    # `position` is stored standardised in snapshots but stays monotonic within a
    # group, so an ordinal rank must recover the 1..N current race order.
    df = pl.DataFrame({
        "session_key": [1, 1, 1],
        "snapshot_lap": [30, 30, 30],
        "driver_number": [44, 1, 16],
        "position": [-1.2, 0.3, 1.5],
        "final_position": [1, 2, 3],
    })
    out = add_current_rank(df)
    ranks = dict(zip(out["driver_number"].to_list(), out["current_rank"].to_list()))
    assert ranks == {44: 1, 1: 2, 16: 3}


def _reshuffle_snapshots(n_races=8):
    # Current order d=1..5 (position d); the race fully reverses, so the final
    # order is 6 - d. A `pace` feature carries the eventual finish, forcing the
    # model to learn non-zero deltas rather than just echoing current position.
    rows = []
    for r in range(n_races):
        for d in range(1, 6):
            final = 6 - d
            rows.append({
                "session_key": r, "snapshot_lap": 30, "driver_number": d,
                "position": float(d), "final_position": final, "pace": float(final),
            })
    return pl.DataFrame(rows)


def test_predict_scores_rank_by_predicted_final_position():
    df = _reshuffle_snapshots(8)
    feats = ["position", "pace"]
    model = train_baseline(df, feats, params={"num_leaves": 7, "min_data_in_leaf": 1, "n_estimators": 60})
    scores = predict(model, df, feats)
    ranked = df.with_columns(pl.Series("score", scores)).filter(
        pl.col("session_key") == 0
    ).sort("score", descending=True)
    # Highest reconstructed score must be the driver who finishes P1.
    assert ranked["final_position"][0] == 1


def test_predict_with_no_movement_reproduces_naive_order():
    # final_position == current position => every delta is 0; the reconstructed
    # score must collapse to -current_rank (the naive persistence baseline).
    rows = [
        {"session_key": 0, "snapshot_lap": 30, "driver_number": d,
         "position": float(d), "final_position": d, "pace": 0.0}
        for d in range(1, 6)
    ]
    df = pl.DataFrame(rows)
    feats = ["position", "pace"]
    model = train_baseline(df, feats, params={"num_leaves": 7, "min_data_in_leaf": 1, "n_estimators": 30})
    scores = predict(model, df, feats)
    order = df.with_columns(pl.Series("score", scores)).sort("score", descending=True)
    assert order["final_position"].to_list() == [1, 2, 3, 4, 5]


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
