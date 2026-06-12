"""Unit tests for the 2026 regime-drift diagnostic (synthetic snapshots)."""
import json

import polars as pl

from f1_predictor.drift import degradation_report, inert_features


def _write_split(path, n_races, start, *, era_flag, learnable):
    """Synthetic snapshots. With learnable=True a `pace` feature carries the
    eventual finish (the race reverses the current order), so a trained model
    beats naive persistence; is_2026_regs is held constant at `era_flag`."""
    rows = []
    for r in range(start, start + n_races):
        for d in range(1, 6):
            final = (6 - d) if learnable else d
            rows.append({
                "session_key": r, "snapshot_lap": 30, "driver_number": d,
                "final_position": final, "relevance": 21 - final,
                "position": float(d), "pace": float(final),
                "is_2026_regs": float(era_flag),
            })
    pl.DataFrame(rows).write_parquet(path)


def _build(tmp_path):
    snap = tmp_path / "snapshots"; snap.mkdir()
    # train + val are the "old regime" (era 0); test is "2026" (era 1).
    _write_split(snap / "train.parquet", 20, 0, era_flag=0, learnable=True)
    _write_split(snap / "val.parquet", 4, 100, era_flag=0, learnable=True)
    _write_split(snap / "test.parquet", 4, 200, era_flag=1, learnable=True)
    (snap / "metadata.json").write_text(json.dumps({
        "feature_columns": ["position", "pace", "is_2026_regs"],
        "data_version": "test",
    }))
    return snap


def test_inert_features_flags_constant_train_column():
    train = pl.DataFrame({
        "position": [1.0, 2.0, 3.0, 4.0],
        "pace": [4.0, 3.0, 2.0, 1.0],
        "is_2026_regs": [0.0, 0.0, 0.0, 0.0],  # constant in train -> inert
    })
    assert inert_features(train, ["position", "pace", "is_2026_regs"]) == ["is_2026_regs"]


def test_degradation_report_structure_and_inert_era_flag(tmp_path):
    snap = _build(tmp_path)
    report = degradation_report(snap, params={
        "num_leaves": 7, "min_data_in_leaf": 1, "n_estimators": 40,
    })

    # Both splits reported with naive + model + uplift and per-lap breakdown.
    for split in ("val", "test"):
        s = report["splits"][split]
        assert {"naive", "model", "uplift", "per_lap"} <= set(s)
        assert s["per_lap"][0]["snapshot_lap"] == 30

    # The era flag is constant in train -> structurally inert, and the trees
    # cannot have used it (zero gain importance). This is the headline finding.
    assert report["is_2026_regs_inert"] is True
    assert "is_2026_regs" in report["inert_train_features"]
    assert report["is_2026_regs_gain_importance"] == 0.0

    # The val->test drift summary is present when both splits exist.
    assert "uplift_retained" in report["val_to_test"]


def test_model_beats_naive_on_learnable_signal(tmp_path):
    snap = _build(tmp_path)
    report = degradation_report(snap, params={
        "num_leaves": 7, "min_data_in_leaf": 1, "n_estimators": 40,
    })
    # The reversal signal is learnable from `pace`, so the model's uplift over
    # naive persistence must be positive on the held-out val split.
    assert report["splits"]["val"]["uplift"] > 0.0
