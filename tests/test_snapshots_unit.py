"""Unit tests for Stage 4 (snapshots) and the run_pipeline helper."""
import json
import numpy as np
import polars as pl
import pytest

from f1_predictor.snapshots import assign_split, extract_snapshots, RELEVANCE_BASE
from f1_predictor.snapshots import fit_scaler, apply_scaler
from f1_predictor.snapshots import build_snapshots


def _mini_features() -> pl.DataFrame:
    # 2 drivers, laps 1..3. Driver 2 has no row at lap 3 (retired earlier).
    return pl.DataFrame({
        "session_key": [900, 900, 900, 900, 900],
        "driver_number": [1, 1, 1, 2, 2],
        "lap_number": [1, 2, 3, 1, 2],
        "final_position": [1, 1, 1, 2, 2],
        "position": [1, 1, 1, 2, 2],
        "gap_to_leader": [0.0, 0.0, 0.0, 1.0, 1.5],
    })


def test_extract_snapshots_picks_snapshot_laps_only():
    snaps = extract_snapshots(_mini_features(), snapshot_laps=[2, 3], feature_columns=["position", "gap_to_leader"])
    assert set(snaps["snapshot_lap"].unique().to_list()) == {2, 3}
    # Lap 2: both drivers active -> 2 rows. Lap 3: only driver 1 -> 1 row.
    assert snaps.filter(pl.col("snapshot_lap") == 2).height == 2
    assert snaps.filter(pl.col("snapshot_lap") == 3).height == 1


def test_extract_snapshots_relevance_is_21_minus_final_position():
    snaps = extract_snapshots(_mini_features(), snapshot_laps=[2], feature_columns=["position"])
    d1 = snaps.filter(pl.col("driver_number") == 1)
    d2 = snaps.filter(pl.col("driver_number") == 2)
    assert d1["relevance"][0] == RELEVANCE_BASE - 1   # final_position 1
    assert d2["relevance"][0] == RELEVANCE_BASE - 2   # final_position 2


def test_extract_snapshots_carries_keys_and_features():
    snaps = extract_snapshots(_mini_features(), snapshot_laps=[2], feature_columns=["position", "gap_to_leader"])
    for c in ["session_key", "snapshot_lap", "driver_number", "final_position", "relevance", "position", "gap_to_leader"]:
        assert c in snaps.columns


def test_assign_split_uses_two_date_boundaries():
    from f1_predictor.snapshots import assign_split
    vs, ts = "2025-09-01", "2026-01-01"
    assert assign_split("2023-03-05T15:00:00+00:00", vs, ts) == "train"
    assert assign_split("2024-09-01T13:00:00+00:00", vs, ts) == "train"
    assert assign_split("2025-04-01T13:00:00+00:00", vs, ts) == "train"  # early 2025 -> train
    assert assign_split("2025-10-01T13:00:00+00:00", vs, ts) == "val"    # late 2025 -> val
    assert assign_split("2026-03-15T13:00:00+00:00", vs, ts) == "test"   # 2026 -> test


def test_fit_scaler_uses_train_only_and_imputes_nulls():
    train = pl.DataFrame({"a": [0.0, 2.0, 4.0], "b": [None, None, None]})
    params = fit_scaler(train, feature_columns=["a", "b"])
    # mean(a)=2, std(a)=sqrt(8/3); b is all-null -> imputed 0 -> mean 0, scale 1.
    assert params["mean"]["a"] == pytest.approx(2.0)
    assert params["scale"]["a"] == pytest.approx(np.std([0.0, 2.0, 4.0]))
    assert params["mean"]["b"] == pytest.approx(0.0)
    assert params["scale"]["b"] == pytest.approx(1.0)  # zero-variance -> scale 1


def test_apply_scaler_standardises_and_passes_through_constant():
    train = pl.DataFrame({"a": [0.0, 2.0, 4.0], "b": [None, None, None]})
    params = fit_scaler(train, ["a", "b"])
    out = apply_scaler(pl.DataFrame({"a": [2.0], "b": [5.0]}), params, ["a", "b"])
    assert out["a"][0] == pytest.approx(0.0)       # (2-2)/std = 0
    # b had scale 1, mean 0 -> passes 5.0 through unchanged (the 2024 skew case)
    assert out["b"][0] == pytest.approx(5.0)


def test_apply_scaler_imputes_nulls_before_scaling():
    train = pl.DataFrame({"a": [0.0, 2.0, 4.0]})
    params = fit_scaler(train, ["a"])
    out = apply_scaler(pl.DataFrame({"a": [None]}), params, ["a"])
    # null -> 0 -> (0-2)/std
    assert out["a"][0] == pytest.approx((0.0 - 2.0) / np.std([0.0, 2.0, 4.0]))


def _write_feature_file(features_dir, raw_dir, key, date, n_laps=4):
    features_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / str(key)).mkdir(parents=True, exist_ok=True)
    # minimal sessions.parquet for the date
    pl.DataFrame({"date_start": [date], "circuit_short_name": ["X"]}).write_parquet(
        raw_dir / str(key) / "sessions.parquet"
    )
    rows = []
    for d in (1, 2):
        for lap in range(1, n_laps + 1):
            rows.append({"session_key": key, "driver_number": d, "lap_number": lap,
                         "final_position": d, "position": d, "gap_to_leader": float(d)})
    pl.DataFrame(rows).write_parquet(features_dir / f"{key}.parquet")


def test_build_snapshots_writes_splits_and_metadata(tmp_path):
    features_dir = tmp_path / "features"
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "snapshots"
    _write_feature_file(features_dir, raw_dir, 700, "2023-05-01T13:00:00+00:00")  # train
    _write_feature_file(features_dir, raw_dir, 800, "2024-03-01T13:00:00+00:00")  # val
    _write_feature_file(features_dir, raw_dir, 900, "2024-09-01T13:00:00+00:00")  # test

    build_snapshots(
        features_dir=features_dir, raw_dir=raw_dir, out_dir=out_dir,
        feature_columns=["position", "gap_to_leader"],
        snapshot_laps=[2, 4], val_start="2024-01-01", test_start="2024-07-01", git_sha="deadbeef",
    )

    for split in ("train", "val", "test"):
        assert (out_dir / f"{split}.parquet").exists()
    meta = json.loads((out_dir / "metadata.json").read_text())
    assert meta["feature_columns"] == ["position", "gap_to_leader"]
    assert meta["splits"]["train"] == [700]
    assert meta["splits"]["test"] == [900]
    assert "data_version" in meta
    # Train 'position' is standardised -> mean ~0
    train = pl.read_parquet(out_dir / "train.parquet")
    assert abs(train["position"].mean()) < 1e-9


def test_run_pipeline_lists_session_keys(tmp_path):
    # run_pipeline.discover_sessions returns the integer keys of raw sessions
    # that have a meta.json (fully pulled), ignoring partial dirs.
    from scripts.run_pipeline import discover_sessions

    (tmp_path / "9001").mkdir()
    (tmp_path / "9001" / "meta.json").write_text("{}")
    (tmp_path / "9002").mkdir()  # no meta.json -> skipped
    keys = discover_sessions(tmp_path)
    assert keys == [9001]
