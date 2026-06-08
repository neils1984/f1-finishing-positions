"""Unit tests for Stage 4 (snapshots) and the run_pipeline helper."""
import numpy as np
import polars as pl
import pytest

from f1_predictor.snapshots import assign_split, extract_snapshots, RELEVANCE_BASE
from f1_predictor.snapshots import fit_scaler, apply_scaler


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


def test_assign_split_2023_is_train():
    assert assign_split("2023-03-05T15:00:00+00:00", "2024-07-01") == "train"


def test_assign_split_2024_before_cutoff_is_val():
    assert assign_split("2024-03-02T15:00:00+00:00", "2024-07-01") == "val"


def test_assign_split_2024_on_or_after_cutoff_is_test():
    assert assign_split("2024-07-07T13:00:00+00:00", "2024-07-01") == "test"
    assert assign_split("2024-07-01T00:00:00+00:00", "2024-07-01") == "test"


def test_assign_split_pre_2023_is_train():
    # Any race earlier than the val season counts as train.
    assert assign_split("2022-11-20T13:00:00+00:00", "2024-07-01") == "train"


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


def test_run_pipeline_lists_session_keys(tmp_path):
    # run_pipeline.discover_sessions returns the integer keys of raw sessions
    # that have a meta.json (fully pulled), ignoring partial dirs.
    from scripts.run_pipeline import discover_sessions

    (tmp_path / "9001").mkdir()
    (tmp_path / "9001" / "meta.json").write_text("{}")
    (tmp_path / "9002").mkdir()  # no meta.json -> skipped
    keys = discover_sessions(tmp_path)
    assert keys == [9001]
