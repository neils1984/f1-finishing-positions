"""Unit tests for Stage 4 (snapshots) and the run_pipeline helper."""
import polars as pl
import pytest

from f1_predictor.snapshots import assign_split


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


def test_run_pipeline_lists_session_keys(tmp_path):
    # run_pipeline.discover_sessions returns the integer keys of raw sessions
    # that have a meta.json (fully pulled), ignoring partial dirs.
    from scripts.run_pipeline import discover_sessions

    (tmp_path / "9001").mkdir()
    (tmp_path / "9001" / "meta.json").write_text("{}")
    (tmp_path / "9002").mkdir()  # no meta.json -> skipped
    keys = discover_sessions(tmp_path)
    assert keys == [9001]
