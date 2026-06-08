"""Unit tests for Stage 4 (snapshots) and the run_pipeline helper."""
import polars as pl
import pytest


def test_run_pipeline_lists_session_keys(tmp_path):
    # run_pipeline.discover_sessions returns the integer keys of raw sessions
    # that have a meta.json (fully pulled), ignoring partial dirs.
    from scripts.run_pipeline import discover_sessions

    (tmp_path / "9001").mkdir()
    (tmp_path / "9001" / "meta.json").write_text("{}")
    (tmp_path / "9002").mkdir()  # no meta.json -> skipped
    keys = discover_sessions(tmp_path)
    assert keys == [9001]
