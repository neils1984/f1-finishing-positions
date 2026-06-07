"""Unit tests for internal sessionise helpers using tiny synthetic DataFrames."""
import polars as pl
import pytest
from f1_predictor.sessionise import _build_lap_table


def test_build_lap_table_selects_correct_columns():
    laps = pl.DataFrame({
        "session_key": [9161, 9161, 9161],
        "driver_number": [1, 1, 44],
        "lap_number": [1, 2, 1],
        "date_start": [
            "2023-03-30T14:00:00+00:00",
            "2023-03-30T14:01:30+00:00",
            "2023-03-30T14:00:00+00:00",
        ],
        "lap_duration": [90.123, 88.456, 91.001],
        "i1_speed": [280, 282, 278],     # extra column that should be dropped
    })

    result = _build_lap_table(laps)

    assert "session_key" in result.columns
    assert "driver_number" in result.columns
    assert "lap_number" in result.columns
    assert "date_start" in result.columns
    assert "lap_time" in result.columns
    assert "i1_speed" not in result.columns, "extra columns must be dropped"


def test_build_lap_table_renames_lap_duration():
    laps = pl.DataFrame({
        "session_key": [9161],
        "driver_number": [1],
        "lap_number": [1],
        "date_start": ["2023-03-30T14:00:00+00:00"],
        "lap_duration": [90.5],
    })
    result = _build_lap_table(laps)
    assert "lap_time" in result.columns
    assert "lap_duration" not in result.columns


def test_build_lap_table_filters_out_lap_zero():
    laps = pl.DataFrame({
        "session_key": [9161, 9161],
        "driver_number": [1, 1],
        "lap_number": [0, 1],
        "date_start": ["2023-03-30T13:58:00+00:00", "2023-03-30T14:00:00+00:00"],
        "lap_duration": [None, 90.5],
    })
    result = _build_lap_table(laps)
    assert result["lap_number"].min() == 1, "lap 0 (formation lap) must be excluded"
