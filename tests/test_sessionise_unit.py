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


from f1_predictor.sessionise import _join_positions, _join_intervals


def make_base_laps() -> pl.DataFrame:
    """Minimal 3-lap, 2-driver lap table for join tests."""
    return pl.DataFrame({
        "session_key": [9161] * 6,
        "driver_number": [1, 1, 1, 44, 44, 44],
        "lap_number": [1, 2, 3, 1, 2, 3],
        "date_start": [
            "2023-03-30T14:00:00+00:00",
            "2023-03-30T14:01:30+00:00",
            "2023-03-30T14:03:00+00:00",
            "2023-03-30T14:00:00+00:00",
            "2023-03-30T14:01:32+00:00",
            "2023-03-30T14:03:02+00:00",
        ],
        "lap_time": [90.0, 88.5, 87.2, 92.0, 91.5, 90.1],
    })


def test_join_positions_adds_position_column():
    laps = make_base_laps()
    pos = pl.DataFrame({
        "driver_number": [1, 1, 44, 44],
        "date": [
            "2023-03-30T14:00:05+00:00",   # lap 1 of driver 1
            "2023-03-30T14:01:35+00:00",   # lap 2 of driver 1
            "2023-03-30T14:00:05+00:00",   # lap 1 of driver 44
            "2023-03-30T14:01:37+00:00",   # lap 2 of driver 44
        ],
        "position": [1, 1, 2, 2],
    })
    result = _join_positions(laps, pos)
    assert "position" in result.columns
    assert result.shape[0] == 6  # row count unchanged


def test_join_positions_picks_last_reading_before_lap_end():
    laps = make_base_laps()
    # Driver 1 lap 1 ends at ~14:01:30. Position changes twice during lap 1.
    pos = pl.DataFrame({
        "driver_number": [1, 1, 1],
        "date": [
            "2023-03-30T14:00:10+00:00",  # early in lap 1, position 3
            "2023-03-30T14:01:20+00:00",  # late in lap 1, position 1 (overtook)
            "2023-03-30T14:01:40+00:00",  # during lap 2
        ],
        "position": [3, 1, 1],
    })
    result = _join_positions(laps, pos)
    driver1_lap1 = result.filter(
        (pl.col("driver_number") == 1) & (pl.col("lap_number") == 1)
    )
    assert driver1_lap1["position"][0] == 1, "should pick position 1 (last reading in lap 1)"


def test_join_intervals_adds_gap_columns():
    laps = make_base_laps()
    intervals = pl.DataFrame({
        "driver_number": [1, 44],
        "date": ["2023-03-30T14:01:25+00:00", "2023-03-30T14:01:27+00:00"],
        "gap_to_leader": [0.0, 1.234],
        "interval": [None, 1.234],
    })
    result = _join_intervals(laps, intervals)
    assert "gap_to_leader" in result.columns
    assert "interval_to_ahead" in result.columns
