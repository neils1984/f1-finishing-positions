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


from f1_predictor.sessionise import _join_stints, _join_pit


def test_join_stints_adds_tyre_columns():
    laps = pl.DataFrame({
        "session_key": [9161] * 4,
        "driver_number": [1, 1, 1, 1],
        "lap_number": [1, 2, 3, 4],
        "date_start": ["2023-01-01T14:00:00+00:00"] * 4,
        "lap_time": [90.0] * 4,
    })
    stints = pl.DataFrame({
        "driver_number": [1, 1],
        "stint_number": [1, 2],
        "lap_start": [1, 3],
        "lap_end": [2, 4],
        "compound": ["SOFT", "MEDIUM"],
        "tyre_age_at_start": [0, 0],
    })
    result = _join_stints(laps, stints)

    assert "tyre_compound" in result.columns
    assert "tyre_age_laps" in result.columns
    assert "stint_number" in result.columns
    assert result.shape[0] == 4

    compounds = result.sort("lap_number")["tyre_compound"].to_list()
    assert compounds == ["SOFT", "SOFT", "MEDIUM", "MEDIUM"]


def test_join_stints_tyre_age_resets_at_stint_boundary():
    laps = pl.DataFrame({
        "session_key": [9161] * 3,
        "driver_number": [1, 1, 1],
        "lap_number": [1, 2, 3],
        "date_start": ["2023-01-01T14:00:00+00:00"] * 3,
        "lap_time": [90.0] * 3,
    })
    stints = pl.DataFrame({
        "driver_number": [1, 1],
        "stint_number": [1, 2],
        "lap_start": [1, 3],
        "lap_end": [2, 3],
        "compound": ["SOFT", "MEDIUM"],
        "tyre_age_at_start": [0, 0],
    })
    result = _join_stints(laps, stints).sort("lap_number")
    assert result["tyre_age_laps"].to_list() == [0, 1, 0]


def test_join_pit_adds_pit_flags():
    laps = pl.DataFrame({
        "session_key": [9161] * 4,
        "driver_number": [1, 1, 1, 1],
        "lap_number": [1, 2, 3, 4],
        "date_start": ["2023-01-01T14:00:00+00:00"] * 4,
        "lap_time": [90.0] * 4,
    })
    pit = pl.DataFrame({
        "driver_number": [1],
        "lap_number": [2],
        "pit_duration": [22.5],
    })
    result = _join_pit(laps, pit)

    assert "pit_this_lap" in result.columns
    assert "stops_completed" in result.columns
    row = result.sort("lap_number")
    assert row["pit_this_lap"].to_list() == [False, True, False, False]
    assert row["stops_completed"].to_list() == [0, 1, 1, 1]


from f1_predictor.sessionise import _add_car_data


def test_add_car_data_max_speed_per_lap():
    laps = pl.DataFrame({
        "session_key": [9161] * 2,
        "driver_number": [1, 1],
        "lap_number": [1, 2],
        "date_start": [
            "2023-03-30T14:00:00+00:00",
            "2023-03-30T14:01:30+00:00",
        ],
        "lap_time": [90.0, 88.5],
    })
    car = pl.DataFrame({
        "driver_number": [1, 1, 1, 1],
        "date": [
            "2023-03-30T14:00:10+00:00",  # lap 1
            "2023-03-30T14:00:50+00:00",  # lap 1
            "2023-03-30T14:01:40+00:00",  # lap 2
            "2023-03-30T14:02:10+00:00",  # lap 2
        ],
        "speed": [250, 310, 290, 320],
    })
    result = _add_car_data(laps, car)
    assert "max_speed_kmh" in result.columns
    row = result.sort("lap_number")
    assert row["max_speed_kmh"][0] == 310  # lap 1
    assert row["max_speed_kmh"][1] == 320  # lap 2
