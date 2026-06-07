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


def test_join_intervals_accepts_datetime_date_start():
    # In the full pipeline _join_positions runs first and converts date_start to
    # datetime in place; _join_intervals (via _lap_end_times) must accept an
    # already-datetime date_start instead of re-parsing it as a string.
    laps = make_base_laps().with_columns(
        pl.col("date_start")
        .str.to_datetime(format="%Y-%m-%dT%H:%M:%S%.f%:z", time_unit="us")
        .cast(pl.Datetime("us", "UTC"))
    )
    intervals = pl.DataFrame({
        "driver_number": [1, 44],
        "date": ["2023-03-30T14:01:25+00:00", "2023-03-30T14:01:27+00:00"],
        "gap_to_leader": [0.0, 1.234],
        "interval": [None, 1.234],
    })
    result = _join_intervals(laps, intervals)
    assert "gap_to_leader" in result.columns
    assert result.shape[0] == 6


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


from f1_predictor.sessionise import _join_stints, _add_pit_from_stints


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


def test_add_pit_from_stints_derives_flags():
    # After _join_stints the lap table carries stint_number. A new stint
    # (number > 1) starting IS a pit stop; stops_completed = stint_number - 1.
    laps = pl.DataFrame({
        "session_key": [9161] * 4,
        "driver_number": [1, 1, 1, 1],
        "lap_number": [1, 2, 3, 4],
        "stint_number": [1, 2, 2, 2],
    })
    result = _add_pit_from_stints(laps).sort("lap_number")

    assert "pit_this_lap" in result.columns
    assert "stops_completed" in result.columns
    assert result["pit_this_lap"].to_list() == [False, True, False, False]
    assert result["stops_completed"].to_list() == [0, 1, 1, 1]


def test_add_pit_from_stints_two_stops():
    laps = pl.DataFrame({
        "session_key": [9161] * 6,
        "driver_number": [1] * 6,
        "lap_number": [1, 2, 3, 4, 5, 6],
        "stint_number": [1, 1, 2, 2, 3, 3],
    })
    result = _add_pit_from_stints(laps).sort("lap_number")
    assert result["pit_this_lap"].to_list() == [False, False, True, False, True, False]
    assert result["stops_completed"].to_list() == [0, 0, 1, 1, 2, 2]


def test_add_pit_from_stints_no_stops():
    laps = pl.DataFrame({
        "session_key": [9161] * 3,
        "driver_number": [1] * 3,
        "lap_number": [1, 2, 3],
        "stint_number": [1, 1, 1],
    })
    result = _add_pit_from_stints(laps).sort("lap_number")
    assert result["pit_this_lap"].to_list() == [False, False, False]
    assert result["stops_completed"].to_list() == [0, 0, 0]


def test_add_pit_from_stints_independent_per_driver():
    laps = pl.DataFrame({
        "session_key": [9161] * 4,
        "driver_number": [1, 1, 44, 44],
        "lap_number": [1, 2, 1, 2],
        "stint_number": [1, 2, 1, 1],
    })
    result = _add_pit_from_stints(laps).sort(["driver_number", "lap_number"])
    # driver 1 pits on lap 2; driver 44 never pits
    assert result["pit_this_lap"].to_list() == [False, True, False, False]
    assert result["stops_completed"].to_list() == [0, 1, 0, 0]


def test_join_positions_empty_input():
    result = _join_positions(make_base_laps(), pl.DataFrame())
    assert "position" in result.columns
    assert result.shape[0] == 6
    assert result["position"].null_count() == 6


def test_join_intervals_empty_input():
    result = _join_intervals(make_base_laps(), pl.DataFrame())
    assert "gap_to_leader" in result.columns
    assert "interval_to_ahead" in result.columns
    assert result.shape[0] == 6


def test_join_stints_empty_input():
    result = _join_stints(make_base_laps(), pl.DataFrame())
    assert "tyre_compound" in result.columns
    assert "tyre_age_laps" in result.columns
    assert "stint_number" in result.columns
    assert result.shape[0] == 6


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


from f1_predictor.sessionise import _add_race_control_flags


def make_rc_events(*rows: dict) -> pl.DataFrame:
    return pl.DataFrame(rows, schema={
        "lap_number": pl.Int64,
        "category": pl.Utf8,
        "message": pl.Utf8,
        "flag": pl.Utf8,
    })


def make_5lap_table() -> pl.DataFrame:
    return pl.DataFrame({
        "session_key": [9161] * 5,
        "driver_number": [1] * 5,
        "lap_number": list(range(1, 6)),
        "date_start": ["2023-03-30T14:00:00+00:00"] * 5,
        "lap_time": [90.0] * 5,
    })


def test_sc_active_marks_correct_laps():
    laps = make_5lap_table()
    rc = make_rc_events(
        {"lap_number": 2, "category": "SafetyCar", "message": "SAFETY CAR DEPLOYED", "flag": "SC"},
        {"lap_number": 4, "category": "SafetyCar", "message": "SAFETY CAR IN THIS LAP", "flag": "SC"},
    )
    result = _add_race_control_flags(laps, rc)
    sc = result.sort("lap_number")["sc_active"].to_list()
    assert sc == [False, True, True, True, False]


def test_laps_since_sc_end_increments():
    laps = make_5lap_table()
    rc = make_rc_events(
        {"lap_number": 1, "category": "SafetyCar", "message": "SAFETY CAR DEPLOYED", "flag": "SC"},
        {"lap_number": 2, "category": "SafetyCar", "message": "SAFETY CAR IN THIS LAP", "flag": "SC"},
    )
    result = _add_race_control_flags(laps, rc).sort("lap_number")
    lssce = result["laps_since_sc_end"].to_list()
    assert lssce == [0, 0, 1, 2, 3]


def test_vsc_and_red_flag_active():
    laps = make_5lap_table()
    rc = make_rc_events(
        {"lap_number": 3, "category": "SafetyCar", "message": "VIRTUAL SAFETY CAR DEPLOYED", "flag": "VSC"},
        {"lap_number": 4, "category": "SafetyCar", "message": "VIRTUAL SAFETY CAR ENDING", "flag": "VSC"},
        {"lap_number": 5, "category": "Flag", "message": "RED FLAG", "flag": "RED"},
    )
    result = _add_race_control_flags(laps, rc).sort("lap_number")
    assert result["vsc_active"].to_list() == [False, False, True, True, False]
    assert result["red_flag_active"].to_list() == [False, False, False, False, True]


from f1_predictor.sessionise import _add_retirements


def _session_result(rows: list[dict]) -> pl.DataFrame:
    """Build a session_result frame mirroring OpenF1's real schema."""
    return pl.DataFrame(
        rows,
        schema={
            "driver_number": pl.Int64,
            "position": pl.Int64,         # null for DNF/DNS/DSQ
            "number_of_laps": pl.Int64,
            "dnf": pl.Boolean,
            "dns": pl.Boolean,
            "dsq": pl.Boolean,
        },
    )


def test_add_retirements_marks_dnf_drivers():
    laps = pl.DataFrame({
        "session_key": [9161] * 6,
        "driver_number": [1, 1, 1, 44, 44, 44],
        "lap_number": [1, 2, 3, 1, 2, 3],
        "date_start": ["2023-01-01T14:00:00+00:00"] * 6,
        "lap_time": [90.0] * 6,
    })
    # Driver 44 retired (DNF) → null position in OpenF1 → ranked after finishers.
    session_result = _session_result([
        {"driver_number": 1, "position": 1, "number_of_laps": 3, "dnf": False, "dns": False, "dsq": False},
        {"driver_number": 44, "position": None, "number_of_laps": 3, "dnf": True, "dns": False, "dsq": False},
    ])
    result = _add_retirements(laps, session_result)
    assert "is_retired" in result.columns
    assert "retirement_lap" in result.columns
    assert "final_position" in result.columns

    driver44 = result.filter(pl.col("driver_number") == 44)
    assert driver44["is_retired"].unique().to_list() == [True]
    assert driver44["retirement_lap"].unique().to_list() == [3]
    # Ranked immediately after the single finisher (max classified position 1).
    assert driver44["final_position"].unique().to_list() == [2]

    driver1 = result.filter(pl.col("driver_number") == 1)
    assert driver1["is_retired"].unique().to_list() == [False]
    assert driver1["final_position"].unique().to_list() == [1]
    assert driver1["retirement_lap"].unique().to_list() == [None]


def test_add_retirements_retirement_lap_is_last_lap():
    laps = pl.DataFrame({
        "session_key": [9161] * 4,
        "driver_number": [55, 55, 55, 55],
        "lap_number": [1, 2, 3, 4],
        "date_start": ["2023-01-01T14:00:00+00:00"] * 4,
        "lap_time": [90.0] * 4,
    })
    session_result = _session_result([
        {"driver_number": 55, "position": None, "number_of_laps": 4, "dnf": True, "dns": False, "dsq": False},
    ])
    result = _add_retirements(laps, session_result)
    row = result.filter(pl.col("driver_number") == 55)
    assert row["retirement_lap"].unique()[0] == 4
    # Only retiree, no finishers → ranked first among unclassified.
    assert row["final_position"].unique()[0] == 1


def test_add_retirements_ranks_unclassified_by_laps():
    # One finisher and two DNFs; the DNF with more laps gets the better position.
    laps = pl.DataFrame({
        "session_key": [9161] * 9,
        "driver_number": [1, 1, 1, 20, 20, 20, 10, 10, 10],
        "lap_number": [1, 2, 3, 1, 2, 3, 1, 2, 3],
        "date_start": ["2023-01-01T14:00:00+00:00"] * 9,
        "lap_time": [90.0] * 9,
    })
    session_result = _session_result([
        {"driver_number": 1, "position": 1, "number_of_laps": 57, "dnf": False, "dns": False, "dsq": False},
        {"driver_number": 20, "position": None, "number_of_laps": 40, "dnf": True, "dns": False, "dsq": False},
        {"driver_number": 10, "position": None, "number_of_laps": 12, "dnf": True, "dns": False, "dsq": False},
    ])
    result = _add_retirements(laps, session_result)
    fp = {r["driver_number"]: r["final_position"]
          for r in result.select(["driver_number", "final_position"]).unique().iter_rows(named=True)}
    assert fp[1] == 1
    assert fp[20] == 2   # 40 laps → ranked above the 12-lap retiree
    assert fp[10] == 3
