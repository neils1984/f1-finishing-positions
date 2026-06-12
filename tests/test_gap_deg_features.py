"""Unit tests for tyre-degradation slope (#3) and gap-trend (#5) features."""
import polars as pl
import pytest

from f1_predictor.features import (
    _add_tyre_deg_slope,
    _add_gap_trends,
    FEATURE_COLUMNS,
)


def _deg_frame(lap_times, ages, *, sc=None, pit=None, n=None):
    n = n or len(lap_times)
    return pl.DataFrame({
        "driver_number": [1] * n,
        "stint_number": [1] * n,
        "lap_number": list(range(1, n + 1)),
        "tyre_age_laps": ages,
        "lap_time": lap_times,
        "pit_this_lap": pit or [False] * n,
        "sc_active": sc or [False] * n,
        "vsc_active": [False] * n,
        "red_flag_active": [False] * n,
    })


def test_tyre_deg_slope_linear_and_null_until_three_laps():
    # lap_time rises exactly 1.0 s per lap of tyre age -> slope 1.0.
    df = _deg_frame([90.0, 91.0, 92.0, 93.0], [1, 2, 3, 4])
    out = _add_tyre_deg_slope(df).sort("lap_number")
    s = out["tyre_deg_slope"].to_list()
    assert s[0] is None and s[1] is None          # <3 clean laps -> null
    assert s[2] == pytest.approx(1.0)             # laps 1-3
    assert s[3] == pytest.approx(1.0)             # laps 1-4


def test_tyre_deg_slope_excludes_sc_and_pit_laps():
    # An SC lap (lap 3, 140 s) must not corrupt the slope.
    df = _deg_frame(
        [90.0, 91.0, 140.0, 93.0, 94.0], [1, 2, 3, 4, 5],
        sc=[False, False, True, False, False],
    )
    out = _add_tyre_deg_slope(df).sort("lap_number")
    # Clean points (1,90),(2,91),(4,93),(5,94) are all lt = 89 + age -> slope 1.0.
    assert out["tyre_deg_slope"].to_list()[-1] == pytest.approx(1.0)


def test_gap_trends_interval_to_behind():
    # P1,P2,P3 on one lap. interval_to_ahead = gap to the car AHEAD.
    # P1's car behind is P2, whose interval_to_ahead (0.5) IS P1's gap-to-behind.
    df = pl.DataFrame({
        "driver_number": [1, 2, 3], "lap_number": [5, 5, 5], "position": [1, 2, 3],
        "gap_to_leader": [0.0, 0.5, 1.3], "interval_to_ahead": [None, 0.5, 0.8],
    })
    out = _add_gap_trends(df).sort("position")
    behind = out["interval_to_behind"].to_list()
    assert behind[0] == pytest.approx(0.5)   # P1 <- P2's interval
    assert behind[1] == pytest.approx(0.8)   # P2 <- P3's interval
    assert behind[2] is None                 # P3 is last, no car behind


def test_gap_trends_in_striking_distance():
    df = pl.DataFrame({
        "driver_number": [1, 2, 3], "lap_number": [5, 5, 5], "position": [1, 2, 3],
        "gap_to_leader": [0.0, 0.5, 3.0], "interval_to_ahead": [None, 0.5, 2.5],
    })
    out = _add_gap_trends(df).sort("position")
    # leader null->0 ; 0.5<1 ->1 ; 2.5 ->0
    assert out["in_striking_distance"].to_list() == [0, 1, 0]


def test_gap_trends_3lap_deltas():
    # One driver over 4 laps; closing on the car ahead 2.0 -> 0.5.
    df = pl.DataFrame({
        "driver_number": [7] * 4, "lap_number": [1, 2, 3, 4], "position": [5] * 4,
        "gap_to_leader": [10.0, 10.5, 11.0, 11.6],
        "interval_to_ahead": [2.0, 1.5, 1.0, 0.5],
    })
    out = _add_gap_trends(df).sort("lap_number")
    r4 = out.filter(pl.col("lap_number") == 4)
    assert r4["interval_to_ahead_delta_3lap"][0] == pytest.approx(-1.5)  # 0.5-2.0
    assert r4["gap_to_leader_delta_3lap"][0] == pytest.approx(1.6)       # 11.6-10.0
    # lap 3 has only 2 prior laps -> shift(3) is null
    assert out.filter(pl.col("lap_number") == 3)["interval_to_ahead_delta_3lap"][0] is None
