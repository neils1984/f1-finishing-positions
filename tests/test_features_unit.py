"""Unit tests for Stage 3 per-race feature transforms using synthetic data."""
import polars as pl
import pytest
from f1_predictor.features import load_circuits, circuit_length_km, is_street_circuit
from f1_predictor.features import _parse_gap_columns


def test_load_circuits_reads_yaml():
    circuits = load_circuits()
    assert "lengths_km" in circuits
    assert "street" in circuits
    assert circuits["lengths_km"]["Sakhir"] == pytest.approx(5.412)


def test_circuit_length_km_known_circuit():
    circuits = load_circuits()
    assert circuit_length_km("Baku", circuits) == pytest.approx(6.003)


def test_circuit_length_km_unknown_returns_none():
    circuits = load_circuits()
    assert circuit_length_km("Nowhere", circuits) is None


def test_is_street_circuit_flags():
    circuits = load_circuits()
    assert is_street_circuit("Baku", circuits) is True
    assert is_street_circuit("Silverstone", circuits) is False


def test_parse_gap_columns_numeric_and_lapped():
    df = pl.DataFrame({
        "gap_to_leader": ["0.0", "1.234", "+1 LAP", "+2 LAPS", None],
        "interval_to_ahead": [None, "1.234", "0.5", "+1 LAP", ""],
    })
    out = _parse_gap_columns(df)
    assert out["gap_to_leader"].dtype == pl.Float64
    assert out["interval_to_ahead"].dtype == pl.Float64
    assert out["gap_to_leader"].to_list()[:2] == [0.0, 1.234]
    # "+1 LAP", "+2 LAPS", null, and "" all become null
    assert out["gap_to_leader"].to_list()[2] is None
    assert out["gap_to_leader"].to_list()[3] is None
    assert out["interval_to_ahead"].to_list()[4] is None


def test_parse_gap_columns_already_float_is_passthrough():
    df = pl.DataFrame({
        "gap_to_leader": [0.0, 1.5],
        "interval_to_ahead": [None, 0.3],
    })
    out = _parse_gap_columns(df)
    assert out["gap_to_leader"].to_list() == [0.0, 1.5]


from f1_predictor.features import _add_active_and_distance


def _mini_sessionised() -> pl.DataFrame:
    """3 drivers, 3 laps. Driver 3 retires on lap 2."""
    return pl.DataFrame({
        "session_key": [9000] * 9,
        "driver_number": [1, 1, 1, 2, 2, 2, 3, 3, 3],
        "lap_number": [1, 2, 3, 1, 2, 3, 1, 2, 3],
        "position": [1, 1, 1, 2, 2, 2, 3, 3, 3],
        "is_retired": [False] * 6 + [True] * 3,
        "retirement_lap": [None] * 6 + [2, 2, 2],
        "lap_time": [90.0, 89.0, 88.0, 91.0, 90.5, 90.0, 92.0, 93.0, None],
    })


def test_num_active_drivers_decreases_after_retirement():
    df = _add_active_and_distance(_mini_sessionised(), circuit_length=5.0)
    by_lap = (
        df.group_by("lap_number").agg(pl.col("num_active_drivers").first())
        .sort("lap_number")
    )
    # Driver 3 retired on lap 2, so it is inactive only from lap 3 onward.
    assert by_lap["num_active_drivers"].to_list() == [3, 3, 2]


def test_distance_remaining_km_uses_circuit_length():
    df = _add_active_and_distance(_mini_sessionised(), circuit_length=5.0)
    # total_laps = max lap_number = 3. circuit_length = 5.0 km.
    row = df.filter((pl.col("driver_number") == 1) & (pl.col("lap_number") == 1))
    assert row["distance_remaining_km"][0] == pytest.approx(5.0 * (3 - 1))
    last = df.filter((pl.col("driver_number") == 1) & (pl.col("lap_number") == 3))
    assert last["distance_remaining_km"][0] == pytest.approx(0.0)


def test_distance_remaining_km_null_when_circuit_unknown():
    df = _add_active_and_distance(_mini_sessionised(), circuit_length=None)
    assert df["distance_remaining_km"].null_count() == df.height


import math
from f1_predictor.features import _add_positions_gained, _add_pace_deltas, _add_gaps_ahead
from f1_predictor.features import _add_rolling_pace


def test_rolling_lap_time_3_norm_and_delta_leader():
    # Two drivers, 3 laps. Driver 1 is the leader (position 1) every lap.
    df = pl.DataFrame({
        "driver_number": [1, 1, 1, 2, 2, 2],
        "lap_number": [1, 2, 3, 1, 2, 3],
        "position": [1, 1, 1, 2, 2, 2],
        "lap_time": [90.0, 90.0, 90.0, 100.0, 100.0, 100.0],
    })
    out = _add_rolling_pace(df).sort(["driver_number", "lap_number"])
    # Lap 3: rolling3 driver1 = 90, driver2 = 100. field median of {90,100} = 95.
    d1_l3 = out.filter((pl.col("driver_number") == 1) & (pl.col("lap_number") == 3))
    d2_l3 = out.filter((pl.col("driver_number") == 2) & (pl.col("lap_number") == 3))
    assert d1_l3["rolling_lap_time_3_norm"][0] == pytest.approx(90.0 / 95.0)
    assert d2_l3["rolling_lap_time_3_norm"][0] == pytest.approx(100.0 / 95.0)
    # delta_leader = driver rolling - leader (position 1) rolling
    assert d1_l3["rolling_lap_time_3_delta_leader"][0] == pytest.approx(0.0)
    assert d2_l3["rolling_lap_time_3_delta_leader"][0] == pytest.approx(10.0)


def test_rolling_pace_partial_window_uses_available_laps():
    # On lap 1 the rolling mean is just that lap's time (min_periods=1).
    df = pl.DataFrame({
        "driver_number": [1, 2],
        "lap_number": [1, 1],
        "position": [1, 2],
        "lap_time": [90.0, 94.0],
    })
    out = _add_rolling_pace(df)
    d1 = out.filter(pl.col("driver_number") == 1)
    assert d1["rolling_lap_time_3_delta_leader"][0] == pytest.approx(0.0)


def test_positions_gained_from_grid_sign():
    df = pl.DataFrame({
        "driver_number": [1, 2, 3],
        "lap_number": [5, 5, 5],
        "position": [1, 2, 3],
    })
    grid = {1: 3, 2: 2, 3: 1}  # driver 1 started P3 now P1 -> gained 2
    out = _add_positions_gained(df, grid).sort("driver_number")
    assert out["positions_gained_from_grid"].to_list() == [2, 0, -2]


def test_pace_deltas_to_ahead_and_behind():
    # One lap, three cars by position with known lap_times.
    df = pl.DataFrame({
        "driver_number": [1, 2, 3],
        "lap_number": [5, 5, 5],
        "position": [1, 2, 3],
        "lap_time": [88.0, 89.5, 91.0],
    })
    out = _add_pace_deltas(df).sort("position")
    # P2 vs ahead (P1): 89.5 - 88.0 = 1.5 ; vs behind (P3): 89.5 - 91.0 = -1.5
    p2 = out.filter(pl.col("position") == 2)
    assert p2["last_lap_pace_delta_to_ahead"][0] == pytest.approx(1.5)
    assert p2["last_lap_pace_delta_to_behind"][0] == pytest.approx(-1.5)
    # Leader has no car ahead -> null ahead delta
    p1 = out.filter(pl.col("position") == 1)
    assert p1["last_lap_pace_delta_to_ahead"][0] is None
    # Last car has no car behind -> null behind delta
    p3 = out.filter(pl.col("position") == 3)
    assert p3["last_lap_pace_delta_to_behind"][0] is None


def test_pace_deltas_no_fanout_on_duplicate_position():
    # Stage 2's position asof-join can tie two drivers at the same position on a
    # lap. _add_pace_deltas must not duplicate rows when that happens.
    df = pl.DataFrame({
        "driver_number": [1, 2, 3, 4],
        "lap_number": [5, 5, 5, 5],
        "position": [1, 2, 3, 3],  # drivers 3 and 4 share position 3
        "lap_time": [88.0, 89.0, 90.0, 90.5],
    })
    out = _add_pace_deltas(df)
    assert out.height == 4, "must not fan out rows when a position is shared"
    assert "last_lap_pace_delta_to_ahead" in out.columns
    assert "last_lap_pace_delta_to_behind" in out.columns


def test_gaps_ahead_mean_and_stdev():
    # Positions 1..4 with cumulative gap_to_leader 0, 1, 3, 6 -> inter-car gaps 1,2,3.
    df = pl.DataFrame({
        "driver_number": [1, 2, 3, 4],
        "lap_number": [5, 5, 5, 5],
        "position": [1, 2, 3, 4],
        "gap_to_leader": [0.0, 1.0, 3.0, 6.0],
    })
    out = _add_gaps_ahead(df).sort("position")
    m = out["mean_gap_cars_ahead"].to_list()
    s = out["stdev_gap_cars_ahead"].to_list()
    # Leader (P1): no cars ahead -> 0, 0
    assert m[0] == pytest.approx(0.0) and s[0] == pytest.approx(0.0)
    # P2: cars ahead = {P1}; no inter-car gap -> 0, 0
    assert m[1] == pytest.approx(0.0) and s[1] == pytest.approx(0.0)
    # P3: inter-car gaps among {P1,P2} = [1] -> mean 1, stdev 0
    assert m[2] == pytest.approx(1.0) and s[2] == pytest.approx(0.0)
    # P4: inter-car gaps among {P1,P2,P3} = [1,2] -> mean 1.5, stdev 0.5 (population)
    assert m[3] == pytest.approx(1.5) and s[3] == pytest.approx(0.5)


def test_gaps_ahead_ignores_tied_positions():
    # Two drivers tie at position 2 (a Stage 2 asof-join artifact). The tie must
    # not inject a spurious ~0 inter-car gap for the car behind, and both tied
    # drivers get the same (deduped) value.
    df = pl.DataFrame({
        "driver_number": [1, 2, 3, 4],
        "lap_number": [5, 5, 5, 5],
        "position": [1, 2, 2, 3],          # drivers 2 and 3 both at P2
        "gap_to_leader": [0.0, 1.0, 1.0, 3.0],
    })
    out = _add_gaps_ahead(df).sort("driver_number")
    g = {r["driver_number"]: (r["mean_gap_cars_ahead"], r["stdev_gap_cars_ahead"])
         for r in out.iter_rows(named=True)}
    # Deduped positions {P1:0, P2:1, P3:3}; for P3 the only inter-car gap among
    # cars ahead is (1-0)=1 -> mean 1, stdev 0 (NOT contaminated by a 0 from the tie).
    assert g[4][0] == pytest.approx(1.0)
    assert g[4][1] == pytest.approx(0.0)
    # Both tied P2 drivers get P2's value (no cars-ahead gap pair) -> 0, 0.
    assert g[2] == g[3]


def test_rolling_pace_leader_tie_is_deterministic():
    # Two drivers tie at position 1 on a lap. The leader rolling time used for
    # delta must be chosen deterministically (lowest driver_number) and stable.
    df = pl.DataFrame({
        "driver_number": [2, 5, 9],
        "lap_number": [1, 1, 1],
        "position": [1, 1, 2],             # drivers 2 and 5 tie at P1
        "lap_time": [90.0, 92.0, 95.0],
    })
    out1 = _add_rolling_pace(df)
    out2 = _add_rolling_pace(df)
    d9_a = out1.filter(pl.col("driver_number") == 9)["rolling_lap_time_3_delta_leader"][0]
    d9_b = out2.filter(pl.col("driver_number") == 9)["rolling_lap_time_3_delta_leader"][0]
    assert d9_a == d9_b                       # deterministic across runs
    # Leader = lowest driver_number among P1 ties = driver 2 (rolling 90).
    assert d9_a == pytest.approx(95.0 - 90.0)


from f1_predictor.features import _add_tyre_onehot, _add_stops_vs_median
from f1_predictor.features import _grid_from_position, FEATURE_COLUMNS


def test_grid_from_position_takes_earliest_reading():
    # Driver 1's earliest reading is grid P3; later readings are mid-race.
    pos = pl.DataFrame({
        "driver_number": [1, 1, 1, 2, 2],
        "date": [
            "2023-03-05T14:01:00+00:00",  # grid
            "2023-03-05T15:03:45+00:00",  # race
            "2023-03-05T15:10:00+00:00",  # race
            "2023-03-05T14:01:00+00:00",  # grid
            "2023-03-05T15:03:45+00:00",  # race
        ],
        "position": [3, 1, 1, 1, 2],
    })
    grid = _grid_from_position(pos)
    assert grid == {1: 3, 2: 1}


def test_feature_columns_constant_complete():
    # The public column contract must list exactly the 24 features + tyre one-hots.
    for c in ["position", "positions_gained_from_grid", "distance_remaining_km",
              "tyre_soft", "tyre_wet", "stops_vs_median",
              "driver_championship_standing", "team_circuit_finish_rate"]:
        assert c in FEATURE_COLUMNS


def test_tyre_onehot_columns():
    df = pl.DataFrame({"tyre_compound": ["SOFT", "MEDIUM", "HARD", "INTER", "WET", None]})
    out = _add_tyre_onehot(df)
    for c in ["tyre_soft", "tyre_medium", "tyre_hard", "tyre_inter", "tyre_wet"]:
        assert c in out.columns
        assert out[c].dtype == pl.Int8
    assert out["tyre_soft"].to_list() == [1, 0, 0, 0, 0, 0]
    assert out["tyre_wet"].to_list() == [0, 0, 0, 0, 1, 0]
    # Unknown/null compound -> all zeros
    assert out.row(5, named=True)["tyre_hard"] == 0


def test_stops_vs_median():
    # One lap, three drivers with stops 0, 1, 2 -> median 1.
    df = pl.DataFrame({
        "lap_number": [5, 5, 5],
        "driver_number": [1, 2, 3],
        "stops_completed": [0, 1, 2],
    })
    out = _add_stops_vs_median(df).sort("driver_number")
    assert out["stops_vs_median"].to_list() == [-1.0, 0.0, 1.0]


def test_is_2026_regs_flags_regulation_era():
    # True for 2026+ races (new technical regs), False otherwise — analogous to
    # is_street_circuit, the principled slot for "this is a different regime".
    from f1_predictor.features import _regulation_era_flag
    assert _regulation_era_flag("2026-03-15T13:00:00+00:00") is True
    assert _regulation_era_flag("2025-03-15T13:00:00+00:00") is False
    assert _regulation_era_flag("2023-07-01T13:00:00+00:00") is False
