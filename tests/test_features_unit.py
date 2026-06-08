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
