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
