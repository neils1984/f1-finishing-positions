"""Unit tests for Stage 3 per-race feature transforms using synthetic data."""
import polars as pl
import pytest
from f1_predictor.features import load_circuits, circuit_length_km, is_street_circuit


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
