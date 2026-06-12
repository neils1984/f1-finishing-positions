"""Speed-trap and sector-time features (Stage 2 passthrough + Stage 3 deltas).

The OpenF1 `laps` endpoint exposes st_speed (speed trap) and duration_sector_1/2/3
for every era (2023-2026), so these are era-agnostic. Stage 2 must carry the raw
columns through; Stage 3 turns them into field-relative deltas (driver minus the
per-lap field median) so circuit/era scale and track-wide effects cancel.
"""
import polars as pl
import pytest

from f1_predictor.sessionise import _build_lap_table
from f1_predictor.features import _add_speed_sector_deltas, FEATURE_COLUMNS

_SPEED_SECTOR_FEATURES = [
    "st_speed_delta_to_field",
    "sector1_time_delta_to_field",
    "sector2_time_delta_to_field",
    "sector3_time_delta_to_field",
]


# ---- Stage 2: _build_lap_table must carry the raw source columns ----

def test_build_lap_table_carries_speed_and_sector_columns():
    laps = pl.DataFrame({
        "session_key": [9161, 9161],
        "driver_number": [1, 44],
        "lap_number": [1, 1],
        "date_start": ["2024-03-30T14:00:00+00:00", "2024-03-30T14:00:00+00:00"],
        "lap_duration": [90.0, 91.0],
        "st_speed": [320, 315],
        "duration_sector_1": [29.0, 29.5],
        "duration_sector_2": [30.0, 30.2],
        "duration_sector_3": [31.0, 31.3],
        "i1_speed": [280, 278],   # not in our set -> still dropped
    })
    out = _build_lap_table(laps)
    for c in ["st_speed", "duration_sector_1", "duration_sector_2", "duration_sector_3"]:
        assert c in out.columns, f"{c} must be carried through Stage 2"
    assert "i1_speed" not in out.columns, "columns we do not feature must still be dropped"


def test_build_lap_table_fills_null_when_source_columns_absent():
    # Older / synthetic laps frames (e.g. some 2023 fixtures) may lack these
    # fields. Stage 2 must add them as null rather than crashing on .select.
    laps = pl.DataFrame({
        "session_key": [9161],
        "driver_number": [1],
        "lap_number": [1],
        "date_start": ["2023-03-30T14:00:00+00:00"],
        "lap_duration": [90.0],
    })
    out = _build_lap_table(laps)
    for c in ["st_speed", "duration_sector_1", "duration_sector_2", "duration_sector_3"]:
        assert c in out.columns
        assert out[c].null_count() == out.height


# ---- Stage 3: field-relative deltas ----

def test_speed_sector_deltas_are_field_relative():
    # One lap, three drivers (odd count avoids median interpolation).
    df = pl.DataFrame({
        "lap_number": [5, 5, 5],
        "driver_number": [1, 2, 3],
        "st_speed": [300, 310, 320],            # median 310
        "duration_sector_1": [30.0, 31.0, 32.0],  # median 31
        "duration_sector_2": [40.0, 41.0, 42.0],  # median 41
        "duration_sector_3": [50.0, 51.0, 52.0],  # median 51
    })
    out = _add_speed_sector_deltas(df).sort("driver_number")
    # Speed trap: higher than field median is positive.
    assert out["st_speed_delta_to_field"].to_list() == pytest.approx([-10.0, 0.0, 10.0])
    # Sector times: faster than field (lower time) is negative.
    assert out["sector1_time_delta_to_field"].to_list() == pytest.approx([-1.0, 0.0, 1.0])
    assert out["sector2_time_delta_to_field"].to_list() == pytest.approx([-1.0, 0.0, 1.0])
    assert out["sector3_time_delta_to_field"].to_list() == pytest.approx([-1.0, 0.0, 1.0])


def test_speed_sector_deltas_median_ignores_nulls():
    # A null source value yields a null delta for that driver; the field median is
    # computed over the non-null values only.
    df = pl.DataFrame({
        "lap_number": [5, 5, 5],
        "driver_number": [1, 2, 3],
        "st_speed": [300, 320, None],            # median of {300,320} = 310
        "duration_sector_1": [30.0, 32.0, None],
        "duration_sector_2": [30.0, 32.0, None],
        "duration_sector_3": [30.0, 32.0, None],
    })
    out = _add_speed_sector_deltas(df).sort("driver_number")
    assert out["st_speed_delta_to_field"].to_list() == pytest.approx([-10.0, 10.0, None], nan_ok=True) \
        or out["st_speed_delta_to_field"].to_list()[2] is None
    assert out["st_speed_delta_to_field"][0] == pytest.approx(-10.0)
    assert out["st_speed_delta_to_field"][2] is None


def test_speed_sector_deltas_missing_source_columns_yield_null_features():
    # If Stage 2 produced no source columns at all, the features still exist and
    # are entirely null (Stage 4 imputes them to 0).
    df = pl.DataFrame({"lap_number": [5, 5], "driver_number": [1, 2]})
    out = _add_speed_sector_deltas(df)
    for f in _SPEED_SECTOR_FEATURES:
        assert f in out.columns
        assert out[f].null_count() == out.height


def test_speed_sector_features_in_feature_columns():
    for f in _SPEED_SECTOR_FEATURES:
        assert f in FEATURE_COLUMNS
