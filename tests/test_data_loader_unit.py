"""Unit tests for the Transformer snapshot data loader."""
import polars as pl
import pytest

from f1_predictor.data_loader import (
    UNKNOWN_DRIVER_INDEX,
    build_driver_index,
    prepare_split,
)


# --- build_driver_index (carried over from the 2026-06-08 plan) ---------------

def test_build_driver_index_assigns_from_one():
    train = pl.DataFrame({"driver_number": [44, 1, 44, 11]})
    idx = build_driver_index(train, max_drivers=30)
    assert UNKNOWN_DRIVER_INDEX == 0
    assert set(idx.values()) == {1, 2, 3}      # 3 distinct drivers -> indices 1..3
    assert 0 not in idx.values()               # 0 reserved for unknown
    assert idx[1] != idx[44]


def test_build_driver_index_is_deterministic():
    train = pl.DataFrame({"driver_number": [11, 1, 44]})
    a = build_driver_index(train, 30)
    b = build_driver_index(train, 30)
    assert a == b                              # sorted driver_number -> stable


def test_build_driver_index_caps_at_max():
    train = pl.DataFrame({"driver_number": list(range(1, 40))})  # 39 drivers
    idx = build_driver_index(train, max_drivers=30)
    assert max(idx.values()) == 29
    assert len(idx) == 29


# --- prepare_split: current_rank + delta target -------------------------------

def test_prepare_split_adds_current_rank_and_delta():
    # one race/lap, 3 drivers; position standardised but monotonic.
    df = pl.DataFrame({
        "session_key": [5, 5, 5], "snapshot_lap": [30, 30, 30],
        "driver_number": [44, 1, 11], "final_position": [1, 3, 2],
        "position": [-1.0, 0.0, 1.0],  # ranks -> 1,2,3
    })
    out = prepare_split(df)
    # by driver 1, 11, 44 (sorted): ranks are 2, 3, 1
    assert out.sort("driver_number")["current_rank"].to_list() == [2, 3, 1]
    # delta = current_rank - final_position
    by_drv = {r["driver_number"]: r["delta"] for r in out.iter_rows(named=True)}
    assert by_drv[44] == 0.0 and by_drv[1] == -1.0 and by_drv[11] == 1.0
