"""Unit tests for the Transformer snapshot data loader."""
import polars as pl
import torch

from f1_predictor.data_loader import (
    PAD_FINAL_POSITION,
    UNKNOWN_DRIVER_INDEX,
    build_driver_index,
    prepare_split,
    snapshot_to_tensors,
)


def _group(n):
    df = pl.DataFrame({
        "session_key": [5] * n, "snapshot_lap": [30] * n,
        "driver_number": list(range(1, n + 1)),
        "final_position": list(range(1, n + 1)),
        "position": [float(p) for p in range(1, n + 1)],
        "f0": [float(p) for p in range(1, n + 1)], "f1": [0.0] * n,
    })
    return prepare_split(df)


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


# --- snapshot_to_tensors: padded delta-regression tensors ---------------------

def test_tensors_pad_and_carry_delta_currentrank():
    t = snapshot_to_tensors(_group(3), ["f0", "f1"], {1: 1, 2: 2, 3: 3}, num_slots=20)
    assert t["features"].shape == (20, 2)
    assert t["valid"].sum().item() == 3
    assert t["current_rank"][0].item() == 1.0
    assert t["delta"][0].item() == 0.0            # P1 stays P1
    assert t["final_position"][5].item() == PAD_FINAL_POSITION
    assert torch.all(t["features"][3:] == 0)
    # padded slots: driver_idx + driver_number are 0
    assert t["driver_idx"][5].item() == 0
    assert t["driver_number"][5].item() == 0


def test_unknown_driver_maps_to_zero():
    t = snapshot_to_tensors(_group(2), ["f0", "f1"], {1: 1}, num_slots=20)  # driver 2 unseen
    assert t["driver_idx"][0].item() == 1 and t["driver_idx"][1].item() == 0
    # raw driver_number is preserved regardless of embedding-index fallback
    assert t["driver_number"][1].item() == 2
