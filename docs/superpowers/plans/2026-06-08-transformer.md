# Cross-Driver Transformer (Stage 5b) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a permutation-equivariant Transformer that attends across the 20 driver slots of a race snapshot and outputs a ranking score per driver, using a valid-pair-masked RankNet loss (then LambdaRank), and beat the LightGBM baseline's test Spearman.

**Architecture:** A data loader pads each `(race, snapshot_lap)` snapshot to 20 driver slots and produces feature/driver-index/mask/target tensors. The model projects features to `d_model`, adds a learned driver-identity embedding, runs a pre-LayerNorm `TransformerEncoder` with **no positional encoding** (drivers are an unordered set) and a padding mask, and scores each slot through a linear head (padded slots forced to `-1e4`). Loss is computed only over valid pairs (both drivers active at the snapshot lap). Evaluation reuses Stage 5a's metrics and adds a lap-by-lap prediction-evolution plot.

**Tech Stack:** Python 3.11+, `uv`, PyTorch, Polars, MLflow (local), matplotlib, Typer, pytest.

**Spec:** `docs/superpowers/specs/2026-06-03-f1-predictor-design.md` (Stage 5: Transformer + Loss + Evaluation).
**This is Plan 4 of 4.** Plans 1–3 are complete. This plan consumes `data/snapshots/{split}.parquet` + `metadata.json` from Plan 3 and the metrics in `src/f1_predictor/evaluate.py`.

---

## Inherited context (from Plans 1–3)

1. **Snapshots are long-format and scaled.** `data/snapshots/{train,val,test}.parquet` have one row per `(session_key, snapshot_lap, driver_number)` for drivers active at that lap, with the 30 `FEATURE_COLUMNS` already imputed + standardised, plus `final_position` and `relevance` (= `21 - final_position`). `metadata.json` holds `feature_columns`. This plan adds padding/masks at load time.
2. **Config:** `num_drivers = 20`, `driver_embedding_count = 30` (index 0 reserved for "unknown driver" — roster turnover between 2023 and 2024).
3. **Active-driver semantics:** a snapshot row exists only for drivers active at that lap, so "valid" slots after padding are exactly the rows present; padded slots are inactive. This is the valid-pair mask the loss needs.
4. **Baseline to beat:** Plan 3's `run_baseline` reports a test Spearman (spec target ~0.6–0.75 at lap 30). This plan must report the same metrics so the two are directly comparable.

---

## File Map

```
src/f1_predictor/data_loader.py            CREATE  snapshot long parquet -> padded tensors + masks + driver index
src/f1_predictor/losses.py                 CREATE  ranknet_loss + lambdarank_loss (valid-pair masked)
src/f1_predictor/models/transformer.py     CREATE  cross-driver Transformer ranker
src/f1_predictor/train.py                  CREATE  training loop (AdamW, warmup+cosine, MLflow, checkpoint)
src/f1_predictor/visualise.py              CREATE  lap-by-lap prediction-evolution plot
scripts/train_transformer.py              CREATE  Typer CLI: train + evaluate the Transformer
tests/test_data_loader_unit.py             CREATE  loader/padding/driver-index tests
tests/test_losses_unit.py                  CREATE  loss correctness tests
tests/test_transformer_unit.py             CREATE  model shape/equivariance/mask tests
tests/test_train_integration.py            CREATE  overfit-a-tiny-set + run-dir integration
```

`runs/` and `data/` are gitignored. `models/__init__.py` already exists (Plan 3).

---

## Tensor contract (one sample = one `(session_key, snapshot_lap)` group)

| Tensor | Shape | Meaning |
|---|---|---|
| `features` | `[20, F]` float | scaled features per slot; padded slots = 0 |
| `driver_idx` | `[20]` long | driver embedding index (0 = unknown/pad) |
| `valid` | `[20]` bool | True for active drivers, False for padding |
| `relevance` | `[20]` float | `21 - final_position`; padded = 0 |
| `final_position` | `[20]` long | for evaluation; padded = 99 sentinel |
| `session_key`, `snapshot_lap` | scalars | group identity (for grouped metrics) |

Batched by stacking: `[B, 20, F]`, `[B, 20]`, etc.

---

## Task 1: Driver index mapping

Map each `driver_number` to a small embedding index (1..29), reserving 0 for unknown/padding. Built from the **train** split so 2024-only drivers fall back to "unknown".

**Files:**
- Create: `src/f1_predictor/data_loader.py`
- Create: `tests/test_data_loader_unit.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_data_loader_unit.py
import polars as pl
import pytest
from f1_predictor.data_loader import build_driver_index, UNKNOWN_DRIVER_INDEX


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
    # Only 29 fit (1..29); the rest are absent (map to unknown at lookup time).
    assert max(idx.values()) == 29
    assert len(idx) == 29
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_data_loader_unit.py::test_build_driver_index_assigns_from_one -v`
Expected: `ModuleNotFoundError: No module named 'f1_predictor.data_loader'`.

- [ ] **Step 3: Create `src/f1_predictor/data_loader.py`**

```python
"""Load snapshot parquets into padded per-group tensors for the Transformer."""
from __future__ import annotations

import polars as pl

UNKNOWN_DRIVER_INDEX = 0  # reserved for padding and unseen drivers


def build_driver_index(train: pl.DataFrame, max_drivers: int) -> dict[int, int]:
    """Map driver_number -> embedding index 1..(max_drivers-1), 0 reserved.

    Built from the train split (sorted driver_number for determinism). Drivers
    beyond the capacity, and any driver unseen in train, map to UNKNOWN at
    lookup time (they are simply absent from this dict).
    """
    drivers = sorted(train["driver_number"].unique().to_list())
    capacity = max_drivers - 1  # index 0 reserved
    return {d: i + 1 for i, d in enumerate(drivers[:capacity])}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_data_loader_unit.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/f1_predictor/data_loader.py tests/test_data_loader_unit.py
git commit -m "feat: transformer — driver identity index mapping (0 = unknown)"
```

---

## Task 2: Snapshot → padded tensors

Convert one `(session_key, snapshot_lap)` group into the fixed `[20, …]` tensors, padding inactive slots.

**Files:**
- Modify: `src/f1_predictor/data_loader.py`
- Modify: `tests/test_data_loader_unit.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_data_loader_unit.py`:

```python
import torch
from f1_predictor.data_loader import snapshot_to_tensors, PAD_FINAL_POSITION


def _group(n_drivers):
    return pl.DataFrame({
        "session_key": [5] * n_drivers,
        "snapshot_lap": [30] * n_drivers,
        "driver_number": list(range(1, n_drivers + 1)),
        "final_position": list(range(1, n_drivers + 1)),
        "relevance": [21 - p for p in range(1, n_drivers + 1)],
        "f0": [float(p) for p in range(1, n_drivers + 1)],
        "f1": [0.0] * n_drivers,
    })


def test_snapshot_to_tensors_pads_to_num_slots():
    grp = _group(3)
    idx = {1: 1, 2: 2, 3: 3}
    t = snapshot_to_tensors(grp, feature_columns=["f0", "f1"], driver_index=idx, num_slots=20)
    assert t["features"].shape == (20, 2)
    assert t["driver_idx"].shape == (20,)
    assert t["valid"].sum().item() == 3                  # 3 active, 17 padded
    assert bool(t["valid"][:3].all()) and not bool(t["valid"][3:].any())
    assert t["relevance"][0].item() == pytest.approx(20.0)
    # padded slots: driver_idx 0, final_position sentinel, features 0
    assert t["driver_idx"][5].item() == 0
    assert t["final_position"][5].item() == PAD_FINAL_POSITION
    assert torch.all(t["features"][3:] == 0)


def test_snapshot_to_tensors_unknown_driver_maps_to_zero():
    grp = _group(2)
    idx = {1: 1}  # driver 2 unseen in train
    t = snapshot_to_tensors(grp, ["f0", "f1"], idx, num_slots=20)
    assert t["driver_idx"][0].item() == 1
    assert t["driver_idx"][1].item() == 0   # unknown -> 0
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_data_loader_unit.py::test_snapshot_to_tensors_pads_to_num_slots -v`
Expected: `ImportError: cannot import name 'snapshot_to_tensors'`.

- [ ] **Step 3: Implement `snapshot_to_tensors`**

Add to `src/f1_predictor/data_loader.py`:

```python
import torch

PAD_FINAL_POSITION = 99  # sentinel for padded slots (never used in metrics)


def snapshot_to_tensors(
    group: pl.DataFrame,
    feature_columns: list[str],
    driver_index: dict[int, int],
    num_slots: int,
) -> dict:
    """Pad one (session_key, snapshot_lap) group to num_slots fixed driver slots."""
    n = group.height
    if n > num_slots:
        group = group.sort("final_position").head(num_slots)
        n = num_slots

    feats = torch.zeros((num_slots, len(feature_columns)), dtype=torch.float32)
    feats[:n] = torch.tensor(group.select(feature_columns).to_numpy(), dtype=torch.float32)

    driver_idx = torch.zeros(num_slots, dtype=torch.long)
    drivers = group["driver_number"].to_list()
    for i, d in enumerate(drivers):
        driver_idx[i] = driver_index.get(d, UNKNOWN_DRIVER_INDEX)

    valid = torch.zeros(num_slots, dtype=torch.bool)
    valid[:n] = True

    relevance = torch.zeros(num_slots, dtype=torch.float32)
    relevance[:n] = torch.tensor(group["relevance"].to_numpy(), dtype=torch.float32)

    final_position = torch.full((num_slots,), PAD_FINAL_POSITION, dtype=torch.long)
    final_position[:n] = torch.tensor(group["final_position"].to_numpy(), dtype=torch.long)

    return {
        "features": feats,
        "driver_idx": driver_idx,
        "valid": valid,
        "relevance": relevance,
        "final_position": final_position,
        "session_key": int(group["session_key"][0]),
        "snapshot_lap": int(group["snapshot_lap"][0]),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_data_loader_unit.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/f1_predictor/data_loader.py tests/test_data_loader_unit.py
git commit -m "feat: transformer — pad snapshot group to fixed driver-slot tensors"
```

---

## Task 3: Dataset + DataLoader

A `torch.utils.data.Dataset` over all groups in a split, plus a helper to read a split parquet and build the driver index.

**Files:**
- Modify: `src/f1_predictor/data_loader.py`
- Modify: `tests/test_data_loader_unit.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_data_loader_unit.py`:

```python
from torch.utils.data import DataLoader
from f1_predictor.data_loader import SnapshotDataset


def test_snapshot_dataset_one_item_per_group():
    df = pl.concat([_group(3).with_columns(pl.lit(k).alias("session_key")) for k in (1, 2)])
    ds = SnapshotDataset(df, feature_columns=["f0", "f1"], driver_index={1: 1, 2: 2, 3: 3}, num_slots=20)
    assert len(ds) == 2  # two (session_key, snapshot_lap) groups


def test_dataloader_batches_stack_correctly():
    df = pl.concat([_group(3).with_columns(pl.lit(k).alias("session_key")) for k in (1, 2, 3)])
    ds = SnapshotDataset(df, ["f0", "f1"], {1: 1, 2: 2, 3: 3}, num_slots=20)
    loader = DataLoader(ds, batch_size=2)
    batch = next(iter(loader))
    assert batch["features"].shape == (2, 20, 2)
    assert batch["valid"].shape == (2, 20)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_data_loader_unit.py::test_snapshot_dataset_one_item_per_group -v`
Expected: `ImportError: cannot import name 'SnapshotDataset'`.

- [ ] **Step 3: Implement `SnapshotDataset` + `load_split`**

Add to `src/f1_predictor/data_loader.py`:

```python
import json
from pathlib import Path

from torch.utils.data import Dataset


class SnapshotDataset(Dataset):
    """One item per (session_key, snapshot_lap) group of a snapshot split."""

    def __init__(self, df: pl.DataFrame, feature_columns: list[str],
                 driver_index: dict[int, int], num_slots: int):
        self.feature_columns = feature_columns
        self.driver_index = driver_index
        self.num_slots = num_slots
        self.groups = [
            grp for _, grp in df.group_by(["session_key", "snapshot_lap"], maintain_order=True)
        ]

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, i: int) -> dict:
        return snapshot_to_tensors(
            self.groups[i], self.feature_columns, self.driver_index, self.num_slots
        )


def load_split(snapshots_dir: Path, split: str) -> pl.DataFrame:
    """Read one split parquet (train/val/test)."""
    return pl.read_parquet(snapshots_dir / f"{split}.parquet")


def load_metadata(snapshots_dir: Path) -> dict:
    return json.loads((snapshots_dir / "metadata.json").read_text())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_data_loader_unit.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/f1_predictor/data_loader.py tests/test_data_loader_unit.py
git commit -m "feat: transformer — SnapshotDataset + split loaders"
```

---

## Task 4: RankNet loss (valid-pair masked)

Pairwise loss over pairs where one driver should outrank another and both are active.

**Files:**
- Create: `src/f1_predictor/losses.py`
- Create: `tests/test_losses_unit.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_losses_unit.py
import torch
import pytest
from f1_predictor.losses import ranknet_loss


def test_ranknet_zero_when_perfectly_ordered():
    # scores already align with relevance with a large margin -> ~0 loss.
    scores = torch.tensor([[10.0, 5.0, 0.0]])
    relevance = torch.tensor([[20.0, 19.0, 18.0]])
    valid = torch.tensor([[True, True, True]])
    loss = ranknet_loss(scores, relevance, valid)
    assert loss.item() < 1e-3


def test_ranknet_higher_when_misordered():
    scores_good = torch.tensor([[5.0, 0.0]])
    scores_bad = torch.tensor([[0.0, 5.0]])
    relevance = torch.tensor([[20.0, 19.0]])
    valid = torch.tensor([[True, True]])
    assert ranknet_loss(scores_bad, relevance, valid) > ranknet_loss(scores_good, relevance, valid)


def test_ranknet_ignores_padded_slots():
    # The padded 3rd slot (valid=False) with a huge score must not affect the loss.
    relevance = torch.tensor([[20.0, 19.0, 0.0]])
    valid = torch.tensor([[True, True, False]])
    s_a = torch.tensor([[5.0, 0.0, 0.0]])
    s_b = torch.tensor([[5.0, 0.0, 999.0]])
    assert ranknet_loss(s_a, relevance, valid).item() == pytest.approx(
        ranknet_loss(s_b, relevance, valid).item()
    )


def test_ranknet_is_differentiable():
    scores = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    relevance = torch.tensor([[20.0, 19.0, 18.0]])
    valid = torch.tensor([[True, True, True]])
    ranknet_loss(scores, relevance, valid).backward()
    assert scores.grad is not None and torch.isfinite(scores.grad).all()
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_losses_unit.py::test_ranknet_zero_when_perfectly_ordered -v`
Expected: `ModuleNotFoundError: No module named 'f1_predictor.losses'`.

- [ ] **Step 3: Create `src/f1_predictor/losses.py`**

```python
"""Pairwise ranking losses with valid-pair masking.

Convention: a HIGHER score means a BETTER (higher-relevance) driver. Only pairs
where both drivers are active at the snapshot lap enter the loss.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _pair_masks(relevance: torch.Tensor, valid: torch.Tensor):
    """Return (score-diff broadcast helpers) — pair_mask[b,i,j] True when i should
    outrank j (r_i > r_j) and both are valid."""
    r_i = relevance.unsqueeze(2)        # [B,N,1]
    r_j = relevance.unsqueeze(1)        # [B,1,N]
    v = valid.unsqueeze(2) & valid.unsqueeze(1)
    pair_mask = v & (r_i > r_j)         # [B,N,N]
    return pair_mask


def ranknet_loss(scores: torch.Tensor, relevance: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Mean RankNet loss over valid (i outranks j) pairs.

    scores/relevance/valid are [B, N]. For each ordered pair where r_i > r_j and
    both valid, the loss is softplus(-(s_i - s_j)) = log(1 + exp(-(s_i - s_j))).
    """
    s_diff = scores.unsqueeze(2) - scores.unsqueeze(1)   # [B,N,N] = s_i - s_j
    pair_mask = _pair_masks(relevance, valid)
    losses = F.softplus(-s_diff) * pair_mask.float()
    denom = pair_mask.float().sum().clamp(min=1.0)
    return losses.sum() / denom
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_losses_unit.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/f1_predictor/losses.py tests/test_losses_unit.py
git commit -m "feat: transformer — valid-pair-masked RankNet loss"
```

---

## Task 5: Cross-driver Transformer model

Project features, add driver embedding, run a pre-LN `TransformerEncoder` with no positional encoding and a padding mask, score each slot.

**Files:**
- Create: `src/f1_predictor/models/transformer.py`
- Create: `tests/test_transformer_unit.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_transformer_unit.py
import torch
import pytest
from f1_predictor.models.transformer import DriverRanker


def _inputs(B=2, N=20, F=6):
    features = torch.randn(B, N, F)
    driver_idx = torch.randint(0, 30, (B, N))
    valid = torch.ones(B, N, dtype=torch.bool)
    valid[:, 15:] = False  # last 5 slots padded
    return features, driver_idx, valid


def test_forward_output_shape():
    model = DriverRanker(num_features=6, d_model=32, n_heads=4, n_layers=2, num_drivers=30)
    features, driver_idx, valid = _inputs()
    scores = model(features, driver_idx, valid)
    assert scores.shape == (2, 20)


def test_padded_slots_scored_very_low():
    model = DriverRanker(num_features=6, d_model=32, n_heads=4, n_layers=2, num_drivers=30)
    features, driver_idx, valid = _inputs()
    scores = model(features, driver_idx, valid)
    assert (scores[~valid] <= -1e3).all()   # padded slots forced very negative


def test_permutation_equivariance_on_valid_slots():
    # Permuting drivers (features+idx) permutes the scores correspondingly,
    # since there is no positional encoding.
    torch.manual_seed(0)
    model = DriverRanker(num_features=6, d_model=32, n_heads=4, n_layers=2, num_drivers=30).eval()
    F = 6
    features = torch.randn(1, 5, F)
    driver_idx = torch.tensor([[1, 2, 3, 4, 5]])
    valid = torch.ones(1, 5, dtype=torch.bool)
    perm = torch.tensor([2, 0, 4, 1, 3])
    with torch.no_grad():
        base = model(features, driver_idx, valid)[0]
        permd = model(features[:, perm], driver_idx[:, perm], valid[:, perm])[0]
    assert torch.allclose(base[perm], permd, atol=1e-5)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_transformer_unit.py::test_forward_output_shape -v`
Expected: `ModuleNotFoundError: No module named 'f1_predictor.models.transformer'`.

- [ ] **Step 3: Create `src/f1_predictor/models/transformer.py`**

```python
"""Permutation-equivariant cross-driver Transformer ranker.

Input [B, N, F] features + [B, N] driver indices + [B, N] validity mask.
Output [B, N] ranking scores (higher = better). No positional encoding (drivers
are an unordered set); a learned driver-identity embedding carries priors.
Padded slots are masked in attention and forced to -1e4 in the score head.
"""
from __future__ import annotations

import torch
import torch.nn as nn

_MASK_SCORE = -1e4


class DriverRanker(nn.Module):
    def __init__(self, num_features: int, d_model: int = 128, n_heads: int = 8,
                 n_layers: int = 4, dropout: float = 0.1, num_drivers: int = 30):
        super().__init__()
        self.input_proj = nn.Linear(num_features, d_model)
        self.driver_embed = nn.Embedding(num_drivers, d_model, padding_idx=0)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True, norm_first=True,  # pre-LayerNorm
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.score_head = nn.Linear(d_model, 1)

    def forward(self, features: torch.Tensor, driver_idx: torch.Tensor,
                valid: torch.Tensor) -> torch.Tensor:
        # [B,N,d] = projected features + driver-identity embedding.
        h = self.input_proj(features) + self.driver_embed(driver_idx)
        # TransformerEncoder: src_key_padding_mask True = ignore that slot.
        pad_mask = ~valid
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        scores = self.score_head(h).squeeze(-1)          # [B,N]
        return scores.masked_fill(pad_mask, _MASK_SCORE)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_transformer_unit.py -v`
Expected: all pass. (If `test_permutation_equivariance_on_valid_slots` fails by a hair, confirm the model is in `.eval()` so dropout is off — it is in the test.)

- [ ] **Step 5: Commit**

```bash
git add src/f1_predictor/models/transformer.py tests/test_transformer_unit.py
git commit -m "feat: transformer — cross-driver ranker (pre-LN, driver embed, padding mask)"
```

---

## Task 6: Training loop + run dir + MLflow

AdamW with a 500-step warmup + cosine decay, RankNet loss, MLflow logging, checkpoint + metrics under `runs/{run_id}/`. Evaluation reuses `evaluate.ranking_metrics`.

**Files:**
- Create: `src/f1_predictor/train.py`
- Create: `tests/test_train_integration.py`

- [ ] **Step 1: Write the failing test (overfit a tiny set)**

```python
# tests/test_train_integration.py
import json
from pathlib import Path
import polars as pl
import pytest
from f1_predictor.train import train_transformer


def _write_snapshots(snap_dir, splits):
    snap_dir.mkdir(parents=True, exist_ok=True)
    feature_columns = ["f0", "f1"]
    for split, (n_races, start) in splits.items():
        rows = []
        for r in range(start, start + n_races):
            for d in range(1, 6):
                rows.append({"session_key": r, "snapshot_lap": 30, "driver_number": d,
                             "final_position": d, "relevance": 21 - d,
                             "f0": float(d), "f1": 0.0})
        pl.DataFrame(rows).write_parquet(snap_dir / f"{split}.parquet")
    (snap_dir / "metadata.json").write_text(json.dumps({"feature_columns": feature_columns}))


def test_train_transformer_overfits_and_writes_run(tmp_path):
    snap_dir = tmp_path / "snapshots"
    _write_snapshots(snap_dir, {"train": (12, 0), "val": (3, 100), "test": (3, 200)})
    runs_dir = tmp_path / "runs"

    result = train_transformer(
        snap_dir, runs_dir,
        config={"d_model": 32, "n_heads": 4, "n_layers": 2, "epochs": 40,
                "lr": 1e-3, "warmup_steps": 5, "batch_size": 4, "num_drivers": 30,
                "num_slots": 20, "loss": "ranknet"},
        use_mlflow=False,
    )
    run_dir = Path(result["run_dir"])
    assert (run_dir / "model.pt").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "predictions_test.parquet").exists()
    # The signal is trivially learnable -> strong positive test Spearman.
    assert result["metrics"]["spearman"] > 0.9
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_train_integration.py -v`
Expected: `ModuleNotFoundError: No module named 'f1_predictor.train'`.

- [ ] **Step 3: Create `src/f1_predictor/train.py`**

```python
"""Train the cross-driver Transformer ranker and persist a run directory."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from f1_predictor.data_loader import (
    SnapshotDataset, build_driver_index, load_metadata, load_split,
)
from f1_predictor.evaluate import ranking_metrics
from f1_predictor.losses import ranknet_loss, lambdarank_loss
from f1_predictor.models.transformer import DriverRanker

_LOSSES = {"ranknet": ranknet_loss, "lambdarank": lambdarank_loss}


def _warmup_cosine(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def _predict_split(model, dataset, device) -> pl.DataFrame:
    model.eval()
    rows = []
    loader = DataLoader(dataset, batch_size=16)
    with torch.no_grad():
        for batch in loader:
            scores = model(batch["features"].to(device), batch["driver_idx"].to(device),
                           batch["valid"].to(device)).cpu()
            B, N = scores.shape
            for b in range(B):
                v = batch["valid"][b]
                for s in range(N):
                    if not bool(v[s]):
                        continue
                    rows.append({
                        "session_key": int(batch["session_key"][b]),
                        "snapshot_lap": int(batch["snapshot_lap"][b]),
                        "final_position": int(batch["final_position"][b, s]),
                        "score": float(scores[b, s]),
                    })
    return pl.DataFrame(rows)


def train_transformer(snapshots_dir: Path, runs_dir: Path, config: dict,
                      use_mlflow: bool = True) -> dict:
    """Train, evaluate on test, persist runs/{run_id}/. Returns run_dir + metrics."""
    meta = load_metadata(snapshots_dir)
    feature_columns = meta["feature_columns"]
    num_slots = config["num_slots"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df = load_split(snapshots_dir, "train")
    test_df = load_split(snapshots_dir, "test")
    driver_index = build_driver_index(train_df, config["num_drivers"])

    train_ds = SnapshotDataset(train_df, feature_columns, driver_index, num_slots)
    test_ds = SnapshotDataset(test_df, feature_columns, driver_index, num_slots)
    loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)

    model = DriverRanker(
        num_features=len(feature_columns), d_model=config["d_model"],
        n_heads=config["n_heads"], n_layers=config["n_layers"],
        num_drivers=config["num_drivers"],
    ).to(device)

    loss_fn = _LOSSES[config["loss"]]
    opt = AdamW(model.parameters(), lr=config["lr"], weight_decay=config.get("weight_decay", 0.01))
    total_steps = max(config["epochs"] * len(loader), 1)
    sched = LambdaLR(opt, lambda s: _warmup_cosine(s, config["warmup_steps"], total_steps))

    model.train()
    for _ in range(config["epochs"]):
        for batch in loader:
            opt.zero_grad()
            scores = model(batch["features"].to(device), batch["driver_idx"].to(device),
                           batch["valid"].to(device))
            loss = loss_fn(scores, batch["relevance"].to(device), batch["valid"].to(device))
            loss.backward()
            opt.step()
            sched.step()

    preds = _predict_split(model, test_ds, device)
    metrics = ranking_metrics(preds)

    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), run_dir / "model.pt")
    preds.write_parquet(run_dir / "predictions_test.parquet")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (run_dir / "config.yaml").write_text(json.dumps(
        {**config, "feature_columns": feature_columns, "data_version": meta.get("data_version")},
        indent=2))

    if use_mlflow:
        import mlflow
        mlflow.set_experiment("f1-transformer")
        with mlflow.start_run(run_name=run_id):
            mlflow.log_params({k: v for k, v in config.items() if not isinstance(v, (list, dict))})
            mlflow.log_metrics(metrics)
            mlflow.log_artifacts(str(run_dir))

    return {"run_dir": str(run_dir), "metrics": metrics}
```

- [ ] **Step 4: Add a `lambdarank_loss` stub so the import resolves (full impl in Task 7)**

To keep Task 6 runnable before Task 7, add to `src/f1_predictor/losses.py`:

```python
def lambdarank_loss(scores: torch.Tensor, relevance: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Placeholder delegating to RankNet until Task 7 implements NDCG weighting."""
    return ranknet_loss(scores, relevance, valid)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_train_integration.py tests/test_losses_unit.py -v`
Expected: all pass (the overfit test reaches Spearman > 0.9).

- [ ] **Step 6: Commit**

```bash
git add src/f1_predictor/train.py src/f1_predictor/losses.py tests/test_train_integration.py
git commit -m "feat: transformer — training loop (AdamW, warmup+cosine, run dir, MLflow)"
```

---

## Task 7: LambdaRank loss (NDCG-weighted)

Replace the placeholder with a true LambdaRank: weight each RankNet pair by the |ΔNDCG| of swapping the two drivers under the current scores. Podium swaps get large gradients.

**Files:**
- Modify: `src/f1_predictor/losses.py`
- Modify: `tests/test_losses_unit.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_losses_unit.py`:

```python
from f1_predictor.losses import lambdarank_loss


def test_lambdarank_weights_top_swaps_more_than_tail_swaps():
    # Same misordering magnitude, but a swap near the top should cost more than
    # one in the tail because NDCG discounts later ranks.
    relevance = torch.tensor([[20.0, 19.0, 2.0, 1.0]])
    valid = torch.tensor([[True, True, True, True]])
    # Case A: top pair misordered (s0 < s1 though r0 > r1).
    top_bad = torch.tensor([[0.0, 1.0, -5.0, -6.0]])
    # Case B: tail pair misordered (s2 < s3 though r2 > r3).
    tail_bad = torch.tensor([[5.0, 4.0, -1.0, 0.0]])
    assert lambdarank_loss(top_bad, relevance, valid) > lambdarank_loss(tail_bad, relevance, valid)


def test_lambdarank_is_differentiable_and_finite():
    scores = torch.tensor([[1.0, 2.0, 0.5, -1.0]], requires_grad=True)
    relevance = torch.tensor([[20.0, 19.0, 18.0, 17.0]])
    valid = torch.tensor([[True, True, True, True]])
    lambdarank_loss(scores, relevance, valid).backward()
    assert torch.isfinite(scores.grad).all()


def test_lambdarank_ignores_padded_slots():
    relevance = torch.tensor([[20.0, 19.0, 0.0]])
    valid = torch.tensor([[True, True, False]])
    s_a = torch.tensor([[3.0, 0.0, 0.0]])
    s_b = torch.tensor([[3.0, 0.0, 999.0]])
    assert torch.allclose(
        lambdarank_loss(s_a, relevance, valid), lambdarank_loss(s_b, relevance, valid)
    )
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_losses_unit.py::test_lambdarank_weights_top_swaps_more_than_tail_swaps -v`
Expected: FAIL (placeholder weights all pairs equally, so the two losses are equal).

- [ ] **Step 3: Replace the `lambdarank_loss` placeholder in `losses.py`**

```python
def _dcg_discounts(scores: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """1/log2(rank+1) per slot, where rank is by descending score among valid."""
    neg = scores.masked_fill(~valid, float("-inf"))
    order = torch.argsort(neg, dim=1, descending=True)         # [B,N] slot indices by rank
    ranks = torch.empty_like(order)
    ar = torch.arange(scores.size(1), device=scores.device).expand_as(order)
    ranks.scatter_(1, order, ar)                                # ranks[b, slot] = 0-based rank
    return 1.0 / torch.log2(ranks.float() + 2.0)               # +2 => log2(rank+1) with rank 1-based


def lambdarank_loss(scores: torch.Tensor, relevance: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """RankNet loss with each pair weighted by |ΔNDCG| of swapping i and j.

    gains = 2^relevance - 1; discounts from current predicted ranking; ideal DCG
    normalises per group. Higher score = better. Only valid (r_i > r_j) pairs count.
    """
    s_diff = scores.unsqueeze(2) - scores.unsqueeze(1)         # [B,N,N]
    pair_mask = _pair_masks(relevance, valid)                  # [B,N,N]

    gains = (torch.pow(2.0, relevance) - 1.0) * valid.float()  # [B,N]
    discounts = _dcg_discounts(scores, valid) * valid.float()  # [B,N]

    # Ideal DCG: gains sorted by true relevance, paired with sorted discounts.
    ideal_gains, _ = torch.sort(gains, dim=1, descending=True)
    ideal_disc, _ = torch.sort(_dcg_discounts(relevance, valid) * valid.float(), dim=1, descending=True)
    idcg = (ideal_gains * ideal_disc).sum(dim=1).clamp(min=1e-6)  # [B]

    g_i, g_j = gains.unsqueeze(2), gains.unsqueeze(1)
    d_i, d_j = discounts.unsqueeze(2), discounts.unsqueeze(1)
    delta_ndcg = torch.abs((g_i - g_j) * (d_i - d_j)) / idcg.view(-1, 1, 1)  # [B,N,N]

    losses = F.softplus(-s_diff) * delta_ndcg * pair_mask.float()
    denom = (delta_ndcg * pair_mask.float()).sum().clamp(min=1e-6)
    return losses.sum() / denom
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_losses_unit.py -v`
Expected: all pass (top swaps now cost more; gradients finite; padding ignored).

- [ ] **Step 5: Commit**

```bash
git add src/f1_predictor/losses.py tests/test_losses_unit.py
git commit -m "feat: transformer — LambdaRank loss (NDCG-weighted pairs)"
```

---

## Task 8: CLI, lap-by-lap evolution plot, head-to-head run

Train CLI, a prediction-evolution visualiser, and a real run comparing the Transformer to the baseline.

**Files:**
- Create: `scripts/train_transformer.py`
- Create: `src/f1_predictor/visualise.py`
- Create: `tests/test_visualise_unit.py`

- [ ] **Step 1: Write the failing test for the visualiser**

```python
# tests/test_visualise_unit.py
from pathlib import Path
import polars as pl
from f1_predictor.visualise import plot_prediction_evolution


def test_plot_prediction_evolution_writes_png(tmp_path):
    # predictions across snapshot laps for one race; plot must produce a file.
    rows = []
    for lap in (10, 20, 30, 40):
        for d, fp in [(1, 1), (2, 2), (3, 3)]:
            rows.append({"session_key": 5, "snapshot_lap": lap, "driver_number": d,
                         "final_position": fp, "score": fp * -1.0 + lap * 0.0})
    preds = pl.DataFrame(rows)
    out = tmp_path / "evo.png"
    plot_prediction_evolution(preds, session_key=5, out_path=out)
    assert out.exists() and out.stat().st_size > 0
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_visualise_unit.py -v`
Expected: `ModuleNotFoundError: No module named 'f1_predictor.visualise'`.

- [ ] **Step 3: Create `src/f1_predictor/visualise.py`**

```python
"""Lap-by-lap prediction-evolution plot: predicted rank per driver vs snapshot lap."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import polars as pl


def plot_prediction_evolution(predictions: pl.DataFrame, session_key: int, out_path: Path) -> None:
    """Plot each driver's predicted rank across snapshot laps for one race.

    predictions: session_key, snapshot_lap, driver_number, final_position, score.
    Predicted rank within a (race, lap) = descending score (1 = predicted winner).
    """
    race = predictions.filter(pl.col("session_key") == session_key)
    race = race.with_columns(
        pl.col("score").rank("ordinal", descending=True).over("snapshot_lap").alias("pred_rank")
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    for d, grp in race.group_by("driver_number", maintain_order=True):
        grp = grp.sort("snapshot_lap")
        ax.plot(grp["snapshot_lap"].to_list(), grp["pred_rank"].to_list(),
                marker="o", label=f"#{d[0] if isinstance(d, tuple) else d}")
    ax.invert_yaxis()  # rank 1 at top
    ax.set_xlabel("snapshot lap")
    ax.set_ylabel("predicted rank")
    ax.set_title(f"Prediction evolution — session {session_key}")
    ax.legend(loc="best", fontsize="x-small", ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
```

- [ ] **Step 4: Create `scripts/train_transformer.py`**

```python
"""CLI: train + evaluate the cross-driver Transformer on the snapshot splits."""
from pathlib import Path

import typer

from f1_predictor.train import train_transformer

app = typer.Typer(add_completion=False)


@app.command()
def main(
    snapshots_dir: Path = typer.Option(Path("data/snapshots"), "--snapshots-dir"),
    runs_dir: Path = typer.Option(Path("runs"), "--runs-dir"),
    loss: str = typer.Option("lambdarank", "--loss", help="ranknet | lambdarank"),
    epochs: int = typer.Option(40, "--epochs"),
    no_mlflow: bool = typer.Option(False, "--no-mlflow"),
) -> None:
    config = {
        "d_model": 128, "n_heads": 8, "n_layers": 4, "dropout": 0.1,
        "lr": 1e-4, "weight_decay": 0.01, "warmup_steps": 500,
        "batch_size": 32, "epochs": epochs, "num_drivers": 30,
        "num_slots": 20, "loss": loss,
    }
    result = train_transformer(snapshots_dir, runs_dir, config, use_mlflow=not no_mlflow)
    typer.echo(f"Run: {result['run_dir']}")
    for k, v in result["metrics"].items():
        typer.echo(f"  {k}: {v}")


if __name__ == "__main__":
    app()
```

- [ ] **Step 5: Run unit tests, then the real Transformer run and compare to baseline**

```bash
uv run pytest tests/test_visualise_unit.py tests/test_transformer_unit.py tests/test_losses_unit.py tests/test_data_loader_unit.py -v
uv run python scripts/build_snapshots.py                       # ensure snapshots exist
uv run python scripts/train_transformer.py --loss ranknet --no-mlflow      # warm-up loss
uv run python scripts/train_transformer.py --loss lambdarank --no-mlflow   # primary
```

Compare the Transformer's test Spearman to the LightGBM baseline (Plan 3). The Transformer should **meet or beat** the baseline; if it is much worse, that is a real finding — inspect `predictions_test.parquet` and the loss curve before accepting it (start with fewer layers / higher lr / more epochs, and confirm the overfit test still passes).

- [ ] **Step 6: Run the full suite**

Run: `uv run pytest tests/ -q`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add scripts/train_transformer.py src/f1_predictor/visualise.py tests/test_visualise_unit.py
git commit -m "feat: transformer — CLI, lap-by-lap evolution plot, head-to-head run"
```

---

## Self-Review

### 1. Spec coverage

| Spec requirement (Stage 5b) | Task |
|---|---|
| Input [batch,20,F] → output [batch,20] | Task 5 |
| No positional encoding (permutation-equivariant) | Task 5 (+ equivariance test) |
| Driver identity embedding, up to 30, index 0 = unknown | Task 1 + 5 |
| Pre-LayerNorm (norm_first=True) | Task 5 |
| Padding mask, retired/absent slots → -1e4 in score head | Task 2 + 5 |
| Hyperparams d_model=128/heads=8/layers=4/dropout 0.1/AdamW lr 1e-4/wd 0.01/warmup 500 + cosine | Task 6 + 8 CLI |
| LambdaRank loss; podium swaps large gradients | Task 7 |
| RankNet fallback with valid-pair masking | Task 4 |
| Only valid pairs (both active at lap N) in loss | Task 4 + 7 |
| Spearman / top-3 / top-1 / mean position error | reuses Plan 3 `evaluate.py` (Task 6/8) |
| Lap-by-lap prediction evolution plot | Task 8 |
| runs/{run_id}/ (config, model.pt, metrics, predictions_test, MLflow) | Task 6 |

The optional "temporal encoding (LSTM wrapper) if plateau" from the spec's build sequence is explicitly out of scope for v1 (it is a contingency, not a requirement) and would be a follow-up plan.

### 2. Placeholder scan

The only deliberate placeholder is `lambdarank_loss` in Task 6 Step 4, which exists solely so the training loop imports cleanly; Task 7 replaces it with the real implementation in the same plan. No "TBD"/"add error handling"/"similar to Task N" placeholders; all code is complete and runnable.

### 3. Type consistency

- `build_driver_index(train, max_drivers) -> dict[int,int]` — Task 1, used in Task 3/6.
- `snapshot_to_tensors(group, feature_columns, driver_index, num_slots) -> dict` — Task 2; the dict keys (`features`, `driver_idx`, `valid`, `relevance`, `final_position`, `session_key`, `snapshot_lap`) are consumed unchanged by `SnapshotDataset` (Task 3), `DriverRanker.forward(features, driver_idx, valid)` (Task 5), the losses (`scores, relevance, valid`, Tasks 4/7), and `_predict_split` (Task 6).
- `ranknet_loss` / `lambdarank_loss(scores, relevance, valid)` — identical signatures (Tasks 4/6/7), selected via `_LOSSES` in Task 6.
- `DriverRanker(num_features, d_model, n_heads, n_layers, dropout, num_drivers)` — Task 5, instantiated in Task 6 with matching kwargs.
- `ranking_metrics` (Plan 3) consumes `session_key, snapshot_lap, final_position, score` — exactly the columns `_predict_split` and the visualiser produce.

### Open items for the executor to confirm

- **CPU training time:** the real run is CPU-bound unless CUDA is present (`train.py` auto-selects device). With ~22 train races × 4 snapshot laps the dataset is small (~hundreds of groups); 40 epochs should be minutes on CPU. Reduce `epochs`/`d_model` if needed for iteration speed.
- **Driver index persistence:** the index is rebuilt deterministically from `train.parquet` each run and saved inside `config.yaml`. If you later serve the model, persist `driver_index` explicitly alongside `model.pt`.
- **Beating the baseline:** if LambdaRank underperforms RankNet on this small dataset, that is a known small-data behaviour — report both and prefer the better test Spearman; the spec treats RankNet as an acceptable fallback.
