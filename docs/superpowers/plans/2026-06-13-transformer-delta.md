# Cross-Driver Transformer (delta-regression) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.
>
> On execution, copy this plan to `docs/superpowers/plans/2026-06-13-transformer-delta.md` and commit it first.

## Context

We want to find out whether **cross-driver attention adds predictive value over the LightGBM baseline** — i.e. does modelling drivers jointly (who's ahead/behind, traffic, relative pace) beat scoring each driver-row independently? A per-row GBM cannot see the field; a permutation-equivariant Transformer over the 20 driver slots can.

Two facts about the current project shape this plan and make the pre-existing [2026-06-08-transformer.md](docs/superpowers/plans/2026-06-08-transformer.md) plan stale:

1. **The baseline is now L1 delta-regression, not LambdaRank ranking.** Each driver predicts *places gained* `delta = current_rank − final_position`; the ranking score is reconstructed as `score = predicted_delta − current_rank` (== `−predicted_final_position`). A predicted delta of 0 reproduces naive persistence, so the model only learns residual movement. This is "the first formulation that beats naive persistence at every snapshot lap" ([baseline_gbm.py](src/f1_predictor/models/baseline_gbm.py)). **The Transformer must use the same target and scoring** so the comparison is apples-to-apples — this is the single biggest change from the old plan, which used pairwise RankNet/LambdaRank losses (those are dropped entirely).

2. **2026 is a new-regulation regime where every model loses to naive** (drift = non-transferable deltas; see [2026-06-11-2026-adaptation.md](docs/superpowers/plans/2026-06-11-2026-adaptation.md) and the `2026-drift-diagnostic` memory). Evaluating an architecture on the 2026 `test` split would conflate "is the architecture good" with "did it survive a distribution shift." So this plan **evaluates on the pre-2026 `val` split (late-2025 races), and never touches `test` (2026).** Within 2023–2025 there is no regulation change, so a win/loss on `val` is a clean read on the architecture.

**Intended outcome:** a trained cross-driver Transformer, evaluated on the 2025 `val` split, reported head-to-head against (a) the GBM baseline trained on the same `train` split and (b) naive persistence — all three via the same `ranking_metrics`. The deliverable is the comparison and the reusable model/loaders, not a guarantee the Transformer wins.

## What carries over verbatim from the 2026-06-08 plan

These pieces are unchanged and should be lifted directly (same code, same tests) — do **not** re-derive them:

- **Task 1 — `build_driver_index`** (driver_number → embedding index 1..29, 0 reserved for unknown/pad), built from the `train` split. See old plan Task 1.
- **The `SnapshotDataset` shell + `load_split` / `load_metadata`** (old plan Task 3) — modified only to carry the extra tensor fields below.
- **The model architecture** (old plan Task 5): `input_proj` (Linear F→d_model) + `driver_embed` (Embedding(30, d_model, padding_idx=0)) + pre-LN `TransformerEncoder` (no positional encoding) + linear head, with `src_key_padding_mask = ~valid`. Renamed/repurposed: the head output is now a **per-driver delta** (regression), not a ranking score. **Drop** the `masked_fill(-1e4)` (that was for softmax-ranking losses; here we mask in the loss and at scoring time).
- **The lap-by-lap evolution plot** `visualise.plot_prediction_evolution` (old plan Task 8) — verbatim.
- **Warmup+cosine LR schedule helper** (old plan Task 6 `_warmup_cosine`) — verbatim.

## What changes (the substance of this plan)

| Concern | Old plan | This plan |
|---|---|---|
| Head output | ranking score | **predicted delta (places gained)** |
| Loss | pairwise RankNet → LambdaRank | **masked L1** over valid slots (matches GBM's `regression_l1`) |
| Per-slot target | `relevance = 21 − final_position` | **`delta = current_rank − final_position`** |
| Scoring | raw head, padded → −1e4 | **`score = delta_hat − current_rank`**, invalid slots dropped |
| Feature count | "30 FEATURE_COLUMNS" | **read `len(metadata.feature_columns)`** (currently ~40; never hard-code) |
| Eval split | `test` (2026) | **`val` (2025)**; `test` untouched |
| Comparison | "beat the LightGBM baseline" (implicit) | **explicit 3-way on `val`: Transformer vs GBM vs naive** |

## Reused existing functions (import, don't reimplement)

- `f1_predictor.models.baseline_gbm.add_current_rank(df)` → adds `current_rank` (ordinal rank of standardised `position` within `(session_key, snapshot_lap)`). Use this so `current_rank` is **identical** to what the GBM uses.
- `f1_predictor.models.baseline_gbm.naive_predict(df)` / `train_baseline` / `predict` → for the head-to-head on `val`.
- `f1_predictor.evaluate.ranking_metrics(predictions)` → needs columns `session_key, snapshot_lap, driver_number, final_position, score`. Reused for all three models.

## File Map

```
src/f1_predictor/data_loader.py            CREATE  driver index, prepare_split (adds current_rank+delta), padded tensors, Dataset
src/f1_predictor/losses.py                 CREATE  masked_l1_loss (the only loss; no pairwise machinery)
src/f1_predictor/models/transformer.py     CREATE  DriverDeltaNet — cross-driver Transformer, per-driver delta head
src/f1_predictor/train.py                  CREATE  training loop (AdamW, warmup+cosine, early-stop on val Spearman, run dir, MLflow)
src/f1_predictor/visualise.py              CREATE  lap-by-lap prediction-evolution plot (verbatim from old plan)
scripts/train_transformer.py               CREATE  Typer CLI: train + eval on val
scripts/compare_val.py                     CREATE  3-way head-to-head on val: Transformer vs GBM vs naive
tests/test_data_loader_unit.py             CREATE  driver index, current_rank/delta carry, padding, unknown-driver
tests/test_losses_unit.py                  CREATE  masked L1 correctness + padding-invariance + differentiable
tests/test_transformer_unit.py             CREATE  shape, permutation-equivariance, padding handled
tests/test_train_integration.py            CREATE  overfit-a-tiny-set + run-dir artifacts + score reconstruction
```

`runs/` and `data/` are gitignored. `models/__init__.py` already exists.

## Tensor contract (one sample = one `(session_key, snapshot_lap)` group, padded to 20 slots)

| Tensor | Shape | Meaning |
|---|---|---|
| `features` | `[20, F]` float | scaled features per slot; padded = 0 |
| `driver_idx` | `[20]` long | embedding index (0 = unknown/pad) |
| `driver_number` | `[20]` long | raw OpenF1 number; for prediction rows; padded = 0 |
| `valid` | `[20]` bool | True for active drivers |
| `current_rank` | `[20]` float | 1..N race order at the snapshot lap; for score reconstruction; padded = 0 |
| `delta` | `[20]` float | regression target `current_rank − final_position`; padded = 0 |
| `final_position` | `[20]` long | for evaluation; padded = 99 sentinel |
| `session_key`, `snapshot_lap` | scalars | group identity |

Batched by default stacking: `[B, 20, F]`, `[B, 20]`, etc.

---

## Task 1: Driver index + `prepare_split` (current_rank + delta)

Lift `build_driver_index` from the old plan Task 1 verbatim. Add a `prepare_split` that augments a loaded split DataFrame with `current_rank` (via the reused `add_current_rank`) and `delta`.

**Files:** Create `src/f1_predictor/data_loader.py`, `tests/test_data_loader_unit.py`.

- [ ] **Step 1 — failing tests.** Reuse the old plan's three `build_driver_index` tests verbatim. Add:

```python
import polars as pl
from f1_predictor.data_loader import prepare_split

def test_prepare_split_adds_current_rank_and_delta():
    # one race/lap, 3 drivers; position standardised but monotonic.
    df = pl.DataFrame({
        "session_key": [5, 5, 5], "snapshot_lap": [30, 30, 30],
        "driver_number": [44, 1, 11], "final_position": [1, 3, 2],
        "position": [-1.0, 0.0, 1.0],  # ranks -> 1,2,3
    })
    out = prepare_split(df)
    assert out.sort("driver_number")["current_rank"].to_list() == [2, 3, 1]  # by driver 1,11,44
    # delta = current_rank - final_position; driver 44: rank1 - final1 = 0; driver 1: 2-3=-1; driver 11: 3-2=1
    by_drv = {r["driver_number"]: r["delta"] for r in out.iter_rows(named=True)}
    assert by_drv[44] == 0.0 and by_drv[1] == -1.0 and by_drv[11] == 1.0
```

- [ ] **Step 2 — implement.** `build_driver_index` exactly as old plan. Then:

```python
from f1_predictor.models.baseline_gbm import add_current_rank

def prepare_split(df: pl.DataFrame) -> pl.DataFrame:
    """Add current_rank (reused from the GBM) and the delta regression target.

    delta = current_rank - final_position (positive = places gained), identical
    to the LightGBM baseline so the two models are directly comparable.
    """
    df = add_current_rank(df)
    return df.with_columns(
        (pl.col("current_rank") - pl.col("final_position")).cast(pl.Float64).alias("delta")
    )
```

- [ ] **Step 3 — verify pass; commit** `feat: transformer — driver index + delta-target split prep`.

---

## Task 2: Snapshot → padded delta-regression tensors

Convert one `(session_key, snapshot_lap)` group to the fixed-20 tensor contract above. Mirrors old plan Task 2 but carries `current_rank`, `delta`, `driver_number` and **drops** `relevance`.

**Files:** Modify `data_loader.py`, `tests/test_data_loader_unit.py`.

- [ ] **Step 1 — failing tests** (adapt old plan Task 2):

```python
import torch
from f1_predictor.data_loader import snapshot_to_tensors, PAD_FINAL_POSITION

def _group(n):
    df = pl.DataFrame({
        "session_key": [5]*n, "snapshot_lap": [30]*n,
        "driver_number": list(range(1, n+1)),
        "final_position": list(range(1, n+1)),
        "position": [float(p) for p in range(1, n+1)],
        "f0": [float(p) for p in range(1, n+1)], "f1": [0.0]*n,
    })
    return prepare_split(df)

def test_tensors_pad_and_carry_delta_currentrank():
    t = snapshot_to_tensors(_group(3), ["f0","f1"], {1:1,2:2,3:3}, num_slots=20)
    assert t["features"].shape == (20, 2)
    assert t["valid"].sum().item() == 3
    assert t["current_rank"][0].item() == 1.0
    assert t["delta"][0].item() == 0.0            # P1 stays P1
    assert t["final_position"][5].item() == PAD_FINAL_POSITION
    assert torch.all(t["features"][3:] == 0)

def test_unknown_driver_maps_to_zero():
    t = snapshot_to_tensors(_group(2), ["f0","f1"], {1:1}, num_slots=20)  # driver 2 unseen
    assert t["driver_idx"][0].item() == 1 and t["driver_idx"][1].item() == 0
```

- [ ] **Step 2 — implement** `snapshot_to_tensors` as old plan Task 2, but the returned dict carries `current_rank`, `delta`, `driver_number` (long, pad 0) and **no** `relevance`. If `n > num_slots`, sort by `final_position` and keep the head (same overflow guard as old plan). `PAD_FINAL_POSITION = 99`.

- [ ] **Step 3 — verify pass; commit** `feat: transformer — padded delta-regression tensors`.

---

## Task 3: `SnapshotDataset` + split loaders

Lift the old plan Task 3 `SnapshotDataset`, `load_split`, `load_metadata` verbatim, with one change: the dataset's `__init__` calls `prepare_split(df)` before grouping (so `current_rank`/`delta` exist), and groups by `(session_key, snapshot_lap)`.

**Files:** Modify `data_loader.py`, `tests/test_data_loader_unit.py`.

- [ ] **Step 1 — failing tests** (old plan Task 3's two tests; assert `batch["delta"].shape == (B, 20)` and `batch["current_rank"].shape == (B, 20)` in addition to `features`/`valid`).
- [ ] **Step 2 — implement.** Same as old plan; add `self.df = prepare_split(df)` and group on that.
- [ ] **Step 3 — verify pass; commit** `feat: transformer — SnapshotDataset + split loaders`.

---

## Task 4: Masked L1 loss

The only loss. Matches the GBM's `regression_l1`: mean absolute error between predicted and target delta over **valid** slots only.

**Files:** Create `src/f1_predictor/losses.py`, `tests/test_losses_unit.py`.

- [ ] **Step 1 — failing tests:**

```python
import torch, pytest
from f1_predictor.losses import masked_l1_loss

def test_zero_when_exact():
    pred = torch.tensor([[1.0, -2.0, 0.5]]); tgt = pred.clone()
    valid = torch.tensor([[True, True, True]])
    assert masked_l1_loss(pred, tgt, valid).item() == pytest.approx(0.0)

def test_matches_manual_mean_abs_error():
    pred = torch.tensor([[1.0, 0.0]]); tgt = torch.tensor([[0.0, 2.0]])
    valid = torch.tensor([[True, True]])
    assert masked_l1_loss(pred, tgt, valid).item() == pytest.approx(1.5)  # (|1|+|2|)/2

def test_ignores_padded_slots():
    valid = torch.tensor([[True, True, False]])
    a = torch.tensor([[1.0, 0.0, 0.0]]); b = torch.tensor([[1.0, 0.0, 999.0]])
    tgt = torch.tensor([[0.0, 0.0, 0.0]])
    assert masked_l1_loss(a, tgt, valid).item() == pytest.approx(masked_l1_loss(b, tgt, valid).item())

def test_differentiable():
    pred = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    tgt = torch.tensor([[0.0, 0.0, 0.0]]); valid = torch.tensor([[True, True, True]])
    masked_l1_loss(pred, tgt, valid).backward()
    assert torch.isfinite(pred.grad).all()
```

- [ ] **Step 2 — implement:**

```python
"""Masked L1 loss for delta-regression over valid driver slots.

Mirrors the LightGBM baseline's robust `regression_l1` objective: the delta
distribution (places gained) is heavy-tailed, so absolute error avoids the
front-of-grid over-correction that squared loss causes. Only slots active at the
snapshot lap contribute.
"""
from __future__ import annotations
import torch

def masked_l1_loss(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    m = valid.float()
    abs_err = (pred - target).abs() * m
    return abs_err.sum() / m.sum().clamp(min=1.0)
```

- [ ] **Step 3 — verify pass; commit** `feat: transformer — masked L1 delta loss`.

---

## Task 5: Cross-driver model (`DriverDeltaNet`)

Architecture identical to old plan Task 5, head reinterpreted as a per-driver delta. **No** `-1e4` masked_fill.

**Files:** Create `src/f1_predictor/models/transformer.py`, `tests/test_transformer_unit.py`.

- [ ] **Step 1 — failing tests.** Reuse old plan Task 5's three tests (output shape `(B,20)`; permutation-equivariance on valid slots in `.eval()`). Replace the "padded slots scored −1e3" test with:

```python
def test_padded_slots_do_not_change_valid_outputs():
    # padding must not leak through attention into the active slots' deltas.
    torch.manual_seed(0)
    model = DriverDeltaNet(num_features=6, d_model=32, n_heads=4, n_layers=2, num_drivers=30).eval()
    feats = torch.randn(1, 5, 6); didx = torch.tensor([[1,2,3,4,5]])
    valid = torch.ones(1, 5, dtype=torch.bool)
    with torch.no_grad():
        base = model(feats, didx, valid)[0]
        # append 3 padded slots with garbage features/idx
        feats2 = torch.cat([feats, torch.randn(1,3,6)], dim=1)
        didx2 = torch.cat([didx, torch.zeros(1,3,dtype=torch.long)], dim=1)
        valid2 = torch.cat([valid, torch.zeros(1,3,dtype=torch.bool)], dim=1)
        out2 = model(feats2, didx2, valid2)[0][:5]
    assert torch.allclose(base, out2, atol=1e-5)
```

- [ ] **Step 2 — implement** `DriverDeltaNet(num_features, d_model=128, n_heads=8, n_layers=4, dropout=0.1, num_drivers=30)`:
  - `input_proj = nn.Linear(num_features, d_model)`
  - `driver_embed = nn.Embedding(num_drivers, d_model, padding_idx=0)`
  - pre-LN `TransformerEncoderLayer(..., batch_first=True, norm_first=True, activation="gelu")`, `n_layers`
  - `head = nn.Linear(d_model, 1)`
  - `forward(features, driver_idx, valid)`: `h = input_proj(features) + driver_embed(driver_idx)`; `h = encoder(h, src_key_padding_mask=~valid)`; `return head(h).squeeze(-1)` (raw delta, `[B,20]`).
  - **Padding-leak note for the executor:** with `padding_idx=0` and `src_key_padding_mask`, padded *keys* are excluded from attention, so valid outputs are independent of padded slots — the new test guards this. (`norm_first` keeps it stable on small data.)

- [ ] **Step 3 — verify pass; commit** `feat: transformer — cross-driver DriverDeltaNet (delta head)`.

---

## Task 6: Training loop (train on `train`, early-stop on `val` Spearman)

AdamW + 500-step warmup→cosine, masked-L1 loss, **early stop on val Spearman** (the eval metric), run dir + MLflow. Reconstructs scores from delta for evaluation.

**Files:** Create `src/f1_predictor/train.py`, `tests/test_train_integration.py`.

Key sub-pieces:
- `_predict_split(model, dataset, device) -> pl.DataFrame`: per valid slot emit `{session_key, snapshot_lap, driver_number, final_position, score}` where **`score = delta_hat − current_rank`** (`delta_hat` = model output, `current_rank` from the tensor). This column set feeds `ranking_metrics` directly.
- `train_transformer(snapshots_dir, runs_dir, config, use_mlflow=True) -> {"run_dir","metrics"}`:
  1. `meta = load_metadata`; `feature_columns = meta["feature_columns"]` (**never hard-code F**).
  2. Load `train` + `val` splits (`load_split`). **`test` is never loaded here.**
  3. `driver_index = build_driver_index(train_df, config["num_drivers"])`.
  4. Datasets/loaders; model on auto device (cuda if available else cpu).
  5. Each epoch: train with `masked_l1_loss(model(...), batch["delta"], batch["valid"])`; then eval `ranking_metrics(_predict_split(model, val_ds))`; **track best val Spearman, patience ~10, keep best state_dict.**
  6. Restore best; final metrics = best val metrics; write `runs/{run_id}/` (`model.pt`, `predictions_val.parquet`, `metrics.json`, `config.json` incl. `feature_columns`, `driver_index`, `data_version`, `eval_split: "val"`).
  7. MLflow experiment `f1-transformer` (guarded by `use_mlflow`).

- [ ] **Step 1 — failing test (overfit a trivially-learnable set).** Build a tiny `train`/`val` (reuse old plan Task 6's snapshot writer but include a `position` column so `prepare_split` works; targets where final order is a deterministic function of features). Assert `model.pt`, `metrics.json`, `predictions_val.parquet` exist and `result["metrics"]["spearman"] > 0.9`. Config: `{"d_model":32,"n_heads":4,"n_layers":2,"epochs":60,"lr":1e-3,"warmup_steps":5,"batch_size":4,"num_drivers":30,"num_slots":20,"patience":20}`.
- [ ] **Step 2 — implement** per above (warmup+cosine helper verbatim from old plan).
- [ ] **Step 3 — verify pass; commit** `feat: transformer — training loop (val early-stop, delta→score, run dir)`.

---

## Task 7: CLI, head-to-head on `val`, evolution plot

**Files:** Create `scripts/train_transformer.py`, `scripts/compare_val.py`, `src/f1_predictor/visualise.py`, `tests/test_visualise_unit.py`.

- [ ] **Step 1 — visualiser** (verbatim from old plan Task 8 + its PNG test). It groups by driver and plots predicted rank vs snapshot lap — works unchanged on the `predictions_val.parquet` columns.
- [ ] **Step 2 — `scripts/train_transformer.py`** Typer CLI: starting hyperparams `d_model=128, n_heads=8, n_layers=4, dropout=0.1, lr=1e-4, weight_decay=0.01, warmup_steps=500, batch_size=32, epochs=100, patience=10, num_drivers=30, num_slots=20`; flags `--epochs`, `--no-mlflow`. Prints run dir + metrics.
- [ ] **Step 3 — `scripts/compare_val.py`** the head-to-head. Load `train`+`val` snapshots and `metadata`. Compute three score sets **on `val`**:
  - **Transformer:** load the run's `model.pt` + `driver_index`, `_predict_split` on `val` → `ranking_metrics`.
  - **GBM:** `train_baseline(train, feature_columns)` then `predict(model, val, feature_columns)` → build preds → `ranking_metrics`.
  - **Naive:** `naive_predict(val)` → build preds → `ranking_metrics`.
  Print a 3-row table (spearman, top1, top3, mpe, n_groups) and the verdict line: does the Transformer beat the GBM and naive on val Spearman? Log to MLflow `f1-transformer-vs-baseline`.
- [ ] **Step 4 — real run (`[data-run]`, lead executes):**
  ```bash
  uv run pytest tests/ -q                                   # all green first
  uv run python scripts/build_snapshots.py                  # ensure snapshots exist (or scripts/run_pipeline.py)
  uv run python scripts/train_transformer.py --no-mlflow
  uv run python scripts/compare_val.py --run-dir runs/<id> --no-mlflow
  ```
  Record the 3-way val numbers. If the Transformer underperforms the GBM, that is a **real finding** (a per-row GBM already captures most signal on this small dataset) — inspect `predictions_val.parquet` and the loss curve, try fewer layers / higher lr / more epochs, and confirm the overfit test still passes before accepting it. Do **not** chase the number by touching `test` (2026).
- [ ] **Step 5 — commit** `feat: transformer — CLI, val head-to-head, evolution plot`.

---

## Verification

- **Unit:** `uv run pytest tests/test_data_loader_unit.py tests/test_losses_unit.py tests/test_transformer_unit.py -v` — driver index, current_rank/delta carry, padding, masked-L1 correctness/invariance, model shape + permutation-equivariance + no padding leak.
- **Integration:** `uv run pytest tests/test_train_integration.py -v` — overfits a trivial set to val Spearman > 0.9 and writes a complete run dir.
- **End-to-end (`[data-run]`):** the Task 7 Step 4 sequence produces a run dir and a 3-way `val` comparison table (Transformer vs GBM vs naive) plus an evolution PNG. Success = the comparison is produced and interpretable; the headline question (does cross-driver attention beat the GBM on the same-regime 2025 split?) is answered with evidence either way.
- **Guardrails to assert during review:** `F` is read from `metadata.feature_columns` (never hard-coded); `current_rank`/`delta` come from the reused `add_current_rank`; evaluation loads **only** `train`+`val` (grep that `train.py`/`compare_val.py` never read `test.parquet`); scores are `delta_hat − current_rank` so they're on the same scale as GBM and naive.

## Out of scope (follow-ups)

- Any 2026 evaluation / walk-forward integration — gated on the adaptation experiment and the ≥10-race data bottleneck (separate plan).
- Temporal encoding (per-driver LSTM wrapper) — only if the cross-driver model plateaus and wins enough to justify it.
- Pairwise ranking losses — deliberately dropped; revisit only if delta-regression underperforms and a ranking objective is worth re-testing on the Transformer.
