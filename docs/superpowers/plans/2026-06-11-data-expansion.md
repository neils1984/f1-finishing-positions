# Data Expansion + 2026 Drift Diagnostic — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Grow the training data to all complete seasons (2023–2025), then measure how badly the model degrades on the regulation-changed 2026 season ("drift diagnostic") — to decide whether the GBM generalises, whether adaptation is needed, and whether the Transformer is still warranted.

**Architecture:** Measure-first. Train on every complete season (2023+2024+2025) and hold out **all of 2026-so-far as a pure test set** to quantify distribution shift. A shared naive-baseline harness is the yardstick. Three data levers are applied as separately-measured increments: (A) add 2025+2026 with a generalised chronological split, (B) denser snapshot laps, (C) backfill `max_speed_kmh`. Adaptation machinery (regulation-era feature, sample weighting, fine-tuning) is **staged but deferred** — it only activates once 2026 enters *training*, which this plan does not do.

**Tech Stack:** Polars, DuckDB, LightGBM, Typer, pytest, OpenF1 REST API, `uv`.

---

## Context & Findings (read before starting)

Verified against the live API and current code on 2026-06-11:

- **OpenF1 has no pre-2023 data.** Backward history is impossible. Complete seasons available: **2023, 2024, 2025**. **2026 is in-progress** (~8 completed races; the API lists the full 24-race calendar but unraced events return empty `/laps`, which `sessionise` already skips).
- **On disk today:** 2023 (21 usable races) + 2024 (23). **2025 and 2026 are NOT pulled.**
- **2026 has new technical regulations** → materially different racing (more overtaking). The hypothesis under test: **current grid position predicts final order less well in 2026, so the naive baseline degrades** — and that degradation measures the drift.
- **Why pooled training does NOT auto-adapt:** a model trained on 2023–2026 learns the *average* regime, dominated by the 2023–2025 majority (~67 races vs ~8). The Transformer's cross-driver attention models *relative race state within a race*, not *temporal regime shift across seasons* — neither GBM nor Transformer "knows the year" unless given a regime signal. Adaptation requires an explicit mechanism (regulation-era feature / sample weighting / fine-tuning), all deferred here until the diagnostic justifies them.
- **`assign_split` uses a single `val_cutoff` + hardcoded `_VAL_YEAR = 2024`** (`src/f1_predictor/snapshots.py:19,26`). Supporting three regime-separated blocks (pre-2026 train / late-2025 val / 2026 test) needs a **two-boundary** split. Task 2 generalises it.
- **Current committed baseline:** Option B L1 delta-regression, test Spearman **0.836** vs naive **0.809** (train=2023 / val=2024H1 / test=2024H2, laps `[10,20,30,40]`).
- **`max_speed_kmh` is null everywhere** — `sessionise._add_car_data` (`sessionise.py:193`) aggregates it *when a `car_data.parquet` exists*, but the ingest never pulls car_data (a per-session query 422s; must be fetched per `driver_number`). See [[openf1-real-data-gotchas]].

**New chronological split (two date boundaries):**
- `train` = races before `val_start` (`2025-09-01`) → 2023 + 2024 + early/mid 2025
- `val` = `[val_start, test_start)` (`2025-09-01` … `2026-01-01`) → late-2025 slice, for early-stopping / a same-regime reference
- `test` = on/after `test_start` (`2026-01-01`) → **all 2026-so-far, the drift probe**

**Standing convention:** real-data pulls and real-training runs (`[data-run]`) are executed directly by the lead, not delegated to synthetic-data subagents. Code tasks (`[code]`, synthetic-data tests) are subagent-friendly.

---

## File Structure

- **Create** `scripts/eval_naive.py` — prints naive (and optional model) Spearman per split + per lap. The yardstick.
- **Modify** `src/f1_predictor/models/baseline_gbm.py` — add tested `naive_predict(df)` (score = −current_rank).
- **Modify** `src/f1_predictor/snapshots.py` — generalise `assign_split` / `build_snapshots` to `val_start` + `test_start`; drop `_VAL_YEAR`.
- **Modify** `scripts/build_snapshots.py` + `config/default.yaml` — two-boundary split config, denser `snapshot_laps`, `seasons: [2023,2024,2025,2026]`.
- **Modify** `src/f1_predictor/ingest.py` — `pull_car_data(session_key, driver_numbers, raw_dir)`; wire into `pull_session`.
- **Modify** `src/f1_predictor/features.py` — add inert-for-now `is_2026_regs` regulation-era feature (adaptation-prep, Task 8).
- **Test** `tests/test_baseline_gbm.py`, `tests/test_snapshots_unit.py`, `tests/test_ingest.py`, `tests/test_features_unit.py`.

---

## Task 1: Naive-baseline measurement harness `[code]`

Yardstick used at every checkpoint. Naive score = −current_rank (predict final order = current order).

**Files:** Modify `src/f1_predictor/models/baseline_gbm.py`; Create `scripts/eval_naive.py`; Test `tests/test_baseline_gbm.py`.

- [ ] **Step 1: Write the failing test**

```python
def test_naive_predict_scores_current_order():
    # Naive = predict no movement: score is strictly decreasing in current
    # position, so P1 (lowest position) gets the highest score.
    df = pl.DataFrame({
        "session_key": [0, 0, 0, 0],
        "snapshot_lap": [20, 20, 20, 20],
        "driver_number": [44, 1, 16, 55],
        "position": [-1.5, -0.4, 0.6, 1.7],   # standardised but monotonic
        "final_position": [2, 1, 4, 3],
    })
    from f1_predictor.models.baseline_gbm import naive_predict
    scores = naive_predict(df)
    order = df.with_columns(pl.Series("score", scores)).sort("score", descending=True)
    assert order["driver_number"].to_list() == [44, 1, 16, 55]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_baseline_gbm.py::test_naive_predict_scores_current_order -v`
Expected: FAIL — `ImportError: cannot import name 'naive_predict'`.

- [ ] **Step 3: Implement**

Add to `src/f1_predictor/models/baseline_gbm.py`:

```python
def naive_predict(df: pl.DataFrame) -> np.ndarray:
    """Naive persistence baseline: score = -current_rank (predict no movement)."""
    df = add_current_rank(df)
    return -df["current_rank"].to_numpy().astype(float)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_baseline_gbm.py::test_naive_predict_scores_current_order -v`
Expected: PASS.

- [ ] **Step 5: Write the eval script**

Create `scripts/eval_naive.py`:

```python
"""CLI: per-split, per-lap Spearman for the naive baseline (and optionally a
run's predictions). The yardstick for the drift diagnostic and every checkpoint.

Reports val (same-regime reference) and test (the drift probe) so naive
degradation from one to the other is visible in a single run."""
from pathlib import Path

import polars as pl
import typer

from f1_predictor.evaluate import ranking_metrics
from f1_predictor.models.baseline_gbm import naive_predict

app = typer.Typer(add_completion=False)


def _report(label: str, preds: pl.DataFrame) -> None:
    if preds.is_empty():
        typer.echo(f"\n=== {label} === (empty)"); return
    m = ranking_metrics(preds)
    typer.echo(f"\n=== {label} ===")
    typer.echo(f"overall spearman={m['spearman']:.4f}  top1={m['top1_accuracy']:.3f}  "
               f"top3={m['top3_accuracy']:.3f}  mpe={m['mean_position_error']:.3f}  "
               f"n_groups={m['n_groups']}")
    for lap in sorted(preds["snapshot_lap"].unique().to_list()):
        ml = ranking_metrics(preds.filter(pl.col("snapshot_lap") == lap))
        typer.echo(f"  lap{lap}: spearman={ml['spearman']:.4f}  n_groups={ml['n_groups']}")


def _naive_frame(df: pl.DataFrame) -> pl.DataFrame:
    return df.select(["session_key", "snapshot_lap", "driver_number", "final_position"]).with_columns(
        pl.Series("score", naive_predict(df))
    )


@app.command()
def main(
    snapshots_dir: Path = typer.Option(Path("data/snapshots"), "--snapshots-dir"),
    run_dir: Path = typer.Option(None, "--run-dir", help="Optional run dir to compare on TEST"),
) -> None:
    for split in ("val", "test"):
        p = snapshots_dir / f"{split}.parquet"
        if p.exists():
            df = pl.read_parquet(p)
            if not df.is_empty():
                _report(f"NAIVE on {split.upper()}", _naive_frame(df))
    if run_dir is not None:
        _report(f"MODEL ({run_dir.name}) on TEST", pl.read_parquet(run_dir / "predictions_test.parquet"))


if __name__ == "__main__":
    app()
```

- [ ] **Step 6: Verify against current data**

Run: `uv run pytest tests/test_baseline_gbm.py -q` → all pass.
Run: `uv run python scripts/eval_naive.py` → prints NAIVE on TEST overall ≈ 0.809 (sanity check vs the known baseline; val may be empty on current data).

- [ ] **Step 7: Commit**

```bash
git add src/f1_predictor/models/baseline_gbm.py scripts/eval_naive.py tests/test_baseline_gbm.py
git commit -m "feat: naive-baseline measurement harness (eval_naive + naive_predict)"
```

---

## Task 2: Generalise the chronological split to two date boundaries `[code]`

Replace `val_cutoff` + hardcoded `_VAL_YEAR` with explicit `val_start` / `test_start` so train can span 2023–mid-2025, val = late-2025, test = 2026.

**Files:** Modify `src/f1_predictor/snapshots.py`, `scripts/build_snapshots.py`, `config/default.yaml`; Test `tests/test_snapshots_unit.py`.

- [ ] **Step 1: Write the failing test** (and update the existing `assign_split` tests to the new signature)

```python
def test_assign_split_uses_two_date_boundaries():
    from f1_predictor.snapshots import assign_split
    vs, ts = "2025-09-01", "2026-01-01"
    assert assign_split("2023-03-05T15:00:00+00:00", vs, ts) == "train"
    assert assign_split("2024-09-01T13:00:00+00:00", vs, ts) == "train"
    assert assign_split("2025-04-01T13:00:00+00:00", vs, ts) == "train"  # early 2025 -> train
    assert assign_split("2025-10-01T13:00:00+00:00", vs, ts) == "val"    # late 2025 -> val
    assert assign_split("2026-03-15T13:00:00+00:00", vs, ts) == "test"   # 2026 -> test
```

Replace the four existing `assign_split` tests (`test_assign_split_2023_is_train`, `..._2024_before_cutoff_is_val`, `..._2024_on_or_after_cutoff_is_test`, `..._pre_2023_is_train`, lines ~46–61) with the test above — they assert the old single-cutoff signature and will otherwise fail to compile. Also update the `build_snapshots(...)` call in `test_snapshots_unit.py` (~line 117): replace `val_cutoff="2024-07-01"` with `val_start="2024-07-01", test_start="2099-01-01"` (so the existing fixture's 2-driver race still lands in val as before — pick boundaries that preserve that test's intent; verify by running it).

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_snapshots_unit.py::test_assign_split_uses_two_date_boundaries -v`
Expected: FAIL — `TypeError: assign_split() takes 2 positional arguments but 3 were given`.

- [ ] **Step 3: Implement**

In `src/f1_predictor/snapshots.py`, replace `assign_split` and drop the `_VAL_YEAR` constant:

```python
def assign_split(date_start: str, val_start: str, test_start: str) -> str:
    """Classify a race into 'train' | 'val' | 'test' by two date boundaries.

    train: before val_start.  val: [val_start, test_start).  test: >= test_start.
    """
    d = datetime.fromisoformat(date_start).date()
    if d < datetime.fromisoformat(val_start).date():
        return "train"
    if d < datetime.fromisoformat(test_start).date():
        return "val"
    return "test"
```

Update `build_snapshots` to take `val_start: str, test_start: str` (replacing `val_cutoff`) and pass them to `assign_split`.

- [ ] **Step 4: Run tests to verify**

Run: `uv run pytest tests/test_snapshots_unit.py -q`
Expected: all pass.

- [ ] **Step 5: Wire config + CLI**

In `config/default.yaml`, replace the `val_cutoff` line and `seasons`:

```yaml
seasons: [2023, 2024, 2025, 2026]
val_start: "2025-09-01"
test_start: "2026-01-01"
```

In `scripts/build_snapshots.py`, change the `build_snapshots(...)` call:

```python
        snapshot_laps=cfg["snapshot_laps"], val_start=cfg["val_start"],
        test_start=cfg["test_start"], git_sha=_git_sha(),
```

- [ ] **Step 6: Commit**

```bash
git add src/f1_predictor/snapshots.py scripts/build_snapshots.py config/default.yaml tests/test_snapshots_unit.py
git commit -m "feat: two-boundary chronological split (enables 2026 hold-out test)"
```

---

## Task 3: Pull 2025 and 2026 `[data-run]`

Network pull executed by the lead. 2025 full (~23 races after Monaco); 2026 partial (completed races only — empty `/laps` for unraced events is skipped downstream).

- [ ] **Step 1: Pull**

Run: `uv run python scripts/pull_season.py --year 2025`
Run: `uv run python scripts/pull_season.py --year 2026`
Expected: ~23 sessions for 2025, ~8 for 2026 (varies with the calendar to date). Sprints + Monaco auto-excluded.

- [ ] **Step 2: Sanity-check**

Run: `uv run python -c "from pathlib import Path; d=Path('data/raw'); print('sessions:', sum(1 for p in d.iterdir() if p.is_dir() and (p/'meta.json').exists()))"`
Expected: count rises by ~31 vs before (was 45).
Confirm a 2026 race has real laps (find one with the latest date_start):
Run: `uv run python -c "import polars as pl, glob; fs=glob.glob('data/raw/*/sessions.parquet'); rows=[(pl.read_parquet(f).row(0,named=True)['date_start'], f) for f in fs]; d,f=sorted(rows)[-1]; print(d, f); print('laps:', pl.read_parquet(f.replace('sessions','laps')).height)"`
Expected: a 2026 date and a non-zero lap count.

- [ ] **Step 3: No commit** — `data/` is gitignored. Log the new session keys.

---

## Task 4: Re-run Stages 2–3, rebuild snapshots, RUN THE DRIFT DIAGNOSTIC `[data-run]`

The centerpiece. Keep the **current `[10,20,30,40]` grid** here so the 2026 number is directly comparable to the committed 0.836; densify later (Task 5).

- [ ] **Step 1: Sessionise + features for all races**

Run: `uv run python scripts/run_pipeline.py`
Expected: "Sessionised M races" / "Built features for M races" with M ≈ 75 (2023+2024+2025+2026-so-far). Watch for new-driver / new-circuit issues; priors use only prior races, so they self-populate chronologically.

- [ ] **Step 2: Rebuild snapshots with the two-boundary split**

Temporarily ensure `config/default.yaml` still has `snapshot_laps: [10, 20, 30, 40]` for this comparison.
Run: `uv run python scripts/build_snapshots.py`
Expected: `train` ≈ 2023+2024+early-2025, `val` = late-2025 slice, `test` = 2026-so-far. New `data_version=`.

- [ ] **Step 3: Train Option B + run the diagnostic**

Run: `uv run python scripts/train_baseline.py --no-mlflow`
Run: `uv run python scripts/eval_naive.py --run-dir runs/<new-run-id>`

- [ ] **Step 4: Interpret — DRIFT DIAGNOSTIC**

Record and compare:
- **Naive on VAL (late-2025) vs Naive on TEST (2026).** A large drop test-vs-val confirms the more-overtaking hypothesis (persistence breaks in 2026).
- **Model on TEST vs Naive on TEST.** Does Option B still beat naive on 2026 (transferred dynamics) or fall behind (overfit to old persistence)?
- **Per-lap, early laps especially** (lap 10) — where overtaking shows most and the project's lap-by-lap-confidence goal lives.

This is the result that decides whether adaptation (a later phase) is needed. Write the three numbers + verdict into a memory updating [[baseline-is-delta-regression-not-lambdarank]].

- [ ] **Step 5: No code commit** (config/data only; data gitignored).

---

## Task 5: Denser snapshot laps `[code]` + re-measure `[data-run]`

Densify the grid for finer early-race resolution, then re-run the diagnostic.

**Files:** Modify `config/default.yaml`; Test `tests/test_snapshots_unit.py`.

- [ ] **Step 1: Characterisation test for absent-lap tolerance**

```python
def test_extract_snapshots_skips_laps_absent_from_short_race():
    feats = pl.DataFrame({
        "session_key": [9001, 9001], "lap_number": [5, 10], "driver_number": [1, 1],
        "final_position": [3, 3], "position": [3, 3], "gap_to_leader": [1.0, 1.0],
    })
    out = extract_snapshots(feats, snapshot_laps=[5, 10, 50],
                            feature_columns=["position", "gap_to_leader"])
    assert sorted(out["snapshot_lap"].unique().to_list()) == [5, 10]
    assert out.height == 2
```

- [ ] **Step 2: Run it**

Run: `uv run pytest tests/test_snapshots_unit.py::test_extract_snapshots_skips_laps_absent_from_short_race -v`
Expected: PASS (characterisation — the `lap_number.is_in(snapshot_laps)` filter already tolerates absent laps). If it FAILS, something regressed — fix `extract_snapshots`.

- [ ] **Step 3: Densify config**

In `config/default.yaml`:

```yaml
snapshot_laps: [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
```

- [ ] **Step 4: Rebuild + re-measure `[data-run]`**

Run: `uv run python scripts/build_snapshots.py`
Run: `uv run python scripts/train_baseline.py --no-mlflow`
Run: `uv run python scripts/eval_naive.py --run-dir runs/<new-run-id>`
**Checkpoint:** does the finer grid change the 2026 model-vs-naive gap, especially at laps 5–15?

- [ ] **Step 5: Commit**

```bash
git add config/default.yaml tests/test_snapshots_unit.py
git commit -m "feat: denser snapshot-lap grid (5..50 by 5) + absent-lap tolerance test"
```

---

## Task 6: Backfill `max_speed_kmh` from per-driver car telemetry `[code]`

Heaviest, most-uncertain lever — **may be deferred** if earlier checkpoints already settle the Transformer question. Per-session car_data 422s; fetch per `driver_number`.

**Files:** Modify `src/f1_predictor/ingest.py`; Test `tests/test_ingest.py`.

- [ ] **Step 1: Write the failing test** (mock HTTP; no network)

```python
def test_pull_car_data_concatenates_per_driver(tmp_path, monkeypatch):
    from f1_predictor import ingest

    calls = []

    class FakeResp:
        status_code = 200
        def json(self):
            dn = int(calls[-1].split("driver_number=")[1])
            return [{"driver_number": dn, "date": "2025-03-16T05:00:00.000000+00:00", "speed": 250 + dn}]
        def raise_for_status(self): pass

    class FakeSession:
        def get(self, url, timeout=30):
            calls.append(url); return FakeResp()
        def __enter__(self): return self
        def __exit__(self, *a): pass

    monkeypatch.setattr(ingest.requests, "Session", lambda: FakeSession())

    out = tmp_path / "9999"; out.mkdir()
    ingest.pull_car_data(9999, driver_numbers=[1, 44], raw_dir=tmp_path)

    import polars as pl
    df = pl.read_parquet(out / "car_data.parquet")
    assert set(df["driver_number"].to_list()) == {1, 44}
    assert df.height == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ingest.py::test_pull_car_data_concatenates_per_driver -v`
Expected: FAIL — `AttributeError: module 'f1_predictor.ingest' has no attribute 'pull_car_data'`.

- [ ] **Step 3: Implement + wire in**

Add to `src/f1_predictor/ingest.py`:

```python
def pull_car_data(session_key: int, driver_numbers: list[int], raw_dir: Path) -> None:
    """Pull car telemetry per driver and write one car_data.parquet.

    A per-session car_data query 422s ("too much data"), so query per
    driver_number. Keep only what _add_car_data needs: driver_number, date, speed.
    """
    frames: list[pl.DataFrame] = []
    with requests.Session() as s:
        for dn in driver_numbers:
            url = f"{OPENF1_BASE}/car_data?session_key={session_key}&driver_number={dn}"
            data = _fetch(s, url)
            if data:
                frames.append(
                    pl.DataFrame(data, infer_schema_length=None)
                    .select(["driver_number", "date", "speed"])
                )
            time.sleep(_BASE_DELAY)
    df = pl.concat(frames, how="vertical") if frames else pl.DataFrame()
    df.write_parquet(raw_dir / str(session_key) / "car_data.parquet")
```

In `pull_session`, after the `ENDPOINTS` loop (which writes `drivers.parquet`):

```python
    # car_data is pulled per driver (a per-session query 422s) — see module docstring.
    drivers_df = pl.read_parquet(session_dir / "drivers.parquet")
    driver_numbers = drivers_df["driver_number"].unique().to_list() if not drivers_df.is_empty() else []
    pull_car_data(session_key, driver_numbers, raw_dir)
    row_counts["car_data"] = pl.read_parquet(session_dir / "car_data.parquet").height
```

(Keep `car_data` out of `ENDPOINTS`; `sessionise.py:21` already reads `car_data` from the raw dir.)

- [ ] **Step 4–5: Verify**

Run: `uv run pytest tests/test_ingest.py::test_pull_car_data_concatenates_per_driver -v` → PASS.
Run: `uv run pytest tests/test_ingest.py -q` → all pass.

- [ ] **Step 6: Commit**

```bash
git add src/f1_predictor/ingest.py tests/test_ingest.py
git commit -m "feat: per-driver car_data pull to backfill max_speed_kmh"
```

---

## Task 7: Re-pull car_data, re-run pipeline, final measurement `[data-run]`

- [ ] **Step 1: Force re-pull so cached races gain car_data**

Run: `uv run python scripts/pull_season.py --year 2023 --force`
Run: `uv run python scripts/pull_season.py --year 2024 --force`
Run: `uv run python scripts/pull_season.py --year 2025 --force`
Run: `uv run python scripts/pull_season.py --year 2026 --force`
**Heaviest network step** (~20 drivers × ~75 races). Honours Retry-After; expect a long run. Spot-check:
Run: `uv run python -c "import polars as pl, glob; f=sorted(glob.glob('data/raw/*/car_data.parquet'))[-1]; df=pl.read_parquet(f); print(f, df.height, df['speed'].max())"`
Expected: large row count, plausible top speed (~300–360).

- [ ] **Step 2: Re-run Stages 2–4**

Run: `uv run python scripts/run_pipeline.py`
Run: `uv run python scripts/build_snapshots.py`
Run: `uv run python -c "import polars as pl; df=pl.read_parquet('data/snapshots/train.parquet'); print('max_speed non-null frac:', df['max_speed_kmh'].is_not_null().mean())"`
Expected: well above 0 (was 0 before).

- [ ] **Step 3: Final measurement `[data-run]`**

Run: `uv run python scripts/train_baseline.py --no-mlflow`
Run: `uv run python scripts/eval_naive.py --run-dir runs/<new-run-id>`
**Checkpoint:** does reviving `max_speed_kmh` move the 2026 model-vs-naive gap?

---

## Task 8: Stage the regulation-era feature (adaptation-prep) `[code]`

Add an `is_2026_regs` context feature now so it's ready when 2026 enters *training* in a future adaptation phase. **It is inert in this plan's diagnostic** (constant-False across all-pre-2026 training → zero variance → scaler maps it to 0 → GBM can't split on it), so it must not change the diagnostic numbers — a guard test asserts a constant feature is harmless.

**Files:** Modify `src/f1_predictor/features.py`; Test `tests/test_features_unit.py`.

- [ ] **Step 1: Write the failing test**

```python
def test_is_2026_regs_flags_regulation_era():
    # The feature is True for 2026+ races, False otherwise — the principled slot
    # for "this is a different regulation regime" (analogous to is_street_circuit).
    from f1_predictor.features import _regulation_era_flag  # helper under test
    assert _regulation_era_flag("2026-03-15T13:00:00+00:00") is True
    assert _regulation_era_flag("2025-03-15T13:00:00+00:00") is False
    assert _regulation_era_flag("2023-07-01T13:00:00+00:00") is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_features_unit.py::test_is_2026_regs_flags_regulation_era -v`
Expected: FAIL — `ImportError: cannot import name '_regulation_era_flag'`.

- [ ] **Step 3: Implement**

In `src/f1_predictor/features.py`:

```python
def _regulation_era_flag(date_start: str) -> bool:
    """True for the 2026+ technical-regulation era (different car/racing dynamics)."""
    from datetime import datetime
    return datetime.fromisoformat(date_start).year >= 2026
```

Add `is_2026_regs` to the per-race feature construction (set from the race's `date_start`, like other race-level constants) and append `"is_2026_regs"` to `FEATURE_COLUMNS`. Cast to Float64 in the snapshot pipeline path as with other booleans (the existing `_impute` handles bool→float).

- [ ] **Step 4: Verify**

Run: `uv run pytest tests/test_features_unit.py -q` → all pass.
Run: `uv run pytest tests/ -q` → full suite green (the new 31st feature must not break snapshot/scaler tests; if a test hardcodes 30 features, update it).

- [ ] **Step 5: Re-run the diagnostic to confirm inertness `[data-run]`**

Run: `uv run python scripts/run_pipeline.py && uv run python scripts/build_snapshots.py && uv run python scripts/train_baseline.py --no-mlflow`
Expected: test Spearman essentially unchanged from Task 7 (the feature is constant in train → inert). Confirms the column is safely staged for adaptation.

- [ ] **Step 6: Commit**

```bash
git add src/f1_predictor/features.py tests/test_features_unit.py
git commit -m "feat: stage is_2026_regs regulation-era feature (inert until 2026 trains)"
```

---

## Decision Checkpoint + Deferred Adaptation Phase

After Task 4 (and refined by 5/7), use the drift numbers to decide:

- **If naive barely degrades on 2026 and Option B still beats it** → 2026 dynamics transfer; no urgent adaptation. Proceed toward the Transformer (`docs/superpowers/plans/2026-06-08-transformer.md`, revised to benchmark Option B) or the lap-by-lap visualiser.
- **If naive collapses and/or the model underperforms naive on 2026** → drift is real. Open a **follow-up adaptation plan** that *includes 2026 in training* (respecting within-2026 chronology) and tries, in order: (1) activate `is_2026_regs` (now staged), (2) per-sample loss weighting with a tuned 2026 upweight measured on a held-out 2026 slice, (3) pre-train pre-2026 → fine-tune on 2026 (only if enough 2026 races have accrued). The Transformer needs the same regime signal — attention alone does not adapt to season drift.

This plan deliberately measures before adapting; adaptation is a separate, evidence-gated plan.

---

## Self-Review Notes

- **Spec coverage:** 2026 test (Tasks 3–4), more seasons (2025; Tasks 2–4), denser laps (Task 5), `max_speed_kmh` backfill (Tasks 6–7), regulation-era feature staged (Task 8), drift accounted for via measure-first + a deferred adaptation phase.
- **Key correction baked in:** the Transformer does not auto-adapt to season drift; the regime feature is inert until 2026 enters training (Task 8 guards this).
- **Ordering:** harness → split refactor → pull → **diagnostic (current grid, directly comparable to 0.836)** → densify → telemetry backfill → stage regime feature. Cheapest/most-informative first; heaviest/most-uncertain last and deferrable.
- **Methodology guardrail:** 2026 is in-progress — it is used only as a *held-out test* here; any future training on it must respect within-2026 chronology.
- **Splittable:** Tasks 6–8 are cleanly separable into follow-up plans; Tasks 1–5 stand alone and deliver the diagnostic.
