# Data Expansion for the LightGBM Baseline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Grow the training data and revive the one dead feature so the Option B LightGBM baseline is measured at its true ceiling — to decide whether the Transformer is still needed.

**Architecture:** Three independent data levers, applied as separately-measured increments so each one's effect on test Spearman vs the naive baseline is attributable: (A) denser snapshot laps, (B) add the 2025 season + re-split chronologically, (C) backfill `max_speed_kmh` from per-driver car telemetry. A shared, tested naive-baseline harness (Task 1) is the yardstick for every checkpoint.

**Tech Stack:** Polars, DuckDB, LightGBM, Typer, pytest, OpenF1 REST API, `uv`.

---

## Context & Findings (read before starting)

These were verified against the live API and current code on 2026-06-11:

- **OpenF1 has no pre-2023 data.** `sessions?year=2021` and `year=2022` return zero races. "More history" backward is impossible; the only new full season available is **2025** (24 races, complete) — currently **not pulled**. 2026 is only ~8 races into the calendar, so it is excluded here.
- **Current data on disk:** 2023 (21 usable races) + 2024 (23 usable races). 2025 roughly doubles the race count (44 → ~67).
- **`assign_split` hardcodes `_VAL_YEAR = 2024`** (`src/f1_predictor/snapshots.py:19`). Adding 2025 *requires* making the val year configurable, or 2024 races wrongly fall into val/test. Task 4 fixes this.
- **The current baseline** (committed): Option B L1 delta-regression, test Spearman **0.836** vs naive **0.809**, on splits train=2023 / val=2024-H1 / test=2024-H2, snapshot_laps `[10,20,30,40]`.
- **Snapshot-lap density is a weaker lever than it looks:** lap 10 and lap 15 of the *same* race are highly correlated, so denser laps add groups but not much *independent* signal, and they shift the evaluation mix earlier. Treat Group A as a measured experiment, not an assumed win.
- **The car_data ingest is the only missing piece for `max_speed_kmh`.** `sessionise._add_car_data` (`src/f1_predictor/sessionise.py:193`) already aggregates `speed`→`max_speed_kmh` per driver-lap *when a `car_data.parquet` exists*; today it doesn't, so the column is null everywhere. Group C only needs the per-driver *pull*. Per-session car_data returns HTTP 422 ("too much data") — it must be fetched per `driver_number` (see [[openf1-real-data-gotchas]]).

**New chronological split (applied in Group B):** train = 2023 + 2024, val = 2025 before `2025-07-01`, test = 2025 on/after `2025-07-01`. This gives ~2× the training races and a clean fully-held-out season for val/test.

**Standing convention (from prior work):** real-data pulls and real-training runs are executed directly by the human partner / lead, not delegated to synthetic-data subagents. Code tasks (with synthetic-data tests) are subagent-friendly. Each task below is tagged `[code]` or `[data-run]` accordingly.

---

## File Structure

- **Create** `scripts/eval_naive.py` — CLI that prints overall + per-lap Spearman for the naive baseline and (optionally) a run's predictions, against a snapshots dir. Reusable yardstick for every checkpoint.
- **Modify** `src/f1_predictor/models/baseline_gbm.py` — add a tested `naive_predict(df)` helper (score = −current_rank).
- **Modify** `src/f1_predictor/snapshots.py` — make `assign_split` / `build_snapshots` take a configurable `val_year`.
- **Modify** `scripts/build_snapshots.py` — pass `val_year` from config.
- **Modify** `config/default.yaml` — denser `snapshot_laps`, `seasons: [2023, 2024, 2025]`, `val_cutoff: "2025-07-01"`, add `val_year: 2025`.
- **Modify** `src/f1_predictor/ingest.py` — add `pull_car_data(session_key, driver_numbers, raw_dir)` writing `car_data.parquet`; wire it into `pull_session`.
- **Test** `tests/test_baseline_gbm.py`, `tests/test_snapshots_unit.py`, `tests/test_ingest.py` — new behaviour.

---

## Task 1: Naive-baseline measurement harness `[code]`

The yardstick used after every lever. Naive score = −current_rank (predict final order = current order).

**Files:**
- Modify: `src/f1_predictor/models/baseline_gbm.py`
- Create: `scripts/eval_naive.py`
- Test: `tests/test_baseline_gbm.py`

- [ ] **Step 1: Write the failing test**

```python
def test_naive_predict_scores_current_order():
    # Naive = predict no movement: score must be a strictly decreasing function
    # of current race position, so P1 (lowest position) gets the highest score.
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
    assert order["driver_number"].to_list() == [44, 1, 16, 55]  # = current order
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_baseline_gbm.py::test_naive_predict_scores_current_order -v`
Expected: FAIL with `ImportError: cannot import name 'naive_predict'`.

- [ ] **Step 3: Write minimal implementation**

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
"""CLI: print overall + per-lap Spearman for the naive baseline (and optionally
a run's predictions) on a snapshots dir. The yardstick for every experiment."""
from pathlib import Path

import polars as pl
import typer

from f1_predictor.evaluate import ranking_metrics
from f1_predictor.models.baseline_gbm import naive_predict

app = typer.Typer(add_completion=False)


def _report(label: str, preds: pl.DataFrame) -> None:
    m = ranking_metrics(preds)
    typer.echo(f"\n=== {label} ===")
    typer.echo(f"overall spearman={m['spearman']:.4f}  top1={m['top1_accuracy']:.3f}  "
               f"top3={m['top3_accuracy']:.3f}  mpe={m['mean_position_error']:.3f}  "
               f"n_groups={m['n_groups']}")
    for lap in sorted(preds["snapshot_lap"].unique().to_list()):
        ml = ranking_metrics(preds.filter(pl.col("snapshot_lap") == lap))
        typer.echo(f"  lap{lap}: spearman={ml['spearman']:.4f}  n_groups={ml['n_groups']}")


@app.command()
def main(
    snapshots_dir: Path = typer.Option(Path("data/snapshots"), "--snapshots-dir"),
    run_dir: Path = typer.Option(None, "--run-dir", help="Optional run dir to compare"),
) -> None:
    test = pl.read_parquet(snapshots_dir / "test.parquet")
    naive = test.select(["session_key", "snapshot_lap", "driver_number", "final_position"]).with_columns(
        pl.Series("score", naive_predict(test))
    )
    _report("NAIVE on TEST", naive)
    if run_dir is not None:
        preds = pl.read_parquet(run_dir / "predictions_test.parquet")
        _report(f"MODEL ({run_dir.name})", preds)


if __name__ == "__main__":
    app()
```

- [ ] **Step 6: Run full baseline test file + the script against current data**

Run: `uv run pytest tests/test_baseline_gbm.py -q`
Expected: all pass.
Run: `uv run python scripts/eval_naive.py`
Expected: prints NAIVE on TEST with overall spearman ≈ 0.809 (this reproduces the known baseline against current snapshots — sanity check the harness).

- [ ] **Step 7: Commit**

```bash
git add src/f1_predictor/models/baseline_gbm.py scripts/eval_naive.py tests/test_baseline_gbm.py
git commit -m "feat: naive-baseline measurement harness (eval_naive + naive_predict)"
```

---

## Task 2: Denser snapshot laps — config + tolerance test `[code]`

Change the snapshot grid from `[10,20,30,40]` to `[5,10,15,20,25,30,35,40,45,50]`. A snapshot lap absent from a short race must simply yield no rows for that race (not crash). Verify that tolerance, then re-measure.

**Files:**
- Modify: `config/default.yaml`
- Test: `tests/test_snapshots_unit.py`

- [ ] **Step 1: Write the failing test**

```python
def test_extract_snapshots_skips_laps_absent_from_short_race():
    # A race that ends at lap 12 must contribute rows only at laps that exist.
    feats = pl.DataFrame({
        "session_key": [9001, 9001],
        "lap_number": [5, 10],
        "driver_number": [1, 1],
        "final_position": [3, 3],
        "position": [3, 3],
        "gap_to_leader": [1.0, 1.0],
    })
    out = extract_snapshots(feats, snapshot_laps=[5, 10, 50],
                            feature_columns=["position", "gap_to_leader"])
    assert sorted(out["snapshot_lap"].unique().to_list()) == [5, 10]  # no lap 50
    assert out.height == 2
```

(Add alongside the existing snapshot unit tests; `extract_snapshots` is already imported there. If not, add `from f1_predictor.snapshots import extract_snapshots`.)

- [ ] **Step 2: Run test to verify it fails or passes**

Run: `uv run pytest tests/test_snapshots_unit.py::test_extract_snapshots_skips_laps_absent_from_short_race -v`
Expected: PASS immediately is acceptable here — this is a **characterisation test** locking in existing tolerant behaviour before we rely on it. If it FAILS, fix `extract_snapshots` so absent laps yield no rows (the `lap_number.is_in(snapshot_laps)` filter already does this; a failure means something else regressed).

- [ ] **Step 3: Update the config grid**

In `config/default.yaml` change:

```yaml
snapshot_laps: [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
```

- [ ] **Step 4: Rebuild snapshots from existing features**

Run: `uv run python scripts/build_snapshots.py`
Expected: prints train/val/test race counts and a new `data_version=`. Group counts per split should rise (more laps per race).

- [ ] **Step 5: Re-measure naive + Option B at the new grid `[data-run]`**

Run: `uv run python scripts/train_baseline.py --no-mlflow`
Run: `uv run python scripts/eval_naive.py --run-dir runs/<new-run-id>`
Record overall + per-lap Spearman for both naive and Option B. **Checkpoint A:** does denser sampling change the gap? (Expect early-lap groups to dominate and the absolute number to drop because early laps are harder — what matters is Option B vs naive *at each lap*, not the overall blend.)

- [ ] **Step 6: Commit**

```bash
git add config/default.yaml tests/test_snapshots_unit.py
git commit -m "feat: denser snapshot-lap grid (5..50 by 5) + absent-lap tolerance test"
```

---

## Task 3: Make the validation year configurable `[code]`

`assign_split` must take `val_year` so 2023+2024 → train when 2025 is the held-out season.

**Files:**
- Modify: `src/f1_predictor/snapshots.py`
- Test: `tests/test_snapshots_unit.py`

- [ ] **Step 1: Write the failing test**

```python
def test_assign_split_respects_configurable_val_year():
    from f1_predictor.snapshots import assign_split
    # val_year=2025: both 2023 and 2024 are train; 2025 splits on the cutoff.
    assert assign_split("2024-09-01T13:00:00", "2025-07-01", val_year=2025) == "train"
    assert assign_split("2025-04-01T13:00:00", "2025-07-01", val_year=2025) == "val"
    assert assign_split("2025-08-01T13:00:00", "2025-07-01", val_year=2025) == "test"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_snapshots_unit.py::test_assign_split_respects_configurable_val_year -v`
Expected: FAIL with `TypeError: assign_split() got an unexpected keyword argument 'val_year'`.

- [ ] **Step 3: Implement**

In `src/f1_predictor/snapshots.py`, replace the module constant usage:

```python
def assign_split(date_start: str, val_cutoff: str, val_year: int = 2024) -> str:
    """Classify a race into 'train' | 'val' | 'test' by its start date.

    train: any race in a season before val_year.
    val:   a val_year race strictly before val_cutoff.
    test:  a val_year race on or after val_cutoff.
    """
    dt = datetime.fromisoformat(date_start)
    cutoff = datetime.fromisoformat(val_cutoff).date()
    if dt.year < val_year:
        return "train"
    return "val" if dt.date() < cutoff else "test"
```

Then thread `val_year` through `build_snapshots` (add a `val_year: int = 2024` parameter and pass it to the `assign_split(...)` call). Remove the now-unused `_VAL_YEAR = 2024` module constant.

- [ ] **Step 4: Run tests to verify**

Run: `uv run pytest tests/test_snapshots_unit.py -q`
Expected: all pass (existing tests still pass because the default is 2024).

- [ ] **Step 5: Wire config through the CLI**

In `config/default.yaml` add:

```yaml
val_year: 2025
```

In `scripts/build_snapshots.py`, pass it:

```python
    meta = build_snapshots(
        features_dir=features_dir, raw_dir=raw_dir, out_dir=out_dir,
        feature_columns=FEATURE_COLUMNS,
        snapshot_laps=cfg["snapshot_laps"], val_cutoff=cfg["val_cutoff"],
        val_year=cfg["val_year"], git_sha=_git_sha(),
    )
```

- [ ] **Step 6: Commit**

```bash
git add src/f1_predictor/snapshots.py scripts/build_snapshots.py config/default.yaml tests/test_snapshots_unit.py
git commit -m "feat: configurable val_year for chronological split (enables 2025 hold-out)"
```

---

## Task 4: Pull the 2025 season `[data-run]`

Network pull executed directly by the lead (not a subagent). Honours rate limits; ~23 usable races after Monaco exclusion.

- [ ] **Step 1: Pull**

Run: `uv run python scripts/pull_season.py --year 2025`
Expected: "Pulled N sessions for 2025" (N ≈ 23). Takes several minutes (rate limiting). Sprints and Monaco are auto-excluded by `pull_season`.

- [ ] **Step 2: Sanity-check the pull**

Run: `uv run python -c "from pathlib import Path; import polars as pl; d=Path('data/raw'); print('sessions on disk:', sum(1 for p in d.iterdir() if p.is_dir() and (p/'meta.json').exists()))"`
Expected: count rises by ~23 vs before (was 45).
Spot-check one 2025 race has non-empty `laps.parquet`:
Run: `uv run python -c "import polars as pl, glob; f=sorted(glob.glob('data/raw/*/laps.parquet'))[-1]; print(f, pl.read_parquet(f).height)"`
Expected: a non-zero lap count.

- [ ] **Step 3: No commit** — `data/` is gitignored. Note the new session keys in the execution log.

---

## Task 5: Re-run Stages 2–3 and rebuild snapshots on 2023+2024+2025 `[data-run]`

- [ ] **Step 1: Sessionise + features for all races (incl. 2025)**

Run: `uv run python scripts/run_pipeline.py`
Expected: "Sessionised M races" / "Built features for M races" with M ≈ 67. Watch for new-season gotchas (new drivers, new circuits). Priors are computed across all sessionised races and use only prior races, so 2025 priors draw on 2023+2024+earlier-2025 automatically.

- [ ] **Step 2: Eyeball a 2025 features table for nulls / sanity**

Run: `uv run python -c "import polars as pl, glob; f=sorted(glob.glob('data/features/*.parquet'))[-1]; df=pl.read_parquet(f); print(df.select(['position','driver_circuit_finish_rate','distance_remaining_km']).describe())"`
Expected: `position` integer-like 1..20, `distance_remaining_km` positive, `driver_circuit_finish_rate` in [0,1] (new drivers may be null/default — acceptable).

- [ ] **Step 3: Rebuild snapshots with the 2025 hold-out split**

Run: `uv run python scripts/build_snapshots.py`
Expected: `train` ≈ 44 races (2023+2024), `val` + `test` = 2025 split on 2025-07-01. New `data_version=`.

- [ ] **Step 4: Re-measure naive + Option B `[data-run]`**

Run: `uv run python scripts/train_baseline.py --no-mlflow`
Run: `uv run python scripts/eval_naive.py --run-dir runs/<new-run-id>`
**Checkpoint B (the big one):** with ~2× training races and a fresh held-out season, does Option B's margin over naive grow? Record overall + per-lap. This is the strongest signal on whether data volume was the bottleneck.

- [ ] **Step 5: No code commit** (config already committed in Tasks 2–3; data is gitignored). Record Checkpoint B numbers in the execution log / a memory.

---

## Task 6: Backfill `max_speed_kmh` from per-driver car telemetry `[code]`

The heaviest, most-uncertain lever — **may be deferred** if Checkpoint B already closes the gap. Per-session car_data returns 422; fetch per `driver_number` and concatenate.

**Files:**
- Modify: `src/f1_predictor/ingest.py`
- Test: `tests/test_ingest.py`

- [ ] **Step 1: Write the failing test** (mock the HTTP layer; no network)

```python
def test_pull_car_data_concatenates_per_driver(tmp_path, monkeypatch):
    from f1_predictor import ingest

    calls = []

    class FakeResp:
        status_code = 200
        def json(self):  # one speed row per driver, keyed off the URL's driver_number
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
    assert df.height == 2  # one row per driver, concatenated
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ingest.py::test_pull_car_data_concatenates_per_driver -v`
Expected: FAIL with `AttributeError: module 'f1_predictor.ingest' has no attribute 'pull_car_data'`.

- [ ] **Step 3: Implement `pull_car_data` and wire it in**

Add to `src/f1_predictor/ingest.py`:

```python
def pull_car_data(session_key: int, driver_numbers: list[int], raw_dir: Path) -> None:
    """Pull car telemetry per driver and write a single car_data.parquet.

    A per-session car_data query returns 422 ("too much data"), so OpenF1 must
    be queried per driver_number. Each query is ~tens of thousands of rows; we
    keep only what _add_car_data needs downstream (driver_number, date, speed).
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

Then, in `pull_session`, after the main endpoint loop writes `drivers.parquet`, backfill car_data using the pulled driver list:

```python
    # car_data is pulled per driver (a per-session query 422s) — see module docstring.
    drivers_df = pl.read_parquet(session_dir / "drivers.parquet")
    driver_numbers = drivers_df["driver_number"].unique().to_list() if not drivers_df.is_empty() else []
    pull_car_data(session_key, driver_numbers, raw_dir)
    row_counts["car_data"] = pl.read_parquet(session_dir / "car_data.parquet").height
```

(Keep `car_data` out of `ENDPOINTS` — it has its own pull path. Sessionise already lists `car_data` among the raw endpoints it reads at `sessionise.py:21`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_ingest.py::test_pull_car_data_concatenates_per_driver -v`
Expected: PASS.

- [ ] **Step 5: Run the full ingest test file**

Run: `uv run pytest tests/test_ingest.py -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/f1_predictor/ingest.py tests/test_ingest.py
git commit -m "feat: per-driver car_data pull to backfill max_speed_kmh"
```

---

## Task 7: Re-pull car_data, re-run pipeline, final measurement `[data-run]`

- [ ] **Step 1: Force a re-pull so cached races get car_data**

Run: `uv run python scripts/pull_season.py --year 2023 --force`
Run: `uv run python scripts/pull_season.py --year 2024 --force`
Run: `uv run python scripts/pull_season.py --year 2025 --force`
Expected: each race now has a non-empty `car_data.parquet`. **This is the heaviest network step** (~20 drivers × ~67 races). Honours Retry-After; expect a long run. Spot-check:
Run: `uv run python -c "import polars as pl, glob; f=sorted(glob.glob('data/raw/*/car_data.parquet'))[-1]; df=pl.read_parquet(f); print(f, df.height, df['speed'].max())"`
Expected: large row count, a plausible top speed (~300–360 km/h).

- [ ] **Step 2: Re-run Stages 2–4**

Run: `uv run python scripts/run_pipeline.py`
Run: `uv run python scripts/build_snapshots.py`
Then verify `max_speed_kmh` is now populated:
Run: `uv run python -c "import polars as pl; df=pl.read_parquet('data/snapshots/train.parquet'); print('max_speed non-null frac:', df['max_speed_kmh'].is_not_null().mean())"`
Expected: well above 0 (it was 0 before). Note: snapshots store the column standardised; check the pre-scaling features table if you want raw km/h.

- [ ] **Step 3: Final measurement `[data-run]`**

Run: `uv run python scripts/train_baseline.py --no-mlflow`
Run: `uv run python scripts/eval_naive.py --run-dir runs/<new-run-id>`
**Checkpoint C:** does reviving `max_speed_kmh` move Option B vs naive? Record overall + per-lap.

- [ ] **Step 4: Record the decision**

Write a memory updating [[baseline-is-delta-regression-not-lambdarank]] with the three checkpoint results and the verdict: **is the gap over naive now large enough, and does the Transformer still look necessary?** This is the question that motivated the whole plan.

---

## Decision Checkpoint: Transformer or not?

After Checkpoint C, compare the best Option B test Spearman (and especially the **early-lap** numbers — lap 5/10/15, where naive is weakest and the project's lap-by-lap-confidence goal lives) against naive:

- **If Option B now clears naive by a healthy, consistent margin across laps** and early-lap accuracy is acceptable → the Transformer may be deferred; bank the GBM and move to the lap-by-lap visualiser (Build Order step 7).
- **If the margin is still thin, or early-lap ranking is poor** → the cross-driver joint-consistency the Transformer provides is the next lever; proceed to `docs/superpowers/plans/2026-06-08-transformer.md` (revise its baseline-comparison section to use Option B, not lambdarank).

This plan deliberately does **not** decide the Transformer's fate up front — it produces the measurements needed to decide it.

---

## Self-Review Notes

- **Spec coverage:** all three user-requested levers (denser snapshot laps → Task 2; more seasons → Tasks 3–5; `max_speed_kmh` backfill → Tasks 6–7) are covered, plus the prerequisite `val_year` fix and a reusable measurement harness.
- **Ordering rationale:** cheapest/most-informative first (no-network denser laps), then the big data lever (2025), then the heaviest/most-uncertain (telemetry backfill) last so it can be dropped if the gap already closes.
- **Attribution:** each lever ends in its own checkpoint against the same naive yardstick, so the Transformer decision rests on per-lever evidence, not a single blended number.
- **Possible split:** if Group C (Tasks 6–7) balloons or its telemetry volume proves impractical, it is cleanly separable into its own follow-up plan — Tasks 1–5 stand alone and produce a working, measured result.
