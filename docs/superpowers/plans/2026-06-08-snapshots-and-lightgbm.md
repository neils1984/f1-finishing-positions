# Snapshots + LightGBM Baseline (Stage 4 + Stage 5a) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert per-race feature tables into a chronologically split, scaled, snapshot training set and train a LightGBM LambdaRank baseline that produces an end-to-end Spearman number on the held-out test races.

**Architecture:** Stage 4 reads `data/features/{session_key}.parquet`, extracts fixed-lap snapshots (one row per active driver at each snapshot lap), assigns each race to train/val/test by date, fits a `StandardScaler` on the train split only, and writes `data/snapshots/{split}.parquet` plus `metadata.json`. Stage 5a trains LightGBM with `objective="lambdarank"`, one ranking group per `(race, snapshot_lap)`, relevance `21 - final_position`, and evaluates with Spearman / top-k / mean position error, saving everything under `runs/{run_id}/`.

**Tech Stack:** Python 3.11+, `uv`, Polars, scikit-learn (`StandardScaler`), LightGBM (`lambdarank`), MLflow (local), Typer, pytest.

**Spec:** `docs/superpowers/specs/2026-06-03-f1-predictor-design.md` (Stages 4 and 5).
**This is Plan 3 of 4.** Plans 1–2 (Pipeline Foundation, Feature Engineering) are complete. Plan 4 covers the Transformer and reuses this plan's snapshots + metadata.

---

## Real-data decisions (carried from Plans 1–2)

Recorded in project memory (`openf1-real-data-gotchas`). These shape the plan:

1. **2024 is not pulled yet.** Plans 1–2 only processed 2023. The chronological split (train 2023, val/test 2024) requires pulling and processing 2024 first — Task 1 does this. The Stage 1–3 code already handles every real-data quirk, so it Just Works for 2024 too.
2. **Nulls are pervasive.** Feature columns contain nulls: gaps for lapped drivers, pace deltas at field edges, and entire columns that are all-null for 2023 — `max_speed_kmh` (car_data deferred) and the circuit finish rates (a circuit appears once per season, so 2023 has no prior same-circuit race). Stage 4 **imputes every null to 0.0 before scaling**. `StandardScaler` sets `scale_=1` for zero-variance columns, so an all-null-in-train column scales to 0 in train and passes 2024's real values through unchanged — no NaNs, no crashes.
3. **Two known train/serve skews, accepted for v1 (documented, not hidden):** `max_speed_kmh` is 0 everywhere until car_data is backfilled; `driver_circuit_finish_rate` / `team_circuit_finish_rate` are 0 across all 2023 (train) but carry real values in 2024 (val/test). The columns are retained so the schema is stable and a later backfill lights them up. Flag both in error analysis.
4. **Snapshot membership = drivers active at the snapshot lap.** A snapshot at lap N contains exactly the drivers who have a feature row at `lap_number == N` (i.e. completed lap N). Drivers retired before N are excluded from that snapshot's ranking group — their fate is already decided. Each included row already carries `final_position`, so relevance is well-defined. (Padding to 20 drivers is a Plan 4 / Transformer concern, not needed for LightGBM's variable-size groups.)

---

## File Map

```
src/f1_predictor/snapshots.py              CREATE  Stage 4: split assignment, snapshot extraction, scaling, build_snapshots()
src/f1_predictor/models/__init__.py        CREATE  empty package marker
src/f1_predictor/models/baseline_gbm.py    CREATE  Stage 5a: LightGBM lambdarank train + predict
src/f1_predictor/evaluate.py               CREATE  ranking metrics (Spearman, top-k, mean position error)
scripts/run_pipeline.py                    CREATE  sessionise + features for all raw sessions (Stage 2+3 convenience)
scripts/build_snapshots.py                 CREATE  Typer CLI for Stage 4
scripts/train_baseline.py                  CREATE  Typer CLI: train + evaluate the LightGBM baseline
tests/test_snapshots_unit.py               CREATE  Stage 4 unit tests (synthetic)
tests/test_evaluate_unit.py                CREATE  metric unit tests (synthetic)
tests/test_baseline_gbm.py                 CREATE  LightGBM train/predict + integration tests
```

`data/snapshots/` and `runs/` are gitignored. The `models/` directory is new; `evaluate.py` and `losses.py` are shared with Plan 4 (this plan creates `evaluate.py`; Plan 4 creates `losses.py`).

---

## Snapshot schema (`data/snapshots/{split}.parquet`)

One row per `(session_key, snapshot_lap, driver_number)` for drivers active at that lap.

Keys/meta: `session_key`, `snapshot_lap`, `driver_number`, `final_position`, `relevance` (= `21 - final_position`), `split`.
Features: the 30 `FEATURE_COLUMNS` from Stage 3, imputed and scaled (Float64).

`metadata.json`:
```json
{
  "feature_columns": ["position", "..."],
  "scaler": {"mean": {"position": 0.0, "...": 0.0}, "scale": {"position": 1.0, "...": 1.0}},
  "snapshot_laps": [10, 20, 30, 40],
  "splits": {"train": [<session_keys>], "val": [...], "test": [...]},
  "data_version": "<sha256 of feature_columns + scaler + git SHA>"
}
```

---

## Task 1: Ingest & process the 2024 season + run_pipeline convenience

Pull 2024, then sessionise + featurise every raw session (2023 + 2024). A small `run_pipeline.py` makes Stage 2+3 a one-command rebuild.

**Files:**
- Create: `scripts/run_pipeline.py`
- Test: `tests/test_snapshots_unit.py` (created here; reused by later tasks)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_snapshots_unit.py
"""Unit tests for Stage 4 (snapshots) and the run_pipeline helper."""
import polars as pl
import pytest


def test_run_pipeline_lists_session_keys(tmp_path):
    # run_pipeline.discover_sessions returns the integer keys of raw sessions
    # that have a meta.json (fully pulled), ignoring partial dirs.
    from scripts.run_pipeline import discover_sessions

    (tmp_path / "9001").mkdir()
    (tmp_path / "9001" / "meta.json").write_text("{}")
    (tmp_path / "9002").mkdir()  # no meta.json -> skipped
    keys = discover_sessions(tmp_path)
    assert keys == [9001]
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_snapshots_unit.py::test_run_pipeline_lists_session_keys -v`
Expected: `ModuleNotFoundError: No module named 'scripts.run_pipeline'` (or import error).

- [ ] **Step 3: Create `scripts/run_pipeline.py`**

```python
"""Run Stage 2 (sessionise) + Stage 3 (features) for all pulled raw sessions."""
from pathlib import Path

import typer

from f1_predictor.sessionise import sessionise
from f1_predictor.features import build_features, load_circuits
from f1_predictor.priors import build_driver_races, compute_priors

app = typer.Typer(add_completion=False)


def discover_sessions(raw_dir: Path) -> list[int]:
    """Integer keys of fully-pulled raw sessions (those with a meta.json)."""
    return sorted(
        int(p.name)
        for p in raw_dir.iterdir()
        if p.is_dir() and (p / "meta.json").exists()
    )


@app.command()
def main(
    raw_dir: Path = typer.Option(Path("data/raw"), "--raw-dir"),
    sessions_dir: Path = typer.Option(Path("data/sessions"), "--sessions-dir"),
    features_dir: Path = typer.Option(Path("data/features"), "--features-dir"),
) -> None:
    keys = discover_sessions(raw_dir)
    if not keys:
        typer.echo("No pulled sessions found.", err=True)
        raise SystemExit(1)

    # Stage 2: sessionise each race (skips cancelled / empty-laps sessions).
    sessionised: list[int] = []
    for key in keys:
        df = sessionise(key, raw_dir, sessions_dir)
        if not df.is_empty():
            sessionised.append(key)
    typer.echo(f"Sessionised {len(sessionised)} races")

    # Stage 3: priors computed once across all sessionised races, then features.
    driver_races = build_driver_races(raw_dir, sessionised)
    priors = compute_priors(driver_races)
    circuits = load_circuits()
    for key in sessionised:
        build_features(key, sessions_dir, raw_dir, features_dir, priors, circuits)
    typer.echo(f"Built features for {len(sessionised)} races")


if __name__ == "__main__":
    app()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_snapshots_unit.py::test_run_pipeline_lists_session_keys -v`
Expected: PASS.

- [ ] **Step 5: Pull and process 2024 (network — run once, ~several minutes)**

```bash
uv run python scripts/pull_season.py --year 2024
uv run python scripts/run_pipeline.py
```

Expected: 2024 races pulled to `data/raw/`, then sessionised + featurised alongside 2023. `data/features/` now holds both seasons. (If a 2024 endpoint diverges further, the Stage 1–3 robustness handles 404/422/429/empty; investigate only if a session errors.)

- [ ] **Step 6: Commit**

```bash
git add scripts/run_pipeline.py tests/test_snapshots_unit.py
git commit -m "feat: run_pipeline (stage 2+3 for all sessions) + ingest 2024 season"
```

---

## Task 2: Split assignment by race date

Each race is train (2023), val (2024 before `val_cutoff`), or test (2024 on/after `val_cutoff`). Dates come from the raw `sessions` endpoint.

**Files:**
- Create: `src/f1_predictor/snapshots.py`
- Modify: `tests/test_snapshots_unit.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_snapshots_unit.py`:

```python
from f1_predictor.snapshots import assign_split


def test_assign_split_2023_is_train():
    assert assign_split("2023-03-05T15:00:00+00:00", "2024-07-01") == "train"


def test_assign_split_2024_before_cutoff_is_val():
    assert assign_split("2024-03-02T15:00:00+00:00", "2024-07-01") == "val"


def test_assign_split_2024_on_or_after_cutoff_is_test():
    assert assign_split("2024-07-07T13:00:00+00:00", "2024-07-01") == "test"
    assert assign_split("2024-07-01T00:00:00+00:00", "2024-07-01") == "test"


def test_assign_split_pre_2023_is_train():
    # Any race earlier than the val season counts as train.
    assert assign_split("2022-11-20T13:00:00+00:00", "2024-07-01") == "train"
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_snapshots_unit.py::test_assign_split_2023_is_train -v`
Expected: `ModuleNotFoundError: No module named 'f1_predictor.snapshots'`.

- [ ] **Step 3: Create `src/f1_predictor/snapshots.py`**

```python
"""Stage 4: build chronologically split, scaled snapshot training tensors.

Snapshots are extracted at fixed laps from the Stage 3 feature tables. The
StandardScaler is fitted on the train split only; nulls are imputed to 0.0
before scaling. Output: data/snapshots/{train,val,test}.parquet + metadata.json.
"""
from __future__ import annotations

from datetime import datetime

import polars as pl

# The validation season; anything earlier is train.
_VAL_YEAR = 2024


def assign_split(date_start: str, val_cutoff: str) -> str:
    """Classify a race into 'train' | 'val' | 'test' by its start date.

    train: any race before the validation season (2024).
    val:   a 2024 race strictly before val_cutoff.
    test:  a 2024 race on or after val_cutoff.
    """
    dt = datetime.fromisoformat(date_start)
    cutoff = datetime.fromisoformat(val_cutoff).date()
    if dt.year < _VAL_YEAR:
        return "train"
    return "val" if dt.date() < cutoff else "test"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_snapshots_unit.py -v`
Expected: all pass (split tests + the run_pipeline test).

- [ ] **Step 5: Commit**

```bash
git add src/f1_predictor/snapshots.py tests/test_snapshots_unit.py
git commit -m "feat: stage 4 — chronological split assignment by race date"
```

---

## Task 3: Snapshot extraction per race

For each snapshot lap, take the active drivers' feature rows and attach `relevance = 21 - final_position`.

**Files:**
- Modify: `src/f1_predictor/snapshots.py`
- Modify: `tests/test_snapshots_unit.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_snapshots_unit.py`:

```python
from f1_predictor.snapshots import extract_snapshots, RELEVANCE_BASE


def _mini_features() -> pl.DataFrame:
    # 2 drivers, laps 1..3. Driver 2 has no row at lap 3 (retired earlier).
    return pl.DataFrame({
        "session_key": [900, 900, 900, 900, 900],
        "driver_number": [1, 1, 1, 2, 2],
        "lap_number": [1, 2, 3, 1, 2],
        "final_position": [1, 1, 1, 2, 2],
        "position": [1, 1, 1, 2, 2],
        "gap_to_leader": [0.0, 0.0, 0.0, 1.0, 1.5],
    })


def test_extract_snapshots_picks_snapshot_laps_only():
    snaps = extract_snapshots(_mini_features(), snapshot_laps=[2, 3], feature_columns=["position", "gap_to_leader"])
    assert set(snaps["snapshot_lap"].unique().to_list()) == {2, 3}
    # Lap 2: both drivers active -> 2 rows. Lap 3: only driver 1 -> 1 row.
    assert snaps.filter(pl.col("snapshot_lap") == 2).height == 2
    assert snaps.filter(pl.col("snapshot_lap") == 3).height == 1


def test_extract_snapshots_relevance_is_21_minus_final_position():
    snaps = extract_snapshots(_mini_features(), snapshot_laps=[2], feature_columns=["position"])
    d1 = snaps.filter(pl.col("driver_number") == 1)
    d2 = snaps.filter(pl.col("driver_number") == 2)
    assert d1["relevance"][0] == RELEVANCE_BASE - 1   # final_position 1
    assert d2["relevance"][0] == RELEVANCE_BASE - 2   # final_position 2


def test_extract_snapshots_carries_keys_and_features():
    snaps = extract_snapshots(_mini_features(), snapshot_laps=[2], feature_columns=["position", "gap_to_leader"])
    for c in ["session_key", "snapshot_lap", "driver_number", "final_position", "relevance", "position", "gap_to_leader"]:
        assert c in snaps.columns
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_snapshots_unit.py::test_extract_snapshots_picks_snapshot_laps_only -v`
Expected: `ImportError: cannot import name 'extract_snapshots'`.

- [ ] **Step 3: Implement `extract_snapshots` in `snapshots.py`**

```python
RELEVANCE_BASE = 21  # relevance = RELEVANCE_BASE - final_position (higher = better)

_META_COLUMNS = ["session_key", "snapshot_lap", "driver_number", "final_position", "relevance"]


def extract_snapshots(
    features: pl.DataFrame,
    snapshot_laps: list[int],
    feature_columns: list[str],
) -> pl.DataFrame:
    """One row per (snapshot_lap, active driver) with relevance + feature columns.

    A driver is "active" at a snapshot lap if it has a feature row at that exact
    lap_number. relevance = RELEVANCE_BASE - final_position.
    """
    snaps = (
        features.filter(pl.col("lap_number").is_in(snapshot_laps))
        .with_columns([
            pl.col("lap_number").alias("snapshot_lap"),
            (RELEVANCE_BASE - pl.col("final_position")).alias("relevance"),
        ])
        .select(_META_COLUMNS + feature_columns)
    )
    return snaps
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_snapshots_unit.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/f1_predictor/snapshots.py tests/test_snapshots_unit.py
git commit -m "feat: stage 4 — snapshot extraction with relevance target"
```

---

## Task 4: Null imputation + train-only StandardScaler

Impute nulls to 0.0, fit `StandardScaler` on the **train** rows only, apply to all splits, and round-trip the scaler params through plain dicts (for `metadata.json`).

**Files:**
- Modify: `src/f1_predictor/snapshots.py`
- Modify: `tests/test_snapshots_unit.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_snapshots_unit.py`:

```python
import numpy as np
from f1_predictor.snapshots import fit_scaler, apply_scaler


def test_fit_scaler_uses_train_only_and_imputes_nulls():
    train = pl.DataFrame({"a": [0.0, 2.0, 4.0], "b": [None, None, None]})
    params = fit_scaler(train, feature_columns=["a", "b"])
    # mean(a)=2, std(a)=sqrt(8/3); b is all-null -> imputed 0 -> mean 0, scale 1.
    assert params["mean"]["a"] == pytest.approx(2.0)
    assert params["scale"]["a"] == pytest.approx(np.std([0.0, 2.0, 4.0]))
    assert params["mean"]["b"] == pytest.approx(0.0)
    assert params["scale"]["b"] == pytest.approx(1.0)  # zero-variance -> scale 1


def test_apply_scaler_standardises_and_passes_through_constant():
    train = pl.DataFrame({"a": [0.0, 2.0, 4.0], "b": [None, None, None]})
    params = fit_scaler(train, ["a", "b"])
    out = apply_scaler(pl.DataFrame({"a": [2.0], "b": [5.0]}), params, ["a", "b"])
    assert out["a"][0] == pytest.approx(0.0)       # (2-2)/std = 0
    # b had scale 1, mean 0 -> passes 5.0 through unchanged (the 2024 skew case)
    assert out["b"][0] == pytest.approx(5.0)


def test_apply_scaler_imputes_nulls_before_scaling():
    train = pl.DataFrame({"a": [0.0, 2.0, 4.0]})
    params = fit_scaler(train, ["a"])
    out = apply_scaler(pl.DataFrame({"a": [None]}), params, ["a"])
    # null -> 0 -> (0-2)/std
    assert out["a"][0] == pytest.approx((0.0 - 2.0) / np.std([0.0, 2.0, 4.0]))
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_snapshots_unit.py::test_fit_scaler_uses_train_only_and_imputes_nulls -v`
Expected: `ImportError: cannot import name 'fit_scaler'`.

- [ ] **Step 3: Implement scaler functions in `snapshots.py`**

```python
import numpy as np
from sklearn.preprocessing import StandardScaler


def _impute(df: pl.DataFrame, feature_columns: list[str]) -> pl.DataFrame:
    """Bool->Int, then fill nulls with 0.0 and cast features to Float64."""
    return df.with_columns([
        pl.col(c).cast(pl.Float64, strict=False).fill_null(0.0).alias(c)
        for c in feature_columns
    ])


def fit_scaler(train: pl.DataFrame, feature_columns: list[str]) -> dict:
    """Fit a StandardScaler on imputed train features; return params as dicts.

    Zero-variance columns get scale 1.0 (sklearn behaviour), so all-null-in-train
    features map to 0 in train and pass real values through unchanged elsewhere.
    """
    x = _impute(train, feature_columns).select(feature_columns).to_numpy()
    scaler = StandardScaler().fit(x)
    scale = np.where(scaler.scale_ == 0.0, 1.0, scaler.scale_)
    return {
        "mean": {c: float(m) for c, m in zip(feature_columns, scaler.mean_)},
        "scale": {c: float(s) for c, s in zip(feature_columns, scale)},
    }


def apply_scaler(df: pl.DataFrame, params: dict, feature_columns: list[str]) -> pl.DataFrame:
    """Impute nulls to 0.0 then standardise each feature with the fitted params."""
    df = _impute(df, feature_columns)
    return df.with_columns([
        ((pl.col(c) - params["mean"][c]) / params["scale"][c]).alias(c)
        for c in feature_columns
    ])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_snapshots_unit.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/f1_predictor/snapshots.py tests/test_snapshots_unit.py
git commit -m "feat: stage 4 — null imputation + train-only StandardScaler"
```

---

## Task 5: Assemble snapshots — `build_snapshots()` + CLI

Wire split assignment, extraction, and scaling into the public builder. Write `data/snapshots/{split}.parquet` and `metadata.json`.

**Files:**
- Modify: `src/f1_predictor/snapshots.py`
- Create: `scripts/build_snapshots.py`
- Modify: `tests/test_snapshots_unit.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_snapshots_unit.py`:

```python
import json
from f1_predictor.snapshots import build_snapshots


def _write_feature_file(features_dir, raw_dir, key, date, n_laps=4):
    features_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / str(key)).mkdir(parents=True, exist_ok=True)
    # minimal sessions.parquet for the date
    pl.DataFrame({"date_start": [date], "circuit_short_name": ["X"]}).write_parquet(
        raw_dir / str(key) / "sessions.parquet"
    )
    rows = []
    for d in (1, 2):
        for lap in range(1, n_laps + 1):
            rows.append({"session_key": key, "driver_number": d, "lap_number": lap,
                         "final_position": d, "position": d, "gap_to_leader": float(d)})
    pl.DataFrame(rows).write_parquet(features_dir / f"{key}.parquet")


def test_build_snapshots_writes_splits_and_metadata(tmp_path):
    features_dir = tmp_path / "features"
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "snapshots"
    _write_feature_file(features_dir, raw_dir, 700, "2023-05-01T13:00:00+00:00")  # train
    _write_feature_file(features_dir, raw_dir, 800, "2024-03-01T13:00:00+00:00")  # val
    _write_feature_file(features_dir, raw_dir, 900, "2024-09-01T13:00:00+00:00")  # test

    build_snapshots(
        features_dir=features_dir, raw_dir=raw_dir, out_dir=out_dir,
        feature_columns=["position", "gap_to_leader"],
        snapshot_laps=[2, 4], val_cutoff="2024-07-01", git_sha="deadbeef",
    )

    for split in ("train", "val", "test"):
        assert (out_dir / f"{split}.parquet").exists()
    meta = json.loads((out_dir / "metadata.json").read_text())
    assert meta["feature_columns"] == ["position", "gap_to_leader"]
    assert meta["splits"]["train"] == [700]
    assert meta["splits"]["test"] == [900]
    assert "data_version" in meta
    # Train 'position' is standardised -> mean ~0
    train = pl.read_parquet(out_dir / "train.parquet")
    assert abs(train["position"].mean()) < 1e-9
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_snapshots_unit.py::test_build_snapshots_writes_splits_and_metadata -v`
Expected: `ImportError: cannot import name 'build_snapshots'`.

- [ ] **Step 3: Implement `build_snapshots` in `snapshots.py`**

```python
import hashlib
import json
from pathlib import Path


def _race_date(raw_dir: Path, session_key: int) -> str:
    ses = pl.read_parquet(raw_dir / str(session_key) / "sessions.parquet").row(0, named=True)
    return ses["date_start"]


def _data_version(feature_columns: list[str], scaler: dict, git_sha: str) -> str:
    payload = json.dumps({"f": feature_columns, "s": scaler, "g": git_sha}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def build_snapshots(
    features_dir: Path,
    raw_dir: Path,
    out_dir: Path,
    feature_columns: list[str],
    snapshot_laps: list[int],
    val_cutoff: str,
    git_sha: str = "unknown",
) -> dict:
    """Build train/val/test snapshot parquets + metadata.json. Returns metadata.

    Scaler is fit on the train split only and applied to all splits.
    """
    keys = sorted(int(p.stem) for p in features_dir.glob("*.parquet"))

    # Group races by split.
    split_keys: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    raw_by_split: dict[str, list[pl.DataFrame]] = {"train": [], "val": [], "test": []}
    for key in keys:
        split = assign_split(_race_date(raw_dir, key), val_cutoff)
        feats = pl.read_parquet(features_dir / f"{key}.parquet")
        snaps = extract_snapshots(feats, snapshot_laps, feature_columns)
        snaps = snaps.with_columns(pl.lit(split).alias("split"))
        split_keys[split].append(key)
        raw_by_split[split].append(snaps)

    train_df = pl.concat(raw_by_split["train"], how="vertical") if raw_by_split["train"] else pl.DataFrame()
    if train_df.is_empty():
        raise ValueError("No train races found — cannot fit scaler.")

    scaler = fit_scaler(train_df, feature_columns)

    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        frames = raw_by_split[split]
        if not frames:
            pl.DataFrame().write_parquet(out_dir / f"{split}.parquet")
            continue
        df = pl.concat(frames, how="vertical")
        scaled = apply_scaler(df, scaler, feature_columns)
        scaled.write_parquet(out_dir / f"{split}.parquet")

    metadata = {
        "feature_columns": feature_columns,
        "scaler": scaler,
        "snapshot_laps": snapshot_laps,
        "splits": split_keys,
        "data_version": _data_version(feature_columns, scaler, git_sha),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata
```

- [ ] **Step 4: Create `scripts/build_snapshots.py`**

```python
"""CLI: build Stage 4 snapshots (train/val/test) from Stage 3 features."""
import subprocess
from pathlib import Path

import typer
import yaml

from f1_predictor.features import FEATURE_COLUMNS
from f1_predictor.snapshots import build_snapshots

app = typer.Typer(add_completion=False)


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


@app.command()
def main(
    features_dir: Path = typer.Option(Path("data/features"), "--features-dir"),
    raw_dir: Path = typer.Option(Path("data/raw"), "--raw-dir"),
    out_dir: Path = typer.Option(Path("data/snapshots"), "--out-dir"),
    config: Path = typer.Option(Path("config/default.yaml"), "--config"),
) -> None:
    cfg = yaml.safe_load(open(config))
    meta = build_snapshots(
        features_dir=features_dir, raw_dir=raw_dir, out_dir=out_dir,
        feature_columns=FEATURE_COLUMNS,
        snapshot_laps=cfg["snapshot_laps"], val_cutoff=cfg["val_cutoff"],
        git_sha=_git_sha(),
    )
    for split, ks in meta["splits"].items():
        typer.echo(f"  {split}: {len(ks)} races")
    typer.echo(f"data_version={meta['data_version']}")


if __name__ == "__main__":
    app()
```

- [ ] **Step 5: Run unit tests + the CLI on real data**

```bash
uv run pytest tests/test_snapshots_unit.py -v          # all pass
uv run python scripts/build_snapshots.py                # writes data/snapshots/*
```

Expected: snapshots written; train/val/test race counts printed (train = all 2023, val/test = 2024 split at 2024-07-01).

- [ ] **Step 6: Commit**

```bash
git add src/f1_predictor/snapshots.py scripts/build_snapshots.py tests/test_snapshots_unit.py
git commit -m "feat: stage 4 — build_snapshots assembly + CLI"
```

---

## Task 6: Evaluation metrics

Spearman (primary), top-3 accuracy, top-1 accuracy, mean position error — computed per `(session_key, snapshot_lap)` group and averaged.

**Files:**
- Create: `src/f1_predictor/evaluate.py`
- Create: `tests/test_evaluate_unit.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_evaluate_unit.py
import polars as pl
import pytest
from f1_predictor.evaluate import ranking_metrics


def _preds(rows):
    # rows: (session_key, snapshot_lap, driver, final_position, score)
    return pl.DataFrame(
        rows,
        schema=["session_key", "snapshot_lap", "driver_number", "final_position", "score"],
        orient="row",
    )


def test_perfect_ranking_scores_one():
    # Higher score should mean better (lower) final_position. Perfect alignment.
    df = _preds([
        (1, 30, 44, 1, 0.9),
        (1, 30, 11, 2, 0.5),
        (1, 30, 16, 3, 0.1),
    ])
    m = ranking_metrics(df)
    assert m["spearman"] == pytest.approx(1.0)
    assert m["top1_accuracy"] == pytest.approx(1.0)
    assert m["top3_accuracy"] == pytest.approx(1.0)
    assert m["mean_position_error"] == pytest.approx(0.0)


def test_reversed_ranking_is_negative_spearman():
    df = _preds([
        (1, 30, 44, 1, 0.1),
        (1, 30, 11, 2, 0.5),
        (1, 30, 16, 3, 0.9),
    ])
    m = ranking_metrics(df)
    assert m["spearman"] == pytest.approx(-1.0)
    assert m["top1_accuracy"] == pytest.approx(0.0)


def test_metrics_average_over_groups():
    # Two groups: one perfect, one reversed -> mean spearman 0.
    df = _preds([
        (1, 30, 1, 1, 0.9), (1, 30, 2, 2, 0.1),
        (2, 30, 3, 1, 0.1), (2, 30, 4, 2, 0.9),
    ])
    m = ranking_metrics(df)
    assert m["spearman"] == pytest.approx(0.0)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_evaluate_unit.py -v`
Expected: `ModuleNotFoundError: No module named 'f1_predictor.evaluate'`.

- [ ] **Step 3: Create `src/f1_predictor/evaluate.py`**

```python
"""Ranking evaluation metrics for snapshot predictions.

A higher predicted score means a better (lower) final_position. Metrics are
computed per (session_key, snapshot_lap) group and averaged across groups.
"""
from __future__ import annotations

import numpy as np
import polars as pl
from scipy.stats import spearmanr


def _predicted_order(scores: np.ndarray) -> np.ndarray:
    """Rank index by descending score (rank 1 = highest score)."""
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks


def ranking_metrics(predictions: pl.DataFrame) -> dict:
    """Average Spearman, top-1, top-3 accuracy, and mean position error.

    predictions must have: session_key, snapshot_lap, final_position, score.
    """
    spearmans, top1s, top3s, mpes = [], [], [], []

    for _, grp in predictions.group_by(["session_key", "snapshot_lap"], maintain_order=True):
        final_pos = grp["final_position"].to_numpy()
        score = grp["score"].to_numpy()
        if len(final_pos) < 2:
            continue

        # Spearman between score and -final_position (so positive = aligned).
        rho, _ = spearmanr(score, -final_pos)
        spearmans.append(0.0 if np.isnan(rho) else rho)

        pred_rank = _predicted_order(score)            # 1 = top predicted
        true_rank = final_pos.argsort().argsort() + 1  # 1 = actual winner

        # top-1: predicted winner is the actual winner.
        top1s.append(float(final_pos[np.argmax(score)] == final_pos.min()))
        # top-3: predicted top-3 set == actual top-3 set (as a hit rate over min(3,n)).
        k = min(3, len(final_pos))
        pred_top = set(np.argsort(-score)[:k])
        true_top = set(np.argsort(final_pos)[:k])
        top3s.append(len(pred_top & true_top) / k)
        # mean position error: |predicted rank - true rank| averaged.
        mpes.append(float(np.mean(np.abs(pred_rank - true_rank))))

    return {
        "spearman": float(np.mean(spearmans)) if spearmans else 0.0,
        "top1_accuracy": float(np.mean(top1s)) if top1s else 0.0,
        "top3_accuracy": float(np.mean(top3s)) if top3s else 0.0,
        "mean_position_error": float(np.mean(mpes)) if mpes else 0.0,
        "n_groups": len(spearmans),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_evaluate_unit.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/f1_predictor/evaluate.py tests/test_evaluate_unit.py
git commit -m "feat: ranking evaluation metrics (spearman, top-k, mpe)"
```

---

## Task 7: LightGBM LambdaRank baseline

Train LightGBM with `objective="lambdarank"`, one group per `(session_key, snapshot_lap)`, label `relevance`. Save model + config under `runs/{run_id}/` and log to MLflow.

**Files:**
- Create: `src/f1_predictor/models/__init__.py`
- Create: `src/f1_predictor/models/baseline_gbm.py`
- Create: `tests/test_baseline_gbm.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_baseline_gbm.py
import polars as pl
import numpy as np
from f1_predictor.models.baseline_gbm import group_sizes, train_baseline, predict


def _toy_snapshots(n_races=6):
    # Each race has a clean signal: lower 'position' -> better relevance.
    rows = []
    for r in range(n_races):
        for d in range(1, 5):
            rows.append({
                "session_key": r, "snapshot_lap": 30, "driver_number": d,
                "final_position": d, "relevance": 21 - d, "position": float(d),
                "gap_to_leader": float(d - 1),
            })
    return pl.DataFrame(rows)


def test_group_sizes_counts_rows_per_group():
    df = _toy_snapshots(2)
    sizes = group_sizes(df)
    assert sizes == [4, 4]   # 4 drivers per (race, lap) group


def test_train_and_predict_learns_obvious_signal():
    df = _toy_snapshots(8)
    model = train_baseline(df, feature_columns=["position", "gap_to_leader"],
                           params={"num_leaves": 7, "min_data_in_leaf": 1, "n_estimators": 20})
    scores = predict(model, df, ["position", "gap_to_leader"])
    # Within a group, the lowest 'position' (driver 1) should get the highest score.
    df = df.with_columns(pl.Series("score", scores))
    g0 = df.filter(pl.col("session_key") == 0).sort("score", descending=True)
    assert g0["driver_number"][0] == 1
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_baseline_gbm.py -v`
Expected: `ModuleNotFoundError: No module named 'f1_predictor.models.baseline_gbm'`.

- [ ] **Step 3: Create the package marker and model**

`src/f1_predictor/models/__init__.py`: empty file.

`src/f1_predictor/models/baseline_gbm.py`:

```python
"""LightGBM LambdaRank baseline for snapshot ranking.

One ranking group per (session_key, snapshot_lap); label = relevance
(21 - final_position). Higher predicted score = better (lower) final position.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import polars as pl

_DEFAULT_PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "min_data_in_leaf": 20,
    "n_estimators": 300,
    "verbose": -1,
}


def group_sizes(df: pl.DataFrame) -> list[int]:
    """Row counts per (session_key, snapshot_lap) group, in row order.

    The DataFrame MUST already be sorted by (session_key, snapshot_lap) so the
    returned sizes line up with LightGBM's contiguous-group expectation.
    """
    return (
        df.group_by(["session_key", "snapshot_lap"], maintain_order=True)
        .len()["len"]
        .to_list()
    )


def _sorted(df: pl.DataFrame) -> pl.DataFrame:
    return df.sort(["session_key", "snapshot_lap"])


def train_baseline(
    train: pl.DataFrame,
    feature_columns: list[str],
    params: dict | None = None,
    valid: pl.DataFrame | None = None,
) -> lgb.Booster:
    """Train a LambdaRank booster. Returns the fitted Booster."""
    train = _sorted(train)
    p = {**_DEFAULT_PARAMS, **(params or {})}
    n_estimators = p.pop("n_estimators")

    dtrain = lgb.Dataset(
        train.select(feature_columns).to_numpy(),
        label=train["relevance"].to_numpy(),
        group=group_sizes(train),
        feature_name=list(feature_columns),
    )
    valid_sets = [dtrain]
    if valid is not None and not valid.is_empty():
        valid = _sorted(valid)
        dvalid = lgb.Dataset(
            valid.select(feature_columns).to_numpy(),
            label=valid["relevance"].to_numpy(),
            group=group_sizes(valid),
            reference=dtrain,
        )
        valid_sets.append(dvalid)

    return lgb.train(p, dtrain, num_boost_round=n_estimators, valid_sets=valid_sets)


def predict(model: lgb.Booster, df: pl.DataFrame, feature_columns: list[str]) -> np.ndarray:
    """Predicted ranking scores aligned to df's current row order."""
    return model.predict(df.select(feature_columns).to_numpy())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_baseline_gbm.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/f1_predictor/models/__init__.py src/f1_predictor/models/baseline_gbm.py tests/test_baseline_gbm.py
git commit -m "feat: stage 5a — LightGBM lambdarank baseline (train + predict)"
```

---

## Task 8: End-to-end baseline run — CLI, run dir, MLflow, Spearman number

Train on real train snapshots, evaluate on test, save `runs/{run_id}/` artefacts, log to MLflow, and assert a sane Spearman with an integration test.

**Files:**
- Create: `scripts/train_baseline.py`
- Modify: `tests/test_baseline_gbm.py`

- [ ] **Step 1: Write the failing integration test**

Add to `tests/test_baseline_gbm.py`:

```python
import json
from pathlib import Path
import pytest
from f1_predictor.models.baseline_gbm import run_baseline


def test_run_baseline_end_to_end(tmp_path):
    # Synthetic snapshots with a learnable signal; verify the run dir + metrics.
    def write(split, n_races, start):
        rows = []
        for r in range(start, start + n_races):
            for d in range(1, 6):
                rows.append({"session_key": r, "snapshot_lap": 30, "driver_number": d,
                             "final_position": d, "relevance": 21 - d,
                             "position": float(d), "gap_to_leader": float(d - 1)})
        pl.DataFrame(rows).write_parquet(snap_dir / f"{split}.parquet")

    snap_dir = tmp_path / "snapshots"; snap_dir.mkdir()
    write("train", 20, 0); write("val", 4, 100); write("test", 4, 200)
    (snap_dir / "metadata.json").write_text(json.dumps({
        "feature_columns": ["position", "gap_to_leader"], "data_version": "test"
    }))

    runs_dir = tmp_path / "runs"
    result = run_baseline(snap_dir, runs_dir, params={"num_leaves": 7, "min_data_in_leaf": 1, "n_estimators": 30}, use_mlflow=False)

    run_dir = Path(result["run_dir"])
    assert (run_dir / "model.lgb").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "predictions_test.parquet").exists()
    # The signal is perfectly learnable -> strong positive test Spearman.
    assert result["metrics"]["spearman"] > 0.9
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_baseline_gbm.py::test_run_baseline_end_to_end -v`
Expected: `ImportError: cannot import name 'run_baseline'`.

- [ ] **Step 3: Add `run_baseline` to `baseline_gbm.py`**

```python
import json
import time
from pathlib import Path

from f1_predictor.evaluate import ranking_metrics


def run_baseline(
    snapshots_dir: Path,
    runs_dir: Path,
    params: dict | None = None,
    use_mlflow: bool = True,
) -> dict:
    """Train on train.parquet, evaluate on test.parquet, persist a run directory.

    Returns {"run_dir": str, "metrics": {...}}.
    """
    meta = json.loads((snapshots_dir / "metadata.json").read_text())
    feature_columns = meta["feature_columns"]

    train = pl.read_parquet(snapshots_dir / "train.parquet")
    test = pl.read_parquet(snapshots_dir / "test.parquet")
    val_path = snapshots_dir / "val.parquet"
    valid = pl.read_parquet(val_path) if val_path.exists() and pl.read_parquet(val_path).height else None

    model = train_baseline(train, feature_columns, params=params, valid=valid)

    scores = predict(model, test, feature_columns)
    preds = test.select(["session_key", "snapshot_lap", "driver_number", "final_position"]).with_columns(
        pl.Series("score", scores)
    )
    metrics = ranking_metrics(preds)

    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(run_dir / "model.lgb"))
    preds.write_parquet(run_dir / "predictions_test.parquet")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (run_dir / "config.yaml").write_text(json.dumps({
        "model": "lightgbm_lambdarank",
        "params": {**_DEFAULT_PARAMS, **(params or {})},
        "feature_columns": feature_columns,
        "data_version": meta.get("data_version"),
    }, indent=2))

    if use_mlflow:
        import mlflow
        mlflow.set_experiment("f1-baseline")
        with mlflow.start_run(run_name=run_id):
            mlflow.log_params({"model": "lightgbm_lambdarank", **(params or {})})
            mlflow.log_metrics(metrics)
            mlflow.log_artifacts(str(run_dir))

    return {"run_dir": str(run_dir), "metrics": metrics}
```

- [ ] **Step 4: Create `scripts/train_baseline.py`**

```python
"""CLI: train + evaluate the LightGBM baseline on the snapshot splits."""
from pathlib import Path

import typer

from f1_predictor.models.baseline_gbm import run_baseline

app = typer.Typer(add_completion=False)


@app.command()
def main(
    snapshots_dir: Path = typer.Option(Path("data/snapshots"), "--snapshots-dir"),
    runs_dir: Path = typer.Option(Path("runs"), "--runs-dir"),
    no_mlflow: bool = typer.Option(False, "--no-mlflow", help="Skip MLflow logging"),
) -> None:
    result = run_baseline(snapshots_dir, runs_dir, use_mlflow=not no_mlflow)
    typer.echo(f"Run: {result['run_dir']}")
    for k, v in result["metrics"].items():
        typer.echo(f"  {k}: {v}")


if __name__ == "__main__":
    app()
```

- [ ] **Step 5: Run the unit/integration tests, then the real baseline**

```bash
uv run pytest tests/test_baseline_gbm.py -v          # all pass
uv run python scripts/build_snapshots.py             # ensure snapshots exist
uv run python scripts/train_baseline.py --no-mlflow  # real end-to-end run
```

Expected: prints a run dir and metrics. Record the **test Spearman** — the spec's target is ~0.6–0.75 from lap-30 snapshots; this is the number everything else must beat. (A much lower number is a real finding — inspect `predictions_test.parquet`; do not silently accept it.)

- [ ] **Step 6: Run the full suite**

Run: `uv run pytest tests/ -q`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add src/f1_predictor/models/baseline_gbm.py scripts/train_baseline.py tests/test_baseline_gbm.py
git commit -m "feat: stage 5a — end-to-end LightGBM baseline run + MLflow + run dir"
```

---

## Self-Review

### 1. Spec coverage

| Spec requirement (Stages 4–5a) | Task |
|---|---|
| Tensor/snapshot per snapshot lap | Task 3 |
| Target relevance = 21 − final_position | Task 3 |
| Snapshots at fixed laps [10,20,30,40] | config + Task 5 |
| Chronological split (2023 / H1-2024 / H2-2024) | Task 2 + 5 |
| StandardScaler fit on train only, params saved | Task 4 + 5 |
| metadata.json (feature names, scaler params, data version) | Task 5 |
| LightGBM, objective lambdarank, group per (race, lap) | Task 7 |
| One row per (race, snapshot_lap, driver) | Task 3 |
| runs/{run_id}/ with config, model, metrics, predictions_test | Task 8 |
| MLflow tracking | Task 8 |
| Spearman (primary), top-3, top-1, mean position error | Task 6 |
| Baseline Spearman number established | Task 8 |
| 2024 data available for val/test | Task 1 |

Padding to 20 drivers and the retirement/target masks are deferred to Plan 4 (the Transformer's tensor loader); LightGBM uses variable-size groups, so they are not needed here. The lap-by-lap prediction evolution plot is a Plan 4 deliverable (it compares models).

### 2. Placeholder scan

No "TBD"/"add error handling"/"similar to Task N" placeholders. All code is complete. Circuit/feature lists come from `FEATURE_COLUMNS` (Stage 3) — no re-listing.

### 3. Type consistency

- `assign_split(date_start: str, val_cutoff: str) -> str` — Task 2, used in Task 5.
- `extract_snapshots(features, snapshot_laps, feature_columns) -> pl.DataFrame` — Task 3, used in Task 5.
- `fit_scaler(train, feature_columns) -> dict` / `apply_scaler(df, params, feature_columns)` — Task 4, used in Task 5.
- `build_snapshots(...)` writes the schema Task 7/8 read; `relevance`, `session_key`, `snapshot_lap`, `final_position` names are consistent across snapshots, evaluate, and baseline.
- `group_sizes`, `train_baseline`, `predict`, `run_baseline` — Task 7/8, used by the CLI.
- `ranking_metrics(predictions) -> dict` keys (`spearman`, `top1_accuracy`, `top3_accuracy`, `mean_position_error`) — Task 6, asserted in Task 8.

### Open items for the executor to confirm

- **Scaling booleans:** the four bool features (`sc_active`, `vsc_active`, `red_flag_active`, `is_street_circuit`) are cast to float and standardised along with the rest. Harmless, but if you prefer to leave 0/1 flags unscaled, exclude them from `feature_columns` passed to the scaler (and document).
- **`max_speed_kmh` and circuit finish rates** are 0 in 2023 (train). Confirm the baseline Spearman is reported per snapshot lap during error analysis so the train/serve skew on those columns is visible.
- **LightGBM params** in `_DEFAULT_PARAMS` are a reasonable start; tune `num_leaves`, `learning_rate`, `min_data_in_leaf`, and `n_estimators` against the val split once the end-to-end number exists.
