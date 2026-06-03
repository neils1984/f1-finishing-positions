# F1 Race Position Predictor — Design Spec

**Date:** 2026-06-03
**Status:** Approved

---

## Problem Framing

Given the state of an F1 race at lap N, predict the final finishing order of all drivers. This is a **ranking problem**, not a regression problem — drivers are interdependent, and the cost of error is non-uniform (a wrong podium call matters more than a wrong P14 call).

A core feature is tracking **prediction confidence over the course of a race**: how does model certainty resolve as the race progresses?

---

## Architecture

Five-stage pipeline. Each stage is independently runnable and produces cached Parquet artefacts. Re-pulling from OpenF1 is rare; re-running a stage is cheap and idempotent.

```
OpenF1 API
    ↓
1. Ingestion       → data/raw/{session_key}/{endpoint}.parquet
    ↓
2. Sessionising    → data/sessions/{session_key}.parquet   (driver-lap rows)
    ↓
3. Features        → data/features/{session_key}.parquet   (engineered)
    ↓
4. Snapshots       → data/snapshots/{split}.parquet        (training-ready tensors)
    ↓
5. Train & Eval    → runs/{run_id}/                        (checkpoints, metrics)
```

**Build order:** one race end-to-end before scaling. LightGBM baseline before any neural work.

---

## Tooling

| Concern | Choice |
|---|---|
| Env management | `uv` + `pyproject.toml` |
| Data transforms (per-race) | Polars |
| Data transforms (cross-race joins) | DuckDB |
| Neural models | PyTorch |
| Baseline model | LightGBM |
| Experiment tracking | MLflow (local) |
| CLI | Typer |
| Orchestration | None in v1 — plain Python scripts |

---

## Data & Ingestion (Stage 1)

**Source:** OpenF1 API — free, no auth.

**Scope:** 2023 and 2024 seasons (~44 races). Monaco excluded entirely. Baku, Singapore, Las Vegas, and Miami included but tagged with `is_street_circuit = True`.

**Endpoints pulled per race:**
`sessions`, `drivers`, `laps`, `position`, `intervals`, `stints`, `pit`, `race_control`, `session_result`, `car_data`

`car_data` is pulled for `max_speed_kmh` extraction only — aggregated to one value per driver-lap in Stage 2, raw telemetry retained on disk but not carried forward.

**Storage layout:**
```
data/raw/
  └── {session_key}/
      ├── meta.json              # session info, pull timestamp
      ├── drivers.parquet
      ├── laps.parquet
      ├── position.parquet
      ├── intervals.parquet
      ├── stints.parquet
      ├── pit.parquet
      ├── race_control.parquet
      ├── session_result.parquet
      └── car_data.parquet
```

**Ingestion behaviour:**
- `pull_session(session_key, force=False)` — skips if already on disk (idempotent)
- `pull_season(year)` — discovers all race session keys then calls `pull_session` for each
- 0.2s sleep between requests; `requests.Session` for connection reuse

---

## Sessionising (Stage 2)

Collapses per-endpoint raw dumps into a canonical race tape: **one row per `(driver_number, lap_number)`**.

**Output schema:**

| Column | Type | Notes |
|---|---|---|
| `session_key` | int | |
| `driver_number` | int | |
| `lap_number` | int | |
| `position` | int | Raw 1–20 |
| `gap_to_leader` | float | Seconds |
| `interval_to_ahead` | float | Seconds to car immediately ahead |
| `tyre_compound` | str | SOFT/MEDIUM/HARD/INTER/WET |
| `tyre_age_laps` | int | |
| `stint_number` | int | |
| `pit_this_lap` | bool | |
| `stops_completed` | int | Total stops by this lap |
| `lap_time` | float | Seconds |
| `max_speed_kmh` | float | `max(speed)` from `car_data`, aggregated here |
| `sc_active` | bool | |
| `vsc_active` | bool | |
| `red_flag_active` | bool | |
| `laps_since_sc_end` | int | |
| `is_retired` | bool | |
| `retirement_lap` | int | Null if not retired |
| `final_position` | int | Official classification; same across all laps for a driver |

**Retirement handling:** DNF flags from `session_result`, cross-referenced with last lap in `laps` to determine `retirement_lap`. Retired drivers are assigned their official F1 classification position (laps-completed order) — not excluded, not zeroed.

**Two masks built in Stage 2:**
- `attention_mask [num_drivers, num_laps]` — 1 if racing, 0 if retired
- `target_mask [num_drivers]` — 1 if classified, 0 if DNF

**Tests are mandatory for Stage 2.** Three fixture races chosen by hand:
- One SC race
- One with multiple retirements
- One clean race

Assert known facts (e.g. "driver 44 was on MEDIUM from lap 18–35"). This is where silent bugs are most expensive.

---

## Feature Engineering (Stage 3)

Pure transformation — no API calls, no randomness. Same input → same output. Raw, human-interpretable values; scaling happens in Stage 4.

**Full feature set:**

| Feature | Notes |
|---|---|
| `position` | Raw integer 1–20 |
| `positions_gained_from_grid` | Signed; negative = lost places |
| `num_active_drivers` | Cars still running at this lap |
| `distance_remaining_km` | `circuit_length_km × (total_laps - lap_number)`, normalised by total race distance. Circuit lengths from a static lookup table (fixed FIA data). |
| `gap_to_leader` | Seconds, from `intervals` |
| `interval_to_ahead` | Seconds to car immediately ahead |
| `rolling_lap_time_3_norm` | 3-lap rolling average, normalised vs field median |
| `rolling_lap_time_3_delta_leader` | Driver rolling avg minus leader's rolling avg |
| `last_lap_pace_delta_to_ahead` | Last lap time minus P-1 driver's last lap time |
| `last_lap_pace_delta_to_behind` | Last lap time minus P+1 driver's last lap time |
| `mean_gap_cars_ahead` | Mean of inter-car gaps from P1 to P(N-1). 0 for race leader. |
| `stdev_gap_cars_ahead` | Stdev of same. 0 for P1 and P2. |
| `max_speed_kmh` | Per lap, from Stage 2 |
| `tyre_compound` | Encoded: SOFT/MEDIUM/HARD/INTER/WET |
| `tyre_age_laps` | Laps on current set |
| `stint_number` | Current stint number |
| `stops_vs_median` | `driver_stops - median_stops_across_field` at this lap. Encodes strategic position relative to field. |
| `sc_active` | bool |
| `vsc_active` | bool |
| `red_flag_active` | bool |
| `laps_since_sc_end` | Continuous; 0 if SC currently active |
| `is_street_circuit` | True for Baku, Singapore, Las Vegas, Miami |
| `driver_circuit_finish_rate` | Historical finishing rate at this circuit — prior races only (no leakage) |
| `driver_championship_standing` | Standing entering this race |
| `team_circuit_finish_rate` | Team-level equivalent |
| `team_championship_standing` | Team standing entering this race |

**Leakage guard:** driver/team priors computed in a separate sub-step using only races prior to the current race. Cached per race.

---

## Snapshots & Training Split (Stage 4)

Converts per-race feature tables into training tensors.

**Tensor shape:** `[num_drivers=20, num_features]` per snapshot lap.
**Target:** relevance vector — `21 - final_position` for all drivers (retirees keep their official classification position).
**Snapshots per race:** multiple at fixed lap intervals (e.g. laps 10, 20, 30, 40).
**Padding:** pad to 20 drivers; retirement mask applied.

**Chronological split — never random:**
- Train: 2023 season
- Val: first half of 2024
- Test: second half of 2024

**Scaling:** StandardScaler fitted on train split only. Parameters saved to `metadata.json` and applied identically to val/test.

```
data/snapshots/
  ├── train.parquet
  ├── val.parquet
  ├── test.parquet
  └── metadata.json     # feature names, scaler params, data version hash
```

---

## Models & Loss (Stage 5)

### Run directory structure

```
runs/{run_id}/
    config.yaml                # hyperparams, data version hash, git SHA
    model.lgb / model.pt
    metrics.json
    predictions_test.parquet   # saved alongside model for free error analysis
    train.log
```

MLflow tracks all runs locally.

### LightGBM Baseline

Build first. One row per `(race, snapshot_lap, driver)`. Each `(race, snapshot_lap)` is one LightGBM ranking group. Relevance: `21 - final_position`.

**Target baseline:** Spearman ~0.6–0.75 from lap-30 snapshots. Everything else must beat this.

### Transformer Across Drivers

Input: `[batch, 20, num_features]` → Output: `[batch, 20]` (one score per driver slot).

Key choices:
- **No positional encoding** — drivers are an unordered set; permutation-equivariance is correct
- **Driver identity embedding** (up to 30 indices) — model learns driver-specific priors; index 0 reserved as "unknown driver" for roster turnover
- **Pre-LayerNorm** (`norm_first=True`) — more stable on small datasets
- **Padding mask** applied in score head — retired/absent slots masked to `-1e4` before loss

Starting hyperparameters: `d_model=128`, `n_heads=8`, `n_layers=4`, dropout 0.1, AdamW lr 1e-4, weight decay 0.01, warmup 500 steps + cosine decay.

### Loss: LambdaRank

Pairwise gradients scaled by NDCG impact. Podium swaps get large gradients; midfield swaps get small ones. Only valid pairs (both drivers active at snapshot lap N) enter the loss.

**Fallback:** plain RankNet with valid-pair masking if LambdaRank proves fiddly to implement first.

### Evaluation Metrics

- Spearman rank correlation (primary)
- Top-3 accuracy (podium hit rate)
- Top-1 accuracy (winner)
- Mean position error (sanity check)
- Lap-by-lap prediction evolution plot

### Build Sequence Within Stage 5

1. LightGBM baseline → establish Spearman number
2. Transformer + RankNet loss → beat the baseline
3. Swap in LambdaRank → squeeze podium accuracy
4. Temporal encoding (LSTM wrapper) if plateau

---

## Repository Layout

```
f1-finishing-positions/
├── pyproject.toml
├── plan.md
├── config/
│   └── default.yaml
├── src/
│   └── f1_predictor/
│       ├── ingest.py
│       ├── sessionise.py
│       ├── features.py
│       ├── snapshots.py
│       ├── models/
│       │   ├── baseline_gbm.py
│       │   └── transformer.py
│       ├── losses.py
│       ├── train.py
│       └── evaluate.py
├── scripts/
│   ├── pull_season.py
│   └── run_pipeline.py
├── tests/
│   └── test_sessionise.py
├── docs/
│   └── superpowers/
│       └── specs/
│           └── 2026-06-03-f1-predictor-design.md
└── data/                      # gitignored
    ├── raw/
    ├── sessions/
    ├── features/
    └── snapshots/
```

---

## Known Risks

- **OpenF1 data quality:** occasional outages and schema changes — `meta.json` pull timestamps help debug stale cache
- **Wet races:** different physics — flag with `is_wet_race` in error analysis; consider excluding from v1 if they degrade calibration significantly
- **Track-specific dynamics:** Baku, Singapore, Monaco (excluded) are outliers — `is_street_circuit` flag enables per-circuit error analysis
- **Driver roster turnover:** 2023 grid ≠ 2024 grid — "unknown driver" embedding index handles unseen drivers
- **`car_data` volume:** largest endpoint by far — aggregate to `max_speed_kmh` in Stage 2 and do not carry raw telemetry into the feature pipeline
