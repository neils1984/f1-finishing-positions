# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

An F1 race position predictor: given the state of a race at lap N, predict the final finishing order of all drivers. This is a **ranking problem** (not regression) — drivers are interdependent and the cost of error is non-uniform (wrong podium call > wrong P14 call). A core goal is tracking prediction confidence as a race progresses, lap by lap.

Full design spec: `docs/superpowers/specs/2026-06-03-f1-predictor-design.md`
Original project plan: `plan.md`

## Environment

- Developed in WSL2 (Ubuntu) on Windows. Run all commands in the WSL2 terminal.
- `uv` for env management — `uv sync` to install, `uv run <cmd>` to execute.
- No Python on the Windows host; everything runs inside WSL2.

## Commands

```bash
uv sync                                          # install dependencies
uv run python scripts/pull_season.py --year 2023 # pull a full season from OpenF1
uv run python scripts/run_pipeline.py            # run all pipeline stages
uv run pytest tests/                             # run all tests
uv run pytest tests/test_sessionise.py           # run sessionising tests (the critical ones)
mlflow ui                                        # view experiment runs at localhost:5000
```

## Pipeline Architecture

Five stages, each independently runnable, each producing cached Parquet artefacts. Re-pulling from OpenF1 is rare; re-running a stage is cheap.

```
OpenF1 API
    ↓
1. ingest.py       → data/raw/{session_key}/{endpoint}.parquet
    ↓
2. sessionise.py   → data/sessions/{session_key}.parquet   (one row per driver-lap)
    ↓
3. features.py     → data/features/{session_key}.parquet   (engineered features)
    ↓
4. snapshots.py    → data/snapshots/{split}.parquet        (training tensors)
    ↓
5. train.py        → runs/{run_id}/
```

`data/` is gitignored. `runs/` contains MLflow artefacts and model checkpoints.

## Constraints That Are Easy to Get Wrong

**Chronological split — never random.** Train on 2023 → val on first half of 2024 → test on second half of 2024. Random splitting leaks race dynamics across boundaries.

**Driver/team priors must use only prior-race data.** Historical finishing rates and championship standings are computed using only races that occurred before the current race. Violation = target leakage. These are computed in a separate sub-step and cached per race.

**Scaling happens in Stage 4, not Stage 3.** Stage 3 outputs raw, human-readable values (position as integer 1–20, times in seconds, gaps in seconds). StandardScaler is fitted on the train split only and applied at snapshot generation time. Never apply scaling in Stage 3.

**Retirement handling.** Retired drivers are assigned their official F1 classification position (laps-completed order) — not excluded, not given position 0. They must stay in the ranking target with their real classification. The loss function uses valid-pair masking (only pairs where both drivers were active at snapshot lap N enter the loss).

**`car_data` is pulled but not carried forward raw.** It is aggregated to `max_speed_kmh` per driver-lap in Stage 2 and the raw telemetry is not included in the feature pipeline. It's the largest endpoint by volume.

**Monaco is excluded from training data entirely.** Other street circuits (Baku, Singapore, Las Vegas, Miami) are included but tagged with `is_street_circuit = True`.

**`position` uses a fixed denominator for normalisation (Stage 4 scaler), not `position / num_active_drivers`.** The latter shifts as drivers retire, distorting the signal.

## Tooling Choices

| Concern | Choice | Why |
|---|---|---|
| Per-race transforms | Polars | Fast, native Parquet |
| Cross-race joins (Stage 3 priors) | DuckDB | Excellent for cross-file SQL |
| Experiment tracking | MLflow (local) | No external account needed |
| CLI | Typer | Type-annotated, minimal boilerplate |
| Baseline model | LightGBM (`objective='lambdarank'`) | Establishes Spearman benchmark |
| Neural model | PyTorch Transformer (cross-driver attention) | Captures relative race state |
| Loss | LambdaRank → RankNet fallback | Podium swaps get large gradients |

## Build Order

Do not skip ahead. The discipline of getting all five stages working end-to-end on one race before deepening any stage is what separates projects that ship from projects that don't.

1. Stage 1 + 2 for a **single race** → eyeball output → write sessionising tests
2. Stage 1 + 2 for the full 2023 + 2024 seasons
3. Stage 3 with the core feature set
4. Stage 4 with a single snapshot lap per race (lap 30)
5. Stage 5 with **LightGBM baseline** → get an end-to-end Spearman number before anything else
6. Richer features, multiple snapshot laps, then the Transformer
7. Lap-by-lap prediction visualiser

## Stage 2 Tests Are Mandatory

`tests/test_sessionise.py` uses three hand-picked fixture races:
- One SC race
- One with multiple retirements
- One clean race

Assert known facts against these (e.g. tyre compound windows, retirement laps). This is where silent data bugs are most expensive and hardest to detect downstream.

## Key Feature Notes

- `stops_vs_median` = driver stops minus median stops across active field at that lap. Encodes strategic position relative to field without hardcoding expected stop counts.
- `mean_gap_cars_ahead` / `stdev_gap_cars_ahead` — inter-car gaps from P1 to P(N-1), derived from `gap_to_leader` differences. Captures traffic density ahead. Both are 0 for the race leader.
- `distance_remaining_km` uses a static circuit-length lookup table (FIA data). Do not use `lap_number / total_laps` — circuit lengths vary significantly and distort the normalisation.
- Tyre compound is one-hot encoded (5 binary columns: SOFT/MEDIUM/HARD/INTER/WET).
- Driver identity embeddings: up to 30 indices, index 0 reserved as "unknown driver" for roster turnover between seasons.
