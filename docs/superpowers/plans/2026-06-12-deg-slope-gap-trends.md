# Tyre-Degradation Slope + Gap-Trend Features Implementation Plan

> **Status: IMPLEMENTED (2026-06-12).** Built TDD-style on branch `claude/bold-gauss-e98pwd` (commits `c2c8692`, `5c4d62a`). See **Outcome (as-built)** below.

**Goal:** Add two Tier-1 feature groups to the Stage 3 pipeline — a within-stint tyre-degradation slope (#3) and gap-trend / proximity features (#5) — so the model sees pace decay and battle dynamics, not just instantaneous state.

**Architecture:** Both groups are pure Stage 3 transforms on columns that are **already in the Stage 2 sessionised table** (`lap_time`, `tyre_age_laps`, `stint_number`, `gap_to_leader`, `interval_to_ahead`, `position`, and the flag columns). So — unlike the speed-trap/sector work — **no Stage 2 / ingest change is needed.** Each group is one new helper in `features.py` (mirroring the existing `_add_rolling_pace` / `_add_pace_deltas` / `_add_speed_sector_deltas` patterns), registered in `FEATURE_COLUMNS`, and called in `build_features`. Everything downstream (snapshots → scaler → metadata → LightGBM) picks the new columns up automatically because `scripts/build_snapshots.py` reads `FEATURE_COLUMNS` dynamically.

**Tech Stack:** Python 3.12, Polars (per-race transforms), LightGBM (model), pytest, `uv` for env. Run all commands in WSL2.

---

## Outcome (as-built)

All four tasks completed; 9 new unit tests pass; full suite green (145 passed). The 6 features populate on real 2026 data and are used by the model. **Ablation (model with vs without the 6, on top of the speed/sector batch):**

| | VAL (same-regime) | TEST (2026, n=5) |
|---|---|---|
| naive | 0.7935 | 0.8790 |
| model without the 6 | 0.7933 | 0.8670 |
| model with the 6 | **0.7998** | **0.8703** |

The features **help both regimes** (+0.0065 val, +0.0033 on 2026) and flip the model from *tied* to **beating** naive on val (uplift +0.0064). On 2026 they narrow the gap to naive (−0.012 → −0.009) without closing it — the residual 2026 gap is structural (stale championship-standing priors + 2026 being more processional) and needs the **adaptation** work, not more features. `is_2026_regs` remains inert (no 2026 race in training). Caveat: only 5 completed 2026 races — treat the 2026 magnitudes as directional.

One small deviation from the plan: Tasks 1 and 2 were committed together (`c2c8692`) because the shared test module would otherwise have left a red test collection between the two commits.

---

## Context

The model currently sees a **snapshot** of race state at lap N but almost no **trend**. The 2026 drift diagnostic showed the model can't beat naive persistence on 2026, and the recently-added speed-trap/sector features helped the 2026 regime specifically (+0.010) — confirming relational pace signals are worth adding. Two Tier-1 ideas remain unbuilt:

- **#3 Tyre-degradation slope** — is a driver managing tyres to the end, or falling off a cliff (about to be swallowed)? A per-stint OLS slope of `lap_time` vs `tyre_age_laps` captures pace decay that a single lap-time can't.
- **#5 Gap trends + proximity** — the model sees the *level* of `gap_to_leader` / `interval_to_ahead` but not their *direction*. Closing on the car ahead at 0.4 s/lap is a near-certain pass; it's currently invisible. It also has **no gap-to-car-behind** signal at all (pressure from behind), and no "within striking distance" flag.

Both are causal (use only laps ≤ N) so they introduce no target leakage.

**Design decisions locked in (from user):**
- Deg slope uses **clean racing laps only** — excludes `pit_this_lap`, `sc_active`, `vsc_active`, `red_flag_active` (these are far slower and would distort a slope much more than they distort the existing 3-lap rolling mean). Computed as an **expanding-within-stint** OLS via masked cumulative sums; null until ≥3 clean laps exist in the stint.
- Gap trends use the **full 5-feature set**. The old `in_drs_range` idea is renamed **`in_striking_distance`** (`interval_to_ahead < 1.0 s`) because 2026 has no DRS.

**New features (6 total):**
| Feature | Meaning |
|---|---|
| `tyre_deg_slope` | sec/lap OLS slope of lap_time vs tyre age, current stint, clean laps (>0 = degrading) |
| `gap_to_leader_delta_3lap` | gap_to_leader[N] − [N−3] (>0 = dropping away from front) |
| `interval_to_ahead_delta_3lap` | interval_to_ahead[N] − [N−3] (<0 = closing on car ahead) |
| `interval_to_behind` | gap to the car directly behind (= that car's interval_to_ahead); null for last place |
| `interval_to_behind_delta_3lap` | change in gap-to-behind over 3 laps (<0 = being caught) |
| `in_striking_distance` | 1 if interval_to_ahead < 1.0 s else 0 |

---

## File Structure

- **Modify:** `src/f1_predictor/features.py` — add two helpers (`_add_tyre_deg_slope`, `_add_gap_trends`), extend the `FEATURE_COLUMNS` list, and call both helpers inside `build_features` after `_add_speed_sector_deltas`. This is the single source file changed.
- **Create:** `tests/test_gap_deg_features.py` — unit tests for both helpers and the `FEATURE_COLUMNS` contract (mirrors `tests/test_speed_sector_features.py`).

No other files change. `scripts/build_snapshots.py`, `scripts/run_pipeline.py`, `snapshots.py`, and the model code are untouched — they read `FEATURE_COLUMNS` dynamically.

**Branch:** `claude/bold-gauss-e98pwd`.

---

## Task 1: Tyre-degradation slope helper

`_add_tyre_deg_slope` in `src/f1_predictor/features.py` (after `_add_rolling_pace`). Clean-lap mask → masked per-lap contributions → cumulative sums over `(driver_number, stint_number)` in lap order → `slope = (n·Σxy − Σx·Σy) / (n·Σxx − Σx²)`, emitted only when `n ≥ 3` and the denominator is non-degenerate. Tests: linear-slope recovery + null-until-3-laps, and SC/pit-lap exclusion (`tests/test_gap_deg_features.py`). See commit `c2c8692` for the exact implementation.

## Task 2: Gap-trend + proximity helper

`_add_gap_trends` in `src/f1_predictor/features.py` (after `_add_tyre_deg_slope`). `interval_to_behind` via a deduped self-join on `position + 1` (mirrors `_add_pace_deltas`); the three `*_delta_3lap` via `shift(3).over("driver_number")` after sorting by `(driver_number, lap_number)`; `in_striking_distance` = `(interval_to_ahead < 1.0).fill_null(False)`. Tests: gap-to-behind, striking-distance, and 3-lap deltas. Commit `c2c8692`.

## Task 3: Register and wire

Add the 6 names to `FEATURE_COLUMNS` (after `sector3_time_delta_to_field`) and call `_add_tyre_deg_slope` + `_add_gap_trends` in `build_features` right after `_add_speed_sector_deltas`. Contract test asserts all 6 are in `FEATURE_COLUMNS`. They propagate to snapshots/scaler/metadata/LightGBM automatically (build_snapshots reads `FEATURE_COLUMNS`). Commit `5c4d62a`.

## Task 4: Rebuild on real data and measure

`uv run python scripts/run_pipeline.py && uv run python scripts/build_snapshots.py` (rebuilds 72 races → train 58 / val 9 / test 5), confirm the 6 columns populate on a 2026 race, then `scripts/eval_degradation.py` + a with/without ablation. Results in **Outcome** above.

---

## Verification (end-to-end)

1. **Unit:** `uv run pytest tests/test_gap_deg_features.py -q` → 9 tests pass; full `uv run pytest tests/ -q` → 145 pass.
2. **Feature realness:** all six columns populate (no all-null) on a 2026 race; `in_striking_distance ∈ {0,1}`.
3. **Model integration:** `eval_degradation.py` runs; the six features appear in `feature_importance_gain`.
4. **Net value:** the ablation table quantifies with-vs-without on val and 2026.

## Notes & caveats

- **No Stage 2 / ingest change** — all inputs already exist in the sessionised table; missing inputs degrade to null (Stage 4 imputes null → 0).
- **`gap_to_leader_delta_3lap` partly overlaps** the existing `rolling_lap_time_3_delta_leader`; first candidate to drop if it adds val noise.
- **Window = 3 laps** matches the existing `rolling_lap_time_3` convention; deltas are null for a driver's first 3 laps (imputed to 0).
- **Large tails** (interval deltas in the 100s–900s) are the usual pit-stop/lapping artifacts the existing gap features already carry; the tree model and Stage 4 scaler absorb them.
- **Commit identity:** this environment has no commit-signing key, so commits show "Unverified" on GitHub; committer is `Claude <noreply@anthropic.com>`.
