# Feature Engineering (Stage 3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Stage 3 of the F1 predictor pipeline: turn each race's sessionised driver-lap table into an engineered feature table (`data/features/{session_key}.parquet`), including cross-race driver/team priors computed with a strict no-leakage guard.

**Architecture:** Per-race features are pure Polars transforms over the Stage 2 output (plus the raw `position`/`drivers`/`sessions` endpoints for grid, team, and circuit identity). Cross-race priors (championship standings, circuit finish rates) are computed once across all races with DuckDB using **only races that occurred before the current race**, then joined back per race. Stage 3 emits **raw, human-readable values** — scaling happens in Stage 4.

**Tech Stack:** Python 3.11+, `uv`, Polars (per-race), DuckDB (cross-race joins), Typer (CLI), pytest.

**Spec:** `docs/superpowers/specs/2026-06-03-f1-predictor-design.md` (Stage 3 section)
**This is Plan 2 of 4.** Plan 1 (Pipeline Foundation, Stages 1–2) is complete. Plans 3–4 cover Snapshots + LightGBM and the Transformer.

---

## Real-data decisions (carried from Plan 1)

These resolve ambiguities the spec leaves open, grounded in what the live OpenF1 data actually contains. They are also recorded in project memory (`openf1-real-data-gotchas`).

1. **`max_speed_kmh` is null.** Plan 1 deferred `car_data` (a per-session query returns HTTP 422). The column exists in Stage 2 output but is all-null. Stage 3 passes it through unchanged; it becomes meaningful when `car_data` is backfilled per-driver. Do **not** drop the column.
2. **Gaps are strings.** `gap_to_leader` / `interval_to_ahead` in the Stage 2 output are `String` (OpenF1 reports `"+1 LAP"` for lapped drivers). Stage 3 parses them to `Float64`: a plain number → that number; `"+N LAP"`/`"+N LAPS"`/null/empty → null. Features derived from gaps inherit those nulls; Stage 4 imputes.
3. **`distance_remaining_km` is raw kilometres.** Per CLAUDE.md ("use a static circuit-length lookup; do not use `lap_number / total_laps`"), output `circuit_length_km × (total_laps − lap_number)` — actual remaining km, **not** normalised to a lap fraction. Stage 4's scaler handles magnitude. `total_laps` = the maximum `lap_number` in the race (the winner's lap count).
4. **Grid position** comes from the raw `position` endpoint: each driver's position at their earliest reading (a pre-race/grid timestamp, ~1h before lights-out). Stage 3 reads `data/raw/{key}/position.parquet` for this.
5. **First-race priors are null.** A driver/team with no prior races (or no prior race at this circuit) gets null for the relevant prior. Stage 3 leaves nulls; Stage 4 imputes. This is correct no-leakage behaviour, not a bug.
6. **Cancelled races are absent.** 2023 Imola (`9086`) has no sessionised file (Plan 1 skips empty-laps sessions); it simply never appears in Stage 3 inputs or the prior history.

---

## File Map

```
config/circuits.yaml                       CREATE  circuit lengths (km) + street-circuit flags
src/f1_predictor/features.py               CREATE  Stage 3: per-race feature transforms + public features()
src/f1_predictor/priors.py                 CREATE  Stage 3 cross-race priors (DuckDB), no-leakage
scripts/build_features.py                  CREATE  Typer CLI: features for one race or all races
tests/test_features_unit.py                CREATE  per-race transform unit tests (synthetic)
tests/test_priors_unit.py                  CREATE  cross-race prior unit tests (synthetic) incl. leakage guard
tests/test_features.py                     CREATE  integration tests on the 3 real 2023 fixtures
```

The Stage 2 fixtures from Plan 1 are reused: SC race `9070`, retirement race `9181`, clean race `9078` (see `tests/conftest.py`). Stage 3 integration tests require `data/sessions/` and `data/raw/` to be populated (run `uv run python scripts/pull_season.py --year 2023` then sessionise). They skip if absent, mirroring Plan 1.

---

## Output schema (`data/features/{session_key}.parquet`)

One row per `(session_key, driver_number, lap_number)`. Keys + 24 engineered feature columns. Raw values (no scaling).

Keys: `session_key`, `driver_number`, `lap_number`, plus `final_position` (target source, carried through unchanged for Stage 4).

Features: `position`, `positions_gained_from_grid`, `num_active_drivers`, `distance_remaining_km`, `gap_to_leader`, `interval_to_ahead`, `rolling_lap_time_3_norm`, `rolling_lap_time_3_delta_leader`, `last_lap_pace_delta_to_ahead`, `last_lap_pace_delta_to_behind`, `mean_gap_cars_ahead`, `stdev_gap_cars_ahead`, `max_speed_kmh`, `tyre_soft`, `tyre_medium`, `tyre_hard`, `tyre_inter`, `tyre_wet`, `tyre_age_laps`, `stint_number`, `stops_vs_median`, `sc_active`, `vsc_active`, `red_flag_active`, `laps_since_sc_end`, `is_street_circuit`, `driver_circuit_finish_rate`, `driver_championship_standing`, `team_circuit_finish_rate`, `team_championship_standing`.

---

## Task 1: Scaffold — circuit reference + module skeleton

**Files:**
- Create: `config/circuits.yaml`
- Create: `src/f1_predictor/features.py`
- Create: `tests/test_features_unit.py`

- [ ] **Step 1: Create `config/circuits.yaml`**

Circuit lengths are FIA track lengths in km, keyed by OpenF1 `circuit_short_name`. Verify against official FIA data before training; these are the documented lengths for the 2023–2024 calendars. Monaco is excluded from training but kept here for completeness.

```yaml
# OpenF1 circuit_short_name -> length in km (FIA). Used for distance_remaining_km.
lengths_km:
  Sakhir: 5.412
  Jeddah: 6.174
  Melbourne: 5.278
  Baku: 6.003
  Miami: 5.412
  Imola: 4.909
  Monte Carlo: 3.337
  Catalunya: 4.657
  Montreal: 4.361
  Spielberg: 4.318
  Silverstone: 5.891
  Hungaroring: 4.381
  Spa-Francorchamps: 7.004
  Zandvoort: 4.259
  Monza: 5.793
  Singapore: 4.940
  Suzuka: 5.807
  Lusail: 5.419
  Austin: 5.513
  Mexico City: 4.304
  Interlagos: 4.309
  Las Vegas: 6.201
  Yas Marina Circuit: 5.281
  Shanghai: 5.451

# Street circuits (is_street_circuit = True). Monaco is excluded from training.
street:
  - Baku
  - Singapore
  - Las Vegas
  - Miami
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_features_unit.py
"""Unit tests for Stage 3 per-race feature transforms using synthetic data."""
import polars as pl
import pytest
from f1_predictor.features import load_circuits, circuit_length_km, is_street_circuit


def test_load_circuits_reads_yaml():
    circuits = load_circuits()
    assert "lengths_km" in circuits
    assert "street" in circuits
    assert circuits["lengths_km"]["Sakhir"] == pytest.approx(5.412)


def test_circuit_length_km_known_circuit():
    circuits = load_circuits()
    assert circuit_length_km("Baku", circuits) == pytest.approx(6.003)


def test_circuit_length_km_unknown_returns_none():
    circuits = load_circuits()
    assert circuit_length_km("Nowhere", circuits) is None


def test_is_street_circuit_flags():
    circuits = load_circuits()
    assert is_street_circuit("Baku", circuits) is True
    assert is_street_circuit("Silverstone", circuits) is False
```

- [ ] **Step 3: Run to verify failure**

Run: `uv run pytest tests/test_features_unit.py -v`
Expected: `ModuleNotFoundError: No module named 'f1_predictor.features'`

- [ ] **Step 4: Create `src/f1_predictor/features.py`**

```python
"""Stage 3: engineer features per race from the Stage 2 sessionised table.

Pure, deterministic transforms producing raw (unscaled) human-readable values.
Scaling happens in Stage 4. Cross-race priors live in priors.py.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import yaml

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


def load_circuits(path: Path | None = None) -> dict:
    """Load the circuit reference (lengths + street-circuit list)."""
    path = path or (_CONFIG_DIR / "circuits.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def circuit_length_km(circuit_short_name: str, circuits: dict) -> float | None:
    """Track length in km for a circuit_short_name, or None if unknown."""
    return circuits.get("lengths_km", {}).get(circuit_short_name)


def is_street_circuit(circuit_short_name: str, circuits: dict) -> bool:
    """True if the circuit is a street circuit (Baku/Singapore/Las Vegas/Miami)."""
    return circuit_short_name in set(circuits.get("street", []))
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_features_unit.py -v`
Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add config/circuits.yaml src/f1_predictor/features.py tests/test_features_unit.py
git commit -m "feat: stage 3 scaffold — circuit reference table + features module"
```

---

## Task 2: Numeric gap parsing

OpenF1 reports `gap_to_leader` / `interval_to_ahead` as strings (`"+1 LAP"` for lapped cars). Convert to `Float64`, mapping non-numeric markers to null.

**Files:**
- Modify: `src/f1_predictor/features.py`
- Modify: `tests/test_features_unit.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_features_unit.py`:

```python
from f1_predictor.features import _parse_gap_columns


def test_parse_gap_columns_numeric_and_lapped():
    df = pl.DataFrame({
        "gap_to_leader": ["0.0", "1.234", "+1 LAP", "+2 LAPS", None],
        "interval_to_ahead": [None, "1.234", "0.5", "+1 LAP", ""],
    })
    out = _parse_gap_columns(df)
    assert out["gap_to_leader"].dtype == pl.Float64
    assert out["interval_to_ahead"].dtype == pl.Float64
    assert out["gap_to_leader"].to_list()[:2] == [0.0, 1.234]
    # "+1 LAP", "+2 LAPS", null, and "" all become null
    assert out["gap_to_leader"].to_list()[2] is None
    assert out["gap_to_leader"].to_list()[3] is None
    assert out["interval_to_ahead"].to_list()[4] is None


def test_parse_gap_columns_already_float_is_passthrough():
    df = pl.DataFrame({
        "gap_to_leader": [0.0, 1.5],
        "interval_to_ahead": [None, 0.3],
    })
    out = _parse_gap_columns(df)
    assert out["gap_to_leader"].to_list() == [0.0, 1.5]
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_features_unit.py::test_parse_gap_columns_numeric_and_lapped -v`
Expected: `ImportError: cannot import name '_parse_gap_columns'`

- [ ] **Step 3: Implement `_parse_gap_columns`**

Add to `src/f1_predictor/features.py`:

```python
_GAP_COLUMNS = ["gap_to_leader", "interval_to_ahead"]


def _parse_gap_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Coerce gap columns to Float64; non-numeric markers (e.g. '+1 LAP') -> null.

    pl.col(...).cast(Float64, strict=False) turns any value that doesn't parse as
    a number into null, which is exactly the desired behaviour for '+N LAP(S)',
    '' and existing nulls. Already-Float64 columns pass through unchanged.
    """
    exprs = []
    for col in _GAP_COLUMNS:
        if col in df.columns and df.schema[col] != pl.Float64:
            exprs.append(pl.col(col).cast(pl.Float64, strict=False).alias(col))
    return df.with_columns(exprs) if exprs else df
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_features_unit.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/f1_predictor/features.py tests/test_features_unit.py
git commit -m "feat: stage 3 — parse string gap columns to Float64"
```

---

## Task 3: Simple per-driver features

Passthrough + single-driver derived features that need no cross-driver context: `position`, `tyre_age_laps`, `stint_number`, `sc_active`, `vsc_active`, `red_flag_active`, `laps_since_sc_end`, `max_speed_kmh`, plus `num_active_drivers` (per-lap count) and `distance_remaining_km`.

**Files:**
- Modify: `src/f1_predictor/features.py`
- Modify: `tests/test_features_unit.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_features_unit.py`:

```python
from f1_predictor.features import _add_active_and_distance


def _mini_sessionised() -> pl.DataFrame:
    """3 drivers, 3 laps. Driver 3 retires on lap 2."""
    return pl.DataFrame({
        "session_key": [9000] * 9,
        "driver_number": [1, 1, 1, 2, 2, 2, 3, 3, 3],
        "lap_number": [1, 2, 3, 1, 2, 3, 1, 2, 3],
        "position": [1, 1, 1, 2, 2, 2, 3, 3, 3],
        "is_retired": [False] * 6 + [True] * 3,
        "retirement_lap": [None] * 6 + [2, 2, 2],
        "lap_time": [90.0, 89.0, 88.0, 91.0, 90.5, 90.0, 92.0, 93.0, None],
    })


def test_num_active_drivers_decreases_after_retirement():
    df = _add_active_and_distance(_mini_sessionised(), circuit_length=5.0)
    by_lap = (
        df.group_by("lap_number").agg(pl.col("num_active_drivers").first())
        .sort("lap_number")
    )
    # Driver 3 retired on lap 2, so it is inactive only from lap 3 onward.
    assert by_lap["num_active_drivers"].to_list() == [3, 3, 2]


def test_distance_remaining_km_uses_circuit_length():
    df = _add_active_and_distance(_mini_sessionised(), circuit_length=5.0)
    # total_laps = max lap_number = 3. circuit_length = 5.0 km.
    row = df.filter((pl.col("driver_number") == 1) & (pl.col("lap_number") == 1))
    assert row["distance_remaining_km"][0] == pytest.approx(5.0 * (3 - 1))
    last = df.filter((pl.col("driver_number") == 1) & (pl.col("lap_number") == 3))
    assert last["distance_remaining_km"][0] == pytest.approx(0.0)


def test_distance_remaining_km_null_when_circuit_unknown():
    df = _add_active_and_distance(_mini_sessionised(), circuit_length=None)
    assert df["distance_remaining_km"].null_count() == df.height
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_features_unit.py::test_num_active_drivers_decreases_after_retirement -v`
Expected: `ImportError: cannot import name '_add_active_and_distance'`

- [ ] **Step 3: Implement `_add_active_and_distance`**

Add to `src/f1_predictor/features.py`:

```python
def _add_active_and_distance(
    df: pl.DataFrame, circuit_length: float | None
) -> pl.DataFrame:
    """Add num_active_drivers (per lap) and distance_remaining_km (raw km).

    A driver is active at lap L if not retired, or retired with retirement_lap >= L.
    total_laps is the maximum lap_number in the race (the winner's lap count).
    distance_remaining_km is null when the circuit length is unknown.
    """
    total_laps = df["lap_number"].max()

    active = (
        pl.col("retirement_lap").is_null() | (pl.col("retirement_lap") >= pl.col("lap_number"))
    )
    df = df.with_columns(active.alias("_active"))

    per_lap = (
        df.group_by("lap_number")
        .agg(pl.col("_active").sum().cast(pl.Int64).alias("num_active_drivers"))
    )

    dist_expr = (
        pl.lit(None, dtype=pl.Float64)
        if circuit_length is None
        else (pl.lit(float(circuit_length)) * (pl.lit(total_laps) - pl.col("lap_number")))
    )

    return (
        df.join(per_lap, on="lap_number", how="left")
        .with_columns(dist_expr.alias("distance_remaining_km"))
        .drop("_active")
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_features_unit.py -v`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add src/f1_predictor/features.py tests/test_features_unit.py
git commit -m "feat: stage 3 — num_active_drivers + distance_remaining_km"
```

---

## Task 4: Field-relative features

Features needing the per-lap ordering of drivers: `positions_gained_from_grid`, `last_lap_pace_delta_to_ahead`, `last_lap_pace_delta_to_behind`, `mean_gap_cars_ahead`, `stdev_gap_cars_ahead`. Grid is supplied as a `{driver_number: grid_position}` map (extracted from the raw `position` endpoint in Task 7's assembly).

**Files:**
- Modify: `src/f1_predictor/features.py`
- Modify: `tests/test_features_unit.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_features_unit.py`:

```python
import math
from f1_predictor.features import _add_positions_gained, _add_pace_deltas, _add_gaps_ahead


def test_positions_gained_from_grid_sign():
    df = pl.DataFrame({
        "driver_number": [1, 2, 3],
        "lap_number": [5, 5, 5],
        "position": [1, 2, 3],
    })
    grid = {1: 3, 2: 2, 3: 1}  # driver 1 started P3 now P1 -> gained 2
    out = _add_positions_gained(df, grid).sort("driver_number")
    assert out["positions_gained_from_grid"].to_list() == [2, 0, -2]


def test_pace_deltas_to_ahead_and_behind():
    # One lap, three cars by position with known lap_times.
    df = pl.DataFrame({
        "driver_number": [1, 2, 3],
        "lap_number": [5, 5, 5],
        "position": [1, 2, 3],
        "lap_time": [88.0, 89.5, 91.0],
    })
    out = _add_pace_deltas(df).sort("position")
    # P2 vs ahead (P1): 89.5 - 88.0 = 1.5 ; vs behind (P3): 89.5 - 91.0 = -1.5
    p2 = out.filter(pl.col("position") == 2)
    assert p2["last_lap_pace_delta_to_ahead"][0] == pytest.approx(1.5)
    assert p2["last_lap_pace_delta_to_behind"][0] == pytest.approx(-1.5)
    # Leader has no car ahead -> null ahead delta
    p1 = out.filter(pl.col("position") == 1)
    assert p1["last_lap_pace_delta_to_ahead"][0] is None
    # Last car has no car behind -> null behind delta
    p3 = out.filter(pl.col("position") == 3)
    assert p3["last_lap_pace_delta_to_behind"][0] is None


def test_gaps_ahead_mean_and_stdev():
    # Positions 1..4 with cumulative gap_to_leader 0, 1, 3, 6 -> inter-car gaps 1,2,3.
    df = pl.DataFrame({
        "driver_number": [1, 2, 3, 4],
        "lap_number": [5, 5, 5, 5],
        "position": [1, 2, 3, 4],
        "gap_to_leader": [0.0, 1.0, 3.0, 6.0],
    })
    out = _add_gaps_ahead(df).sort("position")
    m = out["mean_gap_cars_ahead"].to_list()
    s = out["stdev_gap_cars_ahead"].to_list()
    # Leader (P1): no cars ahead -> 0, 0
    assert m[0] == pytest.approx(0.0) and s[0] == pytest.approx(0.0)
    # P2: cars ahead = {P1}; no inter-car gap -> 0, 0
    assert m[1] == pytest.approx(0.0) and s[1] == pytest.approx(0.0)
    # P3: inter-car gaps among {P1,P2} = [1] -> mean 1, stdev 0
    assert m[2] == pytest.approx(1.0) and s[2] == pytest.approx(0.0)
    # P4: inter-car gaps among {P1,P2,P3} = [1,2] -> mean 1.5, stdev 0.5 (population)
    assert m[3] == pytest.approx(1.5) and s[3] == pytest.approx(0.5)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_features_unit.py::test_positions_gained_from_grid_sign -v`
Expected: `ImportError: cannot import name '_add_positions_gained'`

- [ ] **Step 3: Implement the three functions**

Add to `src/f1_predictor/features.py`:

```python
import numpy as np


def _add_positions_gained(df: pl.DataFrame, grid: dict[int, int]) -> pl.DataFrame:
    """positions_gained_from_grid = grid_position - current position (>0 = gained)."""
    grid_df = pl.DataFrame(
        {"driver_number": list(grid.keys()), "_grid": list(grid.values())},
        schema={"driver_number": df.schema["driver_number"], "_grid": pl.Int64},
    )
    return (
        df.join(grid_df, on="driver_number", how="left")
        .with_columns((pl.col("_grid") - pl.col("position")).alias("positions_gained_from_grid"))
        .drop("_grid")
    )


def _add_pace_deltas(df: pl.DataFrame) -> pl.DataFrame:
    """last_lap_pace_delta_to_ahead/behind: lap_time minus the P-1 / P+1 car's lap_time."""
    pace = df.select(["lap_number", "position", "lap_time"])
    ahead = pace.rename({"position": "_pos_join", "lap_time": "_lt_ahead"})
    behind = pace.rename({"position": "_pos_join", "lap_time": "_lt_behind"})

    return (
        df.with_columns([
            (pl.col("position") - 1).alias("_ahead_pos"),
            (pl.col("position") + 1).alias("_behind_pos"),
        ])
        .join(ahead, left_on=["lap_number", "_ahead_pos"], right_on=["lap_number", "_pos_join"], how="left")
        .join(behind, left_on=["lap_number", "_behind_pos"], right_on=["lap_number", "_pos_join"], how="left")
        .with_columns([
            (pl.col("lap_time") - pl.col("_lt_ahead")).alias("last_lap_pace_delta_to_ahead"),
            (pl.col("lap_time") - pl.col("_lt_behind")).alias("last_lap_pace_delta_to_behind"),
        ])
        .drop(["_ahead_pos", "_behind_pos", "_lt_ahead", "_lt_behind"])
    )


def _gaps_ahead_for_lap(positions: list[int], gaps: list[float]) -> dict[int, tuple[float, float]]:
    """For one lap, return {position: (mean, stdev)} of inter-car gaps among cars ahead.

    Inter-car gap at position k (k>=2) = gap_to_leader[k] - gap_to_leader[k-1].
    For a driver at position P, aggregate the inter-car gaps of positions 2..P-1.
    Leader and P2 have <2 cars ahead -> (0, 0). Population stdev (ddof=0).
    """
    order = sorted(range(len(positions)), key=lambda i: positions[i])
    sorted_pos = [positions[i] for i in order]
    sorted_gap = [gaps[i] for i in order]

    inter = [sorted_gap[k] - sorted_gap[k - 1] for k in range(1, len(sorted_gap))]  # len n-1
    result: dict[int, tuple[float, float]] = {}
    for idx, pos in enumerate(sorted_pos):
        ahead_inter = inter[: max(idx - 1, 0)]  # gaps among positions strictly ahead
        if len(ahead_inter) == 0:
            result[pos] = (0.0, 0.0)
        else:
            arr = np.array(ahead_inter, dtype=float)
            result[pos] = (float(arr.mean()), float(arr.std(ddof=0)))
    return result


def _add_gaps_ahead(df: pl.DataFrame) -> pl.DataFrame:
    """Add mean_gap_cars_ahead / stdev_gap_cars_ahead (traffic density ahead).

    Rows with null position or null gap_to_leader at a lap are excluded from that
    lap's gap computation and receive null features.
    """
    means: list[float | None] = []
    stdevs: list[float | None] = []
    keys: list[tuple] = []

    for (lap,), grp in df.group_by(["lap_number"], maintain_order=True):
        valid = grp.filter(pl.col("position").is_not_null() & pl.col("gap_to_leader").is_not_null())
        lookup = _gaps_ahead_for_lap(
            valid["position"].to_list(), valid["gap_to_leader"].to_list()
        )
        for d, p in zip(grp["driver_number"].to_list(), grp["position"].to_list()):
            keys.append((lap, d))
            m, s = lookup.get(p, (None, None))
            means.append(m)
            stdevs.append(s)

    feat = pl.DataFrame({
        "lap_number": [k[0] for k in keys],
        "driver_number": [k[1] for k in keys],
        "mean_gap_cars_ahead": means,
        "stdev_gap_cars_ahead": stdevs,
    }, schema_overrides={
        "lap_number": df.schema["lap_number"],
        "driver_number": df.schema["driver_number"],
    })
    return df.join(feat, on=["lap_number", "driver_number"], how="left")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_features_unit.py -v`
Expected: 12 passed.

- [ ] **Step 5: Commit**

```bash
git add src/f1_predictor/features.py tests/test_features_unit.py
git commit -m "feat: stage 3 — positions gained, pace deltas, gaps-ahead density"
```

---

## Task 5: Rolling pace features

`rolling_lap_time_3_norm` (3-lap rolling mean lap_time, divided by the field-median rolling mean at that lap) and `rolling_lap_time_3_delta_leader` (driver rolling mean minus the leader's rolling mean at that lap).

**Files:**
- Modify: `src/f1_predictor/features.py`
- Modify: `tests/test_features_unit.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_features_unit.py`:

```python
from f1_predictor.features import _add_rolling_pace


def test_rolling_lap_time_3_norm_and_delta_leader():
    # Two drivers, 3 laps. Driver 1 is the leader (position 1) every lap.
    df = pl.DataFrame({
        "driver_number": [1, 1, 1, 2, 2, 2],
        "lap_number": [1, 2, 3, 1, 2, 3],
        "position": [1, 1, 1, 2, 2, 2],
        "lap_time": [90.0, 90.0, 90.0, 100.0, 100.0, 100.0],
    })
    out = _add_rolling_pace(df).sort(["driver_number", "lap_number"])
    # Lap 3: rolling3 driver1 = 90, driver2 = 100. field median of {90,100} = 95.
    d1_l3 = out.filter((pl.col("driver_number") == 1) & (pl.col("lap_number") == 3))
    d2_l3 = out.filter((pl.col("driver_number") == 2) & (pl.col("lap_number") == 3))
    assert d1_l3["rolling_lap_time_3_norm"][0] == pytest.approx(90.0 / 95.0)
    assert d2_l3["rolling_lap_time_3_norm"][0] == pytest.approx(100.0 / 95.0)
    # delta_leader = driver rolling - leader (position 1) rolling
    assert d1_l3["rolling_lap_time_3_delta_leader"][0] == pytest.approx(0.0)
    assert d2_l3["rolling_lap_time_3_delta_leader"][0] == pytest.approx(10.0)


def test_rolling_pace_partial_window_uses_available_laps():
    # On lap 1 the rolling mean is just that lap's time (min_periods=1).
    df = pl.DataFrame({
        "driver_number": [1, 2],
        "lap_number": [1, 1],
        "position": [1, 2],
        "lap_time": [90.0, 94.0],
    })
    out = _add_rolling_pace(df)
    d1 = out.filter(pl.col("driver_number") == 1)
    assert d1["rolling_lap_time_3_delta_leader"][0] == pytest.approx(0.0)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_features_unit.py::test_rolling_lap_time_3_norm_and_delta_leader -v`
Expected: `ImportError: cannot import name '_add_rolling_pace'`

- [ ] **Step 3: Implement `_add_rolling_pace`**

Add to `src/f1_predictor/features.py`:

```python
def _add_rolling_pace(df: pl.DataFrame) -> pl.DataFrame:
    """Add rolling_lap_time_3_norm and rolling_lap_time_3_delta_leader.

    rolling3 = mean lap_time over the last 3 laps per driver (min_periods=1).
    _norm divides by the field median of rolling3 at that lap; _delta_leader
    subtracts the rolling3 of the car in position 1 at that lap. Pit/SC laps are
    included as-is (they inflate rolling3); this is acceptable for v1.
    """
    df = df.sort(["driver_number", "lap_number"]).with_columns(
        pl.col("lap_time").rolling_mean(window_size=3, min_periods=1).over("driver_number").alias("_roll3")
    )

    field = (
        df.group_by("lap_number").agg(pl.col("_roll3").median().alias("_field_med"))
    )
    leader = (
        df.filter(pl.col("position") == 1)
        .select(["lap_number", pl.col("_roll3").alias("_leader_roll3")])
        .unique("lap_number")
    )

    return (
        df.join(field, on="lap_number", how="left")
        .join(leader, on="lap_number", how="left")
        .with_columns([
            (pl.col("_roll3") / pl.col("_field_med")).alias("rolling_lap_time_3_norm"),
            (pl.col("_roll3") - pl.col("_leader_roll3")).alias("rolling_lap_time_3_delta_leader"),
        ])
        .drop(["_roll3", "_field_med", "_leader_roll3"])
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_features_unit.py -v`
Expected: 14 passed.

- [ ] **Step 5: Commit**

```bash
git add src/f1_predictor/features.py tests/test_features_unit.py
git commit -m "feat: stage 3 — rolling 3-lap pace (field-normalised + delta to leader)"
```

---

## Task 6: Strategy features — tyre one-hot + stops_vs_median

Five binary tyre columns (`tyre_soft/medium/hard/inter/wet`) and `stops_vs_median` (driver `stops_completed` minus the per-lap median across the active field).

**Files:**
- Modify: `src/f1_predictor/features.py`
- Modify: `tests/test_features_unit.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_features_unit.py`:

```python
from f1_predictor.features import _add_tyre_onehot, _add_stops_vs_median


def test_tyre_onehot_columns():
    df = pl.DataFrame({"tyre_compound": ["SOFT", "MEDIUM", "HARD", "INTER", "WET", None]})
    out = _add_tyre_onehot(df)
    for c in ["tyre_soft", "tyre_medium", "tyre_hard", "tyre_inter", "tyre_wet"]:
        assert c in out.columns
        assert out[c].dtype == pl.Int8
    assert out["tyre_soft"].to_list() == [1, 0, 0, 0, 0, 0]
    assert out["tyre_wet"].to_list() == [0, 0, 0, 0, 1, 0]
    # Unknown/null compound -> all zeros
    assert out.row(5, named=True)["tyre_hard"] == 0


def test_stops_vs_median():
    # One lap, three drivers with stops 0, 1, 2 -> median 1.
    df = pl.DataFrame({
        "lap_number": [5, 5, 5],
        "driver_number": [1, 2, 3],
        "stops_completed": [0, 1, 2],
    })
    out = _add_stops_vs_median(df).sort("driver_number")
    assert out["stops_vs_median"].to_list() == [-1.0, 0.0, 1.0]
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_features_unit.py::test_tyre_onehot_columns -v`
Expected: `ImportError: cannot import name '_add_tyre_onehot'`

- [ ] **Step 3: Implement both functions**

Add to `src/f1_predictor/features.py`:

```python
_TYRE_ONEHOT = {
    "SOFT": "tyre_soft",
    "MEDIUM": "tyre_medium",
    "HARD": "tyre_hard",
    "INTER": "tyre_inter",
    "WET": "tyre_wet",
}


def _add_tyre_onehot(df: pl.DataFrame) -> pl.DataFrame:
    """Five binary tyre columns. Null/unknown compound -> all zeros."""
    return df.with_columns([
        (pl.col("tyre_compound") == compound).fill_null(False).cast(pl.Int8).alias(col)
        for compound, col in _TYRE_ONEHOT.items()
    ])


def _add_stops_vs_median(df: pl.DataFrame) -> pl.DataFrame:
    """stops_completed minus the per-lap median stops across the field."""
    med = df.group_by("lap_number").agg(
        pl.col("stops_completed").median().alias("_med_stops")
    )
    return (
        df.join(med, on="lap_number", how="left")
        .with_columns(
            (pl.col("stops_completed").cast(pl.Float64) - pl.col("_med_stops")).alias("stops_vs_median")
        )
        .drop("_med_stops")
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_features_unit.py -v`
Expected: 16 passed.

- [ ] **Step 5: Commit**

```bash
git add src/f1_predictor/features.py tests/test_features_unit.py
git commit -m "feat: stage 3 — tyre one-hot + stops_vs_median"
```

---

## Task 7: Cross-race priors (DuckDB, no leakage)

Driver/team championship standings entering a race, and driver/team finish rates at the race's circuit — both computed using **only races before the current race**. Standings entering race R = cumulative championship points from all prior races in the season-to-date; finish rate at circuit C entering race R = (finishes / starts) across that driver's (or team's) prior races at circuit C.

This task computes priors from the raw `session_result` + `drivers` + `sessions` endpoints across all races, keyed by `(session_key, driver_number)`, one row per driver-race (not per lap).

**Files:**
- Create: `src/f1_predictor/priors.py`
- Create: `tests/test_priors_unit.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_priors_unit.py
"""Unit tests for cross-race priors with the no-leakage guard."""
import polars as pl
import pytest
from f1_predictor.priors import compute_priors


def _race(session_key, date, circuit, rows):
    """rows: list of (driver_number, team, points, finished[bool])."""
    return [
        {
            "session_key": session_key,
            "date_start": date,
            "circuit_short_name": circuit,
            "driver_number": d,
            "team_name": t,
            "points": p,
            "finished": f,
        }
        for (d, t, p, f) in rows
    ]


def _frame(*races) -> pl.DataFrame:
    rows = [r for race in races for r in race]
    return pl.DataFrame(rows)


def test_championship_standing_uses_only_prior_races():
    data = _frame(
        _race(1, "2023-03-01", "Sakhir", [(1, "RB", 25, True), (44, "MER", 18, True)]),
        _race(2, "2023-03-08", "Jeddah", [(1, "RB", 25, True), (44, "MER", 18, True)]),
        _race(3, "2023-03-15", "Melbourne", [(1, "RB", 25, True), (44, "MER", 18, True)]),
    )
    priors = compute_priors(data)

    # Entering race 1: no prior races -> standing 0 (no points yet).
    r1 = priors.filter((pl.col("session_key") == 1) & (pl.col("driver_number") == 1))
    assert r1["driver_championship_standing"][0] == pytest.approx(0.0)
    # Entering race 3: driver 1 has 25+25 = 50 from races 1 and 2 (NOT race 3).
    r3 = priors.filter((pl.col("session_key") == 3) & (pl.col("driver_number") == 1))
    assert r3["driver_championship_standing"][0] == pytest.approx(50.0)


def test_driver_circuit_finish_rate_prior_only():
    # Driver 1 races Sakhir three times: finishes, DNF, then the current race.
    data = _frame(
        _race(1, "2022-03-01", "Sakhir", [(1, "RB", 25, True)]),
        _race(2, "2023-03-01", "Sakhir", [(1, "RB", 0, False)]),
        _race(3, "2024-03-01", "Sakhir", [(1, "RB", 25, True)]),
    )
    priors = compute_priors(data)
    # Entering race 3 at Sakhir: prior Sakhir races = {finish, DNF} -> rate 0.5.
    r3 = priors.filter((pl.col("session_key") == 3) & (pl.col("driver_number") == 1))
    assert r3["driver_circuit_finish_rate"][0] == pytest.approx(0.5)
    # Entering race 1: no prior Sakhir races -> null.
    r1 = priors.filter((pl.col("session_key") == 1) & (pl.col("driver_number") == 1))
    assert r1["driver_circuit_finish_rate"][0] is None


def test_team_priors_aggregate_both_cars():
    data = _frame(
        _race(1, "2023-03-01", "Sakhir", [(1, "RB", 25, True), (11, "RB", 18, False)]),
        _race(2, "2023-03-08", "Jeddah", [(1, "RB", 25, True), (11, "RB", 18, True)]),
    )
    priors = compute_priors(data)
    # Entering race 2: team RB prior points = 25 + 18 = 43 (both cars, race 1).
    r2 = priors.filter((pl.col("session_key") == 2) & (pl.col("driver_number") == 1))
    assert r2["team_championship_standing"][0] == pytest.approx(43.0)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_priors_unit.py -v`
Expected: `ModuleNotFoundError: No module named 'f1_predictor.priors'`

- [ ] **Step 3: Implement `src/f1_predictor/priors.py`**

```python
"""Stage 3 cross-race priors, computed with a strict no-leakage guard.

All priors for race R use ONLY races whose date_start is before R's. DuckDB does
the cross-race windowing. Input is one row per driver-race; output adds the four
prior columns keyed by (session_key, driver_number).
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import polars as pl

# Columns the input driver-race frame must provide.
_INPUT_COLUMNS = [
    "session_key", "date_start", "circuit_short_name",
    "driver_number", "team_name", "points", "finished",
]


def compute_priors(driver_races: pl.DataFrame) -> pl.DataFrame:
    """Return per (session_key, driver_number) prior features (prior races only).

    Columns added: driver_championship_standing, team_championship_standing,
    driver_circuit_finish_rate, team_circuit_finish_rate.
    Standings are cumulative championship points from prior races (0 if none).
    Finish rates are finishes/starts at the same circuit in prior races (null if
    the driver/team has no prior race at that circuit).
    """
    missing = [c for c in _INPUT_COLUMNS if c not in driver_races.columns]
    if missing:
        raise ValueError(f"driver_races missing columns: {missing}")

    con = duckdb.connect()
    con.register("dr", driver_races.to_arrow())

    query = """
    WITH dr AS (
        SELECT
            session_key,
            CAST(date_start AS TIMESTAMP) AS dt,
            circuit_short_name,
            driver_number,
            team_name,
            CAST(points AS DOUBLE) AS points,
            CAST(finished AS INTEGER) AS finished
        FROM dr
    )
    SELECT
        cur.session_key,
        cur.driver_number,
        -- Driver championship standing: sum points of this driver's prior races.
        COALESCE((
            SELECT SUM(p.points) FROM dr p
            WHERE p.driver_number = cur.driver_number AND p.dt < cur.dt
        ), 0.0) AS driver_championship_standing,
        -- Team championship standing: sum points of both team cars' prior races.
        COALESCE((
            SELECT SUM(p.points) FROM dr p
            WHERE p.team_name = cur.team_name AND p.dt < cur.dt
        ), 0.0) AS team_championship_standing,
        -- Driver circuit finish rate: finishes/starts at this circuit, prior races.
        (
            SELECT AVG(CAST(p.finished AS DOUBLE)) FROM dr p
            WHERE p.driver_number = cur.driver_number
              AND p.circuit_short_name = cur.circuit_short_name
              AND p.dt < cur.dt
        ) AS driver_circuit_finish_rate,
        -- Team circuit finish rate: both cars, this circuit, prior races.
        (
            SELECT AVG(CAST(p.finished AS DOUBLE)) FROM dr p
            WHERE p.team_name = cur.team_name
              AND p.circuit_short_name = cur.circuit_short_name
              AND p.dt < cur.dt
        ) AS team_circuit_finish_rate
    FROM dr cur
    """
    result = con.execute(query).arrow()
    con.close()
    return pl.from_arrow(result)


def build_driver_races(raw_dir: Path, session_keys: list[int]) -> pl.DataFrame:
    """Assemble the one-row-per-driver-race input frame from raw endpoints.

    Reads session_result (points + dnf/dns/dsq), drivers (team_name), and
    sessions (date_start, circuit_short_name) for each session_key.
    finished := not (dnf or dns or dsq).
    """
    frames: list[pl.DataFrame] = []
    for key in session_keys:
        sdir = raw_dir / str(key)
        sr = pl.read_parquet(sdir / "session_result.parquet")
        drv = pl.read_parquet(sdir / "drivers.parquet").select(["driver_number", "team_name"]).unique("driver_number")
        ses = pl.read_parquet(sdir / "sessions.parquet").row(0, named=True)

        finished = ~(
            pl.col("dnf").fill_null(False)
            | pl.col("dns").fill_null(False)
            | pl.col("dsq").fill_null(False)
        )
        frame = (
            sr.with_columns(finished.alias("finished"))
            .join(drv, on="driver_number", how="left")
            .with_columns([
                pl.lit(key).alias("session_key"),
                pl.lit(ses["date_start"]).alias("date_start"),
                pl.lit(ses["circuit_short_name"]).alias("circuit_short_name"),
                pl.col("points").cast(pl.Float64),
            ])
            .select(_INPUT_COLUMNS)
        )
        frames.append(frame)
    return pl.concat(frames, how="vertical")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_priors_unit.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/f1_predictor/priors.py tests/test_priors_unit.py
git commit -m "feat: stage 3 — cross-race driver/team priors (DuckDB, no leakage)"
```

---

## Task 8: Assembly — public `build_features()` + CLI

Wire every per-race transform together, attach the cross-race priors and `is_street_circuit`, read grid from the raw `position` endpoint, and write `data/features/{session_key}.parquet`.

**Files:**
- Modify: `src/f1_predictor/features.py`
- Create: `scripts/build_features.py`
- Modify: `tests/test_features_unit.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_features_unit.py`:

```python
from f1_predictor.features import _grid_from_position, FEATURE_COLUMNS


def test_grid_from_position_takes_earliest_reading():
    # Driver 1's earliest reading is grid P3; later readings are mid-race.
    pos = pl.DataFrame({
        "driver_number": [1, 1, 1, 2, 2],
        "date": [
            "2023-03-05T14:01:00+00:00",  # grid
            "2023-03-05T15:03:45+00:00",  # race
            "2023-03-05T15:10:00+00:00",  # race
            "2023-03-05T14:01:00+00:00",  # grid
            "2023-03-05T15:03:45+00:00",  # race
        ],
        "position": [3, 1, 1, 1, 2],
    })
    grid = _grid_from_position(pos)
    assert grid == {1: 3, 2: 1}


def test_feature_columns_constant_complete():
    # The public column contract must list exactly the 24 features + tyre one-hots.
    for c in ["position", "positions_gained_from_grid", "distance_remaining_km",
              "tyre_soft", "tyre_wet", "stops_vs_median",
              "driver_championship_standing", "team_circuit_finish_rate"]:
        assert c in FEATURE_COLUMNS
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_features_unit.py::test_grid_from_position_takes_earliest_reading -v`
Expected: `ImportError: cannot import name '_grid_from_position'`

- [ ] **Step 3: Implement grid extraction, `FEATURE_COLUMNS`, and `build_features()`**

Add to `src/f1_predictor/features.py`:

```python
from f1_predictor.priors import build_driver_races, compute_priors

FEATURE_COLUMNS = [
    "position", "positions_gained_from_grid", "num_active_drivers",
    "distance_remaining_km", "gap_to_leader", "interval_to_ahead",
    "rolling_lap_time_3_norm", "rolling_lap_time_3_delta_leader",
    "last_lap_pace_delta_to_ahead", "last_lap_pace_delta_to_behind",
    "mean_gap_cars_ahead", "stdev_gap_cars_ahead", "max_speed_kmh",
    "tyre_soft", "tyre_medium", "tyre_hard", "tyre_inter", "tyre_wet",
    "tyre_age_laps", "stint_number", "stops_vs_median",
    "sc_active", "vsc_active", "red_flag_active", "laps_since_sc_end",
    "is_street_circuit",
    "driver_circuit_finish_rate", "driver_championship_standing",
    "team_circuit_finish_rate", "team_championship_standing",
]

_KEY_COLUMNS = ["session_key", "driver_number", "lap_number", "final_position"]


def _grid_from_position(pos_df: pl.DataFrame) -> dict[int, int]:
    """Grid position per driver = position at the earliest reading (pre-race)."""
    earliest = (
        pos_df.sort("date")
        .group_by("driver_number", maintain_order=True)
        .agg(pl.col("position").first().alias("grid"))
    )
    return dict(zip(earliest["driver_number"].to_list(), earliest["grid"].to_list()))


def build_features(
    session_key: int,
    sessions_dir: Path,
    raw_dir: Path,
    features_dir: Path,
    priors: pl.DataFrame,
    circuits: dict | None = None,
) -> pl.DataFrame:
    """Engineer all Stage 3 features for one race and write the parquet.

    `priors` is the cross-race prior table from compute_priors() (one row per
    (session_key, driver_number)); it is computed once for all races by the CLI.
    Returns the feature DataFrame.
    """
    circuits = circuits or load_circuits()
    df = pl.read_parquet(sessions_dir / f"{session_key}.parquet")

    ses = pl.read_parquet(raw_dir / str(session_key) / "sessions.parquet").row(0, named=True)
    circuit = ses["circuit_short_name"]
    pos_raw = pl.read_parquet(raw_dir / str(session_key) / "position.parquet")
    grid = _grid_from_position(pos_raw)

    df = _parse_gap_columns(df)
    df = _add_active_and_distance(df, circuit_length_km(circuit, circuits))
    df = _add_positions_gained(df, grid)
    df = _add_pace_deltas(df)
    df = _add_gaps_ahead(df)
    df = _add_rolling_pace(df)
    df = _add_tyre_onehot(df)
    df = _add_stops_vs_median(df)
    df = df.with_columns(pl.lit(is_street_circuit(circuit, circuits)).alias("is_street_circuit"))

    race_priors = priors.filter(pl.col("session_key") == session_key).drop("session_key")
    df = df.join(race_priors, on="driver_number", how="left")

    out = df.select(_KEY_COLUMNS + FEATURE_COLUMNS)

    features_dir.mkdir(parents=True, exist_ok=True)
    out.write_parquet(features_dir / f"{session_key}.parquet")
    return out
```

- [ ] **Step 4: Create `scripts/build_features.py`**

```python
"""CLI: build Stage 3 features for one race or all sessionised races."""
from pathlib import Path

import polars as pl
import typer

from f1_predictor.features import build_features, load_circuits
from f1_predictor.priors import build_driver_races, compute_priors

app = typer.Typer(add_completion=False)


@app.command()
def main(
    session_key: int = typer.Option(None, "--session-key", help="One race; omit for all"),
    sessions_dir: Path = typer.Option(Path("data/sessions"), "--sessions-dir"),
    raw_dir: Path = typer.Option(Path("data/raw"), "--raw-dir"),
    features_dir: Path = typer.Option(Path("data/features"), "--features-dir"),
) -> None:
    keys = sorted(int(p.stem) for p in sessions_dir.glob("*.parquet") if "_masks" not in p.stem)
    if not keys:
        typer.echo("No sessionised races found.", err=True)
        raise SystemExit(1)

    # Priors are computed once across ALL races (the SQL enforces prior-only).
    driver_races = build_driver_races(raw_dir, keys)
    priors = compute_priors(driver_races)
    circuits = load_circuits()

    targets = [session_key] if session_key is not None else keys
    for key in targets:
        out = build_features(key, sessions_dir, raw_dir, features_dir, priors, circuits)
        typer.echo(f"  {key}: {out.shape[0]} rows, {out.shape[1]} cols")
    typer.echo(f"Built features for {len(targets)} race(s)")


if __name__ == "__main__":
    app()
```

- [ ] **Step 5: Run unit tests + CLI help**

Run: `uv run pytest tests/test_features_unit.py -v`
Expected: 18 passed.
Run: `uv run python scripts/build_features.py --help`
Expected: shows `--session-key`, `--sessions-dir`, `--raw-dir`, `--features-dir`.

- [ ] **Step 6: Commit**

```bash
git add src/f1_predictor/features.py scripts/build_features.py tests/test_features_unit.py
git commit -m "feat: stage 3 — build_features assembly + CLI"
```

---

## Task 9: Integration tests on real fixtures

Build features for the three Plan 1 fixtures and assert structural + known-fact invariants, including the leakage guard end-to-end.

**Files:**
- Create: `tests/test_features.py`

- [ ] **Step 1: Pre-req — build features for the fixtures (run once)**

These commands need the 2023 raw + sessionised data from Plan 1 on disk.

```bash
uv run python scripts/build_features.py --session-key 9078
uv run python scripts/build_features.py --session-key 9070
uv run python scripts/build_features.py --session-key 9181
```

Expected: each prints a row/col count and writes `data/features/{key}.parquet`.

- [ ] **Step 2: Write the integration tests**

```python
# tests/test_features.py
"""Stage 3 integration tests on the three real 2023 fixtures."""
from pathlib import Path

import polars as pl
import pytest

from f1_predictor.features import FEATURE_COLUMNS
from tests.conftest import FIXTURE_SC, FIXTURE_RETIREMENTS, FIXTURE_CLEAN

FEATURES_DIR = Path("data") / "features"


def _load_or_skip(session_key: int) -> pl.DataFrame:
    path = FEATURES_DIR / f"{session_key}.parquet"
    if not path.exists():
        pytest.skip(f"Features for {session_key} not built. Run: "
                    f"uv run python scripts/build_features.py --session-key {session_key}")
    return pl.read_parquet(path)


@pytest.fixture(scope="session")
def feat_sc():
    return _load_or_skip(FIXTURE_SC)


@pytest.fixture(scope="session")
def feat_clean():
    return _load_or_skip(FIXTURE_CLEAN)


@pytest.mark.parametrize("fixture_name", ["feat_sc", "feat_clean"])
def test_all_feature_columns_present(fixture_name, request):
    df = request.getfixturevalue(fixture_name)
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    assert not missing, f"Missing feature columns: {missing}"


@pytest.mark.parametrize("fixture_name", ["feat_sc", "feat_clean"])
def test_one_row_per_driver_lap(fixture_name, request):
    df = request.getfixturevalue(fixture_name)
    dupes = df.filter(df.select(["driver_number", "lap_number"]).is_duplicated())
    assert dupes.is_empty()


def test_gaps_are_numeric(feat_clean):
    assert feat_clean["gap_to_leader"].dtype == pl.Float64
    assert feat_clean["interval_to_ahead"].dtype == pl.Float64


def test_leader_has_zero_gaps_ahead(feat_clean):
    leader = feat_clean.filter(pl.col("position") == 1)
    assert (leader["mean_gap_cars_ahead"].fill_null(0) == 0).all()
    assert (leader["stdev_gap_cars_ahead"].fill_null(0) == 0).all()


def test_tyre_onehot_sums_to_at_most_one(feat_clean):
    onehot = feat_clean.select(["tyre_soft", "tyre_medium", "tyre_hard", "tyre_inter", "tyre_wet"])
    row_sums = onehot.sum_horizontal()
    assert (row_sums <= 1).all()


def test_clean_fixture_is_street_circuit_true(feat_clean):
    # FIXTURE_CLEAN is Miami (a street circuit).
    assert feat_clean["is_street_circuit"].unique().to_list() == [True]


def test_distance_remaining_decreases_over_laps(feat_clean):
    d = (
        feat_clean.filter(pl.col("driver_number") == feat_clean["driver_number"][0])
        .sort("lap_number")
    )
    diffs = d["distance_remaining_km"].diff().drop_nulls()
    assert (diffs <= 0).all(), "distance_remaining_km must be non-increasing within a race"


def test_first_2023_race_has_null_priors():
    # The earliest 2023 race has no prior races, so championship standing is 0
    # and circuit finish rate is null (no prior race at that circuit).
    import json
    raw = Path("data/raw")
    keys = [int(p.name) for p in raw.iterdir() if p.is_dir() and (p / "sessions.parquet").exists()]
    dated = []
    for k in keys:
        ses = pl.read_parquet(raw / str(k) / "sessions.parquet").row(0, named=True)
        dated.append((ses["date_start"], k))
    first_key = min(dated)[1]
    path = FEATURES_DIR / f"{first_key}.parquet"
    if not path.exists():
        pytest.skip(f"Features for first race {first_key} not built")
    df = pl.read_parquet(path)
    assert (df["driver_championship_standing"] == 0).all()
    assert df["driver_circuit_finish_rate"].null_count() == df.height
```

- [ ] **Step 3: Build features for the earliest race too, then run**

```bash
uv run python scripts/build_features.py        # builds all races (covers the first-race prior test)
uv run pytest tests/test_features.py -v
```

Expected: all tests pass (or skip if data absent).

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest tests/ -v`
Expected: all tests pass (Plan 1's 77 + Stage 3 unit + integration).

- [ ] **Step 5: Commit**

```bash
git add tests/test_features.py
git commit -m "test: stage 3 integration tests on real fixtures + leakage guard"
```

---

## Self-Review

### 1. Spec coverage

| Spec feature | Task |
|---|---|
| `position` | Task 8 (passthrough via select) |
| `positions_gained_from_grid` | Task 4 |
| `num_active_drivers` | Task 3 |
| `distance_remaining_km` (raw km, circuit lookup) | Task 1 + 3 |
| `gap_to_leader`, `interval_to_ahead` (numeric) | Task 2 |
| `rolling_lap_time_3_norm`, `rolling_lap_time_3_delta_leader` | Task 5 |
| `last_lap_pace_delta_to_ahead/behind` | Task 4 |
| `mean_gap_cars_ahead`, `stdev_gap_cars_ahead` | Task 4 |
| `max_speed_kmh` | Task 8 (passthrough; null until car_data backfill) |
| `tyre_compound` one-hot (5 cols) | Task 6 |
| `tyre_age_laps`, `stint_number` | Task 8 (passthrough) |
| `stops_vs_median` | Task 6 |
| `sc_active`, `vsc_active`, `red_flag_active`, `laps_since_sc_end` | Task 8 (passthrough) |
| `is_street_circuit` | Task 1 + 8 |
| `driver_circuit_finish_rate`, `team_circuit_finish_rate` | Task 7 |
| `driver_championship_standing`, `team_championship_standing` | Task 7 |
| Leakage guard (prior races only) | Task 7 (SQL `dt < cur.dt`) + Task 9 |

Note: `position`, `tyre_age_laps`, `stint_number`, the SC flags, and `max_speed_kmh` are carried straight from the Stage 2 table by the `select(_KEY_COLUMNS + FEATURE_COLUMNS)` in `build_features`; they need no transform task. Verify they exist in the Stage 2 output (they do, per Plan 1's schema).

### 2. Placeholder scan

No "TBD"/"add error handling"/"similar to Task N" placeholders. Circuit lengths are concrete (flagged for FIA re-verification). All code blocks are complete.

### 3. Type consistency

- `build_features(session_key, sessions_dir, raw_dir, features_dir, priors, circuits=None)` — defined Task 8, matches the CLI call.
- `compute_priors(driver_races) -> pl.DataFrame` and `build_driver_races(raw_dir, session_keys)` — defined Task 7, used in Task 8 CLI.
- Prior columns (`driver_championship_standing`, `team_championship_standing`, `driver_circuit_finish_rate`, `team_circuit_finish_rate`) — same names in Task 7 SQL, `FEATURE_COLUMNS`, and Task 9 assertions.
- `_grid_from_position` returns `dict[int, int]`; `_add_positions_gained(df, grid)` consumes that shape.
- Tyre one-hot column names (`tyre_soft/medium/hard/inter/wet`) consistent across Task 6 and `FEATURE_COLUMNS`.

### Open items for the executor to confirm

- **Circuit lengths**: re-verify `config/circuits.yaml` values against official FIA track data before training.
- **`rolling_lap_time_3_norm` definition**: implemented as a ratio to the field median (1.0 = median pace). If a subtractive delta is preferred, adjust Task 5 and its test together.
- **Pit/SC laps in rolling pace**: included as-is (they inflate lap_time). Revisit if they distort the feature during error analysis.
