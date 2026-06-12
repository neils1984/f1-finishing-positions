"""Stage-0 regime-drift diagnostic for the 2026 technical regulations.

Quantifies how far the cross-era LightGBM baseline degrades when it is asked to
predict the 2026 regime it never trained on. The question this answers is
*architectural*: do we need a separate / fine-tuned 2026 model, or does the
existing single model hold up?

It reports, per split (val = same-regime reference, test = the 2026 drift probe)
and per snapshot lap:

* naive persistence Spearman  (does the *grid/pace order itself* carry over?)
* trained-model Spearman      (does the *learned* mapping carry over?)
* uplift = model - naive      (does the model still add value, or has its
                               learned signal gone stale / actively harmful?)

If naive holds up on test but the model's uplift collapses, the learned
feature->finish relationships have shifted (concept drift) — the case for a
2026-specific model. If naive *itself* collapses, the competitive order was
reshuffled (covariate drift in the priors) and the inputs need rethinking
before any model can recover.

It also flags **inert features**: columns that are constant across the train
split and therefore carry no signal the model could have learned. `is_2026_regs`
is the important one — when no training race is from 2026 it is constant 0 in
train, so LightGBM can never split on it. That makes the pipeline's only
regime-awareness hook structurally dead, which is itself a key finding.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import polars as pl

from f1_predictor.evaluate import ranking_metrics
from f1_predictor.models.baseline_gbm import naive_predict, predict, train_baseline

_META = ["session_key", "snapshot_lap", "driver_number", "final_position"]

# A train-split feature whose variance is below this is treated as constant and
# therefore inert: LightGBM cannot find a split on it, so it carries no signal.
_INERT_VAR_EPS = 1e-12


def _preds_frame(df: pl.DataFrame, scores: np.ndarray) -> pl.DataFrame:
    """Attach a score column to the metadata columns ranking_metrics expects."""
    return df.select(_META).with_columns(pl.Series("score", scores))


def _per_lap(preds: pl.DataFrame) -> list[dict]:
    """Per-snapshot-lap ranking metrics, sorted by lap."""
    out = []
    for lap in sorted(preds["snapshot_lap"].unique().to_list()):
        m = ranking_metrics(preds.filter(pl.col("snapshot_lap") == lap))
        out.append({"snapshot_lap": int(lap), **m})
    return out


def _split_report(
    df: pl.DataFrame, model, feature_columns: list[str]
) -> dict:
    """Naive + model metrics (overall and per-lap) with uplift, for one split."""
    naive = _preds_frame(df, naive_predict(df))
    model_p = _preds_frame(df, predict(model, df, feature_columns))

    naive_overall = ranking_metrics(naive)
    model_overall = ranking_metrics(model_p)

    naive_laps = {d["snapshot_lap"]: d for d in _per_lap(naive)}
    model_laps = {d["snapshot_lap"]: d for d in _per_lap(model_p)}
    per_lap = [
        {
            "snapshot_lap": lap,
            "naive_spearman": naive_laps[lap]["spearman"],
            "model_spearman": model_laps[lap]["spearman"],
            "uplift": model_laps[lap]["spearman"] - naive_laps[lap]["spearman"],
            "n_groups": model_laps[lap]["n_groups"],
        }
        for lap in sorted(model_laps)
    ]

    return {
        "n_rows": df.height,
        "n_races": df["session_key"].n_unique(),
        "naive": naive_overall,
        "model": model_overall,
        "uplift": model_overall["spearman"] - naive_overall["spearman"],
        "per_lap": per_lap,
    }


def inert_features(train: pl.DataFrame, feature_columns: list[str]) -> list[str]:
    """Feature columns that are constant across the train split (zero signal).

    Snapshots store scaled values, but a feature constant in train (e.g.
    `is_2026_regs` when no training race is from 2026) stays constant after
    scaling, so a near-zero variance is the reliable test.
    """
    x = train.select(feature_columns).to_numpy().astype(float)
    var = np.nanvar(x, axis=0)
    return [c for c, v in zip(feature_columns, var) if v < _INERT_VAR_EPS]


def degradation_report(
    snapshots_dir: Path,
    params: dict | None = None,
) -> dict:
    """Train on train.parquet and contrast val (same-regime) vs test (2026).

    Returns a structured report; does not write anything. Splits that are empty
    or absent are skipped. The model is trained once on the train split (with val
    as a monitoring set when present), matching the production baseline.
    """
    meta = json.loads((snapshots_dir / "metadata.json").read_text())
    feature_columns = meta["feature_columns"]

    train = pl.read_parquet(snapshots_dir / "train.parquet")
    if train.is_empty():
        raise ValueError("train.parquet is empty — nothing to train on.")

    def _load(split: str) -> pl.DataFrame | None:
        p = snapshots_dir / f"{split}.parquet"
        if not p.exists():
            return None
        df = pl.read_parquet(p)
        return None if df.is_empty() else df

    val = _load("val")
    test = _load("test")

    model = train_baseline(train, feature_columns, params=params, valid=val)

    splits: dict[str, dict] = {}
    if val is not None:
        splits["val"] = _split_report(val, model, feature_columns)
    if test is not None:
        splits["test"] = _split_report(test, model, feature_columns)

    # Regime-awareness diagnostics.
    inert = inert_features(train, feature_columns)
    gain = model.feature_importance(importance_type="gain")
    importance = {c: float(g) for c, g in zip(feature_columns, gain)}

    report: dict = {
        "feature_columns": feature_columns,
        "n_train_races": train["session_key"].n_unique(),
        "splits": splits,
        "inert_train_features": inert,
        "is_2026_regs_inert": "is_2026_regs" in inert
        if "is_2026_regs" in feature_columns
        else None,
        "is_2026_regs_gain_importance": importance.get("is_2026_regs"),
        "feature_importance_gain": dict(
            sorted(importance.items(), key=lambda kv: kv[1], reverse=True)
        ),
    }

    # Headline: how much uplift survived from val (same regime) to test (2026).
    if "val" in splits and "test" in splits:
        report["val_to_test"] = {
            "naive_spearman_drop": splits["val"]["naive"]["spearman"]
            - splits["test"]["naive"]["spearman"],
            "model_spearman_drop": splits["val"]["model"]["spearman"]
            - splits["test"]["model"]["spearman"],
            "uplift_val": splits["val"]["uplift"],
            "uplift_test": splits["test"]["uplift"],
            "uplift_retained": splits["test"]["uplift"] - splits["val"]["uplift"],
        }
    return report


def write_report(report: dict, runs_dir: Path) -> Path:
    """Persist the report JSON under runs/degradation-<ts>/ and return the dir."""
    run_dir = runs_dir / f"degradation-{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "degradation.json").write_text(json.dumps(report, indent=2))
    return run_dir
