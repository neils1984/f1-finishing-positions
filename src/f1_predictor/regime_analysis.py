"""Descriptive stats explaining how persistent a set of races is.

Higher naive-persistence Spearman is driven by (a) fewer retirements and
(b) a tighter spread of position changes from race-state to finish. These
helpers quantify both so per-season differences (e.g. 2026 vs 2024) are
explainable rather than mysterious.
"""
from __future__ import annotations

import polars as pl


def position_change_stats(features: pl.DataFrame, snapshot_lap: int) -> dict:
    """DNF rate and mean absolute position change at a given snapshot lap.

    `features` is the raw Stage 3 feature table(s) for one or more races, with
    columns session_key, lap_number, position, final_position, dnf. Rows are
    filtered to lap_number == snapshot_lap. mean_abs_change is computed per race
    then averaged across races (so a single chaotic race doesn't dominate by
    driver count).
    """
    snap = features.filter(pl.col("lap_number") == snapshot_lap)
    per_race = (
        snap.with_columns((pl.col("final_position") - pl.col("position")).abs().alias("abs_change"))
        .group_by("session_key")
        .agg(pl.col("abs_change").mean().alias("race_mean_abs_change"))
    )
    return {
        "mean_abs_change": per_race["race_mean_abs_change"].mean(),
        "dnf_rate": float(snap["dnf"].mean()) if "dnf" in snap.columns else float("nan"),
        "n_races": snap["session_key"].n_unique(),
    }
