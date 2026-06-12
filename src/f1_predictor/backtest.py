"""Walk-forward (expanding-window) backtest over a chronological set of races.

For each target race, train on every race that started strictly earlier (the
scaler is refit on that past only, so there is no temporal leak) and predict the
target. Matches the real use case: predict the season as it unfolds. Operates on
raw Stage 3 feature tables in memory; reuses the Stage 4 snapshot extraction and
scaler so features are treated identically to the production pipeline.
"""
from __future__ import annotations

import polars as pl

from f1_predictor.evaluate import ranking_metrics
from f1_predictor.models.baseline_gbm import (
    naive_predict, predict, train_baseline,
)
from f1_predictor.snapshots import apply_scaler, extract_snapshots, fit_scaler


def _race_date(race: pl.DataFrame) -> str:
    return race["date_start"][0]


def walk_forward(
    races: dict[int, pl.DataFrame],
    target_keys: list[int],
    feature_columns: list[str],
    snapshot_laps: list[int],
    params: dict | None = None,
    sample_weight_fn=None,
    blend_alpha: float | None = None,
) -> pl.DataFrame:
    """Backtest each target race against an expanding window of earlier races.

    races: {session_key: raw feature DataFrame} (must include date_start,
        lap_number, position, final_position, and feature_columns).
    target_keys: races to evaluate, evaluated chronologically.
    sample_weight_fn: optional callable(train_snapshots_df) -> np.ndarray of
        per-row weights (e.g. recency / regime upweighting). None = uniform.
    blend_alpha: if set, final score = (1-alpha)*naive + alpha*model (both are
        ~ -finish_position so they share a scale). None = pure model.

    Returns one row per target with model_spearman / naive_spearman and the
    other ranking metrics for the model.
    """
    # naive_predict / add_current_rank rank by `position`, so it must survive
    # snapshot extraction even when it is not one of the model's feature_columns
    # (it always is in production, but the engine should not rely on that).
    extract_cols = (
        feature_columns if "position" in feature_columns
        else [*feature_columns, "position"]
    )
    snaps = {
        k: extract_snapshots(df, snapshot_laps, extract_cols).with_columns([
            pl.lit(_race_date(df)).alias("date_start"),
            pl.lit(int(_race_date(df)[:4])).alias("season"),
        ])
        for k, df in races.items()
    }
    out_rows = []
    for tkey in sorted(target_keys, key=lambda k: _race_date(races[k])):
        tdate = _race_date(races[tkey])
        train_frames = [s for k, s in snaps.items() if _race_date(races[k]) < tdate]
        if not train_frames:
            continue
        train = pl.concat(train_frames, how="vertical_relaxed")
        target = snaps[tkey]

        scaler = fit_scaler(train, feature_columns)
        train_s = apply_scaler(train, scaler, feature_columns)
        target_s = apply_scaler(target, scaler, feature_columns)

        weights = sample_weight_fn(train) if sample_weight_fn is not None else None
        model = train_baseline(train_s, feature_columns, params=params, sample_weight=weights)

        model_score = predict(model, target_s, feature_columns)
        naive_score = naive_predict(target_s)
        score = model_score if blend_alpha is None else (
            (1.0 - blend_alpha) * naive_score + blend_alpha * model_score
        )

        meta = target_s.select(["session_key", "snapshot_lap", "driver_number", "final_position"])
        model_m = ranking_metrics(meta.with_columns(pl.Series("score", score)))
        naive_m = ranking_metrics(meta.with_columns(pl.Series("score", naive_score)))
        out_rows.append({
            "session_key": tkey, "date_start": tdate, "n_train_races": len(train_frames),
            "model_spearman": model_m["spearman"], "naive_spearman": naive_m["spearman"],
            "model_top1": model_m["top1_accuracy"], "model_top3": model_m["top3_accuracy"],
            "model_mpe": model_m["mean_position_error"], "n_groups": model_m["n_groups"],
        })
    return pl.DataFrame(out_rows)
