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
