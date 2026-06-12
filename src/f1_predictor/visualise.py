"""Lap-by-lap prediction-evolution plot: predicted rank per driver vs lap."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402
import polars as pl  # noqa: E402


def plot_prediction_evolution(
    predictions: pl.DataFrame, session_key: int, out_path: Path
) -> None:
    """Plot each driver's predicted rank across snapshot laps for one race.

    predictions: session_key, snapshot_lap, driver_number, final_position, score.
    Predicted rank within a (race, lap) = descending score (1 = predicted winner).
    """
    race = predictions.filter(pl.col("session_key") == session_key)
    race = race.with_columns(
        pl.col("score")
        .rank("ordinal", descending=True)
        .over("snapshot_lap")
        .alias("pred_rank")
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    for d, grp in race.group_by("driver_number", maintain_order=True):
        grp = grp.sort("snapshot_lap")
        num = d[0] if isinstance(d, tuple) else d
        ax.plot(grp["snapshot_lap"].to_list(), grp["pred_rank"].to_list(),
                marker="o", label=f"#{num}")
    ax.invert_yaxis()  # rank 1 at top
    ax.set_xlabel("snapshot lap")
    ax.set_ylabel("predicted rank")
    ax.set_title(f"Prediction evolution — session {session_key}")
    ax.legend(loc="best", fontsize="x-small", ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
