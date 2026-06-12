"""CLI: head-to-head on the pre-2026 `val` split.

Reports Transformer vs LightGBM-baseline vs naive-persistence, all trained on the
same `train` split and scored on `val`, via the same ranking_metrics. Answers:
does cross-driver attention beat the per-row GBM on a single (2023-2025) regime?
The 2026 `test` split is never loaded.
"""
import json
from pathlib import Path

import polars as pl
import torch
import typer

from f1_predictor.data_loader import (
    SnapshotDataset,
    load_metadata,
    load_split,
)
from f1_predictor.evaluate import ranking_metrics
from f1_predictor.models.baseline_gbm import naive_predict, predict, train_baseline
from f1_predictor.models.transformer import DriverDeltaNet
from f1_predictor.train import _predict_split

app = typer.Typer(add_completion=False)


def _metrics_frame(df: pl.DataFrame, scores) -> dict:
    preds = df.select(
        ["session_key", "snapshot_lap", "driver_number", "final_position"]
    ).with_columns(pl.Series("score", scores))
    return ranking_metrics(preds)


def _transformer_metrics(run_dir: Path, val_df: pl.DataFrame, device) -> dict:
    cfg = json.loads((run_dir / "config.json").read_text())
    feature_columns = cfg["feature_columns"]
    driver_index = {int(k): v for k, v in cfg["driver_index"].items()}
    model = DriverDeltaNet(
        num_features=len(feature_columns),
        d_model=cfg["d_model"], n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"], dropout=cfg.get("dropout", 0.1),
        num_drivers=cfg["num_drivers"],
    ).to(device)
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location=device))
    val_ds = SnapshotDataset(
        val_df, feature_columns, driver_index, cfg["num_slots"]
    )
    return ranking_metrics(_predict_split(model, val_ds, device))


@app.command()
def main(
    run_dir: Path = typer.Option(..., "--run-dir", help="Transformer run dir"),
    snapshots_dir: Path = typer.Option(Path("data/snapshots"), "--snapshots-dir"),
    no_mlflow: bool = typer.Option(False, "--no-mlflow"),
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta = load_metadata(snapshots_dir)
    feature_columns = meta["feature_columns"]
    train_df = load_split(snapshots_dir, "train")
    val_df = load_split(snapshots_dir, "val")

    tf = _transformer_metrics(run_dir, val_df, device)

    gbm = train_baseline(train_df, feature_columns)
    gbm_m = _metrics_frame(val_df, predict(gbm, val_df, feature_columns))
    naive_m = _metrics_frame(val_df, naive_predict(val_df))

    keys = ["spearman", "top1_accuracy", "top3_accuracy",
            "mean_position_error", "n_groups"]
    typer.echo(f"\n{'model':<14}" + "".join(f"{k:>20}" for k in keys))
    for name, m in [("transformer", tf), ("lightgbm", gbm_m), ("naive", naive_m)]:
        typer.echo(f"{name:<14}" + "".join(f"{m[k]:>20.4f}" for k in keys))

    beats_gbm = tf["spearman"] > gbm_m["spearman"]
    beats_naive = tf["spearman"] > naive_m["spearman"]
    typer.echo(
        f"\nTransformer val Spearman {tf['spearman']:.4f} — "
        f"beats GBM: {beats_gbm} ({gbm_m['spearman']:.4f}), "
        f"beats naive: {beats_naive} ({naive_m['spearman']:.4f})"
    )

    if not no_mlflow:
        import mlflow
        mlflow.set_experiment("f1-transformer-vs-baseline")
        with mlflow.start_run(run_name="val-head-to-head"):
            mlflow.log_metric("transformer_spearman", tf["spearman"])
            mlflow.log_metric("gbm_spearman", gbm_m["spearman"])
            mlflow.log_metric("naive_spearman", naive_m["spearman"])


if __name__ == "__main__":
    app()
