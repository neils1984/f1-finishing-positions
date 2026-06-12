"""Train the cross-driver Transformer (delta-regression) and persist a run dir.

Trains on the `train` snapshot split, early-stops on the **val** split's Spearman
(the eval metric), and writes runs/{run_id}/ with the model, val predictions, and
metrics. The 2026 `test` split is never touched here — this is a same-regime
architecture comparison against the LightGBM baseline (see compare_val.py).

Scoring mirrors the baseline exactly: score = predicted_delta - current_rank
(== -predicted_final_position), so a higher score means a better finish.
"""
from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path

import polars as pl
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from f1_predictor.data_loader import (
    SnapshotDataset,
    build_driver_index,
    load_metadata,
    load_split,
)
from f1_predictor.evaluate import ranking_metrics
from f1_predictor.losses import masked_l1_loss
from f1_predictor.models.transformer import DriverDeltaNet


def _warmup_cosine(step: int, warmup: int, total: int) -> float:
    """Linear warmup to 1.0 over `warmup` steps, then cosine decay to 0."""
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def _predict_split(model, dataset, device) -> pl.DataFrame:
    """Score every valid driver slot: score = delta_hat - current_rank.

    Returns the column set ranking_metrics expects:
    session_key, snapshot_lap, driver_number, final_position, score.
    """
    model.eval()
    rows = []
    loader = DataLoader(dataset, batch_size=16)
    with torch.no_grad():
        for batch in loader:
            delta_hat = model(
                batch["features"].to(device),
                batch["driver_idx"].to(device),
                batch["valid"].to(device),
            ).cpu()
            score = delta_hat - batch["current_rank"]
            B, N = score.shape
            for b in range(B):
                for s in range(N):
                    if not bool(batch["valid"][b, s]):
                        continue
                    rows.append({
                        "session_key": int(batch["session_key"][b]),
                        "snapshot_lap": int(batch["snapshot_lap"][b]),
                        "driver_number": int(batch["driver_number"][b, s]),
                        "final_position": int(batch["final_position"][b, s]),
                        "score": float(score[b, s]),
                    })
    return pl.DataFrame(rows)


def train_transformer(
    snapshots_dir: Path,
    runs_dir: Path,
    config: dict,
    use_mlflow: bool = True,
) -> dict:
    """Train on `train`, early-stop on `val`, persist runs/{run_id}/.

    Returns {"run_dir": str, "metrics": {...}} where metrics are the best val
    metrics. `test` is intentionally never loaded.
    """
    meta = load_metadata(snapshots_dir)
    feature_columns = meta["feature_columns"]
    num_slots = config["num_slots"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df = load_split(snapshots_dir, "train")
    val_df = load_split(snapshots_dir, "val")
    driver_index = build_driver_index(train_df, config["num_drivers"])

    train_ds = SnapshotDataset(train_df, feature_columns, driver_index, num_slots)
    val_ds = SnapshotDataset(val_df, feature_columns, driver_index, num_slots)
    loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)

    model = DriverDeltaNet(
        num_features=len(feature_columns),
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        dropout=config.get("dropout", 0.1),
        num_drivers=config["num_drivers"],
    ).to(device)

    opt = AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config.get("weight_decay", 0.01),
    )
    total_steps = max(config["epochs"] * len(loader), 1)
    sched = LambdaLR(
        opt, lambda s: _warmup_cosine(s, config["warmup_steps"], total_steps)
    )

    patience = config.get("patience", 10)
    best_spearman = float("-inf")
    best_state = copy.deepcopy(model.state_dict())
    best_metrics: dict = {}
    epochs_since_best = 0

    for _ in range(config["epochs"]):
        model.train()
        for batch in loader:
            opt.zero_grad()
            delta_hat = model(
                batch["features"].to(device),
                batch["driver_idx"].to(device),
                batch["valid"].to(device),
            )
            loss = masked_l1_loss(
                delta_hat,
                batch["delta"].to(device),
                batch["valid"].to(device),
            )
            loss.backward()
            opt.step()
            sched.step()

        val_metrics = ranking_metrics(_predict_split(model, val_ds, device))
        if val_metrics["spearman"] > best_spearman:
            best_spearman = val_metrics["spearman"]
            best_metrics = val_metrics
            best_state = copy.deepcopy(model.state_dict())
            epochs_since_best = 0
        else:
            epochs_since_best += 1
            if epochs_since_best >= patience:
                break

    model.load_state_dict(best_state)
    val_preds = _predict_split(model, val_ds, device)

    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), run_dir / "model.pt")
    val_preds.write_parquet(run_dir / "predictions_val.parquet")
    (run_dir / "metrics.json").write_text(json.dumps(best_metrics, indent=2))
    (run_dir / "config.json").write_text(json.dumps({
        "model": "cross_driver_transformer_delta_l1",
        **{k: v for k, v in config.items()},
        "feature_columns": feature_columns,
        "driver_index": {str(k): v for k, v in driver_index.items()},
        "data_version": meta.get("data_version"),
        "eval_split": "val",
    }, indent=2))

    if use_mlflow:
        import mlflow
        mlflow.set_experiment("f1-transformer")
        with mlflow.start_run(run_name=run_id):
            mlflow.log_params({
                k: v for k, v in config.items()
                if not isinstance(v, (list, dict))
            })
            mlflow.log_metrics(best_metrics)
            mlflow.log_artifacts(str(run_dir))

    return {"run_dir": str(run_dir), "metrics": best_metrics}
