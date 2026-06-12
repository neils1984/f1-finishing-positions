"""Integration test: the Transformer training loop overfits a tiny learnable set.

The signal is constructed so the true delta target is leaked in a feature `f0`
(delta = current_rank - final_position = f0), making it trivially regressable.
A model that fits it reconstructs score = delta_hat - current_rank ==
-final_position, so the val Spearman must approach 1.
"""
import json
from pathlib import Path

import polars as pl

from f1_predictor.train import train_transformer


def _write_snapshots(snap_dir: Path, splits: dict[str, tuple[int, int]]) -> None:
    snap_dir.mkdir(parents=True, exist_ok=True)
    feature_columns = ["position", "f0", "f1"]
    n = 5  # drivers per race
    for split, (n_races, start) in splits.items():
        rows = []
        for r in range(start, start + n_races):
            for d in range(1, n + 1):
                final_position = n + 1 - d           # reverse current order
                delta = d - final_position           # = 2d - (n+1)
                rows.append({
                    "session_key": r, "snapshot_lap": 30, "driver_number": d,
                    "final_position": final_position,
                    "position": float(d),            # current_rank == d
                    "f0": float(delta),              # leaked target
                    "f1": 0.0,
                })
        pl.DataFrame(rows).write_parquet(snap_dir / f"{split}.parquet")
    (snap_dir / "metadata.json").write_text(
        json.dumps({"feature_columns": feature_columns, "data_version": "test"})
    )


def test_train_transformer_overfits_and_writes_run(tmp_path):
    snap_dir = tmp_path / "snapshots"
    _write_snapshots(snap_dir, {"train": (12, 0), "val": (3, 100), "test": (3, 200)})
    runs_dir = tmp_path / "runs"

    result = train_transformer(
        snap_dir, runs_dir,
        config={"d_model": 32, "n_heads": 4, "n_layers": 2, "epochs": 60,
                "lr": 1e-3, "warmup_steps": 5, "batch_size": 4, "num_drivers": 30,
                "num_slots": 20, "patience": 20},
        use_mlflow=False,
    )
    run_dir = Path(result["run_dir"])
    assert (run_dir / "model.pt").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "predictions_val.parquet").exists()
    # Evaluated on val, not test.
    assert result["metrics"]["spearman"] > 0.9
