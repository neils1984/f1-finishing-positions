"""CLI: train + evaluate the cross-driver Transformer on the snapshot splits.

Trains on `train`, early-stops on the pre-2026 `val` split, and reports the val
metrics. The 2026 `test` split is intentionally untouched (architecture test on
a single regime). Use compare_val.py for the Transformer-vs-GBM-vs-naive table.
"""
from pathlib import Path

import typer

from f1_predictor.train import train_transformer

app = typer.Typer(add_completion=False)


@app.command()
def main(
    snapshots_dir: Path = typer.Option(Path("data/snapshots"), "--snapshots-dir"),
    runs_dir: Path = typer.Option(Path("runs"), "--runs-dir"),
    epochs: int = typer.Option(100, "--epochs"),
    no_mlflow: bool = typer.Option(False, "--no-mlflow"),
) -> None:
    config = {
        "d_model": 128, "n_heads": 8, "n_layers": 4, "dropout": 0.1,
        "lr": 1e-4, "weight_decay": 0.01, "warmup_steps": 500,
        "batch_size": 32, "epochs": epochs, "patience": 10,
        "num_drivers": 30, "num_slots": 20,
    }
    result = train_transformer(
        snapshots_dir, runs_dir, config, use_mlflow=not no_mlflow
    )
    typer.echo(f"Run: {result['run_dir']}")
    for k, v in result["metrics"].items():
        typer.echo(f"  {k}: {v}")


if __name__ == "__main__":
    app()
