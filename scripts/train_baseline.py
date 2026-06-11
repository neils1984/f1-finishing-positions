"""CLI: train + evaluate the LightGBM baseline on the snapshot splits."""
from pathlib import Path

import typer

from f1_predictor.models.baseline_gbm import run_baseline

app = typer.Typer(add_completion=False)


@app.command()
def main(
    snapshots_dir: Path = typer.Option(Path("data/snapshots"), "--snapshots-dir"),
    runs_dir: Path = typer.Option(Path("runs"), "--runs-dir"),
    no_mlflow: bool = typer.Option(False, "--no-mlflow", help="Skip MLflow logging"),
) -> None:
    result = run_baseline(snapshots_dir, runs_dir, use_mlflow=not no_mlflow)
    typer.echo(f"Run: {result['run_dir']}")
    for k, v in result["metrics"].items():
        typer.echo(f"  {k}: {v}")


if __name__ == "__main__":
    app()
