"""CLI: 2026 walk-forward backtest. Trains on the expanding past, predicts each
2026 race, prints per-race model-vs-naive Spearman, and logs a summary to MLflow.

Knobs for the adaptation experiment:
  --upweight-2026 W   per-row training weight W on 2026 rows (recency lever)
  --blend-alpha A     final score = (1-A)*naive + A*model (naive-anchored fallback)
"""
import glob
from pathlib import Path

import polars as pl
import typer

from f1_predictor.backtest import walk_forward
from f1_predictor.features import FEATURE_COLUMNS
from f1_predictor.models.baseline_gbm import season_weights

app = typer.Typer(add_completion=False)


def _load_races(features_dir: Path, raw_dir: Path) -> dict[int, pl.DataFrame]:
    races = {}
    for f in sorted(glob.glob(str(features_dir / "*.parquet"))):
        key = int(Path(f).stem)
        ses = pl.read_parquet(raw_dir / str(key) / "sessions.parquet").row(0, named=True)
        races[key] = pl.read_parquet(f).with_columns(pl.lit(ses["date_start"]).alias("date_start"))
    return races


@app.command()
def main(
    features_dir: Path = typer.Option(Path("data/features"), "--features-dir"),
    raw_dir: Path = typer.Option(Path("data/raw"), "--raw-dir"),
    test_start: str = typer.Option("2026-01-01", "--test-start"),
    snapshot_laps: str = typer.Option("5,10,15,20,25,30,35,40,45,50", "--laps"),
    upweight_2026: float = typer.Option(1.0, "--upweight-2026"),
    blend_alpha: float = typer.Option(None, "--blend-alpha"),
    use_mlflow: bool = typer.Option(True, "--mlflow/--no-mlflow"),
) -> None:
    races = _load_races(features_dir, raw_dir)
    laps = [int(x) for x in snapshot_laps.split(",")]
    targets = [k for k, df in races.items() if df["date_start"][0] >= test_start]
    wfn = (lambda tr: season_weights(tr, upweight_2026)) if upweight_2026 != 1.0 else None
    res = walk_forward(races, targets, FEATURE_COLUMNS, laps,
                       sample_weight_fn=wfn, blend_alpha=blend_alpha)
    res = res.sort("date_start")
    typer.echo(res.to_pandas().to_string(index=False))
    wins = (res["model_spearman"] > res["naive_spearman"]).sum()
    typer.echo(f"\nmodel beats naive in {wins}/{res.height} 2026 races; "
               f"mean model={res['model_spearman'].mean():.4f} naive={res['naive_spearman'].mean():.4f}")
    if use_mlflow:
        import mlflow
        mlflow.set_experiment("f1-adaptation")
        with mlflow.start_run(run_name=f"wf-w{upweight_2026}-a{blend_alpha}"):
            mlflow.log_params({"upweight_2026": upweight_2026, "blend_alpha": blend_alpha})
            mlflow.log_metric("mean_model_spearman", res["model_spearman"].mean())
            mlflow.log_metric("mean_naive_spearman", res["naive_spearman"].mean())
            mlflow.log_metric("model_win_rate", wins / res.height)


if __name__ == "__main__":
    app()
