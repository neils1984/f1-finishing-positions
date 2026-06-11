"""CLI: per-split, per-lap Spearman for the naive baseline (and optionally a
run's predictions). The yardstick for the drift diagnostic and every checkpoint.

Reports val (same-regime reference) and test (the drift probe) so naive
degradation from one to the other is visible in a single run."""
from pathlib import Path

import polars as pl
import typer

from f1_predictor.evaluate import ranking_metrics
from f1_predictor.models.baseline_gbm import naive_predict

app = typer.Typer(add_completion=False)


def _report(label: str, preds: pl.DataFrame) -> None:
    if preds.is_empty():
        typer.echo(f"\n=== {label} === (empty)"); return
    m = ranking_metrics(preds)
    typer.echo(f"\n=== {label} ===")
    typer.echo(f"overall spearman={m['spearman']:.4f}  top1={m['top1_accuracy']:.3f}  "
               f"top3={m['top3_accuracy']:.3f}  mpe={m['mean_position_error']:.3f}  "
               f"n_groups={m['n_groups']}")
    for lap in sorted(preds["snapshot_lap"].unique().to_list()):
        ml = ranking_metrics(preds.filter(pl.col("snapshot_lap") == lap))
        typer.echo(f"  lap{lap}: spearman={ml['spearman']:.4f}  n_groups={ml['n_groups']}")


def _naive_frame(df: pl.DataFrame) -> pl.DataFrame:
    return df.select(["session_key", "snapshot_lap", "driver_number", "final_position"]).with_columns(
        pl.Series("score", naive_predict(df))
    )


@app.command()
def main(
    snapshots_dir: Path = typer.Option(Path("data/snapshots"), "--snapshots-dir"),
    run_dir: Path = typer.Option(None, "--run-dir", help="Optional run dir to compare on TEST"),
) -> None:
    for split in ("val", "test"):
        p = snapshots_dir / f"{split}.parquet"
        if p.exists():
            df = pl.read_parquet(p)
            if not df.is_empty():
                _report(f"NAIVE on {split.upper()}", _naive_frame(df))
    if run_dir is not None:
        _report(f"MODEL ({run_dir.name}) on TEST", pl.read_parquet(run_dir / "predictions_test.parquet"))


if __name__ == "__main__":
    app()
