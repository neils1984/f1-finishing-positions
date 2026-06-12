"""CLI: per-season persistence diagnostics — naive Spearman, DNF rate, and
position-change dispersion — to explain why some seasons are more predictable."""
import glob
from pathlib import Path

import polars as pl
import typer

from f1_predictor.regime_analysis import position_change_stats

app = typer.Typer(add_completion=False)


def _season(raw_dir: Path, key: int) -> int:
    ses = pl.read_parquet(raw_dir / str(key) / "sessions.parquet").row(0, named=True)
    return int(ses["date_start"][:4])


@app.command()
def main(
    features_dir: Path = typer.Option(Path("data/features"), "--features-dir"),
    sessions_dir: Path = typer.Option(Path("data/sessions"), "--sessions-dir"),
    raw_dir: Path = typer.Option(Path("data/raw"), "--raw-dir"),
    snapshot_lap: int = typer.Option(40, "--lap"),
) -> None:
    rows = []
    for f in sorted(glob.glob(str(features_dir / "*.parquet"))):
        key = int(Path(f).stem)
        df = pl.read_parquet(f)
        if "dnf" not in df.columns:
            # Stage 3 features carry no retirement flag; Stage 2 has is_retired
            # (= dnf|dns|dsq), which is the right "did not finish normally" signal.
            sess = (
                pl.read_parquet(sessions_dir / f"{key}.parquet")
                .select(["driver_number", pl.col("is_retired").alias("dnf")])
                .unique(subset=["driver_number"])
            )
            df = df.join(sess, on="driver_number", how="left")
        df = df.with_columns(pl.lit(_season(raw_dir, key)).alias("season"))
        rows.append(df)
    allf = pl.concat(rows, how="vertical_relaxed")
    for season in sorted(allf["season"].unique().to_list()):
        s = position_change_stats(allf.filter(pl.col("season") == season), snapshot_lap)
        typer.echo(f"{season}: n_races={s['n_races']:2d}  dnf_rate={s['dnf_rate']:.3f}  "
                   f"mean_abs_change@lap{snapshot_lap}={s['mean_abs_change']:.2f}")


if __name__ == "__main__":
    app()
