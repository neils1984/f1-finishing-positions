"""CLI: build Stage 4 snapshots (train/val/test) from Stage 3 features."""
import subprocess
from pathlib import Path

import typer
import yaml

from f1_predictor.features import FEATURE_COLUMNS
from f1_predictor.snapshots import build_snapshots

app = typer.Typer(add_completion=False)


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


@app.command()
def main(
    features_dir: Path = typer.Option(Path("data/features"), "--features-dir"),
    raw_dir: Path = typer.Option(Path("data/raw"), "--raw-dir"),
    out_dir: Path = typer.Option(Path("data/snapshots"), "--out-dir"),
    config: Path = typer.Option(Path("config/default.yaml"), "--config"),
) -> None:
    cfg = yaml.safe_load(open(config))
    meta = build_snapshots(
        features_dir=features_dir, raw_dir=raw_dir, out_dir=out_dir,
        feature_columns=FEATURE_COLUMNS,
        snapshot_laps=cfg["snapshot_laps"], val_start=cfg["val_start"],
        test_start=cfg["test_start"],
        git_sha=_git_sha(),
    )
    for split, ks in meta["splits"].items():
        typer.echo(f"  {split}: {len(ks)} races")
    typer.echo(f"data_version={meta['data_version']}")


if __name__ == "__main__":
    app()
