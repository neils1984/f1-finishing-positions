"""CLI: build Stage 3 features for one race or all sessionised races."""
from pathlib import Path

import polars as pl
import typer

from f1_predictor.features import build_features, load_circuits
from f1_predictor.priors import build_driver_races, compute_priors

app = typer.Typer(add_completion=False)


@app.command()
def main(
    session_key: int = typer.Option(None, "--session-key", help="One race; omit for all"),
    sessions_dir: Path = typer.Option(Path("data/sessions"), "--sessions-dir"),
    raw_dir: Path = typer.Option(Path("data/raw"), "--raw-dir"),
    features_dir: Path = typer.Option(Path("data/features"), "--features-dir"),
) -> None:
    keys = sorted(int(p.stem) for p in sessions_dir.glob("*.parquet") if "_masks" not in p.stem)
    if not keys:
        typer.echo("No sessionised races found.", err=True)
        raise SystemExit(1)

    # Priors are computed once across ALL races (the SQL enforces prior-only).
    driver_races = build_driver_races(raw_dir, keys)
    priors = compute_priors(driver_races)
    circuits = load_circuits()

    targets = [session_key] if session_key is not None else keys
    for key in targets:
        out = build_features(key, sessions_dir, raw_dir, features_dir, priors, circuits)
        typer.echo(f"  {key}: {out.shape[0]} rows, {out.shape[1]} cols")
    typer.echo(f"Built features for {len(targets)} race(s)")


if __name__ == "__main__":
    app()
