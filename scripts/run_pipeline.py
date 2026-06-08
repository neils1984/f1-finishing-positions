"""Run Stage 2 (sessionise) + Stage 3 (features) for all pulled raw sessions."""
from pathlib import Path

import typer

from f1_predictor.sessionise import sessionise
from f1_predictor.features import build_features, load_circuits
from f1_predictor.priors import build_driver_races, compute_priors

app = typer.Typer(add_completion=False)


def discover_sessions(raw_dir: Path) -> list[int]:
    """Integer keys of fully-pulled raw sessions (those with a meta.json)."""
    return sorted(
        int(p.name)
        for p in raw_dir.iterdir()
        if p.is_dir() and (p / "meta.json").exists()
    )


@app.command()
def main(
    raw_dir: Path = typer.Option(Path("data/raw"), "--raw-dir"),
    sessions_dir: Path = typer.Option(Path("data/sessions"), "--sessions-dir"),
    features_dir: Path = typer.Option(Path("data/features"), "--features-dir"),
) -> None:
    keys = discover_sessions(raw_dir)
    if not keys:
        typer.echo("No pulled sessions found.", err=True)
        raise SystemExit(1)

    # Stage 2: sessionise each race (skips cancelled / empty-laps sessions).
    sessionised: list[int] = []
    for key in keys:
        df = sessionise(key, raw_dir, sessions_dir)
        if not df.is_empty():
            sessionised.append(key)
    typer.echo(f"Sessionised {len(sessionised)} races")

    # Stage 3: priors computed once across all sessionised races, then features.
    driver_races = build_driver_races(raw_dir, sessionised)
    priors = compute_priors(driver_races)
    circuits = load_circuits()
    for key in sessionised:
        build_features(key, sessions_dir, raw_dir, features_dir, priors, circuits)
    typer.echo(f"Built features for {len(sessionised)} races")


if __name__ == "__main__":
    app()
