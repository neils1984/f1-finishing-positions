"""CLI: pull one season from OpenF1."""
from pathlib import Path
import typer
from f1_predictor.ingest import pull_season

app = typer.Typer(add_completion=False)


@app.command()
def main(
    year: int = typer.Option(..., "--year", help="Season year, e.g. 2023"),
    force: bool = typer.Option(False, "--force", help="Re-pull even if cached"),
    raw_dir: Path = typer.Option(Path("data/raw"), "--raw-dir"),
) -> None:
    keys = pull_season(year, raw_dir=raw_dir, force=force)
    typer.echo(f"Pulled {len(keys)} sessions for {year}")
    for k in keys:
        typer.echo(f"  {k}")


if __name__ == "__main__":
    app()
