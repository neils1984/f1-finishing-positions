"""Print key facts about a sessionised race — use to write fixture test values."""
from pathlib import Path
import polars as pl
import typer

app = typer.Typer(add_completion=False)


@app.command()
def main(session_key: int = typer.Argument(...)) -> None:
    path = Path(f"data/sessions/{session_key}.parquet")
    if not path.exists():
        typer.echo(f"No sessionised file found at {path}. Run sessionise first.", err=True)
        raise SystemExit(1)

    df = pl.read_parquet(path)

    typer.echo(f"\n=== Session {session_key} ===")
    typer.echo(f"Drivers: {sorted(df['driver_number'].unique().to_list())}")
    typer.echo(f"Laps: {df['lap_number'].min()} – {df['lap_number'].max()}")

    typer.echo("\n--- SC events ---")
    sc_laps = df.filter(pl.col("sc_active")).select("lap_number").unique().sort("lap_number")
    typer.echo(sc_laps if not sc_laps.is_empty() else "  (none)")

    typer.echo("\n--- Retirements ---")
    retirements = (
        df.filter(pl.col("is_retired"))
        .select(["driver_number", "retirement_lap", "final_position"])
        .unique("driver_number")
        .sort("retirement_lap")
    )
    typer.echo(retirements if not retirements.is_empty() else "  (none)")

    typer.echo("\n--- Tyre stints (driver 1st row per stint) ---")
    stints = (
        df.sort(["driver_number", "lap_number"])
        .with_columns(
            (pl.col("tyre_compound") != pl.col("tyre_compound").shift(1).over("driver_number")).alias("new_stint")
        )
        .filter(pl.col("new_stint"))
        .select(["driver_number", "lap_number", "tyre_compound", "stint_number"])
        .sort(["driver_number", "lap_number"])
    )
    typer.echo(stints)

    typer.echo("\n--- Final positions ---")
    typer.echo(
        df.select(["driver_number", "final_position"])
        .unique("driver_number")
        .sort("final_position")
    )


if __name__ == "__main__":
    app()
