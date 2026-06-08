"""Stage 3 cross-race priors, computed with a strict no-leakage guard.

All priors for race R use ONLY races whose date_start is before R's. DuckDB does
the cross-race windowing. Input is one row per driver-race; output adds the four
prior columns keyed by (session_key, driver_number).
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import polars as pl

# Columns the input driver-race frame must provide.
_INPUT_COLUMNS = [
    "session_key", "date_start", "circuit_short_name",
    "driver_number", "team_name", "points", "finished",
]


def compute_priors(driver_races: pl.DataFrame) -> pl.DataFrame:
    """Return per (session_key, driver_number) prior features (prior races only).

    Columns added: driver_championship_standing, team_championship_standing,
    driver_circuit_finish_rate, team_circuit_finish_rate.
    Standings are cumulative championship points from prior races (0 if none).
    Finish rates are finishes/starts at the same circuit in prior races (null if
    the driver/team has no prior race at that circuit).
    """
    missing = [c for c in _INPUT_COLUMNS if c not in driver_races.columns]
    if missing:
        raise ValueError(f"driver_races missing columns: {missing}")

    con = duckdb.connect()
    con.register("dr", driver_races.to_arrow())

    query = """
    WITH dr AS (
        SELECT
            session_key,
            CAST(date_start AS TIMESTAMP) AS dt,
            circuit_short_name,
            driver_number,
            team_name,
            CAST(points AS DOUBLE) AS points,
            CAST(finished AS INTEGER) AS finished
        FROM dr
    )
    SELECT
        cur.session_key,
        cur.driver_number,
        -- Driver championship standing: sum points of this driver's prior races.
        COALESCE((
            SELECT SUM(p.points) FROM dr p
            WHERE p.driver_number = cur.driver_number AND p.dt < cur.dt
        ), 0.0) AS driver_championship_standing,
        -- Team championship standing: sum points of both team cars' prior races.
        COALESCE((
            SELECT SUM(p.points) FROM dr p
            WHERE p.team_name = cur.team_name AND p.dt < cur.dt
        ), 0.0) AS team_championship_standing,
        -- Driver circuit finish rate: finishes/starts at this circuit, prior races.
        (
            SELECT AVG(CAST(p.finished AS DOUBLE)) FROM dr p
            WHERE p.driver_number = cur.driver_number
              AND p.circuit_short_name = cur.circuit_short_name
              AND p.dt < cur.dt
        ) AS driver_circuit_finish_rate,
        -- Team circuit finish rate: both cars, this circuit, prior races.
        (
            SELECT AVG(CAST(p.finished AS DOUBLE)) FROM dr p
            WHERE p.team_name = cur.team_name
              AND p.circuit_short_name = cur.circuit_short_name
              AND p.dt < cur.dt
        ) AS team_circuit_finish_rate
    FROM dr cur
    """
    result = con.execute(query).arrow()
    con.close()
    return pl.from_arrow(result)


def build_driver_races(raw_dir: Path, session_keys: list[int]) -> pl.DataFrame:
    """Assemble the one-row-per-driver-race input frame from raw endpoints.

    Reads session_result (points + dnf/dns/dsq), drivers (team_name), and
    sessions (date_start, circuit_short_name) for each session_key.
    finished := not (dnf or dns or dsq).
    """
    frames: list[pl.DataFrame] = []
    for key in session_keys:
        sdir = raw_dir / str(key)
        sr = pl.read_parquet(sdir / "session_result.parquet")
        drv = pl.read_parquet(sdir / "drivers.parquet").select(["driver_number", "team_name"]).unique("driver_number")
        ses = pl.read_parquet(sdir / "sessions.parquet").row(0, named=True)

        finished = ~(
            pl.col("dnf").fill_null(False)
            | pl.col("dns").fill_null(False)
            | pl.col("dsq").fill_null(False)
        )
        frame = (
            sr.with_columns(finished.alias("finished"))
            .join(drv, on="driver_number", how="left")
            .with_columns([
                pl.lit(key).alias("session_key"),
                pl.lit(ses["date_start"]).alias("date_start"),
                pl.lit(ses["circuit_short_name"]).alias("circuit_short_name"),
                pl.col("points").cast(pl.Float64),
            ])
            .select(_INPUT_COLUMNS)
        )
        frames.append(frame)
    return pl.concat(frames, how="vertical")
