"""Unit tests for cross-race priors with the no-leakage guard."""
import polars as pl
import pytest
from f1_predictor.priors import compute_priors


def _race(session_key, date, circuit, rows):
    """rows: list of (driver_number, team, points, finished[bool])."""
    return [
        {
            "session_key": session_key,
            "date_start": date,
            "circuit_short_name": circuit,
            "driver_number": d,
            "team_name": t,
            "points": p,
            "finished": f,
        }
        for (d, t, p, f) in rows
    ]


def _frame(*races) -> pl.DataFrame:
    rows = [r for race in races for r in race]
    return pl.DataFrame(rows)


def test_championship_standing_uses_only_prior_races():
    data = _frame(
        _race(1, "2023-03-01", "Sakhir", [(1, "RB", 25, True), (44, "MER", 18, True)]),
        _race(2, "2023-03-08", "Jeddah", [(1, "RB", 25, True), (44, "MER", 18, True)]),
        _race(3, "2023-03-15", "Melbourne", [(1, "RB", 25, True), (44, "MER", 18, True)]),
    )
    priors = compute_priors(data)

    # Entering race 1: no prior races -> standing 0 (no points yet).
    r1 = priors.filter((pl.col("session_key") == 1) & (pl.col("driver_number") == 1))
    assert r1["driver_championship_standing"][0] == pytest.approx(0.0)
    # Entering race 3: driver 1 has 25+25 = 50 from races 1 and 2 (NOT race 3).
    r3 = priors.filter((pl.col("session_key") == 3) & (pl.col("driver_number") == 1))
    assert r3["driver_championship_standing"][0] == pytest.approx(50.0)


def test_driver_circuit_finish_rate_prior_only():
    # Driver 1 races Sakhir three times: finishes, DNF, then the current race.
    data = _frame(
        _race(1, "2022-03-01", "Sakhir", [(1, "RB", 25, True)]),
        _race(2, "2023-03-01", "Sakhir", [(1, "RB", 0, False)]),
        _race(3, "2024-03-01", "Sakhir", [(1, "RB", 25, True)]),
    )
    priors = compute_priors(data)
    # Entering race 3 at Sakhir: prior Sakhir races = {finish, DNF} -> rate 0.5.
    r3 = priors.filter((pl.col("session_key") == 3) & (pl.col("driver_number") == 1))
    assert r3["driver_circuit_finish_rate"][0] == pytest.approx(0.5)
    # Entering race 1: no prior Sakhir races -> null.
    r1 = priors.filter((pl.col("session_key") == 1) & (pl.col("driver_number") == 1))
    assert r1["driver_circuit_finish_rate"][0] is None


def test_team_priors_aggregate_both_cars():
    data = _frame(
        _race(1, "2023-03-01", "Sakhir", [(1, "RB", 25, True), (11, "RB", 18, False)]),
        _race(2, "2023-03-08", "Jeddah", [(1, "RB", 25, True), (11, "RB", 18, True)]),
    )
    priors = compute_priors(data)
    # Entering race 2: team RB prior points = 25 + 18 = 43 (both cars, race 1).
    r2 = priors.filter((pl.col("session_key") == 2) & (pl.col("driver_number") == 1))
    assert r2["team_championship_standing"][0] == pytest.approx(43.0)
