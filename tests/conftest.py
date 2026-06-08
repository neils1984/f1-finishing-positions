"""Shared fixtures for sessionise integration tests."""
from pathlib import Path
import polars as pl
import pytest

DATA_DIR = Path("data")


def _load_or_skip(session_key: int) -> pl.DataFrame:
    path = DATA_DIR / "sessions" / f"{session_key}.parquet"
    if not path.exists():
        pytest.skip(f"Fixture {session_key} not sessionised. Run: "
                    f"uv run python -c \"from pathlib import Path; "
                    f"from f1_predictor.sessionise import sessionise; "
                    f"sessionise({session_key}, Path('data/raw'), Path('data/sessions'))\"")
    return pl.read_parquet(path)


# Hand-picked 2023 fixture races (discovered via scripts/inspect_fixture.py):
FIXTURE_SC = 9070           # Baku — single SC window, laps 11–13
FIXTURE_RETIREMENTS = 9181  # 5 retirements across the race
FIXTURE_CLEAN = 9078        # Miami — no SC/VSC, no retirements


@pytest.fixture(scope="session")
def df_sc():
    return _load_or_skip(FIXTURE_SC)


@pytest.fixture(scope="session")
def df_retirements():
    return _load_or_skip(FIXTURE_RETIREMENTS)


@pytest.fixture(scope="session")
def df_clean():
    return _load_or_skip(FIXTURE_CLEAN)
