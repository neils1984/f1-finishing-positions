"""Stage 3 integration tests on the three real 2023 fixtures."""
from pathlib import Path

import polars as pl
import pytest

from f1_predictor.features import FEATURE_COLUMNS
from tests.conftest import FIXTURE_SC, FIXTURE_RETIREMENTS, FIXTURE_CLEAN

FEATURES_DIR = Path("data") / "features"


def _load_or_skip(session_key: int) -> pl.DataFrame:
    path = FEATURES_DIR / f"{session_key}.parquet"
    if not path.exists():
        pytest.skip(f"Features for {session_key} not built. Run: "
                    f"uv run python scripts/build_features.py --session-key {session_key}")
    return pl.read_parquet(path)


@pytest.fixture(scope="session")
def feat_sc():
    return _load_or_skip(FIXTURE_SC)


@pytest.fixture(scope="session")
def feat_clean():
    return _load_or_skip(FIXTURE_CLEAN)


@pytest.mark.parametrize("fixture_name", ["feat_sc", "feat_clean"])
def test_all_feature_columns_present(fixture_name, request):
    df = request.getfixturevalue(fixture_name)
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    assert not missing, f"Missing feature columns: {missing}"


@pytest.mark.parametrize("fixture_name", ["feat_sc", "feat_clean"])
def test_one_row_per_driver_lap(fixture_name, request):
    df = request.getfixturevalue(fixture_name)
    dupes = df.filter(df.select(["driver_number", "lap_number"]).is_duplicated())
    assert dupes.is_empty()


def test_gaps_are_numeric(feat_clean):
    assert feat_clean["gap_to_leader"].dtype == pl.Float64
    assert feat_clean["interval_to_ahead"].dtype == pl.Float64


def test_leader_has_zero_gaps_ahead(feat_clean):
    leader = feat_clean.filter(pl.col("position") == 1)
    assert (leader["mean_gap_cars_ahead"].fill_null(0) == 0).all()
    assert (leader["stdev_gap_cars_ahead"].fill_null(0) == 0).all()


def test_tyre_onehot_sums_to_at_most_one(feat_clean):
    onehot = feat_clean.select(["tyre_soft", "tyre_medium", "tyre_hard", "tyre_inter", "tyre_wet"])
    row_sums = onehot.sum_horizontal()
    assert (row_sums <= 1).all()


def test_clean_fixture_is_street_circuit_true(feat_clean):
    # FIXTURE_CLEAN is Miami (a street circuit).
    assert feat_clean["is_street_circuit"].unique().to_list() == [True]


def test_distance_remaining_decreases_over_laps(feat_clean):
    d = (
        feat_clean.filter(pl.col("driver_number") == feat_clean["driver_number"][0])
        .sort("lap_number")
    )
    diffs = d["distance_remaining_km"].diff().drop_nulls()
    assert (diffs <= 0).all(), "distance_remaining_km must be non-increasing within a race"


def test_first_2023_race_has_null_priors():
    # The earliest 2023 race has no prior races, so championship standing is 0
    # and circuit finish rate is null (no prior race at that circuit).
    raw = Path("data/raw")
    keys = [int(p.name) for p in raw.iterdir() if p.is_dir() and (p / "sessions.parquet").exists()]
    dated = []
    for k in keys:
        ses = pl.read_parquet(raw / str(k) / "sessions.parquet").row(0, named=True)
        dated.append((ses["date_start"], k))
    first_key = min(dated)[1]
    path = FEATURES_DIR / f"{first_key}.parquet"
    if not path.exists():
        pytest.skip(f"Features for first race {first_key} not built")
    df = pl.read_parquet(path)
    assert (df["driver_championship_standing"] == 0).all()
    assert df["driver_circuit_finish_rate"].null_count() == df.height
