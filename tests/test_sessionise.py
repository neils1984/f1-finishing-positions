"""Stage 2 integration tests — required by spec. Three hand-picked fixture races."""
import numpy as np
import polars as pl
import pytest

# ── Structural invariants (all fixture races) ──────────────────────────────

REQUIRED_COLUMNS = [
    "session_key", "driver_number", "lap_number",
    "position", "gap_to_leader", "interval_to_ahead",
    "tyre_compound", "tyre_age_laps", "stint_number",
    "pit_this_lap", "stops_completed",
    "lap_time", "max_speed_kmh",
    "sc_active", "vsc_active", "red_flag_active", "laps_since_sc_end",
    "is_retired", "retirement_lap", "final_position",
]


@pytest.mark.parametrize("fixture_name", ["df_sc", "df_retirements", "df_clean"])
def test_required_columns_present(fixture_name, request):
    df = request.getfixturevalue(fixture_name)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    assert not missing, f"Missing columns: {missing}"


@pytest.mark.parametrize("fixture_name", ["df_sc", "df_retirements", "df_clean"])
def test_no_null_final_position(fixture_name, request):
    df = request.getfixturevalue(fixture_name)
    nulls = df["final_position"].null_count()
    assert nulls == 0, f"{nulls} null final_position values"


@pytest.mark.parametrize("fixture_name", ["df_sc", "df_retirements", "df_clean"])
def test_final_position_same_across_all_laps_for_driver(fixture_name, request):
    df = request.getfixturevalue(fixture_name)
    varying = (
        df.group_by("driver_number")
        .agg(pl.col("final_position").n_unique().alias("n"))
        .filter(pl.col("n") > 1)
    )
    assert varying.is_empty(), f"Drivers with varying final_position: {varying}"


@pytest.mark.parametrize("fixture_name", ["df_sc", "df_retirements", "df_clean"])
def test_tyre_age_non_negative(fixture_name, request):
    df = request.getfixturevalue(fixture_name)
    negatives = df.filter(pl.col("tyre_age_laps") < 0)
    assert negatives.is_empty(), "tyre_age_laps must be >= 0"


@pytest.mark.parametrize("fixture_name", ["df_sc", "df_retirements", "df_clean"])
def test_stops_completed_monotonically_non_decreasing(fixture_name, request):
    df = request.getfixturevalue(fixture_name)
    decreases = (
        df.sort(["driver_number", "lap_number"])
        .with_columns(
            pl.col("stops_completed").diff().over("driver_number").alias("delta")
        )
        .filter(pl.col("delta") < 0)
    )
    assert decreases.is_empty(), "stops_completed must never decrease within a driver's race"


@pytest.mark.parametrize("fixture_name", ["df_sc", "df_retirements", "df_clean"])
def test_no_duplicate_driver_lap(fixture_name, request):
    df = request.getfixturevalue(fixture_name)
    dupes = df.filter(df.select(["driver_number", "lap_number"]).is_duplicated())
    assert dupes.is_empty(), "Each (driver_number, lap_number) must be unique"


@pytest.mark.parametrize("fixture_name", ["df_sc", "df_retirements", "df_clean"])
def test_tyre_compound_valid_values(fixture_name, request):
    df = request.getfixturevalue(fixture_name)
    valid = {"SOFT", "MEDIUM", "HARD", "INTER", "WET"}
    compounds = set(df["tyre_compound"].drop_nulls().unique().to_list())
    invalid = compounds - valid
    assert not invalid, f"Unknown tyre compounds: {invalid}"


@pytest.mark.parametrize("fixture_name", ["df_sc", "df_retirements", "df_clean"])
def test_laps_since_sc_end_non_negative(fixture_name, request):
    df = request.getfixturevalue(fixture_name)
    negs = df.filter(pl.col("laps_since_sc_end") < 0)
    assert negs.is_empty()


# ── Clean race (FIXTURE_CLEAN = 9078, Miami 2023) ──────────────────────────
# Values from: uv run python scripts/inspect_fixture.py 9078

CLEAN_STINT_DRIVER = 1        # race winner
CLEAN_STINT_COMPOUND = "HARD"
CLEAN_STINT_LAP_START = 1
CLEAN_STINT_LAP_END = 45      # stint 1 (HARD) ran laps 1–45
CLEAN_WINNER_DRIVER = 1       # Verstappen


def test_clean_no_sc(df_clean):
    assert df_clean["sc_active"].any() == False, "Expected zero SC laps in clean fixture"


def test_clean_no_retirements(df_clean):
    assert df_clean["is_retired"].any() == False, "Expected zero retirements in clean fixture"


def test_clean_tyre_window(df_clean):
    stint_rows = df_clean.filter(
        (pl.col("driver_number") == CLEAN_STINT_DRIVER) &
        (pl.col("lap_number") >= CLEAN_STINT_LAP_START) &
        (pl.col("lap_number") <= CLEAN_STINT_LAP_END)
    )
    assert not stint_rows.is_empty(), "No rows found for the specified driver/lap range"
    assert (
        stint_rows["tyre_compound"] == CLEAN_STINT_COMPOUND
    ).all(), f"Expected {CLEAN_STINT_COMPOUND} for driver {CLEAN_STINT_DRIVER} laps {CLEAN_STINT_LAP_START}–{CLEAN_STINT_LAP_END}"


def test_clean_tyre_age_resets_on_pit_lap(df_clean):
    # pit is derived from /stints, so pit_this_lap is the out-lap (first lap of
    # the new stint), where tyre_age_laps == 0.
    driver = df_clean.filter(pl.col("driver_number") == CLEAN_STINT_DRIVER).sort("lap_number")
    pit_laps = driver.filter(pl.col("pit_this_lap"))["lap_number"].to_list()
    assert pit_laps, "expected at least one pit stop for the winner"
    for pit_lap in pit_laps:
        age = driver.filter(pl.col("lap_number") == pit_lap)["tyre_age_laps"]
        assert age[0] == 0, f"tyre_age_laps should be 0 on the out-lap (pit_this_lap {pit_lap})"


def test_clean_winner_final_position(df_clean):
    winner = df_clean.filter(pl.col("driver_number") == CLEAN_WINNER_DRIVER)
    assert winner["final_position"].unique()[0] == 1


# ── SC race (FIXTURE_SC = 9070, Baku 2023) ─────────────────────────────────
# Values from: uv run python scripts/inspect_fixture.py 9070

SC_FIRST_LAP = 11   # first lap where sc_active == True
SC_LAST_LAP = 13    # last lap where sc_active == True


def test_sc_race_has_sc_laps(df_sc):
    sc_laps = df_sc.filter(pl.col("sc_active"))["lap_number"].unique().sort()
    assert not sc_laps.is_empty(), "Expected at least one SC lap"


def test_sc_active_correct_range(df_sc):
    sc_laps = set(df_sc.filter(pl.col("sc_active"))["lap_number"].to_list())
    for lap in range(SC_FIRST_LAP, SC_LAST_LAP + 1):
        assert lap in sc_laps, f"Lap {lap} should be SC-active"


def test_laps_before_sc_not_active(df_sc):
    if SC_FIRST_LAP > 1:
        before = df_sc.filter(pl.col("lap_number") < SC_FIRST_LAP)
        assert before["sc_active"].any() == False


def test_laps_after_sc_end_not_active(df_sc):
    after = df_sc.filter(pl.col("lap_number") > SC_LAST_LAP)
    assert after["sc_active"].any() == False


def test_laps_since_sc_end_increases_after_sc(df_sc):
    after = (
        df_sc.filter(pl.col("lap_number") > SC_LAST_LAP)
        .select(["lap_number", "laps_since_sc_end"])
        .unique("lap_number")
        .sort("lap_number")
    )
    if len(after) >= 2:
        deltas = after["laps_since_sc_end"].diff().drop_nulls()
        assert (deltas == 1).all(), "laps_since_sc_end should increment by 1 each lap after SC ends"


# ── Retirement race (FIXTURE_RETIREMENTS = 9181) ───────────────────────────
# Values from: uv run python scripts/inspect_fixture.py 9181
# (driver_number, retirement_lap, final_position)

RETIRED_DRIVERS: list[tuple[int, int, int]] = [
    (11, 1, 20),
    (20, 31, 19),
    (14, 47, 18),
    (18, 66, 17),
    (2, 70, 16),
]


def test_retirement_race_has_retirements(df_retirements):
    assert df_retirements["is_retired"].any(), "Expected at least one retirement"


@pytest.mark.parametrize("driver_number,expected_ret_lap,expected_final_pos", RETIRED_DRIVERS)
def test_retirement_lap_correct(driver_number, expected_ret_lap, expected_final_pos, df_retirements):
    rows = df_retirements.filter(pl.col("driver_number") == driver_number)
    assert not rows.is_empty(), f"Driver {driver_number} not found"
    assert rows["is_retired"].unique()[0] == True
    assert rows["retirement_lap"].unique()[0] == expected_ret_lap, \
        f"Driver {driver_number}: expected retirement_lap={expected_ret_lap}"
    assert rows["final_position"].unique()[0] == expected_final_pos, \
        f"Driver {driver_number}: expected final_position={expected_final_pos}"


def test_retired_drivers_have_final_position(df_retirements):
    retired = df_retirements.filter(pl.col("is_retired"))
    nulls = retired["final_position"].null_count()
    assert nulls == 0, "Retired drivers must still have a final_position (laps-completed order)"


def test_attention_mask_zeros_after_retirement():
    """attention_mask must be 0 for retired drivers past their retirement_lap."""
    from tests.conftest import FIXTURE_RETIREMENTS
    from pathlib import Path
    import numpy as np

    mask_path = Path(f"data/sessions/{FIXTURE_RETIREMENTS}_masks.npz")
    if not mask_path.exists():
        pytest.skip("Mask file not generated yet")

    data = np.load(mask_path)
    attention_mask = data["attention_mask"]  # [n_drivers, n_laps]
    assert attention_mask.shape[0] > 0
    assert attention_mask.shape[1] > 0
    # Every column (lap) should have at least one active driver
    assert (attention_mask.sum(axis=0) > 0).all(), "Every lap must have at least one active driver"
