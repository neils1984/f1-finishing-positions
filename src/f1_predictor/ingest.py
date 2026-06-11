import json
import time
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import requests

OPENF1_BASE = "https://api.openf1.org/v1"

# Pit data is NOT pulled directly. OpenF1 has no /pit data for 2023 (the entire
# training season) while 2024 does — using /pit would skew train vs val/test.
# pit_this_lap / stops_completed are derived from /stints in Stage 2 instead.
#
# car_data (telemetry) is deferred: a per-session query returns 422 ("too much
# data"), and it's only needed for max_speed_kmh. When backfilled later it must
# be pulled per driver_number. Until then max_speed_kmh is null (handled in
# Stage 2's _add_car_data).
ENDPOINTS = [
    "sessions",
    "drivers",
    "laps",
    "position",
    "intervals",
    "stints",
    "race_control",
    "session_result",
]

_BASE_DELAY = 0.5       # seconds between requests (politeness / rate limiting)
_MAX_RETRIES = 8        # attempts on HTTP 429 before giving up
_MAX_BACKOFF = 30.0     # cap on exponential backoff (seconds)


def _retry_wait(resp: requests.Response, attempt: int) -> float:
    """Seconds to wait before retrying a 429. Honour Retry-After if present.

    OpenF1's burst allowance is tiny (~3 rapid requests) but it sets
    Retry-After: 1 and recovers within ~1s, so honouring the header is the
    reliable path; exponential backoff is the fallback.
    """
    retry_after = resp.headers.get("Retry-After")
    if retry_after is not None:
        try:
            return float(retry_after)
        except (TypeError, ValueError):
            pass
    return min(_BASE_DELAY * (2 ** attempt), _MAX_BACKOFF)


def _fetch(s: requests.Session, url: str) -> list[dict]:
    """GET a JSON list endpoint, tolerating OpenF1's quirks.

    - OpenF1 answers zero-row queries with 404 {"detail": "No results found."},
      so a 404 means "no data" → empty list, not an error.
    - An oversized query returns 422 ("too much data at once"); treat as empty
      rather than crashing.
    - 429 (rate limited) is retried, honouring the Retry-After header.
    """
    for attempt in range(_MAX_RETRIES):
        resp = s.get(url, timeout=30)
        if resp.status_code in (404, 422):
            return []
        if resp.status_code == 429:
            time.sleep(_retry_wait(resp, attempt))
            continue
        resp.raise_for_status()
        return resp.json()
    resp.raise_for_status()  # exhausted retries while still rate-limited
    return resp.json()


def pull_car_data(session_key: int, driver_numbers: list[int], raw_dir: Path) -> None:
    """Pull car telemetry per driver and write one car_data.parquet.

    A per-session car_data query 422s ("too much data"), so query per
    driver_number. Keep only what _add_car_data needs: driver_number, date, speed.
    """
    frames: list[pl.DataFrame] = []
    with requests.Session() as s:
        for dn in driver_numbers:
            url = f"{OPENF1_BASE}/car_data?session_key={session_key}&driver_number={dn}"
            data = _fetch(s, url)
            if data:
                frames.append(
                    pl.DataFrame(data, infer_schema_length=None)
                    .select(["driver_number", "date", "speed"])
                )
            time.sleep(_BASE_DELAY)
    df = pl.concat(frames, how="vertical") if frames else pl.DataFrame()
    df.write_parquet(raw_dir / str(session_key) / "car_data.parquet")


def pull_session(session_key: int, raw_dir: Path, force: bool = False) -> None:
    """Pull all endpoints for one race session. Skips if already cached."""
    session_dir = raw_dir / str(session_key)
    meta_path = session_dir / "meta.json"

    if meta_path.exists() and not force:
        return

    session_dir.mkdir(parents=True, exist_ok=True)

    row_counts: dict[str, int] = {}
    with requests.Session() as s:
        for endpoint in ENDPOINTS:
            url = f"{OPENF1_BASE}/{endpoint}?session_key={session_key}"
            data = _fetch(s, url)

            # infer_schema_length=None scans all rows so mixed-type columns
            # (e.g. /intervals gap_to_leader: float or "+1 LAP" for lapped
            # drivers) resolve to a String supertype instead of raising.
            df = pl.DataFrame(data, infer_schema_length=None) if data else pl.DataFrame()
            df.write_parquet(session_dir / f"{endpoint}.parquet")
            row_counts[endpoint] = len(df)

            time.sleep(_BASE_DELAY)

    # car_data is pulled per driver (a per-session query 422s) — see module docstring.
    drivers_df = pl.read_parquet(session_dir / "drivers.parquet")
    driver_numbers = drivers_df["driver_number"].unique().to_list() if not drivers_df.is_empty() else []
    pull_car_data(session_key, driver_numbers, raw_dir)
    row_counts["car_data"] = pl.read_parquet(session_dir / "car_data.parquet").height

    meta = {
        "session_key": session_key,
        "pull_timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "row_counts": row_counts,
    }
    meta_path.write_text(json.dumps(meta, indent=2))


def pull_season(year: int, raw_dir: Path, force: bool = False) -> list[int]:
    """Pull all Grand Prix race sessions for a season. Returns session keys.

    The year=...&session_type=Race query also returns Sprint sessions; only the
    main race (session_name == "Race") is kept. Monaco is excluded entirely
    (OpenF1 names its circuit "Monte Carlo").
    """
    with requests.Session() as s:
        resp = s.get(
            f"{OPENF1_BASE}/sessions?year={year}&session_type=Race",
            timeout=30,
        )
        resp.raise_for_status()
        sessions = resp.json()

    keys: list[int] = []
    for session in sessions:
        # Keep only the main Grand Prix race; drop Sprints.
        if session.get("session_name") != "Race":
            continue
        circuit = session.get("circuit_short_name", "").lower()
        # OpenF1 names the Monaco circuit "Monte Carlo" (not "Monaco"). Match
        # both spellings; note Montreal/Monza must NOT match.
        if "monaco" in circuit or "monte carlo" in circuit:
            continue
        key = int(session["session_key"])
        pull_session(key, raw_dir, force=force)
        keys.append(key)
        time.sleep(_BASE_DELAY)

    return keys
