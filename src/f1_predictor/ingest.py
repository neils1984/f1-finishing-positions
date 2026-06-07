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
ENDPOINTS = [
    "sessions",
    "drivers",
    "laps",
    "position",
    "intervals",
    "stints",
    "race_control",
    "session_result",
    "car_data",
]

_BASE_DELAY = 0.3       # seconds between requests (politeness / rate limiting)
_MAX_RETRIES = 5        # attempts on HTTP 429 before giving up


def _fetch(s: requests.Session, url: str) -> list[dict]:
    """GET a JSON list endpoint, tolerating OpenF1's quirks.

    - OpenF1 answers zero-row queries with 404 {"detail": "No results found."},
      so a 404 means "no data" → empty list, not an error.
    - 429 (rate limited) is retried with exponential backoff.
    """
    for attempt in range(_MAX_RETRIES):
        resp = s.get(url, timeout=30)
        if resp.status_code == 404:
            return []
        if resp.status_code == 429:
            time.sleep(_BASE_DELAY * (2 ** attempt))
            continue
        resp.raise_for_status()
        return resp.json()
    resp.raise_for_status()  # exhausted retries while still rate-limited
    return resp.json()


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
