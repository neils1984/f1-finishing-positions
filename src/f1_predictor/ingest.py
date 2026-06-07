import json
import time
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import requests

OPENF1_BASE = "https://api.openf1.org/v1"

ENDPOINTS = [
    "sessions",
    "drivers",
    "laps",
    "position",
    "intervals",
    "stints",
    "pit",
    "race_control",
    "session_result",
    "car_data",
]


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
            resp = s.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            df = pl.DataFrame(data) if data else pl.DataFrame()
            df.write_parquet(session_dir / f"{endpoint}.parquet")
            row_counts[endpoint] = len(df)

            time.sleep(0.2)

    meta = {
        "session_key": session_key,
        "pull_timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "row_counts": row_counts,
    }
    meta_path.write_text(json.dumps(meta, indent=2))


def pull_season(year: int, raw_dir: Path, force: bool = False) -> list[int]:
    """Pull all race sessions for a season. Skips Monaco. Returns session keys."""
    with requests.Session() as s:
        resp = s.get(
            f"{OPENF1_BASE}/sessions?year={year}&session_type=Race",
            timeout=30,
        )
        resp.raise_for_status()
        sessions = resp.json()

    keys: list[int] = []
    for session in sessions:
        circuit = session.get("circuit_short_name", "").lower()
        if "monaco" in circuit:
            continue
        key = int(session["session_key"])
        pull_session(key, raw_dir, force=force)
        keys.append(key)
        time.sleep(0.2)

    return keys
