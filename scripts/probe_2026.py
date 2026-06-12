"""CLI: probe what OpenF1 actually exposes for a 2026 race vs a reference race.

The 2026 technical regulations (active aero replacing DRS, the new power units,
narrower cars) change the racing dynamics and may add / remove / repurpose
fields in the API. Before building or gating any 2026 feature we need ground
truth on what the endpoints return, because the production pipeline narrows
several endpoints on ingest and would hide new channels:

  * ingest.pull_car_data keeps only [driver_number, date, speed] — it would
    silently drop a `drs` / active-aero channel if one exists.
  * race_control is collapsed to SC/VSC/red-flag booleans in Stage 2 — any
    "active aero", "override mode" or DRS-related messages are discarded.

This script hits the RAW endpoints (all fields), reports the schema of each for
a 2026 race, diffs it against a reference-year race, and specifically:

  * checks car_data for a `drs` channel and reports its value distribution
    (so we can see whether DRS telemetry is still emitted under active aero);
  * scans race_control text for active-aero / override / DRS keywords.

Runnable wherever OpenF1 is reachable (api.openf1.org must be allowlisted).
Writes a JSON report to data/probe/ and prints a summary.

    uv run python scripts/probe_2026.py
    uv run python scripts/probe_2026.py --year 2026 --compare-year 2024
    uv run python scripts/probe_2026.py --session-key 9999
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import requests
import typer

from f1_predictor.ingest import OPENF1_BASE, _fetch

app = typer.Typer(add_completion=False)

# Endpoints cheap enough to pull whole at session level. car_data is handled
# separately (one driver only) because a session-wide query 422s.
_SESSION_ENDPOINTS = [
    "sessions", "drivers", "laps", "position", "intervals",
    "stints", "race_control", "session_result",
]

# Substrings that would flag the new overtaking / aero regime in race_control.
_AERO_KEYWORDS = ["aero", "active aero", "override", "drs", "mode", "x-mode", "z-mode"]


def _first_race_key(s: requests.Session, year: int) -> int | None:
    """The session_key of the first main Grand Prix race of a season, or None."""
    data = _fetch(s, f"{OPENF1_BASE}/sessions?year={year}&session_type=Race")
    races = [d for d in data if d.get("session_name") == "Race"]
    races.sort(key=lambda d: d.get("date_start", ""))
    return int(races[0]["session_key"]) if races else None


def _fields(rows: list[dict]) -> list[str]:
    """Union of keys seen across rows (OpenF1 rows can be ragged), sorted."""
    keys: set[str] = set()
    for r in rows:
        keys.update(r.keys())
    return sorted(keys)


def _probe_car_data(s: requests.Session, session_key: int) -> dict:
    """Pull car_data for one driver and inspect its channels, esp. `drs`."""
    drv = _fetch(s, f"{OPENF1_BASE}/drivers?session_key={session_key}")
    if not drv:
        return {"available": False, "reason": "no drivers"}
    dn = drv[0]["driver_number"]
    rows = _fetch(s, f"{OPENF1_BASE}/car_data?session_key={session_key}&driver_number={dn}")
    if not rows:
        return {"available": False, "reason": "no car_data rows", "driver_number": dn}

    out = {
        "available": True,
        "driver_number": dn,
        "n_rows_sampled": len(rows),
        "fields": _fields(rows),
        "has_drs": "drs" in rows[0],
    }
    if out["has_drs"]:
        vc = Counter(r.get("drs") for r in rows)
        out["drs_value_counts"] = {str(k): v for k, v in vc.most_common()}
    return out


def _scan_race_control(rows: list[dict]) -> dict:
    """Surface race_control messages mentioning aero / override / DRS keywords."""
    hits: list[str] = []
    for r in rows:
        msg = str(r.get("message", "")).lower()
        if any(k in msg for k in _AERO_KEYWORDS):
            hits.append(r.get("message", ""))
    return {
        "categories": sorted({str(r.get("category")) for r in rows}),
        "keyword_hits": sorted(set(hits))[:50],
        "n_keyword_hits": len(hits),
    }


def _probe_session(s: requests.Session, session_key: int) -> dict:
    """Schema + row counts for every endpoint of one session."""
    schema: dict[str, dict] = {}
    race_control_rows: list[dict] = []
    for ep in _SESSION_ENDPOINTS:
        rows = _fetch(s, f"{OPENF1_BASE}/{ep}?session_key={session_key}")
        schema[ep] = {"n_rows": len(rows), "fields": _fields(rows)}
        if ep == "race_control":
            race_control_rows = rows
    schema["car_data"] = _probe_car_data(s, session_key)
    return {
        "session_key": session_key,
        "endpoints": schema,
        "race_control_scan": _scan_race_control(race_control_rows),
    }


def _diff_fields(target: dict, reference: dict) -> dict:
    """Per-endpoint added/removed field diff (target relative to reference)."""
    diff = {}
    for ep in _SESSION_ENDPOINTS + ["car_data"]:
        t = set(target["endpoints"].get(ep, {}).get("fields", []))
        r = set(reference["endpoints"].get(ep, {}).get("fields", []))
        added, removed = sorted(t - r), sorted(r - t)
        if added or removed:
            diff[ep] = {"added": added, "removed": removed}
    return diff


@app.command()
def main(
    year: int = typer.Option(2026, "--year", help="Season to probe"),
    compare_year: int = typer.Option(2024, "--compare-year", help="Reference season to diff against"),
    session_key: int = typer.Option(None, "--session-key", help="Probe this exact session instead of year's first race"),
    out_dir: Path = typer.Option(Path("data/probe"), "--out-dir"),
) -> None:
    with requests.Session() as s:
        if session_key is None:
            session_key = _first_race_key(s, year)
            if session_key is None:
                typer.echo(f"No {year} race sessions found in OpenF1 yet."); raise typer.Exit(1)
        typer.echo(f"Probing {year} session {session_key} ...")
        target = _probe_session(s, session_key)

        ref_key = _first_race_key(s, compare_year)
        reference = _probe_session(s, ref_key) if ref_key else None

    report = {"target": target, "reference": reference,
              "year": year, "compare_year": compare_year}
    if reference is not None:
        report["field_diff_vs_reference"] = _diff_fields(target, reference)

    # ---- summary ----
    typer.echo(f"\n=== {year} session {session_key} endpoint schema ===")
    for ep, info in target["endpoints"].items():
        if ep == "car_data":
            continue
        typer.echo(f"  {ep:<15} rows={info['n_rows']:<7} fields={info['fields']}")

    cd = target["endpoints"]["car_data"]
    typer.echo(f"\n=== car_data (driver {cd.get('driver_number')}) ===")
    typer.echo(f"  available={cd['available']}  fields={cd.get('fields')}")
    typer.echo(f"  has_drs channel: {cd.get('has_drs')}")
    if cd.get("drs_value_counts"):
        typer.echo(f"  drs value counts: {cd['drs_value_counts']}")

    rc = target["race_control_scan"]
    typer.echo(f"\n=== race_control aero/override/DRS scan ===")
    typer.echo(f"  categories: {rc['categories']}")
    typer.echo(f"  keyword hits ({rc['n_keyword_hits']}):")
    for h in rc["keyword_hits"]:
        typer.echo(f"    - {h}")

    if report.get("field_diff_vs_reference"):
        typer.echo(f"\n=== field diff vs {compare_year} (session {reference['session_key']}) ===")
        for ep, d in report["field_diff_vs_reference"].items():
            typer.echo(f"  {ep}: +{d['added']}  -{d['removed']}")
    elif reference is not None:
        typer.echo(f"\n  no field differences vs {compare_year}")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"probe_{year}_{session_key}.json"
    out_path.write_text(json.dumps(report, indent=2))
    typer.echo(f"\nWrote {out_path}")


if __name__ == "__main__":
    app()
