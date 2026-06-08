import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import json
import polars as pl
from f1_predictor.ingest import pull_session, pull_season, ENDPOINTS

def make_mock_response(data: list[dict]) -> MagicMock:
    m = MagicMock()
    m.json.return_value = data
    m.raise_for_status.return_value = None
    return m

def test_pull_session_creates_parquets(tmp_path):
    responses = {ep: [{"session_key": 9161, "driver_number": 1}] for ep in ENDPOINTS}

    call_count = 0
    def fake_get(url, **kwargs):
        nonlocal call_count
        call_count += 1
        for ep in ENDPOINTS:
            if f"/{ep}" in url:
                return make_mock_response(responses[ep])
        return make_mock_response([])

    with patch("f1_predictor.ingest.requests.Session") as MockSession:
        session_obj = MagicMock()
        session_obj.get.side_effect = fake_get
        MockSession.return_value.__enter__ = lambda s: session_obj
        MockSession.return_value.__exit__ = MagicMock(return_value=False)
        pull_session(9161, tmp_path)

    session_dir = tmp_path / "9161"
    assert session_dir.exists()
    for ep in ENDPOINTS:
        assert (session_dir / f"{ep}.parquet").exists(), f"missing {ep}.parquet"
    meta = json.loads((session_dir / "meta.json").read_text())
    assert meta["session_key"] == 9161
    assert "pull_timestamp" in meta

def _status_response(status_code, data, headers=None):
    m = MagicMock()
    m.status_code = status_code
    m.json.return_value = data
    m.headers = headers or {}
    m.raise_for_status.return_value = None
    return m


def test_pull_session_treats_404_as_empty(tmp_path):
    # OpenF1 answers zero-row queries with 404 {"detail": "No results found."}.
    # A 404 on an endpoint must mean "no data" (empty parquet), not a crash.
    def fake_get(url, **kwargs):
        if "/stints" in url:
            return _status_response(404, {"detail": "No results found."})
        return _status_response(200, [{"session_key": 9161}])

    with patch("f1_predictor.ingest.requests.Session") as MockSession:
        session_obj = MagicMock()
        session_obj.get.side_effect = fake_get
        MockSession.return_value.__enter__ = lambda s: session_obj
        MockSession.return_value.__exit__ = MagicMock(return_value=False)
        pull_session(9161, tmp_path)

    stints = pl.read_parquet(tmp_path / "9161" / "stints.parquet")
    assert stints.height == 0, "404 endpoint should produce an empty parquet"
    laps = pl.read_parquet(tmp_path / "9161" / "laps.parquet")
    assert laps.height == 1


def test_pull_session_treats_422_as_empty(tmp_path):
    # Oversized queries (e.g. car_data) return 422 "too much data" — treat as
    # no data rather than crashing the whole pull.
    def fake_get(url, **kwargs):
        if "/intervals" in url:
            return _status_response(422, {"detail": "too much data at once"})
        return _status_response(200, [{"session_key": 9161}])

    with patch("f1_predictor.ingest.requests.Session") as MockSession:
        session_obj = MagicMock()
        session_obj.get.side_effect = fake_get
        MockSession.return_value.__enter__ = lambda s: session_obj
        MockSession.return_value.__exit__ = MagicMock(return_value=False)
        pull_session(9161, tmp_path)

    intervals = pl.read_parquet(tmp_path / "9161" / "intervals.parquet")
    assert intervals.height == 0, "422 endpoint should produce an empty parquet"


def test_pull_session_retries_on_429(tmp_path):
    # /laps returns 429 twice, then 200. The pull should back off and succeed.
    state = {"laps_429": 0}

    def fake_get(url, **kwargs):
        if "/laps" in url and state["laps_429"] < 2:
            state["laps_429"] += 1
            return _status_response(429, {"detail": "rate limited"})
        return _status_response(200, [{"session_key": 9161}])

    with (
        patch("f1_predictor.ingest.requests.Session") as MockSession,
        patch("f1_predictor.ingest.time.sleep"),  # don't actually sleep in tests
    ):
        session_obj = MagicMock()
        session_obj.get.side_effect = fake_get
        MockSession.return_value.__enter__ = lambda s: session_obj
        MockSession.return_value.__exit__ = MagicMock(return_value=False)
        pull_session(9161, tmp_path)

    assert state["laps_429"] == 2, "should have retried past both 429s"
    laps = pl.read_parquet(tmp_path / "9161" / "laps.parquet")
    assert laps.height == 1


def test_retry_wait_honours_retry_after():
    from f1_predictor.ingest import _retry_wait
    resp = _status_response(429, {}, headers={"Retry-After": "1"})
    # Header takes precedence over exponential backoff, even on a late attempt.
    assert _retry_wait(resp, attempt=5) == 1.0
    # No header → exponential backoff fallback.
    resp_nohdr = _status_response(429, {}, headers={})
    assert _retry_wait(resp_nohdr, attempt=0) == 0.5


def test_pull_session_handles_mixed_type_column(tmp_path):
    # OpenF1 /intervals reports gap_to_leader as a float for most drivers but a
    # string like "+1 LAP" for lapped drivers — and lapped rows can appear well
    # past the first 100 rows, defeating Polars' default schema inference.
    mixed_rows = [{"driver_number": 1, "gap_to_leader": float(i)} for i in range(150)]
    mixed_rows[140]["gap_to_leader"] = "+1 LAP"

    def fake_get(url, **kwargs):
        if "/intervals" in url:
            return make_mock_response(mixed_rows)
        return make_mock_response([{"session_key": 9161}])

    with patch("f1_predictor.ingest.requests.Session") as MockSession:
        session_obj = MagicMock()
        session_obj.get.side_effect = fake_get
        MockSession.return_value.__enter__ = lambda s: session_obj
        MockSession.return_value.__exit__ = MagicMock(return_value=False)
        # Must not raise ComputeError on the mixed-type column.
        pull_session(9161, tmp_path)

    intervals = pl.read_parquet(tmp_path / "9161" / "intervals.parquet")
    assert intervals.height == 150
    assert "+1 LAP" in intervals["gap_to_leader"].to_list()


def test_pull_session_skips_if_cached(tmp_path):
    session_dir = tmp_path / "9161"
    session_dir.mkdir()
    (session_dir / "meta.json").write_text('{"session_key": 9161, "pull_timestamp": "2026-01-01"}')

    with patch("f1_predictor.ingest.requests.Session") as MockSession:
        pull_session(9161, tmp_path)
        MockSession.assert_not_called()

def test_pull_session_force_repulls(tmp_path):
    session_dir = tmp_path / "9161"
    session_dir.mkdir()
    (session_dir / "meta.json").write_text('{"session_key": 9161, "pull_timestamp": "2026-01-01"}')

    with patch("f1_predictor.ingest.requests.Session") as MockSession:
        session_obj = MagicMock()
        session_obj.get.return_value = make_mock_response([{"session_key": 9161}])
        MockSession.return_value.__enter__ = lambda s: session_obj
        MockSession.return_value.__exit__ = MagicMock(return_value=False)
        pull_session(9161, tmp_path, force=True)
        assert session_obj.get.called


def test_pull_season_excludes_monaco(tmp_path):
    # OpenF1 returns the Monaco circuit as "Monte Carlo". Montreal and Monza
    # share the "mon" prefix and must NOT be excluded.
    sessions = [
        {"session_key": 9001, "circuit_short_name": "Sakhir", "session_name": "Race"},
        {"session_key": 9002, "circuit_short_name": "Monte Carlo", "session_name": "Race"},
        {"session_key": 9003, "circuit_short_name": "Catalunya", "session_name": "Race"},
        {"session_key": 9004, "circuit_short_name": "Montreal", "session_name": "Race"},
        {"session_key": 9005, "circuit_short_name": "Monza", "session_name": "Race"},
    ]

    pulled = []

    def fake_pull_session(key, raw_dir, force=False):
        pulled.append(key)

    with (
        patch("f1_predictor.ingest.requests.Session") as MockSession,
        patch("f1_predictor.ingest.pull_session", side_effect=fake_pull_session),
    ):
        session_obj = MagicMock()
        session_obj.get.return_value = make_mock_response(sessions)
        MockSession.return_value.__enter__ = lambda s: session_obj
        MockSession.return_value.__exit__ = MagicMock(return_value=False)

        keys = pull_season(2023, tmp_path)

    assert 9002 not in keys, "Monaco (Monte Carlo) must be excluded"
    assert 9001 in keys
    assert 9003 in keys
    assert 9004 in keys, "Montreal must NOT be excluded"
    assert 9005 in keys, "Monza must NOT be excluded"


def test_pull_season_excludes_sprints(tmp_path):
    # The year=...&session_type=Race query also returns Sprint sessions.
    sessions = [
        {"session_key": 9001, "circuit_short_name": "Sakhir", "session_name": "Race"},
        {"session_key": 9069, "circuit_short_name": "Baku", "session_name": "Sprint"},
        {"session_key": 9070, "circuit_short_name": "Baku", "session_name": "Race"},
    ]

    with (
        patch("f1_predictor.ingest.requests.Session") as MockSession,
        patch("f1_predictor.ingest.pull_session"),
    ):
        session_obj = MagicMock()
        session_obj.get.return_value = make_mock_response(sessions)
        MockSession.return_value.__enter__ = lambda s: session_obj
        MockSession.return_value.__exit__ = MagicMock(return_value=False)

        keys = pull_season(2023, tmp_path)

    assert 9069 not in keys, "Sprint sessions must be excluded"
    assert 9001 in keys
    assert 9070 in keys


def test_pull_season_returns_session_keys(tmp_path):
    sessions = [
        {"session_key": 9001, "circuit_short_name": "Bahrain", "session_name": "Race"},
        {"session_key": 9003, "circuit_short_name": "Spain", "session_name": "Race"},
    ]

    with (
        patch("f1_predictor.ingest.requests.Session") as MockSession,
        patch("f1_predictor.ingest.pull_session"),
    ):
        session_obj = MagicMock()
        session_obj.get.return_value = make_mock_response(sessions)
        MockSession.return_value.__enter__ = lambda s: session_obj
        MockSession.return_value.__exit__ = MagicMock(return_value=False)

        keys = pull_season(2023, tmp_path)

    assert keys == [9001, 9003]
