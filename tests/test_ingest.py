import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import json
import polars as pl
from f1_predictor.ingest import pull_session, ENDPOINTS

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
