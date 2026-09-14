from __future__ import annotations

import threading
import time

from fastapi.testclient import TestClient

from app.main import app
from app.routers import research as research_router


def _research_result(ticker: str) -> dict:
    return {
        "run_id": f"run-{ticker}",
        "ticker": ticker,
        "status": "COMPLETED",
        "modules": {
            "supply_chain": {
                "status": "COMPLETED",
                "summary": f"{ticker} chain result",
                "metrics": {"relationship_count": 1},
                "details": {
                    "relationships": [{
                        "source_company": ticker,
                        "target_company": "SUPPLIER",
                        "relationship_type": "supplier",
                        "description": "validated relationship",
                        "verified": True,
                    }],
                },
            },
        },
    }


def _reset_jobs() -> None:
    research_router._JOBS.clear()
    research_router._ACTIVE_BY_TICKER.clear()


def _poll(client: TestClient, job_id: str) -> dict:
    deadline = time.time() + 2
    while time.time() < deadline:
        payload = client.get(f"/api/chat/jobs/{job_id}").json()
        if payload["status"] != "running":
            return payload
        time.sleep(0.02)
    raise AssertionError("chat job did not complete")


def test_chain_runs_as_background_job(monkeypatch):
    _reset_jobs()

    class FakeEngine:
        def run(self, ticker, *, refresh=False, progress=None):
            return _research_result(ticker)

    monkeypatch.setattr(research_router, "_new_engine", FakeEngine)

    with TestClient(app) as client:
        started = client.post("/api/chat", json={"text": "/chain AAPL"})
        assert started.status_code == 200
        assert started.json()["status"] == "running"

        completed = _poll(client, started.json()["job_id"])
        assert completed["status"] == "completed"
        assert completed["intent"] == "chain"
        assert "AAPL · 产业链" in completed["html"]
        assert completed["data"]["relationships"][0]["target_company"] == "SUPPLIER"


def test_duplicate_running_chain_is_deduplicated(monkeypatch):
    _reset_jobs()
    release = threading.Event()

    class SlowEngine:
        def run(self, ticker, *, refresh=False, progress=None):
            release.wait(timeout=1)
            return _research_result(ticker)

    monkeypatch.setattr(research_router, "_new_engine", SlowEngine)

    with TestClient(app) as client:
        first = client.post("/api/chat", json={"text": "/chain TSLA"}).json()
        second = client.post("/api/chat", json={"text": "/chain TSLA"}).json()
        assert first["job_id"] == second["job_id"]
        assert second["deduplicated"] is True

        release.set()
        assert _poll(client, first["job_id"])["status"] == "completed"
