from __future__ import annotations

import time

from app.main import app
from app.routers import agent_v2
from fastapi.testclient import TestClient


class _Result:
    def __init__(self, text: str) -> None:
        self.text = text

    def to_dict(self):
        return {
            "run_id": "agent-v2-test",
            "status": "completed",
            "answer": f"answer: {self.text}",
            "route": {"kind": "general_knowledge"},
        }


class _Agent:
    def __init__(self) -> None:
        self.calls = []

    def run(self, text, **kwargs):
        self.calls.append((text, kwargs))
        return _Result(text)


def test_agent_v2_web_entrypoint_is_independent_and_web_is_double_gated(monkeypatch):
    fake = _Agent()
    monkeypatch.setattr(agent_v2, "_AGENT", fake)
    monkeypatch.setattr(agent_v2, "_AGENT_WEB_ENABLED", True)
    monkeypatch.setenv("AGENT_V2_WEB_ENABLED", "true")
    with TestClient(app) as client:
        response = client.post(
            "/api/agent-v2/ask",
            json={"text": "什么是自由现金流？", "session_id": "browser-1", "allow_web": True},
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["interface"] == "web"
    assert payload["policy"] == {"web_requested": True, "web_enabled": True, "web_allowed": True}
    assert fake.calls[0][1]["session_id"] == "browser-1"
    assert fake.calls[0][1]["allow_web"] is True


def test_agent_v2_long_request_runs_as_a_pollable_job(monkeypatch):
    fake = _Agent()
    monkeypatch.setattr(agent_v2, "_AGENT", fake)
    monkeypatch.setattr(agent_v2, "_AGENT_WEB_ENABLED", False)
    monkeypatch.delenv("AGENT_V2_WEB_ENABLED", raising=False)
    with agent_v2._JOBS_LOCK:
        agent_v2._JOBS.clear()
    with TestClient(app) as client:
        started = client.post(
            "/api/agent-v2/ask",
            json={"text": "完整报告：回测标普动量策略", "session_id": "job-1"},
        ).json()
        assert started["job_id"]
        final = started
        for _ in range(100):
            final = client.get(f"/api/agent-v2/jobs/{started['job_id']}").json()
            if final["status"] != "running":
                break
            time.sleep(0.01)
    assert final["status"] == "completed"
    assert final["result"]["answer"].startswith("answer:")


def test_agent_v2_job_poller_returns_404_for_unknown_job():
    with TestClient(app) as client:
        response = client.get("/api/agent-v2/jobs/missing")
    assert response.status_code == 404
