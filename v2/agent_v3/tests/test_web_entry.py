import sys
import time
from pathlib import Path
import pytest

pytest.importorskip("fastapi")
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/"web/backend"))
from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.routers import agent_v3
from app.auth import require_owner
from v2.agent_v3.demo import build_demo_agent


def test_v3_web_confirmation_round_trip(monkeypatch, tmp_path):
    from v2.agent_v2.models import EvidenceItem, ResultStatus, ToolEnvelope
    from v2.agent_v3.contracts import SemanticIntent
    from v2.agent_v3.demo import DemoBrain
    from v2.agent_v3.graph import AgentV3, AgentV3Config
    from v2.agent_v3.tools import Registry
    monkeypatch.setenv("AGENT_V3_DATA_DIR", str(tmp_path))
    calls = []
    registry = Registry()
    def mutate(args, ctx):
        calls.append(args)
        return ToolEnvelope("state.mutate", ResultStatus.COMPLETED, evidence=[EvidenceItem("written", "NVDA", "已添加 NVDA 关注。", source_id="test_state")])
    registry.register("state.mutate", mutate)
    class Write(DemoBrain):
        def classify(self, *args):
            return SemanticIntent(kind="command", tickers=["NVDA"], command={"operation": "watchlist.add", "ticker": "NVDA"})
    agent = AgentV3(registry=registry, brain=Write(), config=AgentV3Config(debate=False))
    monkeypatch.setattr(agent_v3, "_get_agent", lambda: agent)
    app = FastAPI()
    app.include_router(agent_v3.router)
    app.dependency_overrides[require_owner] = lambda: None
    try:
        with TestClient(app) as client:
            pending = client.post("/api/agent-v3/ask", json={"text": "加入关注 NVDA", "session_id": "owner", "background": False}).json()
            assert pending["status"] == "waiting_confirmation"
            assert pending["pending_mutation"] == {"operation": "watchlist.add", "payload": {"ticker": "NVDA"}, "description": pending["pending_mutation"]["description"]}
            assert calls == []
            wrong = client.post(f"/api/agent-v3/runs/{pending['run_id']}/confirm", json={"session_id": "someone-else", "approve": True})
            assert wrong.status_code == 409 and calls == []
            done = client.post(f"/api/agent-v3/runs/{pending['run_id']}/confirm", json={"session_id": "owner", "approve": True})
            assert done.status_code == 200 and done.json()["status"] == "completed" and "已添加" in done.json()["answer"]
            assert calls == [{"operation": "watchlist.add", "payload": {"ticker": "NVDA"}}]
            again = client.post(f"/api/agent-v3/runs/{pending['run_id']}/confirm", json={"session_id": "owner", "approve": True})
            assert again.status_code == 409 and len(calls) == 1
            rejected = client.post("/api/agent-v3/ask", json={"text": "加入关注 NVDA", "session_id": "owner", "background": False}).json()
            cancelled = client.post(f"/api/agent-v3/runs/{rejected['run_id']}/confirm", json={"session_id": "owner", "approve": False}).json()
            assert cancelled["status"] == "cancelled" and len(calls) == 1
            assert client.post("/api/agent-v3/runs/nope/confirm", json={"session_id": "owner"}).status_code == 422
    finally:
        agent.store.close()


def test_v3_callers_without_a_session_id_do_not_share_one(monkeypatch,tmp_path):
    monkeypatch.setenv("AGENT_V3_DATA_DIR",str(tmp_path))
    seen=[]
    class Recorder:
        def run(self,text,**kwargs):
            seen.append(kwargs["session_id"])
            return build_demo_agent().run(text,session_id=kwargs["session_id"])
    monkeypatch.setattr(agent_v3,"_get_agent",lambda:Recorder())
    app=FastAPI()
    app.include_router(agent_v3.router)
    app.dependency_overrides[require_owner]=lambda:None
    with TestClient(app) as client:
        for _ in range(2):
            client.post("/api/agent-v3/ask",json={"text":"fixture","background":False},cookies={"workbench_session":"cookie-a"})
        client.post("/api/agent-v3/ask",json={"text":"fixture","background":False})
    assert seen[0]==seen[1] and seen[0].startswith("web:") and seen[2] not in {seen[0],"web"}


def test_v3_web_job_runs_graph_and_returns_compatible_evidence(monkeypatch,tmp_path):
    monkeypatch.setenv("AGENT_V3_DATA_DIR",str(tmp_path))
    agent=build_demo_agent()
    monkeypatch.setattr(agent_v3,"_get_agent",lambda:agent)
    monkeypatch.setenv("AGENT_V3_WEB_ENABLED","0")
    app=FastAPI()
    app.include_router(agent_v3.router)
    app.dependency_overrides[require_owner]=lambda:None
    try:
        with TestClient(app) as client:
            assert client.post("/api/agent-v3/ask",json={"text":"  "}).status_code==422
            assert client.post("/api/agent-v3/ask",json={"text":"fixture","page_context":{"section":"core","allow_web":True}}).status_code==422
            initial=client.post("/api/agent-v3/ask",json={"text":"fixture","session_id":"test","allow_web":True,"page_context":{"section":"core","selection":{"kind":"position","ticker":"NVDA"}}}).json()
            for _ in range(100):
                job=client.get("/api/agent-v3/jobs/"+initial["job_id"]).json()
                if job["status"]!="running":
                    break
                time.sleep(.01)
            assert job["result"]["synthesis"]["framework"]=="langgraph"
            assert job["result"]["policy"]["web_allowed"] is False
            assert job["result"]["evidence"]
            assert job["result"]["request"]["metadata"]["page_context"]["selection"]["ticker"]=="NVDA"
            assert client.get("/api/agent-v3/jobs/missing").status_code==404
            agent_v3._JOBS.pop(initial["job_id"])
            assert client.get("/api/agent-v3/jobs/"+initial["job_id"]).json()["result"]["answer"]==job["result"]["answer"]
    finally:
        agent.store.close()


def test_v3_router_keeps_owner_auth():
    from fastapi import HTTPException
    app=FastAPI()
    app.include_router(agent_v3.router)
    def reject():
        raise HTTPException(401,"owner required")
    app.dependency_overrides[require_owner]=reject
    with TestClient(app) as client:
        assert client.post("/api/agent-v3/ask",json={"text":"fixture"}).status_code==401


def test_interrupted_job_is_durable_and_retryable(monkeypatch,tmp_path):
    monkeypatch.setenv("AGENT_V3_DATA_DIR",str(tmp_path))
    agent_v3._journal().save("old-instance", {"job_id":"old-instance", "instance":"previous-process", "status":"running", "request":{"text":"fixture","session_id":"test"}})
    result=agent_v3._job_view("old-instance")
    assert result["status"]=="failed" and result["restartable"]
    assert "request" not in result and "instance" not in result
    monkeypatch.setattr(agent_v3,"_start_job",lambda body:{"retry_text":body.text})
    app=FastAPI();app.include_router(agent_v3.router)
    app.dependency_overrides[require_owner]=lambda:None
    with TestClient(app) as client:
        assert client.post("/api/agent-v3/jobs/old-instance/retry").json()=={"retry_text":"fixture"}
