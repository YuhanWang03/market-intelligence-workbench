import json
import time

import httpx
import pytest

from v2.agent_v3.brain import ModelBrain
from v2.agent_v3.context import RunContext
from v2.agent_v3.runtime import build_model


def configure(monkeypatch):
    monkeypatch.setenv("AGENT_V3_MODEL", "fixture")
    monkeypatch.setenv("AGENT_V3_BASE_URL", "https://model.invalid/v1")
    monkeypatch.setenv("AGENT_V3_API_KEY", "offline-placeholder")


def test_explicit_thinking_mode_reaches_structured_request(monkeypatch):
    configure(monkeypatch)
    monkeypatch.setenv("AGENT_V3_THINKING", "disabled")

    def respond(request):
        body = json.loads(request.content)
        assert body["thinking"] == {"type": "disabled"}
        assert body["tool_choice"]["function"]["name"] == "SemanticIntent"
        return httpx.Response(200, json={"id":"fixture", "model":"fixture", "object":"chat.completion",
            "choices":[{"index":0,"finish_reason":"tool_calls","message":{"role":"assistant","content":None,
            "tool_calls":[{"id":"call-1","type":"function","function":{"name":"SemanticIntent",
            "arguments":json.dumps({"kind":"knowledge"})}}]}}],
            "usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}})

    model = build_model()
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        from openai import OpenAI
        model.client = OpenAI(api_key="offline-placeholder",base_url="https://model.invalid/v1",http_client=client).chat.completions
        result = ModelBrain(model).classify("Explain valuation",{},RunContext("test",time.monotonic()+5))
    assert result.kind == "knowledge"


def test_invalid_thinking_config_fails_before_network(monkeypatch):
    configure(monkeypatch)
    monkeypatch.setenv("AGENT_V3_THINKING", "sometimes")
    with pytest.raises(ValueError,match="AGENT_V3_THINKING"):
        build_model()


def test_quote_semantics_select_structured_market_read():
    from v2.agent_v3.contracts import SemanticIntent
    from v2.agent_v3.tools import Registry
    from v2.agent_v2.models import NormalizedRequest
    from v2.agent_v2.routing import route
    request = NormalizedRequest("fixture", "fixture", entities=("NVDA",))
    decision = route(request, intent=SemanticIntent(kind="lookup", tickers=["NVDA"], wants=["performance"]).domain())
    plan = ModelBrain(None).plan(request,decision,Registry(),RunContext("test",time.monotonic()+5))
    assert [(task.capability,task.arguments) for task in plan.tasks] == [("market.performance",{"ticker":"NVDA"})]


def test_failed_evidence_followup_does_not_query_again():
    from v2.agent_v3.demo import build_demo_agent
    from v2.agent_v3.contracts import SemanticIntent
    agent = build_demo_agent()
    agent.store.put("test", {"previous":{"answer":"No evidence available", "results":[], "evidence":[]}})
    agent.brain.classify = lambda *args: SemanticIntent(kind="lookup", refers_back=True, follow_up_mode="restate")
    agent.brain.plan = lambda *args: pytest.fail("A restatement must not execute another plan")
    try:
        result = agent.run("Restate the previous result", session_id="test")
        assert result.answer == "No evidence available"
        assert result.stop_reason == "follow_up_without_evidence"
        assert result.results == []
        assert result.status.value == "partial"
    finally:
        agent.store.close()
