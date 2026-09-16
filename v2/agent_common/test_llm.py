"""The shared chat-completions client: request shape, not the network."""
from __future__ import annotations

import io
import json
import sys
import types

import pytest

from v2.agent_common.llm import OpenAICompatLLM


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def _capture(monkeypatch):
    """Route the client's single HTTP call into a list and stub the usage ledger."""

    sent: list[dict] = []

    def fake_urlopen(request, timeout=None):
        sent.append(json.loads(request.data.decode("utf-8")))
        body = {"choices": [{"message": {"content": "ok"}}], "usage": {}}
        return _Response(json.dumps(body).encode("utf-8"))

    ledger = types.ModuleType("v2.data.usage_ledger")
    ledger.record_llm = lambda *a, **kw: None
    ledger.record = lambda *a, **kw: None
    monkeypatch.setitem(sys.modules, "v2.data.usage_ledger", ledger)
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    return sent


def test_thinking_is_off_the_wire_unless_configured(monkeypatch):
    monkeypatch.delenv("AGENT_LLM_THINKING", raising=False)
    sent = _capture(monkeypatch)
    OpenAICompatLLM(api_key="k").complete([{"role": "user", "content": "hi"}])
    assert "thinking" not in sent[0]


def test_env_switch_disables_thinking_on_every_call(monkeypatch):
    monkeypatch.setenv("AGENT_LLM_THINKING", "disabled")
    sent = _capture(monkeypatch)
    llm = OpenAICompatLLM(api_key="k")
    llm.complete([{"role": "user", "content": "hi"}])
    llm.complete([{"role": "user", "content": "hi"}], tools=[{"type": "function", "function": {"name": "finish", "parameters": {}}}], tool_choice={"type": "function", "function": {"name": "finish"}})
    assert [call["thinking"] for call in sent] == [{"type": "disabled"}, {"type": "disabled"}]
    assert sent[1]["tool_choice"]["function"]["name"] == "finish"


def test_constructor_argument_wins_over_env_and_is_validated(monkeypatch):
    monkeypatch.setenv("AGENT_LLM_THINKING", "disabled")
    assert OpenAICompatLLM(api_key="k", thinking="enabled").thinking == "enabled"
    monkeypatch.setenv("AGENT_LLM_THINKING", "sometimes")
    with pytest.raises(ValueError, match="AGENT_LLM_THINKING"):
        OpenAICompatLLM(api_key="k")
