"""Agent V3 model calls reach the shared usage ledger the way V2's do."""
import sys
import time
import types
from types import SimpleNamespace

from v2.agent_v3.context import RunContext, model_identity


def _ledger(monkeypatch):
    calls = []
    module = types.ModuleType("v2.data.usage_ledger")
    module.record = lambda *args, **kwargs: calls.append((args, kwargs))
    monkeypatch.setitem(sys.modules, "v2.data.usage_ledger", module)
    return calls


def _message(**usage):
    return SimpleNamespace(usage_metadata=usage, response_metadata={"model_name": "deepseek-v4-flash"})


def test_model_identity_uses_the_v2_provider_rule():
    deepseek = SimpleNamespace(model_name="deepseek-v4-flash", openai_api_base="https://api.deepseek.com/v1")
    other = SimpleNamespace(model_name="gpt-x", openai_api_base="https://api.example.com/v1")
    assert model_identity(deepseek) == ("DeepSeek", "deepseek-v4-flash")
    assert model_identity(other) == ("Other LLM", "gpt-x")
    assert model_identity(None) == ("", "")
    assert model_identity(object()) == ("", "")


def test_record_writes_a_normalised_llm_row(monkeypatch):
    calls = _ledger(monkeypatch)
    run = RunContext("r1", time.monotonic() + 10, provider="DeepSeek", model="deepseek-v4-flash")
    run.record("synthesizer", _message(input_tokens=1200, output_tokens=300, total_tokens=1500, input_token_details={"cache_read": 1000}, output_token_details={"reasoning": 40}))
    assert run.usage[0]["source"] == "synthesizer" and run.usage[0]["input_tokens"] == 1200
    (category, provider, model, usage), options = calls[0]
    assert (category, provider, model) == ("llm", "DeepSeek", "deepseek-v4-flash")
    assert usage == {"input_tokens": 1200, "output_tokens": 300, "cached_tokens": 1000, "reasoning_tokens": 40}
    assert options == {"source": "agent_v3.synthesizer", "endpoint": "chat", "usage_basis": "reported", "requested_model": "deepseek-v4-flash"}


def test_runs_without_a_provider_are_not_accounted(monkeypatch):
    calls = _ledger(monkeypatch)
    run = RunContext("r2", time.monotonic() + 10)  # demo brain / scripted model
    run.record("debater", _message(input_tokens=5, output_tokens=1))
    assert run.usage and calls == []
