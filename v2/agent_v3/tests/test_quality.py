from langchain_core.messages import AIMessage

from v2.agent_v3.contracts import SemanticIntent
from v2.agent_v3.quality_eval import evaluate
from v2.agent_v3.tests.test_specialists import ScriptedModel, call


def test_synthesis_uses_short_citation_handles_and_restores_only_known_ids():
    import time
    from v2.agent_v3.brain import ModelBrain
    from v2.agent_v3.context import RunContext
    from v2.agent_v3.tools import Registry
    original="evidence-market-price-a1744854192cf6bf"
    model=ScriptedModel(responses=[AIMessage(content="价格120美元。[E1] 未知引用[E99]")])
    state={"text":"price","evidence":[{"id":original,"entity":"EXMP","claim":"价格120美元"}]}
    answer=ModelBrain(model).draft(state,Registry(),RunContext("citation",time.monotonic()+10))
    assert f"[{original}]" in answer and "[E99]" in answer
    assert original not in model._seen[0][-1].content and '"id": "E1"' in model._seen[0][-1].content


def test_presentation_is_v3_only_and_survives_followup():
    semantic = SemanticIntent(kind="lookup",response_style="brief")
    assert "response_style" not in semantic.domain().to_dict()
    responses = []
    for index in range(3):
        responses.extend([call("SemanticIntent",{"kind":"lookup","tickers":["NVDA"],"wants":["performance"],"response_style":"brief","refers_back":index==2,"follow_up_mode":"restate" if index==2 else "expand"},index),
            AIMessage(content="合成评测数据中 NVDA 在 2026-09-11 收盘价为 120 美元。[fixture-close]")])
    responses.append(call("RestatementVerdict", {"adds_facts":False}, 4))
    model = ScriptedModel(responses=responses)
    report = evaluate(model)
    assert [row["observations"]["new_tool_calls"] for row in report["runs"]] == [1,1,0]
    assert all(row["result"]["verification"]["ok"] for row in report["runs"])


def test_unknown_price_source_does_not_invent_provenance():
    from v2.agent_v3.market import source_descriptor
    assert source_descriptor(object(),"NVDA") == {"provider":"","url":""}


def test_known_source_records_provider_and_symbol_without_calling_network():
    from v2.agent_v3.market import source_descriptor
    from v2.data.price_source import YFinancePriceSource, FDPriceSource
    assert source_descriptor(YFinancePriceSource(),"BRK.B")["url"] == "https://finance.yahoo.com/quote/BRK-B/history/"
    assert source_descriptor(FDPriceSource(None),"NVDA")["provider"] == "Financial Datasets"


def test_provenance_follows_actual_source_and_preserves_evidence(monkeypatch):
    from dataclasses import replace
    from v2.agent_v3.market import register_market_capabilities
    from v2.agent_v3.quality_eval import fixture
    from v2.agent_v3.tools import Registry
    from v2.data.price_source import YFinancePriceSource
    from v2.agent_v2.adapters import market
    original = fixture()
    original.evidence[1] = replace(original.evidence[1],metadata={"benchmark":"SPY"})
    source = YFinancePriceSource()
    def calculate(ticker, context, supplied_source, *, now):
        assert getattr(supplied_source,'upstream',supplied_source) is source
        return original
    monkeypatch.setattr(market,"_performance_envelope",calculate)
    registry = Registry()
    register_market_capabilities(registry,price_source_factory=lambda:source)
    result = registry.handlers["market.performance"]({"ticker":"NVDA"},None)
    assert result.evidence[0].claim == original.evidence[0].claim
    assert result.evidence[0].source_title == "Yahoo Finance via yfinance"
    assert result.evidence[1].metadata["sources"][1]["url"].endswith("/SPY/history/")
    assert original.evidence[0].source_title == "合成评测数据（非真实行情）"


def test_historical_attribution_does_not_use_current_market_date(monkeypatch):
    from datetime import datetime, timezone
    from v2.agent_v3.market import register_market_capabilities
    from v2.agent_v3.quality_eval import fixture
    from v2.agent_v3.tools import Registry
    from v2.agent_v2.adapters import market
    seen=[]
    def calculate(ticker, context, source, *, now):
        seen.append(now.date().isoformat())
        return fixture()
    monkeypatch.setattr(market,"_performance_envelope",calculate)
    registry=Registry()
    register_market_capabilities(registry,price_source_factory=lambda:object(),now_factory=lambda:datetime(2026,9,13,tzinfo=timezone.utc))
    registry.handlers["market.performance"]({"ticker":"NVDA","_as_of":"2026-08-20"},None)
    assert seen==["2026-08-20"]
