import time
from datetime import date
from v2.agent_v3.brain import ModelBrain
from v2.agent_v3.context import RunContext
from v2.agent_v3.tests.test_specialists import ScriptedModel, call
from v2.agent_v3.tests.test_source_support import run
from v2.agent_v3.fomc import read_statements, CALENDAR


def test_judge_cannot_quote_rule_as_answer_violation():
    model = ScriptedModel(responses=[call("ClaimVerdict", {"violations":[{"index":0,"quote":"未授权","reason":"fixture"}]},1)])
    result = ModelBrain(model).judge([{"id":"permission","text":"本轮检索到了报道。","claim":"未授权"}],RunContext("test",time.monotonic()+5))
    assert result == {}


def test_judge_preserves_actual_violation_quote():
    model = ScriptedModel(responses=[call("ClaimVerdict", {"violations":[{"index":0,"quote":"本轮未授权","reason":"fixture"}]},1)])
    result = ModelBrain(model).judge([{"id":"permission","text":"本轮未授权网页检索。","claim":"未授权"}],RunContext("test",time.monotonic()+5))
    assert result == {"permission":"本轮未授权"}


def test_news_numbers_must_trace_even_if_semantic_judge_accepts():
    quote = "The shares rose 6% to 7% during the session."
    source = lambda *a,**k:[{"url":"https://example.org/news","title":"Fixture","raw_content":quote}]
    result = run([call("search_news",{"query":"fixture"},1),call("read_page",{"id":"p1"},2),
        call("FindingsReport",{"findings":[{"text":"Shares rose 5% to 7%.","source":"p1","quote":quote}]},3),
        call("SupportedFindings",{"supported_indices":[0]},4)],source=source)
    assert result.evidence == []
    assert result.metadata["rejection_counts"]["numeric_not_in_quote"] == 1


def test_fomc_only_published_official_documents_and_full_article():
    links = '<a href="/newsevents/pressreleases/monetary20260729a.htm">HTML</a><a href="/newsevents/pressreleases/monetary20260916a.htm">HTML</a>'
    body = "The Committee decided to maintain its target range. " * 8
    def get(url):
        return links if url == CALENDAR else '<div id="article"><p>'+body+'</p></div><footer>not article</footer>'
    result = read_statements(get=get,today=date(2026,9,13))
    assert len(result.evidence) == 1 and result.evidence[0].as_of == "2026-07-29"
    assert result.evidence[0].claim == body.strip()


def test_completed_tools_survive_interrupted_graph_and_new_agent(tmp_path):
    import sqlite3
    import pytest
    from langgraph.checkpoint.sqlite import SqliteSaver
    from v2.agent_v3.demo import build_demo_agent
    from v2.agent_v3.persistence import SessionStore
    from v2.agent_v2.models import ExecutionPlan, PlanTask, RouteKind
    class Crash(BaseException):
        pass
    calls = []
    conn = sqlite3.connect(tmp_path / "checkpoint.sqlite",check_same_thread=False)
    store = SessionStore(tmp_path / "session.sqlite")
    def build(crash):
        agent = build_demo_agent(store=store,checkpointer=SqliteSaver(conn))
        agent.brain.plan = lambda *args: ExecutionPlan("fixture",RouteKind.FAST_LOOKUP,tasks=(PlanTask("first","market.performance",{"ticker":"NVDA"}),PlanTask("second","market.performance",{"ticker":"NVDA"},depends_on=("first",))))
        base = agent.registry.handlers["market.performance"]
        def reader(args,context):
            calls.append(context.run_id)
            if crash and len(calls) == 2:
                raise Crash()
            return base(args,context)
        agent.registry.register("market.performance",reader)
        return agent
    try:
        first = build(True)
        with pytest.raises(Crash):
            first.run("fixture",session_id="recovery")
        recovered = build(False).recover(session_id="recovery",run_id=calls[0])
        assert recovered.answer and len(calls) == 3
        assert recovered.run_id == calls[0]
    finally:
        store.close()
        conn.close()


def test_issuer_file_preserves_full_rows_and_date_without_normalizing():
    from v2.agent_v3.fund_holdings import issuer_holdings
    fixture = 'Fund Holdings as of,"Sep 10, 2026"\nTicker,Name,Asset Class,Weight (%)\nAAA,Company A,Equity,8.0\nBBB,Company B,Equity,7.0\n'
    result = issuer_holdings("IVV",top=1,get=lambda url:fixture)
    assert result.as_of == "2026-09-10"
    assert len(result.metadata["holdings"]) == 2
    assert result.evidence[0].value == 8.0
    assert result.metadata["weight_sum_pct"] == 15.0


def test_source_repair_is_reviewed_again_and_keeps_original_quote():
    from v2.agent_v3.specialists import register_specialists
    from v2.agent_v3.tools import Registry
    from v2.agent_v2.models import PlanTask, ExecutionPlan, NormalizedRequest, RouteKind
    quote = "The shares rose 6% to 7% during the session."
    original = {"text":"Shares rose 5% to 7%.","source":"p1","quote":quote}
    fixed = {**original,"text":"Shares rose 6% to 7%."}
    model = ScriptedModel(responses=[call("search_news",{"query":"fixture"},1),call("read_page",{"id":"p1"},2),call("FindingsReport",{"findings":[original]},3),call("SupportedFindings",{"supported_indices":[]},4),call("SourceRepairs",{"repairs":[{"index":0,"text":fixed["text"]}]},5),call("SupportedFindings",{"supported_indices":[0]},6)])
    registry = Registry()
    register_specialists(registry,model,search=lambda *a,**k:[{"url":"https://example.org/news","title":"Fixture","raw_content":quote}])
    result = registry.execute(PlanTask("news","web.research",{"query":"fixture","topic":"news"}),ExecutionPlan("fixture",RouteKind.RESEARCH),NormalizedRequest("fixture","fixture",allow_web=True),RunContext("test",time.monotonic()+30))
    assert len(result.evidence)==1 and "5%" not in result.evidence[0].claim
    assert result.evidence[0].metadata["quote"] == quote


def test_issuer_failure_is_visible_and_does_not_claim_complete_holdings(monkeypatch):
    import pandas as pd
    import yfinance as yf
    from types import SimpleNamespace
    from v2.agent_v3 import fund_holdings
    from v2.agent_v3.data_extensions import register_data_extensions
    from v2.agent_v3.tools import Registry
    def unavailable(*args):
        raise OSError("provider unavailable")
    monkeypatch.setattr(fund_holdings,"issuer_holdings",unavailable)
    monkeypatch.setattr(yf,"Ticker",lambda ticker:SimpleNamespace(funds_data=SimpleNamespace(top_holdings=pd.DataFrame({"Holding Percent":[.08],"Name":["Company"]},index=["AAA"]))))
    registry = Registry()
    register_data_extensions(registry)
    result = registry.handlers["etf.holdings"]({"ticker":"IVV"},None)
    assert result.status.value == "partial_data"
    assert not result.metadata.get("complete_provider_file")
    assert result.evidence[0].as_of == "" and any("读取失败" in note for note in result.limitations)
