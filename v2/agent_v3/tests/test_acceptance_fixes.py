from v2.agent_v3.contracts import SemanticIntent, plain
from v2.agent_v3.demo import DemoBrain, build_demo_agent
from v2.agent_v2.models import EvidenceItem, ExecutionPlan, PlanTask, RouteKind, ResultStatus, ToolEnvelope
from v2.agent_v3.source_scope import scope_rejection
from v2.agent_v3.specialists import Finding
from v2.agent_v3.research import attach_comparison_sources
import pytest


@pytest.mark.parametrize("ticker,web,mode", [("ORCL", False, "restate"), ("NVDA", True, "restate"), ("NVDA", False, "expand")])
def test_history_requires_same_subject_permission_and_pure_rewrite(ticker, web, mode):
    class Brain(DemoBrain):
        def classify(self, *args):
            return SemanticIntent(kind="lookup", tickers=[ticker], wants=["performance"], refers_back=True, follow_up_mode=mode)
    a = build_demo_agent()
    a.brain = Brain()
    from dataclasses import replace
    a.config = replace(a.config, enable_web=True)
    a.store.put("switch", {"previous": {"answer":"Old NVDA answer [old]", "tickers":["NVDA"], "allow_web":False, "results":[], "evidence":[plain(EvidenceItem("old", "NVDA", "Old NVDA answer"))]}})
    try:
        r = a.run("synthetic semantic fixture", session_id="switch", allow_web=web)
        assert "execute" in r.synthesis["nodes"]
        assert all(e.id != "old" for e in r.evidence)
    finally:
        a.store.close()


def test_news_failure_does_not_dump_filing_directory():
    a = build_demo_agent()
    try:
        state = {"intent":{"wants":["news"],"tickers":["ORCL"]}, "allow_web":True, "evidence":[plain(EvidenceItem("f", "ORCL", "Form 4"))]}
        result = a._fallback(state, None)
        assert "Form 4" not in result["answer"]
        assert "新闻" in result["answer"] and result["fallback"]
        state["allow_web"] = False
        assert "未授权" in a._fallback(state, None)["answer"]
    finally:
        a.store.close()


def test_fallback_excludes_other_security():
    a = build_demo_agent()
    try:
        result = a._fallback({"intent":{"tickers":["ORCL"]}, "evidence":[plain(EvidenceItem("old","NVDA","NVIDIA announcement"))]}, None)
        assert "NVIDIA" not in result["answer"]
    finally:
        a.store.close()


def test_news_synthesis_failure_preserves_located_sources():
    a = build_demo_agent()
    try:
        evidence = plain(EvidenceItem("source", "AMD", "Reported event", metadata={"quote_located": True}))
        result = a._fallback({"intent":{"tickers":["AMD"],"wants":["news"]},"allow_web":True,"evidence":[evidence]},None)
        assert "Reported event [source]" in result["answer"]
        assert "未取得" not in result["answer"] and result["fallback"]
    finally:
        a.store.close()


def test_selected_monitor_evidence_cannot_override_current_stock():
    a = build_demo_agent()
    a.brain.classify = lambda *args: SemanticIntent(kind="research", tickers=["ORCL"], wants=["attribution"], use_selected_record=True)
    try:
        result = a._classify({"text":"fixture", "allow_web":False, "monitor_evidence":[plain(EvidenceItem("old", "NVDA", "Old alert"))]}, None)
        assert result["monitor_evidence"] == []
        assert not result.get("record_answer")
    finally:
        a.store.close()


def test_restatement_preserves_partial_status():
    from types import SimpleNamespace
    a = build_demo_agent()
    try:
        state = {"answer":"Supported rewrite", "inherited_partial":True, "report":{"ok":True}}
        assert a._finish(state, SimpleNamespace(cancelled=False, usage=[]))["status"] == "partial"
    finally:
        a.store.close()


@pytest.mark.parametrize("wants,scope,days", [(["news"],"none",14),(["attribution"],"recent",30),(["news","attribution"],"recent",30)])
def test_implicit_recent_window_is_enforced(wants, scope, days):
    from datetime import date, timedelta
    a = build_demo_agent()
    a.brain.classify = lambda *args: SemanticIntent(kind="research", tickers=["AMD"], wants=wants, scope=scope)
    try:
        result = a._classify({"text":"fixture", "allow_web":True}, None)
        assert result["intent"]["date_window"] == {"start":(date.today()-timedelta(days=days-1)).isoformat(), "end":date.today().isoformat(), "basis":"event" if "attribution" in wants else "publication"}
    finally:
        a.store.close()


@pytest.mark.parametrize("explicit",[False,True])
def test_generated_recent_window_cannot_override_user_window_policy(explicit):
    from datetime import date, timedelta
    from v2.agent_v3.contracts import DateWindow
    window=DateWindow(start="2026-09-01",end="2026-09-07",basis="event")
    a=build_demo_agent()
    a.brain.classify=lambda *args:SemanticIntent(kind="research",tickers=["AAPL"],wants=["attribution"],scope="recent",date_window=window,date_window_explicit=explicit)
    try:
        result=a._classify({"text":"fixture","allow_web":True},None)
        expected=window.model_dump() if explicit else {"start":(date.today()-timedelta(days=29)).isoformat(),"end":date.today().isoformat(),"basis":"event"}
        assert result["intent"]["date_window"]==expected
    finally:a.store.close()


def test_company_analysis_replaces_inherited_news_and_reads_research():
    from v2.agent_v3.brain import ModelBrain
    from v2.agent_v2.routing import route
    from v2.agent_v2.models import NormalizedRequest
    a = build_demo_agent()
    a.brain.classify = lambda *args: SemanticIntent(kind="research", tickers=["MU"], wants=["news"], analysis_scope="company", refers_back=True)
    try:
        update = a._classify({"text":"fixture", "allow_web":True}, None)
        intent = SemanticIntent.model_validate(update["intent"])
        assert "news" not in intent.wants and not intent.refers_back
        request = NormalizedRequest("fixture", "fixture", entities=("MU",))
        plan = ModelBrain(None).plan(request, route(request,intent=intent.domain()), a.registry, None)
        assert [t.capability for t in plan.tasks] == ["market.performance","research.stock"]
        assert plan.tasks[1].arguments == {"ticker":"MU","focus":"overview"}
        assert "analysis_scope" not in intent.domain().to_dict()
    finally:
        a.store.close()


def test_news_with_catalysts_keeps_news_only_plan():
    a = build_demo_agent()
    a.brain.classify = lambda *args: SemanticIntent(kind="research", tickers=["AMD"], wants=["news","catalysts"])
    try:
        result = a._classify({"text":"fixture", "allow_web":True}, None)
        assert result["intent"]["wants"] == ["news"]
    finally:
        a.store.close()


def test_cached_partial_research_remains_partial():
    from types import SimpleNamespace
    a = build_demo_agent()
    try:
        state = {"answer":"fixture", "report":{"ok":True}, "results":[plain(ToolEnvelope("research.stock", ResultStatus.CACHED, limitations=["valuation: PARTIAL_DATA"]))]}
        assert a._finish(state, SimpleNamespace(cancelled=False,usage=[]))["status"] == "partial"
    finally:
        a.store.close()


def test_committee_semantics_take_priority_over_company_overview():
    a=build_demo_agent()
    a.brain.classify=lambda *args:SemanticIntent(kind="research",tickers=["MU"],lab="committee",analysis_scope="company")
    try:
        result=a._classify({"text":"fixture","allow_web":False},None)
        assert result["intent"]["kind"]=="lab"
        assert result["intent"]["analysis_scope"]=="focused"
    finally:a.store.close()


def test_missing_security_clarifies_without_account_fallback():
    class Brain(DemoBrain):
        def classify(self,*args):
            return SemanticIntent(kind="lookup",wants=["performance"])
        def plan(self,*args):
            raise AssertionError("Missing subject must not reach planner")
    a=build_demo_agent();a.brain=Brain()
    try:
        r=a.run("查询收盘价")
        assert r.status.value=="waiting_clarification"
        assert not r.results
    finally:a.store.close()


def test_explicit_clarification_does_not_depend_on_confidence():
    class Brain(DemoBrain):
        def classify(self,*args):
            return SemanticIntent(kind="research",confidence=.99,clarification="请提供比较对象")
    a=build_demo_agent();a.brain=Brain()
    try:assert a.run("比较一下").status.value=="waiting_clarification"
    finally:a.store.close()


def test_unregistered_lab_has_actionable_boundary_without_execution():
    class Brain(DemoBrain):
        def classify(self,*args):return SemanticIntent(kind="lab",lab="backtest",tickers=["NVDA"])
        def plan(self,*args):return ExecutionPlan("test",RouteKind.LAB,tasks=(PlanTask("lab","lab.backtest",{}),))
    a=build_demo_agent();a.brain=Brain()
    try:
        r=a.run("回测")
        assert r.stop_reason=="capability_unavailable"
        assert "策略回测" in r.answer and not r.results
    finally:a.store.close()


def test_macro_identifier_and_domain_boundary():
    intent=SemanticIntent(kind="lookup",release="CPI")
    assert intent.release=="cpi" and intent.domain().release=="cpi"


def test_optional_unavailable_tool_does_not_block_required_query():
    class Brain(DemoBrain):
        def plan(self,*args):
            return ExecutionPlan("test",RouteKind.RESEARCH,tasks=(
                PlanTask("price","market.performance",{"ticker":"NVDA"}),
                PlanTask("optional","market.anomaly_history",{"ticker":"NVDA"},required=False)))
    a=build_demo_agent();a.brain=Brain()
    try:
        r=a.run("查询")
        assert r.stop_reason!="capability_unavailable"
        assert any(e.id=="demo-price" for e in r.evidence)
        assert "execute" in r.synthesis["nodes"]
    finally:a.store.close()


def test_etf_holdings_does_not_call_stock_research():
    class Brain(DemoBrain):
        def classify(self,*args):return SemanticIntent(kind="lookup",tickers=["SPY"],data_target="etf_holdings")
        def plan(self,*args):raise AssertionError("ETF holdings must not fall through to stock research")
    a=build_demo_agent();a.brain=Brain()
    try:
        r=a.run("前五大持仓")
        assert r.stop_reason=="capability_unavailable" and "ETF" in r.answer
    finally:a.store.close()


def test_restatement_filters_unused_evidence_and_fails_closed():
    a=build_demo_agent()
    prior="NVDA 收盘价为 120 美元。[price]"
    a.store.put("r",{"previous":{"answer":prior,"results":[],"evidence":[
        plain(EvidenceItem("price","NVDA","收盘价120美元",value=120)),
        plain(EvidenceItem("extra","NVDA","额外指标50",value=50))]}})
    class Brain(DemoBrain):
        def classify(self,*args):return SemanticIntent(kind="lookup",refers_back=True,follow_up_mode="restate",tickers=["NVDA"])
        def draft(self,state,*args,**kwargs):
            assert [r["id"] for r in state["evidence"]]==["price"]
            assert state["results"]==[]
            return "NVDA 收盘价120美元，且有额外结论。[price]"
        def check_restatement(self,previous,answer,run):
            assert previous==prior
            return True
    a.brain=Brain()
    try:
        r=a.run("缩短上一答",session_id="r")
        assert r.status.value=="partial" and r.answer.endswith(prior)
        assert "额外结论" not in r.answer and "execute" not in r.synthesis["nodes"]
    finally:a.store.close()


def test_news_scope_reasons_are_separate():
    window={"basis":"event","start":"2026-09-07","end":"2026-09-13"}
    def f(day):return Finding(date=day,text="fixture",source="p1",quote="A sufficiently long quote")
    assert scope_rejection(f(""),window,[])=="missing_event_date"
    assert scope_rejection(f("2026-09-01"),window,[])=="event_date_out_of_scope"
    assert scope_rejection(f("bad"),window,[])=="invalid_event_date"
    assert scope_rejection(f("2026-09-11"),window,[])==""


def test_comparison_keeps_distinct_source_periods():
    a=EvidenceItem("a","NVDA","a",period="2026-Q2",source_url="https://example.com/a")
    b=EvidenceItem("b","AMD","b",period="2026-Q1",source_url="https://example.com/b")
    c=EvidenceItem("c","NVDA,AMD","compare",metadata={"comparison":True,"from":["a","b"]})
    result=attach_comparison_sources(ToolEnvelope("research.compare",ResultStatus.COMPLETED,evidence=[c,a,b]))
    row=result.evidence[0]
    assert [x["period"] for x in row.metadata["source_components"]]==["2026-Q2","2026-Q1"]
    assert row.metadata["periods_verified_equal"] is False
    assert row.as_of=="" and row.source_url==""
