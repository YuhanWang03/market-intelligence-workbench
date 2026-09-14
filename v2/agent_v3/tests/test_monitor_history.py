from datetime import datetime, timezone
import sqlite3
import pytest
from v2.agent_v3.monitor_history import MonitorArchive
from v2.agent_v3.demo import DemoBrain, build_demo_agent


@pytest.fixture
def archive(tmp_path):
    path = tmp_path / "archive.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE pushes (id INTEGER,ts TEXT,agent TEXT,msg_type TEXT,tickers TEXT,text_html TEXT)")
        conn.executemany("INSERT INTO pushes VALUES (?,?,?,?,?,?)", [
            (1,"2026-09-10T15:00:00+00:00","intraday_anomaly","intraday_anomaly","NVDA","<b>原始历史记录</b>"),
            (2,"2026-09-11T15:00:00+00:00","alert","alert_fire","NVDA, AMD","价格提醒"),
            (3,"2026-09-11T15:00:00+00:00","anomaly","anomaly","NVDAX","别的股票"),
            (4,"2026-09-14T15:00:00+00:00","anomaly","anomaly","NVDA","未来记录"),
        ])
    return MonitorArchive(path, now=lambda: datetime(2026,9,13,tzinfo=timezone.utc))


def test_exact_record_and_chronological_recall(archive):
    assert archive.get("1","NVDA")["id"] == 1
    with pytest.raises(ValueError): archive.get("1","AMD")
    with pytest.raises(ValueError): archive.get("1 OR 1=1")
    with pytest.raises(LookupError): archive.get("99")
    rows = archive.recall("NVDA", "任意语义不做关键词过滤", 10)
    assert [r.metadata["monitor_record"]["id"] for r in rows] == [2,1]
    assert [r.metadata["monitor_record"]["id"] for r in archive.recall("AMD","",10)] == [2]
    assert archive.recall("NVDA","",1) == []


def test_missing_archive_is_error_not_empty_and_never_created(tmp_path):
    path = tmp_path / "missing.db"
    with pytest.raises(sqlite3.OperationalError): MonitorArchive(path).recall("NVDA","",30)
    assert not path.exists()


def test_graph_resolves_server_record_before_semantics(archive):
    agent = build_demo_agent()
    agent.event_source = archive
    class Brain(DemoBrain):
        def classify(self,text,history,run):
            selection = history["page_context"]["selection"]
            assert selection["occurred_at"] == "2026-09-10T15:00:00+00:00"
            assert "伪造" not in selection["excerpt"]
            return super().classify(text,history,run)
    agent.brain = Brain()
    try:
        context = {"section":"core","selection":{"kind":"anomaly","record_id":"1","ticker":"NVDA","excerpt":"伪造浏览器数据","occurred_at":"2026-09-13"}}
        result = agent.run("这条记录呢",page_context=context)
        assert result.synthesis["nodes"][0] == "resolve_context"
        item = next(e for e in result.evidence if e.id == "monitor-1")
        assert item.as_of == "2026-09-10T15:00:00+00:00"
        context["selection"]["record_id"] = "99"
        missing = agent.run("这条记录呢",page_context=context)
        assert missing.stop_reason == "monitor_record_unavailable"
        assert "classify" not in missing.synthesis["nodes"]
    finally:
        agent.store.close()


def test_specialist_memory_returns_traceable_archive_evidence(archive):
    import time
    from v2.agent_v3.tests.test_specialists import ScriptedModel, call
    from v2.agent_v3.specialists import register_specialists
    from v2.agent_v3.tools import Registry
    from v2.agent_v3.context import RunContext
    from v2.agent_v2.models import PlanTask, ExecutionPlan, RouteKind, NormalizedRequest
    model = ScriptedModel(responses=[
        call("recall_memory", {"ticker":"NVDA","query":"历史记录"},1),
        call("FindingsReport", {"findings":[],"note":""},2),
    ])
    registry = Registry()
    register_specialists(registry,model,recall=archive.recall)
    result = registry.execute(PlanTask("news","web.research",{"query":"fixture","topic":"news"}),
        ExecutionPlan("fixture",RouteKind.RESEARCH),NormalizedRequest("fixture","fixture",allow_web=True),RunContext("test",time.monotonic()+10))
    assert {e.id for e in result.evidence} == {"monitor-1","monitor-2"}
    assert any("不是语义相似检索" in note for note in result.limitations)


def test_record_only_semantic_intent_skips_external_research(archive):
    from v2.agent_v3.contracts import SemanticIntent
    agent = build_demo_agent()
    agent.event_source = archive
    class Brain(DemoBrain):
        def classify(self,*args):
            return SemanticIntent(kind="lookup",tickers=["NVDA"],use_selected_record=True)
        def plan(self,*args):
            pytest.fail("Verified record-only question must not start external research")
        def draft(self,*args,**kwargs):
            return "档案记录了 NVDA 的历史监控内容。[monitor-1]"
    agent.brain = Brain()
    try:
        result = agent.run("复述这条记录",page_context={"section":"core","selection":{"kind":"anomaly","ticker":"NVDA","record_id":"1"}})
        assert result.verification.ok
        assert "execute" not in result.synthesis["nodes"]
        assert "synthesize" in result.synthesis["nodes"]
    finally:
        agent.store.close()
