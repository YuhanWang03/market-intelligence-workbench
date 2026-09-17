from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import sqlite3
import threading
import time

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver

from v2.agent_v2.models import AnswerMode, EvidenceItem, ExecutionPlan, PlanTask, ResultStatus, RouteKind, RunStatus, ToolEnvelope
from v2.agent_v3.contracts import SemanticIntent
from v2.agent_v3.demo import DEMO_QUESTION, DemoBrain, build_demo_agent
from v2.agent_v3.execution import validate_plan
from v2.agent_v3.graph import AgentV3, AgentV3Config
from v2.agent_v3.persistence import SessionStore
from v2.agent_v3.tools import Registry


def test_offline_graph_uses_real_nodes_and_checkpoint():
    agent = build_demo_agent()
    result = agent.run(DEMO_QUESTION, session_id="demo")
    assert result.status == RunStatus.COMPLETED, result.to_dict()
    assert result.verification.ok
    assert {"classify", "plan", "execute", "synthesize", "verify", "finish"} <= set(result.synthesis["nodes"])
    snapshot = agent.graph.get_state({"configurable": {"thread_id": result.run_id}})
    json.dumps(snapshot.values, allow_nan=False)
    assert snapshot.values["evidence"][0]["value"] == 120


def test_source_review_avoids_duplicate_debate_but_keeps_answer_verification():
    agent=build_demo_agent()
    try:
        state={"report":{"ok":True},"route":"research","results":[{"metadata":{"review_stage":"source_cross_review"}}]}
        assert agent._after_verify(state)=="finish"
        state["report"]={"ok":False}
        assert agent._after_verify(state)=="repair"
    finally:agent.store.close()


def test_prose_enum_arguments_are_coerced_not_fatal():
    from v2.agent_v3.execution import coerce_enum_value, coerce_plan_arguments
    allowed = ["overview", "fundamentals", "valuation", "earnings", "market", "ownership", "catalysts", "filings", "supply_chain", "risk", "full"]
    assert coerce_enum_value("行业与供应链风险", allowed) == "supply_chain"
    assert coerce_enum_value("估值贵不贵", allowed) == "valuation" and coerce_enum_value("risk", allowed) == "risk"
    assert coerce_enum_value("天气", allowed) is None and coerce_enum_value("full", allowed) == "full"
    registry = Registry()
    plan = ExecutionPlan("q", RouteKind.RESEARCH, tasks=(PlanTask("r", "research.stock", {"ticker": "QCOM", "focus": "行业与供应链风险"}), PlanTask("m", "market.performance", {"ticker": "QCOM"})))
    fixed = coerce_plan_arguments(plan, registry)
    assert fixed.tasks[0].arguments == {"ticker": "QCOM", "focus": "supply_chain"} and fixed.tasks[1] is plan.tasks[1]
    assert any("read as 'supply_chain'" in note for note in fixed.assumptions)
    validate_plan(fixed, registry)
    with pytest.raises(Exception):
        validate_plan(plan, registry)
    # end to end: a brain that plans with a prose focus no longer fails the run
    agent = build_demo_agent()
    agent.registry.register("research.stock", lambda args, ctx: ToolEnvelope("research.stock", ResultStatus.COMPLETED, subject=args["ticker"], evidence=[EvidenceItem("r1", "NVDA", "NVDA 供应链集中。", source_id="research_engine")]))
    class Prose(DemoBrain):
        def classify(self, *args):
            return SemanticIntent(kind="research", wants=["risk"], tickers=["NVDA"], focus=["行业与供应链风险"])
        def plan(self, request, decision, registry, run):
            return ExecutionPlan("q", RouteKind.RESEARCH, tasks=(PlanTask("r", "research.stock", {"ticker": "NVDA", "focus": "行业与供应链风险"}),), answer_mode=AnswerMode.RESEARCH_GROUNDED)
        def draft(self, state, registry, run, **kwargs):
            return "NVDA 供应链集中。[r1]"
    agent.brain = Prose()
    result = agent.run("上面第二点展开讲")
    assert result.status != RunStatus.FAILED, result.error
    assert result.plan.tasks[0].arguments["focus"] == "supply_chain"


def test_repair_is_a_graph_cycle_not_hidden_in_synthesizer():
    class Repair(DemoBrain):
        def draft(self, state, registry, run, **kwargs):
            return super().draft(state, registry, run) if kwargs.get("repair") else "NVDA 收盘价为 999 美元。[fake-id]"
    agent = build_demo_agent()
    agent.brain = Repair()
    result = agent.run(DEMO_QUESTION)
    assert result.status == RunStatus.COMPLETED
    assert result.synthesis["nodes"].count("repair") == 1
    assert result.synthesis["nodes"].count("verify") == 2
    assert "999" not in result.answer


def test_failed_repairs_are_bounded_and_fallback_is_partial():
    class Bad(DemoBrain):
        def draft(self, *args, **kwargs):
            return "NVDA 值 999 美元。[fake-id]"
    agent = build_demo_agent()
    agent.brain = Bad()
    result = agent.run(DEMO_QUESTION)
    assert result.synthesis["nodes"].count("repair") == 2
    assert result.status == RunStatus.PARTIAL
    assert "999" not in result.answer
    assert "demo-price" in result.answer


@pytest.mark.parametrize("tasks", [
    [PlanTask("a", "market.performance", {"ticker": "NVDA"}, depends_on=("b",))],
    [PlanTask("a", "market.performance", {"ticker": "NVDA"}, depends_on=("a",))],
    [PlanTask("a", "market.performance", {"ticker": 123})],
    [PlanTask("a", "market.performance", {"ticker": "NVDA", "unexpected": True})],
])
def test_invalid_plans_rejected_before_tools(tasks):
    with pytest.raises(Exception):
        validate_plan(ExecutionPlan("test", RouteKind.RESEARCH, tasks=tuple(tasks)), Registry())


def test_parallel_fanout_waits_for_source_and_aggregates_evidence():
    from v2.agent_v3.execution import build_executor
    from v2.agent_v3.context import RunContext
    from v2.agent_v3.contracts import plain
    from v2.agent_v2.models import NormalizedRequest
    registry = Registry()
    barrier = threading.Barrier(2)
    def holdings(args, ctx):
        return ToolEnvelope("account.portfolio", ResultStatus.COMPLETED, metadata={"tickers": ["NVDA", "AMD"]})
    def price(args, ctx):
        barrier.wait(timeout=3)
        return ToolEnvelope("market.performance", ResultStatus.COMPLETED, subject=args["ticker"])
    registry.register("account.portfolio", holdings)
    registry.register("market.performance", price)
    plan = ExecutionPlan("prices", RouteKind.RESEARCH, tasks=(PlanTask("scope", "account.portfolio"), PlanTask("prices", "market.performance", fan_out={"from": "scope", "field": "tickers", "argument": "ticker", "max": 2})))
    validate_plan(plan, registry)
    output = build_executor(registry).invoke({"plan": plain(plan), "request": plain(NormalizedRequest("prices", "prices")), "updates": []}, context=RunContext("test", time.monotonic()+5), config={"max_concurrency": 2})
    assert {row["subject"] for key,row in output["done"].items() if key != "scope"} == {"AMD", "NVDA"}
    assert all(row["status"] == "completed" for row in output["done"].values())


def mutation_agent(store=None, checkpointer=None, calls=None):
    registry = Registry()
    def mutate(args, ctx):
        calls.append(args)
        return ToolEnvelope("state.mutate", ResultStatus.COMPLETED, evidence=[EvidenceItem("written", "NVDA", "已添加 NVDA 关注。", source_id="test_state")])
    registry.register("state.mutate", mutate)
    class Write(DemoBrain):
        def classify(self, *args):
            return SemanticIntent(kind="command", tickers=["NVDA"], command={"operation": "watchlist.add", "ticker": "NVDA"})
    return AgentV3(registry=registry, brain=Write(), store=store, checkpointer=checkpointer, config=AgentV3Config(debate=False))


def test_confirmation_persists_and_cross_session_or_duplicate_resume_refused(tmp_path):
    path = tmp_path / "checkpoints.sqlite"
    conn = sqlite3.connect(path, check_same_thread=False)
    calls = []
    store = SessionStore(tmp_path / "session.sqlite")
    agent = mutation_agent(store, SqliteSaver(conn), calls)
    result = agent.run("加入关注", session_id="owner")
    assert result.status == RunStatus.WAITING_CONFIRMATION
    assert calls == []
    store.close()
    conn.close()
    conn2 = sqlite3.connect(path, check_same_thread=False)
    restored = mutation_agent(SessionStore(tmp_path / "session.sqlite"), SqliteSaver(conn2), calls)
    with pytest.raises(PermissionError):
        restored.resume(session_id="other", run_id=result.run_id, approve=True)
    resumed = restored.resume(session_id="owner", run_id=result.run_id, approve=True)
    assert len(calls) == 1
    assert resumed.status == RunStatus.COMPLETED
    assert "已添加" in resumed.answer
    with pytest.raises(PermissionError):
        restored.resume(session_id="owner", run_id=result.run_id, approve=True)
    assert len(calls) == 1
    restored.store.close()
    conn2.close()


def test_cancelled_or_expired_confirmation_never_writes():
    calls=[]
    agent=mutation_agent(calls=calls)
    pending=agent.run("加入关注",session_id="owner")
    result=agent.resume(session_id="owner",run_id=pending.run_id,approve=False)
    assert result.status == RunStatus.CANCELLED and calls == []
    agent.config=replace(agent.config,confirmation_ttl=-1)
    pending=agent.run("加入关注",session_id="owner")
    result=agent.resume(session_id="owner",run_id=pending.run_id,approve=True)
    assert result.status == RunStatus.CANCELLED and calls == []


def test_write_journal_does_not_repeat_uncertain_side_effect():
    store=SessionStore()
    task=PlanTask("m","state.mutate",{"operation":"watchlist.add","payload":{"ticker":"NVDA"}})
    calls=[]
    def crash():
        calls.append(1)
        raise RuntimeError("connection lost after write")
    with pytest.raises(RuntimeError):
        store.mutate("key",task,crash)
    with pytest.raises(RuntimeError,match="uncertain"):
        store.mutate("key",task,crash)
    assert calls == [1]


def test_expanding_followup_replans_instead_of_blind_reuse():
    class Follow(DemoBrain):
        def classify(self,text,history,run):
            return SemanticIntent(kind="research",tickers=["NVDA"],refers_back=bool(history.get("previous")), wants=["performance"])
    agent=build_demo_agent()
    agent.brain=Follow()
    first=agent.run(DEMO_QUESTION,session_id="a")
    second=agent.run("解释这个数字",session_id="a")
    assert "execute" in second.synthesis["nodes"]
    assert second.evidence == first.evidence


def test_tool_timeout_returns_partial_and_cancellation_is_local():
    agent=build_demo_agent()
    release=threading.Event()
    def slow(args,ctx):
        release.wait(2)
        return ToolEnvelope("market.performance",ResultStatus.COMPLETED)
    agent.registry.register("market.performance",slow)
    agent.config=replace(agent.config,max_seconds=.2,answer_reserve_seconds=.05)
    try:
        started=time.monotonic()
        result=agent.run(DEMO_QUESTION)
        assert time.monotonic()-started < 1.5
        assert result.status == RunStatus.PARTIAL
    finally:
        release.set()
    cancelled=threading.Event()
    cancelled.set()
    stopped=agent.run(DEMO_QUESTION,cancel_event=cancelled)
    assert stopped.status == RunStatus.CANCELLED


def test_concurrent_sessions_do_not_share_cancellation():
    agent=build_demo_agent()
    event=threading.Event()
    event.set()
    with ThreadPoolExecutor(max_workers=2) as pool:
        a=pool.submit(agent.run,DEMO_QUESTION,session_id="a",cancel_event=event)
        b=pool.submit(agent.run,DEMO_QUESTION,session_id="b")
        assert a.result().status == RunStatus.CANCELLED
        assert b.result().status == RunStatus.COMPLETED


def test_clarification_is_merged_as_context_not_keyword_routed():
    class Clarify(DemoBrain):
        seen=[]
        def classify(self,text,history,run):
            self.seen.append(text)
            if not history.get("clarification"):
                return SemanticIntent(kind="lookup",confidence=.2,clarification="需要查询哪只股票？")
            return super().classify(text,history,run)
    agent=build_demo_agent()
    brain=Clarify()
    agent.brain=brain
    first=agent.run("查一下行情",session_id="a")
    assert first.status == RunStatus.WAITING_CLARIFICATION
    second=agent.run("英伟达",session_id="a")
    assert second.status == RunStatus.COMPLETED
    assert "查一下行情" in brain.seen[-1] and "英伟达" in brain.seen[-1]


def test_web_denied_before_handler_and_bad_mutation_payload_rejected():
    from v2.agent_v3.context import RunContext
    from v2.agent_v2.models import NormalizedRequest
    registry=Registry()
    calls=[]
    registry.register("web.research",lambda args,ctx: calls.append(args))
    result=registry.execute(PlanTask("w","web.research",{"query":"news","topic":"news"}),ExecutionPlan("news",RouteKind.RESEARCH),NormalizedRequest("news","news",allow_web=False),RunContext("test",time.monotonic()+5))
    assert not result.ok and calls == []
    with pytest.raises(Exception):
        registry.validate(PlanTask("w","state.mutate",{"operation":"alert.add","payload":{"ticker":"NVDA","direction":"above","target_price":-2}}))


def test_required_dependency_failure_skips_consumer():
    from v2.agent_v3.execution import build_executor
    from v2.agent_v3.context import RunContext
    from v2.agent_v3.contracts import plain
    from v2.agent_v2.models import NormalizedRequest
    registry=Registry()
    registry.register("account.portfolio",lambda args,ctx: ToolEnvelope("account.portfolio",ResultStatus.FAILED,errors=["provider down"]))
    called=[]
    registry.register("market.performance",lambda args,ctx: called.append(args))
    plan=ExecutionPlan("test",RouteKind.RESEARCH,tasks=(PlanTask("a","account.portfolio"),PlanTask("b","market.performance",{"ticker":"NVDA"},depends_on=("a",))))
    output=build_executor(registry).invoke({"plan":plain(plan),"request":plain(NormalizedRequest("test","test")),"updates":[]},context=RunContext("test",time.monotonic()+5))
    assert output["done"]["b"]["status"] == "skipped"
    assert called == []


def _research_agent(reviewer, revised_text):
    """Demo agent routed to research, with a scripted reviewer and a scripted revision."""
    agent = build_demo_agent()
    class Research(DemoBrain):
        def classify(self, *args):
            return SemanticIntent(kind="research", wants=["performance"], tickers=["NVDA"])
        def draft(self, state, registry, run, **kwargs):
            return revised_text if kwargs.get("objections") else super().draft(state, registry, run)
    agent.brain = Research()
    agent.config = replace(agent.config, debate=True)
    agent.reviewer = reviewer
    return agent


OBJECTION = {"objection": "答案未说明价格口径", "evidence_id": "demo-price"}


def test_failed_debate_does_not_replace_verified_answer():
    def broken(*args):
        raise RuntimeError("review provider down")
    agent = _research_agent(broken, "unused")
    result = agent.run(DEMO_QUESTION)
    assert result.status == RunStatus.COMPLETED
    assert result.verification.ok
    assert result.synthesis["debate"]["ran"] is True
    assert result.synthesis["debate"]["error"].startswith("RuntimeError")


def test_debate_revision_that_verifies_replaces_the_answer_and_is_recorded():
    agent = _research_agent(lambda state, run: [OBJECTION], "离线演示：NVDA 收盘价为 120 美元（收盘口径）。[demo-price]")
    result = agent.run(DEMO_QUESTION)
    assert result.status == RunStatus.COMPLETED
    assert "收盘口径" in result.answer
    assert result.synthesis["debate"] == {"ran": True, "objections": [OBJECTION], "revised": True, "revised_verified": True, "material": 1}
    assert result.synthesis["objections"] == [OBJECTION]
    assert len(result.synthesis["attempts"]) == 2 and result.synthesis["attempts"][-1]["ok"]
    assert not result.verification.warnings


def test_debate_revision_that_fails_keeps_the_answer_and_surfaces_objections():
    agent = _research_agent(lambda state, run: [OBJECTION], "NVDA 收盘价为 999 美元。[fake-id]")
    result = agent.run(DEMO_QUESTION)
    assert result.status == RunStatus.COMPLETED
    assert "999" not in result.answer and "demo-price" in result.answer
    assert result.verification.ok
    assert result.synthesis["debate"] == {"ran": True, "objections": [OBJECTION], "revised": True, "revised_verified": False, "material": 1}
    assert any("审阅异议" in note and "demo-price" in note for note in result.verification.warnings)
    assert result.synthesis["attempts"][-1]["rejected_draft"].startswith("NVDA 收盘价为 999")


def test_minor_objections_are_shown_without_a_redraft():
    minor = {**OBJECTION, "severity": "minor", "claim": "NVDA 收盘价为 120 美元。"}
    drafted = []
    agent = _research_agent(lambda state, run: [minor], "unused")
    original_draft = agent.brain.draft
    def spy(state, registry, run, **kwargs):
        drafted.append(kwargs)
        return original_draft(state, registry, run, **kwargs)
    agent.brain.draft = spy
    result = agent.run(DEMO_QUESTION)
    assert result.status == RunStatus.COMPLETED and result.verification.ok
    assert not any(kw.get("objections") for kw in drafted), "a minor-only review must not redraft"
    assert result.synthesis["debate"] == {"ran": True, "objections": [minor], "revised": False, "revised_verified": False, "material": 0}
    assert any(note.startswith("审阅提示：") and "demo-price" in note for note in result.verification.warnings)


def test_objection_schema_defaults_and_bounds():
    from pydantic import ValidationError
    from v2.agent_v3.contracts import Objection, Review
    assert Objection(objection="x", evidence_id="e").severity == "material"
    assert Review().objections == []
    with pytest.raises(ValidationError):
        Objection(objection="x", evidence_id="e", severity="severe")
    with pytest.raises(ValidationError):
        Review(objections=[Objection(objection=str(i), evidence_id="e") for i in range(7)])


def test_skipped_debate_records_why():
    agent = build_demo_agent()  # debate=False in the demo config
    result = agent.run(DEMO_QUESTION)
    assert result.synthesis["debate"] == {"ran": False, "skipped": "disabled"}
    agent.config = replace(agent.config, debate=True)
    agent.reviewer = lambda state, run: []
    result = agent.run(DEMO_QUESTION)
    assert result.synthesis["debate"] == {"ran": False, "skipped": "route=fast_lookup"}


def test_new_question_invalidates_pending_confirmation():
    agent=mutation_agent(calls=[])
    pending=agent.run("加入关注",session_id="a")
    agent.brain=DemoBrain()
    agent.run(DEMO_QUESTION,session_id="a")
    with pytest.raises(PermissionError):
        agent.resume(session_id="a",run_id=pending.run_id,approve=True)
