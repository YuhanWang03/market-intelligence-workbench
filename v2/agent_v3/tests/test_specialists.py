import time
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr

from v2.agent_v3.context import RunContext
from v2.agent_v3.contracts import SemanticIntent
from v2.agent_v3.specialists import FindingsReport, specialist_graph, register_specialists
from v2.agent_v3.tools import Registry
from v2.agent_v2.models import ExecutionPlan, NormalizedRequest, PlanTask, RouteKind


class ScriptedModel(BaseChatModel):
    responses: list[Any]
    _cursor: int = PrivateAttr(default=0)
    _seen: list = PrivateAttr(default_factory=list)

    @property
    def _llm_type(self):
        return "offline-scripted-tools"

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self

    def with_structured_output(self, schema, **kwargs):
        kwargs.pop("method", None)
        return super().with_structured_output(schema, **kwargs)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self._seen.append(messages)
        if self._cursor >= len(self.responses):
            raise ValueError("Script exhausted")
        message=self.responses[self._cursor]
        self._cursor+=1
        return ChatResult(generations=[ChatGeneration(message=message)])


def call(name, args, index):
    return AIMessage(content="", tool_calls=[{"name":name,"args":args,"id":f"call-{index}","type":"tool_call"}])


def test_reviewer_uses_inline_evidence_in_a_single_round():
    from v2.agent_v3.specialists import make_reviewer
    kept = {"objection": "估值判断无来源", "evidence_id": "e1", "severity": "material", "claim": "估值中等"}
    model = ScriptedModel(responses=[call("Review", {"objections": [kept, {"objection": "幽灵", "evidence_id": "ghost"}]}, 1)])
    run = RunContext("test", time.monotonic() + 10)
    state = {"text": "分析一下 NVDA", "answer": "估值中等 [e1]", "evidence": [{"id": "e1", "claim": "valuation 43"}]}
    assert make_reviewer(model)(state, run) == [kept]
    assert len(model._seen) == 1, "no tool rounds: the evidence is already in the prompt"
    assert "valuation 43" in str(model._seen[0])


def test_langchain_agent_executes_tool_then_validated_finish(monkeypatch):
    from v2.agent_v2.agents.base import BoundedLoop
    monkeypatch.setattr(BoundedLoop,"run",lambda *args,**kwargs: pytest.fail("V2 loop must not run"))
    observed=[]
    def read_source(source: str) -> str:
        """Read a fixture source."""
        observed.append(source)
        return "The filing discloses a material risk in the supply agreement."
    model=ScriptedModel(responses=[call("read_source",{"source":"filing"},1),call("FindingsReport",{"findings":[],"note":"No confirmed event"},2)])
    run=RunContext("test",time.monotonic()+10)
    graph,observer=specialist_graph(model,[read_source],FindingsReport,"Read the source and report.",run,"reader")
    result=graph.invoke({"messages":[("human","Check the filing")]},context=run)
    assert observed==["filing"]
    assert isinstance(result["structured_response"],FindingsReport)
    assert [row["action"] for row in observer.trace]==["read_source","FindingsReport"]


@pytest.mark.parametrize("basis,expected", [("publication",1),("event",0)])
def test_publication_metadata_never_substitutes_for_event_date(basis,expected):
    quote="The company reported a change in its business outlook."
    model=ScriptedModel(responses=[call("search_news",{"query":"fixture"},1),call("read_page",{"id":"p1"},2),call("FindingsReport",{"findings":[{"source":"p1","date":"","text":"The company reported a change in its business outlook.","quote":quote}]},3),call("SupportedFindings",{"supported_indices":[0]},4)])
    registry=Registry()
    register_specialists(registry,model,search=lambda *a,**k:[{"url":"https://example.org/news","title":"Fixture","raw_content":quote,"published_date":"2026-09-11"}])
    request=NormalizedRequest("fixture","fixture",allow_web=True,metadata={"wants":["news"],"date_window":{"start":"2026-09-01","end":"2026-09-13","basis":basis}})
    result=registry.execute(PlanTask("news","web.research",{"query":"fixture","topic":"news"}),ExecutionPlan("fixture",RouteKind.RESEARCH),request,RunContext("fixture",time.monotonic()+15))
    assert len(result.evidence)==expected
    if expected:
        assert result.evidence[0].metadata["date_basis"]=="publication"
        assert "非事件发生日" in result.evidence[0].claim


def test_specialist_refuses_multiple_calls_before_tools_execute():
    observed=[]
    def read_source(source: str) -> str:
        """Read fixture."""
        observed.append(source)
        return source
    response=AIMessage(content="",tool_calls=[{"name":"read_source","args":{"source":str(i)},"id":str(i),"type":"tool_call"} for i in (1,2)])
    run=RunContext("test",time.monotonic()+10)
    graph,_=specialist_graph(ScriptedModel(responses=[response]),[read_source],FindingsReport,"Read.",run,"reader")
    with pytest.raises(ValueError,match="one tool"):
        graph.invoke({"messages":[("human","read")]},context=run)
    assert observed==[]


def test_specialist_cannot_keep_invented_quotes():
    quote="This invented quotation does not appear in any tool output."
    model=ScriptedModel(responses=[call("FindingsReport",{"findings":[{"text":"An event happened", "source":"p99","quote":quote}],"note":""},1)])
    registry=Registry()
    register_specialists(registry,model)
    result=registry.execute(PlanTask("news","web.research",{"query":"fixture","topic":"news"}),ExecutionPlan("fixture",RouteKind.RESEARCH),NormalizedRequest("fixture","fixture",allow_web=True),RunContext("test",time.monotonic()+10))
    assert result.evidence==[]
    assert any("Discarded" in note for note in result.limitations)


def test_model_intent_uses_schema_without_word_matching():
    from v2.agent_v3.brain import ModelBrain
    model=ScriptedModel(responses=[call("SemanticIntent",{"kind":"research","scope":"since_purchase","direction":"down","tickers":["ARM"],"wants":["attribution","drawdown"]},1)])
    result=ModelBrain(model).classify("套住好久了，这是怎么回事？",{"turns":[{"question":"我持有ARM"}]},RunContext("test",time.monotonic()+10))
    assert result.scope=="since_purchase" and result.tickers==["ARM"]
    assert "我持有ARM" in str(model._seen)


def test_real_model_adapter_end_to_end_with_scripted_provider(monkeypatch):
    from v2.agent_v2.orchestrator import AgentV2
    from v2.agent_v2.execution import ExecutionEngine
    monkeypatch.setattr(AgentV2,"run",lambda *args,**kwargs: pytest.fail("V2 orchestrator must not run"))
    monkeypatch.setattr(ExecutionEngine,"run",lambda *args,**kwargs: pytest.fail("V2 executor must not run"))
    from v2.agent_v3.brain import ModelBrain
    from v2.agent_v3.demo import build_demo_agent,DEMO_QUESTION
    from v2.agent_v2.models import RunStatus
    model=ScriptedModel(responses=[call("SemanticIntent",{"kind":"lookup","tickers":["NVDA"],"wants":["performance"]},1),AIMessage(content="NVDA 收盘价为 120 美元。[demo-price]")])
    agent=build_demo_agent()
    agent.brain=ModelBrain(model)
    result=agent.run(DEMO_QUESTION)
    assert result.status == RunStatus.COMPLETED
    assert [row["source"] for row in result.synthesis["usage"]] == ["classifier","synthesizer"]


def test_workspace_runtime_composes_without_invoking_provider(tmp_path):
    from v2.agent_v3.runtime import build_workspace_agent
    agent=build_workspace_agent(model=ScriptedModel(responses=[]),data_dir=tmp_path)
    try:
        assert agent.registry.registered("research.stock")
        assert agent.registry.registered("market.attribute_move")
        assert agent.registry.registered("account.portfolio")
        assert not agent.registry.registered("state.mutate")
        assert (tmp_path/"checkpoints.sqlite").exists()
    finally:
        agent.store.close()
        agent.checkpoint_connection.close()


def test_move_agent_preserves_actual_market_evidence_without_inventing_cause():
    from v2.agent_v2.models import EvidenceItem,ToolEnvelope,ResultStatus
    registry=Registry()
    registry.register("market.performance",lambda args,ctx: ToolEnvelope("market.performance",ResultStatus.COMPLETED,evidence=[EvidenceItem("market-price","NVDA","NVDA 收盘价为 120 美元。",value=120,source_id="fixture")]))
    model=ScriptedModel(responses=[call("FindingsReport",{"findings":[],"note":"No supported cause"},1)])
    register_specialists(registry,model)
    result=registry.execute(PlanTask("move","market.explain_move",{"ticker":"NVDA"}),ExecutionPlan("why",RouteKind.RESEARCH),NormalizedRequest("why","why"),RunContext("test",time.monotonic()+10))
    assert result.evidence[0].id == "market-price"
    assert result.metrics["confirmed_driver_count"] == 0
    assert result.status == ResultStatus.PARTIAL_DATA
