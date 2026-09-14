import time
import pytest
from types import SimpleNamespace

from langchain_core.messages import AIMessage
from v2.agent_v3.tests.test_specialists import ScriptedModel, call
from v2.agent_v3.specialists import specialist_graph, FindingsReport
from v2.agent_v3.context import RunContext
from v2.agent_v3.tools import Registry


@pytest.mark.parametrize("citation",["【price-1744854192】",r"\[price-1744854192\]","［price-1744854192］","[price-1744854192, price-1744854192]"])
def test_known_citation_format_does_not_become_financial_number(citation):
    from v2.agent_v3.demo import build_demo_agent
    agent=build_demo_agent()
    state={"plan":{"answer_mode":"research_grounded"},"evidence":[{"id":"price-1744854192","entity":"NVDA","claim":"NVDA price 120 USD","value":120}],"results":[]}
    try:
        _,report=agent._report(f"NVDA price 120 USD {citation}",state,RunContext("test",time.monotonic()+5))
        assert report["ok"], report
        _,bad=agent._report(f"NVDA price 999 USD {citation}",state,RunContext("test",time.monotonic()+5))
        assert not bad["ok"]
    finally:
        agent.store.close()


def test_sec_failure_has_no_negative_factual_evidence():
    from v2.agent_v3.sec import register_history_capabilities
    def fail(*args):
        raise ConnectionError("fixture provider failure")
    registry=Registry()
    register_history_capabilities(registry,fetch=fail)
    result=registry.handlers["filings.recent"]({"ticker":"NVDA","forms":["10-K"]},SimpleNamespace(run_id="test"))
    assert result.errors and not result.evidence
    assert not result.metadata


def test_protocol_retry_does_not_execute_rejected_batch():
    observed=[]
    def read(source: str) -> str:
        """Read fixture."""
        observed.append(source)
        return source
    multiple=AIMessage(content="",tool_calls=[{"name":"read","args":{"source":str(i)},"id":str(i),"type":"tool_call"} for i in (1,2)])
    model=ScriptedModel(responses=[multiple,call("read",{"source":"corrected"},3),call("FindingsReport",{"findings":[]},4)])
    run=RunContext("test",time.monotonic()+5)
    graph,_=specialist_graph(model,[read],FindingsReport,"Read",run,"test")
    graph.invoke({"messages":[("human","read")]},context=run)
    assert observed==["corrected"]


def test_exhausted_read_can_finish_with_existing_context():
    def read(source: str) -> str:
        """Read fixture."""
        raise ValueError("Read limit reached")
    model=ScriptedModel(responses=[call("read",{"source":"fixture"},1),call("FindingsReport",{"findings":[],"note":"Read unavailable"},2)])
    run=RunContext("test",time.monotonic()+5)
    graph,_=specialist_graph(model,[read],FindingsReport,"Read",run,"test")
    result=graph.invoke({"messages":[("human","read")]},context=run)
    assert result["structured_response"].note=="Read unavailable"


def test_registered_source_url_number_is_not_a_financial_claim():
    from v2.agent_v3.demo import build_demo_agent
    agent=build_demo_agent()
    url="https://example.org/article-11929060"
    state={"plan":{"answer_mode":"research_grounded"},"evidence":[{"id":"price","entity":"NVDA","claim":"NVDA price 120 USD","value":120,"source_url":url}],"results":[]}
    answer=f"NVDA price 120 USD [price]. {url}"
    try:
        shown,report=agent._report(answer,state,RunContext("test",time.monotonic()+5))
        assert shown==answer and report["ok"]
        _,bad=agent._report(f"NVDA price 999 USD [price]. {url}",state,RunContext("test",time.monotonic()+5))
        assert not bad["ok"]
    finally:
        agent.store.close()
