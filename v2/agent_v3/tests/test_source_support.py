import time
import pytest

from v2.agent_v3.tests.test_specialists import ScriptedModel, call
from v2.agent_v3.specialists import register_specialists
from v2.agent_v3.specialist_eval import search, NEWS, FixtureFilings, FILING
from v2.agent_v3.tools import Registry
from v2.agent_v3.context import RunContext
from v2.agent_v2.models import PlanTask, ExecutionPlan, NormalizedRequest, RouteKind


def run(responses, *, capability="web.research", source=search):
    registry = Registry()
    register_specialists(registry,ScriptedModel(responses=responses),search=source,filing_source=FixtureFilings())
    args = {"query":"fixture","topic":"news"} if capability == "web.research" else {"ticker":"EXMP"}
    return registry.execute(PlanTask("test",capability,args),ExecutionPlan("fixture",RouteKind.RESEARCH),NormalizedRequest("fixture","fixture",allow_web=True),RunContext("test",time.monotonic()+10))


@pytest.mark.parametrize("read,raw",[(False,True),(True,False)])
def test_snippet_is_not_original_evidence(read,raw):
    def source(*args,**kwargs):
        row=search(*args,**kwargs)[0]
        row.update(content=NEWS,raw_content=NEWS if raw else "")
        return [row]
    responses=[call("search_news",{"query":"fixture"},1)]
    if read:
        responses.append(call("read_page",{"id":"p1"},2))
    responses.append(call("FindingsReport",{"findings":[{"text":"An agreement was announced","source":"p1","quote":NEWS}]},3))
    assert run(responses,source=source).evidence == []


@pytest.mark.parametrize("accepted",[[],[0]])
def test_support_review_controls_admission(accepted):
    result=run([call("search_news",{"query":"fixture"},1),call("read_page",{"id":"p1"},2),
        call("FindingsReport",{"findings":[{"text":"EXMP announced a pilot agreement","date":"2026-09-10","source":"p1","quote":NEWS}]},3),
        call("SupportedFindings",{"supported_indices":accepted},4)])
    assert len(result.evidence) == len(accepted)
    if accepted:
        assert result.evidence[0].metadata["support_review"] == "model_supported"


def test_filing_reads_section_and_preserves_event_date():
    result=run([call("list_filings",{"ticker":"EXMP","forms":["8-K"]},1),call("read_filing",{"filing":1,"section":"s1"},2),
        call("FindingsReport",{"findings":[{"text":"Agreement entered","date":"2026-09-09","source":"f1:s1","quote":FILING}]},3),
        call("SupportedFindings",{"supported_indices":[0]},4)],capability="filings.read_events")
    assert result.evidence[0].as_of == "2026-09-09"
    assert result.evidence[0].source_url == "https://example.org/filing"


def test_attribution_stays_candidate_and_review_failure_drops_findings():
    responses=[call("search_news",{"query":"fixture"},1),call("read_page",{"id":"p1"},2),
        call("FindingsReport",{"findings":[{"text":"Possible context only","source":"p1","quote":NEWS}]},3)]
    result=run(responses+[call("SupportedFindings",{"supported_indices":[0]},4)],capability="market.explain_move")
    assert result.metrics["confirmed_driver_count"] == 0
    assert result.evidence[0].metadata["claim_role"] == "candidate_driver"
    assert run(responses).evidence == []


def test_unreviewed_note_cannot_become_final_answer_evidence():
    result=run([call("FindingsReport",{"findings":[],"note":"Unverified statement: all other news is unimportant."},1)])
    assert all("unimportant" not in value for value in result.limitations)
    assert result.evidence == []
