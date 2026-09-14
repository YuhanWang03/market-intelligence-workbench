from types import SimpleNamespace
import pytest

from v2.agent_v3.contracts import DateWindow, SemanticIntent
from v2.agent_v3.source_scope import ScopedFilings, finding_in_scope
from v2.agent_v3.sec import ItemFilingSource


def test_dates_are_validated_and_excluded_from_legacy_intent():
    with pytest.raises(ValueError):
        DateWindow(start="2026-09-12",end="2026-09-01",basis="event")
    with pytest.raises(ValueError):
        DateWindow(start="not-a-date",end="2026-09-12",basis="event")
    intent=SemanticIntent(kind="research",date_window=DateWindow(start="2026-09-06",end="2026-09-12",basis="event"))
    assert "date_window" not in intent.domain().to_dict()


@pytest.mark.parametrize("day,accepted",[("2026-09-05",False),("",False),("2026-09-06",True),("2026-09-12",True),("2026-09-13",False)])
def test_news_requires_event_within_inclusive_bounds(day,accepted):
    finding=SimpleNamespace(date=day,source="p1")
    assert finding_in_scope(finding,{"start":"2026-09-06","end":"2026-09-12","basis":"event"},[]) is accepted


def test_filing_dates_do_not_filter_described_events_and_items_are_exact():
    window={"start":"2026-01-01","end":"2026-12-31","basis":"filing"}
    assert finding_in_scope(SimpleNamespace(date="2025-12-01",source="f1:Item 1A"),window,["Item 1A"])
    assert not finding_in_scope(SimpleNamespace(date="",source="f1:Item 1"),window,["Item 1A"])


def test_sdk_item_index_exposes_risk_section_without_business_chunking():
    class Report:
        items=["Item 1","Item 1A"]
        def __getitem__(self,key):
            return {"Item 1":"Business description","Item 1A":"Supplier failure could disrupt production."}[key]
    ref=SimpleNamespace(form="10-K",accession="fixture",raw=SimpleNamespace(obj=lambda:Report()))
    source=ItemFilingSource()
    scoped=ScopedFilings(source,items=["Item 1A"])
    assert scoped.outline(ref)[0].id == "Item 1A"
    assert scoped.read(ref,"Item 1A") == "Supplier failure could disrupt production."
    with pytest.raises(ValueError):
        scoped.read(ref,"Item 1")


def test_user_filing_bounds_override_tool_default_lookback():
    observed=[]
    class Source:
        def list_filings(self,*args):
            observed.append(args)
            return [SimpleNamespace(filing_date="2025-12-31"),SimpleNamespace(filing_date="2026-02-25")]
    scoped=ScopedFilings(Source(),{"start":"2026-01-01","end":"2026-09-12","basis":"filing"},forms=["10-K"])
    rows=scoped.list_filings("NVDA","2026-06-01","2026-09-12",["8-K"])
    assert observed==[("NVDA","2026-01-01","2026-09-12",["10-K"])]
    assert len(rows)==1
