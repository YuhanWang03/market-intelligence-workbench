from v2.agent_v2.models import ExecutionPlan, NormalizedRequest, PlanTask, RouteKind
from v2.agent_v3.adapters import register_account
from v2.agent_v3.context import RunContext
from v2.agent_v3.tools import Registry
import time


def test_portfolio_consumes_structured_fields_and_preserves_ratio_units():
    registry=Registry()
    register_account(registry,portfolio_source=lambda: {"account":{"paper":True},"positions":[{"symbol":"ARM","unrealized_pl_pct":-.125,"avg_entry_price":100}]})
    result=registry.execute(PlanTask("positions","account.portfolio"),ExecutionPlan("positions",RouteKind.FAST_LOOKUP),NormalizedRequest("positions","positions"),RunContext("test",time.monotonic()+5))
    assert result.ok
    assert result.metadata["tickers"]==["ARM"]
    assert result.metadata["positions"][0]["pl_pct"]==-12.5
    assert result.metadata["positions"][0]["unrealized_pl_pct"]==-.125
    assert result.evidence[0].metadata["structured"]["pl_pct_unit"]=="percent"
