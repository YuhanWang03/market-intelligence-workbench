"""An explicitly offline fixture, not a natural-language or live-data fallback."""

from v2.agent_v2.models import EvidenceItem, ResultStatus, ToolEnvelope
from v2.agent_v2.planning import IntentPlanner
from v2.agent_v3.contracts import SemanticIntent
from v2.agent_v3.graph import AgentV3, AgentV3Config
from v2.agent_v3.tools import Registry


DEMO_QUESTION = "查询 NVDA 的行情（离线演示）"


class DemoBrain:
    def classify(self, text, history, run):
        return SemanticIntent(kind="lookup", wants=["performance"], tickers=["NVDA"])

    def plan(self, request, route, registry, run):
        return IntentPlanner().plan(request, route)

    def draft(self, state, registry, run, **kwargs):
        return "离线演示：NVDA 收盘价为 120 美元。[demo-price]"

    def judge(self, rows, run):
        return {}  # offline fixture only, never selected by the live runtime


def build_demo_agent(**kwargs):
    registry = Registry()
    def price(arguments, context):
        return ToolEnvelope("market.performance", ResultStatus.COMPLETED, subject="NVDA", evidence=[EvidenceItem(id="demo-price", entity="NVDA", claim="NVDA 收盘价为 120 美元。", metric="close", value=120, unit="USD", source_id="offline_fixture")])
    registry.register("market.performance", price)
    return AgentV3(registry=registry, brain=DemoBrain(), config=AgentV3Config(debate=False), **kwargs)
