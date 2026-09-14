"""Offline smoke demo for the Agent V2 orchestration contracts."""

from __future__ import annotations

import json

from v2.agent_v2.catalog import default_catalog
from v2.agent_v2.execution import CapabilityRegistry
from v2.agent_v2.models import EvidenceItem, ResultStatus, ToolEnvelope
from v2.agent_v2.orchestrator import AgentV2


def _research(arguments, context) -> ToolEnvelope:
    ticker = arguments["ticker"]
    claim = f"{ticker} 的演示研究结果来自结构化证据。"
    evidence = EvidenceItem(
        id=f"demo-{ticker}-001",
        entity=ticker,
        claim=claim,
        source_id="offline_fixture",
    )
    return ToolEnvelope(
        "research.stock",
        ResultStatus.COMPLETED,
        subject=ticker,
        summary=claim,
        evidence=[evidence],
    )


def main() -> int:
    catalog = default_catalog()
    registry = CapabilityRegistry(catalog)
    registry.register("research.stock", _research)
    result = AgentV2(catalog=catalog, registry=registry).run("分析 NVDA 的风险")
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
