"""Recorded capability envelopes for zero-network Agent V2 evaluation."""

from __future__ import annotations

import hashlib
from typing import Any

from v2.agent_v2.catalog import default_catalog
from v2.agent_v2.execution import CapabilityRegistry
from v2.agent_v2.models import EvidenceItem, ResultStatus, ToolEnvelope


class EvalSynthesizer:
    supports_general_knowledge = True

    def synthesize(self, request, plan, results, evidence) -> str:
        if not results:
            if plan.answer_mode.value == "general_knowledge":
                return "这是通用知识回答，未使用实时数据：自由现金流用于衡量经营现金流扣除资本支出后的剩余现金。"
            return "没有执行任何数据能力。"
        return "\n".join(f"{item.claim} [{item.id}]" for item in evidence)


def build_eval_registry() -> CapabilityRegistry:
    catalog = default_catalog()
    registry = CapabilityRegistry(catalog)

    def make_handler(capability: str):
        def handler(arguments: dict[str, Any], context) -> ToolEnvelope:
            subject = str(arguments.get("ticker") or ",".join(arguments.get("tickers", [])) or "portfolio")
            digest = hashlib.sha256(f"{capability}:{subject}".encode()).hexdigest()[:10]
            claim = f"{capability} returned evidence for {subject}."
            item = EvidenceItem(f"eval-{digest}", subject, claim, source_id="eval_fixture")
            metadata = {"tickers": ["NVDA", "AMD"]} if capability in {"account.portfolio", "state.read"} else {}
            return ToolEnvelope(capability, ResultStatus.COMPLETED, subject=subject, summary=claim, evidence=[item], metadata=metadata)

        return handler

    for spec in catalog.specs():
        if not spec.mutating and spec.pack != "web":
            registry.register(spec.name, make_handler(spec.name))
    return registry
