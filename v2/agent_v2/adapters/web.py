"""Registration helper for a bounded WebSearchPort implementation."""

from __future__ import annotations

from v2.agent_v2.execution import CapabilityRegistry, ExecutionContext
from v2.agent_v2.ports import WebSearchPort


def register_web_capability(registry: CapabilityRegistry, search: WebSearchPort) -> None:
    def handler(arguments, context: ExecutionContext):
        return search.search(
            str(arguments.get("query") or ""),
            topic=str(arguments.get("topic") or "general"),
            ticker=str(arguments.get("ticker") or ""),
            recency_days=int(arguments.get("recency_days") or 30),
            run_id=context.run_id,
        )

    registry.register("web.research", handler)
