"""Registration helper for in-process or queued quantitative-lab services."""

from __future__ import annotations

from v2.agent_v2.execution import CapabilityRegistry, ExecutionContext
from v2.agent_v2.ports import LabPort

_LAB_CAPABILITIES = (
    "lab.screen",
    "lab.backtest",
    "lab.sweep",
    "lab.event_study",
    "lab.committee",
)


def register_lab_capabilities(registry: CapabilityRegistry, lab: LabPort) -> None:
    for name in _LAB_CAPABILITIES:

        def handler(arguments, context: ExecutionContext, capability=name):
            return lab.run(capability, arguments, context)

        registry.register(name, handler)
