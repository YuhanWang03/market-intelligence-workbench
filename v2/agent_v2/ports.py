"""Dependency inversion points for models, tools, search, and user channels."""

from __future__ import annotations

from typing import Any, Callable, Protocol

from v2.agent_v2.models import (
    AgentResult,
    EvidenceItem,
    ExecutionPlan,
    NormalizedRequest,
    ProgressEvent,
    RouteDecision,
    SessionResolution,
    ToolEnvelope,
)

ProgressSink = Callable[[ProgressEvent], None]


class PlannerPort(Protocol):
    def plan(self, request: NormalizedRequest, route: RouteDecision) -> ExecutionPlan:
        ...


class SynthesizerPort(Protocol):
    def synthesize(
        self,
        request: NormalizedRequest,
        plan: ExecutionPlan,
        results: list[ToolEnvelope],
        evidence: list[EvidenceItem],
    ) -> str:
        ...


class CapabilityPort(Protocol):
    def execute(self, arguments: dict[str, Any], context: Any) -> ToolEnvelope:
        ...


class WebSearchPort(Protocol):
    def search(
        self,
        query: str,
        *,
        topic: str,
        ticker: str = "",
        recency_days: int = 30,
        run_id: str = "",
    ) -> ToolEnvelope:
        ...


class LabPort(Protocol):
    def run(self, capability: str, arguments: dict[str, Any], context: Any) -> ToolEnvelope:
        ...


class SessionPort(Protocol):
    def resolve(self, session_id: str, text: str) -> SessionResolution:
        ...

    def record(self, result: AgentResult) -> None:
        ...

    def set_pending(self, session_id: str, plan: ExecutionPlan) -> None:
        """Hold a write plan until the user confirms or moves on."""
        ...

    def pop_pending(self, session_id: str) -> ExecutionPlan | None:
        ...

    # Optional (read through getattr): recent_turns(session_id, n), previous_turn(session_id),
    # set_clarification(session_id, original_text, question), pop_clarification(session_id).


class ChannelPort(Protocol):
    async def progress(self, event: ProgressEvent) -> None:
        ...

    async def deliver(self, result: AgentResult) -> None:
        ...


class AgentPort(Protocol):
    def run(
        self,
        text: str,
        *,
        session_id: str = "",
        allow_web: bool = False,
        on_progress: ProgressSink | None = None,
    ) -> AgentResult:
        ...


CapabilityHandler = Callable[[dict[str, Any], Any], ToolEnvelope]
