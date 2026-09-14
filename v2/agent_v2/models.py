"""Stable data contracts shared by Agent V2 components and channel adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RouteKind(str, Enum):
    GENERAL_KNOWLEDGE = "general_knowledge"
    FAST_LOOKUP = "fast_lookup"
    RESEARCH = "research"
    LAB = "lab"
    COMMAND = "command"
    ASYNC = "async"


class RunStatus(str, Enum):
    RECEIVED = "received"
    ROUTED = "routed"
    PLANNED = "planned"
    WAITING_CONFIRMATION = "waiting_confirmation"
    WAITING_CLARIFICATION = "waiting_clarification"
    EXECUTING = "executing"
    SYNTHESIZING = "synthesizing"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ResultStatus(str, Enum):
    COMPLETED = "completed"
    PARTIAL_DATA = "partial_data"
    PARTIAL_ERROR = "partial_error"
    FAILED = "failed"
    CACHED = "cached"
    SKIPPED = "skipped"


class AnswerMode(str, Enum):
    TOOL_GROUNDED = "tool_grounded"
    RESEARCH_GROUNDED = "research_grounded"
    WEB_GROUNDED = "web_grounded"
    GENERAL_KNOWLEDGE = "general_knowledge"
    MIXED = "mixed"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class BudgetClass(str, Enum):
    DIRECT = "direct"
    FOCUSED = "focused"
    STANDARD = "standard"
    COMPARISON = "comparison"
    PORTFOLIO = "portfolio"
    LAB = "lab"
    DEEP = "deep"


@dataclass(frozen=True)
class NormalizedRequest:
    original_text: str
    text: str
    session_id: str = ""
    entities: tuple[str, ...] = ()
    forced_agent: bool = False
    allow_web: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SessionResolution:
    text: str
    rewritten: bool = False
    antecedent: str = ""
    note: str = ""
    #: What the previous turn was about beyond the stock itself (the column,
    #: the period, the value), so a follow-up "为什么跌" keeps its referent.
    frame: dict = field(default_factory=dict)


@dataclass(frozen=True)
class RouteDecision:
    kind: RouteKind
    packs: tuple[str, ...]
    reason: str
    asynchronous: bool = False
    #: The intent the route was decided from (``intent.Intent``); the planner reads it.
    intent: Any = None



@dataclass(frozen=True)
class PlanTask:
    id: str
    capability: str
    arguments: dict[str, Any] = field(default_factory=dict)
    depends_on: tuple[str, ...] = ()
    required: bool = True
    purpose: str = ""
    #: Expand this task once per value a finished task produced, e.g.
    #: ``{"from": "holdings", "field": "tickers", "argument": "ticker", "max": 6}``
    #: runs the capability for each ticker the ``holdings`` result listed in
    #: ``metadata["tickers"]``.  The plan stays static; the data decides the width.
    fan_out: dict[str, Any] | None = None


@dataclass(frozen=True)
class ExecutionPlan:
    objective: str
    route: RouteKind
    tasks: tuple[PlanTask, ...] = ()
    budget: BudgetClass = BudgetClass.DIRECT
    answer_mode: AnswerMode = AnswerMode.TOOL_GROUNDED
    requires_confirmation: bool = False
    web_fallback_allowed: bool = False
    assumptions: tuple[str, ...] = ()
    #: A complete answer the planner can give without running anything: a
    #: capability overview, or the exact clarification a request needs.
    direct_answer: str = ""
    #: What the question refers to beyond its words (a position's loss, a
    #: stretch of decline), when the planner built the plan around it.
    frame: dict = field(default_factory=dict)


@dataclass(frozen=True)
class EvidenceItem:
    id: str
    entity: str
    claim: str
    metric: str = ""
    value: Any = None
    unit: str = ""
    period: str = ""
    as_of: str = ""
    source_id: str = ""
    source_title: str = ""
    source_url: str = ""
    confidence: float | None = None
    producer_run_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "entity": self.entity,
            "claim": self.claim,
            "metric": self.metric,
            "value": self.value,
            "unit": self.unit,
            "period": self.period,
            "as_of": self.as_of,
            "source_id": self.source_id,
            "source_title": self.source_title,
            "source_url": self.source_url,
            "confidence": self.confidence,
            "producer_run_id": self.producer_run_id,
            "metadata": dict(self.metadata),
        }


@dataclass
class ToolEnvelope:
    capability: str
    status: ResultStatus
    subject: str = ""
    as_of: str = ""
    summary: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    findings: list[dict[str, Any]] = field(default_factory=list)
    evidence: list[EvidenceItem] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    run_id: str = ""
    cache_hit: bool = False
    elapsed_ms: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status in {
            ResultStatus.COMPLETED,
            ResultStatus.CACHED,
            ResultStatus.PARTIAL_DATA,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability": self.capability,
            "status": self.status.value,
            "subject": self.subject,
            "as_of": self.as_of,
            "summary": self.summary,
            "metrics": dict(self.metrics),
            "findings": list(self.findings),
            "evidence": [item.to_dict() for item in self.evidence],
            "limitations": list(self.limitations),
            "errors": list(self.errors),
            "run_id": self.run_id,
            "cache_hit": self.cache_hit,
            "elapsed_ms": self.elapsed_ms,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ProgressEvent:
    run_id: str
    status: RunStatus
    message: str
    task_id: str = ""
    capability: str = ""


@dataclass(frozen=True)
class PendingMutation:
    """A confirmed-in-principle write the user still has to approve verbatim."""

    operation: str
    payload: dict[str, Any]
    description: str


@dataclass(frozen=True)
class VerificationReport:
    ok: bool = True
    unknown_citations: tuple[str, ...] = ()
    ungrounded_numbers: tuple[str, ...] = ()
    traced_numbers: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass
class AgentResult:
    run_id: str
    request: NormalizedRequest
    route: RouteDecision
    plan: ExecutionPlan
    status: RunStatus
    answer: str
    answer_mode: AnswerMode
    results: list[ToolEnvelope] = field(default_factory=list)
    evidence: list[EvidenceItem] = field(default_factory=list)
    verification: VerificationReport = field(default_factory=VerificationReport)
    elapsed_ms: int = 0
    error: str = ""
    stop_reason: str = ""
    pending_mutation: PendingMutation | None = None
    #: How the answer text was produced (model draft, repaired draft,
    #: deterministic fallback) and what the verifier said about each attempt.
    synthesis: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "request": {
                "original_text": self.request.original_text,
                "text": self.request.text,
                "session_id": self.request.session_id,
                "entities": list(self.request.entities),
                "metadata": dict(self.request.metadata),
            },
            "route": {
                "kind": self.route.kind.value,
                "packs": list(self.route.packs),
                "reason": self.route.reason,
                "asynchronous": self.route.asynchronous,
            },
            "plan": {
                "objective": self.plan.objective,
                "route": self.plan.route.value,
                "budget": self.plan.budget.value,
                "answer_mode": self.plan.answer_mode.value,
                "requires_confirmation": self.plan.requires_confirmation,
                "web_fallback_allowed": self.plan.web_fallback_allowed,
                "assumptions": list(self.plan.assumptions),
                "tasks": [
                    {
                        "id": task.id,
                        "capability": task.capability,
                        "arguments": dict(task.arguments),
                        "depends_on": list(task.depends_on),
                        "required": task.required,
                        "purpose": task.purpose,
                        "fan_out": dict(task.fan_out) if task.fan_out else None,
                    }
                    for task in self.plan.tasks
                ],
                "direct_answer": self.plan.direct_answer,
            },
            "status": self.status.value,
            "answer": self.answer,
            "answer_mode": self.answer_mode.value,
            "results": [result.to_dict() for result in self.results],
            "sub_agents": sub_agent_summaries(self.results),
            "evidence": [item.to_dict() for item in self.evidence],
            "verification": {
                "ok": self.verification.ok,
                "unknown_citations": list(self.verification.unknown_citations),
                "ungrounded_numbers": list(self.verification.ungrounded_numbers),
                "warnings": list(self.verification.warnings),
            },
            "elapsed_ms": self.elapsed_ms,
            "error": self.error,
            "stop_reason": self.stop_reason,
            "synthesis": dict(self.synthesis),
            "pending_mutation": (
                {
                    "operation": self.pending_mutation.operation,
                    "payload": dict(self.pending_mutation.payload),
                    "description": self.pending_mutation.description,
                }
                if self.pending_mutation
                else None
            ),
        }


def sub_agent_summaries(results: list[ToolEnvelope]) -> list[dict[str, Any]]:
    """What each sub-agent did during the run: rounds, calls, stop reason and its per-round trace.

    A sub-agent's envelope carries ``metadata["agent"]`` (its summary) and
    ``metadata["trace"]`` (one entry per model turn); a reader the
    attributor called shows up nested under it.  Surfaces render this list
    instead of digging through envelopes.
    """

    summaries: list[dict[str, Any]] = []
    for result in results:
        agent = result.metadata.get("agent")
        if not isinstance(agent, dict):
            continue
        entry = {
            "capability": result.capability,
            "name": str(agent.get("name") or ""),
            "label": str(agent.get("label") or agent.get("name") or ""),
            "subject": str(agent.get("subject") or result.subject),
            "rounds": int(agent.get("rounds") or 0),
            "llm_calls": int(agent.get("llm_calls") or 0),
            "elapsed_ms": int(agent.get("elapsed_ms") or 0),
            "seconds_allowed": agent.get("seconds_allowed"),
            "stop_reason": str(agent.get("stop_reason") or ""),
            "calls": dict(agent.get("calls") or {}),
            "intraday": bool(agent.get("intraday", False)),
            "yield": dict(agent.get("yield") or {}),
            "memory": dict(agent.get("memory") or {}),
            "challenge": dict(agent.get("challenge") or {}),
            "notes": [str(note) for note in (agent.get("notes") or []) if note],
            "stance": str(agent.get("stance") or ""),
            "trace": [dict(step) for step in (result.metadata.get("trace") or []) if isinstance(step, dict)],
            "nested": [
                {"label": "申报阅读", "rounds": int(run.get("rounds") or 0), "elapsed_ms": int(run.get("elapsed_ms") or 0), "stop_reason": str(run.get("stop_reason") or ""), "calls": {"filings": run.get("filings"), "sections_read": run.get("sections_read"), "events": run.get("events")}, "trace": [dict(step) for step in (run.get("trace") or []) if isinstance(step, dict)]}
                for run in (agent.get("reader_runs") or [])
                if isinstance(run, dict)
            ],
        }
        summaries.append(entry)
    return summaries
