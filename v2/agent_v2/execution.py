"""Policy-gated capability registry and deterministic DAG executor."""

from __future__ import annotations

import logging
import json
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, wait
from v2.usage_context import ContextExecutor as ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from v2.agent_v2.catalog import CapabilityCatalog
from v2.agent_v2.evidence import EvidenceLedger
from v2.agent_v2.models import (
    BudgetClass,
    EvidenceItem,
    ExecutionPlan,
    NormalizedRequest,
    PlanTask,
    ProgressEvent,
    ResultStatus,
    RunStatus,
    ToolEnvelope,
)
from v2.agent_v2.ports import CapabilityHandler, ProgressSink

#: Upper bound on children one fan-out task may spawn, whatever the source lists.
FAN_OUT_MAX = 12

_TASK_LIMITS = {
    BudgetClass.DIRECT: 1,
    BudgetClass.FOCUSED: 2,
    BudgetClass.STANDARD: 5,
    BudgetClass.COMPARISON: 5,
    BudgetClass.PORTFOLIO: 7,
    BudgetClass.LAB: 2,
    BudgetClass.DEEP: 12,
}

# Wall-clock allowance for the whole capability plan.  A capability that is
# still running when the deadline passes is reported as failed and the tasks
# that have not started are skipped; the run always ends with an answer.
_TIME_LIMITS_SECONDS = {
    BudgetClass.DIRECT: 30.0,
    BudgetClass.FOCUSED: 60.0,
    BudgetClass.STANDARD: 180.0,
    BudgetClass.COMPARISON: 240.0,
    BudgetClass.PORTFOLIO: 240.0,
    BudgetClass.LAB: 600.0,
    BudgetClass.DEEP: 1800.0,
}

logger = logging.getLogger(__name__)


def task_limit(budget: BudgetClass) -> int:
    """Maximum number of tasks the executor accepts for one budget class."""
    return _TASK_LIMITS[budget]


def time_limit(budget: BudgetClass) -> float:
    """Wall-clock seconds the executor grants one budget class."""
    return _TIME_LIMITS_SECONDS[budget]


class PlanValidationError(ValueError):
    """The planned task graph cannot be executed safely."""


class RunBoard:
    """What the run knows so far, shared by every task, plus bounded follow-up requests.

    Tasks read the results and evidence that finished before them (a
    sub-agent can see what another already read) and may request a
    follow-up task; the engine schedules requested tasks after the current
    wave, within the deadline and a cap, and never a mutation.
    """

    def __init__(self, *, max_requests: int = 4) -> None:
        self._lock = threading.Lock()
        self.results: list[ToolEnvelope] = []
        self.evidence: dict[str, EvidenceItem] = {}
        self.requested: list[PlanTask] = []
        self.accepted: list[str] = []
        self.refused: list[str] = []
        self.max_requests = max(0, max_requests)
        self._known: set[str] = set()

    def post(self, result: ToolEnvelope) -> None:
        with self._lock:
            self.results.append(result)
            for item in result.evidence:
                self.evidence.setdefault(item.id, item)

    def results_for(self, capability: str) -> list[ToolEnvelope]:
        with self._lock:
            return [result for result in self.results if result.capability == capability]

    def evidence_where(self, **fields: Any) -> list[EvidenceItem]:
        """Evidence whose fields or metadata carry every given value (``entity="ARM"``, ``evidence_scope="filing_event"``)."""

        with self._lock:
            return [item for item in self.evidence.values() if all((getattr(item, key, None) if hasattr(item, key) else item.metadata.get(key)) == value for key, value in fields.items())]

    def request(self, capability: str, arguments: dict[str, Any] | None = None, *, purpose: str = "", requested_by: str = "") -> bool:
        """Ask for a follow-up task; False when the cap is reached or the same request was already made."""

        key = f"{capability}:{json.dumps(arguments or {}, sort_keys=True, ensure_ascii=False)}"
        with self._lock:
            if key in self._known:
                return False
            if len(self.accepted) >= self.max_requests:
                self.refused.append(key)
                return False
            self._known.add(key)
            task = PlanTask(f"followup-{len(self.accepted) + 1}", capability, dict(arguments or {}), purpose=purpose or f"follow-up requested by {requested_by or 'a task'}", required=False)
            self.requested.append(task)
            self.accepted.append(key)
            return True

    def take_requests(self) -> list[PlanTask]:
        with self._lock:
            tasks, self.requested = self.requested, []
            return tasks


@dataclass(frozen=True)
class ExecutionContext:
    run_id: str
    request: NormalizedRequest
    budget: BudgetClass
    allow_mutations: bool = False
    allow_web: bool = False
    on_progress: ProgressSink | None = None
    #: Absolute ``time.monotonic()`` deadline; ``None`` means the budget default.
    deadline: float | None = None
    #: Shared state of the run; every task can read it and request follow-ups.
    board: RunBoard = field(default_factory=RunBoard)
    #: Set by the user to stop the run; the engine and the sub-agent loops check it between steps.
    cancel_event: threading.Event | None = None

    def remaining_seconds(self) -> float:
        limit = self.deadline if self.deadline is not None else time.monotonic() + time_limit(self.budget)
        return limit - time.monotonic()

    @property
    def cancelled(self) -> bool:
        return self.cancel_event is not None and self.cancel_event.is_set()

    def emit(self, message: str, *, task_id: str = "", capability: str = "") -> None:
        if not self.on_progress:
            return
        try:
            self.on_progress(ProgressEvent(self.run_id, RunStatus.EXECUTING, message, task_id, capability))
        except Exception:  # noqa: BLE001 — progress reporting never breaks a run
            pass


@dataclass
class ExecutionOutcome:
    results: list[ToolEnvelope] = field(default_factory=list)
    ledger: EvidenceLedger = field(default_factory=EvidenceLedger)
    #: ``completed`` or ``deadline``; the orchestrator surfaces it on the result.
    stop_reason: str = "completed"
    elapsed_ms: int = 0


_WINDOW_LABELS = {"1d": "当日", "5d": "近 5 日", "1m": "近 1 月", "3m": "近 3 月", "1y": "近 1 年"}


def _fan_out_table(parent_id: str, children: list[ToolEnvelope]) -> ToolEnvelope | None:
    """One citeable ranking per return window across a performance fan-out.

    Twelve per-holding envelopes leave "who did best this week" to the
    synthesizer's arithmetic; the table is the sorted list under one id, so
    the answer cites a row instead of reconstructing the order.
    """

    rows = [child for child in children if child.capability == "market.performance" and child.ok and isinstance(child.metrics.get("returns"), dict)]
    if len(rows) < 2:
        return None
    intraday = any(child.metrics.get("is_intraday") for child in rows)
    evidence: list[EvidenceItem] = []
    for window, label in _WINDOW_LABELS.items():
        pairs = [(child.subject, float(child.metrics["returns"][window])) for child in rows if isinstance(child.metrics["returns"].get(window), (int, float))]
        if len(pairs) < 2:
            continue
        pairs.sort(key=lambda pair: pair[1], reverse=True)
        note = "（截至查询时，盘中口径）" if intraday and window == "1d" else ""
        claim = f"{label}回报排序{note}：" + "、".join(f"{ticker} {value:+.2%}" for ticker, value in pairs) + f"；最高 {pairs[0][0]}，最低 {pairs[-1][0]}。"
        evidence.append(
            EvidenceItem(
                id=f"evidence-ranking-{window}-{parent_id}",
                entity=",".join(ticker for ticker, _ in pairs),
                claim=claim,
                metric=f"return_{window}",
                source_id="market_data",
                source_title="行情排序",
                metadata={"citation_kind": "metrics", "ranking": True, "window": window, "verified": True, "returns": dict(pairs)},
            )
        )
    if not evidence:
        return None
    return ToolEnvelope(
        "market.performance",
        ResultStatus.COMPLETED,
        subject=parent_id,
        summary=evidence[0].claim,
        evidence=evidence,
        metadata={"fan_out_table": True, "answer_guidance": "排名问题先引用对应窗口的“回报排序”那条证据点名最高和最低，再谈个股。"},
    )


class CapabilityRegistry:
    def __init__(self, catalog: CapabilityCatalog) -> None:
        self.catalog = catalog
        self._handlers: dict[str, CapabilityHandler] = {}

    def register(self, name: str, handler: CapabilityHandler) -> None:
        if self.catalog.get(name) is None:
            raise KeyError(f"capability is not declared in the catalog: {name}")
        self._handlers[name] = handler

    def registered(self, name: str) -> bool:
        return name in self._handlers

    def execute(self, task: PlanTask, context: ExecutionContext) -> ToolEnvelope:
        spec = self.catalog.get(task.capability)
        if spec is None:
            return ToolEnvelope(task.capability, ResultStatus.FAILED, errors=["unknown capability"])
        schema = spec.input_schema
        properties = schema.get("properties", {})
        missing = [name for name in schema.get("required", []) if name not in task.arguments]
        extra = [name for name in task.arguments if name not in properties]
        if missing or extra:
            details = []
            if missing:
                details.append("missing: " + ", ".join(missing))
            if extra:
                details.append("unknown: " + ", ".join(extra))
            return ToolEnvelope(task.capability, ResultStatus.FAILED, errors=["invalid arguments (" + "; ".join(details) + ")"])
        if spec.mutating and not context.allow_mutations:
            return ToolEnvelope(
                task.capability,
                ResultStatus.FAILED,
                errors=["mutation requires explicit confirmation"],
            )
        if spec.pack == "web" and not context.allow_web:
            return ToolEnvelope(task.capability, ResultStatus.FAILED, errors=["web fallback is disabled"])
        handler = self._handlers.get(task.capability)
        if handler is None:
            return ToolEnvelope(
                task.capability,
                ResultStatus.FAILED,
                errors=["capability adapter is not registered"],
            )
        started = time.time()
        try:
            result = handler(dict(task.arguments), context)
            result.elapsed_ms = result.elapsed_ms or int((time.time() - started) * 1000)
            return result
        except Exception as exc:  # a capability failure is data for the orchestrator
            logger.warning("capability %s failed: %s: %s", task.capability, type(exc).__name__, exc)
            return ToolEnvelope(
                task.capability,
                ResultStatus.FAILED,
                errors=[f"{type(exc).__name__}: {str(exc)[:300]}"],
                elapsed_ms=int((time.time() - started) * 1000),
            )


class ExecutionEngine:
    def __init__(self, registry: CapabilityRegistry, *, max_parallel: int = 4) -> None:
        self.registry = registry
        self.max_parallel = max(1, max_parallel)

    def validate(self, plan: ExecutionPlan) -> None:
        if len(plan.tasks) > _TASK_LIMITS[plan.budget]:
            raise PlanValidationError(f"plan has {len(plan.tasks)} tasks; {plan.budget.value} budget allows {_TASK_LIMITS[plan.budget]}")
        ids = [task.id for task in plan.tasks]
        if len(ids) != len(set(ids)):
            raise PlanValidationError("plan task ids must be unique")
        known = set(ids)
        if any(set(task.depends_on) - known for task in plan.tasks):
            raise PlanValidationError("plan contains an unknown dependency")
        for task in plan.tasks:
            if task.fan_out is None:
                continue
            source = str(task.fan_out.get("from") or "")
            if source not in known or source == task.id:
                raise PlanValidationError(f"fan-out task {task.id} names an unknown source")
            if not task.fan_out.get("argument"):
                raise PlanValidationError(f"fan-out task {task.id} has no target argument")
            rank = task.fan_out.get("rank")
            if rank is not None and (not isinstance(rank, dict) or not rank.get("field") or not rank.get("key")):
                raise PlanValidationError(f"fan-out task {task.id} has an invalid rank spec")

    @staticmethod
    def _ordered_values(spec: dict[str, Any], source: ToolEnvelope | None) -> list[Any]:
        """Source values in fan-out order: ranked by a row field when the plan asks for it."""

        field_name = str(spec.get("field") or "tickers")
        values = list((source.metadata.get(field_name) if source is not None else None) or [])
        rank = spec.get("rank")
        if not isinstance(rank, dict) or source is None:
            return values
        rows = source.metadata.get(str(rank.get("field") or ""))
        key = str(rank.get("key") or "")
        argument = str(spec.get("argument") or "")
        if not isinstance(rows, list) or not key:
            return values
        ranked: list[tuple[float, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            value = row.get(argument, row.get("ticker"))
            number = row.get(key)
            if value is None or not isinstance(number, (int, float)):
                continue
            ranked.append((float(number), value))
        if not ranked:
            return values
        ranked.sort(key=lambda pair: pair[0], reverse=bool(rank.get("descending")))
        ordered = [value for _, value in ranked]
        # Values the table did not cover keep their original place after the ranked ones.
        return list(dict.fromkeys([*ordered, *values]))

    @classmethod
    def _expand(cls, task: PlanTask, completed: dict[str, ToolEnvelope]) -> tuple[list[PlanTask], ToolEnvelope | None]:
        """Replace a fan-out template with one child per source value.

        Returns the children and, when the source listed more values than the
        cap allows, a coverage note the answer must disclose.
        """

        spec = task.fan_out or {}
        source = completed.get(str(spec.get("from") or ""))
        values = cls._ordered_values(spec, source)
        limit = max(1, min(int(spec.get("max") or FAN_OUT_MAX), FAN_OUT_MAX))
        note: ToolEnvelope | None = None
        if len(values) > limit:
            chosen = ", ".join(str(value) for value in values[:limit])
            rest = ", ".join(str(value) for value in values[limit:])
            ordering = "按相关性排序后" if isinstance(spec.get("rank"), dict) else "按列表顺序"
            claim = f"{task.capability} 只覆盖了 {len(values)} 个对象中的 {limit} 个（{ordering}）：{chosen}；未覆盖：{rest}。"
            note = ToolEnvelope(
                task.capability,
                ResultStatus.PARTIAL_DATA,
                subject=task.id,
                summary="",
                evidence=[
                    EvidenceItem(
                        id=f"fan-out-coverage-{task.id}",
                        entity=task.id,
                        claim=claim,
                        source_id="execution_engine",
                        source_title="Fan-out coverage",
                        metadata={"citation_kind": "limitations", "verified": True, "covered": list(values[:limit]), "uncovered": list(values[limit:])},
                    )
                ],
                limitations=[claim],
                metadata={"fan_out_coverage": {"covered": list(values[:limit]), "uncovered": list(values[limit:])}},
            )
        children: list[PlanTask] = []
        for value in values[:limit]:
            children.append(
                PlanTask(
                    id=f"{task.id}[{value}]",
                    capability=task.capability,
                    arguments={**task.arguments, str(spec["argument"]): value},
                    depends_on=task.depends_on,
                    required=task.required,
                    purpose=task.purpose,
                )
            )
        return children, note

    def run(self, plan: ExecutionPlan, context: ExecutionContext) -> ExecutionOutcome:
        self.validate(plan)
        started = time.monotonic()
        deadline = context.deadline if context.deadline is not None else started + time_limit(context.budget)
        # Handlers (sub-agents in particular) bound their own loops by the
        # coordinator's remaining time, so the deadline must be visible to
        # them.  The context is frozen for everything else; this one field is
        # set once here, before any handler runs.
        if context.deadline is None:
            object.__setattr__(context, "deadline", deadline)

        pending = {task.id: task for task in plan.tasks}
        completed: dict[str, ToolEnvelope] = {}
        #: fan-out template id -> child ids, so dependants of a template wait for every child.
        expanded: dict[str, list[str]] = {}
        outcome = ExecutionOutcome()

        def finish(task: PlanTask, result: ToolEnvelope) -> None:
            completed[task.id] = result
            outcome.results.append(result)
            pending.pop(task.id, None)
            outcome.ledger.ingest(result)
            context.board.post(result)

        def adopt_requests() -> None:
            # Follow-ups a task asked for on the board join the plan; a
            # mutation or an unknown capability is refused, and a request
            # for the same work twice was refused by the board already.
            for task in context.board.take_requests():
                spec = self.registry.catalog.get(task.capability)
                if spec is None or spec.mutating:
                    context.board.refused.append(task.capability)
                    continue
                pending[task.id] = task
                context.emit(f"follow-up: {task.purpose}", task_id=task.id, capability=task.capability)

        def satisfied(dependency: str) -> bool:
            if dependency in expanded:
                return all(child in completed for child in expanded[dependency])
            return dependency in completed

        def dependency_ok(dependency: str) -> bool:
            if dependency in expanded:
                return all(completed[child].ok for child in expanded[dependency])
            return completed[dependency].ok

        while pending:
            if context.cancelled:
                for task in list(pending.values()):
                    finish(task, ToolEnvelope(task.capability, ResultStatus.SKIPPED, errors=["cancelled by the user"]))
                outcome.stop_reason = "cancelled"
                break
            ready = [task for task in pending.values() if all(satisfied(dependency) for dependency in task.depends_on)]
            if not ready:
                raise PlanValidationError("plan dependency cycle detected")

            runnable: list[PlanTask] = []
            for task in ready:
                failed_dependency = any(not dependency_ok(dep) for dep in task.depends_on)
                if failed_dependency and task.required:
                    finish(task, ToolEnvelope(task.capability, ResultStatus.SKIPPED, errors=["required dependency failed"]))
                    continue
                if task.fan_out is not None:
                    children, note = self._expand(task, completed)
                    pending.pop(task.id, None)
                    if note is not None:
                        # The truncation is a limitation of this run's evidence,
                        # so it travels with the results and is citable.
                        outcome.results.append(note)
                        outcome.ledger.ingest(note)
                    if not children:
                        expanded[task.id] = []
                        result = ToolEnvelope(task.capability, ResultStatus.SKIPPED, errors=["fan-out source listed no values"])
                        completed[task.id] = result
                        outcome.results.append(result)
                        continue
                    expanded[task.id] = [child.id for child in children]
                    for child in children:
                        pending[child.id] = child
                    runnable.extend(children)
                    continue
                runnable.append(task)
            if not runnable:
                continue

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                for task in list(pending.values()):
                    finish(task, ToolEnvelope(task.capability, ResultStatus.SKIPPED, errors=["wall-clock budget exhausted before the task started"]))
                outcome.stop_reason = "deadline"
                break

            for task in runnable:
                context.emit(task.purpose or f"running {task.capability}", task_id=task.id, capability=task.capability)
            pool = ThreadPoolExecutor(max_workers=min(self.max_parallel, len(runnable)))
            futures: dict[Future[ToolEnvelope], PlanTask] = {pool.submit(self.registry.execute, task, context): task for task in runnable}
            timed_out = False
            try:
                outstanding = set(futures)
                while outstanding:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        timed_out = True
                        break
                    done, outstanding = wait(outstanding, timeout=remaining, return_when=FIRST_COMPLETED)
                    for future in done:
                        finish(futures[future], future.result())
                if timed_out:
                    for future in outstanding:
                        task = futures[future]
                        finish(task, ToolEnvelope(task.capability, ResultStatus.FAILED, errors=[f"timed out after {time_limit(context.budget):.0f}s wall-clock budget"]))
            finally:
                # Threads that overran the deadline cannot be killed; they are
                # left to finish in the background without blocking the answer.
                pool.shutdown(wait=False, cancel_futures=True)
            if timed_out:
                for task in list(pending.values()):
                    finish(task, ToolEnvelope(task.capability, ResultStatus.SKIPPED, errors=["wall-clock budget exhausted"]))
                outcome.stop_reason = "deadline"
                break
            adopt_requests()

        for parent_id, child_ids in expanded.items():
            table = _fan_out_table(parent_id, [completed[child] for child in child_ids if child in completed])
            if table is not None:
                outcome.results.append(table)
                outcome.ledger.ingest(table)
        outcome.elapsed_ms = int((time.monotonic() - started) * 1000)
        return outcome
