"""Top-level Agent V2 state machine."""

from __future__ import annotations

import logging
import os
import re
import time
import uuid
from dataclasses import dataclass, replace
from typing import Any

from v2.agent_v2.catalog import CapabilityCatalog, default_catalog
from v2.agent_v2.evidence import EvidenceConflictError
from v2.agent_v2.execution import (
    CapabilityRegistry,
    ExecutionContext,
    ExecutionEngine,
    ExecutionOutcome,
    PlanValidationError,
    time_limit,
)
from v2.agent_v2.models import (
    BudgetClass,
    AgentResult,
    AnswerMode,
    ExecutionPlan,
    NormalizedRequest,
    PendingMutation,
    PlanTask,
    ProgressEvent,
    RouteDecision,
    RouteKind,
    RunStatus,
    VerificationReport,
)
from v2.agent_v2.planning import RulePlanner
from v2.agent_v2.ports import PlannerPort, ProgressSink, SessionPort, SynthesizerPort
from v2.agent_v2.routing import normalize_request, route
from v2.agent_v2.synthesis import EvidenceSummarySynthesizer
from v2.agent_v2.verification import verify_answer

#: Seconds the one web attempt gets when the internal step spent the budget.
WEB_FALLBACK_GRACE_SECONDS = 45.0

_CONFIRM = re.compile(r"^\s*(?:确认|确定|是的?|好的?|执行|yes|y|ok|confirm)\s*[。！!.]?\s*$", re.I)
_CANCEL = re.compile(r"^\s*(?:取消|不用了?|不要|算了|否|no|n|cancel)\s*[。！!.]?\s*$", re.I)


logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class AgentV2Config:
    max_parallel: int = 4
    enable_web_fallback: bool = False
    #: Bypass the confirmation step; only for tests and trusted automation.
    allow_mutations: bool = False
    #: Override the per-budget wall-clock allowance (seconds) for every run.
    max_seconds: float | None = None
    #: Append every sub-agent run to the run ledger (data/agent_v2_subagents.jsonl).
    record_sub_agents: bool = True
    #: Append every capability outcome to data/agent_v2_capabilities.jsonl (the data-source health report reads it).
    record_capabilities: bool = True
    #: After a research answer is verified, one adversarial pass lists the objections the run's evidence supports.
    debate: bool = True
    #: When the debater objects, one bounded rewrite that must pass the verifier again; off keeps objections display-only.
    debate_revision: bool = True
    #: Append every routing decision (intent, its source, the plan) to
    #: data/agent_v2_intents.jsonl.  ``None`` means "when the live runtime
    #: has a model"; tests and offline evals stay silent.
    record_intents: bool | None = None
    #: Ask the user one question instead of planning when the model classifier's
    #: confidence is below this and it named the gap; 0 turns clarification off.
    clarify_below: float = 0.6


class AgentV2:
    def __init__(
        self,
        *,
        catalog: CapabilityCatalog | None = None,
        registry: CapabilityRegistry | None = None,
        planner: PlannerPort | None = None,
        synthesizer: SynthesizerPort | None = None,
        session: SessionPort | None = None,
        config: AgentV2Config | None = None,
        classifier: Any = None,
        memory: Any = None,
    ) -> None:
        self.catalog = catalog or default_catalog()
        self.registry = registry or CapabilityRegistry(self.catalog)
        self.planner = planner or RulePlanner()
        self.synthesizer = synthesizer or EvidenceSummarySynthesizer()
        #: Turns the question into an intent with one model call; None means the recorded labels (tests, evals).
        self.classifier = classifier
        #: User memory (feedback, preferences); None means none is kept.
        self.memory = memory
        self.session = session
        self.config = config or AgentV2Config()
        self.executor = ExecutionEngine(self.registry, max_parallel=self.config.max_parallel)

    @staticmethod
    def _emit(sink: ProgressSink | None, run_id: str, status: RunStatus, message: str) -> None:
        if not sink:
            return
        try:
            sink(ProgressEvent(run_id, status, message))
        except Exception:  # noqa: BLE001 — progress reporting never breaks a run
            pass

    def run(
        self,
        text: str,
        *,
        session_id: str = "",
        allow_web: bool = False,
        on_progress: ProgressSink | None = None,
        cancel_event: Any = None,
    ) -> AgentResult:
        started = time.time()
        run_id = f"agent-v2-{uuid.uuid4().hex[:12]}"
        # Every provider call made for this question carries the run id in
        # the usage ledger, so token use can be read back per question.
        from v2.usage_context import usage_run

        self._cancel_event = cancel_event
        with usage_run(run_id):
            result = self._run(run_id, text, session_id=session_id, allow_web=allow_web, on_progress=on_progress, started=started)
        if self.memory is not None and session_id:
            try:
                self.memory.remember_answer(session_id, question=text, answer=result.answer, run_id=run_id)
            except Exception:  # noqa: BLE001 — memory never breaks an answer
                pass
        return result

    def _run(self, run_id: str, text: str, *, session_id: str, allow_web: bool, on_progress: ProgressSink | None, started: float) -> AgentResult:
        # A pending write is resolved before anything else: "确认" executes it,
        # "取消" drops it, and any other message drops it and proceeds normally.
        pending = self.session.pop_pending(session_id) if self.session is not None and session_id else None
        if pending is not None:
            if _CONFIRM.match(text or ""):
                return self._execute_confirmed(run_id, pending, session_id, on_progress, started)
            if _CANCEL.match(text or ""):
                request = normalize_request(text, session_id=session_id)
                decision = RouteDecision(RouteKind.COMMAND, ("command",), "pending mutation cancelled")
                return self._result(run_id, request, decision, pending, RunStatus.CANCELLED, "已取消，未执行任何修改。", AnswerMode.TOOL_GROUNDED, started)
        elif _CANCEL.match(text or ""):
            # A bare "取消" with nothing pending is not a request to remove
            # an alert; the surface cancels a running answer before this.
            asked = self.session.pop_clarification(session_id) if self.session is not None and session_id else None
            request = normalize_request(text, session_id=session_id)
            decision = RouteDecision(RouteKind.COMMAND, ("command",), "clarification dropped" if asked else "nothing to cancel")
            plan = ExecutionPlan(objective=text, route=RouteKind.COMMAND, direct_answer="已取消，那个问题不再处理。" if asked else "当前没有待确认的操作，也没有正在处理的问题。")
            return self._result(run_id, request, decision, plan, RunStatus.CANCELLED if asked else RunStatus.COMPLETED, plan.direct_answer, AnswerMode.TOOL_GROUNDED, started)

        resolved_text = text
        resolution_metadata = {}
        clarified = False
        if self.session is not None and session_id:
            asked = self.session.pop_clarification(session_id)
            if asked is not None and not _CANCEL.match(text or ""):
                # The message answers the question we asked: read it together with
                # the original question, and do not ask again on this turn.
                original_text, question = asked
                text = f"{original_text}（补充：{text}）"
                resolution_metadata = {"clarified": True, "clarification": question}
                clarified = True
            recent_turns = getattr(self.session, "recent_turns", lambda *_: [])(session_id, 3)
            if recent_turns:
                resolution_metadata = {**resolution_metadata, "recent_turns": recent_turns}
            resolution = self.session.resolve(session_id, text)
            resolved_text = resolution.text
            if resolution.rewritten:
                resolution_metadata = {
                    **resolution_metadata,
                    "rewritten": True,
                    "antecedent": resolution.antecedent,
                    "resolution_note": resolution.note,
                }
                if resolution.frame:
                    resolution_metadata["context_frame"] = dict(resolution.frame)
        if self.memory is not None and session_id:
            try:
                preferences = self.memory.preferences(session_id)
            except Exception:  # noqa: BLE001
                preferences = []
            if preferences:
                resolution_metadata = {**resolution_metadata, "preferences": list(preferences)}
        request = normalize_request(
            resolved_text,
            session_id=session_id,
            allow_web=allow_web and self.config.enable_web_fallback,
            metadata=resolution_metadata,
        )
        from v2.agent_v2.intent import classify_and_time

        intent, classified_ms = classify_and_time(self.classifier, request)
        decision = route(request, intent=intent)
        self._emit(on_progress, run_id, RunStatus.ROUTED, decision.reason)
        if not clarified and self._should_clarify(intent):
            # The classifier is unsure and named the gap: one question now beats a
            # confident answer to the wrong reading. The next message is merged
            # with this one and never asked about again.
            plan = ExecutionPlan(objective=request.text, route=decision.kind, budget=BudgetClass.DIRECT, direct_answer=intent.clarification, assumptions=(f"clarification: confidence {intent.confidence}",))
            self._record_intent(run_id, request, intent, plan, classified_ms)
            return self._ask(run_id, request, decision, plan, intent.clarification, started, asked_about=text)
        previous = getattr(self.session, "previous_turn", lambda *_: None)(session_id) if self.session is not None and session_id else None
        if intent.refers_back and intent.source == "model" and previous and (previous.get("evidence") or previous.get("results")):
            # "第二点展开讲": the answer is written from the previous turn's evidence, no new calls.
            plan = ExecutionPlan(objective=request.text, route=decision.kind, budget=BudgetClass.DIRECT, answer_mode=AnswerMode.RESEARCH_GROUNDED, assumptions=("follow_up: 针对上一条回答的追问，用上一轮的证据回答；上一条回答在 previous_answer 里。",))
            self._record_intent(run_id, request, intent, plan, classified_ms)
            return self._answer_from_previous(run_id, request, decision, plan, previous, on_progress, started)
        plan = self.planner.plan(request, decision)
        self._emit(on_progress, run_id, RunStatus.PLANNED, f"planned {len(plan.tasks)} task(s)")
        self._record_intent(run_id, request, intent, plan, classified_ms)

        if plan.requires_confirmation and not self.config.allow_mutations:
            return self._await_confirmation(run_id, request, decision, plan, started)
        if plan.direct_answer and not plan.tasks:
            # A capability overview is a complete answer; a clarification request is not.
            complete = plan.answer_mode == AnswerMode.GENERAL_KNOWLEDGE
            if not complete and not clarified and decision.kind == RouteKind.COMMAND:
                # "设置提醒需要方向和目标价": the next message ("跌到 150") completes the command.
                return self._ask(run_id, request, decision, plan, plan.direct_answer, started, asked_about=text)
            return self._result(run_id, request, decision, plan, RunStatus.COMPLETED if complete else RunStatus.PARTIAL, plan.direct_answer, AnswerMode.GENERAL_KNOWLEDGE if complete else AnswerMode.INSUFFICIENT_EVIDENCE, started)

        context = self._context(run_id, request, plan, on_progress, started)
        return self._execute(run_id, request, decision, plan, context, on_progress, started)

    # -- debate revision ----------------------------------------------------------

    #: Seconds the revision needs; below this the objections are shown, not applied.
    REVISION_MIN_SECONDS = 15.0

    def _revise(self, request, plan, results, evidence, answer: str, debate: ToolEnvelope, context: ExecutionContext) -> str | None:
        """One rewrite from the debater's objections, when there are any, a model can do it and time remains."""

        objections = list(debate.metadata.get("objections") or [])
        revise = getattr(self.synthesizer, "revise", None)
        if not objections or not callable(revise) or not self.config.debate_revision:
            return None
        if context.remaining_seconds() < self.REVISION_MIN_SECONDS and getattr(context, "deadline", None) is not None:
            self._revision_note = "剩余时间不足"
            return None
        self._revision_note = ""
        try:
            revised = revise(request, plan, results, evidence, answer, objections)
        except Exception as exc:  # noqa: BLE001 — a failed revision leaves the answer as it was
            logger.warning("debate revision failed: %s: %s", type(exc).__name__, exc)
            self._revision_note = f"修订出错（{type(exc).__name__}）"
            return None
        if revised is None:
            self._revision_note = "修订稿未通过校验"
        return revised

    def _revision_reason(self) -> str:
        return str(getattr(self, "_revision_note", "") or "")

    # -- intent ledger ----------------------------------------------------------

    def _record_intent(self, run_id: str, request: NormalizedRequest, intent, plan: ExecutionPlan, elapsed_ms: int) -> None:
        """Append the decision (intent, source, plan) to the ledger; never touches the run."""

        if not self.config.record_intents:
            return
        from v2.agent_v2.intent import decision_row, record_decision

        try:
            record_decision(decision_row(request, intent, plan, run_id=run_id, elapsed_ms=elapsed_ms, channel=str(request.metadata.get("channel") or "")))
        except Exception as exc:  # noqa: BLE001 — ledger work must never surface
            logger.warning("intent ledger failed: %s: %s", type(exc).__name__, exc)

    # -- confirmation ---------------------------------------------------------

    @staticmethod
    def _pending_mutation(plan: ExecutionPlan) -> PendingMutation | None:
        task = next((task for task in plan.tasks if task.capability == "state.mutate"), None)
        if task is None:
            return None
        return PendingMutation(
            operation=str(task.arguments.get("operation") or ""),
            payload=dict(task.arguments.get("payload") or {}),
            description=task.purpose or task.capability,
        )

    def _answer_from_previous(self, run_id, request, decision, plan, previous: dict, on_progress, started) -> AgentResult:
        """Answer a follow-up about the previous answer from that turn's evidence; verified like any answer."""

        results = list(previous.get("results") or [])
        evidence = list(previous.get("evidence") or [])
        request.metadata["previous_answer"] = str(previous.get("answer") or "")[:4000]
        request.metadata["previous_question"] = str(previous.get("question") or "")[:300]
        self._emit(on_progress, run_id, RunStatus.SYNTHESIZING, "answering from the previous turn's evidence")
        answer = self.synthesizer.synthesize(request, plan, results, evidence)
        synthesis = {**self._synthesis_diagnostics(), "follow_up": {"previous_run_id": previous.get("run_id"), "evidence": len(evidence)}}
        self._emit(on_progress, run_id, RunStatus.VERIFYING, "verifying citations")
        verification = verify_answer(answer, evidence, answer_mode=plan.answer_mode, results=results, judge=getattr(self.synthesizer, "judge", None))
        status = RunStatus.COMPLETED if verification.ok else RunStatus.PARTIAL
        return self._result(run_id, request, decision, plan, status, answer, plan.answer_mode, started, results=results, evidence=evidence, verification=verification, synthesis=synthesis)

    def _should_clarify(self, intent) -> bool:
        """A model classification below the threshold that carries a question, on a question that could go several ways."""

        threshold = float(self.config.clarify_below or 0)
        if threshold <= 0 or not intent.clarification or intent.source != "model" or intent.confidence is None:
            return False
        if intent.kind in {"knowledge", "help"}:
            return False
        return intent.confidence < threshold

    def _ask(self, run_id, request, decision, plan, question: str, started, *, asked_about: str) -> AgentResult:
        """Return the question and remember it, so the next message on the session is read as its answer.

        ``asked_about`` is the user's own wording (before the session's
        rewrite), so the merged turn reads as they would have typed it.
        """

        if self.session is not None and request.session_id:
            self.session.set_clarification(request.session_id, asked_about, question)
            answer = question
        else:
            answer = f"{question}（该渠道没有会话，请把补充的信息和问题一起再发一次。）"
        return self._result(run_id, request, decision, plan, RunStatus.WAITING_CLARIFICATION, answer, AnswerMode.INSUFFICIENT_EVIDENCE, started)

    def _await_confirmation(self, run_id, request, decision, plan, started) -> AgentResult:
        mutation = self._pending_mutation(plan)
        if mutation is None:
            return self._result(run_id, request, decision, plan, RunStatus.PARTIAL, self.synthesizer.synthesize(request, plan, [], []), AnswerMode.INSUFFICIENT_EVIDENCE, started)
        if self.session is not None and request.session_id:
            self.session.set_pending(request.session_id, plan)
            answer = f"将执行写操作：{mutation.description}。回复「确认」执行，回复「取消」放弃；当前没有执行任何修改。"
        else:
            answer = f"将执行写操作：{mutation.description}。该渠道没有会话，无法接收确认；当前没有执行任何修改。"
        return self._result(run_id, request, decision, plan, RunStatus.WAITING_CONFIRMATION, answer, AnswerMode.TOOL_GROUNDED, started, pending_mutation=mutation)

    def _execute_confirmed(self, run_id, plan: ExecutionPlan, session_id: str, on_progress, started) -> AgentResult:
        request = normalize_request(plan.objective, session_id=session_id, metadata={"confirmed_mutation": True})
        decision = RouteDecision(RouteKind.COMMAND, ("command",), "user confirmed a pending mutation")
        context = self._context(run_id, request, plan, on_progress, started, allow_mutations=True)
        return self._execute(run_id, request, decision, plan, context, on_progress, started)

    # -- execution ------------------------------------------------------------

    def _context(self, run_id, request, plan, on_progress, started, *, allow_mutations: bool | None = None) -> ExecutionContext:
        limit = self.config.max_seconds if self.config.max_seconds is not None else time_limit(plan.budget)
        elapsed = time.time() - started
        return ExecutionContext(
            run_id=run_id,
            request=request,
            budget=plan.budget,
            cancel_event=getattr(self, "_cancel_event", None),
            allow_mutations=self.config.allow_mutations if allow_mutations is None else allow_mutations,
            allow_web=request.allow_web,
            on_progress=on_progress,
            deadline=time.monotonic() + max(0.0, limit - elapsed),
        )

    def _execute(self, run_id, request, decision, plan, context: ExecutionContext, on_progress, started) -> AgentResult:
        self._emit(on_progress, run_id, RunStatus.EXECUTING, "executing capability plan")
        try:
            outcome = self.executor.run(plan, context)
        except Exception as exc:
            if isinstance(exc, EvidenceConflictError):
                answer = "研究结果包含冲突的证据标识，任务已安全停止。"
                warning = "证据完整性检查失败"
            elif isinstance(exc, PlanValidationError):
                answer = "执行计划无效，任务没有完成。"
                warning = "执行计划校验失败"
            else:
                answer = "工具执行发生未预期错误，任务没有完成。"
                warning = "工具执行失败"
            return self._result(
                run_id,
                request,
                decision,
                plan,
                RunStatus.FAILED,
                answer,
                AnswerMode.INSUFFICIENT_EVIDENCE,
                started,
                error=f"{type(exc).__name__}: {exc}",
                verification=VerificationReport(ok=False, warnings=(warning,)),
            )

        plan, outcome = self._web_fallback(request, decision, plan, outcome, context)
        results = outcome.results
        from v2.agent_v2.agents.move_attributor import mark_read_filings

        mark_read_filings(results)

        self._emit(on_progress, run_id, RunStatus.SYNTHESIZING, "synthesizing evidence")
        evidence = outcome.ledger.items()
        # A confirmed write has one result, the store's own message; a model
        # draft read the plan's "not executed until confirmed" note as the
        # present state and told the user to confirm again.
        synthesizer = EvidenceSummarySynthesizer() if any(task.capability == "state.mutate" for task in plan.tasks) else self.synthesizer
        answer = synthesizer.synthesize(request, plan, results, evidence)
        synthesis = self._synthesis_diagnostics()
        answer_mode = plan.answer_mode
        if not plan.tasks and decision.kind != RouteKind.GENERAL_KNOWLEDGE:
            answer_mode = AnswerMode.INSUFFICIENT_EVIDENCE
        self._emit(on_progress, run_id, RunStatus.VERIFYING, "verifying citations")
        verification = verify_answer(answer, evidence, answer_mode=answer_mode, results=results, judge=getattr(self.synthesizer, "judge", None))
        failures = [result for result in results if not result.ok]
        debate = self._debate(request, decision, plan, answer, evidence, results, context)
        if debate is not None:
            results = [*results, debate]
            revised = self._revise(request, plan, results, evidence, answer, debate, context) if self.config.debate_revision else None
            if revised is not None:
                answer = revised
                verification = verify_answer(answer, evidence, answer_mode=answer_mode, results=results, judge=getattr(self.synthesizer, "judge", None))
                synthesis = {**self._synthesis_diagnostics(), "debate_revision": {"applied": True, "objections": len(debate.metadata.get("objections") or [])}}
            elif self.config.debate_revision and debate.metadata.get("objections"):
                synthesis = {**synthesis, "debate_revision": {"applied": False, "objections": len(debate.metadata.get("objections") or []), "reason": self._revision_reason()}}
        knowledge_unavailable = not results and decision.kind == RouteKind.GENERAL_KNOWLEDGE and not bool(getattr(self.synthesizer, "supports_general_knowledge", False))
        if outcome.stop_reason == "cancelled":
            status = RunStatus.CANCELLED
        elif failures or not verification.ok or knowledge_unavailable or outcome.stop_reason != "completed" or (not results and decision.kind != RouteKind.GENERAL_KNOWLEDGE):
            status = RunStatus.PARTIAL
        else:
            status = RunStatus.COMPLETED
        return self._result(
            run_id,
            request,
            decision,
            plan,
            status,
            answer,
            answer_mode,
            started,
            results=results,
            evidence=evidence,
            verification=verification,
            stop_reason=outcome.stop_reason,
            synthesis=synthesis,
        )

    def _debate(self, request, decision, plan, answer: str, evidence, results, context: ExecutionContext):
        """One adversarial pass over a research answer, when configured, a model is present and time remains.

        Returns the debater's display envelope (no evidence of its own) or None.
        """

        if not self.config.debate or decision.kind != RouteKind.RESEARCH or plan.answer_mode != AnswerMode.RESEARCH_GROUNDED:
            return None
        if not any(result.ok and (result.capability.startswith("research.") or result.capability in {"market.performance", "market.explain_move", "market.attribute_move", "web.research"}) for result in results):
            return None
        llm = getattr(self.synthesizer, "llm", None)
        if llm is None or not evidence:
            return None
        from v2.agent_v2.agents.debater import DEBATE_MIN_SECONDS, Debater

        if context.remaining_seconds() < DEBATE_MIN_SECONDS:
            return None
        try:
            return Debater(llm).run(request.text, answer, evidence, context, subject=request.entities[0] if request.entities else "")
        except Exception:  # noqa: BLE001 — the debate never breaks an answer
            return None

    def _synthesis_diagnostics(self) -> dict:
        diagnostics = getattr(self.synthesizer, "diagnostics", None)
        if not callable(diagnostics):
            return {}
        try:
            return dict(diagnostics())
        except Exception:  # noqa: BLE001 — diagnostics never break an answer
            return {}

    def _web_fallback(self, request, decision, plan, outcome: ExecutionOutcome, context: ExecutionContext):
        """Make one bounded web attempt only after internal evidence is absent or failed."""

        eligible = plan.web_fallback_allowed and request.allow_web and decision.kind in {RouteKind.FAST_LOOKUP, RouteKind.RESEARCH} and self.registry.registered("web.research")
        existing_evidence = outcome.ledger.items()
        internal_failed = any(not result.ok for result in outcome.results)
        if not eligible or (existing_evidence and not internal_failed):
            return plan, outcome
        notes = ["Web fallback ran because internal evidence was missing or failed."]
        if context.remaining_seconds() < WEB_FALLBACK_GRACE_SECONDS:
            # The internal step used the budget (a timeout, typically); the
            # one web attempt gets its own grace rather than nothing, so the
            # run still ends with an answer instead of "未完成".
            object.__setattr__(context, "deadline", time.monotonic() + WEB_FALLBACK_GRACE_SECONDS)
            notes.append(f"Web fallback ran with a {WEB_FALLBACK_GRACE_SECONDS:.0f} s grace after internal capabilities used the budget.")
        topic = "company_event" if request.entities else "financial_research"
        task = PlanTask(
            id="web-fallback",
            capability="web.research",
            arguments={
                "query": request.text[:500],
                "topic": topic,
                "ticker": request.entities[0] if request.entities else "",
                # A comparison question names several stocks; the checker looks at all of them.
                **({"tickers": list(request.entities[:4])} if len(request.entities) > 1 else {}),
                "recency_days": 30,
            },
            required=False,
            purpose="fill an evidence gap left by internal capabilities",
        )
        context.emit(task.purpose, task_id=task.id, capability=task.capability)
        result = self.registry.execute(task, context)
        outcome.results.append(result)
        outcome.ledger.ingest(result)
        mode = AnswerMode.MIXED if existing_evidence else AnswerMode.WEB_GROUNDED
        plan = replace(
            plan,
            tasks=(*plan.tasks, task),
            answer_mode=mode,
            assumptions=(*plan.assumptions, *notes),
        )
        return plan, outcome

    def _result(
        self,
        run_id,
        request: NormalizedRequest,
        decision: RouteDecision,
        plan: ExecutionPlan,
        status: RunStatus,
        answer: str,
        answer_mode: AnswerMode,
        started: float,
        *,
        results=None,
        evidence=None,
        verification=None,
        error: str = "",
        stop_reason: str = "",
        pending_mutation: PendingMutation | None = None,
        synthesis: dict | None = None,
    ) -> AgentResult:
        result = AgentResult(
            run_id=run_id,
            request=request,
            route=decision,
            plan=plan,
            status=status,
            answer=answer,
            answer_mode=answer_mode,
            results=list(results or []),
            evidence=list(evidence or []),
            verification=verification or VerificationReport(),
            elapsed_ms=int((time.time() - started) * 1000),
            error=error,
            stop_reason=stop_reason,
            pending_mutation=pending_mutation,
            synthesis=dict(synthesis or {}),
        )
        if self.session is not None:
            self.session.record(result)
        if result.results and self.config.record_sub_agents:
            from v2.agent_v2.eval.subagent_ledger import record_runs

            record_runs(result)
        if result.results and self.config.record_capabilities:
            from v2.agent_v2.eval.capability_ledger import record_capabilities

            record_capabilities(result)
        return result
