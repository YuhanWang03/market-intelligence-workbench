"""Capability-level scoring for one Agent V2 result."""

from __future__ import annotations

from dataclasses import dataclass

from v2.agent_v2.eval.answer_cases import AnswerCase
from v2.agent_v2.eval.cases import EvalCase
from v2.agent_v2.models import AgentResult
from v2.agent_v2.verification import verify_answer


@dataclass(frozen=True)
class CaseScore:
    case_id: str
    passed: bool
    route_ok: bool
    capability_recall: float
    discipline_ok: bool
    status_ok: bool
    answer_mode_ok: bool
    evidence_ok: bool
    called: tuple[str, ...]


def score_case(case: EvalCase, result: AgentResult) -> CaseScore:
    called = tuple(item.capability for item in result.results)
    required = set(case.required_capabilities)
    acquired = required.intersection(called)
    recall = len(acquired) / len(required) if required else 1.0
    route_ok = result.route.kind == case.expected_route
    discipline_ok = not set(case.forbidden_capabilities).intersection(called)
    status_ok = result.status in case.expected_statuses
    answer_mode_ok = case.expected_answer_mode is None or result.answer_mode == case.expected_answer_mode
    evidence_ok = result.verification.ok and (not called or bool(result.evidence) or result.status in {status for status in case.expected_statuses if status.value in {"queued", "waiting_confirmation"}})
    passed = route_ok and recall == 1.0 and discipline_ok and status_ok and answer_mode_ok and evidence_ok
    return CaseScore(case.id, passed, route_ok, recall, discipline_ok, status_ok, answer_mode_ok, evidence_ok, called)


@dataclass(frozen=True)
class AnswerScore:
    case_id: str
    passed: bool
    verifier_ok: bool
    expected_ok: bool
    warnings: tuple[str, ...]


def score_answer_case(case: AnswerCase) -> AnswerScore:
    envelope = case.envelope()
    answer = case.answer(envelope)
    report = verify_answer(answer, envelope.evidence, answer_mode=case.answer_mode, results=[envelope], judge=case.judge)
    warnings = tuple(report.warnings) + tuple(f"ungrounded: {value}" for value in report.ungrounded_numbers) + tuple(f"unknown citation: {value}" for value in report.unknown_citations)
    passed = report.ok == case.expect_ok
    if passed and case.expected_warning:
        # A rejected answer must fail for the expected reason; an accepted one may still carry a soft warning it must report.
        passed = any(case.expected_warning in warning for warning in warnings)
    return AnswerScore(case.id, passed, report.ok, case.expect_ok, warnings)
