"""Offline suite runner for the Agent V2 seed evaluation set."""

from __future__ import annotations

from dataclasses import dataclass

from v2.agent_v2.eval.answer_cases import ANSWER_CASES, AnswerCase
from v2.agent_v2.eval.cases import CASES, EvalCase
from v2.agent_v2.eval.fixtures import build_eval_registry, EvalSynthesizer
from v2.agent_v2.eval.scenario_cases import SCENARIO_CASES, ScenarioCase, ScenarioScore, score_scenario
from v2.agent_v2.eval.scoring import AnswerScore, CaseScore, score_answer_case, score_case
from v2.agent_v2.orchestrator import AgentV2, AgentV2Config


@dataclass(frozen=True)
class SuiteReport:
    scores: tuple[CaseScore, ...]
    answer_scores: tuple[AnswerScore, ...] = ()
    scenario_scores: tuple[ScenarioScore, ...] = ()

    @property
    def passed(self) -> int:
        return sum(score.passed for score in self.scores) + sum(score.passed for score in self.answer_scores) + sum(score.passed for score in self.scenario_scores)

    @property
    def total(self) -> int:
        return len(self.scores) + len(self.answer_scores) + len(self.scenario_scores)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0


def run_suite(cases: tuple[EvalCase, ...] = CASES, answer_cases: tuple[AnswerCase, ...] = ANSWER_CASES, scenario_cases: tuple[ScenarioCase, ...] = SCENARIO_CASES) -> SuiteReport:
    registry = build_eval_registry()
    # An offline eval writes no production ledgers (the gate runs it on the server).
    agent = AgentV2(catalog=registry.catalog, registry=registry, synthesizer=EvalSynthesizer(), config=AgentV2Config(record_sub_agents=False, record_capabilities=False))
    scores = tuple(score_case(case, agent.run(case.query)) for case in cases)
    answer_scores = tuple(score_answer_case(case) for case in answer_cases)
    scenario_scores = tuple(score_scenario(case) for case in scenario_cases)
    return SuiteReport(scores, answer_scores, scenario_scores)


def render(report: SuiteReport) -> str:
    lines = [f"Agent V2 offline eval: {report.passed}/{report.total} ({report.pass_rate:.0%})"]
    for score in report.scores:
        mark = "PASS" if score.passed else "FAIL"
        lines.append(f"{mark} {score.case_id}: route={score.route_ok} recall={score.capability_recall:.0%} discipline={score.discipline_ok} status={score.status_ok} evidence={score.evidence_ok}")
    for score in report.answer_scores:
        mark = "PASS" if score.passed else "FAIL"
        lines.append(f"{mark} {score.case_id}: verdict={'ok' if score.verifier_ok else 'rejected'} expected={'ok' if score.expected_ok else 'rejected'} warnings={'; '.join(score.warnings) or '-'}")
    for score in report.scenario_scores:
        mark = "PASS" if score.passed else "FAIL"
        lines.append(f"{mark} {score.case_id}: plan={score.capabilities_ok} discipline={score.discipline_ok} phrases={score.phrases_ok} rewritten={score.rewritten_ok} verified={score.verified_ok} agents={score.agents_ok} called={','.join(score.called)}" + (f" sub_agents={'; '.join(score.agents)}" if score.agents else "") + (f" missing={' | '.join(score.missing_phrases)}" if score.missing_phrases else ""))
    return "\n".join(lines)
