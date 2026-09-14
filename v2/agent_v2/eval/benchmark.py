"""Run the preserved historical question set against Agent V2.

Two modes share one question set and one answer key:

``v2_rules``
    V2 with the rule planner and the deterministic synthesizer.  No model, so
    it measures routing, entity extraction and capability choice; its fact
    recall is a floor because the synthesizer only echoes evidence.
``v2_llm``
    V2 as deployed: LLM planner and LLM synthesizer behind the verifier.
    Needs a model; ``llm_factory`` can inject a scripted client for tests.

Every mode runs on recorded observations, so a score depends only on the code
under test and, for ``v2_llm``, the model.

Two fixture layers feed the V2 modes.  ``v1`` serves V1's cards, so V1's fact
keys gate the pass.  ``engine`` serves engine-shaped research and market
envelopes (live recordings when present, offline synthesis otherwise); V1's
fact keys do not describe those numbers, so in that layer a case passes on
tool recall, verification and error-freedom, and fact recall is reported
without gating.
"""

from __future__ import annotations

import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable

from v2.agent_eval.scoring import fact_present, normalise
from v2.agent_v2.eval.benchmark_cases import DEV_CASES, HOLDOUT_CASES, BenchmarkCase, by_category
from v2.agent_v2.eval.benchmark_fixtures import FIXTURE_MODES, RecordedCalls, build_benchmark_registry
from v2.agent_v2.llm import LLMEvidenceSynthesizer, StructuredLLMPlanner
from v2.agent_v2.orchestrator import AgentV2, AgentV2Config
from v2.agent_v2.synthesis import EvidenceSummarySynthesizer

MODES = ("v2_rules", "v2_llm")


@dataclass
class CountingLLM:
    """Wrap any LLM client to count calls and tokens for the cost columns."""

    inner: Any
    calls: int = 0
    tokens: int = 0

    def complete(self, messages, tools=None):
        self.calls += 1
        response = self.inner.complete(messages, tools)
        self.tokens += int(getattr(response, "prompt_tokens", 0) or 0) + int(getattr(response, "completion_tokens", 0) or 0)
        return response


@dataclass(frozen=True)
class BenchmarkScore:
    case_id: str
    category: str
    mode: str
    holdout: bool
    tool_recall: float
    fact_recall: float
    missing_tools: tuple[str, ...]
    missing_facts: tuple[str, ...]
    forbidden_hit: tuple[str, ...]
    waste: tuple[str, ...]
    unmapped_tools: tuple[str, ...]
    grounded: bool
    verify_outcome: str
    status: str
    route: str
    tool_calls: int
    llm_calls: int
    tokens: int
    elapsed_ms: int
    stop_reason: str
    error: str
    answer: str
    called: tuple[str, ...]
    #: Whether the fixtures behind this run are the ones the fact keys describe.
    keyed: bool = True
    fixtures: str = "v1"
    #: v2_llm only: did the *first draft* satisfy the answer key?  None when no draft exists.
    draft_keys_ok: bool | None = None
    draft: str = ""

    @property
    def answer_correct(self) -> bool:
        facts_ok = self.fact_recall == 1.0 if self.keyed else True
        return self.tool_recall == 1.0 and facts_ok and not self.forbidden_hit and not self.error

    @property
    def passed(self) -> bool:
        return self.answer_correct and self.grounded

    @property
    def capability_gap(self) -> bool:
        return bool(self.unmapped_tools)

    @property
    def checker_false_positive(self) -> bool:
        """The verifier rejected an answer whose every keyed assertion passes.

        V1's lesson: a case's own facts and forbidden lists already decide
        whether the answer is right, so a rejection on top of a fully keyed
        pass is a rejection of a correct answer.  Only meaningful with keys.
        """

        return self.keyed and self.answer_correct and not self.grounded

    @property
    def repair_regressed(self) -> bool:
        """The draft satisfied the key and the shipped answer does not."""

        return self.keyed and self.draft_keys_ok is True and self.fact_recall < 1.0

    def failure_reason(self) -> str:
        if self.error:
            return f"运行错误：{self.error[:80]}"
        if self.missing_tools:
            return "缺能力：" + ", ".join(self.missing_tools)
        if self.missing_facts and self.keyed:
            return "缺事实：" + ", ".join(self.missing_facts[:3])
        if self.forbidden_hit:
            return "错误归属：" + ", ".join(self.forbidden_hit[:2])
        if not self.grounded:
            return f"校验未通过（{self.verify_outcome}）" + ("；答案键全部通过，疑似校验器误报" if self.checker_false_positive else "")
        return ""


@dataclass
class ModeReport:
    mode: str
    scores: list[BenchmarkScore] = field(default_factory=list)
    repeat: int = 1

    @property
    def total(self) -> int:
        return len(self.scores)

    @property
    def passed(self) -> int:
        return sum(1 for score in self.scores if score.passed)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    def mean(self, attribute: str) -> float:
        values = [float(getattr(score, attribute)) for score in self.scores]
        return sum(values) / len(values) if values else 0.0

    def rate(self, predicate: Callable[[BenchmarkScore], bool]) -> float:
        return sum(1 for score in self.scores if predicate(score)) / self.total if self.total else 0.0

    def verify_outcomes(self) -> Counter:
        return Counter(score.verify_outcome for score in self.scores if score.verify_outcome)

    def stability(self) -> dict[str, tuple[int, int]]:
        out: dict[str, tuple[int, int]] = {}
        for score in self.scores:
            passed, total = out.get(score.case_id, (0, 0))
            out[score.case_id] = (passed + int(score.passed), total + 1)
        return out

    def stable_failures(self) -> list[str]:
        return sorted(case_id for case_id, (passed, _) in self.stability().items() if passed == 0)

    def flaky(self) -> list[str]:
        return sorted(case_id for case_id, (passed, total) in self.stability().items() if 0 < passed < total)

    def checker_false_positives(self) -> list[BenchmarkScore]:
        return [score for score in self.scores if score.checker_false_positive]

    def repair_regressions(self) -> list[BenchmarkScore]:
        return [score for score in self.scores if score.repair_regressed]

    def by_category(self) -> dict[str, tuple[int, int]]:
        out: dict[str, tuple[int, int]] = {}
        for score in self.scores:
            passed, total = out.get(score.category, (0, 0))
            out[score.category] = (passed + int(score.passed), total + 1)
        return out


def _keys_ok(case: BenchmarkCase, text: str) -> bool:
    required = tuple(case.facts) + tuple(case.behaviors)
    haystack = normalise(text)
    return all(fact_present(forms, text) for forms in required) and not any(normalise(value) in haystack for value in case.forbidden)


def _score(case: BenchmarkCase, *, mode: str, answer: str, called: Iterable[str], grounded: bool, verify_outcome: str = "", status: str = "", route: str = "", tool_calls: int = 0, llm_calls: int = 0, tokens: int = 0, elapsed_ms: int = 0, stop_reason: str = "", error: str = "", fixtures: str = "v1", draft: str = "") -> BenchmarkScore:
    called_set = set(called)
    if "research.compare" in called_set:
        called_set.add("research.stock")  # compare is V2's own way of researching several tickers
    missing_tools = tuple(name for name in case.must_call if name not in called_set)
    denominator = len(case.must_call) + len(case.unmapped_tools)
    tool_recall = 1.0 if not denominator else 1.0 - (len(missing_tools) + len(case.unmapped_tools)) / denominator
    required = tuple(case.facts) + tuple(case.behaviors)
    missing_facts = tuple(forms[0] for forms in required if not fact_present(forms, answer))
    fact_recall = 1.0 if not required else 1.0 - len(missing_facts) / len(required)
    haystack = normalise(answer)
    forbidden_hit = tuple(value for value in case.forbidden if normalise(value) in haystack)
    return BenchmarkScore(
        case_id=case.id,
        category=case.category,
        mode=mode,
        holdout=case.holdout,
        tool_recall=tool_recall,
        fact_recall=fact_recall,
        missing_tools=missing_tools,
        missing_facts=missing_facts,
        forbidden_hit=forbidden_hit,
        waste=tuple(sorted(called_set & set(case.wasteful))),
        unmapped_tools=case.unmapped_tools,
        grounded=grounded,
        verify_outcome=verify_outcome,
        status=status,
        route=route,
        tool_calls=tool_calls,
        llm_calls=llm_calls,
        tokens=tokens,
        elapsed_ms=elapsed_ms,
        stop_reason=stop_reason,
        error=error,
        answer=answer,
        called=tuple(called),
        keyed=fixtures == "v1",
        fixtures=fixtures,
        draft_keys_ok=_keys_ok(case, draft) if draft else None,
        draft=draft,
    )


# -- modes ------------------------------------------------------------------


def _v2_agent(mode: str, llm_factory: Callable[[], Any] | None, fixtures: str) -> tuple[AgentV2, RecordedCalls, CountingLLM | None, LLMEvidenceSynthesizer | None]:
    registry, calls = build_benchmark_registry(fixtures=fixtures)
    if mode == "v2_rules":
        return AgentV2(catalog=registry.catalog, registry=registry, synthesizer=EvidenceSummarySynthesizer(), config=AgentV2Config(max_seconds=60, record_sub_agents=False, record_capabilities=False)), calls, None, None
    if llm_factory is None:
        from v2.agent_common.llm import build_llm

        llm_factory = build_llm
    llm = CountingLLM(llm_factory())
    synthesizer = LLMEvidenceSynthesizer(llm, catalog=registry.catalog)
    # The benchmark counts the model calls a synthesis needs; the debate is a separate pass, measured by the sub-agent ledger.
    agent = AgentV2(catalog=registry.catalog, registry=registry, planner=StructuredLLMPlanner(llm, registry.catalog), synthesizer=synthesizer, config=AgentV2Config(max_seconds=120, debate=False, record_sub_agents=False, record_capabilities=False))
    return agent, calls, llm, synthesizer


def run_v2(case: BenchmarkCase, *, mode: str, llm_factory: Callable[[], Any] | None = None, fixtures: str = "v1") -> BenchmarkScore:
    agent, calls, llm, synthesizer = _v2_agent(mode, llm_factory, fixtures)
    started = time.time()
    try:
        result = agent.run(case.query)
    except Exception as exc:  # noqa: BLE001 — a crash is a scored failure
        return _score(case, mode=mode, answer="", called=calls.names(), grounded=False, error=f"{type(exc).__name__}: {exc}", elapsed_ms=int((time.time() - started) * 1000), fixtures=fixtures)
    outcome = synthesizer.last_outcome if synthesizer is not None else "deterministic"
    draft = synthesizer.last_draft if synthesizer is not None and synthesizer.last_draft != result.answer else ""
    return _score(
        case,
        mode=mode,
        answer=result.answer,
        called=calls.names(),
        grounded=result.verification.ok,
        verify_outcome=outcome,
        status=result.status.value,
        route=result.route.kind.value,
        tool_calls=len(calls.calls),
        llm_calls=llm.calls if llm else 0,
        tokens=llm.tokens if llm else 0,
        elapsed_ms=result.elapsed_ms,
        stop_reason=result.stop_reason,
        error=result.error,
        fixtures=fixtures,
        draft=draft,
    )


def run_mode(mode: str, cases: tuple[BenchmarkCase, ...], *, repeat: int = 1, llm_factory: Callable[[], Any] | None = None, fixtures: str = "v1", workers: int = 1, on_case: Callable[[BenchmarkScore], None] | None = None) -> ModeReport:
    """Run every case under one mode; ``workers`` parallelises model-bound v2_llm runs."""

    if mode not in MODES:
        raise ValueError(f"unknown mode: {mode}")
    if fixtures not in FIXTURE_MODES:
        raise ValueError(f"unknown fixture mode: {fixtures}")
    if mode != "v2_llm":
        repeat = 1  # deterministic modes cannot flake
        workers = 1
    label = mode if fixtures == "v1" else f"{mode}@{fixtures}"
    report = ModeReport(mode=label, repeat=repeat)
    work = [case for case in cases for _ in range(repeat)]

    def one(case: BenchmarkCase) -> BenchmarkScore:
        score = run_v2(case, mode=mode, llm_factory=llm_factory, fixtures=fixtures)
        if on_case is not None:
            on_case(score)
        return score

    if workers <= 1:
        report.scores = [one(case) for case in work]
    else:
        # Each run builds its own agent, registry and model client, so runs
        # share nothing but the recorded fixtures; order is restored afterwards.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            report.scores = list(pool.map(one, work))
    return report


def progress_printer(total: int, stream=sys.stderr) -> Callable[[BenchmarkScore], None]:
    """An ``on_case`` hook that keeps a long model-in-the-loop run observable."""

    state = {"done": 0, "passed": 0, "started": time.time()}

    def hook(score: BenchmarkScore) -> None:
        state["done"] += 1
        state["passed"] += int(score.passed)
        elapsed = time.time() - state["started"]
        eta = (elapsed / state["done"]) * (total - state["done"]) if state["done"] else 0
        mark = "ok " if score.passed else "FAIL"
        print(f"[{state['done']:4}/{total}] {mark} {score.case_id:5} {score.verify_outcome or '-':13} llm={score.llm_calls} {score.elapsed_ms:6}ms  pass={state['passed']}/{state['done']}  eta={eta / 60:.1f}m", file=stream, flush=True)

    return hook


def run_benchmark(modes: Iterable[str] = ("v2_rules",), *, holdout: bool = False, repeat: int = 1, llm_factory: Callable[[], Any] | None = None, fixtures: str = "v1", workers: int = 1, progress: bool = False) -> list[ModeReport]:
    cases = HOLDOUT_CASES if holdout else DEV_CASES
    reports = []
    for mode in modes:
        hook = progress_printer(len(cases) * (repeat if mode == "v2_llm" else 1)) if progress and mode == "v2_llm" else None
        reports.append(run_mode(mode, cases, repeat=repeat, llm_factory=llm_factory, fixtures=fixtures, workers=workers, on_case=hook))
    return reports


# -- rendering ----------------------------------------------------------------


def _pct(value: float) -> str:
    return f"{value:.0%}"


def render_comparison(reports: list[ModeReport]) -> str:
    rows = [
        ("通过率", lambda r: _pct(r.pass_rate)),
        ("工具召回", lambda r: _pct(r.mean("tool_recall"))),
        ("事实召回", lambda r: _pct(r.mean("fact_recall")) + ("" if all(s.keyed for s in r.scores) else "*")),
        ("错误归属命中", lambda r: str(sum(1 for s in r.scores if s.forbidden_hit))),
        ("校验通过率", lambda r: _pct(r.rate(lambda s: s.grounded))),
        ("能力缺口用例", lambda r: str(sum(1 for s in r.scores if s.capability_gap))),
        ("超预算", lambda r: str(sum(1 for s in r.scores if s.stop_reason == "deadline"))),
        ("工具调用 / 例", lambda r: f"{r.mean('tool_calls'):.1f}"),
        ("LLM 调用 / 例", lambda r: f"{r.mean('llm_calls'):.1f}"),
        ("token / 例", lambda r: f"{r.mean('tokens'):.0f}"),
        ("耗时 ms / 例", lambda r: f"{r.mean('elapsed_ms'):.0f}"),
        ("稳定失败", lambda r: str(len(r.stable_failures()))),
        ("不稳定用例", lambda r: str(len(r.flaky())) if r.repeat > 1 else "-"),
        ("校验器疑似误报", lambda r: str(len(r.checker_false_positives())) if all(s.keyed for s in r.scores) else "-"),
        ("重写丢事实", lambda r: str(len(r.repair_regressions())) if any(s.draft for s in r.scores) else "-"),
    ]
    width = max(14, *(len(r.mode) + 2 for r in reports))
    header = f"{'':14}" + "".join(f"{r.mode:>{width}}" for r in reports)
    lines = [header, "─" * len(header)]
    for label, cell in rows:
        lines.append(f"{label:14}" + "".join(f"{cell(r):>{width}}" for r in reports))
    for report in reports:
        outcomes = report.verify_outcomes()
        if outcomes and set(outcomes) != {"deterministic"}:
            lines.append(f"校验结果 [{report.mode}]: " + ", ".join(f"{k}={v}" for k, v in sorted(outcomes.items())))
    if any(not s.keyed for r in reports for s in r.scores):
        lines.append("* engine fixtures: V1 fact keys are informational, not gating")
    return "\n".join(lines)


def render_categories(reports: list[ModeReport]) -> str:
    categories = list(dict.fromkeys(score.category for report in reports for score in report.scores))
    width = max(14, *(len(r.mode) + 2 for r in reports))
    header = f"{'类别':14}" + "".join(f"{r.mode:>{width}}" for r in reports)
    lines = [header, "─" * len(header)]
    for category in categories:
        cells = []
        for report in reports:
            passed, total = report.by_category().get(category, (0, 0))
            cells.append(f"{passed}/{total}")
        lines.append(f"{category:14}" + "".join(f"{cell:>{width}}" for cell in cells))
    return "\n".join(lines)


def render_failures(report: ModeReport, limit: int = 30) -> str:
    seen: set[str] = set()
    lines = [f"[{report.mode}] 失败用例（最多 {limit} 条）"]
    for score in report.scores:
        if score.passed or score.case_id in seen:
            continue
        seen.add(score.case_id)
        gap = " ⚠能力缺口" if score.capability_gap else ""
        lines.append(f"  {score.case_id:5} {score.category:14} {score.failure_reason()}{gap}  called={list(score.called)}")
        if len(lines) > limit:
            break
    return "\n".join(lines)


def render_stability(report: ModeReport) -> str:
    if report.repeat <= 1:
        return ""
    stability = report.stability()
    lines = [f"[{report.mode}] 稳定性（每例 {report.repeat} 次）：稳定通过 {sum(1 for p, n in stability.values() if p == n)} · 不稳定 {len(report.flaky())} · 稳定失败 {len(report.stable_failures())}"]
    for case_id in report.flaky()[:20]:
        passed, total = stability[case_id]
        lines.append(f"  {case_id:5} {passed}/{total}")
    return "\n".join(lines)


def render_checker(report: ModeReport) -> str:
    rows = report.checker_false_positives()
    regressions = report.repair_regressions()
    if not rows and not regressions:
        return ""
    lines = [f"[{report.mode}] 校验器轴：疑似误报 {len(rows)} · 重写丢事实 {len(regressions)}"]
    seen: set[str] = set()
    for score in rows[:10]:
        if score.case_id in seen:
            continue
        seen.add(score.case_id)
        lines.append(f"  误报? {score.case_id:5} outcome={score.verify_outcome} answer={score.answer[:60]!r}")
    for score in regressions[:10]:
        lines.append(f"  丢事实 {score.case_id:5} 缺：{', '.join(score.missing_facts[:3])}")
    return "\n".join(lines)


def render(reports: list[ModeReport], *, failures: bool = True) -> str:
    parts = [render_comparison(reports), "", render_categories(reports)]
    for report in reports:
        for block in (render_stability(report), render_checker(report)):
            if block:
                parts.extend(["", block])
    if failures:
        parts.extend(["", *(render_failures(report) for report in reports)])
    return "\n".join(parts)


def to_markdown(reports: list[ModeReport], *, title: str, note: str = "") -> str:
    """Render benchmark results as Markdown tables."""

    def cell(value: str) -> str:
        return value.replace("|", "\\|")

    lines = [f"## {title}", ""]
    if note:
        lines.extend([note, ""])
    lines.append("| | " + " | ".join(cell(r.mode) for r in reports) + " |")
    lines.append("|---|" + "---|" * len(reports))
    rows = [
        ("通过率", lambda r: _pct(r.pass_rate)),
        ("工具召回", lambda r: _pct(r.mean("tool_recall"))),
        ("事实召回", lambda r: _pct(r.mean("fact_recall")) + ("" if all(s.keyed for s in r.scores) else "*")),
        ("校验通过率", lambda r: _pct(r.rate(lambda s: s.grounded))),
        ("校验结果", lambda r: ", ".join(f"{k}={v}" for k, v in sorted(r.verify_outcomes().items())) or "-"),
        ("错误归属命中", lambda r: str(sum(1 for s in r.scores if s.forbidden_hit))),
        ("校验器疑似误报", lambda r: str(len(r.checker_false_positives())) if all(s.keyed for s in r.scores) else "-"),
        ("重写丢事实", lambda r: str(len(r.repair_regressions())) if any(s.draft for s in r.scores) else "-"),
        ("超预算", lambda r: str(sum(1 for s in r.scores if s.stop_reason == "deadline"))),
        ("工具调用 / 例", lambda r: f"{r.mean('tool_calls'):.1f}"),
        ("LLM 调用 / 例", lambda r: f"{r.mean('llm_calls'):.1f}"),
        ("token / 例", lambda r: f"{r.mean('tokens'):.0f}"),
        ("每通过 token", lambda r: f"{(sum(s.tokens for s in r.scores) / r.passed):.0f}" if r.passed and any(s.tokens for s in r.scores) else "-"),
        ("稳定失败", lambda r: str(len(r.stable_failures()))),
        ("不稳定用例", lambda r: str(len(r.flaky())) if r.repeat > 1 else "-"),
    ]
    for label, fn in rows:
        lines.append(f"| **{label}** | " + " | ".join(cell(fn(r)) for r in reports) + " |")
    lines.extend(["", "| 类别 | " + " | ".join(cell(r.mode) for r in reports) + " |", "|---|" + "---|" * len(reports)])
    categories = list(dict.fromkeys(score.category for report in reports for score in report.scores))
    for category in categories:
        cells = []
        for report in reports:
            passed, total = report.by_category().get(category, (0, 0))
            cells.append(f"{passed}/{total}")
        lines.append(f"| {category} | " + " | ".join(cells) + " |")
    if any(not s.keyed for r in reports for s in r.scores):
        lines.extend(["", "\\* engine fixtures：V1 事实答案键仅供参考，不作门槛。"])
    return "\n".join(lines) + "\n"


def to_json(reports: list[ModeReport]) -> dict[str, Any]:
    return {
        "modes": [
            {
                "mode": report.mode,
                "repeat": report.repeat,
                "pass_rate": report.pass_rate,
                "scores": [asdict(score) for score in report.scores],
            }
            for report in reports
        ]
    }


def gap_summary(cases: tuple[BenchmarkCase, ...] = DEV_CASES) -> dict[str, list[str]]:
    """Which V1 tools the port could not map, and which cases they block."""

    gaps: dict[str, list[str]] = {}
    for case in cases:
        for tool in case.unmapped_tools:
            gaps.setdefault(tool, []).append(case.id)
    return gaps


__all__ = ["MODES", "BenchmarkScore", "ModeReport", "run_benchmark", "run_mode", "render", "to_json", "to_markdown", "gap_summary", "by_category"]
