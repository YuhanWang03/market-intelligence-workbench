"""Scoring — turning one run into numbers that say *why* it failed.

A single pass/fail per case is almost useless for improving anything: a run that
called the wrong tools and a run that called the right ones but dropped a fact
look identical, and they need opposite fixes. So every case is scored on four
independent axes, and ``passed`` is their conjunction:

* **tool recall** — did it fetch what the answer needs? Extra tools are not
  penalised here; they are charged to the cost metrics, where over-calling
  actually costs something.
* **fact recall** — did the required facts survive into the reply? Each fact is
  a set of acceptable surface forms, so this measures correctness rather than
  phrasing.
* **discipline** — did it call a tool the case marks as wrong or wasteful, or
  emit a string the case forbids (the classic being a figure borrowed from a
  different ticker when the real one has no data).
* **grounding** — every figure traces to an observation. An answer with an
  invented number is not a correct answer, so this is part of ``passed`` rather
  than a footnote.

The baseline's grounding is 1.0 by construction (it returns the card verbatim
and writes nothing), which is a real advantage of that design and is left
visible rather than normalised away.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from typing import Any, Iterable

from v2.agent_eval.cases import CATEGORIES, EvalCase

_WS = re.compile(r"\s+")


_DATE_CJK_FULL = re.compile(r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日?")
_DATE_CJK = re.compile(r"(?<!\d)(\d{1,2})\s*月\s*(\d{1,2})\s*日?")
_DATE_ISO = re.compile(r"(?<!\d)(\d{4})-(\d{1,2})-(\d{1,2})(?!\d)")
_DATE_MD = re.compile(r"(?<![\d.])(\d{1,2})-(\d{1,2})(?![\d.])")


def _canonical_dates(text: str) -> str:
    """「2026 年 9 月 6 日」, ``2026-09-06``, 「9月6日」, ``9-06`` → ``2026-9-6`` / ``9-6``.

    The labels were written in the ISO forms deepseek-chat copies off the cards.
    gpt-4.1-mini writes 「2026 年 9 月 6 日」, and its first clean sweep booked
    that as 事实缺失 on p03, p06 and others — the key was scoring the date's
    spelling, not the answer's correctness. Both sides of the match go through
    this, so a label written either way accepts an answer written either way.
    """
    text = _DATE_CJK_FULL.sub(lambda m: f"{int(m[1])}-{int(m[2])}-{int(m[3])}", text)
    text = _DATE_CJK.sub(lambda m: f"{int(m[1])}-{int(m[2])}", text)
    text = _DATE_ISO.sub(lambda m: f"{int(m[1])}-{int(m[2])}-{int(m[3])}", text)
    text = _DATE_MD.sub(lambda m: f"{int(m[1])}-{int(m[2])}", text)
    return text


def normalise(text: str) -> str:
    """Lowercase, drop thousands separators, collapse whitespace, canonicalise dates.

    Removing commas is what lets a case assert ``184,320.55`` and still match a
    model that wrote ``184320.55``. Whitespace is collapsed rather than removed
    so that English quotes keep their word boundaries.
    """
    return _WS.sub(" ", _canonical_dates((text or "").replace(",", "")).lower()).strip()


def fact_present(forms: Iterable[str], answer: str) -> bool:
    """True when any acceptable surface form of one fact appears.

    Matching is also tried with all whitespace removed, because Chinese answers
    space dates differently from the labels — a model writing 「9月6日」 was
    being scored as a miss against the form "9 月 6". That measured the label's
    spacing, not the answer's correctness.
    """
    haystack = normalise(answer)
    tight = haystack.replace(" ", "")
    for form in forms:
        if not form:
            continue
        needle = normalise(form)
        if needle in haystack or needle.replace(" ", "") in tight:
            return True
    return False


@dataclass
class CaseScore:
    case_id: str
    category: str
    mode: str

    tool_recall: float = 0.0
    fact_recall: float = 0.0
    grounded: bool = True
    missing_tools: tuple[str, ...] = ()
    missing_facts: tuple[str, ...] = ()
    violations: tuple[str, ...] = ()
    #: Unnecessary tools called. A cost signal, never a failure.
    waste: tuple[str, ...] = ()
    forbidden_hit: tuple[str, ...] = ()
    #: The figures that failed to trace. "数字无法溯源" without naming them is
    #: the same unactionable verdict this package criticises elsewhere.
    ungrounded: tuple[str, ...] = ()
    #: {kind: [figures]} — why each rejected figure failed to trace.
    ungrounded_kinds: dict[str, list[str]] = field(default_factory=dict)
    #: Figures accepted because the answer showed the arithmetic behind them.
    derived: int = 0
    #: Misattribution findings raised against this answer, as
    #: "entity←figure(实为 owners)" strings.
    #:
    #: These are scored as **failures of the checker, not of the model**. Every
    #: other assertion on the case already decides whether the answer put the
    #: right number against the right company: ``facts`` require the correct
    #: pairings and ``forbidden`` names the borrowed ones. So when a case's own
    #: assertions all pass and attribution still fires, the finding is a false
    #: positive by construction — the check rejecting a correct answer.
    #:
    #: This axis did not exist for the first nine rounds. Seven false positives
    #: reached production, each found by a human reading a Telegram message, and
    #: every one of them was invisible here: the case passed, the warning was
    #: never looked at.
    misattributed: tuple[str, ...] = ()

    #: tool name -> times called in this run. Overspend says a run used 25
    #: calls; only this says whether that was one tool fanned out 25 ways or
    #: eight tools called three times each — and those have different fixes.
    calls_by_tool: dict[str, int] = field(default_factory=dict)
    #: Calls refused because the tool had hit its per-run cap.
    capped_calls: int = 0
    #: Calls turned away before reaching a tool (duplicate, capped, malformed).
    #: Kept out of ``tool_calls`` — a refusal costs nothing and must not inflate
    #: the cost metric, least of all the one the cap exists to reduce.
    refused_calls: int = 0

    tool_calls: int = 0
    llm_calls: int = 0
    tokens: int = 0
    elapsed_ms: int = 0
    overspend: bool = False

    path: str = ""
    path_correct: bool = True
    stop_reason: str = ""
    error: str = ""

    #: The calls that reached a tool, in order, as ``tool(arg=value)``. Kept
    #: per run so that a passing and a failing run of the same case can be
    #: laid side by side — the pass rate says a case is flaky, only this says
    #: where the two runs parted.
    trace: tuple[str, ...] = ()
    #: The final answer text. Not scored here (every assertion above already
    #: is); recorded so a wording flake can be read instead of guessed at.
    answer: str = ""
    #: Repair rounds taken (0 or 1), the answer the checks rejected, and why.
    repairs: int = 0
    draft: str = ""
    draft_findings: tuple[str, ...] = ()
    #: Whether the *draft* carried every fact the case asks for and none of
    #: the forbidden ones — the answer-key part of the score, applied to the
    #: text the checks refused.
    draft_facts_ok: bool = False
    #: The rewrite dropped more than half of the draft's traced figures — it
    #: came back as the corrected fragment rather than the whole answer.
    partial_rewrite: bool = False

    @property
    def answer_correct(self) -> bool:
        """Everything the case asserts about the answer itself."""
        return (self.tool_recall >= 1.0
                and self.fact_recall >= 1.0
                and not self.violations
                and not self.forbidden_hit
                and self.grounded
                and not self.error)

    @property
    def false_misattribution(self) -> bool:
        """A misattribution warning on an answer the case says is correct."""
        return self.answer_correct and bool(self.misattributed)

    @property
    def repair_regressed(self) -> bool:
        """The draft had the facts; the rewrite the checks demanded lost them.

        This is the failure the false-positive axis could not see. A warning
        on a *final* answer that is correct counts as the check's error; a
        warning on a draft that is correct triggers a rewrite, the rewrite
        drops the flagged fact along with the rest, and the case fails on
        「事实缺失」 — booked against the model. m07 and r09 both failed this
        way in the first 6×10 sweep, and the checker's own score stayed at 0.

        A rewrite that keeps the facts and is *still* ungrounded (c02, v01
        in the first full sweep: the same invented figures, twice) is not a
        regression — the check was right and the model did not comply. Only
        a final answer missing a fact, tool or forbidden-string assertion
        the draft satisfied counts.
        """
        return (self.repairs > 0 and self.draft_facts_ok
                and bool(self.missing_facts or self.forbidden_hit or self.missing_tools))

    @property
    def passed(self) -> bool:
        return self.answer_correct and not self.misattributed

    def failure_reason(self) -> str:
        """The single most actionable reason, for the per-case failure list."""
        if self.error:
            return f"运行错误：{self.error}"
        if self.repair_regressed:
            shape = "重写只回了改动的那一段" if self.partial_rewrite else "重写丢了事实"
            return (f"{shape}（初稿事实齐全，被打回：{'；'.join(self.draft_findings[:4])}）"
                    f" → {self._own_reason()}")
        return self._own_reason()

    def _own_reason(self) -> str:
        if self.tool_recall < 1.0:
            return f"工具漏调：{', '.join(self.missing_tools)}"
        if self.violations:
            return f"调用了不该调的：{', '.join(self.violations)}"
        if self.forbidden_hit:
            return f"输出了禁止内容：{', '.join(self.forbidden_hit)}"
        if not self.grounded:
            figures = ", ".join(self.ungrounded[:6]) if self.ungrounded else "未记录"
            return f"数字无法溯源：{figures}"
        if self.fact_recall < 1.0:
            return f"事实缺失：{', '.join(self.missing_facts)}"
        return ""


def score_case(
    case: EvalCase,
    *,
    mode: str,
    answer: str,
    tools_called: Iterable[str],
    grounded: bool = True,
    ungrounded: Iterable[str] = (),
    ungrounded_kinds: dict[str, list[str]] | None = None,
    derived: int = 0,
    misattributed: Iterable[str] = (),
    calls_by_tool: dict[str, int] | None = None,
    capped_calls: int = 0,
    refused_calls: int = 0,
    tool_calls: int = 0,
    llm_calls: int = 0,
    tokens: int = 0,
    elapsed_ms: int = 0,
    path: str = "",
    stop_reason: str = "",
    error: str = "",
    trace: Iterable[str] = (),
    repairs: int = 0,
    draft: str = "",
    draft_findings: Iterable[str] = (),
    partial_rewrite: bool = False,
) -> CaseScore:
    called = set(tools_called)

    missing_tools = tuple(t for t in case.must_call if t not in called)
    tool_recall = 1.0 if not case.must_call else 1.0 - len(missing_tools) / len(case.must_call)

    # Data facts and behavioural requirements gate a case identically; they
    # differ only in whether the answer key check expects to find them in the
    # fixtures.
    required = tuple(case.facts) + tuple(case.behaviors)
    missing_facts = tuple(forms[0] for forms in required if not fact_present(forms, answer))
    fact_recall = 1.0 if not required else 1.0 - len(missing_facts) / len(required)

    haystack = normalise(answer)
    forbidden_hit = tuple(f for f in case.forbidden if normalise(f) in haystack)

    draft_facts_ok = bool(draft) and all(
        fact_present(forms, draft) for forms in required) and not any(
        normalise(f) in normalise(draft) for f in case.forbidden)

    return CaseScore(
        case_id=case.id, category=case.category, mode=mode,
        tool_recall=tool_recall, fact_recall=fact_recall, grounded=grounded,
        missing_tools=missing_tools, missing_facts=missing_facts,
        violations=tuple(sorted(called & set(case.must_not_call))),
        waste=tuple(sorted(called & set(case.wasteful_tools))),
        forbidden_hit=forbidden_hit, ungrounded=tuple(ungrounded),
        ungrounded_kinds=dict(ungrounded_kinds or {}), derived=derived,
        misattributed=tuple(misattributed),
        calls_by_tool=dict(calls_by_tool or {}), capped_calls=capped_calls,
        refused_calls=refused_calls,
        tool_calls=tool_calls, llm_calls=llm_calls, tokens=tokens,
        elapsed_ms=elapsed_ms, overspend=tool_calls > case.max_tool_calls,
        path=path, path_correct=(not path or path == case.expected_path),
        stop_reason=stop_reason, error=error,
        trace=tuple(trace), answer=answer,
        repairs=repairs, draft=draft, draft_findings=tuple(draft_findings),
        draft_facts_ok=draft_facts_ok, partial_rewrite=partial_rewrite,
    )


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


# ---------------------------------------------------------------------------
# Where a flaky case forks
# ---------------------------------------------------------------------------
#
# Six cases stayed flaky across sweeps with the sampling temperature already at
# zero, so "lower the temperature" is not available and "run it again" only
# re-rolls the dice. What can be acted on is *where* the runs part:
#
#   error        a failing run never finished (provider error, exception) — noise
#                from outside the loop; nothing in the loop to fix
#   budget       same calls as a passing run, then the failing one ran out of
#                steps, calls or seconds before writing — a stop-rule problem
#   early_stop   same calls as a passing run, then the failing one decided it
#                had enough and wrote — the evidence it needed was usually
#                already in hand (m07: explain_move(ARM) called, 7.42% not
#                written), so this points at the answer, not the tool table
#   tool_choice  the failing run called a different tool, or the same tool with
#                different arguments, somewhere before the end — the model chose
#                differently on identical context; the prompt or a description
#                left the choice open
#   wording      identical path, identical evidence, different answer text —
#                the fork is in what the model wrote, not what it saw
#
# The order matters: a run that errored out has a shorter trace too, and would
# otherwise be misread as a tool-choice fork.

FLAKE_KINDS = ("error", "budget", "early_stop", "tool_choice", "wording")

_BUDGET_STOPS = frozenset({"max_steps", "budget_exhausted"})


@dataclass(frozen=True)
class Divergence:
    case_id: str
    kind: str
    #: The passing runs' most common path.
    good_trace: tuple[str, ...]
    #: One failing run's path — the first failing run of the majority kind.
    bad_trace: tuple[str, ...]
    #: Index into the traces where the failing path first differs from the
    #: passing one; ``None`` when the paths are identical.
    fork_at: int | None
    #: Distinct failure reasons across the failing runs.
    reasons: tuple[str, ...]
    #: Stop reasons of the failing runs, deduplicated.
    bad_stops: tuple[str, ...]


def _fork_index(good: tuple[str, ...], bad: tuple[str, ...]) -> int | None:
    for i, (a, b) in enumerate(zip(good, bad)):
        if a != b:
            return i
    return None if len(good) == len(bad) else min(len(good), len(bad))


def _classify_one(good: tuple[str, ...], bad: CaseScore) -> str:
    if bad.error:
        return "error"
    fork = _fork_index(good, bad.trace)
    if fork is None:
        return "wording"
    # Same calls as far as it got, then stopped by a limit: the loop had the
    # evidence and ran out of room, which is a different bug from choosing badly.
    if bad.trace == good[:len(bad.trace)]:
        return "budget" if bad.stop_reason in _BUDGET_STOPS else "early_stop"
    return "tool_choice"


def diverge(runs: list[CaseScore]) -> Divergence:
    """Classify one case's runs by where its failing runs left the passing path.

    With no passing run there is no path to compare against and the kind is
    whatever the failing runs say on their own (an error, a budget stop, or
    ``tool_choice`` as the honest "it never found the path"). With no failing
    run there is nothing to classify.
    """
    if not runs:
        raise ValueError("no runs")
    case_id = runs[0].case_id
    passing = [r for r in runs if r.passed]
    failing = [r for r in runs if not r.passed]
    if passing:
        counts: dict[tuple[str, ...], int] = {}
        for r in passing:
            counts[r.trace] = counts.get(r.trace, 0) + 1
        good = max(counts, key=lambda t: (counts[t], -len(t)))
    else:
        good = ()
    if not failing:
        return Divergence(case_id, "stable_pass", good, good, None, (), ())
    kinds = [_classify_one(good, r) for r in failing] if passing else [
        "error" if r.error else "budget" if r.stop_reason in _BUDGET_STOPS
        else "tool_choice" for r in failing]
    kind = max(FLAKE_KINDS, key=kinds.count)
    bad = failing[kinds.index(kind)]
    return Divergence(
        case_id, kind, good, bad.trace,
        _fork_index(good, bad.trace) if passing else None,
        tuple(dict.fromkeys(r.failure_reason() for r in failing)),
        tuple(dict.fromkeys(r.stop_reason for r in failing if r.stop_reason)),
    )


@dataclass
class SuiteReport:
    mode: str
    scores: list[CaseScore] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.scores)

    @property
    def passed(self) -> int:
        return sum(1 for s in self.scores if s.passed)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    #: How many times each case was run (1 unless the sweep asked for repeats).
    repeat: int = 1

    def stability(self) -> dict[str, tuple[int, int]]:
        """case_id -> (times passed, times run)."""
        out: dict[str, tuple[int, int]] = {}
        for score in self.scores:
            passed, total = out.get(score.case_id, (0, 0))
            out[score.case_id] = (passed + (1 if score.passed else 0), total + 1)
        return out

    def stable_failures(self) -> list[str]:
        """Cases that failed every time — the ones worth acting on."""
        return sorted(cid for cid, (p, n) in self.stability().items() if p == 0)

    def flaky(self) -> list[tuple[str, int, int]]:
        """Cases that passed sometimes and failed others."""
        return sorted((cid, p, n) for cid, (p, n) in self.stability().items()
                      if 0 < p < n)

    def runs(self, case_id: str) -> list[CaseScore]:
        return [s for s in self.scores if s.case_id == case_id]

    def divergence(self, case_id: str) -> "Divergence":
        """Where the passing and failing runs of one flaky case part ways."""
        return diverge(self.runs(case_id))

    def flaky_by_kind(self) -> dict[str, list[str]]:
        """kind -> flaky case ids, in the order of FLAKE_KINDS."""
        out: dict[str, list[str]] = {}
        for case_id, _p, _n in self.flaky():
            out.setdefault(self.divergence(case_id).kind, []).append(case_id)
        return {k: out[k] for k in FLAKE_KINDS if k in out}

    def by_category(self) -> dict[str, tuple[int, int]]:
        out: dict[str, tuple[int, int]] = {}
        for category in CATEGORIES:
            group = [s for s in self.scores if s.category == category]
            if group:
                out[category] = (sum(1 for s in group if s.passed), len(group))
        return out

    def summary(self) -> dict[str, Any]:
        tokens = [float(s.tokens) for s in self.scores]
        latency = [float(s.elapsed_ms) for s in self.scores]
        passed = self.passed
        return {
            "mode": self.mode,
            "total": self.total,
            "passed": passed,
            "pass_rate": self.pass_rate,
            "tool_recall": _mean([s.tool_recall for s in self.scores]),
            "fact_recall": _mean([s.fact_recall for s in self.scores]),
            "grounded_rate": _mean([1.0 if s.grounded else 0.0 for s in self.scores]),
            "violation_rate": _mean([1.0 if s.violations else 0.0 for s in self.scores]),
            "waste_rate": _mean([1.0 if s.waste else 0.0 for s in self.scores]),
            "overspend_rate": _mean([1.0 if s.overspend else 0.0 for s in self.scores]),
            "routing_accuracy": _mean([1.0 if s.path_correct else 0.0 for s in self.scores]),
            "ungrounded_kinds": self.ungrounded_breakdown(),
            "derived_figures": sum(sc.derived for sc in self.scores),
            # A warning raised on an answer the case says is correct. This is
            # the check's own error rate, and the number that should have been
            # on the table for the last nine rounds.
            "false_misattribution_rate": _mean(
                [1.0 if s.false_misattribution else 0.0 for s in self.scores]),
            "repair_rate": _mean([1.0 if s.repairs else 0.0 for s in self.scores]),
            # The check's *other* error rate: rewrites it demanded that lost
            # facts the draft had. Invisible to false_misattribution_rate.
            "repair_regression_rate": _mean(
                [1.0 if s.repair_regressed else 0.0 for s in self.scores]),
            "mean_tool_calls": _mean([float(s.tool_calls) for s in self.scores]),
            "mean_llm_calls": _mean([float(s.llm_calls) for s in self.scores]),
            "total_tokens": int(sum(tokens)),
            "median_tokens": _median(tokens),
            "median_latency_ms": _median(latency),
            # The metric that decides whether a mode is worth its cost.
            "tokens_per_pass": (sum(tokens) / passed) if passed else float("inf"),
            "repeat": self.repeat,
            "stable_failures": len(self.stable_failures()),
            "flaky": len(self.flaky()),
            "flaky_by_kind": {k: len(v) for k, v in self.flaky_by_kind().items()},
        }

    def ungrounded_breakdown(self) -> dict[str, int]:
        """How many rejected figures of each kind, across the whole suite."""
        counts: dict[str, int] = {}
        for score in self.scores:
            for kind, figures in score.ungrounded_kinds.items():
                counts[kind] = counts.get(kind, 0) + len(figures)
        return counts

    def failures(self) -> list[CaseScore]:
        return [s for s in self.scores if not s.passed]

    def repair_regressions(self) -> list[CaseScore]:
        """Runs whose draft had the facts and whose rewrite lost them."""
        return [s for s in self.scores if s.repair_regressed]

    def false_misattributions(self) -> list[tuple[str, tuple[str, ...]]]:
        """(case id, findings) for every warning raised on a correct answer."""
        return [(s.case_id, s.misattributed)
                for s in self.scores if s.false_misattribution]
