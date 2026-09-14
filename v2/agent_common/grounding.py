"""Numeric grounding check — the anti-hallucination guarantee, kept.

The bot's design rule was "the LLM never decides what to *say*, only which tool
to *call*". Letting a model write the final answer gives that rule up, so it has
to be replaced by something enforceable rather than simply dropped.

The replacement is narrow and mechanical: **every figure in the answer must
appear in some observation the run actually collected.** No judgement, no second
LLM, no cost. It cannot catch a wrong *claim* built out of right numbers, and it
is not meant to — it catches invented numbers, which is the failure mode that
matters when the output looks like a research note.

Two categories are exempt, both under-approximations chosen to keep the signal
honest rather than to flatter the score:

* structural small integers (ranks, counts, "top 3") and 4-digit years;
* figures the model derived by arithmetic, which are reported separately as
  ``derived`` rather than counted as grounded.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# Matches 1,234.56 / 22.3% / $1.2B / -3.6 — the shapes that appear in the cards.
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")

#: SEC item numbers and similar identifiers look like decimals but are labels.
#: "8-K Item 5.02" is not a quantity, and demanding it trace to an observation
#: rejects an answer for correctly naming the section it is talking about.
_IDENTIFIER_CONTEXT = re.compile(
    r"(?:item|section|条款|项)\s*$", re.IGNORECASE)

#: Text that reads as a number but names something — a filing, a date, a
#: countdown, a duration. Blanked with equal-length spaces before extraction so
#: every offset downstream still lines up.
#:
#: Owned here and shared with the attribution check, because for a while each
#: check kept its own list and they disagreed: attribution knew 13F, 「近 30
#: 天」 and 「52 周高点」 were not quantities, grounding did not, and 「你能帮我
#: 做什么」 — a capability answer with no observations at all — failed 0/3 for
#: mentioning them. Two checks with two ideas of what a figure is will drift
#: apart again if they are not the same function.
# `\b` is the wrong boundary for this text: CJK characters are \w, so `\b`
# never fires between 「构」 and 「13F」 or between 「秒」 and 「超」. 「机构13F持仓」
# left a bare 13 behind, and d06 (a capability answer, no tool called, every
# surviving digit ungrounded by construction) failed 3 runs in 10 on it. The
# boundaries below are "not a letter or digit", which is what was meant.
_L = r"(?<![A-Za-z0-9])"
_R = r"(?![A-Za-z0-9])"
NON_QUANTITY = re.compile(
    _L + r"\d{4}-\d{1,2}-\d{1,2}" + _R          # 2026-11-17
    + "|" + _L + r"\d{1,2}-\d{1,2}" + _R          # 10-21, 09-30
    + "|" + _L + r"\d{1,2}/\d{1,2}" + _R          # 9/30
    + "|" + _L + r"\d{1,4}\s?(?:ms|s|秒|分钟|小时)" + _R   # 30s 超时
    + "|" + _L + r"[A-Z]-\d{1,4}" + _R             # D-74
    + "|" + _L + r"\d{1,2}-[A-Z]" + _R             # 8-K, 10-Q
    + "|" + _L + r"\d{1,2}[FKQD]" + _R             # 13F, 13D
    # An indicator's lookback parameter: CMF(20), RSI(14). c07's table
    # header 「资金流 CMF(20)」 had its 20 attributed to SMH.
    + "|" + _L + r"[A-Z]{2,6}\(\d{1,3}\)"
    + r"|(?:[Ii]tem|[Ss]ection)\s*\d+(?:\.\d+)?")

#: A figure followed by a time unit is the size of a window, not a value:
#: 「52 周高点」「200 日均线」「过去 30 天」.
WINDOW_UNIT = re.compile(r"^\s*(?:个)?\s*(?:周|日|天|月|年|季|季度|小时)")

# A displayed figure can be a faithful scale conversion of a raw observation:
# 349,029,376 -> 3.49 亿, or 0.857 -> 85.7%.  Keep the unit next to the
# answer figure in the check so these conversions remain mechanical rather
# than becoming a general tolerance relaxation.
_SCALE_UNIT = re.compile(r"^\s*(万|亿|[KMBT])(?![A-Za-z])", re.IGNORECASE)
_SCALE_FACTORS = {"万": 1e4, "亿": 1e8, "K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}


def _scaled_candidates(value: float, suffix: str) -> tuple[float, ...]:
    candidates = [value]
    scale = _SCALE_UNIT.match(suffix)
    if scale:
        candidates.append(value * _SCALE_FACTORS[scale.group(1).upper()])
    if re.match(r"^\s*%", suffix):
        candidates.append(value / 100)
    return tuple(candidates)


def mask_non_quantities(text: str) -> str:
    return NON_QUANTITY.sub(lambda m: " " * len(m.group(0)), text or "")


def _normalise(token: str) -> str:
    return token.replace(",", "").lstrip("-").rstrip(".")


def _digit_signature(value: str) -> str:
    """Significant digits, ignoring the decimal point and trailing zeros.

    This is what makes a unit conversion traceable. A card printing ``$57.80B``
    and an answer writing ``578 亿`` describe the same quantity; only the scale
    word differs, and the scale word is not a figure the check can verify. Both
    reduce to ``578``. Same for ``$9.19M`` written as ``919 万``.
    """
    digits = _normalise(value).replace(".", "").lstrip("0")
    return digits.rstrip("0") or digits


def extract_numbers(text: str) -> list[str]:
    return [m.group(0) for m in _NUMBER.finditer(text or "")]


def extract_numbers_with_context(text: str) -> list[tuple[str, str, int, int]]:
    """Each figure with its preceding context and its span in the text."""
    body = text or ""
    return [(m.group(0), body[max(0, m.start() - 12): m.start()], m.start(), m.end())
            for m in _NUMBER.finditer(body)]


#: How far around a derived figure to look for the arithmetic that produced it.
_DERIVATION_WINDOW_BEFORE = 90
_DERIVATION_WINDOW_AFTER = 30


def _shows_its_working(
    target: float,
    answer: str,
    start: int,
    end: int,
    is_traceable,
) -> bool:
    """True when the answer displays traceable inputs that produce ``target``.

    Three sweeps of the evaluation showed the model would not comply with a
    prompt telling it to show its arithmetic: sums stayed ~30% of all rejections
    across every run. Instructing harder was not going to work, and accepting
    bare sums would gut the guarantee — a fabricated figure often has *some*
    arithmetic explanation in a card full of numbers.

    So the rule became precise instead of loose: a derived figure is traceable
    when the answer **shows** the derivation, and every input of that derivation
    is itself traceable. "前三合计 22.4% + 18.2% + 14.1% = 54.7%" passes; a bare
    "54.7%" does not. Faking this requires inventing addends that are each
    individually traceable *and* sum to the target, which is a far higher bar
    than inventing one number.
    """
    window = answer[max(0, start - _DERIVATION_WINDOW_BEFORE): start] \
        + answer[end: end + _DERIVATION_WINDOW_AFTER]
    inputs: list[float] = []
    for token in extract_numbers(window):
        try:
            value = abs(float(_normalise(token)))
        except ValueError:
            continue
        if value and is_traceable(value, token) and value not in inputs:
            inputs.append(value)
    if len(inputs) < 2:
        return False

    count = min(len(inputs), 8)
    inputs = inputs[:count]
    for i in range(count):
        for j in range(i + 1, count):
            pair = inputs[i] + inputs[j]
            if _approx(target, pair) or _approx(target, abs(inputs[i] - inputs[j])):
                return True
            for k in range(j + 1, count):
                triple = pair + inputs[k]
                if _approx(target, triple):
                    return True
                for m in range(k + 1, count):
                    if _approx(target, triple + inputs[m]):
                        return True
    return False


def _approx(a: float, b: float) -> bool:
    return abs(a - b) <= max(0.005, abs(b) * 0.005)


@dataclass
class GroundingReport:
    """Per-answer verdict, reported as a metric rather than a hard gate."""

    total: int = 0
    grounded: int = 0
    ungrounded: list[str] = field(default_factory=list)
    #: Figures that traced, as written. The repair round names them so the
    #: rewrite keeps them: told only what was wrong, the model deletes what
    #: was right along with it (p04 dropped the day's P&L to fix one
    #: percentage), and nothing in the loop rewards keeping a figure it was
    #: not told about.
    traced: list[str] = field(default_factory=list)
    exempt: int = 0
    #: Figures accepted because the answer *showed* the arithmetic producing them
    #: from figures that are themselves traceable.
    derived: int = 0

    @property
    def ratio(self) -> float:
        return 1.0 if self.total == 0 else self.grounded / self.total

    @property
    def ok(self) -> bool:
        return not self.ungrounded

    def summary(self) -> str:
        if self.total == 0:
            return "no figures in answer"
        state = "clean" if self.ok else f"{len(self.ungrounded)} unsupported"
        return f"{self.grounded}/{self.total} grounded ({self.ratio:.0%}) — {state}"


def check(
    answer: str,
    observations: str,
    *,
    exempt_below: int = 13,
    tolerate_years: bool = True,
    rounding_tolerance: float = 0.005,
) -> GroundingReport:
    """Verify every figure in ``answer`` traces back to ``observations``.

    Four ways a figure can trace, all of them meaning "this number came from the
    data" rather than "this number was written identically to the data". The
    first version only implemented the last one, and an evaluation sweep showed
    it rejecting mostly *correct* answers: 42% of its rejections were values it
    had rounded, and several more were unit conversions and SEC item numbers.
    Those were bugs in the check, not laxity in the model, and fixing them is
    not the same as relaxing the rule — a figure derived by arithmetic the
    answer does not show is still rejected.
    """
    haystack = (observations or "").replace(",", "")
    values: list[float] = []
    signatures: set[str] = set()
    for token in extract_numbers(observations):
        signatures.add(_digit_signature(token))
        try:
            values.append(abs(float(_normalise(token))))
        except ValueError:
            continue

    def traceable(value: float, token: str) -> bool:
        """Whether one figure would pass the direct checks on its own."""
        text = _normalise(token)
        if text in haystack or _digit_signature(token) in signatures:
            return True
        return any(abs(value - v) <= max(rounding_tolerance * v, 0.005) for v in values)

    report = GroundingReport()

    masked = mask_non_quantities(answer)
    for raw, before, start, end in extract_numbers_with_context(masked):
        token = _normalise(raw)
        if not token:
            continue
        if WINDOW_UNIT.match(masked[end:]):
            report.exempt += 1
            continue

        # Structural exemptions: ordinals, counts, years, identifiers.
        try:
            as_float = float(token)
        except ValueError:
            continue
        if as_float.is_integer() and abs(as_float) < exempt_below:
            report.exempt += 1
            continue
        if tolerate_years and as_float.is_integer() and 1900 <= as_float <= 2100:
            report.exempt += 1
            continue
        if _IDENTIFIER_CONTEXT.search(before):
            report.exempt += 1
            continue

        report.total += 1
        target = abs(as_float)
        candidates = _scaled_candidates(target, masked[end:])

        # 1. Written the same way the card wrote it.
        variants = {token, token.rstrip("0").rstrip("."), f"{as_float:g}"}
        if any(v and v in haystack for v in variants):
            report.grounded += 1
            report.traced.append(raw)
            continue
        # 2. The same quantity at a different scale (57.80B -> 578 亿).
        if _digit_signature(raw) in signatures:
            report.grounded += 1
            report.traced.append(raw)
            continue
        # 3. The card's value, rounded (15,851.57 -> 15,852).
        if any(abs(candidate - v) <= max(rounding_tolerance * v, 0.005) for candidate in candidates for v in values):
            report.grounded += 1
            report.traced.append(raw)
            continue
        # 4. Arithmetic the answer actually shows, over inputs that themselves trace.
        if _shows_its_working(target, answer, start, end, traceable):
            report.grounded += 1
            report.derived += 1
            report.traced.append(raw)
            continue
        report.ungrounded.append(raw)

    return report


# ---------------------------------------------------------------------------
# Diagnosis — what is the check actually rejecting?
# ---------------------------------------------------------------------------

#: Classification of one rejected figure, ordered from most benign to least.
#: Ratios were tried and removed: with dozens of numbers in the observations,
#: some a/b×100 lands within tolerance of almost any target by chance, which
#: launders fabricated figures into "legitimate arithmetic" — the exact error
#: this diagnosis exists to prevent.
FIGURE_KINDS = ("rounding", "sum", "difference", "unknown")


def _observation_numbers(observations: str, limit: int = 80) -> list[float]:
    """Distinct numeric values appearing in the observations."""
    seen: list[float] = []
    for token in extract_numbers(observations):
        try:
            value = abs(float(_normalise(token)))
        except ValueError:
            continue
        if value and value not in seen:
            seen.append(value)
        if len(seen) >= limit:
            break
    return seen


def _close(a: float, b: float, rel: float = 0.002, abs_tol: float = 0.005) -> bool:
    return abs(a - b) <= max(abs_tol, abs(b) * rel)


def classify_figure(figure: str, values: list[float]) -> str:
    """Explain *why* a figure failed to trace, rather than only that it did.

    "18% of answers were ungrounded" is not actionable: a model that invented a
    statistic and one that added two published numbers without showing its work
    look identical in that statistic, and they call for opposite responses —
    tighten the prompt, or relax the check. This splits them.

    The classification is a **hypothesis, not proof**. With dozens of numbers in
    the observations some combination will match by coincidence, so a "sum"
    label means "there exists an arithmetic explanation", not "the model did
    that arithmetic". Only ``unknown`` is safe to act on hard: nothing in the
    observations produces that figure by any of these routes, so it was almost
    certainly invented. Tolerances are tight and ratios are not attempted, both
    to keep coincidental matches down.
    """
    try:
        target = abs(float(_normalise(figure)))
    except ValueError:
        return "unknown"
    if not target:
        return "unknown"

    for value in values:
        if _close(target, value, rel=0.02, abs_tol=0.051):
            return "rounding"

    count = len(values)
    for i in range(count):
        for j in range(i + 1, count):
            if _close(target, values[i] + values[j]):
                return "sum"
            if _close(target, abs(values[i] - values[j])):
                return "difference"

    # Triples are common in "top three combined" style claims but quadratic ×
    # linear gets slow, so only try them on a small observation set.
    if count <= 45:
        for i in range(count):
            for j in range(i + 1, count):
                partial = values[i] + values[j]
                for k in range(j + 1, count):
                    if _close(target, partial + values[k]):
                        return "sum"
    return "unknown"


def diagnose(report: GroundingReport, observations: str) -> dict[str, list[str]]:
    """Group a report's rejected figures by why they failed."""
    values = _observation_numbers(observations)
    grouped: dict[str, list[str]] = {kind: [] for kind in FIGURE_KINDS}
    for figure in report.ungrounded:
        grouped[classify_figure(figure, values)].append(figure)
    return {kind: figures for kind, figures in grouped.items() if figures}


def repair_instruction(report: GroundingReport) -> str:
    """Message appended for the one repair round when figures don't trace."""
    return (
        "GROUNDING CHECK FAILED. These figures in your answer do not appear in "
        f"any tool result from this run: {', '.join(report.ungrounded[:12])}.\n"
        "Rewrite the answer using only figures present in the observations above. "
        "If a number was your own arithmetic, write the arithmetic out in full "
        "(e.g. 22.4% + 18.2% + 14.1% = 54.7%) — a shown derivation over traceable "
        "inputs is accepted, a bare result is not. "
        "If you cannot support it, drop the claim — omitting a number is always "
        "better than inventing one."
        + (f"\nEvery other figure traced and must stay exactly as written: "
           f"{', '.join(dict.fromkeys(report.traced))[:400]}."
           if report.traced else "")
        # A fact the model cannot know: this is not a conversation where the
        # draft stays visible. Told to "fix only the listed figures", it sent
        # back the one corrected sentence, and that sentence became the whole
        # answer (p04, the day's P&L gone).
        + "\nSend the COMPLETE answer again, every section, not only the corrected "
          "part: your reply replaces the draft in full and is the only text the "
          "user will see."
    )
