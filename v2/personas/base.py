"""The persona contract and the shared verdict rules.

A persona is a checklist, not a chatbot.  ``evaluate()`` turns a snapshot
into scored parts and facts with no LLM in the loop; ``decide()`` turns
those numbers into a signal and a confidence by fixed rules; ``analyze()``
strings the two together and writes a templated one-line reasoning.  An
LLM only ever enters later, in :mod:`v2.personas.narrate`, to phrase what
the numbers already said.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from v2.personas.models import Evaluation, PersonaSignal, Signal, SubScore
from v2.personas.snapshot import PersonaSnapshot

BULLISH_RATIO = 0.7
BEARISH_RATIO = 0.3


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def classic_verdict(ratio: float, *, bullish_at: float = BULLISH_RATIO, bearish_at: float = BEARISH_RATIO) -> Signal:
    """The rule the pre-2026 upstream agents applied before calling the LLM."""
    if ratio >= bullish_at:
        return "bullish"
    if ratio <= bearish_at:
        return "bearish"
    return "neutral"


def confidence_from_ratio(ratio: float, signal: Signal, *, bullish_at: float = BULLISH_RATIO, bearish_at: float = BEARISH_RATIO) -> int:
    """Map score ratio to a 10-95 confidence, monotone in distance from the cut.

    * bullish: 55 at the threshold, 95 at a perfect score.
    * bearish: 55 at the threshold, 95 at zero.
    * neutral: 50 in the middle of the band, tapering to 35 at either edge
      (a neutral right next to a cut is the least certain kind).
    """
    if signal == "bullish":
        span = max(1e-9, 1.0 - bullish_at)
        return int(round(clamp(55 + 40 * (ratio - bullish_at) / span, 10, 95)))
    if signal == "bearish":
        span = max(1e-9, bearish_at)
        return int(round(clamp(55 + 40 * (bearish_at - ratio) / span, 10, 95)))
    mid = (bullish_at + bearish_at) / 2
    half = max(1e-9, (bullish_at - bearish_at) / 2)
    return int(round(clamp(50 - 15 * abs(ratio - mid) / half, 10, 95)))


def apply_margin_of_safety(signal: Signal, confidence: int, mos: float | None, *, overvalued_at: float = -0.25) -> tuple[Signal, int]:
    """Valuation gate shared by the value-style personas.

    A business can score well on quality and still be a poor purchase; the
    upstream prompts all said some version of "bullish only with a margin of
    safety".  Encoded once here so every persona applies it the same way.
    """
    if mos is None:
        return signal, confidence
    if signal == "bullish" and mos <= 0:
        return "neutral", int(clamp(confidence - 15, 10, 69))
    if signal == "neutral" and mos < overvalued_at:
        return "bearish", int(clamp(max(confidence, 50), 10, 95))
    if signal == "bullish" and mos > 0.3:
        return signal, int(clamp(confidence + 5, 10, 95))
    return signal, confidence


def _as_date(value: Any):
    """YYYY-MM-DD (or any ISO prefix) → date, else None."""
    if not value:
        return None
    try:
        from datetime import date

        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


class Persona(ABC):
    """Base class for one simulated investor."""

    #: stable identifier used in APIs, storage and the CLI (e.g. ``warren_buffett``)
    key: ClassVar[str]
    #: display name in English / Chinese
    name: ClassVar[str]
    name_zh: ClassVar[str] = ""
    #: one-line description of the style, as the upstream README phrased it
    style: ClassVar[str] = ""
    #: reporting period the rules were written for
    period: ClassVar[str] = "ttm"
    #: how many periods the rules look back over
    lookback: ClassVar[int] = 10
    #: optional data this persona reads; drives what the snapshot fetches
    needs: ClassVar[frozenset[str]] = frozenset()
    #: inputs without which the checklist is meaningless — missing any → abstain.
    #: Every upstream agent scores mostly off line items, so a snapshot with
    #: ratios but no line items must not be read as "everything scores zero".
    requires: ClassVar[frozenset[str]] = frozenset({"metrics", "line_items"})
    #: minimum line-item periods the rules need before a verdict is honest
    min_periods: ClassVar[int] = 3
    #: upstream system prompt (voice + decision checklist); narration only
    system_prompt: ClassVar[str] = ""

    # -- the contract ----------------------------------------------------------

    @abstractmethod
    def evaluate(self, snap: PersonaSnapshot) -> Evaluation:
        """Score the snapshot. Deterministic; no network, no LLM."""

    def decide(self, ev: Evaluation) -> tuple[Signal, int]:
        """Signal + confidence from an evaluation. Override for persona-specific rules."""
        signal = classic_verdict(ev.ratio)
        confidence = confidence_from_ratio(ev.ratio, signal)
        return apply_margin_of_safety(signal, confidence, ev.margin_of_safety)

    def analyze(self, snap: PersonaSnapshot) -> PersonaSignal:
        gaps = list(snap.gaps)
        missing = self.missing_inputs(snap)
        if missing:
            return self._abstain(snap, f"missing inputs: {', '.join(missing)}", gaps)
        ev = self.evaluate(snap)
        if ev.insufficient or ev.max_score <= 0:
            note = "; ".join(ev.notes) or "insufficient data"
            return self._abstain(snap, note, gaps, ev)
        signal, confidence = self.decide(ev)
        confidence = int(clamp(confidence, 0, 100))
        return PersonaSignal(
            persona=self.key,
            ticker=snap.ticker,
            as_of=snap.as_of,
            signal=signal,
            confidence=confidence,
            score=ev.score,
            max_score=ev.max_score,
            parts=list(ev.parts),
            facts=dict(ev.facts),
            margin_of_safety=ev.margin_of_safety,
            reasoning=self.explain(ev, signal, confidence),
            data_gaps=gaps,
            snapshot_hash=snap.content_hash,
        )

    # -- helpers subclasses may override --------------------------------------

    def missing_inputs(self, snap: PersonaSnapshot) -> list[str]:
        """Names of required inputs the snapshot lacks; non-empty means abstain.

        Each entry names the input and, when the snapshot recorded why the
        fetch failed, quotes that reason so the UI can show it.
        """
        missing: list[str] = []

        def gap_for(prefix: str) -> str:
            for g in snap.gaps:
                if g.startswith(prefix):
                    return f" ({g})"
            return ""

        if "metrics" in self.requires and not snap.metrics(self.period):
            missing.append(f"{self.period} metrics" + gap_for(f"metrics_{self.period}"))
        if "line_items" in self.requires:
            items = snap.line_items(self.period)
            if not items:
                missing.append(f"{self.period} line items" + gap_for(f"line_items_{self.period}"))
            elif len(items) < self.min_periods:
                missing.append(f"only {len(items)} {self.period} line-item period(s), need {self.min_periods}")
        if "prices" in self.requires and not snap.prices:
            missing.append("daily prices" + gap_for("prices"))
        if "market_cap" in self.requires and not snap.market_cap:
            missing.append("market cap" + gap_for("market_cap"))
        return missing

    def explain(self, ev: Evaluation, signal: Signal, confidence: int) -> str:
        """Templated, deterministic one-liner: score, verdict, the loudest parts."""
        head = f"{signal} ({confidence}%) · score {round(ev.score, 2):g}/{round(ev.max_score, 2):g}"
        if ev.margin_of_safety is not None:
            head += f" · margin of safety {ev.margin_of_safety:+.0%}"
        bits = [f"{p.name}: {p.details}" for p in ev.parts if p.details]
        text = head + (" · " + " | ".join(bits) if bits else "")
        return text if len(text) <= 600 else text[:597] + "..."

    def _abstain(self, snap: PersonaSnapshot, note: str, gaps: list[str], ev: Evaluation | None = None) -> PersonaSignal:
        return PersonaSignal(
            persona=self.key,
            ticker=snap.ticker,
            as_of=snap.as_of,
            signal="neutral",
            confidence=0,
            score=ev.score if ev else 0.0,
            max_score=ev.max_score if ev else 0.0,
            parts=list(ev.parts) if ev else [],
            facts=dict(ev.facts) if ev else {},
            margin_of_safety=ev.margin_of_safety if ev else None,
            reasoning=f"abstain · {note}",
            abstained=True,
            data_gaps=gaps,
            snapshot_hash=snap.content_hash,
        )

    # -- small shared arithmetic ----------------------------------------------

    @staticmethod
    def series(rows: list[Any], field: str, *, positive_only: bool = False) -> list[float]:
        """Extract a numeric field from rows (latest first), skipping nulls."""
        out: list[float] = []
        for row in rows:
            value = getattr(row, field, None)
            if value is None:
                continue
            try:
                f = float(value)
            except (TypeError, ValueError):
                continue
            if f != f:  # NaN
                continue
            if positive_only and f <= 0:
                continue
            out.append(f)
        return out

    @staticmethod
    def latest(rows: list[Any], field: str) -> float | None:
        for row in rows:
            value = getattr(row, field, None)
            if value is not None:
                try:
                    f = float(value)
                except (TypeError, ValueError):
                    return None
                return None if f != f else f
        return None

    @staticmethod
    def years_spanned(rows: list[Any], *, period: str | None = None) -> float | None:
        """Elapsed years between the newest and oldest row (rows are newest-first).

        Upstream divided growth by ``len(rows) - 1`` — right for annual rows,
        wrong for TTM rows, where ten periods span about two years, not nine.
        Use the report dates when the rows carry them; otherwise assume a
        quarter per step unless the period label says annual.
        """
        if len(rows) < 2:
            return None
        newest = _as_date(getattr(rows[0], "report_period", None))
        oldest = _as_date(getattr(rows[-1], "report_period", None))
        if newest and oldest and newest > oldest:
            return (newest - oldest).days / 365.25
        label = str(period or getattr(rows[0], "period", "") or "").lower()
        step = 1.0 if label in ("annual", "fy", "yearly", "year") else 0.25
        return (len(rows) - 1) * step

    @classmethod
    def cagr(cls, rows: list[Any], field: str, *, positive_only: bool = True, min_years: float = 0.5) -> tuple[float, float] | None:
        """``(annualised growth, years)`` of ``field`` from oldest to newest row, or None.

        Skips rows where the field is missing (or non-positive when
        ``positive_only``), measures the span in calendar years, and refuses
        spans shorter than ``min_years`` — a two-quarter CAGR is noise.
        """
        picked: list[tuple[Any, float]] = []
        for row in rows:
            value = getattr(row, field, None)
            if value is None:
                continue
            try:
                f = float(value)
            except (TypeError, ValueError):
                continue
            if f != f or (positive_only and f <= 0):
                continue
            picked.append((row, f))
        if len(picked) < 2:
            return None
        years = cls.years_spanned([r for r, _ in picked])
        if not years or years < min_years:
            return None
        newest, oldest = picked[0][1], picked[-1][1]
        if newest <= 0 or oldest <= 0:
            return None
        return (newest / oldest) ** (1.0 / years) - 1.0, years

    @staticmethod
    def part(name: str, score: float, max_score: float, details: list[str] | str, **data: Any) -> SubScore:
        text = details if isinstance(details, str) else "; ".join(d for d in details if d)
        return SubScore(name=name, score=float(score), max_score=float(max_score), details=text, data=data)

    def __repr__(self) -> str:
        return f"<Persona {self.key}>"
