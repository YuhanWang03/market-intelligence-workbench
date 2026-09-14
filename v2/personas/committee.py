"""Run many personas over many tickers and add the votes up — transparently.

Upstream, a "portfolio manager" LLM read the thirteen opinions and made the
call.  Here the aggregation is arithmetic: every non-abstaining persona casts
a vote weighted by its confidence, and the result carries the vote counts
and the dispersion so a reader can see a 7-6 split for what it is.
"""

from __future__ import annotations

import logging
import time
from v2.usage_context import ContextExecutor as ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable, Iterable

from v2.personas.base import Persona
from v2.personas.models import PersonaSignal
from v2.personas.registry import PERSONAS, list_personas
from v2.personas.snapshot import OPTIONAL_NEEDS, PersonaSnapshot, build_snapshot

logger = logging.getLogger(__name__)

Progress = Callable[[str, str], None]


@dataclass
class TickerVerdict:
    """The committee's view of one ticker."""

    ticker: str
    as_of: str
    consensus: float           # -1 (all bearish, fully confident) .. +1
    net_votes: int             # bullish minus bearish, unweighted
    bullish: int
    bearish: int
    neutral: int
    abstained: int
    agreement: float           # share of voters on the majority side, 0..1
    avg_confidence: float      # over voters (non-abstaining)
    signals: list[PersonaSignal] = field(default_factory=list)
    snapshot_hash: str = ""
    data_gaps: list[str] = field(default_factory=list)
    rank: int | None = None

    @property
    def voters(self) -> int:
        return self.bullish + self.bearish + self.neutral

    @property
    def stance(self) -> str:
        if self.voters == 0:
            return "abstain"
        if self.consensus >= 0.2:
            return "bullish"
        if self.consensus <= -0.2:
            return "bearish"
        return "neutral"

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "as_of": self.as_of,
            "stance": self.stance,
            "consensus": round(self.consensus, 4),
            "net_votes": self.net_votes,
            "bullish": self.bullish,
            "bearish": self.bearish,
            "neutral": self.neutral,
            "abstained": self.abstained,
            "voters": self.voters,
            "agreement": round(self.agreement, 4),
            "avg_confidence": round(self.avg_confidence, 1),
            "rank": self.rank,
            "snapshot_hash": self.snapshot_hash,
            "data_gaps": list(self.data_gaps),
            "signals": [s.to_dict() for s in self.signals],
        }


@dataclass
class CommitteeResult:
    as_of: str
    personas: list[str]
    verdicts: list[TickerVerdict]          # sorted by consensus, best first
    elapsed_s: float = 0.0
    errors: dict[str, str] = field(default_factory=dict)
    #: the snapshots the verdicts were computed from (not serialized; for caching)
    snapshots: dict[str, PersonaSnapshot] = field(default_factory=dict, repr=False)

    def verdict(self, ticker: str) -> TickerVerdict | None:
        for v in self.verdicts:
            if v.ticker == ticker:
                return v
        return None

    def top(self, n: int = 15, *, min_voters: int = 1, min_agreement: float = 0.0) -> list[TickerVerdict]:
        """Best ``n`` tickers by consensus, with optional quality floors."""
        picks = [v for v in self.verdicts if v.voters >= min_voters and v.agreement >= min_agreement]
        return picks[:n]

    def matrix(self) -> dict[str, dict[str, PersonaSignal]]:
        """``persona -> ticker -> signal`` for a grid view."""
        grid: dict[str, dict[str, PersonaSignal]] = {p: {} for p in self.personas}
        for v in self.verdicts:
            for s in v.signals:
                grid.setdefault(s.persona, {})[v.ticker] = s
        return grid

    def to_dict(self) -> dict[str, Any]:
        return {
            "as_of": self.as_of,
            "personas": list(self.personas),
            "elapsed_s": round(self.elapsed_s, 2),
            "errors": dict(self.errors),
            "verdicts": [v.to_dict() for v in self.verdicts],
        }


def tally(signals: Iterable[PersonaSignal], *, ticker: str = "", as_of: str = "") -> TickerVerdict:
    """Aggregate one ticker's persona signals into a :class:`TickerVerdict`."""
    signals = list(signals)
    voters = [s for s in signals if not s.abstained]
    bullish = sum(1 for s in voters if s.signal == "bullish")
    bearish = sum(1 for s in voters if s.signal == "bearish")
    neutral = len(voters) - bullish - bearish
    weight = sum(s.confidence for s in voters)
    consensus = sum(s.direction * s.confidence for s in voters) / (100.0 * len(voters)) if voters else 0.0
    majority = max(bullish, bearish, neutral) if voters else 0
    gaps: list[str] = []
    for s in signals:
        for g in s.data_gaps:
            if g not in gaps:
                gaps.append(g)
    return TickerVerdict(
        ticker=ticker or (signals[0].ticker if signals else ""),
        as_of=as_of or (signals[0].as_of if signals else ""),
        consensus=consensus,
        net_votes=bullish - bearish,
        bullish=bullish,
        bearish=bearish,
        neutral=neutral,
        abstained=len(signals) - len(voters),
        agreement=majority / len(voters) if voters else 0.0,
        avg_confidence=weight / len(voters) if voters else 0.0,
        signals=signals,
        snapshot_hash=signals[0].snapshot_hash if signals else "",
        data_gaps=gaps,
    )


def needs_for(personas: Iterable[Persona]) -> set[str]:
    """Which optional snapshot inputs this set of personas actually reads."""
    wanted: set[str] = set()
    for p in personas:
        wanted.update(p.needs)
    return wanted & set(OPTIONAL_NEEDS)


def analyze_snapshot(snap: PersonaSnapshot, personas: Iterable[Persona]) -> TickerVerdict:
    """Run every persona over one already-built snapshot."""
    signals: list[PersonaSignal] = []
    for p in personas:
        try:
            signals.append(p.analyze(snap))
        except Exception as exc:  # noqa: BLE001 — one persona's bug must not sink the vote
            logger.exception("%s failed on %s", p.key, snap.ticker)
            signals.append(PersonaSignal(
                persona=p.key, ticker=snap.ticker, as_of=snap.as_of, signal="neutral", confidence=0,
                score=0.0, max_score=0.0, reasoning=f"abstain · error: {type(exc).__name__}: {str(exc)[:120]}",
                abstained=True, data_gaps=list(snap.gaps), snapshot_hash=snap.content_hash,
            ))
    return tally(signals, ticker=snap.ticker, as_of=snap.as_of)


def run_committee(
    tickers: Iterable[str],
    client: Any | None = None,
    *,
    personas: Iterable[str] | None = None,
    as_of: str | date | None = None,
    snapshots: dict[str, PersonaSnapshot] | None = None,
    max_workers: int = 4,
    progress: Progress | None = None,
    exclude_needs: Iterable[str] = (),
) -> CommitteeResult:
    """Build one snapshot per ticker (in parallel) and let the personas vote.

    ``snapshots`` lets a caller reuse cached snapshots (keyed by ticker) and
    skip the fetch; anything missing from it is fetched through ``client``.
    """
    started = time.time()
    keys = list(personas or PERSONAS)
    people = list_personas(keys)
    need = needs_for(people) - set(exclude_needs)
    wanted = []
    for t in tickers:
        t = t.strip().upper()
        if t and t not in wanted:
            wanted.append(t)
    ready = dict(snapshots or {})
    errors: dict[str, str] = {}

    def fetch(ticker: str) -> PersonaSnapshot | None:
        if ticker in ready:
            return ready[ticker]
        if progress:
            progress(ticker, "fetching")
        try:
            return build_snapshot(ticker, as_of, client, need=need)
        except Exception as exc:  # noqa: BLE001
            logger.exception("snapshot failed for %s", ticker)
            errors[ticker] = f"{type(exc).__name__}: {str(exc)[:160]}"
            return None

    to_fetch = [t for t in wanted if t not in ready]
    if to_fetch:
        with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
            for ticker, snap in zip(to_fetch, pool.map(fetch, to_fetch)):
                if snap is not None:
                    ready[ticker] = snap

    verdicts: list[TickerVerdict] = []
    for ticker in wanted:
        snap = ready.get(ticker)
        if snap is None:
            continue
        if progress:
            progress(ticker, "scoring")
        verdicts.append(analyze_snapshot(snap, people))

    verdicts.sort(key=lambda v: (v.voters > 0, v.consensus, v.agreement, v.avg_confidence), reverse=True)
    for i, v in enumerate(verdicts, start=1):
        v.rank = i
    return CommitteeResult(
        as_of=verdicts[0].as_of if verdicts else (str(as_of)[:10] if as_of else date.today().isoformat()),
        personas=keys,
        verdicts=verdicts,
        elapsed_s=time.time() - started,
        errors=errors,
        snapshots={t: ready[t] for t in wanted if t in ready},
    )
