"""Data models for the Earnings Agent.

Three shapes:

- :class:`EarningsEvent` — a *forward* calendar entry (release in the future,
  or today before/after market). Sourced from yfinance.
- :class:`EarningsHistorical` — a *past* filing, normalised from
  ``v2.data.models.EarningsRecord``. The shape the rest of v2 already uses
  ("BEAT" / "MISS" / "MEET" plus actual/estimate numbers).
- :class:`EarningsSummary` — the post-release card payload assembled by the
  summarizer for Telegram + dashboard rendering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


# ---------------------------------------------------------------------------
# Forward calendar
# ---------------------------------------------------------------------------

When = Literal["bmo", "amc", "unknown"]
"""bmo = before market open, amc = after market close, unknown = time TBA."""


@dataclass(frozen=True)
class EarningsEvent:
    """One upcoming earnings release.

    All numeric fields are optional — yfinance often only knows the date.
    """

    ticker: str
    release_date: str               # ISO date, e.g. "2026-07-31"
    when: When = "unknown"
    eps_estimate: float | None = None
    revenue_estimate: float | None = None
    n_analysts: int | None = None
    source: str = "yfinance"

    def d_minus(self, today_iso: str) -> int:
        """Days until release. Negative if release is in the past.

        ``today_iso`` is passed in (not computed from datetime.today()) so the
        caller controls the timezone — the cron runs in ET.
        """
        from datetime import date
        d_rel = date.fromisoformat(self.release_date)
        d_today = date.fromisoformat(today_iso)
        return (d_rel - d_today).days


# ---------------------------------------------------------------------------
# Historical filing
# ---------------------------------------------------------------------------

EpsSurprise = Literal["BEAT", "MISS", "MEET", "UNKNOWN"]


@dataclass(frozen=True)
class EarningsHistorical:
    """A past filing, normalised from ``v2.data.models.EarningsRecord``."""

    ticker: str
    report_period: str              # ISO date — quarter end
    filing_date: str                # ISO date — when 8-K/10-Q/10-K landed
    source_type: str                # "8-K" | "10-Q" | "10-K" | "20-F"
    eps_actual: float | None = None
    eps_estimate: float | None = None
    eps_surprise: EpsSurprise = "UNKNOWN"
    revenue_actual: float | None = None
    revenue_estimate: float | None = None

    @property
    def has_quarterly_data(self) -> bool:
        """True iff we have anything useful to render."""
        return self.eps_surprise != "UNKNOWN" or self.eps_actual is not None

    def eps_surprise_pct(self) -> float | None:
        """Signed surprise as a fraction (0.05 = +5%). None if not computable."""
        if self.eps_actual is None or self.eps_estimate is None:
            return None
        if self.eps_estimate == 0:
            return None
        return (self.eps_actual - self.eps_estimate) / abs(self.eps_estimate)

    def revenue_surprise_pct(self) -> float | None:
        if self.revenue_actual is None or self.revenue_estimate is None:
            return None
        if self.revenue_estimate <= 0:
            return None
        return (self.revenue_actual - self.revenue_estimate) / self.revenue_estimate


# ---------------------------------------------------------------------------
# Post-release summary card
# ---------------------------------------------------------------------------

@dataclass
class EarningsSummary:
    """Assembled payload for the post-release summary card.

    Numeric facts come from Python (FD); only the qualitative slots
    (``bull``, ``bear``, ``narrative``) are filled by the LLM under
    Template-Fill discipline.
    """

    ticker: str
    report_period: str
    filing_date: str
    eps_surprise: EpsSurprise
    eps_actual: float | None = None
    eps_estimate: float | None = None
    revenue_actual: float | None = None
    revenue_estimate: float | None = None

    # Optional context
    last_4q_surprises: list[EpsSurprise] = field(default_factory=list)
    transcript_url: str | None = None

    # LLM-authored, never numeric:
    bull: str = ""
    bear: str = ""
    narrative: str = ""

    # Phase 3.5 — optional 10-Q delta block surfaced as a separate
    # card section. None when the ticker has no recent 10-Q (just
    # filed an 8-K-only earnings release / parse failed). Typed as
    # ``object`` to avoid a v2.earnings → v2.sec runtime import; the
    # formatter duck-types on the field names of TenQDelta.
    ten_q_delta: object | None = None
