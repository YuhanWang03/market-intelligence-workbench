"""Data classes for v2/macro — Phase 4 Macro Agent.

The agent consumes FRED (canonical EOD numerics) + yfinance (intraday
market levels) + Tavily (FOMC sell-side aggregate) and produces three
shapes of report depending on the cron context:

- :class:`MacroSnapshot` — the 16:30 ET ⑭ daily close snapshot.
- :class:`MacroRelease` — one specific data release (CPI / PCE / NFP /
  GDP / PPI / Claims). Multiple may fire on the same day.
- :class:`FOMCEvent` — FOMC-specific shape (statement diff + SEP dots
  + sell-side consensus). FOMC happens 8× / year on pre-announced dates.

All numeric fields default to ``None`` where the data source can omit
them; renderers downstream gate on ``None`` rather than crash, mirroring
the Phase 1/2/3 convention.

Stage 0 design ack'd decisions baked in:
- LLM does NOT produce numbers (template-fill, only labels).
- LLM does NOT predict forward direction (Layer 1 prompt + Layer 2 regex).
- FOMC hawkish/dovish judgment uses Python diff + Tavily majority vote
  (Layer 3), never LLM verdict.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Vintage handling for revised data (CPI prelim → final, NFP -100K revision)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReleaseVintage:
    """One version of a release's value.

    BLS / BEA frequently revise prior months. Tracking vintages lets the
    summarizer say "May NFP revised down 50K" when that's the story,
    instead of pretending the new print is the only data point.
    """

    value: float
    vintage: str            # "prelim" / "second" / "final"
    published_at: str       # ISO datetime — the FRED vintage_date
    is_revision: bool       # False for first print; True for any revision


# ---------------------------------------------------------------------------
# One data release event
# ---------------------------------------------------------------------------

@dataclass
class MacroRelease:
    """A single macro data release (CPI / PCE / NFP / GDP / PPI / Claims).

    ``surprise_pct`` and ``surprise_sigma`` are populated only when a
    consensus number is available (we keep a small in-module table of
    sell-side consensus for the big monthly prints; otherwise the field
    is None and the summarizer treats the release as "no consensus").

    ``trailing_3mo_trend`` is computed by Python from the FRED series
    history — the LLM never decides "accelerating" vs "decelerating".
    """

    release_type: str       # "CPI" / "PCE" / "NFP" / "GDP" / "PPI" / "Claims"
    release_date: str       # ISO date — when the print hit the wire
    period: str             # "2026-05" (monthly) / "2026Q1" (quarterly)

    # Core numbers — Python computes; LLM never produces these.
    headline: float | None = None
    core: float | None = None
    mom_pct: float | None = None
    yoy_pct: float | None = None

    # Consensus + surprise
    consensus: float | None = None
    surprise_pct: float | None = None      # (actual - consensus) / abs(consensus)
    surprise_sigma: float | None = None    # standardized
    surprise_label: str = "no_consensus"   # "in_line" / "above_1sigma" / ...

    # Trend (Python compute)
    trailing_3mo_trend: str = "unknown"    # "accelerating" / "decelerating" / "flat"
    prior_value: float | None = None

    # Revisions
    vintage_history: list[ReleaseVintage] = field(default_factory=list)

    # Optional LLM-produced qualitative labels (Layer 1+2 enforced)
    bull_takeaway: str | None = None
    bear_takeaway: str | None = None
    narrative: str | None = None
    tone: str = "neutral"                  # "hawkish" / "dovish" / "neutral"


# ---------------------------------------------------------------------------
# FOMC event — Layer 3 defense path
# ---------------------------------------------------------------------------

@dataclass
class FOMCEvent:
    """FOMC meeting outcome: statement + (Mar/Jun/Sep/Dec) SEP dot plot.

    Per Stage 0 design ack: the LLM is NOT asked to judge hawkish vs
    dovish from the statement. Python diffs key phrases; the SEP dot
    plot extracts numbers; Tavily aggregates sell-side reactions. The
    summary the user sees is the union of (a) Python-tagged statement
    changes, (b) numeric SEP shift, (c) external desk majority vote.
    """

    meeting_date: str
    statement_text: str = ""
    statement_diff: dict = field(default_factory=dict)
    has_sep: bool = False                          # True only for Mar/Jun/Sep/Dec
    sep_median_dots: dict | None = None            # {2025: 4.25, "longer_run": 2.875}
    sep_dot_plot_change: str = "no_change"         # "hawkish_shift" / "dovish_shift"

    # Tavily sell-side aggregate (Layer 3)
    sell_side_sentiment: str | None = None         # "hawkish" / "dovish" / "mixed"
    sell_side_sources: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Daily 16:30 ET market snapshot
# ---------------------------------------------------------------------------

@dataclass
class MacroSnapshot:
    """Post-close snapshot pushed by ⑭ cron.

    Markets (VIX, DXY, WTI, gold) come from yfinance (15-min delayed
    quote is fine for post-close). Rates come from FRED only — Yahoo
    Finance's ^TNX / ^FVX / ^TYX historically return values × 10 which
    burned us in Stage 0 exploration; FRED is canonical.

    Anomaly flags are computed by Python from the same values — no LLM
    decides "this is a spike", the thresholds are explicit and pinned.
    """

    snapshot_date: str

    # Markets (yfinance)
    vix: float | None = None
    vix_pct_change_1d: float | None = None
    dxy: float | None = None
    wti_crude: float | None = None
    gold: float | None = None

    # Rates (FRED canonical EOD)
    fed_funds_upper: float | None = None
    fed_funds_lower: float | None = None
    dgs2: float | None = None
    dgs10: float | None = None
    t10y2y: float | None = None              # 10Y - 2Y spread
    t10y2y_prior: float | None = None        # prior trading day

    # Anomaly flags (Python compute against fixed thresholds)
    vix_spike: bool = False                  # +20% single-day
    vix_elevated: bool = False               # +10% single-day
    curve_flip: bool = False                 # T10Y2Y flipped sign today
    rates_shocked: bool = False              # |DGS10 daily Δ| >= 20bps

    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Top-level report
# ---------------------------------------------------------------------------

@dataclass
class MacroReport:
    """Envelope returned by the pipeline. Different crons populate
    different subsets:

    - ⑭ daily snapshot: ``snapshot`` only.
    - ⑮ release-day fire: ``today_releases`` (+ optional ``fomc_event``).
    - ⑯ Thursday claims: a single ``today_releases`` entry (Claims).
    - ⑰ Friday weekly recap: a dict-shaped recap (handled separately).

    ``warnings`` aggregates sub-failures across all sub-fetches so the
    cron path can ship a degraded card rather than crash silently.
    """

    report_date: str
    snapshot: MacroSnapshot | None = None
    today_releases: list[MacroRelease] = field(default_factory=list)
    fomc_event: FOMCEvent | None = None
    warnings: list[str] = field(default_factory=list)
