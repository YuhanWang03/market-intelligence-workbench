"""Data models for the Phase-2 Portfolio Risk Agent.

All shapes are intentionally simple frozen dataclasses (not pydantic) — the
risk pipeline runs as a one-shot cron, there's no schema validation surface
to expose, and the formatters need positional readability over runtime
schema enforcement.

Five clusters of metrics roll up into a single :class:`RiskReport`:

- :class:`PositionFlat` — a single ticker's normalized snapshot
- :class:`ConcentrationMetrics` — Top-N + HHI
- :class:`ExposureMetrics` — sector ETF aggregation
- :class:`PnLMetrics` — 1D / 1W / 1M return
- :class:`DrawdownMetrics` — peak-trough max drawdown over 1M
- :class:`EarningsRiskItem` — one upcoming earnings release in next 7d

Every numeric field that could be unavailable is typed ``float | None``
so callers can render "数据不足" gracefully instead of crashing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


# ---------------------------------------------------------------------------
# Position-level
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PositionFlat:
    """One holding, normalized from Alpaca's :class:`Position` model.

    ``weight`` is the position's share of ``portfolio_value`` (NOT cash).
    ``sector_etf`` comes from :func:`v2.universe.sector_bucket_for` — so
    unmapped tickers carry ``"OTHER"`` rather than the SPY fallback used
    in the signal path.
    """

    ticker: str
    market_value: float
    weight: float                  # 0.0 - 1.0
    sector_etf: str                # ETF symbol or "OTHER"


# ---------------------------------------------------------------------------
# Concentration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConcentrationMetrics:
    """Top-N and Herfindahl-Hirschman concentration measures.

    HHI is the sum of squared weights, scaled to [0, 1]. A perfectly equal
    20-position book has HHI = 0.05; a single-position book has HHI = 1.0.
    Industry rule of thumb: HHI > 0.25 is "highly concentrated".
    """

    top_1_pct: float               # share of single largest position
    top_3_pct: float               # cumulative share of top 3
    top_5_pct: float               # cumulative share of top 5
    hhi: float                     # Herfindahl-Hirschman Index, [0, 1]
    n_positions: int               # number of non-zero positions

    @classmethod
    def empty(cls) -> "ConcentrationMetrics":
        return cls(
            top_1_pct=0.0, top_3_pct=0.0, top_5_pct=0.0,
            hhi=0.0, n_positions=0,
        )


# ---------------------------------------------------------------------------
# Sector exposure
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExposureMetrics:
    """Position weights aggregated by sector ETF.

    ``by_sector`` keys are ETF symbols (``"SMH"``, ``"XLK"``, ``"XLF"`` …)
    or the literal ``"OTHER"`` for unmapped tickers. Values are fractions
    in [0, 1]; they sum to the total invested portion (1 - cash %).
    """

    by_sector: dict[str, float] = field(default_factory=dict)
    largest_sector: str = ""
    largest_sector_pct: float = 0.0

    @classmethod
    def empty(cls) -> "ExposureMetrics":
        return cls()


# ---------------------------------------------------------------------------
# P&L
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PnLMetrics:
    """Period P&L in dollar + percent terms.

    Daily numbers come from the Alpaca account snapshot (no history call
    needed). Weekly / monthly come from ``portfolio_history(1M, 1D)``
    walking back N trading days. None means insufficient history (e.g.
    account is < 1 week old) — caller renders "数据不足".
    """

    daily_pnl: float | None         # USD
    daily_pnl_pct: float | None     # fraction
    weekly_pnl_pct: float | None    # fraction over last 5 trading days
    monthly_pnl_pct: float | None   # fraction over last ~21 trading days

    @classmethod
    def unavailable(cls) -> "PnLMetrics":
        return cls(daily_pnl=None, daily_pnl_pct=None,
                   weekly_pnl_pct=None, monthly_pnl_pct=None)


# ---------------------------------------------------------------------------
# Drawdown
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DrawdownMetrics:
    """Peak-trough drawdown over the 1-month window.

    Both fields are **non-negative magnitudes** (Stage 5 convention).
    Renderers prepend ``-`` always; this contract removes a class of
    sign-flip bugs that bit Stage 2.5's dry-run output.

    ``current_drawdown_pct`` measures distance from the most recent
    peak as a positive fraction (``0.034`` = "3.4% below peak").
    ``max_drawdown_pct`` is the worst peak-to-trough drop in the
    window, also a positive magnitude.
    """

    current_drawdown_pct: float | None    # ≥ 0 fraction, or None if unavailable
    max_drawdown_pct: float | None        # ≥ 0 fraction, or None if unavailable
    peak_value: float | None              # USD at the peak point
    peak_date: date | None                # date of that peak

    @classmethod
    def unavailable(cls) -> "DrawdownMetrics":
        return cls(current_drawdown_pct=None, max_drawdown_pct=None,
                   peak_value=None, peak_date=None)


# ---------------------------------------------------------------------------
# Earnings risk
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EarningsRiskItem:
    """One upcoming earnings release for a held ticker, ≤ 7 days out."""

    ticker: str
    release_date: str              # ISO date
    days_until: int                # 0 = today, 7 = a week out
    estimated_eps: float | None
    estimated_revenue: float | None


# ---------------------------------------------------------------------------
# Top-level report
# ---------------------------------------------------------------------------

@dataclass
class RiskReport:
    """Full risk snapshot — what the 18:30 ET cron pushes.

    All sub-metrics are populated independently; a single failure (e.g.
    portfolio_history unavailable → ``PnLMetrics.unavailable()``) does
    not block the others. The cron renders whatever fields it has.

    Field semantics (Stage 2.5 clarification — Alpaca's
    ``account.portfolio_value`` is the TOTAL equity, not invested-only):

    - ``portfolio_value`` = Alpaca account.portfolio_value =
      invested equity + cash.
    - ``cash`` = uninvested cash (Alpaca account.cash).
    - ``cash_pct`` = cash / portfolio_value (cash as share of TOTAL).
    - ``invested_value`` = portfolio_value - cash (derived @property,
      cannot drift from the source fields).
    """

    snapshot_date: str             # ISO date YYYY-MM-DD
    portfolio_value: float         # TOTAL equity (invested + cash)
    cash: float
    cash_pct: float                # cash / portfolio_value

    positions: list[PositionFlat] = field(default_factory=list)
    concentration: ConcentrationMetrics = field(default_factory=ConcentrationMetrics.empty)
    exposure: ExposureMetrics = field(default_factory=ExposureMetrics.empty)
    pnl: PnLMetrics = field(default_factory=PnLMetrics.unavailable)
    drawdown: DrawdownMetrics = field(default_factory=DrawdownMetrics.unavailable)
    earnings_risk_next_7d: list[EarningsRiskItem] = field(default_factory=list)

    # Per-section error notes — surfaced in cards as small italics so the
    # user knows WHY a section is empty rather than wondering if their
    # cron broke.
    warnings: list[str] = field(default_factory=list)

    @property
    def invested_value(self) -> float:
        """Invested equity = portfolio_value - cash.

        Derived rather than stored so it can't drift from the source
        fields if a caller mutates ``cash`` or ``portfolio_value`` later.
        Clamps to 0 to absorb Alpaca rounding (cash > portfolio_value
        is theoretically impossible but can show up at $0.01 scale).
        """
        return max(0.0, self.portfolio_value - self.cash)
