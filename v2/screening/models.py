"""Pydantic models for the screening pipeline."""

from __future__ import annotations

from pydantic import BaseModel, Field


class FilterConfig(BaseModel):
    """Hard rules for the basic-fundamental screen.

    All thresholds are inclusive boundaries. Missing data fails the check.
    """

    market_cap_min: float = 10_000_000_000        # $10B
    market_cap_max: float = 5_000_000_000_000     # $5T (room for AAPL/NVDA)
    revenue_growth_min: float = 0.05              # 5% YoY — post-AI-cycle reality
    gross_margin_min: float = 0.50                # 50% — software-like quality
    volatility_max: float = 0.60                  # 60% annualized log-return std


class ScreenCandidate(BaseModel):
    """One ticker's snapshot — the data we use to filter and narrate."""

    ticker: str
    price: float
    price_change: float | None = None            # latest 1-day return
    market_cap: float | None = None
    revenue_growth: float | None = None
    gross_margin: float | None = None
    volatility: float | None = None              # annualized
    high_52w: float | None = None                # for "市值突破" tag

    # Phase-2 改进 ④: Earnings surprise (actual vs Wall-Street estimate)
    revenue_actual: float | None = None
    revenue_estimate: float | None = None
    revenue_surprise_pct: float | None = None    # (actual - est) / est
    eps_actual: float | None = None
    eps_estimate: float | None = None
    eps_surprise_pct: float | None = None

    # Phase-2 改进 ②: Hard tags computed from the metrics above
    tags: list[str] = []

    # Phase-2 改进 ①: Dynamic delta context for the narrator
    return_1w: float | None = None               # 1-week price return (5 trading days)
    peer_diff_1w: float | None = None            # return_1w - cohort_median_return_1w
    news_headlines: list[dict] = []              # [{title, snippet}, ...] from Tavily

    # ETF benchmarking — context for sector-relative narration.
    sector_etf: str | None = None
    sector_return_1w: float | None = None         # sector ETF 1-week return
    sector_diff_1w_pp: float | None = None        # return_1w - sector_return_1w, in pp

    bull: str = ""                               # filled by narrator (pure logic, no numbers)
    bear: str = ""                               # filled by narrator (pure logic, no numbers)


class ScreenResult(BaseModel):
    """Top-level result returned by run_screening()."""

    date: str
    universe_size: int
    candidates: list[ScreenCandidate] = Field(default_factory=list)
    api_calls: int = 0          # legacy: total external calls (kept for compat)
    fd_calls: int = 0           # Financial Datasets only
    tavily_calls: int = 0       # Tavily news search only
    llm_tokens: int = 0
