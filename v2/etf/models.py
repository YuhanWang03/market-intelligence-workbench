"""Data models for the ETF holdings module."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ETFHolding:
    etf: str
    date: str
    ticker: str | None
    cusip: str | None
    company: str
    shares: float
    market_value: float
    weight_pct: float


@dataclass
class ETFChange:
    """One position's day-over-day delta within an ETF."""
    etf: str
    ticker: str
    company: str
    cusip: str | None
    shares_today: float
    shares_yesterday: float
    shares_diff: float
    shares_diff_pct: float          # (today - yesterday) / yesterday, 0 if new
    weight_pct: float               # today's weight
    weight_diff_pp: float           # today_weight - yesterday_weight (pp)
    is_new: bool                    # held today, not yesterday
    is_exit: bool                   # held yesterday, not today
