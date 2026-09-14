"""Hard-rule filtering for screening candidates."""

from __future__ import annotations

from v2.screening.models import FilterConfig, ScreenCandidate


DEFAULT_FILTERS = FilterConfig()


def passes_filter(c: ScreenCandidate, cfg: FilterConfig) -> bool:
    """Return True iff *c* satisfies all hard rules.

    Missing data (None) on any required field fails the check —
    we'd rather skip a candidate than guess.
    """
    if c.market_cap is None or not (cfg.market_cap_min <= c.market_cap <= cfg.market_cap_max):
        return False
    if c.revenue_growth is None or c.revenue_growth < cfg.revenue_growth_min:
        return False
    if c.gross_margin is None or c.gross_margin < cfg.gross_margin_min:
        return False
    if c.volatility is None or c.volatility > cfg.volatility_max:
        return False
    return True
