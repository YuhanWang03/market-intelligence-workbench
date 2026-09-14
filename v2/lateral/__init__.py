"""Lateral expansion — LLM-driven discovery of supply-chain neighbors."""

from v2.lateral.models import (
    CATEGORIES,
    CATEGORY_LABEL_CN,
    Label,
    LateralResult,
    Neighbor,
)
from v2.lateral.verify import verify
from v2.screening.models import FilterConfig

# Default seeds: 7 mega-cap tech stocks with the richest supply-chain footprints
DEFAULT_SEEDS: list[str] = [
    "NVDA", "AAPL", "MSFT", "AMD", "TSLA", "META", "GOOGL",
]

# Looser filter for ③ — lateral discovery should be permissive about smaller
# adjacent names. Screening (① TECH_30) keeps the strict quality bar.
LATERAL_FILTERS = FilterConfig(
    market_cap_min=3_000_000_000,         # $3B  (was $10B in ①)
    market_cap_max=5_000_000_000_000,     # $5T
    revenue_growth_min=0.03,              # 3%   (was 5%)
    gross_margin_min=0.35,                # 35%  (was 50%)
    volatility_max=0.70,                  # 70%  (was 60%)
)


def discover(*args, **kwargs):
    from v2.lateral.discover import discover as _discover
    return _discover(*args, **kwargs)


def run_lateral_expansion(*args, **kwargs):
    from v2.lateral.orchestrator import run_lateral_expansion as _run
    return _run(*args, **kwargs)

__all__ = [
    "CATEGORIES",
    "CATEGORY_LABEL_CN",
    "DEFAULT_SEEDS",
    "LATERAL_FILTERS",
    "Label",
    "LateralResult",
    "Neighbor",
    "discover",
    "run_lateral_expansion",
    "verify",
]
