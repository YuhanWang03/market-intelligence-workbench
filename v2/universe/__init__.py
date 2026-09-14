"""Tradable universes + sector ETF mappings.

Centralizes ticker → sector ETF lookups so the monitoring, screening, and
streaming layers can ask the same question: "what is this ticker's
benchmark for relative-strength comparison?"
"""

from v2.universe.etfs import (
    BENCHMARK_ETF,
    BROAD_BUCKET,
    BROAD_MARKET_ETFS,
    OTHER_BUCKET,
    SECTOR_ETFS,
    TICKER_TO_SECTOR,
    sector_bucket_for,
    sector_etf_for,
)

__all__ = [
    "BENCHMARK_ETF",
    "BROAD_BUCKET",
    "BROAD_MARKET_ETFS",
    "OTHER_BUCKET",
    "SECTOR_ETFS",
    "TICKER_TO_SECTOR",
    "sector_bucket_for",
    "sector_etf_for",
]
