"""Sector ETF mappings — used for context-aware screening + anomaly detection
plus Phase-2 portfolio sector-exposure reporting.

Why this module: a "+5% volume spike" on NVDA means something very different
when SMH is +4.5% (board-wide rally — beta move) versus when SMH is -1%
(contrarian, ticker-specific signal — the kind that drives PMs to call up
sell-side desks). Tagging every signal with its sector-relative move
elevates real signal from market beta noise.

For Phase 2 (`v2/portfolio/exposure.py`), the same ``TICKER_TO_SECTOR``
mapping is used to aggregate position weights by sector ETF so the risk
card can flag "你对 SMH 暴露 38%" type concentrations. Unmapped tickers
fall back to the ``OTHER_BUCKET`` label rather than SPY so the bucket
stays visible to the user instead of being silently merged into "broad
market".
"""

from __future__ import annotations

# Sector / benchmark ETF symbols we track for relative-strength comparison.
# Pre-fetched ONCE per monitoring run by the orchestrator, then passed into
# every detect() call.
SECTOR_ETFS: list[str] = ["SPY", "XLK", "SMH"]

# Broad-market benchmark — used as a fallback for tickers without a sector
# mapping in the *signal* path (sector_etf_for), and for "vs market"
# comparisons. The portfolio-exposure path uses OTHER_BUCKET instead so
# unmapped names stay visible.
BENCHMARK_ETF: str = "SPY"

# Phase-2 exposure aggregation label for tickers without an explicit
# sector ETF mapping. Surfaced verbatim in the risk card so the user can
# tell "I'm 12% in stuff outside the known sectors" at a glance.
OTHER_BUCKET: str = "OTHER"

# Phase-2.5-mini label for broad-market ETFs (S&P 500, total-market,
# NASDAQ-100 style). These are *internally diversified* — concentrating
# in IVV is not the same risk as concentrating in a single name. The
# bucket lets the risk card surface that distinction in its alert
# footer instead of mislabeling broad-market exposure as "未分类标的".
BROAD_BUCKET: str = "BROAD"

BROAD_MARKET_ETFS: frozenset[str] = frozenset({
    "SPY",   # SPDR S&P 500
    "IVV",   # iShares Core S&P 500
    "VOO",   # Vanguard S&P 500
    "VTI",   # Vanguard Total Stock Market
    "QQQ",   # Invesco NASDAQ-100
    "DIA",   # SPDR Dow Jones
    "IWM",   # iShares Russell 2000
})

# Ticker → primary sector ETF. Extended in Phase 2 beyond the original
# tech focus so non-tech holdings (financials / healthcare / energy /
# staples / industrials / China ADRs) get sensible buckets in the
# portfolio risk report.
TICKER_TO_SECTOR: dict[str, str] = {
    # ---- Semiconductors → SMH (VanEck Semiconductor ETF) ----
    "NVDA": "SMH",
    "AMD":  "SMH",
    "AVGO": "SMH",
    "QCOM": "SMH",
    "INTC": "SMH",
    "TXN":  "SMH",
    "MU":   "SMH",
    "SMCI": "SMH",  # AI server / chip-adjacent
    "ARM":  "SMH",  # IP licensor — semi exposure
    "MRVL": "SMH",
    "LRCX": "SMH",
    "AMAT": "SMH",
    "KLAC": "SMH",
    "ASML": "SMH",  # ADR

    # ---- Mega-cap tech + software + internet → XLK ----
    "AAPL":  "XLK",
    "MSFT":  "XLK",
    "GOOGL": "XLK",
    "GOOG":  "XLK",
    "META":  "XLK",
    "AMZN":  "XLK",
    "TSLA":  "XLK",
    "ORCL":  "XLK",
    "CRM":   "XLK",
    "ADBE":  "XLK",
    "NOW":   "XLK",
    "SNOW":  "XLK",
    "PLTR":  "XLK",
    "CRWD":  "XLK",
    "PANW":  "XLK",
    "DDOG":  "XLK",
    "NFLX":  "XLK",
    "SHOP":  "XLK",
    "UBER":  "XLK",
    "IBM":   "XLK",
    "CSCO":  "XLK",
    "INTU":  "XLK",
    "ANET":  "XLK",

    # ---- Financials → XLF ----
    "JPM":   "XLF",
    "BAC":   "XLF",
    "WFC":   "XLF",
    "GS":    "XLF",
    "MS":    "XLF",
    "C":     "XLF",
    "BRK.B": "XLF",
    "BRK.A": "XLF",
    "BLK":   "XLF",
    "V":     "XLF",
    "MA":    "XLF",
    "AXP":   "XLF",
    "SCHW":  "XLF",
    "PYPL":  "XLF",

    # ---- Healthcare → XLV ----
    "JNJ":   "XLV",
    "PFE":   "XLV",
    "UNH":   "XLV",
    "LLY":   "XLV",
    "ABBV":  "XLV",
    "MRK":   "XLV",
    "TMO":   "XLV",
    "ABT":   "XLV",
    "DHR":   "XLV",
    "BMY":   "XLV",
    "AMGN":  "XLV",
    "GILD":  "XLV",

    # ---- Consumer staples → XLP ----
    "WMT":   "XLP",
    "PG":    "XLP",
    "KO":    "XLP",
    "PEP":   "XLP",
    "COST":  "XLP",
    "TGT":   "XLP",
    "MDLZ":  "XLP",

    # ---- Energy → XLE ----
    "XOM":   "XLE",
    "CVX":   "XLE",
    "COP":   "XLE",
    "OXY":   "XLE",
    "SLB":   "XLE",
    "VST":   "XLE",  # power utility / nuclear
    "GEV":   "XLE",  # GE Vernova
    "EOG":   "XLE",

    # ---- Communications services → XLC ----
    "DIS":   "XLC",
    "T":     "XLC",
    "VZ":    "XLC",
    "TMUS":  "XLC",
    "CMCSA": "XLC",

    # ---- Industrials → XLI ----
    "CAT":   "XLI",
    "HON":   "XLI",
    "RTX":   "XLI",
    "BA":    "XLI",
    "GE":    "XLI",
    "LMT":   "XLI",
    "DE":    "XLI",
    "UPS":   "XLI",

    # ---- China ADRs → KWEB ----
    "BABA":  "KWEB",
    "JD":    "KWEB",
    "PDD":   "KWEB",
    "BIDU":  "KWEB",
    "NIO":   "KWEB",
    "TCEHY": "KWEB",
}


def sector_etf_for(ticker: str) -> str:
    """Return the primary sector ETF symbol for *ticker*. Defaults to SPY.

    Used by the *signal* path (anomaly / screening) where SPY is the right
    fallback because it's the broad-market baseline. Portfolio exposure
    aggregation uses :func:`sector_bucket_for` instead so unmapped
    positions stay visible under the ``OTHER`` label.
    """
    return TICKER_TO_SECTOR.get(ticker.upper(), BENCHMARK_ETF)


def sector_bucket_for(ticker: str) -> str:
    """Return the sector ETF for *ticker*, or a special bucket if unmapped.

    Resolution order:

    1. Broad-market ETFs (``SPY``, ``IVV``, ``VOO``, …) → :data:`BROAD_BUCKET`.
       These are internally diversified, so the risk card can qualify
       concentration alerts ("you have 86% in IVV, but IVV is itself
       500 names").
    2. Explicit single-sector mapping in :data:`TICKER_TO_SECTOR`.
    3. Unmapped → :data:`OTHER_BUCKET`, so the user can see at a glance
       how much of the book is outside the known sectors instead of it
       being silently merged into SPY-fallback.

    This is the *portfolio exposure* path. The *signal* path
    (:func:`sector_etf_for`) keeps SPY as its single fallback so
    relative-strength comparisons still have a benchmark.
    """
    t = ticker.upper()
    if t in BROAD_MARKET_ETFS:
        return BROAD_BUCKET
    return TICKER_TO_SECTOR.get(t, OTHER_BUCKET)

