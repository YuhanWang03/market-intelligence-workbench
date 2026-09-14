"""FRED series catalog — Phase 4 Stage 1.

22 series across inflation / labor / growth / rates / markets /
reference. Each entry pins:

- ``name``: human-readable label for the card
- ``freq``: D / W / M / Q (mirrors FRED's frequency)
- ``transform``: which :mod:`v2.macro.transforms` function to apply

The ⭐ markers in comments call out the "headline" series — the ones
the daily snapshot card surfaces first.

Series IDs verified against FRED catalog 2026-06; see
https://fred.stlouisfed.org/ for each ID.
"""

from __future__ import annotations


FRED_SERIES: dict[str, dict] = {
    # ---- Inflation ----
    "CPIAUCSL":      {"name": "CPI Headline",        "freq": "M", "transform": "mom_yoy"},
    "CPILFESL":      {"name": "CPI Core",            "freq": "M", "transform": "mom_yoy"},
    "PCEPI":         {"name": "PCE Headline",        "freq": "M", "transform": "mom_yoy"},
    "PCEPILFE":      {"name": "PCE Core ⭐",          "freq": "M", "transform": "mom_yoy"},
    "PPIFIS":        {"name": "PPI Final",           "freq": "M", "transform": "mom_yoy"},

    # ---- Labor ----
    "PAYEMS":        {"name": "NFP",                 "freq": "M", "transform": "mom_change_k"},
    "UNRATE":        {"name": "Unemployment",        "freq": "M", "transform": "level_pct"},
    "ICSA":          {"name": "Initial Claims",      "freq": "W", "transform": "level_4wma"},
    "CES0500000003": {"name": "Avg Hourly Earnings", "freq": "M", "transform": "yoy"},

    # ---- Growth ----
    "GDPC1":         {"name": "Real GDP",            "freq": "Q", "transform": "qoq_annualized"},
    "INDPRO":        {"name": "Industrial Prod",     "freq": "M", "transform": "mom"},
    "RSAFS":         {"name": "Retail Sales",        "freq": "M", "transform": "mom"},

    # ---- Rates (Treasury + Fed Funds) ----
    "DFEDTARU":      {"name": "Fed Funds Upper",     "freq": "D", "transform": "level"},
    "DFEDTARL":      {"name": "Fed Funds Lower",     "freq": "D", "transform": "level"},
    "DGS2":          {"name": "2Y Treasury",         "freq": "D", "transform": "level"},
    "DGS10":         {"name": "10Y Treasury",        "freq": "D", "transform": "level"},
    "T10Y2Y":        {"name": "10Y-2Y Spread ⭐",     "freq": "D", "transform": "level"},
    "T10Y3M":        {"name": "10Y-3M Spread",       "freq": "D", "transform": "level"},

    # ---- Markets ----
    "VIXCLS":        {"name": "VIX Close",           "freq": "D", "transform": "level"},

    # ---- Reference (money + FX) ----
    "M2SL":          {"name": "M2 Money Supply",     "freq": "M", "transform": "yoy"},
    "DEXUSEU":       {"name": "USD/EUR",             "freq": "D", "transform": "level"},
    "DEXCHUS":       {"name": "USD/CNY",             "freq": "D", "transform": "level"},
}


# Which FRED series carry each release type. The pipeline pulls all
# series for a release and routes them to the same MacroRelease.
RELEASE_TO_SERIES: dict[str, list[str]] = {
    "CPI":    ["CPIAUCSL", "CPILFESL"],
    "PCE":    ["PCEPI", "PCEPILFE"],
    "NFP":    ["PAYEMS", "UNRATE", "CES0500000003"],
    "GDP":    ["GDPC1"],
    "PPI":    ["PPIFIS"],
    "Claims": ["ICSA"],
}


# Series IDs that the ⑭ daily snapshot needs from FRED. (Markets come
# from yfinance — see v2.macro.market_client.)
SNAPSHOT_FRED_SERIES: tuple[str, ...] = (
    "DFEDTARU", "DFEDTARL", "DGS2", "DGS10", "T10Y2Y", "VIXCLS",
)


__all__ = [
    "FRED_SERIES",
    "RELEASE_TO_SERIES",
    "SNAPSHOT_FRED_SERIES",
]
