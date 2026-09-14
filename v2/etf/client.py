"""ARK Invest CSV downloader.

ARK publishes one CSV per fund, updated every trading day, at predictable
URLs on their CDN. The format is stable: a CSV preceded by a header row
and (sometimes) trailed by disclaimer/total rows we ignore.
"""

from __future__ import annotations

import csv
import io
import logging
import re
from datetime import date

import requests

from v2.etf.models import ETFHolding

logger = logging.getLogger(__name__)

# ARK fund symbol → CSV URL on assets.ark-funds.com CDN.
# Migrated 2026 from www.ark-funds.com/wp-content/uploads/funds-etf-csv/.
# Note: ARKQ's CSV URL was deprecated on assets.ark-funds.com without a
# documented replacement — fund still trades but daily CSV isn't published
# on the legacy path. Dropped from tracking until ARK exposes the new URL.
_ARK_BASE = "https://assets.ark-funds.com/fund-documents/funds-etf-csv"
_ARK_URLS: dict[str, str] = {
    "ARKK": f"{_ARK_BASE}/ARK_INNOVATION_ETF_ARKK_HOLDINGS.csv",
    "ARKW": f"{_ARK_BASE}/ARK_NEXT_GENERATION_INTERNET_ETF_ARKW_HOLDINGS.csv",
    "ARKG": f"{_ARK_BASE}/ARK_GENOMIC_REVOLUTION_ETF_ARKG_HOLDINGS.csv",
    "ARKF": f"{_ARK_BASE}/ARK_FINTECH_INNOVATION_ETF_ARKF_HOLDINGS.csv",
}

SUPPORTED_FUNDS: list[str] = list(_ARK_URLS.keys())

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0.0.0 Safari/537.36"
)

_HEADER_ALIASES: dict[str, list[str]] = {
    "date":         ["date", "as of date"],
    "fund":         ["fund"],
    "company":      ["company", "company name"],
    "ticker":       ["ticker"],
    "cusip":        ["cusip"],
    "shares":       ["shares", "share quantity"],
    "market_value": ["market value ($)", "market value($)", "market value"],
    "weight_pct":   ["weight (%)", "weight(%)", "weight"],
}


def _normalize_headers(fieldnames: list[str]) -> dict[str, str]:
    """Map canonical name → actual header in the CSV (lowercased compare)."""
    lookup: dict[str, str] = {}
    lower = {h.strip().lower(): h for h in fieldnames if h}
    for canonical, aliases in _HEADER_ALIASES.items():
        for alias in aliases:
            if alias in lower:
                lookup[canonical] = lower[alias]
                break
    return lookup


def _clean_num(s: str | None) -> float | None:
    if s is None:
        return None
    s = s.strip().replace(",", "").replace("$", "").replace("%", "")
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def fetch_holdings(symbol: str) -> tuple[list[ETFHolding], str]:
    """Fetch and parse ARK's latest CSV for *symbol*.

    Returns (holdings, snapshot_date_iso). snapshot_date_iso comes from the
    CSV's own date column if present; falls back to today.
    """
    import time as _time
    from v2.observability import emit
    symbol = symbol.upper()
    url = _ARK_URLS.get(symbol)
    if url is None:
        raise ValueError(f"Unsupported ETF: {symbol}")

    t0 = _time.perf_counter()
    resp = requests.get(url, headers={"User-Agent": _UA}, timeout=20)
    resp.raise_for_status()
    text = resp.text
    elapsed_ms = int((_time.perf_counter() - t0) * 1000)
    emit(
        "api_call",
        provider="ark_csv",
        endpoint="fetch_holdings",
        ticker=symbol,
        bytes=len(text),
        elapsed_ms=elapsed_ms,
    )

    # Strip any leading blank/comment lines until we hit the header row.
    lines = text.splitlines()
    start = 0
    for i, line in enumerate(lines):
        if "ticker" in line.lower() and "shares" in line.lower():
            start = i
            break
    cleaned = "\n".join(lines[start:])

    reader = csv.DictReader(io.StringIO(cleaned))
    if not reader.fieldnames:
        return [], date.today().isoformat()

    hmap = _normalize_headers(reader.fieldnames)
    missing = {"ticker", "shares", "market_value", "weight_pct"} - hmap.keys()
    if missing:
        logger.warning("ARK CSV %s missing columns %s, headers=%s",
                       symbol, missing, reader.fieldnames)
        return [], date.today().isoformat()

    holdings: list[ETFHolding] = []
    seen_date: str | None = None

    for row in reader:
        ticker_raw = (row.get(hmap["ticker"]) or "").strip()
        if not ticker_raw or ticker_raw.lower() in {"total", "cash", "disclaimer"}:
            continue
        # ARK uses "TICKER UW" / "TICKER UN" / "TICKER US" suffixes — strip
        ticker = re.split(r"\s+", ticker_raw)[0].upper()
        if not re.fullmatch(r"[A-Z.\-]{1,8}", ticker):
            continue

        shares = _clean_num(row.get(hmap["shares"]))
        mv = _clean_num(row.get(hmap["market_value"]))
        wt = _clean_num(row.get(hmap["weight_pct"]))
        if shares is None or mv is None or wt is None:
            continue

        date_str = (row.get(hmap.get("date", ""), "") or "").strip()
        if date_str and seen_date is None:
            seen_date = _parse_date(date_str)

        company = (row.get(hmap.get("company", ""), "") or "").strip()
        cusip = (row.get(hmap.get("cusip", ""), "") or "").strip() or None

        holdings.append(ETFHolding(
            etf=symbol,
            date=seen_date or date.today().isoformat(),
            ticker=ticker,
            cusip=cusip,
            company=company,
            shares=shares,
            market_value=mv,
            weight_pct=wt,
        ))

    final_date = seen_date or date.today().isoformat()
    return holdings, final_date


def _parse_date(raw: str) -> str:
    """Accept m/d/yyyy or yyyy-mm-dd or yyyy/mm/dd. Return ISO yyyy-mm-dd."""
    raw = raw.strip()
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y"):
        try:
            from datetime import datetime
            return datetime.strptime(raw, fmt).date().isoformat()
        except ValueError:
            continue
    return date.today().isoformat()
