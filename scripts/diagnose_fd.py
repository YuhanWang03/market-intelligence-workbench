"""Diagnose FD API access — show raw status + body for each endpoint.

The base FDClient hides response bodies, so when you get 402 / 404 you
can't see WHY. This script bypasses the client and prints everything.

Usage:
    poetry run python scripts/diagnose_fd.py
"""

from __future__ import annotations

import os

import requests
from dotenv import load_dotenv

load_dotenv()

KEY = os.environ.get("FINANCIAL_DATASETS_API_KEY", "")
BASE = "https://api.financialdatasets.ai"
HEADERS = {"X-API-Key": KEY}


def check(label: str, path: str, params: dict) -> None:
    print(f"\n── {label}")
    print(f"   GET {path}  params={params}")
    try:
        r = requests.get(BASE + path, params=params, headers=HEADERS, timeout=15)
    except requests.RequestException as exc:
        print(f"   ERROR: {exc}")
        return
    print(f"   status: {r.status_code}")
    body = r.text.strip()
    if len(body) > 400:
        body = body[:400] + "..."
    print(f"   body:   {body}")


def main() -> None:
    print(f"API key: {KEY[:8]}...{KEY[-4:] if len(KEY) > 12 else ''}  (len={len(KEY)})")
    if not KEY:
        print("⚠️  No FINANCIAL_DATASETS_API_KEY in env!")
        return

    # Probably-free endpoint
    check("company facts (AAPL)", "/company/facts/", {"ticker": "AAPL"})

    # Prices — used by PEAD which worked before
    check("prices (AAPL, 1 month)", "/prices/", {
        "ticker": "AAPL",
        "interval": "day",
        "interval_multiplier": 1,
        "start_date": "2024-12-01",
        "end_date": "2024-12-31",
    })

    # Financial metrics — failed in screening
    check("financial-metrics (AAPL)", "/financial-metrics/", {
        "ticker": "AAPL",
        "period": "ttm",
        "limit": 1,
    })

    # Earnings — used by PEAD which worked before
    check("earnings (AAPL)", "/earnings/", {"ticker": "AAPL", "limit": 1})

    # A non-mega-cap ticker — to see if access is ticker-restricted
    check("prices (PLTR)", "/prices/", {
        "ticker": "PLTR",
        "interval": "day",
        "interval_multiplier": 1,
        "start_date": "2024-12-01",
        "end_date": "2024-12-31",
    })


if __name__ == "__main__":
    main()
