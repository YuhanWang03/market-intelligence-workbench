"""Per-request prices for financialdatasets.ai, used only to *estimate* spend.

Defaults are what the account's Billing → Endpoints page shows on the credits
plan: every stock endpoint $0.02 / request (checked 2026-09). Override any of
them with a JSON object in ``FD_PRICES``, e.g. ``FD_PRICES='{"news":0.04}'``,
should the published prices change.
"""

from __future__ import annotations

import json
import os

DEFAULT_PRICES_USD: dict[str, float] = {
    # financialdatasets.ai Billing → Endpoints: every stock endpoint is
    # $0.02 / request on the credits plan (verified on the account page, 2026-09).
    "financial_metrics": 0.02,
    "line_items": 0.02,
    "prices": 0.02,
    "earnings": 0.02,
    "insider_trades": 0.02,
    "news": 0.02,
    "company_facts": 0.02,
    "filings": 0.02,
}


def prices() -> dict[str, float]:
    table = dict(DEFAULT_PRICES_USD)
    raw = os.environ.get("FD_PRICES", "").strip()
    if raw:
        try:
            for k, v in json.loads(raw).items():
                if k in table:
                    table[k] = float(v)
        except (ValueError, TypeError, AttributeError):
            pass
    return table


def cost(counts: dict[str, int]) -> float:
    table = prices()
    return round(sum(table.get(k, 0.02) * n for k, n in counts.items()), 4)
