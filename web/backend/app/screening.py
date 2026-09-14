"""Lab screener — the same inputs as ``v2/screening`` with pick-and-mix criteria.

The production screener hard-codes five thresholds and fails a ticker that is
missing any of them. The Lab needs criteria the user can switch on and off
individually, and a few more of them, so the per-ticker snapshot is built
here (metrics row + daily prices, exactly the way ``build_candidate`` does)
and filtered against an explicit list of ``(field, op, value)`` rules.

A rule only applies when it is in the list; a ticker missing that field fails
that rule and only that rule. Fields come from the metrics row (FD or the
yfinance shim) plus price-derived statistics computed from the closes.
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Any, Callable, Literal

import numpy as np
from pydantic import BaseModel, Field

Op = Literal["gte", "lte"]

#: fields a rule may reference: label, unit, where it comes from
CRITERIA: dict[str, dict[str, str]] = {
    "market_cap": {"label": "市值", "unit": "usd", "source": "metrics"},
    "price": {"label": "股价", "unit": "usd", "source": "prices"},
    "revenue_growth": {"label": "营收增长", "unit": "pct", "source": "metrics"},
    "earnings_growth": {"label": "盈利增长", "unit": "pct", "source": "metrics"},
    "gross_margin": {"label": "毛利率", "unit": "pct", "source": "metrics"},
    "operating_margin": {"label": "营业利润率", "unit": "pct", "source": "metrics"},
    "net_margin": {"label": "净利率", "unit": "pct", "source": "metrics"},
    "return_on_equity": {"label": "ROE", "unit": "pct", "source": "metrics"},
    "return_on_invested_capital": {"label": "ROIC", "unit": "pct", "source": "metrics"},
    "debt_to_equity": {"label": "负债/权益", "unit": "x", "source": "metrics"},
    "current_ratio": {"label": "流动比率", "unit": "x", "source": "metrics"},
    "price_to_earnings_ratio": {"label": "市盈率", "unit": "x", "source": "metrics"},
    "price_to_sales_ratio": {"label": "市销率", "unit": "x", "source": "metrics"},
    "price_to_book_ratio": {"label": "市净率", "unit": "x", "source": "metrics"},
    "free_cash_flow_yield": {"label": "自由现金流收益率", "unit": "pct", "source": "metrics"},
    "payout_ratio": {"label": "派息率", "unit": "pct", "source": "metrics"},
    "volatility": {"label": "年化波动率", "unit": "pct", "source": "prices"},
    "return_1w": {"label": "1 周涨跌", "unit": "pct", "source": "prices"},
    "return_1m": {"label": "1 月涨跌", "unit": "pct", "source": "prices"},
    "return_3m": {"label": "3 月涨跌", "unit": "pct", "source": "prices"},
    "pct_from_52w_high": {"label": "距 52 周高点", "unit": "pct", "source": "prices"},
    "pct_from_52w_low": {"label": "距 52 周低点", "unit": "pct", "source": "prices"},
}

_METRIC_FIELDS = [k for k, v in CRITERIA.items() if v["source"] == "metrics"]
_MIN_CLOSES = 60
_TRADING_DAYS = 252


class Rule(BaseModel):
    field: str
    op: Op
    value: float

    def passes(self, row: dict[str, Any]) -> bool:
        v = row.get(self.field)
        if v is None or not isinstance(v, (int, float)) or v != v:
            return False
        return v >= self.value if self.op == "gte" else v <= self.value

    def describe(self) -> str:
        meta = CRITERIA.get(self.field, {"label": self.field, "unit": ""})
        sym = "≥" if self.op == "gte" else "≤"
        if meta["unit"] == "pct":
            val = f"{self.value * 100:g}%"
        elif meta["unit"] == "usd":
            val = f"${self.value / 1e9:g}B" if self.field == "market_cap" else f"${self.value:g}"
        else:
            val = f"{self.value:g}"
        return f"{meta['label']} {sym} {val}"


DEFAULT_RULES: list[Rule] = [
    Rule(field="market_cap", op="gte", value=10e9),
    Rule(field="revenue_growth", op="gte", value=0.05),
    Rule(field="gross_margin", op="gte", value=0.50),
    Rule(field="volatility", op="lte", value=0.60),
]


def _num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if (math.isnan(f) or math.isinf(f)) else f


def _field(obj: Any, name: str) -> Any:
    return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)


def price_stats(prices: list[Any]) -> dict[str, float | None] | None:
    """Volatility, returns and 52-week distances from daily closes (oldest first)."""
    closes = np.array([c for c in (_num(_field(p, "close")) for p in prices) if c is not None and c > 0], dtype=float)
    if len(closes) < _MIN_CLOSES:
        return None
    latest = float(closes[-1])
    ret = lambda n: float(closes[-1] / closes[-1 - n] - 1) if len(closes) > n and closes[-1 - n] > 0 else None  # noqa: E731
    window = closes[-_TRADING_DAYS:]
    log_returns = np.clip(np.diff(np.log(closes)), -0.25, 0.25)
    return {
        "price": latest,
        "price_change": ret(1),
        "return_1w": ret(5),
        "return_1m": ret(21),
        "return_3m": ret(63),
        "volatility": float(log_returns.std(ddof=1) * math.sqrt(_TRADING_DAYS)),
        "high_52w": float(window.max()),
        "low_52w": float(window.min()),
        "pct_from_52w_high": float(latest / window.max() - 1),
        "pct_from_52w_low": float(latest / window.min() - 1),
    }


def build_row(ticker: str, metrics_row: Any, prices: list[Any]) -> dict[str, Any] | None:
    stats = price_stats(prices)
    if stats is None:
        return None
    row: dict[str, Any] = {"ticker": ticker, **stats}
    for name in _METRIC_FIELDS:
        row[name] = _num(_field(metrics_row, name)) if metrics_row is not None else None
    return row


def screen(
    tickers: list[str],
    client: Any,
    price_source: Any,
    rules: list[Rule],
    *,
    on_tick: Callable[[int], None] | None = None,
    with_earnings: bool = False,
    today: date | None = None,
) -> dict[str, Any]:
    """Run every rule over every ticker. Returns candidates, rejects, and counts."""
    today = today or date.today()
    today_str = today.isoformat()
    history_start = (today - timedelta(days=400)).isoformat()
    candidates: list[dict[str, Any]] = []
    rejected: dict[str, list[str]] = {}
    no_data: list[str] = []

    for i, ticker in enumerate(tickers):
        if on_tick:
            on_tick(i)
        metrics = client.get_financial_metrics(ticker, today_str, limit=1) or []
        prices = price_source.get_prices(ticker, history_start, today_str) or []
        row = build_row(ticker, metrics[0] if metrics else None, prices)
        if row is None:
            no_data.append(ticker)
            continue
        failed = [r.describe() for r in rules if not r.passes(row)]
        if failed:
            rejected[ticker] = failed
            continue
        candidates.append(row)

    if with_earnings and candidates:
        from v2.screening.models import ScreenCandidate
        from v2.screening.screener import enrich_with_earnings

        for row in candidates:
            c = ScreenCandidate(ticker=row["ticker"], price=row["price"], price_change=row.get("price_change"), market_cap=row.get("market_cap"),
                                revenue_growth=row.get("revenue_growth"), gross_margin=row.get("gross_margin"), volatility=row.get("volatility"),
                                high_52w=row.get("high_52w"), return_1w=row.get("return_1w"))
            try:
                enrich_with_earnings(c, client)
            except Exception:  # noqa: BLE001 — enrichment is optional
                continue
            for name in ("revenue_actual", "revenue_estimate", "revenue_surprise_pct", "eps_actual", "eps_estimate", "eps_surprise_pct"):
                row[name] = getattr(c, name, None)

    candidates.sort(key=lambda r: -(r.get("market_cap") or 0))
    return {
        "date": today_str,
        "universe_size": len(tickers),
        "rules": [r.model_dump() for r in rules],
        "rules_text": [r.describe() for r in rules],
        "candidates": candidates,
        "rejected_count": len(rejected),
        "no_data": no_data,
        "reject_reasons": _reason_counts(rejected),
    }


def _reason_counts(rejected: dict[str, list[str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for reasons in rejected.values():
        for r in reasons:
            counts[r] = counts.get(r, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))
