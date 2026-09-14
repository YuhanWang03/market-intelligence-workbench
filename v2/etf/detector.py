"""Diff today's ETF snapshot vs yesterday's → ETFChange list."""

from __future__ import annotations

from v2.etf.models import ETFChange, ETFHolding
from v2.observability import emit

# Filter out micro-moves: at least this much of yesterday's share count to flag
_REBALANCE_THRESHOLD = 0.01  # 1% share change


def _by_ticker_today(holdings: list[ETFHolding]) -> dict[str, ETFHolding]:
    return {h.ticker: h for h in holdings if h.ticker}


def _by_ticker_yest(rows: list[dict]) -> dict[str, dict]:
    return {r["ticker"]: r for r in rows if r.get("ticker")}


def compute_daily_changes(
    yesterday_rows: list[dict],
    today_holdings: list[ETFHolding],
) -> list[dict]:
    """Return change dicts ready for format_etf_snapshot.

    Returns dicts (not ETFChange objects) since the formatter only needs a
    bag of fields. Keeps the formatter layer schema-light.
    """
    today_map = _by_ticker_today(today_holdings)
    yest_map = _by_ticker_yest(yesterday_rows)

    if not today_map or not yest_map:
        return []

    changes: list[dict] = []

    # Today's positions
    for ticker, t in today_map.items():
        y = yest_map.get(ticker)
        if y is None:
            changes.append({
                "etf": t.etf,
                "ticker": ticker,
                "company": t.company,
                "shares_diff": float(t.shares),
                "shares_diff_pct": 0.0,
                "weight_pct": t.weight_pct,
                "weight_diff_pp": t.weight_pct,
                "is_new": True,
                "is_exit": False,
            })
            continue

        diff = t.shares - y["shares"]
        if y["shares"] <= 0:
            continue
        pct = diff / y["shares"]
        if abs(pct) < _REBALANCE_THRESHOLD:
            continue
        changes.append({
            "etf": t.etf,
            "ticker": ticker,
            "company": t.company,
            "shares_diff": float(diff),
            "shares_diff_pct": pct,
            "weight_pct": t.weight_pct,
            "weight_diff_pp": t.weight_pct - y["weight_pct"],
            "is_new": False,
            "is_exit": False,
        })

    # Yesterday-only = exits
    for ticker, y in yest_map.items():
        if ticker not in today_map:
            changes.append({
                "etf": y["etf"],
                "ticker": ticker,
                "company": y.get("company") or "",
                "shares_diff": -float(y["shares"]),
                "shares_diff_pct": -1.0,
                "weight_pct": 0.0,
                "weight_diff_pp": -float(y["weight_pct"]),
                "is_new": False,
                "is_exit": True,
            })

    # Sort: new/exits first by absolute share impact, then rebalances by |pct|
    changes.sort(
        key=lambda c: (
            not (c["is_new"] or c["is_exit"]),
            -abs(c.get("shares_diff_pct", 0.0)),
        ),
    )
    etf_symbol = today_holdings[0].etf if today_holdings else "?"
    emit(
        "transform",
        op="etf_diff",
        etf=etf_symbol,
        today_positions=len(today_map),
        yesterday_positions=len(yest_map),
        significant_changes=len(changes),
    )
    return changes
