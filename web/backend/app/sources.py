"""Ticker universes shared by the Lab tools.

Every Lab engine (screener, committee, backtest, event study) takes the same
question first — *which tickers?* — so the answer lives in one place:

* ``custom``               the list the user typed
* ``tech30``               the production TECH_30 monitoring pool
* ``sp500`` / ``nasdaq100`` / ``dow30``   index constituents (``v2/screening/universes.py``)
* ``holdings``             long positions in the Alpaca account
* ``watchlist``            the bot's watchlist
* ``holdings_watchlist``   union of the two

Index universes are large; only the screener accepts them (``BIG_LIMIT``).
"""

from __future__ import annotations

from typing import Any, Literal

Universe = Literal["custom", "tech30", "sp500", "nasdaq100", "dow30", "holdings", "watchlist", "holdings_watchlist"]

MAX_TICKERS = 60
#: cap for the screener, which is the only engine that may take an index
BIG_LIMIT = 600
INDEX_UNIVERSES = ("sp500", "nasdaq100", "dow30")


def normalize_tickers(values: list[str], *, limit: int = MAX_TICKERS) -> list[str]:
    out: list[str] = []
    for raw in values:
        ticker = str(raw).strip().upper()
        if not ticker or len(ticker) > 8 or not all(ch.isalpha() or ch in ".-" for ch in ticker):
            raise ValueError(f"invalid ticker: {raw!r}")
        if ticker not in out:
            out.append(ticker)
    if not out:
        raise ValueError("at least one ticker is required")
    return out[:limit]


def holdings() -> tuple[list[str], dict[str, dict[str, Any]]]:
    """Long positions from Alpaca with their portfolio weight."""
    from v2.broker.alpaca_client import get_portfolio

    pf = get_portfolio()
    total = float((pf.get("account") or {}).get("portfolio_value") or 0.0)
    positions: dict[str, dict[str, Any]] = {}
    for p in pf.get("positions") or []:
        symbol = str(p.get("symbol") or "").upper()
        # alpaca-py hands back an enum; str() of it is "PositionSide.LONG".
        side = str(p.get("side") or "long").lower().rsplit(".", 1)[-1]
        if not symbol or side == "short":
            continue
        mv = float(p.get("market_value") or 0.0)
        positions[symbol] = {
            "weight": (mv / total) if total > 0 else None,
            "market_value": mv,
            "current_price": float(p.get("current_price") or 0.0) or None,
            "unrealized_pl_pct": p.get("unrealized_pl_pct"),
        }
    return list(positions), positions


def watchlist() -> list[str]:
    from v2.bot.state import watchlist_list

    return [str(item["ticker"]).upper() for item in watchlist_list()]


def resolve_universe(universe: Universe, tickers: list[str] | None = None, *, limit: int = MAX_TICKERS) -> tuple[list[str], dict[str, Any]]:
    """Return ``(tickers, meta)``; meta carries positions for holdings-based universes."""
    meta: dict[str, Any] = {"universe": universe}
    if universe == "custom":
        return normalize_tickers(tickers or [], limit=limit), meta
    if universe == "tech30":
        from v2.screening.universe import TECH_30

        return list(TECH_30)[:limit], meta
    if universe in INDEX_UNIVERSES:
        from v2.screening.universes import load_universe

        tickers, as_of = load_universe(universe)
        if len(tickers) > limit:
            raise ValueError(f"{universe} has {len(tickers)} tickers; this tool accepts at most {limit}")
        meta["as_of"] = as_of
        return tickers, meta
    picked: list[str] = []
    if universe in ("holdings", "holdings_watchlist"):
        held, positions = holdings()
        picked.extend(held)
        meta["positions"] = positions
    if universe in ("watchlist", "holdings_watchlist"):
        for t in watchlist():
            if t not in picked:
                picked.append(t)
    if not picked:
        raise ValueError({"holdings": "no long positions in the account", "watchlist": "watchlist is empty", "holdings_watchlist": "no holdings and empty watchlist"}[universe])
    return picked[:limit], meta
