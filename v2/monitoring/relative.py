"""Relative-strength computation: ticker vs its sector ETF.

Pure functions — given pre-fetched price series, return a dict ready to
plug into the Anomaly model. Pre-fetching ETF prices once per run keeps
the FD call budget bounded (3 extra calls for the whole universe).
"""

from __future__ import annotations

from v2.data.models import Price

# Threshold for the "contrarian" flag. With ticker_return and sector_return:
#   - both ≥0 or both ≤0: same direction → not contrarian
#   - signs differ AND |gap| ≥ this many percentage points: contrarian
_CONTRARIAN_GAP_PP = 0.015  # 1.5 percentage points


def _last_return(prices: list[Price]) -> float | None:
    if not prices or len(prices) < 2:
        return None
    last, prev = prices[-1].close, prices[-2].close
    if prev <= 0:
        return None
    return (last - prev) / prev


def _window_return(prices: list[Price], window: int) -> float | None:
    if not prices or len(prices) < window + 1:
        return None
    last, base = prices[-1].close, prices[-1 - window].close
    if base <= 0:
        return None
    return (last - base) / base


def compute_relative(
    ticker_prices: list[Price],
    etf_prices: list[Price] | None,
) -> dict:
    """Return a dict with relative-strength fields. Always returns a dict —
    fields set to None when the comparison is not possible.

    Schema:
        ticker_return_1d   — 1-day return for the ticker
        sector_return_1d   — 1-day return for the sector ETF
        relative_1d_pp     — (ticker - sector) in percentage points
        contrarian         — bool: opposing signs AND |gap| ≥ 1.5pp
        direction          — "leading", "lagging", "aligned", "unknown"
    """
    t1 = _last_return(ticker_prices)
    e1 = _last_return(etf_prices) if etf_prices else None

    if t1 is None or e1 is None:
        return {
            "ticker_return_1d": t1,
            "sector_return_1d": e1,
            "relative_1d_pp":   None,
            "contrarian":       False,
            "direction":        "unknown",
        }

    rel = t1 - e1
    same_sign = (t1 >= 0) == (e1 >= 0)
    contrarian = (not same_sign) and abs(rel) >= _CONTRARIAN_GAP_PP

    if contrarian:
        direction = "leading" if t1 > 0 else "lagging"
    elif abs(rel) >= _CONTRARIAN_GAP_PP:
        direction = "leading" if rel > 0 else "lagging"
    else:
        direction = "aligned"

    return {
        "ticker_return_1d": t1,
        "sector_return_1d": e1,
        "relative_1d_pp":   rel,
        "contrarian":       contrarian,
        "direction":        direction,
    }


def compute_relative_window(
    ticker_prices: list[Price],
    etf_prices: list[Price] | None,
    *,
    window: int = 5,
) -> dict:
    """Same as compute_relative but over a multi-day window — used by the
    screener narrator for medium-horizon context.
    """
    t = _window_return(ticker_prices, window)
    e = _window_return(etf_prices, window) if etf_prices else None

    if t is None or e is None:
        return {
            "ticker_return": t,
            "sector_return": e,
            "relative_pp":   None,
            "window":        window,
        }

    return {
        "ticker_return": t,
        "sector_return": e,
        "relative_pp":   t - e,
        "window":        window,
    }
