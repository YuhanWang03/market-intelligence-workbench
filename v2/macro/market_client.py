"""yfinance wrapper for the ⑭ daily snapshot — Phase 4 Stage 1.

Pulls intraday market levels for VIX / DXY / WTI / Gold. Each fetch
is independent: a single yfinance failure on, say, DXY won't break
the whole snapshot — the caller logs a warning and proceeds with the
other fields. The pipeline aggregates warnings into the report.

⚠️ DO NOT USE FROM YAHOO for Treasury yields:
    ``^TNX`` (10Y), ``^FVX`` (5Y), ``^TYX`` (30Y) return raw values
    that are 10× the actual yield (e.g. 42.5 instead of 4.25%). Yahoo
    documents this but the bug burns every new user. ``^IRX`` (13-week
    T-bill) is also not 2Y. All Treasury yields go through FRED
    (DGS2 / DGS10 / T10Y2Y / T10Y3M) via :mod:`v2.macro.fred_client`.
"""

from __future__ import annotations

import logging
from typing import Callable

try:
    import yfinance as yf
except ImportError:                                  # sandbox safety
    yf = None                                        # type: ignore[assignment]


logger = logging.getLogger(__name__)


VIX_SYMBOL = "^VIX"
DXY_SYMBOL = "DX-Y.NYB"
WTI_SYMBOL = "CL=F"
GOLD_SYMBOL = "GC=F"


def _ticker_factory(symbol: str):
    """Default Ticker builder. Tests override via ``ticker_factory=`` kwarg."""
    if yf is None:
        raise RuntimeError("yfinance package not installed")
    return yf.Ticker(symbol)


def _safe_quote(symbol: str, *, ticker_factory: Callable | None = None) -> dict | None:
    """Return a dict with the latest close + 1d pct change, or None on
    any failure. Never raises — sub-failures aggregate at the caller."""
    factory = ticker_factory or _ticker_factory
    try:
        t = factory(symbol)
        hist = t.history(period="5d", interval="1d")
        if hist is None or len(hist) == 0:
            return None
        # Latest close
        last = float(hist["Close"].iloc[-1])
        # 1-day pct change if at least 2 closes
        if len(hist) >= 2:
            prev = float(hist["Close"].iloc[-2])
            pct = (last / prev - 1.0) if prev else None
        else:
            pct = None
        return {"value": last, "pct_change_1d": pct}
    except Exception as exc:                          # noqa: BLE001
        logger.info("yfinance fetch failed for %s: %s", symbol, exc)
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_vix_intraday(
    *, ticker_factory: Callable | None = None,
) -> dict | None:
    """Return ``{"value": float, "pct_change_1d": float | None}``.

    Used by ⑭. We compare against FRED ``VIXCLS`` overnight (the
    canonical EOD print) — if the two disagree by more than 0.1
    we log a warning but do not block the push.
    """
    return _safe_quote(VIX_SYMBOL, ticker_factory=ticker_factory)


def get_dxy_spot(*, ticker_factory: Callable | None = None) -> float | None:
    q = _safe_quote(DXY_SYMBOL, ticker_factory=ticker_factory)
    return q["value"] if q else None


def get_wti_crude(*, ticker_factory: Callable | None = None) -> float | None:
    q = _safe_quote(WTI_SYMBOL, ticker_factory=ticker_factory)
    return q["value"] if q else None


def get_gold_spot(*, ticker_factory: Callable | None = None) -> float | None:
    q = _safe_quote(GOLD_SYMBOL, ticker_factory=ticker_factory)
    return q["value"] if q else None


__all__ = [
    "VIX_SYMBOL",
    "DXY_SYMBOL",
    "WTI_SYMBOL",
    "GOLD_SYMBOL",
    "get_vix_intraday",
    "get_dxy_spot",
    "get_wti_crude",
    "get_gold_spot",
]
