"""Structured dashboard endpoints — risk report, macro snapshot, equity
history, and per-ticker money-flow status. Each calls a v2 module directly
and serializes to plain JSON. Independent by design: one failing endpoint
never blanks the others (matches v2's degrade-gracefully philosophy).
"""

from __future__ import annotations

import math
from datetime import date, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.concurrency import run_in_threadpool

from app.auth import require_owner

router = APIRouter(prefix="/api", tags=["dashboard"])

_VALID_PERIODS = {"1M", "3M", "1A"}


# --------------------------------------------------------------------------
# Risk report — powers the KPI strip (multi-period P&L) + risk card.
# --------------------------------------------------------------------------

def _fetch_risk() -> dict:
    from v2.portfolio.pipeline import build_risk_report

    r = build_risk_report()
    return {
        "portfolio_value": r.portfolio_value,
        "cash": r.cash,
        "cash_pct": r.cash_pct,
        "invested_value": r.invested_value,
        "pnl": {
            "daily_pnl": r.pnl.daily_pnl,
            "daily_pnl_pct": r.pnl.daily_pnl_pct,
            "weekly_pnl_pct": r.pnl.weekly_pnl_pct,
            "monthly_pnl_pct": r.pnl.monthly_pnl_pct,
        },
        "concentration": {
            "top_1_pct": r.concentration.top_1_pct,
            "top_3_pct": r.concentration.top_3_pct,
            "hhi": r.concentration.hhi,
            "n_positions": r.concentration.n_positions,
        },
        "exposure": {
            "by_sector": r.exposure.by_sector,
            "largest_sector": r.exposure.largest_sector,
            "largest_sector_pct": r.exposure.largest_sector_pct,
        },
        "drawdown": {
            "current_drawdown_pct": r.drawdown.current_drawdown_pct,
            "max_drawdown_pct": r.drawdown.max_drawdown_pct,
        },
        "earnings_risk": [
            {"ticker": e.ticker, "release_date": e.release_date, "days_until": e.days_until}
            for e in r.earnings_risk_next_7d
        ],
        "warnings": r.warnings,
    }


@router.get("/risk", dependencies=[Depends(require_owner)])
async def risk() -> dict:
    try:
        return await run_in_threadpool(_fetch_risk)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))


# --------------------------------------------------------------------------
# Macro snapshot — VIX / yields / DXY market-context strip.
# --------------------------------------------------------------------------

def _fetch_macro() -> dict:
    from v2.macro.pipeline import build_macro_snapshot

    s = build_macro_snapshot(date.today().isoformat())
    return {
        "vix": s.vix,
        "vix_pct_change_1d": s.vix_pct_change_1d,
        "dxy": s.dxy,
        "wti_crude": s.wti_crude,
        "gold": s.gold,
        "dgs2": s.dgs2,
        "dgs10": s.dgs10,
        "t10y2y": s.t10y2y,
        "fed_funds_upper": s.fed_funds_upper,
        "vix_spike": s.vix_spike,
        "curve_flip": s.curve_flip,
        "rates_shocked": s.rates_shocked,
        "warnings": s.warnings,
    }


@router.get("/macro", dependencies=[Depends(require_owner)])
async def macro() -> dict:
    try:
        return await run_in_threadpool(_fetch_macro)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))


# --------------------------------------------------------------------------
# Equity curve — for the 1M / 3M / 1Y toggle.
# --------------------------------------------------------------------------

@router.get("/history", dependencies=[Depends(require_owner)])
async def history(period: str = Query("1M")) -> dict:
    if period not in _VALID_PERIODS:
        period = "1M"

    def _fetch() -> dict:
        from v2.broker.alpaca_client import get_portfolio_history

        h = get_portfolio_history(period=period, timeframe="1D")
        return {"period": period, "timestamp": h["timestamp"], "equity": h["equity"]}

    try:
        return await run_in_threadpool(_fetch)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))


# --------------------------------------------------------------------------
# Money-flow status per ticker — position badges (accumulation/distribution).
# --------------------------------------------------------------------------

def _fetch_flow(tickers: list[str]) -> dict:
    from v2.data import CachedFDClient
    from v2.moneyflow import DEFAULT_CONFIG, detect_divergence, read_axes

    today = date.today()
    start = (today - timedelta(days=120)).isoformat()
    out: dict[str, dict] = {}
    with CachedFDClient() as fd:
        for t in tickers:
            try:
                prices = fd.get_prices(t, start, today.isoformat())
                reading = read_axes(t, prices, DEFAULT_CONFIG)
                sig = detect_divergence(t, prices, DEFAULT_CONFIG)
            except Exception:
                out[t] = {"state": "unknown", "strength": None, "cmf": None, "rsi": None}
                continue
            out[t] = {
                "state": sig.kind if sig else "none",
                "strength": sig.strength if sig else None,
                "cmf": reading.cmf if reading else None,
                "rsi": reading.rsi if reading else None,
            }
    return out


@router.get("/flow_status", dependencies=[Depends(require_owner)])
async def flow_status(tickers: str = Query("")) -> dict:
    syms = [t.strip().upper() for t in tickers.split(",") if t.strip()][:20]
    if not syms:
        return {}
    try:
        return await run_in_threadpool(_fetch_flow, syms)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))


# --------------------------------------------------------------------------
# Ticker tape — a curated set of market levels for the scrolling strip.
# --------------------------------------------------------------------------

# yfinance symbols. Treasury yields deliberately excluded here (Yahoo returns
# 10× wrong values) — Treasury series are appended from FRED below.
_TAPE_SYMBOLS = [
    ("sp500", "SPY", "标普500"),
    ("nasdaq100", "QQQ", "纳指100"),
    ("dow", "DIA", "道指"),
    ("russell2000", "IWM", "罗素2000"),
    ("semiconductor_etf", "SOXX", "半导体ETF"),
    ("philadelphia_semiconductor", "^SOX", "费城半导体指数"),
    ("vix", "^VIX", "VIX"),
    ("gold", "GC=F", "黄金"),
    ("silver", "SI=F", "白银"),
    ("wti", "CL=F", "WTI原油"),
    ("brent", "BZ=F", "布伦特原油"),
    ("dollar", "DX-Y.NYB", "美元"),
    ("bitcoin", "BTC-USD", "比特币"),
]

_TAPE_FRED_SERIES = [
    ("us5y", "DGS5", "美债5Y"),
    ("us10y", "DGS10", "美债10Y"),
]


def _finite_number(value) -> float | None:
    """Normalize provider scalars and reject NaN/Infinity before JSON output."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _fetch_tape() -> dict:
    from v2.macro.market_client import _safe_quote

    items: list[dict] = []
    for key, sym, label in _TAPE_SYMBOLS:
        q = _safe_quote(sym)
        value = _finite_number(q.get("value")) if q else None
        items.append({
            "key": key,
            "label": label,
            "value": value,
            "change_pct": _finite_number(q.get("pct_change_1d")) if q else None,
            "unit": "",
        })

    try:
        from v2.macro.fred_client import get_latest_value
    except Exception:
        get_latest_value = None
    for key, series, label in _TAPE_FRED_SERIES:
        try:
            value = _finite_number(get_latest_value(series)) if get_latest_value else None
        except Exception:
            value = None
        items.append({
            "key": key,
            "label": label,
            "value": value,
            "change_pct": None,
            "unit": "%",
        })

    return {"items": items}


@router.get("/tickertape", dependencies=[Depends(require_owner)])
async def tickertape() -> dict:
    try:
        return await run_in_threadpool(_fetch_tape)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))


# --------------------------------------------------------------------------
# Recommendations — TECH_30 money-flow accumulation scan (吸筹榜). Honest
# labeling: these show accumulation signals, not a buy call.
# --------------------------------------------------------------------------

def _fetch_recommendations() -> dict:
    from v2.data import CachedFDClient
    from v2.moneyflow import DEFAULT_CONFIG, detect_divergence
    from v2.screening import TECH_30

    today = date.today()
    start = (today - timedelta(days=120)).isoformat()
    out: list[dict] = []
    with CachedFDClient() as fd:
        for t in TECH_30:
            try:
                prices = fd.get_prices(t, start, today.isoformat())
                sig = detect_divergence(t, prices, DEFAULT_CONFIG)
            except Exception:
                continue
            if sig and sig.kind == "accumulation":
                out.append({
                    "ticker": t, "strength": sig.strength,
                    "cmf": sig.cmf, "rsi": sig.rsi,
                    "rsi_divergence": sig.rsi_divergence,
                })
    # strong first, then by money-flow strength (CMF desc)
    out.sort(key=lambda x: (x["strength"] != "strong", -(x["cmf"] or 0)))
    return {"items": out}


@router.get("/recommendations", dependencies=[Depends(require_owner)])
async def recommendations() -> dict:
    try:
        return await run_in_threadpool(_fetch_recommendations)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))
