"""Portfolio snapshots and a consistent, timeframe-aware OHLCV surface."""

from __future__ import annotations

import copy
import math
import re
import threading
import time
from datetime import datetime, time as wall_time, timedelta
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.concurrency import run_in_threadpool

from app.auth import require_owner
from app.market_analysis import build_technical_analysis, enrich_bars

router = APIRouter(prefix="/api", tags=["portfolio"])
_TICKER_PATTERN = re.compile(r"^[A-Z][A-Z0-9.-]{0,9}$")
_ET = ZoneInfo("America/New_York")
_SOURCE = "YAHOO_FINANCE"
_ADJUSTMENT_MODE = "SPLIT_ADJUSTED"
_RETURN_ADJUSTMENT_MODE = "TOTAL_RETURN_ADJUSTED_CLOSE"
_PRICE_RANGES = {
    "1W": {"days": 7, "timeframe": "30m", "warmup_days": 45},
    "1M": {"days": 31, "timeframe": "1h", "warmup_days": 150},
    "3M": {"days": 93, "timeframe": "1d", "warmup_days": 400},
    "6M": {"days": 186, "timeframe": "1d", "warmup_days": 400},
    "1Y": {"days": 366, "timeframe": "1d", "warmup_days": 400},
    "3Y": {"days": 1096, "timeframe": "1w", "warmup_days": 1900},
}
_CACHE_LOCK = threading.Lock()
_PRICE_CACHE: dict[tuple[str, str, bool], tuple[float, dict]] = {}


def _fetch() -> dict:
    from v2.broker.alpaca_client import get_pnl, get_portfolio, get_portfolio_history

    pf = get_portfolio()
    pnl = get_pnl()
    try:
        hist = get_portfolio_history(period="1M", timeframe="1D")
        history = {"timestamp": hist["timestamp"], "equity": hist["equity"]}
    except Exception:
        history = {"timestamp": [], "equity": []}
    return {
        "account": pf["account"],
        "positions": pf["positions"],
        "pnl": pnl,
        "history": history,
        "quoteSource": "ALPACA_POSITION_SNAPSHOT",
    }


@router.get("/portfolio", dependencies=[Depends(require_owner)])
async def portfolio() -> dict:
    try:
        return await run_in_threadpool(_fetch)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _market_session(now_et: datetime, latest_quote_at: datetime | None) -> str:
    if latest_quote_at is None or latest_quote_at.date() != now_et.date() or now_et.weekday() >= 5:
        return "CLOSED"
    current = now_et.time()
    if wall_time(4, 0) <= current < wall_time(9, 30):
        return "PRE_MARKET"
    if wall_time(9, 30) <= current < wall_time(16, 0):
        return "REGULAR"
    if wall_time(16, 0) <= current < wall_time(20, 0):
        return "AFTER_HOURS"
    return "CLOSED"


def _bar_session(timestamp: datetime, timeframe: str) -> str:
    if timeframe in {"1d", "1w"}:
        return "REGULAR"
    current = timestamp.astimezone(_ET).time()
    if current < wall_time(9, 30):
        return "PRE_MARKET"
    if current >= wall_time(16, 0):
        return "AFTER_HOURS"
    return "REGULAR"


def _bar_is_final(timestamp: datetime, timeframe: str, now_et: datetime) -> bool:
    local = timestamp.astimezone(_ET)
    if timeframe == "30m":
        bar_end = local + timedelta(minutes=30)
        if _bar_session(local, timeframe) == "REGULAR":
            bar_end = min(bar_end, local.replace(hour=16, minute=0, second=0, microsecond=0))
        return bar_end <= now_et
    if timeframe == "1h":
        bar_end = local + timedelta(hours=1)
        if _bar_session(local, timeframe) == "REGULAR":
            # Yahoo's last hourly regular-session bar starts at 15:30 ET and
            # closes at the official 16:00 boundary, not at 16:30.
            bar_end = min(bar_end, local.replace(hour=16, minute=0, second=0, microsecond=0))
        return bar_end <= now_et
    if timeframe == "1d":
        return local.date() < now_et.date() or (
            local.date() == now_et.date() and now_et.time() >= wall_time(16, 0)
        )
    week_end = local.date() + timedelta(days=max(4 - local.weekday(), 0))
    return week_end < now_et.date() or (
        week_end == now_et.date() and now_et.time() >= wall_time(16, 0)
    )


def _decode_frame(frame, symbol: str, timeframe: str, now_et: datetime) -> list[dict]:
    bars: list[dict] = []
    if frame is None or frame.empty:
        return bars
    for timestamp, row in frame.iterrows():
        try:
            open_price = float(row["Open"])
            high = float(row["High"])
            low = float(row["Low"])
            close = float(row["Close"])
            volume_value = float(row["Volume"])
        except (KeyError, TypeError, ValueError):
            continue
        if not all(math.isfinite(value) and value > 0 for value in (open_price, high, low, close)):
            continue
        volume = max(int(volume_value), 0) if math.isfinite(volume_value) else 0
        parsed = timestamp.to_pydatetime() if hasattr(timestamp, "to_pydatetime") else datetime.fromisoformat(str(timestamp))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=_ET)
        parsed = parsed.astimezone(_ET)
        is_final = _bar_is_final(parsed, timeframe, now_et)
        bars.append({
            "symbol": symbol,
            "timestamp": parsed.isoformat(),
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "timeframe": timeframe,
            "session": _bar_session(parsed, timeframe),
            "source": _SOURCE,
            "isDelayed": True,
            "adjustmentMode": _ADJUSTMENT_MODE,
            "isFinal": is_final,
            "status": "FINAL" if is_final else "FORMING",
            "isStale": False,
        })
    return sorted(bars, key=lambda item: item["timestamp"])


def _daily_closes(ticker_obj, now_et: datetime) -> list[dict]:
    frame = ticker_obj.history(period="10d", interval="1d", auto_adjust=False, prepost=False)
    closes: list[dict] = []
    if frame is None or frame.empty:
        return closes
    for timestamp, row in frame.iterrows():
        try:
            close = float(row["Close"])
            adjusted_close = float(row.get("Adj Close", close))
        except (KeyError, TypeError, ValueError):
            continue
        parsed = timestamp.to_pydatetime() if hasattr(timestamp, "to_pydatetime") else datetime.fromisoformat(str(timestamp))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=_ET)
        parsed = parsed.astimezone(_ET)
        if math.isfinite(close) and _bar_is_final(parsed, "1d", now_et):
            closes.append({
                "timestamp": parsed.replace(hour=16, minute=0, second=0, microsecond=0),
                "close": close,
                "adjustedClose": adjusted_close if math.isfinite(adjusted_close) else close,
            })
    return closes


def _build_quote(ticker_obj, symbol: str, now_et: datetime) -> dict:
    minute_frame = ticker_obj.history(period="5d", interval="1m", auto_adjust=True, prepost=True)
    latest_timestamp: datetime | None = None
    latest_price: float | None = None
    if minute_frame is not None and not minute_frame.empty:
        timestamp = minute_frame.index[-1]
        latest_timestamp = timestamp.to_pydatetime() if hasattr(timestamp, "to_pydatetime") else datetime.fromisoformat(str(timestamp))
        if latest_timestamp.tzinfo is None:
            latest_timestamp = latest_timestamp.replace(tzinfo=_ET)
        latest_timestamp = latest_timestamp.astimezone(_ET)
        try:
            candidate = float(minute_frame.iloc[-1]["Close"])
            latest_price = candidate if math.isfinite(candidate) else None
        except (KeyError, TypeError, ValueError):
            latest_price = None
    completed = _daily_closes(ticker_obj, now_et)
    regular_close_is_official = bool(completed)
    regular_close = completed[-1]["close"] if completed else latest_price
    session = _market_session(now_et, latest_timestamp)
    if session in {"PRE_MARKET", "REGULAR"}:
        previous_close = regular_close
    else:
        previous_close = completed[-2]["close"] if len(completed) > 1 else regular_close
    if session == "CLOSED" and regular_close is not None:
        latest_price = regular_close
        latest_timestamp = completed[-1]["timestamp"] if completed else latest_timestamp
    price = latest_price or regular_close
    if price is None:
        raise ValueError(f"no quote data for {symbol}")
    if session in {"PRE_MARKET", "AFTER_HOURS"} and regular_close:
        extended_change = price / regular_close - 1
        daily_change = (
            price / previous_close - 1 if session == "PRE_MARKET" and previous_close
            else regular_close / previous_close - 1 if previous_close
            else None
        )
    else:
        extended_change = None
        daily_change = price / previous_close - 1 if previous_close else None
    adjusted_day_return = None
    if session in {"AFTER_HOURS", "CLOSED"} and len(completed) > 1:
        adjusted_day_return = completed[-1]["adjustedClose"] / completed[-2]["adjustedClose"] - 1
    return {
        "symbol": symbol,
        "timestamp": (latest_timestamp or now_et).isoformat(),
        "price": round(price, 6),
        "regularClose": round(regular_close, 6) if regular_close is not None else None,
        "previousClose": round(previous_close, 6) if previous_close is not None else None,
        "dailyChangePct": round(daily_change, 8) if daily_change is not None else None,
        "dayReturn": round(daily_change, 8) if daily_change is not None else None,
        "adjustedDayReturn": round(adjusted_day_return, 8) if adjusted_day_return is not None else None,
        "extendedHoursChangePct": round(extended_change, 8) if extended_change is not None else None,
        "source": _SOURCE,
        "session": session,
        "isDelayed": True,
        "isFinal": session == "CLOSED",
        "regularCloseIsOfficial": regular_close_is_official,
        "adjustmentMode": _ADJUSTMENT_MODE,
        "returnAdjustmentMode": _RETURN_ADJUSTMENT_MODE,
    }


def _synchronize_forming_bar(bars: list[dict], quote: dict, timeframe: str) -> None:
    """Synchronize live bars and finalize the last regular bar from daily close."""
    if not bars:
        return
    quote_at = datetime.fromisoformat(quote["timestamp"]).astimezone(_ET)
    if quote.get("regularCloseIsOfficial") and quote["session"] in {"AFTER_HOURS", "CLOSED"}:
        regular_candidates = [
            bar for bar in bars
            if bar["session"] == "REGULAR"
            and datetime.fromisoformat(bar["timestamp"]).astimezone(_ET).date() == quote_at.date()
        ]
        if regular_candidates:
            regular_bar = regular_candidates[-1]
            regular_at = datetime.fromisoformat(regular_bar["timestamp"]).astimezone(_ET)
            valid_terminal_bar = timeframe in {"1d", "1w"} or regular_at.time() == wall_time(15, 30)
            if valid_terminal_bar:
                official_close = float(quote["regularClose"])
                regular_bar["close"] = official_close
                regular_bar["high"] = max(float(regular_bar["high"]), official_close)
                regular_bar["low"] = min(float(regular_bar["low"]), official_close)
                regular_bar["isFinal"] = True
                regular_bar["status"] = "FINAL"
                regular_bar["isStale"] = False
                regular_bar["quoteSynchronized"] = True
                regular_bar["finalizationSource"] = "DAILY_REGULAR_CLOSE"
            else:
                regular_bar["isFinal"] = False
                regular_bar["status"] = "STALE"
                regular_bar["isStale"] = True

    latest = bars[-1]
    bar_at = datetime.fromisoformat(latest["timestamp"]).astimezone(_ET)
    if timeframe == "1d":
        same_bar = quote["session"] == "REGULAR" and bar_at.date() == quote_at.date()
    elif timeframe == "1w":
        same_bar = quote["session"] == "REGULAR" and bar_at.isocalendar()[:2] == quote_at.isocalendar()[:2]
    else:
        duration = timedelta(minutes=30) if timeframe == "30m" else timedelta(hours=1)
        bar_end = bar_at + duration
        if latest["session"] == "REGULAR":
            bar_end = min(bar_end, bar_at.replace(hour=16, minute=0, second=0, microsecond=0))
        same_bar = latest["session"] == quote["session"] and bar_at <= quote_at <= bar_end
    if not same_bar:
        return
    price = float(quote["price"])
    latest["close"] = price
    latest["high"] = max(float(latest["high"]), price)
    latest["low"] = min(float(latest["low"]), price)
    latest["quoteSynchronized"] = True
    latest["isFinal"] = False
    latest["status"] = "FORMING"
    latest["isStale"] = False


def _fetch_price_history(ticker: str, range_key: str, include_extended: bool = False) -> dict:
    symbol = ticker.strip().upper()
    if not _TICKER_PATTERN.fullmatch(symbol):
        raise ValueError("invalid ticker")
    normalized_range = range_key.strip().upper()
    if normalized_range not in _PRICE_RANGES:
        raise ValueError("invalid price range")
    cache_key = (symbol, normalized_range, include_extended)
    with _CACHE_LOCK:
        cached = _PRICE_CACHE.get(cache_key)
        if cached and cached[0] > time.monotonic():
            return copy.deepcopy(cached[1])

    import yfinance as yf
    from v2.data.price_source import YFinancePriceSource

    config = _PRICE_RANGES[normalized_range]
    timeframe = config["timeframe"]
    now_et = datetime.now(_ET)
    visible_start = now_et.date() - timedelta(days=config["days"])
    fetch_start = visible_start - timedelta(days=config["warmup_days"])
    # Keep the public symbol/cache key unchanged; Yahoo spells share classes
    # with a dash (BRK-B), while broker/FD symbols use a dot (BRK.B).
    ticker_obj = yf.Ticker(YFinancePriceSource.yfinance_symbol(symbol))
    frame = ticker_obj.history(
        start=fetch_start.isoformat(),
        end=(now_et.date() + timedelta(days=1)).isoformat(),
        interval={"1w": "1wk"}.get(timeframe, timeframe),
        auto_adjust=False,
        prepost=include_extended and timeframe in {"30m", "1h"},
    )
    decoded_bars = _decode_frame(frame, symbol, timeframe, now_et)
    quote = _build_quote(ticker_obj, symbol, now_et)
    _synchronize_forming_bar(decoded_bars, quote, timeframe)
    all_bars = enrich_bars(decoded_bars, timeframe)
    visible_bars = [
        bar for bar in all_bars
        if datetime.fromisoformat(bar["timestamp"]).astimezone(_ET).date() >= visible_start
    ]
    if not visible_bars:
        raise ValueError(f"no historical data for {symbol}")
    analysis = build_technical_analysis(all_bars, timeframe, quote, _ADJUSTMENT_MODE)
    result = {
        "symbol": symbol,
        "range": normalized_range,
        "timeframe": timeframe,
        "periodDays": config["days"],
        "visibleStart": visible_start.isoformat(),
        "visibleEnd": now_et.date().isoformat(),
        "warmupBars": max(len(all_bars) - len(visible_bars), 0),
        "source": _SOURCE,
        "adjustmentMode": _ADJUSTMENT_MODE,
        "includesExtendedHours": include_extended and timeframe in {"30m", "1h"},
        "quote": quote,
        "bars": visible_bars,
        "technicalAnalysis": analysis,
    }
    ttl = 20 if timeframe in {"30m", "1h"} else 300
    with _CACHE_LOCK:
        _PRICE_CACHE[cache_key] = (time.monotonic() + ttl, copy.deepcopy(result))
    return result


@router.get("/price-history/{ticker}", dependencies=[Depends(require_owner)])
async def price_history(
    ticker: str,
    range_key: str = Query("1M", alias="range"),
    include_extended: bool = Query(False, alias="extended"),
) -> dict:
    try:
        return await run_in_threadpool(_fetch_price_history, ticker, range_key, include_extended)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
