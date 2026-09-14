"""Intraday TECH_30 anomaly scanner — runs inside the streamer loop.

Triggered every minute during US market hours. For each ticker:
  1. Fetch latest trade price (real-time via Alpaca)
  2. Fetch today's daily bar (cumulative volume + open)
  3. Compare price move vs threshold AND volume pace vs 30d avg
  4. Compute sector-ETF relative strength (reuse v2.universe)
  5. Cooldown 30 min per ticker so we don't spam

Cost philosophy: NO LLM, NO Tavily during market hours. Intraday is for
fast signal surfacing only — deeper attribution happens at 17:35 ET when
the post-market Anomaly Monitor runs.
"""

from __future__ import annotations

import logging
import os
from datetime import date, datetime, time as dtime
from zoneinfo import ZoneInfo

from v2.archive.store import Archive
from v2.bot import state
from v2.observability import capture_trace_with_framing
from v2.reporting import format_intraday_anomaly
from v2.reporting.notifier import TelegramNotifier
from v2.reporting.priority import compute_importance
from v2.universe import SECTOR_ETFS, sector_etf_for

logger = logging.getLogger(__name__)

_ET = ZoneInfo("US/Eastern")

# Thresholds
_PRICE_PCT_THRESHOLD = 0.03        # ≥ ±3% intraday move
_VOLUME_PACE_THRESHOLD = 2.5       # today_volume / (avg_30d * progress) ≥ this
_SECTOR_GAP_PP = 0.015             # contrarian if rel ≥ 1.5pp AND opposite sign

# Market window in ET. We only fire during regular session — pre-market
# moves often reverse, and Alpaca's day-bar volume conventions differ
# outside regular hours.
_REGULAR_START = dtime(9, 30)
_REGULAR_END = dtime(16, 0)

# 7 trading days of baseline rolling refresh — re-fetch if older than this.
_BASELINE_REFRESH_DAYS = 7


def scan_universe(notifier: TelegramNotifier) -> int:
    """One scan pass. Returns the number of anomalies fired this pass."""
    if not _in_regular_hours():
        return 0

    api_key = os.environ.get("APCA_API_KEY_ID", "").strip()
    secret = os.environ.get("APCA_API_SECRET_KEY", "").strip()
    if not api_key or not secret:
        logger.debug("Alpaca creds missing — intraday scan skipped")
        return 0

    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest, StockLatestTradeRequest
        from alpaca.data.timeframe import TimeFrame
    except ImportError as exc:
        logger.warning("alpaca-py missing: %s", exc)
        return 0

    client = StockHistoricalDataClient(api_key, secret)

    # Read the operator-managed universe every pass. Changes made from the Web
    # workbench therefore take effect on the next minute without a restart.
    universe = state.intraday_universe_list()
    if not universe:
        logger.info("Intraday universe is empty — scan skipped")
        return 0

    # Ensure 30-day volume baseline is fresh for the whole universe + ETFs.
    targets = list(dict.fromkeys(universe + list(SECTOR_ETFS)))
    _refresh_baselines(client, targets)

    # Pull latest trades for both universe + sector ETFs in ONE batched call
    try:
        trades = client.get_stock_latest_trade(
            StockLatestTradeRequest(symbol_or_symbols=targets),
        )
    except Exception as exc:
        logger.warning("Latest-trade batch failed: %s", exc)
        return 0

    latest_price: dict[str, float] = {}
    for sym, tr in (trades or {}).items():
        try:
            p = float(getattr(tr, "price", 0) or 0)
            if p > 0:
                latest_price[str(sym).upper()] = p
        except (TypeError, ValueError):
            continue

    # Pull today's day bar (open + cumulative volume) for both
    today_iso = date.today().isoformat()
    try:
        bar_req = StockBarsRequest(
            symbol_or_symbols=targets,
            timeframe=TimeFrame.Day,
            start=today_iso,
        )
        bars_data = client.get_stock_bars(bar_req)
    except Exception as exc:
        logger.warning("Day-bar fetch failed: %s", exc)
        return 0

    open_price: dict[str, float] = {}
    volume_today: dict[str, int] = {}
    try:
        bars_df = bars_data.df  # alpaca-py returns a multi-index DataFrame
    except AttributeError:
        bars_df = None

    if bars_df is not None and not bars_df.empty:
        for (sym, _ts), row in bars_df.iterrows():
            sym_u = str(sym).upper()
            try:
                if sym_u not in open_price:
                    open_price[sym_u] = float(row["open"])
                # Bars from `start=today` should give just today's row, but be
                # defensive — last row wins, which is the latest bar
                volume_today[sym_u] = int(row["volume"])
            except (KeyError, ValueError):
                continue

    progress = _market_progress()
    et_now_str = datetime.now(_ET).strftime("%H:%M")

    fired_count = 0
    for ticker in universe:
        signal = _evaluate_ticker(
            ticker,
            latest_price=latest_price,
            open_price=open_price,
            volume_today=volume_today,
            progress=progress,
            et_now=et_now_str,
        )
        if signal is None:
            continue

        if state.intraday_in_cooldown(ticker, minutes=30):
            logger.debug("%s in cooldown — skipping", ticker)
            continue

        try:
            with capture_trace_with_framing(
                agent="intraday_anomaly", intent="explain_move",
                text=f"(盘中扫描) {ticker} 异动",
                responder_name="_r_intraday_scan",
            ) as trace:
                text = format_intraday_anomaly(signal)
                trace.emit("chat_message", role="bot", text=text[:500])
            # Dedicated Archive agent so the dashboard's AGENT_TO_INTENT
            # routes intraday cards to the intraday_anomaly pipeline.
            intraday_notifier = TelegramNotifier(
                archive=Archive("intraday_anomaly"),
                archive_after_send=True,
                retention_trading_days=2,
            )
            # Surface contrarian / sector-relative flags so the scorer
            # can promote noteworthy moves vs market-beta noise.
            flags = list(signal.get("flags") or [])
            priority = compute_importance("intraday_anomaly", {
                "flags": flags,
                "price_change_pct": signal.get("price_change_pct"),
            })
            intraday_notifier.send_text(
                text,
                trace=trace,
                title=f"盘中异动 · {ticker}",
                tickers=[ticker],
                priority=priority,
            )
            state.intraday_record_fire(ticker)
            fired_count += 1
            logger.info(
                "Intraday fire: %s %s%.2f%% pace=%.1fx contrarian=%s",
                ticker,
                "+" if signal["price_change_pct"] >= 0 else "",
                signal["price_change_pct"] * 100,
                signal["volume_pace"],
                signal.get("contrarian", False),
            )
        except Exception as exc:
            logger.warning("Failed to push intraday fire for %s: %s", ticker, exc)

    return fired_count


# ---------------------------------------------------------------------------
# Internal: signal evaluation per ticker
# ---------------------------------------------------------------------------


def _evaluate_ticker(
    ticker: str,
    *,
    latest_price: dict[str, float],
    open_price: dict[str, float],
    volume_today: dict[str, int],
    progress: float,
    et_now: str,
) -> dict | None:
    """Return an anomaly signal dict if both thresholds met, else None."""
    p_now = latest_price.get(ticker)
    p_open = open_price.get(ticker)
    v_today = volume_today.get(ticker)
    if not p_now or not p_open or v_today is None:
        return None

    pct = (p_now - p_open) / p_open if p_open > 0 else 0
    if abs(pct) < _PRICE_PCT_THRESHOLD:
        return None

    avg_v, _ = state.baseline_get(ticker)
    if not avg_v or avg_v <= 0 or progress <= 0:
        return None

    expected_so_far = avg_v * progress
    pace = v_today / expected_so_far if expected_so_far > 0 else 0
    if pace < _VOLUME_PACE_THRESHOLD:
        return None

    # Sector relative strength — use ETF latest vs ETF open
    sector_etf = sector_etf_for(ticker)
    etf_p_now = latest_price.get(sector_etf)
    etf_p_open = open_price.get(sector_etf)
    sec_ret = None
    rel_pp = None
    contrarian = False
    if etf_p_now and etf_p_open and etf_p_open > 0:
        sec_ret = (etf_p_now - etf_p_open) / etf_p_open
        rel_pp = pct - sec_ret
        if (pct >= 0) != (sec_ret >= 0) and abs(rel_pp) >= _SECTOR_GAP_PP:
            contrarian = True

    return {
        "ticker":             ticker,
        "price":              p_now,
        "open_price":         p_open,
        "price_change_pct":   pct,
        "volume_today":       v_today,
        "volume_avg_30d":     avg_v,
        "volume_pace":        pace,
        "market_progress":    progress,
        "sector_etf":         sector_etf if sec_ret is not None else None,
        "sector_return":      sec_ret,
        "relative_pp":        rel_pp,
        "contrarian":         contrarian,
        "time_et":            et_now,
    }


# ---------------------------------------------------------------------------
# Market time helpers
# ---------------------------------------------------------------------------


def _in_regular_hours(now: datetime | None = None) -> bool:
    n = (now or datetime.now(_ET)).astimezone(_ET)
    if n.weekday() not in {0, 1, 2, 3, 4}:
        return False
    t = n.time()
    return _REGULAR_START <= t <= _REGULAR_END


def _market_progress(now: datetime | None = None) -> float:
    """Fraction of the 9:30-16:00 ET session elapsed. 0.0 at open, 1.0 at close."""
    n = (now or datetime.now(_ET)).astimezone(_ET)
    open_dt = n.replace(hour=9, minute=30, second=0, microsecond=0)
    close_dt = n.replace(hour=16, minute=0, second=0, microsecond=0)
    if n < open_dt:
        return 0.0
    if n > close_dt:
        return 1.0
    elapsed = (n - open_dt).total_seconds()
    total = (close_dt - open_dt).total_seconds()
    return elapsed / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# 30-day volume baseline refresh
# ---------------------------------------------------------------------------


def _refresh_baselines(client, tickers: list[str]) -> None:
    """For each ticker, refresh its 30-day avg volume if stale.

    Stale = no row, OR updated_at is more than _BASELINE_REFRESH_DAYS old.
    One Alpaca daily-bars request per stale ticker (could batch, but this
    only runs at most once per day per ticker so it's fine).
    """
    from datetime import timedelta
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    today = date.today()
    stale = []
    for t in tickers:
        _, updated_at = state.baseline_get(t)
        if updated_at is None:
            stale.append(t)
            continue
        try:
            ts = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
            age = today - ts.date()
            if age.days >= _BASELINE_REFRESH_DAYS:
                stale.append(t)
        except (ValueError, AttributeError):
            stale.append(t)

    if not stale:
        return

    logger.info("Refreshing 30d volume baselines for %d tickers", len(stale))

    end = today
    start = end - timedelta(days=45)  # ~30 trading days + weekend buffer

    try:
        req = StockBarsRequest(
            symbol_or_symbols=stale,
            timeframe=TimeFrame.Day,
            start=start.isoformat(),
            end=end.isoformat(),
        )
        data = client.get_stock_bars(req)
    except Exception as exc:
        logger.warning("Baseline batch fetch failed: %s", exc)
        return

    try:
        df = data.df
    except AttributeError:
        return

    if df is None or df.empty:
        return

    # Average daily volume per symbol over the requested range
    grouped = df.groupby(level=0)["volume"].mean()
    for sym, avg in grouped.items():
        try:
            avg_f = float(avg)
            if avg_f > 0:
                state.baseline_set(str(sym).upper(), avg_f)
        except (TypeError, ValueError):
            continue
