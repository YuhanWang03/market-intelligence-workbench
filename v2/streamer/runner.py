"""Streamer main loop — minute-level polling of open price alerts.

Price source: Alpaca's get_latest_trade endpoint (paper account included).
Alpaca data is real-time, free, and doesn't require a separate subscription
for IEX-routed quotes — which is enough for trigger detection.

Cooldown: there's no need for an explicit cooldown table because alerts
are one-shot: once fired_at is set, the alert never re-fires. To get a
new alert at the same price the user creates a new one.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, time as dtime
from typing import Iterable
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

from v2.bot import state
from v2.reporting import format_alert_fired
from v2.archive.store import Archive
from v2.observability import capture_trace_with_framing, install_all
from v2.reporting.notifier import TelegramNotifier
from v2.reporting.priority import compute_importance
from v2.streamer.intraday_scan import scan_universe

logger = logging.getLogger(__name__)

_ET = ZoneInfo("US/Eastern")

# US equities regular session: 9:30 - 16:00 ET. We add a 30-min pre-open
# poll window because pre-market alerts are useful and Alpaca returns
# pre-market quotes during that window.
_POLL_START = dtime(9, 0)
_POLL_END = dtime(16, 30)

_POLL_INTERVAL_SECONDS = 60
_OFF_HOURS_SLEEP_SECONDS = 300  # how long to wait outside market hours
_TRADING_WEEKDAYS = {0, 1, 2, 3, 4}  # Mon-Fri


def run_streamer(test_now: bool = False) -> None:
    """Block forever, polling for alert triggers each minute during market hours.

    If *test_now* is True, ignore the market-hours gate and run ONE poll
    immediately, then exit. Useful for verifying the pipeline off-hours.
    """
    load_dotenv()
    # Arm the observability hooks so capture_trace_with_framing inside
    # the alert / intraday push paths collects Alpaca / db events.
    install_all()

    # The runner-level notifier sends to Telegram. Archive writes happen
    # via per-push-type notifier wrappers below (_push_alert /
    # scan_universe), each pointing at a distinct Archive agent so the
    # dashboard's AGENT_TO_INTENT lookup picks the correct pipeline.
    notifier = TelegramNotifier()

    if test_now:
        logger.info("Test mode: forcing one poll regardless of market hours")
        _check_user_alerts(notifier)
        try:
            # Bypass the regular-hours gate inside scan_universe for testing
            import v2.streamer.intraday_scan as scan_mod
            original = scan_mod._in_regular_hours
            scan_mod._in_regular_hours = lambda *a, **kw: True
            try:
                fired = scan_mod.scan_universe(notifier)
                logger.info("Test scan complete. %d intraday anomalies fired.", fired)
            finally:
                scan_mod._in_regular_hours = original
        except Exception as exc:
            logger.exception("Test scan failed: %s", exc)
        return

    logger.info("Streamer started. Polling every %ds during market hours.",
                _POLL_INTERVAL_SECONDS)

    while True:
        try:
            if _market_is_open():
                _poll_once(notifier)
                time.sleep(_POLL_INTERVAL_SECONDS)
            else:
                logger.debug("Outside market hours — sleeping %ds",
                             _OFF_HOURS_SLEEP_SECONDS)
                time.sleep(_OFF_HOURS_SLEEP_SECONDS)
        except KeyboardInterrupt:
            logger.info("Streamer shutting down (Ctrl+C)")
            return
        except Exception as exc:
            # Never die — log and continue. The systemd unit will also
            # restart us if the process crashes outright.
            logger.exception("Streamer loop error: %s", exc)
            time.sleep(_POLL_INTERVAL_SECONDS)


def _market_is_open(now: datetime | None = None) -> bool:
    """True during NYSE/NASDAQ regular hours (Mon-Fri 9:00 - 16:30 ET).

    We deliberately ignore market holidays — firing an alert on a holiday
    is harmless (no one's trading anyway) and we'd rather not maintain a
    holiday calendar.
    """
    now = (now or datetime.now(_ET)).astimezone(_ET)
    if now.weekday() not in _TRADING_WEEKDAYS:
        return False
    t = now.time()
    return _POLL_START <= t <= _POLL_END


def _poll_once(notifier: TelegramNotifier) -> None:
    """One poll cycle does TWO things:

    1. User-set price alerts — check `alerts` table, fire any that crossed
    2. Universe-wide intraday scan — TECH_30 anomaly detection

    Both are independent: a failure in one won't suppress the other.
    """
    _check_user_alerts(notifier)
    try:
        scan_universe(notifier)
    except Exception as exc:
        logger.exception("Intraday scan failed: %s", exc)


def _check_user_alerts(notifier: TelegramNotifier) -> None:
    """Check the alerts table and fire any whose threshold was crossed."""
    tickers = state.alert_unfired_tickers()
    if not tickers:
        return

    logger.debug("Polling %d tickers with open alerts: %s",
                 len(tickers), tickers)

    prices = _latest_prices(tickers)
    if not prices:
        logger.warning("No prices returned for %d tickers", len(tickers))
        return

    for ticker, price in prices.items():
        fired = state.alert_fire_check(ticker, price)
        for f in fired:
            _push_alert(notifier, f)


def _push_alert(notifier: TelegramNotifier, fired: dict) -> None:
    """Send the triggered-alert card to Telegram + dashboard archive."""
    try:
        ticker = str(fired.get("ticker") or "?")
        with capture_trace_with_framing(
            agent="alert", intent="alert_set",
            text=f"(自动触发) {ticker} 价格提醒",
            responder_name="_r_alert_fire",
        ) as trace:
            text = format_alert_fired(fired)
            trace.emit("chat_message", role="bot", text=text[:500])
        # Dedicated Archive agent "alert" so the dashboard's
        # AGENT_TO_INTENT maps to the alert_fire pipeline.
        alert_notifier = TelegramNotifier(
            archive=Archive("alert"),
            archive_after_send=True,
            retention_trading_days=2,
        )
        priority = compute_importance("alert_fire", {})  # always P0
        alert_notifier.send_text(
            text,
            trace=trace,
            title=f"价格提醒 · {ticker}",
            tickers=[ticker],
            priority=priority,
        )
        logger.info(
            "Alert fired: #%d %s %s $%.2f (current $%.2f)",
            fired["id"], fired["ticker"], fired["direction"],
            fired["target_price"], fired["fired_price"],
        )
    except Exception as exc:
        logger.exception("Failed to push alert #%s: %s", fired.get("id"), exc)


def _latest_prices(tickers: Iterable[str]) -> dict[str, float]:
    """Fetch the latest trade price for each ticker via Alpaca.

    Falls back silently for any ticker that fails — partial data is better
    than no data because other alerts still fire.
    """
    if not tickers:
        return {}

    api_key = os.environ.get("APCA_API_KEY_ID", "").strip()
    secret = os.environ.get("APCA_API_SECRET_KEY", "").strip()
    if not api_key or not secret:
        logger.warning("Alpaca credentials missing — cannot poll prices")
        return {}

    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockLatestTradeRequest
    except ImportError as exc:
        logger.error("alpaca-py not installed: %s", exc)
        return {}

    client = StockHistoricalDataClient(api_key, secret)
    out: dict[str, float] = {}
    ticker_list = list(tickers)
    try:
        req = StockLatestTradeRequest(symbol_or_symbols=ticker_list)
        latest = client.get_stock_latest_trade(req)
    except Exception as exc:
        logger.warning("Bulk latest-trade fetch failed (%d tickers): %s",
                       len(ticker_list), exc)
        return {}

    for sym, trade in (latest or {}).items():
        try:
            price = float(getattr(trade, "price", 0) or 0)
            if price > 0:
                out[str(sym).upper()] = price
        except (TypeError, ValueError):
            continue
    return out
