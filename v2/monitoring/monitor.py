"""Monitoring orchestration: universe -> prices -> detect -> list of anomalies."""

from __future__ import annotations

import logging
from datetime import date, timedelta

from v2.data.client import FDClient
from v2.data.price_source import PriceSource, default_price_source
from v2.monitoring.detectors import detect
from v2.monitoring.models import Anomaly, MonitorConfig
from v2.monitoring.options import OptionsTracker
from v2.universe import SECTOR_ETFS

logger = logging.getLogger(__name__)

# 270 calendar days ≈ 252 trading days + buffer for weekends/holidays
_HISTORY_CALENDAR_DAYS = 380


def run_monitoring(
    tickers: list[str],
    fd_client: FDClient,
    config: MonitorConfig,
    *,
    price_source: PriceSource | None = None,
) -> list[Anomaly]:
    """Scan *tickers*, return all anomalies found on the latest trading day.

    ``price_source`` provides daily OHLCV (Phase 4.5-mini default:
    yfinance real-time EOD). ``fd_client`` still serves non-price
    endpoints inside :func:`v2.monitoring.detectors.detect`
    (financials / earnings / insider).
    """
    if price_source is None:
        price_source = default_price_source()

    end = date.today()
    start = (end - timedelta(days=_HISTORY_CALENDAR_DAYS)).isoformat()
    end_str = end.isoformat()

    # Shared OptionsTracker — one DB connection lifetime across all tickers
    options_tracker = OptionsTracker()

    # Pre-fetch sector ETF price series ONCE per run. Bounded extra cost:
    # |SECTOR_ETFS| price calls (through the injected price_source).
    etf_prices = {}
    for etf in SECTOR_ETFS:
        try:
            series = price_source.get_prices(etf, start, end_str)
        except Exception as exc:
            logger.warning("ETF %s prefetch failed: %s", etf, exc)
            continue
        if series:
            etf_prices[etf] = series

    anomalies: list[Anomaly] = []
    for ticker in tickers:
        prices = price_source.get_prices(ticker, start, end_str)
        if not prices:
            continue
        anomaly = detect(
            ticker, prices, config,
            fd_client=fd_client,
            options_tracker=options_tracker,
            etf_prices=etf_prices,
        )
        if anomaly is not None:
            anomalies.append(anomaly)
            logger.info("Anomaly: %s %s", ticker, anomaly.flags)

    return anomalies
