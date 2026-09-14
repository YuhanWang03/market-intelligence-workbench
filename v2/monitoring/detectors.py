"""Pure anomaly detection functions.

Each takes the latest prices for one ticker and returns either an Anomaly
(if any signal fires) or None. No IO except optional FD client for the
insider trading check (Phase B 3a).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from v2.data.client import FDClient
from v2.data.models import Price
from v2.monitoring.models import (
    Anomaly,
    InsiderActivity,
    InsiderExec,
    MonitorConfig,
)
from v2.monitoring.options import OptionsTracker
from v2.monitoring.relative import compute_relative
from v2.universe import sector_etf_for

logger = logging.getLogger(__name__)

_MIN_HISTORY = 252  # need a full year for the 52-week reference

# FD's get_insider_trades returns transaction_type as human-readable strings,
# not single-letter SEC Form 4 codes. We match the descriptive English phrases.
# We deliberately exclude "Tax or exercise-price share withholding", "Gift",
# option exercises, and grants — they don't reflect insider conviction.
_BUY_KEYWORDS = ("open market purchase", "open market buy")
_SELL_KEYWORDS = ("open market sale", "open market sell")

# Title keywords that promote a trade to "notable executive" rank.
_EXEC_KEYWORDS = (
    "ceo", "chief executive",
    "cfo", "chief financial",
    "coo", "chief operating",
    "president", "chairman",
)


def detect(
    ticker: str,
    prices: list[Price],
    config: MonitorConfig,
    fd_client: FDClient | None = None,
    options_tracker: OptionsTracker | None = None,
    etf_prices: dict[str, list[Price]] | None = None,
) -> Anomaly | None:
    """Run all detectors. Return an Anomaly with all flags that fired, or None.

    Augmentations triggered only when a price-based anomaly fires:
      - *fd_client* → insider-trading check (Phase B 3a)
      - *options_tracker* → snapshot + burst detection (Phase B 3b)
      - *etf_prices* → sector-relative strength comparison (Phase D)
    """
    if len(prices) < _MIN_HISTORY:
        return None

    latest = prices[-1]
    prev = prices[-2]

    # ---- Volume spike ----
    vols_30d = [p.volume for p in prices[-31:-1]]
    avg_30d = sum(vols_30d) / len(vols_30d) if vols_30d else 0.0
    volume_ratio = latest.volume / avg_30d if avg_30d > 0 else 1.0

    # ---- 52-week reference (close-based) ----
    last_252_closes = [p.close for p in prices[-_MIN_HISTORY:]]
    high_52w = max(last_252_closes)
    low_52w = min(last_252_closes)

    # ---- Price-based flags ----
    flags: list[str] = []
    if volume_ratio >= config.volume_spike_threshold:
        flags.append("volume_spike")
    if latest.close >= high_52w * config.high_52w_threshold:
        flags.append("52w_high")
    if latest.close <= low_52w * config.low_52w_threshold:
        flags.append("52w_low")

    if not flags:
        return None

    # ---- Insider augment (Phase B 3a) ----
    insider = None
    if fd_client is not None:
        insider = _detect_insider_activity(
            ticker, fd_client, latest.time[:10], config,
        )
        if insider is not None:
            if insider.net_value > 0:
                flags.append("insider_buying")
            else:
                flags.append("insider_selling")

    # ---- Options augment (Phase B 3b) ----
    options_snapshot = None
    options_burst = None
    if options_tracker is not None:
        options_snapshot = options_tracker.take_snapshot(ticker)
        if options_snapshot is not None:
            options_burst = options_tracker.detect_burst(ticker, options_snapshot)
            if options_burst is not None:
                if options_burst.side == "call":
                    flags.append("options_call_burst")
                else:
                    flags.append("options_put_burst")

    # ---- Sector relative strength (Phase D) ----
    sector_etf = sector_etf_for(ticker)
    sector_series = (etf_prices or {}).get(sector_etf)
    rel = compute_relative(prices, sector_series)

    if rel["contrarian"]:
        flags.append("contrarian_move")

    # ---- Build snapshot ----
    price_change_pct = (latest.close - prev.close) / prev.close if prev.close > 0 else 0.0
    recent = [p.close for p in prices[-config.sparkline_days:]]

    return Anomaly(
        ticker=ticker,
        date=latest.time[:10],
        price=float(latest.close),
        price_change_pct=float(price_change_pct),
        volume_today=int(latest.volume),
        volume_avg_30d=float(avg_30d),
        volume_ratio=float(volume_ratio),
        high_52w=float(high_52w),
        low_52w=float(low_52w),
        flags=flags,
        recent_prices=[float(p) for p in recent],
        insider=insider,
        options_snapshot=options_snapshot,
        options_burst=options_burst,
        sector_etf=sector_etf,
        sector_return_1d=rel["sector_return_1d"],
        relative_1d_pp=rel["relative_1d_pp"],
        contrarian=bool(rel["contrarian"]),
    )


# ---------------------------------------------------------------------------
# Insider trading detection (Phase B 3a)
# ---------------------------------------------------------------------------


def _detect_insider_activity(
    ticker: str,
    fd_client: FDClient,
    asof_date: str,
    config: MonitorConfig,
) -> InsiderActivity | None:
    """Aggregate Form 4 open-market trades over the last *insider_lookback_days*.

    Returns InsiderActivity if net buy ≥ buy threshold OR net sell ≥ sell threshold,
    else None. Detection-only — formatting happens in the reporting layer.
    """
    try:
        end_dt = datetime.fromisoformat(asof_date[:10])
    except (ValueError, TypeError):
        return None

    start = (end_dt - timedelta(days=config.insider_lookback_days)).isoformat()[:10]

    try:
        trades = fd_client.get_insider_trades(
            ticker, asof_date, start_date=start, limit=200,
        )
    except Exception as exc:
        logger.debug("insider fetch failed for %s: %s", ticker, exc)
        return None

    if not trades:
        return None

    # Open-market only — drop option exercises, grants, tax withholdings.
    open_market = [t for t in trades if _is_open_market(t)]
    if not open_market:
        return None

    buy_value = 0.0
    sell_value = 0.0
    executives: list[InsiderExec] = []

    for t in open_market:
        value = t.transaction_value
        if value is None or value <= 0:
            continue
        direction = _trade_direction(t)
        if direction is None:
            continue

        if direction == "buy":
            buy_value += value
        else:
            sell_value += value

        # Track named executives (CEO/CFO/etc) for display
        if _is_executive(t):
            executives.append(InsiderExec(
                name=(t.name or "")[:60],
                title=(t.title or "")[:60],
                direction=direction,
                value=float(value),
            ))

    net = buy_value - sell_value

    # Significance check
    if (
        net >= config.insider_buy_min_value
        or -net >= config.insider_sell_min_value
    ):
        return InsiderActivity(
            net_value=float(net),
            buy_value=float(buy_value),
            sell_value=float(sell_value),
            trade_count=len(open_market),
            executives=executives[:5],  # cap at 5 in display
        )
    return None


def _is_open_market(trade) -> bool:
    """True if the transaction is an open-market buy or sell (no insider grant/tax/etc)."""
    code = (trade.transaction_type or "").lower()
    return any(kw in code for kw in _BUY_KEYWORDS + _SELL_KEYWORDS)


def _trade_direction(trade) -> str | None:
    code = (trade.transaction_type or "").lower()
    if any(kw in code for kw in _BUY_KEYWORDS):
        return "buy"
    if any(kw in code for kw in _SELL_KEYWORDS):
        return "sell"
    return None


def _is_executive(trade) -> bool:
    title = (trade.title or "").lower()
    return any(kw in title for kw in _EXEC_KEYWORDS)
