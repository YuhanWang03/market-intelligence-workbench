"""Daily / weekly / monthly P&L from Alpaca account + portfolio_history.

Two data sources:

- **Daily** comes from the account snapshot (``equity - last_equity``) —
  already available via the existing ``v2/broker/get_pnl()`` wrapper, so
  no extra API call is needed for the daily numbers.
- **Weekly / monthly** come from ``portfolio_history(1M, 1D)`` walking
  back 5 / 21 trading days.

If the account has less history than the requested window, the
unavailable fields return ``None`` and the caller renders "数据不足".
Single-point history (account opened today) → all weekly / monthly
fields are None but daily still works.
"""

from __future__ import annotations

import logging
from typing import Any

from v2.portfolio.models import PnLMetrics

logger = logging.getLogger(__name__)


# Trading-day lookbacks. Alpaca's portfolio_history with timeframe="1D"
# returns one bar per trading day — so these are direct array offsets
# from the end.
_TRADING_DAYS_WEEK = 5
_TRADING_DAYS_MONTH = 21


def compute_pnl(broker: Any | None = None) -> tuple[PnLMetrics, list[str]]:
    """Pull intraday + portfolio_history and derive 1D / 1W / 1M returns.

    Returns ``(metrics, warnings)``. Both fields populated independently
    — a portfolio_history failure leaves the daily numbers intact.
    """
    if broker is None:
        from v2 import broker as _broker
        broker = _broker

    warnings: list[str] = []
    daily_pnl: float | None = None
    daily_pnl_pct: float | None = None
    weekly_pct: float | None = None
    monthly_pct: float | None = None

    # ----- Daily (no history call needed) -----
    try:
        pnl_snap = broker.get_pnl()
        daily_pnl = float(pnl_snap.get("intraday_pl") or 0.0)
        daily_pnl_pct = float(pnl_snap.get("intraday_pl_pct") or 0.0)
    except Exception as exc:
        msg = f"Alpaca 当日 P&L 不可用：{exc}"
        logger.warning(msg)
        warnings.append(msg)

    # ----- Weekly + monthly (one history call) -----
    try:
        hist = broker.get_portfolio_history(period="1M", timeframe="1D")
    except Exception as exc:
        msg = f"Alpaca 历史 equity 不可用：{exc}"
        logger.warning(msg)
        warnings.append(msg)
        return PnLMetrics(
            daily_pnl=daily_pnl, daily_pnl_pct=daily_pnl_pct,
            weekly_pnl_pct=None, monthly_pnl_pct=None,
        ), warnings

    equity = hist.get("equity") or []
    if len(equity) < 2:
        # Single-point history — account too new. Daily still applies.
        return PnLMetrics(
            daily_pnl=daily_pnl, daily_pnl_pct=daily_pnl_pct,
            weekly_pnl_pct=None, monthly_pnl_pct=None,
        ), warnings

    current = equity[-1]

    # Walking back N+1 bars: equity[-(N+1)] is the equity N trading days
    # ago, so the return is (current - then) / then.
    def _period_return(n_days: int) -> float | None:
        if len(equity) <= n_days:
            return None
        prior = equity[-(n_days + 1)]
        if prior <= 0:
            return None
        return (current - prior) / prior

    weekly_pct = _period_return(_TRADING_DAYS_WEEK)
    monthly_pct = _period_return(_TRADING_DAYS_MONTH)

    return PnLMetrics(
        daily_pnl=daily_pnl,
        daily_pnl_pct=daily_pnl_pct,
        weekly_pnl_pct=weekly_pct,
        monthly_pnl_pct=monthly_pct,
    ), warnings
