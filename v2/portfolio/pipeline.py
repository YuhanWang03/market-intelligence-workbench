"""Risk report orchestrator.

One entry point — :func:`build_risk_report` — that fans out to the six
metric modules and stitches their outputs into a single :class:`RiskReport`.

Independence contract: a failure in any sub-module (positions /
concentration / exposure / pnl / drawdown / earnings_risk) populates a
warning string on the report but never raises. The cron scripts can ship
"partial" cards on degraded days instead of going silent.

A trace event is emitted at the end so the dashboard can attribute the
report end-to-end. The trace stays a no-op outside a
``capture_trace_with_framing`` context (production cron sets one up).
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

from v2.observability import emit
from v2.portfolio.concentration import compute_concentration
from v2.portfolio.drawdown import compute_drawdown
from v2.portfolio.earnings_risk import compute_earnings_risk
from v2.portfolio.exposure import compute_exposure
from v2.portfolio.models import RiskReport
from v2.portfolio.pnl import compute_pnl
from v2.portfolio.positions import get_flat_positions

logger = logging.getLogger(__name__)


def build_risk_report(
    *,
    today_iso: str | None = None,
    broker: Any | None = None,
    calendar_fetcher=None,
) -> RiskReport:
    """Build the full risk snapshot.

    Args:
        today_iso: today's ISO date — pass explicitly from the cron so
            the timezone (US/Eastern) is the cron's decision, not the
            pipeline's. Defaults to ``date.today().isoformat()``.
        broker: optional shim for tests — defaults to :mod:`v2.broker`.
        calendar_fetcher: optional shim for tests — defaults to
            :func:`v2.earnings.get_upcoming_batch`.

    Always returns a :class:`RiskReport`. Sub-section failures populate
    ``RiskReport.warnings`` for the formatter to surface.
    """
    today_iso = today_iso or date.today().isoformat()

    # ----- 1. positions + dollar totals -----
    # Alpaca's ``portfolio_value`` IS the TOTAL equity already — adding
    # ``cash`` again would double-count it. ``cash_pct`` is cash as a
    # fraction of TOTAL. ``invested_value`` is exposed on RiskReport
    # as a derived @property (portfolio_value - cash).
    positions, portfolio_value, cash, position_warnings = get_flat_positions(broker=broker)
    cash_pct = (cash / portfolio_value) if portfolio_value > 0 else 0.0

    report = RiskReport(
        snapshot_date=today_iso,
        portfolio_value=portfolio_value,
        cash=cash,
        cash_pct=cash_pct,
        positions=positions,
    )
    report.warnings.extend(position_warnings)

    # ----- 2. concentration (pure, no IO) -----
    try:
        report.concentration = compute_concentration(positions)
    except Exception as exc:
        msg = f"集中度计算失败：{exc}"
        logger.exception("concentration failed")
        report.warnings.append(msg)

    # ----- 3. sector exposure (pure, no IO) -----
    try:
        report.exposure = compute_exposure(positions)
    except Exception as exc:
        msg = f"行业暴露计算失败：{exc}"
        logger.exception("exposure failed")
        report.warnings.append(msg)

    # ----- 4. P&L (Alpaca account + history) -----
    pnl_metrics, pnl_warnings = compute_pnl(broker=broker)
    report.pnl = pnl_metrics
    report.warnings.extend(pnl_warnings)

    # ----- 5. drawdown (Alpaca history + today realtime) -----
    # Phase 2.5 full: pass today's real-time portfolio_value so an
    # intraday decline shows up in current_drawdown_pct on the SAME
    # day. Prior to this, Alpaca's get_portfolio_history series ended
    # at yesterday's close and today's realtime drop was invisible
    # to the drawdown walk — produced cards reading "今日 -3.72%"
    # next to "drawdown 0.00%". When portfolio_value is unavailable
    # (Alpaca down → 0.0), we pass None to preserve the old
    # behaviour rather than appending a meaningless 0.
    realtime_value = portfolio_value if portfolio_value > 0 else None
    dd_metrics, dd_warnings = compute_drawdown(
        broker=broker, today_realtime_value=realtime_value,
    )
    report.drawdown = dd_metrics
    report.warnings.extend(dd_warnings)

    # ----- 6. earnings risk (yfinance via v2.earnings.calendar) -----
    earnings_items, earnings_warnings = compute_earnings_risk(
        positions, today_iso, calendar_fetcher=calendar_fetcher,
    )
    report.earnings_risk_next_7d = earnings_items
    report.warnings.extend(earnings_warnings)

    # ----- Trace event -----
    emit(
        "transform",
        op="portfolio_risk_report",
        n_positions=len(positions),
        portfolio_value=portfolio_value,
        cash_pct=cash_pct,
        top_1_pct=report.concentration.top_1_pct,
        largest_sector=report.exposure.largest_sector,
        largest_sector_pct=report.exposure.largest_sector_pct,
        daily_pnl_pct=report.pnl.daily_pnl_pct,
        max_drawdown_pct=report.drawdown.max_drawdown_pct,
        n_earnings_next_7d=len(earnings_items),
        warnings=len(report.warnings),
    )

    return report
