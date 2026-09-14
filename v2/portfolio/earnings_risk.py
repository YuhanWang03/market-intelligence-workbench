"""Upcoming earnings releases for held tickers, ≤ 7 days out.

Pure reuse of :func:`v2.earnings.get_upcoming_batch` — no new yfinance
plumbing. The portfolio cron just asks "for my current holdings, which
release in the next 7 days?" and surfaces those as risk events.

Caveats inherited from Stage 0 of Phase 1:

- yfinance per-ticker failures are silently skipped inside
  ``get_upcoming_batch``; we never see them here.
- ADRs / non-US tickers are filtered out by ``is_supported_ticker``
  inside the calendar module — they won't show up in the risk card
  even if held.
- Calendar dates are NEVER persisted — re-fetched each cron run so
  date amendments self-heal.
"""

from __future__ import annotations

import logging

from v2.portfolio.models import EarningsRiskItem, PositionFlat

logger = logging.getLogger(__name__)


# Default horizon for the daily 18:30 ET cron. The user spec says
# "未来 7 天财报风险" — 7 = a week of trading days plus weekend.
_DEFAULT_HORIZON_DAYS = 7


def compute_earnings_risk(
    positions: list[PositionFlat],
    today_iso: str,
    *,
    horizon_days: int = _DEFAULT_HORIZON_DAYS,
    calendar_fetcher=None,
) -> tuple[list[EarningsRiskItem], list[str]]:
    """For each held ticker, return upcoming earnings within ``horizon_days``.

    Args:
        positions: from ``positions.get_flat_positions``.
        today_iso: today's date as ISO string — the calendar is
            relative-to-today and the cron timezone is ET, so the caller
            passes its computed today rather than using ``date.today()``.
        horizon_days: how many calendar days forward to include.
            Default 7. The risk card formatter sees one item per release.
        calendar_fetcher: test seam — defaults to
            ``v2.earnings.get_upcoming_batch``.

    Returns ``(items, warnings)``. Sorted by ``days_until`` ascending.
    """
    if calendar_fetcher is None:
        from v2.earnings import get_upcoming_batch as calendar_fetcher

    warnings: list[str] = []

    if not positions:
        return [], warnings

    tickers = [p.ticker for p in positions]
    try:
        batch = calendar_fetcher(tickers)
    except Exception as exc:
        msg = f"yfinance 财报日历查询失败：{exc}"
        logger.warning(msg)
        warnings.append(msg)
        return [], warnings

    items: list[EarningsRiskItem] = []
    for ticker, event in batch.events.items():
        d_minus = event.d_minus(today_iso)
        if not (0 <= d_minus <= horizon_days):
            continue
        items.append(EarningsRiskItem(
            ticker=ticker,
            release_date=event.release_date,
            days_until=d_minus,
            estimated_eps=event.eps_estimate,
            estimated_revenue=event.revenue_estimate,
        ))

    items.sort(key=lambda x: (x.days_until, x.ticker))
    return items, warnings
