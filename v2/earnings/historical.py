"""Historical earnings data via FD — the authoritative source for actuals.

Stage 0 verified FD exposes exactly two methods we care about:

- ``get_earnings(ticker)`` → one ``EarningsRecord`` (most recent filing)
- ``get_earnings_history(ticker, limit)`` → ``list[EarningsRecord]``

This module normalises FD's ``EarningsRecord`` / ``EarningsData`` shape to
the agent-internal ``EarningsHistorical``, so downstream summarizer / card
code never reaches into FD's raw types.

All FD calls are wrapped in try/except — the cron loop must survive
transient API issues. Caller sees ``None`` / empty list, never an exception.
"""

from __future__ import annotations

import logging
from typing import Any

from v2.earnings.models import EarningsHistorical, EpsSurprise

logger = logging.getLogger(__name__)


# Values FD has been seen to put in EarningsData.eps_surprise. Anything not
# in this set degrades to "UNKNOWN".
_VALID_SURPRISE: frozenset[str] = frozenset({"BEAT", "MISS", "MEET"})


def get_latest_actual(fd: Any, ticker: str) -> EarningsHistorical | None:
    """Return the most recent filed earnings for ``ticker``.

    ``fd`` is duck-typed (we accept any object with the FD methods so the
    backtest's ``MockFDClient`` slots in cleanly). Returns ``None`` if FD
    has nothing, the call failed, or the record lacks quarterly data.
    """
    try:
        record = fd.get_earnings(ticker)
    except Exception as exc:
        logger.warning("fd.get_earnings(%s) failed: %s", ticker, exc)
        return None

    if record is None:
        return None
    return _from_record(record, ticker)


def get_recent(fd: Any, ticker: str, limit: int = 4) -> list[EarningsHistorical]:
    """Return the last ``limit`` filings for ``ticker``, newest first.

    Records that can't be normalised (no quarterly block) are filtered out
    — the caller wants useful rows, not raw FD output.
    """
    try:
        records = fd.get_earnings_history(ticker, limit=limit)
    except Exception as exc:
        logger.warning("fd.get_earnings_history(%s) failed: %s", ticker, exc)
        return []

    out: list[EarningsHistorical] = []
    for r in records or []:
        h = _from_record(r, ticker)
        if h is not None and h.has_quarterly_data:
            out.append(h)
    return out


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _from_record(record: Any, ticker: str) -> EarningsHistorical | None:
    """Convert an FD ``EarningsRecord`` into our ``EarningsHistorical``.

    FD's shape (verified in Stage 0 via v2/backtesting/test_backtest.py and
    v2/screening/screener.py):

        EarningsRecord(
            ticker, report_period, source_type, filing_date,
            quarterly: EarningsData | None,
        )

        EarningsData has at least:
            eps_surprise: "BEAT" | "MISS" | "MEET" | None
            earnings_per_share, estimated_earnings_per_share
            revenue, estimated_revenue
    """
    if record is None:
        return None

    quarterly = getattr(record, "quarterly", None)
    if quarterly is None:
        # Cron path may still want the bare record for filing-date tracking.
        return EarningsHistorical(
            ticker=ticker,
            report_period=getattr(record, "report_period", "") or "",
            filing_date=getattr(record, "filing_date", "") or "",
            source_type=getattr(record, "source_type", "") or "",
        )

    raw_surprise = getattr(quarterly, "eps_surprise", None)
    surprise: EpsSurprise = (
        raw_surprise if raw_surprise in _VALID_SURPRISE else "UNKNOWN"
    )

    return EarningsHistorical(
        ticker=ticker,
        report_period=getattr(record, "report_period", "") or "",
        filing_date=getattr(record, "filing_date", "") or "",
        source_type=getattr(record, "source_type", "") or "",
        eps_actual=_to_float(getattr(quarterly, "earnings_per_share", None)),
        eps_estimate=_to_float(
            getattr(quarterly, "estimated_earnings_per_share", None)
        ),
        eps_surprise=surprise,
        revenue_actual=_to_float(getattr(quarterly, "revenue", None)),
        revenue_estimate=_to_float(getattr(quarterly, "estimated_revenue", None)),
    )


def _to_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def surprise_history(records: list[EarningsHistorical]) -> list[EpsSurprise]:
    """Helper: just the BEAT/MISS/MEET sequence, newest first.

    Useful for the post-release card's "last 4Q surprise streak" context
    and the summarizer's bull/bear prompt.
    """
    return [r.eps_surprise for r in records if r.eps_surprise != "UNKNOWN"]
