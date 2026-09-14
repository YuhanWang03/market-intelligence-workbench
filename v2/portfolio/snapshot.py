"""Per-day positions snapshots + weekly attribution — Phase 2.5 full.

Phase 2 shipped ⑩ Friday Portfolio Weekly without per-position
attribution because Alpaca doesn't expose per-position historical
equity curves: the only way to reconstruct each holding's weekly
return is to take daily snapshots ourselves. This module is that
plumbing.

Three responsibilities:

- :func:`write_daily_snapshot` — called by ⑨b sub-cron at 16:25 ET
  Mon-Fri. Writes one row per current holding into
  ``positions_snapshot`` (PRIMARY KEY on snapshot_date + ticker so
  same-day reruns overwrite).

- :func:`read_weekly_window` — called by ⑩ at Fri 19:00 ET. Reads
  the last N calendar days of snapshots, grouped by ticker. Tickers
  that appear in any day inside the window are included (so a
  position added mid-week is still tracked even if it lacks the full
  5-day history).

- :func:`compute_weekly_attribution` — pure-Python aggregation. Per
  ticker: ``weekly_return = (last_mv / first_mv) - 1``, ``avg_weight
  = mean(weight observations)``, ``contribution = avg_weight ×
  weekly_return``. Returns sorted by contribution descending so ⑩'s
  card can render the best at the top and the worst at the bottom.

The minimum viable weekly window is 5 snapshots; below that the ⑩
card displays "归因数据累积中 (N/5)" instead of best/worst rows.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Iterable

from v2.portfolio.models import PositionFlat

logger = logging.getLogger(__name__)


# Minimum number of daily snapshots before per-position attribution is
# meaningful. Below this the ⑩ card falls back to "归因累积中".
WEEKLY_MIN_DAYS = 5


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PositionSnapshot:
    """One ticker's value at one EOD checkpoint.

    Mirrors :class:`PositionFlat` (4 fields) + ``snapshot_date``. We
    keep the dataclass separate from PositionFlat because the snapshot
    grain (per-day historical) is semantically different from the
    flat (current live).
    """

    snapshot_date: str          # ISO YYYY-MM-DD
    ticker: str
    market_value: float
    weight: float               # share of invested_value at snapshot time
    sector_etf: str | None      # nullable — column allows NULL


@dataclass(frozen=True)
class AttributionItem:
    """One ticker's contribution to weekly portfolio return.

    ``contribution = avg_weight × weekly_return`` — the standard
    Brinson-style attribution. Summing across positions approximates
    the total weekly P&L; the small residual is the cash/sector-shift
    cross-term, which we ignore for clarity.
    """

    ticker: str
    avg_weight: float           # mean over observed days in window
    weekly_return: float        # (last_mv - first_mv) / first_mv
    contribution: float         # avg_weight × weekly_return


# ---------------------------------------------------------------------------
# Public API — called by ⑨b cron + ⑩ Friday weekly + tests
# ---------------------------------------------------------------------------

def write_daily_snapshot(
    positions: list[PositionFlat],
    snapshot_date: str,
    archive,
) -> int:
    """Write one day's snapshot of current holdings to ``archive``.

    INSERT OR REPLACE on (snapshot_date, ticker) — a same-day rerun
    overwrites the prior entry, so ops can re-trigger ⑨b without
    inflating the table.

    Empty ``positions`` list → writes 0 rows + logs INFO (all-cash
    days are normal; we don't want a warning log noise floor).
    """
    if not positions:
        logger.info(
            "positions_snapshot: empty positions on %s — skipping write",
            snapshot_date,
        )
        return 0

    rows = [
        {
            "snapshot_date": snapshot_date,
            "ticker": p.ticker,
            "market_value": float(p.market_value),
            "weight": float(p.weight),
            "sector_etf": p.sector_etf,
        }
        for p in positions
    ]
    written = archive.write_position_snapshots(rows)
    logger.info(
        "positions_snapshot: wrote %d rows for %s", written, snapshot_date,
    )
    return written


def read_weekly_window(
    archive,
    end_date_iso: str,
    window_days: int = 7,
) -> dict[str, list[PositionSnapshot]]:
    """Read the last ``window_days`` of snapshots ending at
    ``end_date_iso``, grouped by ticker.

    Weekend days are simply absent from the result (cron only writes
    Mon-Fri), so a 7-day window typically yields 5 weekday snapshots
    per ticker. ``window_days`` is the lookback in *calendar* days,
    not trading days.

    Tickers that appear on any day inside the window are included.
    This handles the mid-week add/remove case cleanly: a position
    added on Wed is tracked from Wed-Fri (3 snapshots) and
    :func:`compute_weekly_attribution` will see only 3 points for
    that ticker but still compute a partial-window return.

    Returns: ``{ticker: [snapshots sorted by snapshot_date ASC]}``.
    Empty dict if no rows in window.
    """
    try:
        end_d = date.fromisoformat(end_date_iso)
    except ValueError:
        logger.warning(
            "read_weekly_window: bad end_date_iso %r — returning empty",
            end_date_iso,
        )
        return {}

    start_d = end_d - timedelta(days=window_days)
    rows = archive.get_position_snapshots(
        since=start_d.isoformat(), until=end_d.isoformat(),
    )

    grouped: dict[str, list[PositionSnapshot]] = {}
    for r in rows:
        snap = PositionSnapshot(
            snapshot_date=r["snapshot_date"],
            ticker=r["ticker"],
            market_value=float(r["market_value"]),
            weight=float(r["weight"]),
            sector_etf=r.get("sector_etf"),
        )
        grouped.setdefault(snap.ticker, []).append(snap)

    # get_position_snapshots already sorts by (snapshot_date, ticker),
    # so each grouped list is already ASC — no extra sort needed.
    return grouped


def compute_weekly_attribution(
    snapshots: dict[str, list[PositionSnapshot]],
    current_positions: Iterable[PositionFlat] | None = None,
) -> list[AttributionItem]:
    """Brinson-style per-position contribution for the window.

    For each ticker with at least 2 snapshots:
        weekly_return = (last.market_value - first.market_value) / first.market_value
        avg_weight    = mean(snap.weight for snap in snapshots)
        contribution  = avg_weight × weekly_return

    Tickers with fewer than 2 snapshots (e.g. just added today) are
    skipped — there's no return to compute over a single point.
    Tickers whose ``first.market_value`` is 0 are skipped to avoid
    a ZeroDivisionError on a delisted/de-funded position.

    ``current_positions`` is accepted for forward-compat with a future
    "include all live tickers even if no snapshots" mode; currently
    unused — the snapshots dict is the only source of truth.

    Returns a list sorted by ``contribution`` descending — ⑩'s card
    can pick ``[0]`` for best and ``[-1]`` for worst.
    """
    items: list[AttributionItem] = []
    for ticker, snaps in snapshots.items():
        if len(snaps) < 2:
            continue
        first = snaps[0]
        last = snaps[-1]
        if first.market_value <= 0:
            continue
        weekly_return = (last.market_value - first.market_value) / first.market_value
        avg_weight = sum(s.weight for s in snaps) / len(snaps)
        contribution = avg_weight * weekly_return
        items.append(AttributionItem(
            ticker=ticker,
            avg_weight=avg_weight,
            weekly_return=weekly_return,
            contribution=contribution,
        ))

    items.sort(key=lambda x: x.contribution, reverse=True)
    return items


__all__ = [
    "AttributionItem",
    "PositionSnapshot",
    "WEEKLY_MIN_DAYS",
    "compute_weekly_attribution",
    "read_weekly_window",
    "write_daily_snapshot",
]
