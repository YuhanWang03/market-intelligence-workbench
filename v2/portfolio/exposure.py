"""Sector exposure aggregation.

Pure function over ``PositionFlat`` list — each position's weight is
added to its ``sector_etf`` bucket. Unmapped tickers carry the
``OTHER`` label per :func:`v2.universe.sector_bucket_for`, so the cron
card can surface "你有 N% 在未分类标的" instead of silently merging
into broad-market exposure.
"""

from __future__ import annotations

from v2.portfolio.models import ExposureMetrics, PositionFlat


def compute_exposure(positions: list[PositionFlat]) -> ExposureMetrics:
    """Aggregate position weights by sector ETF bucket.

    Returns an empty :class:`ExposureMetrics` when there are no positions.
    """
    if not positions:
        return ExposureMetrics.empty()

    by_sector: dict[str, float] = {}
    for p in positions:
        if not p.sector_etf:
            continue
        by_sector[p.sector_etf] = by_sector.get(p.sector_etf, 0.0) + p.weight

    if not by_sector:
        return ExposureMetrics.empty()

    largest_sector = max(by_sector, key=by_sector.get)
    return ExposureMetrics(
        by_sector=by_sector,
        largest_sector=largest_sector,
        largest_sector_pct=by_sector[largest_sector],
    )
