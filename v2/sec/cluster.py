"""Same-day same-direction Form 4 cluster detection.

Stage 0 real-data calibration #1: a "30-day rolling cluster ≥ 3" threshold
fires on 80% of the universe (every ticker with a couple of officers
buying vests + selling for taxes hits it weekly). Tightened to
**same-day same-direction ≥ 3** — that demands coordination, which is
implausible at random and therefore a real signal.

Only P (Purchase) and S (Sale) transactions count toward clusters. Awards,
exercises, tax withholding, gifts, conversions don't qualify — see
``Form4Transaction.direction`` which returns ``None`` for non-P/S codes.

Cluster output drives the Stage-2 cron's priority +15 P1 adjustment when
``len(form4_clusters) >= 1`` for a ticker on a given day.
"""

from __future__ import annotations

from collections import defaultdict

from v2.sec.models import Form4Cluster, Form4Transaction


# Stage 0 spec — same-day cluster needs ≥ 3 same-direction P/S transactions.
# 2-insider events happen occasionally by coincidence (especially with
# joint filings); 3+ is rare enough that the signal is meaningful.
_DEFAULT_MIN_COUNT = 3


def find_clusters(
    transactions: list[Form4Transaction],
    min_count: int = _DEFAULT_MIN_COUNT,
) -> list[Form4Cluster]:
    """Group P/S transactions by (ticker, filing_date, direction).

    Returns clusters with at least ``min_count`` distinct insiders.
    "Distinct" matters because a single insider with three lot-sized
    transactions on one day doesn't count as a "cluster" in the
    signal-theoretic sense — same-actor activity is one decision.

    Args:
        transactions: any iterable of Form4Transaction. Non-P/S codes
            are ignored (``direction`` property returns None).
        min_count: distinct-insider threshold. Default 3 per Stage 0.

    Returns:
        List of Form4Cluster, sorted by ``(ticker, cluster_date)`` for
        deterministic test output.
    """
    # Group by (ticker, filing_date, direction)
    buckets: dict[tuple[str, str, str], list[Form4Transaction]] = defaultdict(list)
    for tx in transactions:
        direction = tx.direction
        if direction is None:    # A / M / F / G / C — not signal codes
            continue
        key = (tx.filing.ticker, tx.filing.filing_date, direction)
        buckets[key].append(tx)

    clusters: list[Form4Cluster] = []
    for (ticker, cluster_date, direction), txs in buckets.items():
        # Distinct-insider count — same actor's multiple lots are one decision
        unique_insiders = {tx.insider_name for tx in txs if tx.insider_name}
        if len(unique_insiders) < min_count:
            continue

        total_usd = sum(
            (tx.transaction_usd or 0.0) for tx in txs
        )
        # Preserve a stable ordering for the insider list in the card.
        insider_names = sorted(unique_insiders)

        clusters.append(Form4Cluster(
            ticker=ticker,
            cluster_date=cluster_date,
            direction=direction,
            transaction_count=len(txs),
            total_usd=total_usd,
            insider_names=insider_names,
            transactions=list(txs),
        ))

    clusters.sort(key=lambda c: (c.ticker, c.cluster_date, c.direction))
    return clusters
