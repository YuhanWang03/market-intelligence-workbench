"""Concentration metrics — Top-N share + Herfindahl-Hirschman index.

Pure functions over ``PositionFlat`` lists. No IO, no broker calls.
"""

from __future__ import annotations

from v2.portfolio.models import ConcentrationMetrics, PositionFlat


def compute_concentration(positions: list[PositionFlat]) -> ConcentrationMetrics:
    """Compute Top-1 / Top-3 / Top-5 cumulative share + HHI.

    Empty input → ``ConcentrationMetrics.empty()`` (all zeros). Negative
    weights (short positions, if Alpaca ever returns them) are still
    summed by absolute value into HHI — they represent risk regardless
    of direction — but Top-N uses signed weights so a 30% short doesn't
    masquerade as a 30% long.
    """
    if not positions:
        return ConcentrationMetrics.empty()

    # Assume positions arrive sorted by weight descending (positions.py
    # contract). Re-sort defensively in case a caller hand-built the list.
    sorted_pos = sorted(positions, key=lambda p: p.weight, reverse=True)
    weights = [p.weight for p in sorted_pos]
    nonzero = [w for w in weights if w != 0.0]

    top_1 = weights[0] if weights else 0.0
    top_3 = sum(weights[:3])
    top_5 = sum(weights[:5])

    # HHI: Σ w_i² over invested portion. Captures dispersion in a single
    # scalar that the cron card can render as "中等集中 0.18".
    hhi = sum(w * w for w in weights)

    return ConcentrationMetrics(
        top_1_pct=top_1,
        top_3_pct=top_3,
        top_5_pct=top_5,
        hhi=hhi,
        n_positions=len(nonzero),
    )
