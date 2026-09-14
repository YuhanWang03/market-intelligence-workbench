"""Numeric transforms for FRED series — Phase 4 Stage 1.

Each function returns a single float (or None on insufficient data).
The pipeline routes each FRED series through the transform named in
:data:`v2.macro.series.FRED_SERIES`.

Stage 0 design ack: LLM never produces these numbers — Python computes,
LLM only labels. Keeping these in their own module makes the unit
tests (and the responsibility split) explicit.
"""

from __future__ import annotations

import math
from typing import Sequence


def _to_list(series) -> list[float]:
    """Accept a pandas.Series, list, or tuple; return a list of floats
    with NaN dropped. Avoids a hard pandas dependency in this module."""
    try:
        # pandas.Series exposes .dropna() + .tolist()
        clean = series.dropna()
        return [float(v) for v in clean.tolist()]
    except AttributeError:
        return [float(v) for v in series if v is not None and not _isnan(v)]


def _isnan(v) -> bool:
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Pct-change family
# ---------------------------------------------------------------------------

def mom_pct(series) -> float | None:
    """Latest / prior - 1. Returns None when fewer than 2 points."""
    vals = _to_list(series)
    if len(vals) < 2:
        return None
    prior, latest = vals[-2], vals[-1]
    if prior == 0:
        return None
    return latest / prior - 1.0


def yoy_pct(series) -> float | None:
    """Latest / value 12 periods ago - 1. None if window too short."""
    vals = _to_list(series)
    if len(vals) < 13:
        return None
    year_ago, latest = vals[-13], vals[-1]
    if year_ago == 0:
        return None
    return latest / year_ago - 1.0


def mom_change_k(series) -> float | None:
    """NFP-style: latest - prior, expressed in thousands (the series
    is already in thousands of jobs, so we just diff)."""
    vals = _to_list(series)
    if len(vals) < 2:
        return None
    return vals[-1] - vals[-2]


def qoq_annualized(quarterly_series) -> float | None:
    """GDP: (latest / prior) ** 4 - 1. None when < 2 points or prior=0."""
    vals = _to_list(quarterly_series)
    if len(vals) < 2:
        return None
    prior, latest = vals[-2], vals[-1]
    if prior <= 0:
        return None
    return (latest / prior) ** 4 - 1.0


# ---------------------------------------------------------------------------
# Smoothing / level
# ---------------------------------------------------------------------------

def four_week_ma(weekly_series) -> float | None:
    """For ICSA — average of last 4 weeks. Partial windows (< 4 weeks
    in series) return whatever average we can compute, or None if empty."""
    vals = _to_list(weekly_series)
    if not vals:
        return None
    tail = vals[-4:]
    return sum(tail) / len(tail)


def latest_level(series) -> float | None:
    """Pass-through latest value (for rates / spreads)."""
    vals = _to_list(series)
    if not vals:
        return None
    return vals[-1]


# ---------------------------------------------------------------------------
# Surprise / sigma
# ---------------------------------------------------------------------------

def surprise_pct(actual: float | None, consensus: float | None) -> float | None:
    """(actual - consensus) / abs(consensus). None when either is None
    or consensus == 0 (avoid divide-by-zero)."""
    if actual is None or consensus is None or consensus == 0:
        return None
    return (actual - consensus) / abs(consensus)


def surprise_sigma(
    actual: float | None,
    consensus: float | None,
    hist_std: float | None,
) -> float | None:
    """Standardized surprise = (actual - consensus) / hist_std.

    Returns ``math.inf`` (or ``-math.inf``) when hist_std is exactly 0
    and the gap is non-zero — surfaces the pathological case rather
    than silently dividing. None when any input is None.
    """
    if actual is None or consensus is None or hist_std is None:
        return None
    gap = actual - consensus
    if hist_std == 0:
        if gap == 0:
            return 0.0
        return math.inf if gap > 0 else -math.inf
    return gap / hist_std


def surprise_label(sigma: float | None) -> str:
    """Bucket a sigma value into a fixed enum.

    | sigma | < 1   in_line
    | 1 ≤ s < 2     above_1sigma  / below_1sigma
    | 2 ≤ s < 3     above_2sigma  / below_2sigma
    | s ≥ 3         extreme_above_3sigma / extreme_below_3sigma
    """
    if sigma is None:
        return "no_consensus"
    abs_s = abs(sigma)
    direction = "above" if sigma > 0 else "below"
    if abs_s < 1.0:
        return "in_line"
    if abs_s < 2.0:
        return f"{direction}_1sigma"
    if abs_s < 3.0:
        return f"{direction}_2sigma"
    return f"extreme_{direction}_3sigma"


# ---------------------------------------------------------------------------
# Trend (last 3 values)
# ---------------------------------------------------------------------------

def trend_label(series, window: int = 3) -> str:
    """Look at the last ``window`` values; return:

    - "accelerating" — strictly increasing
    - "decelerating" — strictly decreasing
    - "flat"         — otherwise (mixed / equal)
    - "unknown"      — fewer than ``window`` points available
    """
    vals = _to_list(series)
    if len(vals) < window:
        return "unknown"
    tail = vals[-window:]
    if all(tail[i] > tail[i - 1] for i in range(1, window)):
        return "accelerating"
    if all(tail[i] < tail[i - 1] for i in range(1, window)):
        return "decelerating"
    return "flat"


__all__ = [
    "mom_pct",
    "yoy_pct",
    "mom_change_k",
    "qoq_annualized",
    "four_week_ma",
    "latest_level",
    "surprise_pct",
    "surprise_sigma",
    "surprise_label",
    "trend_label",
]
