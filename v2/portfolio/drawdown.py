"""Peak-trough drawdown over the 1-month window.

Two scalars, both **non-negative magnitudes** (Stage 5 convention):

- ``current_drawdown_pct`` ≥ 0 — distance from the running peak as a
  positive fraction. ``0.034`` = "3.4% below the recent peak". At a
  fresh all-time high, ``current_drawdown_pct == 0``.
- ``max_drawdown_pct`` ≥ 0 — worst peak-to-trough drop observed in the
  window, also a positive magnitude.

This contract change happened in Stage 5 because:

1. Renderers want to always prepend ``-`` (drawdown is always a loss);
   storing a signed value just to flip it back was redundant.
2. Synthetic test data was easy to get wrong (Stage 2.5 dry-run accidentally
   produced ``+12.00%`` for max drawdown). With a non-negative invariant
   asserted at compute time, that class of bug is impossible.
3. ``compute_importance`` already used ``abs()`` defensively, so it
   doesn't care which convention we use.

Edge cases (unchanged):

- Single-point history → both fields = 0.0 (peak == current, no trough yet).
- Empty equity series → ``DrawdownMetrics.unavailable()`` (all None).
- Alpaca down → warnings collected, unavailable() returned.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from typing import Any

from v2.portfolio.models import DrawdownMetrics

logger = logging.getLogger(__name__)


def compute_drawdown(
    broker: Any | None = None,
    *,
    today_realtime_value: float | None = None,
) -> tuple[DrawdownMetrics, list[str]]:
    """Compute current + max drawdown over 1M of daily equity history.

    Returns ``(metrics, warnings)``. ``current_drawdown_pct`` and
    ``max_drawdown_pct`` are non-negative magnitudes (see module docstring).

    ``today_realtime_value`` (Phase 2.5 full): when supplied, appends
    today's intraday portfolio value to the EOD series before walking,
    so an intraday drop appears in ``current_drawdown_pct`` instead of
    waiting for tomorrow's EOD print. Fixes the prod UX bug where the
    portfolio card showed "今日 P/L 🔴 -3.72%" alongside
    "drawdown 0.00%" — the EOD series's last point was yesterday's
    close, so today's real-time decline was invisible to the drawdown
    walk. Backward compat: ``None`` → behaviour identical to Phase 2.

    Idempotent on same-day reruns: if the latest series point already
    bears today's ISO date, the realtime point replaces it instead of
    appending — avoids inflating the series length and double-counting
    today.
    """
    if broker is None:
        from v2 import broker as _broker
        broker = _broker

    warnings: list[str] = []

    try:
        hist = broker.get_portfolio_history(period="1M", timeframe="1D")
    except Exception as exc:
        msg = f"Alpaca 历史 equity 不可用：{exc}"
        logger.warning(msg)
        warnings.append(msg)
        return DrawdownMetrics.unavailable(), warnings

    equity = list(hist.get("equity") or [])
    timestamps = list(hist.get("timestamp") or [])

    # Phase 2.5 full — append today's realtime value if supplied and
    # the EOD series doesn't already cover today. This makes a same-day
    # intraday drawdown visible in the card.
    if today_realtime_value is not None and equity:
        today_d = date.today()
        last_date = _ts_to_date(timestamps[-1]) if timestamps else None
        # Synthesize a timestamp for today at UTC midnight so _ts_to_date
        # round-trips to today's ISO date. Alpaca's daily bars are unix
        # seconds at midnight UTC; we match that shape.
        today_unix = int(
            datetime(
                today_d.year, today_d.month, today_d.day, tzinfo=timezone.utc,
            ).timestamp()
        )
        if last_date == today_d:
            # EOD print already has today — overwrite with the fresher
            # realtime value rather than appending a duplicate point.
            equity[-1] = float(today_realtime_value)
            timestamps[-1] = today_unix
        else:
            equity.append(float(today_realtime_value))
            timestamps.append(today_unix)

    if not equity:
        return DrawdownMetrics.unavailable(), warnings

    if len(equity) == 1:
        peak_value = equity[0]
        peak_date = _ts_to_date(timestamps[0]) if timestamps else None
        return DrawdownMetrics(
            current_drawdown_pct=0.0,
            max_drawdown_pct=0.0,
            peak_value=peak_value,
            peak_date=peak_date,
        ), warnings

    # Walk the series tracking running peak + worst drawdown seen.
    # Internally we compute signed drops (trough - peak)/peak; flip to
    # magnitude at the end so the contract holds.
    running_peak = equity[0]
    running_peak_idx = 0
    worst_signed_dd = 0.0       # ≤ 0 internally
    worst_signed_peak_idx = 0

    for i, val in enumerate(equity):
        if val > running_peak:
            running_peak = val
            running_peak_idx = i
            continue
        signed = (val - running_peak) / running_peak if running_peak > 0 else 0.0
        if signed < worst_signed_dd:
            worst_signed_dd = signed
            worst_signed_peak_idx = running_peak_idx

    current = equity[-1]
    current_signed = (current - running_peak) / running_peak if running_peak > 0 else 0.0

    # Magnitude convention: drawdown ≥ 0 always.
    max_dd_mag = abs(worst_signed_dd)
    current_dd_mag = abs(current_signed)

    # Sanity — internal invariant. ``worst_signed_dd`` was monotonically
    # non-increasing and starts at 0, so after abs() the result is ≥ 0.
    assert max_dd_mag >= 0.0, f"max drawdown magnitude must be ≥ 0, got {max_dd_mag}"
    assert current_dd_mag >= 0.0, f"current drawdown magnitude must be ≥ 0, got {current_dd_mag}"

    # The peak attributed to the user is the one that produced max_dd —
    # that's the "from this peak you fell N%" story.
    peak_idx = worst_signed_peak_idx if max_dd_mag > 0 else running_peak_idx
    peak_value = equity[peak_idx]
    peak_date = _ts_to_date(timestamps[peak_idx]) if peak_idx < len(timestamps) else None

    return DrawdownMetrics(
        current_drawdown_pct=current_dd_mag,
        max_drawdown_pct=max_dd_mag,
        peak_value=peak_value,
        peak_date=peak_date,
    ), warnings


def _ts_to_date(ts: int | float) -> date | None:
    """Alpaca history timestamps are unix seconds. Convert to ISO date."""
    try:
        return datetime.fromtimestamp(int(ts), tz=timezone.utc).date()
    except (TypeError, ValueError, OSError):
        return None
