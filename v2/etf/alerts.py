"""ARK significant-rebalance alerts — Phase 5a.

Consumes the day-over-day diff dicts produced by
:func:`v2.etf.detector.compute_daily_changes` and filters them into a
narrow set of actionable :class:`ArkAlert` rows for the ⑬ ARK Alerts
cron (Mon-Fri 08:30 ET) to push.

Why this layer exists:

⑤ ETF Daily Snapshot already fetches the CSVs and computes changes —
but its output is **archive-only** (dashboard feed). ⑬ wants the same
diff signal but as a Telegram-pushable alert with priority gating.
This module is the alerts classifier sitting on top of the existing
``v2.etf`` machinery; nothing in client.py / detector.py / tracker.py
changes.

Thresholds (Stage 0 calibration choices):

- ``new_position``  → today's weight ≥ 0.5%
- ``liquidated``    → yesterday's weight ≥ 0.5%
- ``increase``      → relative share change ≥ +20%
- ``decrease``      → relative share change ≤ -20%

Anything below threshold is filtered. Multi-fund coordination (same
ticker, same direction, ≥2 funds same day) marks ``is_multi_fund=True``
so the Stage-2 priority layer can apply a +15 escalation. User-universe
membership (held + watchlist) marks ``is_in_user_universe=True`` for a
separate +10 escalation.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Literal

from v2.etf.models import ETFHolding


# ---------------------------------------------------------------------------
# Thresholds — single source of truth, tested via test_alerts.py
# ---------------------------------------------------------------------------

# Weight thresholds use CSV-native percentage units (e.g. 0.5 means 0.5%).
# Detector's compute_daily_changes() preserves this unit.
_NEW_POSITION_MIN_WEIGHT_PCT = 0.5
_LIQUIDATED_MIN_PRIOR_WEIGHT_PCT = 0.5

# Relative share change uses decimal fraction (0.20 = 20%). Detector's
# shares_diff_pct field is signed: + for increase, - for decrease.
_REBALANCE_MIN_RELATIVE = 0.20


ArkAction = Literal["new_position", "liquidated", "increase", "decrease"]


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ArkAlert:
    """One actionable ARK rebalance signal for a single (fund, ticker).

    Fields stay native to the underlying CSV units so the formatter can
    render them without re-conversion:

    - ``today_weight`` / ``yesterday_weight`` are percentage units
      (e.g. ``1.85`` means 1.85%). ``None`` when the action makes the
      side undefined (``new_position`` has no yesterday; ``liquidated``
      has no today).
    - ``weight_change_relative`` is a signed decimal fraction
      (``+0.25`` means +25% relative to yesterday's share count). For
      ``new_position`` we set ``+1.0`` (entirely new); for
      ``liquidated`` we set ``-1.0`` (entirely gone).
    - ``shares_change`` is the signed integer delta in share count
      (``+250_000`` for new buys, ``-180_000`` for trims).
    - ``market_value_usd`` represents the dollar weight of the action:
      today's full position for ``new_position`` / ``increase``; the
      lost yesterday position for ``liquidated`` / ``decrease`` (so
      readers see "$31.5M new" or "$12.1M gone").
    """

    fund: str
    ticker: str
    company: str
    action: ArkAction
    yesterday_weight: float | None
    today_weight: float | None
    weight_change_relative: float
    shares_change: int
    market_value_usd: float
    is_in_user_universe: bool
    is_multi_fund: bool


@dataclass
class ArkScanResult:
    """Aggregated output of one ⑬ run.

    ``funds_scanned`` is the funds that produced usable data (today's
    CSV + yesterday's baseline both present); ``funds_attempted`` is
    every fund the cron tried — used by the summary card to render
    "(N/M)" coverage fraction so partial failures are transparent.

    ``warnings`` carries per-fund failures the cron / dashboard can
    surface without aborting (e.g. one ARK CSV 404 should not block
    the other 3 funds' alerts)."""

    scan_date: str
    funds_scanned: list[str]
    alerts: list[ArkAlert] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    funds_attempted: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_alerts(
    changes_by_fund: dict[str, list[dict]],
    today_holdings_by_fund: dict[str, list[ETFHolding]],
    yesterday_rows_by_fund: dict[str, list[dict]],
    user_universe: set[str],
) -> list[ArkAlert]:
    """Filter detector output → actionable :class:`ArkAlert` list.

    ``changes_by_fund`` is the dict-of-list shape returned by
    :func:`v2.etf.detector.compute_daily_changes` per-fund — keys are
    fund symbols (``"ARKK"`` etc.), values are the change-dict lists.

    ``today_holdings_by_fund`` + ``yesterday_rows_by_fund`` are needed
    only to look up ``market_value`` (the detector dict doesn't carry
    dollar weight — keeping detector schema unchanged was a Stage 0
    constraint).

    ``user_universe`` should be the union of broker holdings and the
    user's bot watchlist (uppercased ticker strings). Membership marks
    ``is_in_user_universe`` so Stage 2's priority can bump.

    Returns alerts in insertion order (per-fund per-change). Multi-fund
    coordination is computed AFTER the per-fund pass so we can detect
    "same ticker, same direction, ≥2 funds" → mark all matching alerts.
    """
    if not changes_by_fund:
        return []

    raw: list[ArkAlert] = []

    for fund, changes in changes_by_fund.items():
        today_map = _index_by_ticker_today(today_holdings_by_fund.get(fund, []))
        yest_map = _index_by_ticker_yesterday(yesterday_rows_by_fund.get(fund, []))

        for change in changes:
            classified = _maybe_classify_change(
                fund=fund,
                change=change,
                today_map=today_map,
                yest_map=yest_map,
                user_universe=user_universe,
            )
            if classified is not None:
                raw.append(classified)

    return _mark_multi_fund(raw)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

# Map ArkAction → coarse direction for multi-fund detection. "new_position"
# and "increase" both count as a buy; "liquidated" and "decrease" as sell.
_DIRECTION: dict[str, str] = {
    "new_position": "buy",
    "increase":     "buy",
    "liquidated":   "sell",
    "decrease":     "sell",
}


def _index_by_ticker_today(holdings: list[ETFHolding]) -> dict[str, ETFHolding]:
    return {h.ticker: h for h in holdings if h.ticker}


def _index_by_ticker_yesterday(rows: list[dict]) -> dict[str, dict]:
    return {r["ticker"]: r for r in rows if r.get("ticker")}


def _maybe_classify_change(
    *,
    fund: str,
    change: dict,
    today_map: dict[str, ETFHolding],
    yest_map: dict[str, dict],
    user_universe: set[str],
) -> ArkAlert | None:
    """Apply the threshold ladder. Returns None to filter out."""
    ticker = change.get("ticker")
    if not ticker:
        return None

    is_new = bool(change.get("is_new"))
    is_exit = bool(change.get("is_exit"))
    today_weight = float(change.get("weight_pct") or 0.0)
    weight_diff_pp = float(change.get("weight_diff_pp") or 0.0)
    shares_diff_pct = float(change.get("shares_diff_pct") or 0.0)
    shares_diff = float(change.get("shares_diff") or 0.0)

    # Yesterday's weight derived from detector: today_weight - diff_pp.
    # For exits today_weight is 0, so yesterday = -diff_pp.
    yesterday_weight_raw = today_weight - weight_diff_pp

    today_h = today_map.get(ticker)
    yest_row = yest_map.get(ticker)

    if is_new:
        if today_weight < _NEW_POSITION_MIN_WEIGHT_PCT:
            return None
        action: ArkAction = "new_position"
        market_value = float(today_h.market_value) if today_h is not None else 0.0
        return ArkAlert(
            fund=fund,
            ticker=ticker,
            company=str(change.get("company") or ""),
            action=action,
            yesterday_weight=None,
            today_weight=today_weight,
            weight_change_relative=1.0,         # entirely new
            shares_change=int(shares_diff),
            market_value_usd=market_value,
            is_in_user_universe=ticker in user_universe,
            is_multi_fund=False,
        )

    if is_exit:
        if yesterday_weight_raw < _LIQUIDATED_MIN_PRIOR_WEIGHT_PCT:
            return None
        action = "liquidated"
        market_value = float(yest_row["market_value"]) if yest_row else 0.0
        return ArkAlert(
            fund=fund,
            ticker=ticker,
            company=str(change.get("company") or ""),
            action=action,
            yesterday_weight=yesterday_weight_raw,
            today_weight=None,
            weight_change_relative=-1.0,        # entirely gone
            shares_change=int(shares_diff),     # already negative
            market_value_usd=market_value,
            is_in_user_universe=ticker in user_universe,
            is_multi_fund=False,
        )

    # Increase / decrease — both gated by |relative change| ≥ threshold.
    if abs(shares_diff_pct) < _REBALANCE_MIN_RELATIVE:
        return None

    action = "increase" if shares_diff_pct > 0 else "decrease"

    if today_h is not None and yest_row is not None:
        if action == "increase":
            market_value = max(
                0.0, float(today_h.market_value) - float(yest_row["market_value"]),
            )
        else:
            market_value = max(
                0.0, float(yest_row["market_value"]) - float(today_h.market_value),
            )
    else:
        market_value = 0.0

    return ArkAlert(
        fund=fund,
        ticker=ticker,
        company=str(change.get("company") or ""),
        action=action,
        yesterday_weight=yesterday_weight_raw,
        today_weight=today_weight,
        weight_change_relative=shares_diff_pct,
        shares_change=int(shares_diff),
        market_value_usd=market_value,
        is_in_user_universe=ticker in user_universe,
        is_multi_fund=False,
    )


def _mark_multi_fund(alerts: list[ArkAlert]) -> list[ArkAlert]:
    """Re-emit alerts with ``is_multi_fund=True`` where ≥2 funds hit the
    same (ticker, direction) today.

    Frozen dataclasses → we rebuild rather than mutate. Order is
    preserved so the cron renders in the same order it received."""
    if not alerts:
        return []

    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for idx, a in enumerate(alerts):
        direction = _DIRECTION.get(a.action)
        if direction is None:
            continue
        groups[(a.ticker, direction)].append(idx)

    multi_indices: set[int] = set()
    for indices in groups.values():
        if len(indices) >= 2:
            multi_indices.update(indices)

    if not multi_indices:
        return alerts

    out: list[ArkAlert] = []
    for idx, a in enumerate(alerts):
        if idx in multi_indices:
            out.append(_with(a, is_multi_fund=True))
        else:
            out.append(a)
    return out


def _with(alert: ArkAlert, **kwargs) -> ArkAlert:
    """Functional replace for a frozen dataclass."""
    from dataclasses import replace
    return replace(alert, **kwargs)


__all__ = [
    "ArkAction",
    "ArkAlert",
    "ArkScanResult",
    "classify_alerts",
]
