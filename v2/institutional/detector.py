"""Detect significant position changes between two consecutive 13F filings."""

from __future__ import annotations

from v2.institutional.models import PositionChange
from v2.observability import emit

# Position size floor — ignore dust positions (likely passive index allocations)
_MIN_VALUE = 10_000_000.0     # $10M

# Portfolio-percentage floor — only show "concentrated bets" (filter quant noise).
# Quant funds rotate 1000s of tiny positions every quarter; we only care about
# their high-conviction holdings. Value funds easily clear this.
_MIN_PCT_OF_PORTFOLIO = 0.005  # 0.5% of relevant portfolio

# Ratio thresholds for "significant" change
_INCREASE_THRESHOLD = 1.5     # current ≥ 1.5 × previous
_DECREASE_THRESHOLD = 0.5     # current ≤ 0.5 × previous


def detect_changes(
    *,
    cik: str,
    manager_name: str,
    quarter: str,
    current_positions: list[dict],
    prev_positions: list[dict],
    current_total: float,
    prev_total: float,
) -> list[PositionChange]:
    """Compare two quarters of holdings, return list of significant changes."""
    cur_by_cusip = {p["cusip"]: p for p in current_positions}
    prev_by_cusip = {p["cusip"]: p for p in prev_positions}
    all_cusips = set(cur_by_cusip) | set(prev_by_cusip)

    changes: list[PositionChange] = []
    for cusip in all_cusips:
        cur = cur_by_cusip.get(cusip)
        prev = prev_by_cusip.get(cusip)

        cur_value = float(cur["market_value"]) if cur else 0.0
        prev_value = float(prev["market_value"]) if prev else 0.0

        # Skip dust positions (looking at the *larger* of the two so we catch
        # both new tiny positions and exits of tiny positions).
        if max(cur_value, prev_value) < _MIN_VALUE:
            continue

        # Skip positions that are tiny relative to the manager's portfolio —
        # this is what filters out quant-fund market-making noise.
        cur_pct = (cur_value / current_total) if current_total > 0 else 0
        prev_pct = (prev_value / prev_total) if prev_total > 0 else 0
        if max(cur_pct, prev_pct) < _MIN_PCT_OF_PORTFOLIO:
            continue

        change_type = _classify(
            cur_shares=cur["shares"] if cur else 0,
            prev_shares=prev["shares"] if prev else 0,
        )
        if change_type is None:
            continue

        info = cur or prev
        changes.append(PositionChange(
            cik=cik,
            manager_name=manager_name,
            quarter=quarter,
            change_type=change_type,
            ticker=info.get("ticker"),
            issuer_name=info["issuer_name"],
            cusip=cusip,
            current_shares=int(cur["shares"]) if cur else 0,
            current_value=cur_value,
            current_pct=(cur_value / current_total) if current_total > 0 else 0,
            prev_shares=int(prev["shares"]) if prev else 0,
            prev_value=prev_value,
            prev_pct=(prev_value / prev_total) if prev_total > 0 else 0,
        ))

    # Sort: exits + new first (highest signal), then big increases, then decreases.
    # Within each type, larger absolute value first.
    type_order = {"exit": 0, "new": 1, "increase": 2, "decrease": 3}
    changes.sort(key=lambda c: (
        type_order.get(c.change_type, 99),
        -max(c.current_value, c.prev_value),
    ))
    emit(
        "transform",
        op="detect_changes",
        current_positions=len(current_positions),
        prior_positions=len(prev_positions) if prev_positions else 0,
        significant_changes=len(changes),
        manager=manager_name,
        quarter=quarter,
        changes=[
            {
                "ticker": c.ticker or (c.cusip or "")[:8],
                "issuer": (c.issuer_name or "")[:40],
                "action": c.change_type,  # "new" | "exit" | "increase" | "decrease"
                "current_value": c.current_value,
                "change_value": c.current_value - c.prev_value,
            }
            for c in changes
        ],
    )
    return changes


def _classify(*, cur_shares: int, prev_shares: int) -> str | None:
    """Return one of new/exit/increase/decrease, or None if not significant."""
    if prev_shares == 0 and cur_shares > 0:
        return "new"
    if prev_shares > 0 and cur_shares == 0:
        return "exit"
    if cur_shares > 0 and prev_shares > 0:
        ratio = cur_shares / prev_shares
        if ratio >= _INCREASE_THRESHOLD:
            return "increase"
        if ratio <= _DECREASE_THRESHOLD:
            return "decrease"
    return None
