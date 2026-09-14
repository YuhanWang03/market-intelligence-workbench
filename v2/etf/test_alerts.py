"""Phase 5a Stage 1 smoke tests for ARK alerts classifier.

Pure-Python tests — no httpx, no SQLite, no ARK CSV. The classifier
takes the detector's dict-of-list output (already battle-tested by
⑤ since Phase 0) plus today/yesterday holdings lookups and emits
ArkAlert rows.

Threshold tests assert the ladder is correctly gated:
- new_position needs ≥ 0.5% today_weight
- liquidated needs ≥ 0.5% yesterday_weight
- increase/decrease need |relative| ≥ 20%

Multi-fund + user-universe tests check the cross-cutting tags applied
after per-fund classification.
"""

from __future__ import annotations

import dataclasses

import pytest

from v2.etf.alerts import (
    ArkAlert,
    ArkScanResult,
    classify_alerts,
)
from v2.etf.models import ETFHolding


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

def _today_holding(
    etf: str, ticker: str, *, shares: float = 100_000.0,
    market_value: float = 10_000_000.0, weight_pct: float = 1.5,
    company: str = "",
) -> ETFHolding:
    return ETFHolding(
        etf=etf, date="2026-06-09", ticker=ticker, cusip=None,
        company=company or ticker, shares=shares,
        market_value=market_value, weight_pct=weight_pct,
    )


def _yest_row(
    etf: str, ticker: str, *, shares: float = 100_000.0,
    market_value: float = 10_000_000.0, weight_pct: float = 1.5,
    company: str = "",
) -> dict:
    return {
        "etf": etf, "date": "2026-06-06", "ticker": ticker, "cusip": None,
        "company": company or ticker, "shares": shares,
        "market_value": market_value, "weight_pct": weight_pct,
    }


def _change_new(etf: str, ticker: str, *, today_weight: float = 1.5,
                shares: float = 100_000.0, company: str = "") -> dict:
    return {
        "etf": etf, "ticker": ticker, "company": company or ticker,
        "shares_diff": float(shares), "shares_diff_pct": 0.0,
        "weight_pct": today_weight, "weight_diff_pp": today_weight,
        "is_new": True, "is_exit": False,
    }


def _change_exit(etf: str, ticker: str, *, yesterday_weight: float = 0.8,
                 shares: float = 50_000.0, company: str = "") -> dict:
    return {
        "etf": etf, "ticker": ticker, "company": company or ticker,
        "shares_diff": -float(shares), "shares_diff_pct": -1.0,
        "weight_pct": 0.0, "weight_diff_pp": -yesterday_weight,
        "is_new": False, "is_exit": True,
    }


def _change_rebalance(
    etf: str, ticker: str, *, today_weight: float = 1.5,
    weight_diff_pp: float = 0.3, shares_diff: float = 50_000.0,
    shares_diff_pct: float = 0.25, company: str = "",
) -> dict:
    return {
        "etf": etf, "ticker": ticker, "company": company or ticker,
        "shares_diff": shares_diff, "shares_diff_pct": shares_diff_pct,
        "weight_pct": today_weight, "weight_diff_pp": weight_diff_pp,
        "is_new": False, "is_exit": False,
    }


# ---------------------------------------------------------------------------
# new_position threshold
# ---------------------------------------------------------------------------

def test_classify_new_position_above_threshold_p1():
    """new_position with today_weight 1.5% (≥ 0.5% threshold) → alert."""
    change = _change_new("ARKK", "NVDA", today_weight=1.5, shares=250_000.0)
    today_h = _today_holding(
        "ARKK", "NVDA", shares=250_000.0,
        market_value=31_500_000.0, weight_pct=1.5,
    )

    alerts = classify_alerts(
        changes_by_fund={"ARKK": [change]},
        today_holdings_by_fund={"ARKK": [today_h]},
        yesterday_rows_by_fund={"ARKK": []},
        user_universe=set(),
    )

    assert len(alerts) == 1
    a = alerts[0]
    assert a.fund == "ARKK"
    assert a.ticker == "NVDA"
    assert a.action == "new_position"
    assert a.yesterday_weight is None
    assert a.today_weight == 1.5
    assert a.shares_change == 250_000
    assert a.market_value_usd == 31_500_000.0
    assert a.weight_change_relative == 1.0
    assert a.is_in_user_universe is False
    assert a.is_multi_fund is False


def test_classify_new_position_below_threshold_filtered():
    """new_position with today_weight 0.3% (< 0.5%) → no alert."""
    change = _change_new("ARKK", "TINY", today_weight=0.3, shares=5_000.0)
    today_h = _today_holding(
        "ARKK", "TINY", shares=5_000.0,
        market_value=500_000.0, weight_pct=0.3,
    )

    alerts = classify_alerts(
        changes_by_fund={"ARKK": [change]},
        today_holdings_by_fund={"ARKK": [today_h]},
        yesterday_rows_by_fund={"ARKK": []},
        user_universe=set(),
    )
    assert alerts == []


# ---------------------------------------------------------------------------
# liquidated threshold
# ---------------------------------------------------------------------------

def test_classify_liquidated_above_threshold_p1():
    """liquidated with yesterday_weight 0.8% → action='liquidated'."""
    change = _change_exit("ARKQ", "IRBT", yesterday_weight=0.8, shares=60_000.0)
    yest = _yest_row(
        "ARKQ", "IRBT", shares=60_000.0,
        market_value=12_100_000.0, weight_pct=0.8,
    )

    alerts = classify_alerts(
        changes_by_fund={"ARKQ": [change]},
        today_holdings_by_fund={"ARKQ": []},
        yesterday_rows_by_fund={"ARKQ": [yest]},
        user_universe=set(),
    )

    assert len(alerts) == 1
    a = alerts[0]
    assert a.action == "liquidated"
    assert a.yesterday_weight == pytest.approx(0.8)
    assert a.today_weight is None
    assert a.shares_change == -60_000
    assert a.market_value_usd == 12_100_000.0
    assert a.weight_change_relative == -1.0


def test_classify_liquidated_below_threshold_filtered():
    """liquidated where yesterday was only 0.2% → not significant."""
    change = _change_exit("ARKQ", "DUST", yesterday_weight=0.2, shares=2_000.0)
    yest = _yest_row(
        "ARKQ", "DUST", shares=2_000.0,
        market_value=200_000.0, weight_pct=0.2,
    )

    alerts = classify_alerts(
        changes_by_fund={"ARKQ": [change]},
        today_holdings_by_fund={"ARKQ": []},
        yesterday_rows_by_fund={"ARKQ": [yest]},
        user_universe=set(),
    )
    assert alerts == []


# ---------------------------------------------------------------------------
# increase / decrease relative threshold
# ---------------------------------------------------------------------------

def test_classify_increase_20pct_relative_p1():
    """+25% share change → action='increase'."""
    change = _change_rebalance(
        "ARKK", "TSLA",
        today_weight=10.2, weight_diff_pp=2.1,
        shares_diff=500_000.0, shares_diff_pct=0.25,
    )
    today_h = _today_holding(
        "ARKK", "TSLA", shares=2_500_000.0,
        market_value=625_000_000.0, weight_pct=10.2,
    )
    yest = _yest_row(
        "ARKK", "TSLA", shares=2_000_000.0,
        market_value=500_000_000.0, weight_pct=8.1,
    )

    alerts = classify_alerts(
        changes_by_fund={"ARKK": [change]},
        today_holdings_by_fund={"ARKK": [today_h]},
        yesterday_rows_by_fund={"ARKK": [yest]},
        user_universe=set(),
    )
    assert len(alerts) == 1
    a = alerts[0]
    assert a.action == "increase"
    assert a.today_weight == 10.2
    assert a.yesterday_weight == pytest.approx(8.1)
    assert a.weight_change_relative == 0.25
    assert a.shares_change == 500_000
    assert a.market_value_usd == 125_000_000.0


def test_classify_decrease_below_threshold_filtered():
    """-10% share change → below 20% threshold → no alert."""
    change = _change_rebalance(
        "ARKK", "SMALL",
        today_weight=1.5, weight_diff_pp=-0.1,
        shares_diff=-10_000.0, shares_diff_pct=-0.10,
    )
    today_h = _today_holding("ARKK", "SMALL", shares=90_000.0)
    yest = _yest_row("ARKK", "SMALL", shares=100_000.0)

    alerts = classify_alerts(
        changes_by_fund={"ARKK": [change]},
        today_holdings_by_fund={"ARKK": [today_h]},
        yesterday_rows_by_fund={"ARKK": [yest]},
        user_universe=set(),
    )
    assert alerts == []


def test_classify_decrease_above_threshold_alerts():
    """-30% share change → action='decrease', market_value = yesterday - today."""
    change = _change_rebalance(
        "ARKK", "OLDIE",
        today_weight=1.0, weight_diff_pp=-0.5,
        shares_diff=-30_000.0, shares_diff_pct=-0.30,
    )
    today_h = _today_holding(
        "ARKK", "OLDIE", shares=70_000.0,
        market_value=7_000_000.0, weight_pct=1.0,
    )
    yest = _yest_row(
        "ARKK", "OLDIE", shares=100_000.0,
        market_value=10_000_000.0, weight_pct=1.5,
    )

    alerts = classify_alerts(
        changes_by_fund={"ARKK": [change]},
        today_holdings_by_fund={"ARKK": [today_h]},
        yesterday_rows_by_fund={"ARKK": [yest]},
        user_universe=set(),
    )
    assert len(alerts) == 1
    a = alerts[0]
    assert a.action == "decrease"
    assert a.weight_change_relative == -0.30
    assert a.market_value_usd == 3_000_000.0   # 10M yesterday − 7M today


# ---------------------------------------------------------------------------
# user_universe / multi-fund tags
# ---------------------------------------------------------------------------

def test_classify_user_universe_boost():
    """Ticker in held + watchlist → is_in_user_universe=True."""
    change = _change_new("ARKK", "NVDA", today_weight=1.5, shares=250_000.0)
    today_h = _today_holding(
        "ARKK", "NVDA", shares=250_000.0,
        market_value=31_500_000.0, weight_pct=1.5,
    )

    alerts = classify_alerts(
        changes_by_fund={"ARKK": [change]},
        today_holdings_by_fund={"ARKK": [today_h]},
        yesterday_rows_by_fund={"ARKK": []},
        user_universe={"NVDA", "TSLA"},
    )
    assert len(alerts) == 1
    assert alerts[0].is_in_user_universe is True


def test_classify_multi_fund_coordinated():
    """Same ticker increase in 2 funds same day → both marked is_multi_fund."""
    change_arkk = _change_rebalance(
        "ARKK", "TSMC",
        today_weight=3.5, weight_diff_pp=1.2,
        shares_diff=100_000.0, shares_diff_pct=0.30,
    )
    change_arkq = _change_rebalance(
        "ARKQ", "TSMC",
        today_weight=2.0, weight_diff_pp=0.9,
        shares_diff=80_000.0, shares_diff_pct=0.40,
    )
    today_arkk = _today_holding(
        "ARKK", "TSMC", shares=300_000.0,
        market_value=45_000_000.0, weight_pct=3.5,
    )
    today_arkq = _today_holding(
        "ARKQ", "TSMC", shares=200_000.0,
        market_value=30_000_000.0, weight_pct=2.0,
    )
    yest_arkk = _yest_row(
        "ARKK", "TSMC", shares=200_000.0,
        market_value=30_000_000.0, weight_pct=2.3,
    )
    yest_arkq = _yest_row(
        "ARKQ", "TSMC", shares=120_000.0,
        market_value=18_000_000.0, weight_pct=1.1,
    )

    alerts = classify_alerts(
        changes_by_fund={
            "ARKK": [change_arkk], "ARKQ": [change_arkq],
        },
        today_holdings_by_fund={
            "ARKK": [today_arkk], "ARKQ": [today_arkq],
        },
        yesterday_rows_by_fund={
            "ARKK": [yest_arkk], "ARKQ": [yest_arkq],
        },
        user_universe=set(),
    )
    assert len(alerts) == 2
    assert all(a.is_multi_fund for a in alerts)
    assert {a.fund for a in alerts} == {"ARKK", "ARKQ"}


def test_classify_multi_fund_3_funds_extreme():
    """3 funds same-direction same-ticker → all 3 marked is_multi_fund."""
    changes = {
        fund: [_change_rebalance(
            fund, "TSMC",
            today_weight=2.0 + i, weight_diff_pp=0.5,
            shares_diff=50_000.0, shares_diff_pct=0.25,
        )]
        for i, fund in enumerate(["ARKK", "ARKQ", "ARKX"])
    }
    today = {
        fund: [_today_holding(fund, "TSMC", weight_pct=2.0 + i,
                              market_value=20_000_000.0 * (i + 1))]
        for i, fund in enumerate(["ARKK", "ARKQ", "ARKX"])
    }
    yest = {
        fund: [_yest_row(fund, "TSMC", weight_pct=1.5,
                         market_value=10_000_000.0 * (i + 1))]
        for i, fund in enumerate(["ARKK", "ARKQ", "ARKX"])
    }

    alerts = classify_alerts(
        changes_by_fund=changes,
        today_holdings_by_fund=today,
        yesterday_rows_by_fund=yest,
        user_universe=set(),
    )
    assert len(alerts) == 3
    assert all(a.is_multi_fund for a in alerts)


def test_classify_multi_fund_opposite_directions_not_coordinated():
    """Same ticker buy in ARKK + sell in ARKQ → NOT coordinated."""
    change_buy = _change_rebalance(
        "ARKK", "TSMC",
        today_weight=3.5, weight_diff_pp=1.2,
        shares_diff=100_000.0, shares_diff_pct=0.30,
    )
    change_sell = _change_rebalance(
        "ARKQ", "TSMC",
        today_weight=1.0, weight_diff_pp=-0.6,
        shares_diff=-50_000.0, shares_diff_pct=-0.35,
    )
    today = {
        "ARKK": [_today_holding("ARKK", "TSMC", weight_pct=3.5,
                                shares=300_000.0,
                                market_value=45_000_000.0)],
        "ARKQ": [_today_holding("ARKQ", "TSMC", weight_pct=1.0,
                                shares=80_000.0,
                                market_value=12_000_000.0)],
    }
    yest = {
        "ARKK": [_yest_row("ARKK", "TSMC", weight_pct=2.3,
                           shares=200_000.0,
                           market_value=30_000_000.0)],
        "ARKQ": [_yest_row("ARKQ", "TSMC", weight_pct=1.6,
                           shares=130_000.0,
                           market_value=20_000_000.0)],
    }

    alerts = classify_alerts(
        changes_by_fund={"ARKK": [change_buy], "ARKQ": [change_sell]},
        today_holdings_by_fund=today,
        yesterday_rows_by_fund=yest,
        user_universe=set(),
    )
    assert len(alerts) == 2
    assert not any(a.is_multi_fund for a in alerts), (
        "opposite-direction same-ticker should NOT be tagged coordinated"
    )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_classify_empty_changes_returns_empty():
    assert classify_alerts(
        changes_by_fund={},
        today_holdings_by_fund={},
        yesterday_rows_by_fund={},
        user_universe={"NVDA"},
    ) == []


def test_classify_handles_zero_weight_safely():
    """Change dict with weight 0 / shares_diff 0 should silently drop."""
    change = {
        "etf": "ARKK", "ticker": "ZERO", "company": "Zero Inc",
        "shares_diff": 0.0, "shares_diff_pct": 0.0,
        "weight_pct": 0.0, "weight_diff_pp": 0.0,
        "is_new": False, "is_exit": False,
    }
    alerts = classify_alerts(
        changes_by_fund={"ARKK": [change]},
        today_holdings_by_fund={"ARKK": []},
        yesterday_rows_by_fund={"ARKK": []},
        user_universe=set(),
    )
    assert alerts == []


def test_classify_handles_missing_ticker():
    """Change without ticker field skipped silently."""
    change = {
        "etf": "ARKK", "ticker": "", "company": "?",
        "shares_diff": 50_000.0, "shares_diff_pct": 0.30,
        "weight_pct": 1.5, "weight_diff_pp": 0.3,
        "is_new": False, "is_exit": False,
    }
    alerts = classify_alerts(
        changes_by_fund={"ARKK": [change]},
        today_holdings_by_fund={"ARKK": []},
        yesterday_rows_by_fund={"ARKK": []},
        user_universe=set(),
    )
    assert alerts == []


# ---------------------------------------------------------------------------
# Dataclass shape pins
# ---------------------------------------------------------------------------

def test_alert_dataclass_immutability():
    """ArkAlert is frozen — fields shouldn't be reassignable."""
    a = ArkAlert(
        fund="ARKK", ticker="NVDA", company="NVIDIA",
        action="new_position",
        yesterday_weight=None, today_weight=1.5,
        weight_change_relative=1.0,
        shares_change=250_000,
        market_value_usd=31_500_000.0,
        is_in_user_universe=False,
        is_multi_fund=False,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.is_multi_fund = True


def test_scan_result_aggregates_warnings():
    """ArkScanResult should default-init alerts + warnings to empty
    lists (mutable defaults via dataclass field(default_factory=list))."""
    result = ArkScanResult(scan_date="2026-06-09", funds_scanned=["ARKK"])
    assert result.alerts == []
    assert result.warnings == []
    result.warnings.append("ARKQ fetch 503")
    assert result.warnings == ["ARKQ fetch 503"]
    result.alerts.append(ArkAlert(
        fund="ARKK", ticker="NVDA", company="NVIDIA",
        action="new_position",
        yesterday_weight=None, today_weight=1.5,
        weight_change_relative=1.0, shares_change=250_000,
        market_value_usd=31_500_000.0,
        is_in_user_universe=True, is_multi_fund=False,
    ))
    assert len(result.alerts) == 1
