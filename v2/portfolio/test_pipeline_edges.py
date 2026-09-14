"""Pipeline edge-case tests — sandbox runnable.

10 cases covering the failure / degenerate paths of
:func:`v2.portfolio.build_risk_report`. The Stage-1 smoke covered the
happy paths; this file pins the contract that sub-section failures
NEVER raise — they populate ``RiskReport.warnings`` for the formatter
to surface.

Mirrors the Phase-1 ``v2/earnings/test_pipeline.py`` pattern.
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


from v2.portfolio import build_risk_report   # noqa: E402
from v2.earnings.calendar import CalendarBatchResult   # noqa: E402
from v2.earnings.models import EarningsEvent   # noqa: E402


_TODAY = date.today()
_TODAY_ISO = _TODAY.isoformat()


def _future_iso(days: int) -> str:
    return (_TODAY + timedelta(days=days)).isoformat()


def _calendar_empty(tickers):
    return CalendarBatchResult(events={}, skipped_unsupported=[],
                               skipped_empty=[], errors=[])


# ---------------------------------------------------------------------------
# Helper: parametric broker builder
# ---------------------------------------------------------------------------

class _FakeBroker:
    """Configurable Alpaca shim. Each method can be set to raise or return
    a canned value. Lets each test specify exactly which sub-call fails."""

    def __init__(self, *, portfolio=None, pnl=None, history=None,
                 raise_on=None):
        self._portfolio = portfolio
        self._pnl = pnl
        self._history = history
        self._raise_on = set(raise_on or ())

    def get_portfolio(self):
        if "portfolio" in self._raise_on:
            raise RuntimeError("Alpaca 503")
        return self._portfolio or {"account": {}, "positions": []}

    def get_pnl(self):
        if "pnl" in self._raise_on:
            raise RuntimeError("Alpaca 503")
        return self._pnl or {"intraday_pl": 0.0, "intraday_pl_pct": 0.0}

    def get_portfolio_history(self, period, timeframe):
        if "history" in self._raise_on:
            raise RuntimeError("Alpaca history 503")
        return self._history or {"equity": [], "timestamp": []}


# ---------------------------------------------------------------------------
# 1. Total Alpaca outage
# ---------------------------------------------------------------------------

def test_pipeline_alpaca_down_returns_empty_report_no_exception():
    """Alpaca completely down → RiskReport with all-None numeric fields,
    no exception escapes. Warnings carry the failure reasons."""
    broker = _FakeBroker(raise_on={"portfolio", "pnl", "history"})

    report = build_risk_report(
        today_iso=_TODAY_ISO, broker=broker,
        calendar_fetcher=_calendar_empty,
    )

    assert report.snapshot_date == _TODAY_ISO
    assert report.positions == []
    assert report.portfolio_value == 0.0
    assert report.cash == 0.0
    assert report.cash_pct == 0.0
    assert report.invested_value == 0.0    # derived: max(0, 0-0)
    # Sub-sections degrade independently
    assert report.pnl.daily_pnl_pct is None
    assert report.pnl.weekly_pnl_pct is None
    assert report.pnl.monthly_pnl_pct is None
    assert report.drawdown.max_drawdown_pct is None
    assert report.drawdown.current_drawdown_pct is None
    # All 3 outages collected as warnings (positions + pnl + history)
    # Note: history is queried twice (pnl + drawdown) so we expect ≥ 3
    assert len(report.warnings) >= 3
    assert any("503" in w for w in report.warnings)


# ---------------------------------------------------------------------------
# 2. All cash, no positions
# ---------------------------------------------------------------------------

def test_pipeline_all_cash_no_positions():
    """100% cash → concentration/exposure return empty (well-formed,
    not None), PnL still computes from the account snapshot, drawdown
    is 0 magnitude on the single-point equity history."""
    broker = _FakeBroker(
        portfolio={
            "account": {"portfolio_value": 100_000.0, "cash": 100_000.0},
            "positions": [],
        },
        pnl={"intraday_pl": 0.0, "intraday_pl_pct": 0.0},
        history={"equity": [100_000.0], "timestamp": [1_700_000_000]},
    )

    report = build_risk_report(
        today_iso=_TODAY_ISO, broker=broker,
        calendar_fetcher=_calendar_empty,
    )

    assert report.portfolio_value == 100_000.0
    assert report.cash == 100_000.0
    assert report.cash_pct == 1.0
    assert report.invested_value == 0.0
    assert report.positions == []
    # Concentration/exposure empty but well-formed (no AttributeError downstream)
    assert report.concentration.top_1_pct == 0.0
    assert report.concentration.n_positions == 0
    assert report.concentration.hhi == 0.0
    assert report.exposure.by_sector == {}
    assert report.exposure.largest_sector == ""
    # PnL still valid (account snapshot works even with no positions)
    assert report.pnl.daily_pnl_pct == 0.0
    # Single-point history → 0 magnitude drawdown, not unavailable
    assert report.drawdown.max_drawdown_pct == 0.0
    assert report.earnings_risk_next_7d == []


# ---------------------------------------------------------------------------
# 3. Single-position concentration extreme
# ---------------------------------------------------------------------------

def test_pipeline_single_position():
    """1 position → top_1=100%, HHI=1.0 (perfectly concentrated)."""
    broker = _FakeBroker(
        portfolio={
            "account": {"portfolio_value": 100_000.0, "cash": 0.0},
            "positions": [{"symbol": "NVDA", "market_value": "100000"}],
        },
        pnl={"intraday_pl": 0.0, "intraday_pl_pct": 0.0},
        history={"equity": [100_000.0], "timestamp": [1_700_000_000]},
    )

    report = build_risk_report(
        today_iso=_TODAY_ISO, broker=broker,
        calendar_fetcher=_calendar_empty,
    )

    assert len(report.positions) == 1
    assert report.positions[0].ticker == "NVDA"
    assert report.concentration.top_1_pct == 1.0
    assert report.concentration.top_3_pct == 1.0
    assert report.concentration.top_5_pct == 1.0
    assert report.concentration.hhi == 1.0
    assert report.concentration.n_positions == 1


# ---------------------------------------------------------------------------
# 4. Partial outage: history endpoint dead, positions/daily survive
# ---------------------------------------------------------------------------

def test_pipeline_history_endpoint_fails_but_positions_succeed():
    """Independence contract: drawdown / weekly_pnl / monthly_pnl =
    None, but daily / concentration / exposure / earnings still good."""
    broker = _FakeBroker(
        portfolio={
            "account": {"portfolio_value": 100_000.0, "cash": 25_000.0},
            "positions": [
                {"symbol": "NVDA", "market_value": "30000"},
                {"symbol": "AAPL", "market_value": "25000"},
                {"symbol": "MSFT", "market_value": "20000"},
            ],
        },
        pnl={"intraday_pl": 500.0, "intraday_pl_pct": 0.005},
        raise_on={"history"},
    )

    report = build_risk_report(
        today_iso=_TODAY_ISO, broker=broker,
        calendar_fetcher=_calendar_empty,
    )

    # Survived
    assert len(report.positions) == 3
    assert report.pnl.daily_pnl_pct == 0.005
    assert report.concentration.n_positions == 3
    assert "SMH" in report.exposure.by_sector

    # Lost
    assert report.pnl.weekly_pnl_pct is None
    assert report.pnl.monthly_pnl_pct is None
    assert report.drawdown.max_drawdown_pct is None
    assert report.drawdown.current_drawdown_pct is None

    # History failure surfaces as warnings (pnl + drawdown both tried it)
    assert any("history" in w.lower() or "equity" in w for w in report.warnings)


# ---------------------------------------------------------------------------
# 5. yfinance earnings calendar failure → empty earnings_risk
# ---------------------------------------------------------------------------

def test_pipeline_yfinance_earnings_calendar_fails():
    """earnings calendar throws → empty earnings_risk_next_7d, warning
    captured, rest of report intact."""
    def boom_calendar(tickers):
        raise RuntimeError("yfinance 503")

    broker = _FakeBroker(
        portfolio={
            "account": {"portfolio_value": 100_000.0, "cash": 0.0},
            "positions": [{"symbol": "AAPL", "market_value": "100000"}],
        },
        pnl={"intraday_pl": 0.0, "intraday_pl_pct": 0.0},
        history={"equity": [100_000.0], "timestamp": [1_700_000_000]},
    )

    report = build_risk_report(
        today_iso=_TODAY_ISO, broker=broker,
        calendar_fetcher=boom_calendar,
    )

    assert report.earnings_risk_next_7d == []
    assert any("yfinance" in w for w in report.warnings)
    # Other sections still good
    assert report.concentration.top_1_pct == 1.0


# ---------------------------------------------------------------------------
# 6. Unmapped ticker → OTHER bucket (not silent fall-through to SPY)
# ---------------------------------------------------------------------------

def test_pipeline_position_with_unmapped_ticker():
    """A ticker not in TICKER_TO_SECTOR routes to the OTHER bucket
    (Stage 2.5's sector_bucket_for, NOT sector_etf_for which falls
    back to SPY). The position is still included in concentration."""
    broker = _FakeBroker(
        portfolio={
            "account": {"portfolio_value": 100_000.0, "cash": 0.0},
            "positions": [
                {"symbol": "AAPL", "market_value": "50000"},     # XLK
                {"symbol": "NEWIPO", "market_value": "50000"},   # OTHER
            ],
        },
        pnl={"intraday_pl": 0.0, "intraday_pl_pct": 0.0},
        history={"equity": [100_000.0], "timestamp": [1_700_000_000]},
    )

    report = build_risk_report(
        today_iso=_TODAY_ISO, broker=broker,
        calendar_fetcher=_calendar_empty,
    )

    # Both positions present
    by_ticker = {p.ticker: p for p in report.positions}
    assert "NEWIPO" in by_ticker
    assert by_ticker["NEWIPO"].sector_etf == "OTHER"
    assert by_ticker["AAPL"].sector_etf == "XLK"

    # Sector aggregation shows OTHER bucket explicitly
    assert report.exposure.by_sector.get("OTHER") == 0.5
    assert report.exposure.by_sector.get("XLK") == 0.5


# ---------------------------------------------------------------------------
# 7. Short position concentration math
# ---------------------------------------------------------------------------

def test_pipeline_short_position_concentration():
    """Negative market_value (short) still aggregates into HHI by
    signed-square (i.e. (-0.30)² = 0.09, same as a 30% long). Top-N
    uses signed weights so a short shows as negative share."""
    broker = _FakeBroker(
        portfolio={
            "account": {"portfolio_value": 100_000.0, "cash": 0.0},
            "positions": [
                # Long 70k, short 30k (net invested = 40k, but absolute
                # exposure 100k — Alpaca returns mv as the signed dollar
                # value).
                {"symbol": "AAPL", "market_value": "70000"},
                {"symbol": "TSLA", "market_value": "-30000"},
            ],
        },
        pnl={"intraday_pl": 0.0, "intraday_pl_pct": 0.0},
        history={"equity": [100_000.0], "timestamp": [1_700_000_000]},
    )

    report = build_risk_report(
        today_iso=_TODAY_ISO, broker=broker,
        calendar_fetcher=_calendar_empty,
    )

    by_ticker = {p.ticker: p for p in report.positions}
    # Both positions present
    assert {"AAPL", "TSLA"} == set(by_ticker)
    # Long position has positive weight, short has negative
    assert by_ticker["AAPL"].weight > 0
    assert by_ticker["TSLA"].weight < 0
    # HHI uses signed weights^2 — captures concentration regardless of direction
    expected_hhi = (
        by_ticker["AAPL"].weight ** 2 + by_ticker["TSLA"].weight ** 2
    )
    assert abs(report.concentration.hhi - expected_hhi) < 1e-9


# ---------------------------------------------------------------------------
# 8. Warning list truncation (formatter contract)
# ---------------------------------------------------------------------------

def test_pipeline_warnings_collected_when_many_failures():
    """Multiple sub-section failures → all warnings collected.
    Truncation to 3 happens in the FORMATTER, not the pipeline — the
    full list stays in RiskReport for the dashboard trace."""
    broker = _FakeBroker(raise_on={"portfolio", "pnl", "history"})

    def boom_calendar(tickers):
        raise RuntimeError("yfinance 503")

    report = build_risk_report(
        today_iso=_TODAY_ISO, broker=broker,
        calendar_fetcher=boom_calendar,
    )

    # Pipeline preserves ALL warnings (no truncation here)
    # positions=1, pnl=1, drawdown=1 (history hit twice but distinct call sites)
    # earnings short-circuits when positions empty → no calendar call →
    # no calendar warning. So expect 3 from broker only.
    assert len(report.warnings) >= 3

    # Now verify the formatter truncates to 3 (Stage-5 contract).
    from v2.portfolio._bot_cards import format_risk_card
    rendered = format_risk_card(report)
    warning_lines = [
        ln for ln in rendered.split("\n") if "<i>  • " in ln
    ]
    assert len(warning_lines) <= 3, (
        f"formatter must cap warnings at 3, got {len(warning_lines)}"
    )


# ---------------------------------------------------------------------------
# 9. invested_value derivation
# ---------------------------------------------------------------------------

def test_pipeline_invested_value_derivation():
    """Stage-2.5 contract: invested_value = portfolio_value - cash, derived
    @property. The spec case $128.6K total / $25.4K cash → $103.2K invested."""
    broker = _FakeBroker(
        portfolio={
            "account": {"portfolio_value": 128_600.0, "cash": 25_400.0},
            "positions": [{"symbol": "NVDA", "market_value": "103200"}],
        },
        pnl={"intraday_pl": 0.0, "intraday_pl_pct": 0.0},
        history={"equity": [128_600.0], "timestamp": [1_700_000_000]},
    )

    report = build_risk_report(
        today_iso=_TODAY_ISO, broker=broker,
        calendar_fetcher=_calendar_empty,
    )

    assert report.portfolio_value == 128_600.0
    assert report.cash == 25_400.0
    assert report.invested_value == 103_200.0
    # cash_pct = cash / portfolio_value (TOTAL), Stage 2.5 fix
    assert abs(report.cash_pct - 0.19751166) < 1e-5


# ---------------------------------------------------------------------------
# 10. Zero portfolio_value (brand-new account)
# ---------------------------------------------------------------------------

def test_pipeline_zero_portfolio_value_no_div_by_zero():
    """portfolio_value=0 must not raise ZeroDivisionError when computing
    cash_pct or position weights. Brand-new accounts are real."""
    broker = _FakeBroker(
        portfolio={
            "account": {"portfolio_value": 0.0, "cash": 0.0},
            "positions": [],
        },
        pnl={"intraday_pl": 0.0, "intraday_pl_pct": 0.0},
        history={"equity": [], "timestamp": []},
    )

    # Must not raise
    report = build_risk_report(
        today_iso=_TODAY_ISO, broker=broker,
        calendar_fetcher=_calendar_empty,
    )

    assert report.portfolio_value == 0.0
    assert report.cash == 0.0
    assert report.cash_pct == 0.0   # NOT NaN, NOT exception
    assert report.invested_value == 0.0
    assert report.positions == []
    # No drawdown observable, history empty
    assert report.drawdown.max_drawdown_pct is None
