"""Stage-1 smoke test for v2/portfolio/.

Offline by design — network and Alpaca creds are not available in the
dev sandbox, so we exercise each module via:

- pure-function probes for concentration / exposure (no IO)
- a synthetic broker shim for positions / pnl / drawdown
- a stubbed calendar fetcher for earnings_risk

Run: ``poetry run python -m v2.portfolio._smoke_stage1``
"""

from __future__ import annotations

import sys
import traceback
from datetime import date, timedelta
from types import SimpleNamespace


def _section(title: str) -> None:
    print()
    print("=" * 64)
    print(title)
    print("=" * 64)


# ---------------------------------------------------------------------------
# Shared synthetic data
# ---------------------------------------------------------------------------

def _today_iso() -> str:
    return date.today().isoformat()


def _future_iso(days: int) -> str:
    return (date.today() + timedelta(days=days)).isoformat()


def _flats(*items):
    """Tuples of (ticker, weight) → list[PositionFlat]."""
    from v2.portfolio.models import PositionFlat
    from v2.universe import sector_bucket_for
    out = []
    for ticker, weight in items:
        out.append(PositionFlat(
            ticker=ticker,
            market_value=weight * 100_000,
            weight=weight,
            sector_etf=sector_bucket_for(ticker),
        ))
    return out


# ---------------------------------------------------------------------------
# 1. concentration
# ---------------------------------------------------------------------------

def smoke_concentration():
    from v2.portfolio.concentration import compute_concentration

    cases = [
        ("empty", [], (0.0, 0.0, 0.0, 0.0, 0)),
        ("single ticker 100%",
         _flats(("NVDA", 1.0)),
         (1.0, 1.0, 1.0, 1.0, 1)),
        ("balanced 5",
         _flats(("NVDA", 0.20), ("AAPL", 0.20), ("MSFT", 0.20),
                ("AMD", 0.20), ("META", 0.20)),
         (0.20, 0.60, 1.00, 0.20, 5)),  # HHI = 5 * 0.04 = 0.20
        ("concentrated",
         _flats(("NVDA", 0.40), ("AAPL", 0.30), ("MSFT", 0.20),
                ("AMD", 0.05), ("META", 0.05)),
         (0.40, 0.90, 1.00, 0.295, 5)),  # HHI = 0.16+0.09+0.04+0.0025+0.0025
    ]

    for label, positions, expected in cases:
        c = compute_concentration(positions)
        got = (round(c.top_1_pct, 4), round(c.top_3_pct, 4),
               round(c.top_5_pct, 4), round(c.hhi, 4), c.n_positions)
        expected_r = tuple(round(v, 4) if isinstance(v, float) else v for v in expected)
        ok = got == expected_r
        print(f"  {'ok' if ok else 'FAIL':4s} {label:25s} → top1/3/5/hhi/n = {got}")
        assert ok, f"{label}: got {got}, want {expected_r}"


# ---------------------------------------------------------------------------
# 2. exposure
# ---------------------------------------------------------------------------

def smoke_exposure():
    from v2.portfolio.exposure import compute_exposure

    # NVDA → SMH, AAPL/MSFT → XLK, JPM → XLF, XYZ123 → OTHER
    flats = _flats(
        ("NVDA", 0.30), ("AMD", 0.10),
        ("AAPL", 0.20), ("MSFT", 0.10),
        ("JPM", 0.20),
        ("XYZ123", 0.10),
    )
    e = compute_exposure(flats)
    print(f"  ok   by_sector keys: {sorted(e.by_sector.keys())}")
    print(f"  ok   largest: {e.largest_sector} ({e.largest_sector_pct:.0%})")

    # Float arithmetic — use tolerance.
    assert abs(e.by_sector["SMH"] - 0.40) < 1e-9
    assert abs(e.by_sector["XLK"] - 0.30) < 1e-9
    assert abs(e.by_sector["XLF"] - 0.20) < 1e-9
    assert abs(e.by_sector["OTHER"] - 0.10) < 1e-9
    assert e.largest_sector == "SMH"
    assert abs(e.largest_sector_pct - 0.40) < 1e-9


def smoke_exposure_empty():
    from v2.portfolio.exposure import compute_exposure
    e = compute_exposure([])
    assert e.by_sector == {}
    assert e.largest_sector == ""
    assert e.largest_sector_pct == 0.0
    print("  ok   empty positions → empty ExposureMetrics")


# ---------------------------------------------------------------------------
# 3. positions
# ---------------------------------------------------------------------------

def smoke_positions():
    from v2.portfolio.positions import get_flat_positions

    class FakeBroker:
        def get_portfolio(self):
            return {
                "account": {"portfolio_value": 100_000.0, "cash": 25_000.0},
                "positions": [
                    {"symbol": "NVDA", "market_value": "30000"},
                    {"symbol": "AAPL", "market_value": "20000"},
                    {"symbol": "msft", "market_value": "15000"},  # lowercase
                    {"symbol": "BAD",  "market_value": "not_a_number"},  # malformed
                ],
            }

    flats, pv, cash, warnings = get_flat_positions(broker=FakeBroker())
    print(f"  ok   portfolio_value={pv} cash={cash} warnings={warnings}")
    print(f"  ok   flat tickers: {[(p.ticker, round(p.weight, 3), p.sector_etf) for p in flats]}")

    assert pv == 100_000.0
    assert cash == 25_000.0
    assert warnings == []
    tickers = [p.ticker for p in flats]
    # Largest first; malformed BAD silently dropped
    assert tickers == ["NVDA", "AAPL", "MSFT"]
    assert "BAD" not in tickers
    # Stage 2.5: weight denominator is sum of invested market_values
    # (30k + 20k + 15k = 65k), NOT Alpaca's portfolio_value (which is
    # TOTAL = invested + cash). NVDA = 30/65 ≈ 0.4615.
    assert abs(flats[0].weight - (30000 / 65000)) < 1e-9
    assert abs(flats[1].weight - (20000 / 65000)) < 1e-9
    assert abs(flats[2].weight - (15000 / 65000)) < 1e-9
    # Weights sum to 1.0 (share-of-invested semantics)
    assert abs(sum(p.weight for p in flats) - 1.0) < 1e-9
    assert flats[0].sector_etf == "SMH"

    # Sector mapping uppercases the lowercased input
    assert flats[2].ticker == "MSFT"
    assert flats[2].sector_etf == "XLK"


def smoke_positions_alpaca_unavailable():
    from v2.portfolio.positions import get_flat_positions

    class BoomBroker:
        def get_portfolio(self):
            raise RuntimeError("Alpaca 503")

    flats, pv, cash, warnings = get_flat_positions(broker=BoomBroker())
    assert flats == []
    assert pv == 0.0 and cash == 0.0
    assert any("503" in w for w in warnings)
    print(f"  ok   Alpaca unavailable → soft return; warning='{warnings[0][:60]}...'")


# ---------------------------------------------------------------------------
# 4. PnL
# ---------------------------------------------------------------------------

def smoke_pnl_full_history():
    from v2.portfolio.pnl import compute_pnl

    # 22 days of equity going from 100k → 102.5k. Daily wiggle: last day +1%.
    equity = [100_000 + i * 100 for i in range(22)]   # 100000 → 102100
    equity[-1] = equity[-2] * 1.01                     # last day +1%
    timestamps = list(range(22))

    class FakeBroker:
        def get_pnl(self):
            return {"intraday_pl": 1021.0, "intraday_pl_pct": 0.01}
        def get_portfolio_history(self, period, timeframe):
            return {"equity": equity, "timestamp": timestamps,
                    "profit_loss": [], "profit_loss_pct": [],
                    "base_value": 100_000.0, "period": period, "timeframe": timeframe}

    metrics, warnings = compute_pnl(broker=FakeBroker())
    print(f"  ok   daily={metrics.daily_pnl_pct:.4f} "
          f"weekly={metrics.weekly_pnl_pct:.4f} monthly={metrics.monthly_pnl_pct:.4f}")
    assert metrics.daily_pnl_pct == 0.01
    assert metrics.weekly_pnl_pct is not None
    assert metrics.monthly_pnl_pct is not None
    assert warnings == []


def smoke_pnl_short_history():
    from v2.portfolio.pnl import compute_pnl

    class FakeBroker:
        def get_pnl(self):
            return {"intraday_pl": 50.0, "intraday_pl_pct": 0.005}
        def get_portfolio_history(self, period, timeframe):
            # Only 3 days — weekly and monthly should be None
            return {"equity": [100_000, 100_500, 101_000], "timestamp": [1, 2, 3],
                    "profit_loss": [], "profit_loss_pct": [],
                    "base_value": 100_000.0}

    metrics, warnings = compute_pnl(broker=FakeBroker())
    assert metrics.daily_pnl_pct == 0.005
    assert metrics.weekly_pnl_pct is None
    assert metrics.monthly_pnl_pct is None
    print("  ok   3-day history → daily ok, weekly/monthly = None")


def smoke_pnl_history_fails_daily_survives():
    from v2.portfolio.pnl import compute_pnl

    class HalfBroker:
        def get_pnl(self):
            return {"intraday_pl": -200.0, "intraday_pl_pct": -0.02}
        def get_portfolio_history(self, period, timeframe):
            raise RuntimeError("history endpoint 500")

    metrics, warnings = compute_pnl(broker=HalfBroker())
    assert metrics.daily_pnl_pct == -0.02
    assert metrics.weekly_pnl_pct is None
    assert metrics.monthly_pnl_pct is None
    assert any("500" in w for w in warnings)
    print(f"  ok   history broken, daily preserved; warning='{warnings[0][:60]}...'")


# ---------------------------------------------------------------------------
# 5. Drawdown
# ---------------------------------------------------------------------------

def smoke_drawdown_normal():
    from v2.portfolio.drawdown import compute_drawdown

    # Peak at idx 5 (110k), trough at idx 10 (95k), partial recovery to 100k
    equity = [100, 102, 105, 108, 110, 112, 108, 105, 100, 97, 95, 100]
    equity = [e * 1000 for e in equity]
    timestamps = [1700_000_000 + i * 86400 for i in range(len(equity))]

    class FakeBroker:
        def get_portfolio_history(self, period, timeframe):
            return {"equity": equity, "timestamp": timestamps}

    metrics, warnings = compute_drawdown(broker=FakeBroker())
    print(f"  ok   current_dd={metrics.current_drawdown_pct:.4f} "
          f"max_dd={metrics.max_drawdown_pct:.4f} peak={metrics.peak_value}")
    # Stage 5: drawdowns are non-negative magnitudes. (95-112)/112 = -0.152,
    # magnitude 0.152.
    assert abs(metrics.max_drawdown_pct - abs((95 - 112) / 112)) < 1e-6
    assert metrics.peak_value == 112_000
    # Not at the peak → some positive drawdown magnitude
    assert metrics.current_drawdown_pct > 0
    assert metrics.peak_date is not None


def smoke_drawdown_single_point():
    from v2.portfolio.drawdown import compute_drawdown

    class FakeBroker:
        def get_portfolio_history(self, period, timeframe):
            return {"equity": [100_000], "timestamp": [1700_000_000]}

    metrics, warnings = compute_drawdown(broker=FakeBroker())
    assert metrics.current_drawdown_pct == 0.0
    assert metrics.max_drawdown_pct == 0.0
    assert metrics.peak_value == 100_000
    print("  ok   single-point history → 0% drawdown, no exception")


def smoke_drawdown_empty():
    from v2.portfolio.drawdown import compute_drawdown

    class FakeBroker:
        def get_portfolio_history(self, period, timeframe):
            return {"equity": [], "timestamp": []}

    metrics, warnings = compute_drawdown(broker=FakeBroker())
    assert metrics.current_drawdown_pct is None
    assert metrics.max_drawdown_pct is None
    print("  ok   empty history → unavailable()")


def smoke_drawdown_alpaca_down():
    from v2.portfolio.drawdown import compute_drawdown

    class BoomBroker:
        def get_portfolio_history(self, period, timeframe):
            raise RuntimeError("Alpaca 502")

    metrics, warnings = compute_drawdown(broker=BoomBroker())
    assert metrics.current_drawdown_pct is None
    assert any("502" in w for w in warnings)
    print(f"  ok   Alpaca down → unavailable(); warning='{warnings[0][:60]}...'")


# ---------------------------------------------------------------------------
# 6. Earnings risk
# ---------------------------------------------------------------------------

def smoke_earnings_risk():
    from v2.portfolio.earnings_risk import compute_earnings_risk
    from v2.earnings.calendar import CalendarBatchResult
    from v2.earnings.models import EarningsEvent

    positions = _flats(("AAPL", 0.30), ("NVDA", 0.20), ("MSFT", 0.10))
    today = _today_iso()

    def fake_batch(tickers):
        events = {
            "AAPL": EarningsEvent("AAPL", _future_iso(3), "amc", 1.5, 92e9),
            "NVDA": EarningsEvent("NVDA", _future_iso(8), "amc"),   # > 7d, drop
            # MSFT not in result — no upcoming
        }
        return CalendarBatchResult(events=events, skipped_unsupported=[],
                                   skipped_empty=["MSFT"], errors=[])

    items, warnings = compute_earnings_risk(positions, today, calendar_fetcher=fake_batch)
    print(f"  ok   items: {[(i.ticker, i.days_until) for i in items]}")
    assert [(i.ticker, i.days_until) for i in items] == [("AAPL", 3)]
    assert warnings == []


def smoke_earnings_risk_calendar_down():
    from v2.portfolio.earnings_risk import compute_earnings_risk

    positions = _flats(("AAPL", 1.0))
    def fake_batch(tickers):
        raise RuntimeError("yfinance 503")

    items, warnings = compute_earnings_risk(positions, _today_iso(),
                                            calendar_fetcher=fake_batch)
    assert items == []
    assert any("503" in w for w in warnings)
    print(f"  ok   calendar down → empty list; warning='{warnings[0][:60]}...'")


def smoke_earnings_risk_no_positions():
    from v2.portfolio.earnings_risk import compute_earnings_risk
    items, warnings = compute_earnings_risk([], _today_iso(),
                                            calendar_fetcher=lambda t: None)
    assert items == []
    assert warnings == []
    print("  ok   empty positions → empty list, no calendar call")


# ---------------------------------------------------------------------------
# 7. Pipeline orchestration
# ---------------------------------------------------------------------------

def smoke_pipeline_full():
    from v2.portfolio.pipeline import build_risk_report
    from v2.earnings.calendar import CalendarBatchResult
    from v2.earnings.models import EarningsEvent

    class FakeBroker:
        def get_portfolio(self):
            return {
                "account": {"portfolio_value": 100_000.0, "cash": 25_000.0},
                "positions": [
                    {"symbol": "NVDA", "market_value": "30000"},
                    {"symbol": "AAPL", "market_value": "20000"},
                    {"symbol": "JPM",  "market_value": "15000"},
                ],
            }
        def get_pnl(self):
            return {"intraday_pl": -1850.0, "intraday_pl_pct": -0.018}
        def get_portfolio_history(self, period, timeframe):
            # 22 days, peak at idx 10 (105k), gentle decline to 100k current
            equity = ([100_000 + i * 500 for i in range(11)] +
                      [105_000 - i * 1000 for i in range(11)])
            return {"equity": equity, "timestamp": list(range(22)),
                    "profit_loss": [], "profit_loss_pct": [],
                    "base_value": 100_000.0, "period": period, "timeframe": timeframe}

    def fake_calendar(tickers):
        return CalendarBatchResult(
            events={"AAPL": EarningsEvent("AAPL", _future_iso(2), "amc", 1.5, 92e9)},
            skipped_unsupported=[], skipped_empty=[], errors=[],
        )

    report = build_risk_report(
        today_iso=_today_iso(),
        broker=FakeBroker(),
        calendar_fetcher=fake_calendar,
    )

    print(f"  ok   pv={report.portfolio_value} cash_pct={report.cash_pct:.1%}")
    print(f"  ok   top_1={report.concentration.top_1_pct:.1%} "
          f"top_3={report.concentration.top_3_pct:.1%} HHI={report.concentration.hhi:.3f}")
    print(f"  ok   sectors: {report.exposure.by_sector}")
    print(f"  ok   pnl daily={report.pnl.daily_pnl_pct:.3f} "
          f"weekly={report.pnl.weekly_pnl_pct} monthly={report.pnl.monthly_pnl_pct}")
    print(f"  ok   drawdown current={report.drawdown.current_drawdown_pct:.4f} "
          f"max={report.drawdown.max_drawdown_pct:.4f}")
    print(f"  ok   earnings_risk: {[(i.ticker, i.days_until) for i in report.earnings_risk_next_7d]}")
    print(f"  ok   warnings count: {len(report.warnings)}")

    # Structure assertions
    assert report.snapshot_date == _today_iso()
    assert report.portfolio_value == 100_000.0
    # Stage 2.5: cash_pct = cash / portfolio_value (TOTAL), not /(pv+cash).
    # 25k / 100k = 0.25.
    assert report.cash_pct == 0.25
    # invested_value derived from portfolio_value - cash.
    assert report.invested_value == 75_000.0
    # Stage 2.5: weights are share-of-invested. Positions sum to 65k,
    # NVDA = 30/65 ≈ 0.4615. (Old buggy behavior would have been 0.30.)
    assert abs(report.concentration.top_1_pct - (30000 / 65000)) < 1e-9
    assert report.exposure.largest_sector == "SMH"
    assert report.pnl.daily_pnl_pct == -0.018
    # Stage 5: drawdown magnitude is non-negative
    assert report.drawdown.max_drawdown_pct > 0
    assert len(report.earnings_risk_next_7d) == 1
    assert report.warnings == []


def smoke_pipeline_alpaca_down_partial():
    """Alpaca completely down → empty positions + warnings, but the
    pipeline still returns a RiskReport with structure intact."""
    from v2.portfolio.pipeline import build_risk_report

    class BoomBroker:
        def get_portfolio(self): raise RuntimeError("Alpaca 503")
        def get_pnl(self):       raise RuntimeError("Alpaca 503")
        def get_portfolio_history(self, *a, **kw): raise RuntimeError("Alpaca 503")

    def fake_calendar(tickers):
        from v2.earnings.calendar import CalendarBatchResult
        return CalendarBatchResult(events={}, skipped_unsupported=[],
                                   skipped_empty=[], errors=[])

    report = build_risk_report(
        today_iso=_today_iso(),
        broker=BoomBroker(),
        calendar_fetcher=fake_calendar,
    )
    print(f"  ok   alpaca-down report: warnings={len(report.warnings)} "
          f"positions={len(report.positions)}")
    assert report.positions == []
    assert report.pnl.daily_pnl_pct is None
    assert report.drawdown.current_drawdown_pct is None
    assert len(report.warnings) >= 3  # positions + pnl + drawdown
    # earnings_risk gets short-circuited by empty positions, no warning


# ---------------------------------------------------------------------------
# Stage 2.5 — portfolio_value semantics (TOTAL = invested + cash)
# ---------------------------------------------------------------------------

def _calendar_empty(tickers):
    from v2.earnings.calendar import CalendarBatchResult
    return CalendarBatchResult(events={}, skipped_unsupported=[],
                               skipped_empty=[], errors=[])


def smoke_pipeline_cash_zero():
    """All-invested account (cash=0): cash_pct = 0, invested_value
    equals portfolio_value, and concentration weights still sum to 1.0."""
    from v2.portfolio.pipeline import build_risk_report

    class FakeBroker:
        def get_portfolio(self):
            return {
                "account": {"portfolio_value": 100_000.0, "cash": 0.0},
                "positions": [
                    {"symbol": "NVDA", "market_value": "60000"},
                    {"symbol": "AAPL", "market_value": "40000"},
                ],
            }
        def get_pnl(self):
            return {"intraday_pl": 0.0, "intraday_pl_pct": 0.0}
        def get_portfolio_history(self, period, timeframe):
            return {"equity": [100_000, 100_000], "timestamp": [1, 2]}

    r = build_risk_report(today_iso=_today_iso(), broker=FakeBroker(),
                          calendar_fetcher=_calendar_empty)
    print(f"  ok   pv={r.portfolio_value} cash={r.cash} "
          f"cash_pct={r.cash_pct:.4f} invested={r.invested_value}")
    assert r.portfolio_value == 100_000.0
    assert r.cash == 0.0
    assert r.cash_pct == 0.0
    assert r.invested_value == 100_000.0
    # Weights sum to 1.0 — denominator is invested_total (sum of mv)
    total_weight = sum(p.weight for p in r.positions)
    assert abs(total_weight - 1.0) < 1e-9
    # Top 1 NVDA is 60/100 = 60% of invested
    assert abs(r.concentration.top_1_pct - 0.60) < 1e-9


def smoke_pipeline_all_cash_no_positions():
    """All-cash account: cash_pct = 1.0, invested_value = 0,
    concentration/exposure are empty but don't crash."""
    from v2.portfolio.pipeline import build_risk_report

    class FakeBroker:
        def get_portfolio(self):
            return {
                "account": {"portfolio_value": 50_000.0, "cash": 50_000.0},
                "positions": [],
            }
        def get_pnl(self):
            return {"intraday_pl": 0.0, "intraday_pl_pct": 0.0}
        def get_portfolio_history(self, period, timeframe):
            return {"equity": [50_000], "timestamp": [1]}

    r = build_risk_report(today_iso=_today_iso(), broker=FakeBroker(),
                          calendar_fetcher=_calendar_empty)
    print(f"  ok   all-cash: pv={r.portfolio_value} cash_pct={r.cash_pct:.4f} "
          f"invested={r.invested_value} n_positions={len(r.positions)}")
    assert r.portfolio_value == 50_000.0
    assert r.cash == 50_000.0
    assert r.cash_pct == 1.0
    assert r.invested_value == 0.0
    assert r.positions == []
    # Concentration / exposure empty but well-formed
    assert r.concentration.n_positions == 0
    assert r.concentration.top_1_pct == 0.0
    assert r.concentration.hhi == 0.0
    assert r.exposure.by_sector == {}
    assert r.exposure.largest_sector == ""
    # No earnings call attempted (positions empty short-circuits)
    assert r.earnings_risk_next_7d == []


def smoke_pipeline_normal_split():
    """Stage 2 dry-run case: $128.6K total / $25.4K cash → 19.7% cash,
    $103.2K invested. The exact numbers from the spec."""
    from v2.portfolio.pipeline import build_risk_report

    class FakeBroker:
        def get_portfolio(self):
            return {
                "account": {"portfolio_value": 128_600.0, "cash": 25_400.0},
                "positions": [
                    {"symbol": "NVDA", "market_value": "36120"},  # 35% of invested
                    {"symbol": "AAPL", "market_value": "20640"},  # 20%
                    {"symbol": "JPM",  "market_value": "15480"},  # 15%
                    {"symbol": "CRM",  "market_value": "10320"},  # 10%
                    {"symbol": "BAC",  "market_value":  "5160"},  # 5%
                    {"symbol": "MSFT", "market_value": "15480"},  # 15%
                ],
            }
        def get_pnl(self):
            return {"intraday_pl": -1856.0, "intraday_pl_pct": -0.0178}
        def get_portfolio_history(self, period, timeframe):
            return {"equity": [120_000, 125_000, 128_600], "timestamp": [1, 2, 3]}

    r = build_risk_report(today_iso=_today_iso(), broker=FakeBroker(),
                          calendar_fetcher=_calendar_empty)
    print(f"  ok   normal split: pv={r.portfolio_value} cash={r.cash} "
          f"cash_pct={r.cash_pct:.3%} invested={r.invested_value}")

    assert r.portfolio_value == 128_600.0
    assert r.cash == 25_400.0
    # 25400 / 128600 = 0.19751166...  spec said "≈ 19.7%"
    assert abs(r.cash_pct - 0.197511) < 1e-5
    assert r.invested_value == 103_200.0

    # Weights are share of invested, NOT share of TOTAL.
    # 36120 / 103200 = 0.35 (intuitive: NVDA is 35% of invested book)
    assert abs(r.concentration.top_1_pct - 0.35) < 1e-9
    # vs the old wrong denominator (128600), top_1 would be 0.281

    # Sum of weights == 1.0 (since denominator = sum-of-parts)
    total_weight = sum(p.weight for p in r.positions)
    assert abs(total_weight - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    failed: list[str] = []
    for name, fn in [
        ("concentration",         smoke_concentration),
        ("exposure",              smoke_exposure),
        ("exposure_empty",        smoke_exposure_empty),
        ("positions",             smoke_positions),
        ("positions_alpaca_down", smoke_positions_alpaca_unavailable),
        ("pnl_full",              smoke_pnl_full_history),
        ("pnl_short_history",     smoke_pnl_short_history),
        ("pnl_history_partial",   smoke_pnl_history_fails_daily_survives),
        ("drawdown_normal",       smoke_drawdown_normal),
        ("drawdown_single_point", smoke_drawdown_single_point),
        ("drawdown_empty",        smoke_drawdown_empty),
        ("drawdown_alpaca_down",  smoke_drawdown_alpaca_down),
        ("earnings_risk",         smoke_earnings_risk),
        ("earnings_risk_down",    smoke_earnings_risk_calendar_down),
        ("earnings_risk_empty",   smoke_earnings_risk_no_positions),
        ("pipeline_full",         smoke_pipeline_full),
        ("pipeline_alpaca_down",  smoke_pipeline_alpaca_down_partial),
        ("pipeline_cash_zero",    smoke_pipeline_cash_zero),
        ("pipeline_all_cash",     smoke_pipeline_all_cash_no_positions),
        ("pipeline_normal_split", smoke_pipeline_normal_split),
    ]:
        _section(name)
        try:
            fn()
        except Exception:
            traceback.print_exc()
            failed.append(name)

    print()
    if failed:
        print(f"FAILED: {failed}")
        return 1
    print("ALL SMOKE TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
