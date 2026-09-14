"""Tests for the event study engine."""

from __future__ import annotations

import os

import numpy as np
import pytest

from v2.event_study.models import EventCAR, EventStudyResult, MarketModelFit
from v2.event_study.stats import (
    bootstrap_ci,
    compute_abnormal_returns,
    fit_market_model,
    sum_car,
    ttest_cars,
)


# ---------------------------------------------------------------------------
# Unit tests — stats.py
# ---------------------------------------------------------------------------


class TestFitMarketModel:
    def test_known_alpha_beta(self):
        rng = np.random.default_rng(42)
        n = 240
        market = rng.normal(0.0005, 0.01, n)
        noise = rng.normal(0, 0.005, n)
        stock = 0.001 + 1.2 * market + noise

        fit = fit_market_model(stock, market)
        assert abs(fit.alpha - 0.001) < 0.002
        assert abs(fit.beta - 1.2) < 0.15
        assert fit.r_squared > 0.3
        assert fit.n_obs == n

    def test_identical_returns(self):
        returns = np.array([0.01, -0.005, 0.003, 0.002, -0.001] * 50)
        fit = fit_market_model(returns, returns)
        assert abs(fit.beta - 1.0) < 0.01
        assert abs(fit.alpha) < 0.001

    def test_zero_variance_market(self):
        market = np.zeros(100)
        stock = np.random.default_rng(0).normal(0, 0.01, 100)
        fit = fit_market_model(stock, market)
        assert fit.n_obs == 100


class TestComputeAbnormalReturns:
    def test_exact(self):
        stock = np.array([0.03, -0.01, 0.02])
        market = np.array([0.02, 0.00, 0.01])
        alpha, beta = 0.001, 1.2
        ar = compute_abnormal_returns(stock, market, alpha, beta)
        expected = stock - (alpha + beta * market)
        np.testing.assert_allclose(ar, expected)


class TestSumCar:
    def test_window(self):
        daily_ar = np.array([0.01, 0.005, -0.002, 0.003, 0.001, 0.002])
        assert abs(sum_car(daily_ar, 0, 1) - 0.015) < 1e-10
        assert abs(sum_car(daily_ar, 0, 5) - 0.019) < 1e-10

    def test_single_day(self):
        daily_ar = np.array([0.05, -0.01])
        assert abs(sum_car(daily_ar, 0, 0) - 0.05) < 1e-10


class TestTtestCars:
    def test_positive_mean(self):
        cars = np.array([0.02, 0.03, 0.01, 0.04, 0.02, 0.01, 0.03, 0.02])
        t, p = ttest_cars(cars)
        assert t > 0
        assert p < 0.05

    def test_too_few(self):
        t, p = ttest_cars(np.array([0.01]))
        assert t == 0.0
        assert p == 1.0


class TestBootstrapCI:
    def test_brackets_mean(self):
        rng = np.random.default_rng(99)
        cars = rng.normal(0.02, 0.01, 50)
        ci = bootstrap_ci(cars, n_bootstrap=5000, rng_seed=42)
        assert ci.lower < cars.mean() < ci.upper
        assert ci.confidence == 0.95

    def test_deterministic_with_seed(self):
        cars = np.array([0.01, 0.02, 0.03, 0.04])
        ci1 = bootstrap_ci(cars, rng_seed=123)
        ci2 = bootstrap_ci(cars, rng_seed=123)
        assert ci1.lower == ci2.lower
        assert ci1.upper == ci2.upper


# ---------------------------------------------------------------------------
# Unit tests — engine helpers
# ---------------------------------------------------------------------------


class TestRetrospectiveFilter:
    def test_filters_stale_records(self):
        from v2.data.models import EarningsRecord
        from v2.event_study.engine import _filter_retrospective

        good = EarningsRecord(
            ticker="GS", report_period="2026-03-31", source_type="8-K",
            filing_date="2026-04-13",
        )
        stale = EarningsRecord(
            ticker="GS", report_period="2025-12-31", source_type="8-K",
            filing_date="2026-04-13",
        )
        result = _filter_retrospective([good, stale])
        assert len(result) == 1
        assert result[0].report_period == "2026-03-31"


class TestDedupeAndGrouping:
    def test_one_event_per_report_period_prefers_the_8k(self):
        from v2.data.models import EarningsRecord
        from v2.event_study.engine import _dedupe_records

        rec = lambda src, filed, period="2026-06-30": EarningsRecord(ticker="X", report_period=period, source_type=src, filing_date=filed)  # noqa: E731
        kept = _dedupe_records([rec("10-Q", "2026-08-01"), rec("8-K", "2026-07-30"), rec("10-K", "2026-08-05", "2026-03-31"), rec("8-K", "2026-04-28", "2026-03-31")])
        assert [(r.report_period, r.source_type, r.filing_date) for r in kept] == [("2026-06-30", "8-K", "2026-07-30"), ("2026-03-31", "8-K", "2026-04-28")]
        # without an 8-K the earliest filing wins
        only = _dedupe_records([rec("10-K", "2026-08-05"), rec("10-Q", "2026-08-01")])
        assert len(only) == 1 and only[0].source_type == "10-Q"

    def test_aggregate_by_surprise_puts_all_first_then_beat_and_miss(self):
        from v2.event_study.engine import _aggregate

        def ev(i, surprise):
            return EventCAR(ticker="T", event_date=f"2025-01-{10 + i:02d}", source_type="8-K", report_period="2024-12-31", eps_surprise=surprise,
                            market_model=MarketModelFit(alpha=0.0, beta=1.0, r_squared=0.5, n_obs=240), daily_ar=[0.0] * 21,
                            car_0_1=0.02 if surprise == "BEAT" else -0.02, car_0_5=0.03 if surprise == "BEAT" else -0.03, car_0_20=0.04 if surprise == "BEAT" else -0.04)

        events = [ev(i, "BEAT") for i in range(4)] + [ev(i + 4, "MISS") for i in range(3)] + [ev(8, None)]
        groups = _aggregate(events, 200, 42, group_by="surprise")
        assert [g.group for g in groups] == ["ALL", "BEAT", "MISS"]  # a single unlabeled event is too small for stats
        assert groups[0].n_events == 8 and groups[1].n_events == 4 and groups[2].n_events == 3
        assert groups[1].windows[0].mean_car > 0 > groups[2].windows[0].mean_car
        by_source = _aggregate(events, 200, 42, group_by="source")
        assert [g.group for g in by_source] == ["8-K"] and by_source[0].source_type == "8-K"

    def test_aggregate_by_reaction_splits_terciles_of_the_two_day_car(self):
        from v2.event_study.engine import _aggregate, _reaction_groups

        def ev(i, react):
            return EventCAR(ticker="T", event_date=f"2025-01-{10 + i:02d}", source_type="8-K", report_period="2024-12-31", eps_surprise=None,
                            market_model=MarketModelFit(alpha=0.0, beta=1.0, r_squared=0.5, n_obs=240), daily_ar=[0.0] * 21,
                            car_0_1=react, car_0_5=react, car_0_20=react * 1.5, car_2_20=react * 0.5)

        events = [ev(i, r) for i, r in enumerate([-0.10, -0.08, -0.06, -0.01, 0.0, 0.01, 0.05, 0.07, 0.09])]
        terciles = _reaction_groups(events)
        assert [len(terciles[k]) for k in ("REACT_DOWN", "REACT_MID", "REACT_UP")] == [3, 3, 3]
        assert all(e.car_0_1 <= -0.06 for e in terciles["REACT_DOWN"]) and all(e.car_0_1 >= 0.05 for e in terciles["REACT_UP"])
        groups = _aggregate(events, 200, 42, group_by="reaction")
        assert [g.group for g in groups] == ["ALL", "REACT_UP", "REACT_MID", "REACT_DOWN"]
        drift = {w.window: w.mean_car for w in groups[1].windows}
        assert "[+2,+20]" in drift and drift["[+2,+20]"] > 0 > {w.window: w.mean_car for w in groups[3].windows}["[+2,+20]"]
        assert _reaction_groups(events[:2]) == {}  # too few to cut into thirds


# ---------------------------------------------------------------------------
# Unit tests — plot (smoke test, no visual assertion)
# ---------------------------------------------------------------------------


class TestPlots:
    @pytest.fixture()
    def synthetic_result(self):
        events = []
        for i in range(10):
            events.append(EventCAR(
                ticker="TEST",
                event_date=f"2025-01-{10 + i:02d}",
                source_type="8-K" if i < 6 else "10-Q",
                report_period=f"2024-12-{10 + i:02d}",
                market_model=MarketModelFit(alpha=0.001, beta=1.1, r_squared=0.5, n_obs=240),
                daily_ar=[0.005 * ((-1) ** j) for j in range(21)],
                car_0_1=0.01 + i * 0.001,
                car_0_5=0.02 + i * 0.002,
                car_0_20=0.03 + i * 0.003,
            ))
        return EventStudyResult(events=events, aggregates=[], skipped_tickers=[])

    def test_plot_car_by_source(self, synthetic_result):
        from v2.event_study.plot import plot_car_by_source
        from v2.event_study.engine import _aggregate

        synthetic_result.aggregates = _aggregate(synthetic_result.events, 1000, 42)
        fig = plot_car_by_source(synthetic_result)
        assert fig is not None
        plt_mod = __import__("matplotlib.pyplot", fromlist=["close"])
        plt_mod.close(fig)

    def test_plot_car_distribution(self, synthetic_result):
        from v2.event_study.plot import plot_car_distribution

        fig = plot_car_distribution(synthetic_result, "[0,+1]")
        assert fig is not None
        plt_mod = __import__("matplotlib.pyplot", fromlist=["close"])
        plt_mod.close(fig)

    def test_plot_cumulative_ar(self, synthetic_result):
        from v2.event_study.plot import plot_cumulative_ar

        fig = plot_cumulative_ar(synthetic_result)
        assert fig is not None
        plt_mod = __import__("matplotlib.pyplot", fromlist=["close"])
        plt_mod.close(fig)


# ---------------------------------------------------------------------------
# Integration tests — require API key
# ---------------------------------------------------------------------------

pytestmark_live = pytest.mark.skipif(
    not os.environ.get("FINANCIAL_DATASETS_API_KEY"),
    reason="live tests require FINANCIAL_DATASETS_API_KEY",
)


@pytest.fixture(scope="module")
def fd():
    from v2.data import FDClient
    with FDClient() as client:
        yield client


@pytestmark_live
def test_compute_car_live(fd):
    from v2.event_study import compute_car

    result = compute_car(["AAPL"], fd, earnings_limit=4, rng_seed=42)
    assert len(result.events) > 0, "Expected at least one event for AAPL"
    for e in result.events:
        assert e.ticker == "AAPL"
        assert e.source_type in {"8-K", "10-Q", "10-K", "20-F"}
        if e.car_0_1 is not None:
            assert np.isfinite(e.car_0_1)


@pytestmark_live
def test_compute_car_multi_ticker(fd):
    from v2.event_study import compute_car

    result = compute_car(["AAPL", "MSFT", "NVDA"], fd, earnings_limit=4, rng_seed=42)
    tickers_seen = {e.ticker for e in result.events}
    assert len(tickers_seen) >= 2, f"Expected multiple tickers, got {tickers_seen}"
    source_types_seen = {e.source_type for e in result.events}
    assert len(source_types_seen) >= 1
