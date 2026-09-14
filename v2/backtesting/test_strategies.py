"""Momentum, insider-cluster and committee strategies + the price cache."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import pytest

from v2.backtesting.engine import BacktestEngine
from v2.backtesting.strategies import (
    BacktestData,
    CommitteeStrategy,
    InsiderClusterStrategy,
    MomentumStrategy,
    PriceCache,
    rebalance_dates,
)

TODAY = date(2026, 9, 4)  # a Friday


@dataclass
class Bar:
    time: str
    close: float


def _bars(path, start: date, n: int) -> list[Bar]:
    """Daily bars on weekdays; ``path(i)`` gives the close of the i-th day."""
    out, d, i = [], start, 0
    while len(out) < n:
        if d.weekday() < 5:
            out.append(Bar(d.isoformat(), float(path(i))))
            i += 1
        d += timedelta(days=1)
    return out


class Prices:
    def __init__(self, series: dict[str, list[Bar]]):
        self.series = series
        self.calls: list[tuple[str, str, str]] = []

    def get_prices(self, ticker, start, end):
        self.calls.append((ticker, start, end))
        return [b for b in self.series.get(ticker, []) if start <= b.time <= end]


def _data(series: dict[str, list[Bar]], *, chunk=None, fd=None) -> BacktestData:
    return BacktestData(prices=PriceCache(Prices(series), chunk_days=chunk, today=TODAY), fd=fd, today=TODAY)


# ------------------------------------------------------------------ price cache

def test_price_cache_fetches_once_and_serves_slices_and_chunks():
    start = TODAY - timedelta(days=800)
    src = Prices({"AAA": _bars(lambda i: 100 + i, start, 560)})
    cache = PriceCache(src, today=TODAY)
    a = cache.get_prices("AAA", (TODAY - timedelta(days=30)).isoformat(), TODAY.isoformat())
    b = cache.get_prices("AAA", (TODAY - timedelta(days=10)).isoformat(), TODAY.isoformat())
    assert len(src.calls) == 1 and cache.requests == 1 and len(b) < len(a) and all(x.time >= b[0].time for x in b)
    cache.get_prices("AAA", start.isoformat(), TODAY.isoformat())  # earlier start → one more (older) fetch
    assert len(src.calls) == 2 and src.calls[-1][2] < src.calls[0][1]
    assert cache.closes("AAA", start)[0][1] == 100.0

    chunked = PriceCache(Prices(src.series), chunk_days=90, today=TODAY)
    rows = chunked.get_prices("AAA", start.isoformat(), TODAY.isoformat())
    assert chunked.requests == 9 and len(rows) == 560  # 800 days / 90 → 9 windows, nothing lost or duplicated

    class Boom:
        def get_prices(self, *a):
            raise RuntimeError("404 EMPTY_DATA")

    dead = PriceCache(Boom(), today=TODAY)
    assert dead.get_prices("ZZZ", start.isoformat(), TODAY.isoformat()) == [] and "ZZZ" in dead.failed and dead.requests == 0


# --------------------------------------------------------------------- momentum

def test_momentum_ranks_top_n_and_never_looks_ahead():
    start = TODAY - timedelta(days=1200)
    n = 800
    series = {
        "UP": _bars(lambda i: 100 * (1.002 ** i), start, n),        # strong steady uptrend
        "FLAT": _bars(lambda i: 100, start, n),
        "DOWN": _bars(lambda i: 100 * (0.999 ** i), start, n),
        "NEW": _bars(lambda i: 50 + i, TODAY - timedelta(days=100), 60),  # too short for the lookback
    }
    data = _data(series)
    ticks = []
    strat = MomentumStrategy(lookback_days=252, skip_days=21, holding_days=21, top_n=2, history_days=365, progress=ticks.append)
    signals = strat.generate_signals(list(series), data)
    assert ticks == [0, 1, 2, 3]
    assert signals and {s.ticker for s in signals} == {"UP"}          # FLAT / DOWN have no positive momentum
    first = signals[0]
    assert first.metadata["momentum"] > 0 and first.entry_date == (date.fromisoformat(first.metadata["signal_date"]) + timedelta(days=1)).isoformat()
    assert all(s.holding_days == 21 and s.direction == "long" for s in signals)
    assert first.metadata["signal_date"] >= (TODAY - timedelta(days=365)).isoformat()
    # rebalances are `holding_days` trading days apart
    dates = sorted({s.metadata["signal_date"] for s in signals})
    assert len(dates) >= 10

    # engine fills through the same cache: one provider call per ticker in total
    engine = BacktestEngine(capital=100_000, per_trade=10_000)
    result = engine.run_signals(signals, data)
    assert result.metrics and result.metrics.n_trades >= 8 and result.metrics.win_rate == 1.0  # the last periods cannot close before today
    assert data.prices.requests == 4

    # the 52-week-high filter keeps UP (at its high) and drops a name 30 % off its high
    series["FALLEN"] = _bars(lambda i: 200 if i < 700 else 140, start, n)
    data2 = _data(series)
    picks = MomentumStrategy(lookback_days=252, skip_days=21, holding_days=21, top_n=5, history_days=200, near_high_pct=0.05).generate_signals(list(series), data2)
    assert picks and all(s.ticker == "UP" and s.metadata["pct_from_52w_high"] >= -0.05 for s in picks)


def test_momentum_ranks_only_the_constituents_of_each_date():
    """A name that joined the index later must not be bought before it joined."""
    start = TODAY - timedelta(days=1200)
    series = {"OLD": _bars(lambda i: 100 * (1.001 ** i), start, 800), "NEW": _bars(lambda i: 100 * (1.003 ** i), start, 800), "GONE": _bars(lambda i: 100, start, 5)}
    joined = (TODAY - timedelta(days=120)).isoformat()
    members = lambda d: ["OLD", "GONE"] + (["NEW"] if d >= joined else [])  # noqa: E731
    strat = MomentumStrategy(lookback_days=252, skip_days=21, holding_days=21, top_n=1, history_days=365, universe_at=members)
    signals = strat.generate_signals(list(series), _data(series))
    before = [s for s in signals if s.metadata["signal_date"] < joined]
    after = [s for s in signals if s.metadata["signal_date"] >= joined]
    assert before and all(s.ticker == "OLD" for s in before)   # NEW had the higher momentum but was not a member yet
    assert after and all(s.ticker == "NEW" for s in after)
    assert strat.no_data == ["GONE"]                            # a former member without a usable series is reported
    first = strat.periods[0]
    assert first["pool"] == 2 and first["priced"] == 1 and first["ranked"] == 1 and first["picked"] == ["OLD"]
    assert strat.periods[-1]["pool"] == 3 and strat.periods[-1]["picked"] == ["NEW"]


def test_momentum_rejects_bad_params():
    with pytest.raises(ValueError):
        MomentumStrategy(lookback_days=20, skip_days=21)


# ---------------------------------------------------------------- insider cluster

@dataclass
class Insider:
    name: str
    filing_date: str
    transaction_shares: float
    transaction_price_per_share: float | None = 50.0
    transaction_value: float | None = None
    transaction_type: str | None = None


def test_insider_cluster_needs_distinct_buyers_inside_the_window():
    strat = InsiderClusterStrategy(window_days=30, min_insiders=2, min_value_usd=100_000, holding_days=63)
    rows = [
        Insider("Ann", "2026-01-05", 1000),                       # $50k alone
        Insider("Ann", "2026-01-20", 1000),                       # same person again → still one insider
        Insider("Bob", "2026-01-25", 2000),                       # 2 insiders, $200k within 30 days → signal
        Insider("Cid", "2026-01-28", 1500),                       # inside the cool-down → no second signal
        Insider("Dan", "2026-06-01", -5000),                      # a sale never counts
        Insider("Eve", "2026-06-02", 3000, transaction_type="purchase"),
        Insider("Fay", "2026-06-20", 3000, transaction_value=150_000),  # 2 buyers, $300k → second signal
    ]
    signals = strat.cluster_signals("XYZ", rows)
    assert [s.metadata["signal_date"] for s in signals] == ["2026-01-25", "2026-06-20"]
    assert signals[0].entry_date == "2026-01-26" and signals[0].metadata["insiders"] == 2 and signals[0].metadata["cluster_value_usd"] == 200_000
    assert signals[1].metadata["cluster_value_usd"] == 300_000

    # value floor blocks small clusters; a lone big buyer is not a cluster
    assert InsiderClusterStrategy(min_value_usd=1_000_000).cluster_signals("XYZ", rows) == []
    assert InsiderClusterStrategy(min_insiders=1).cluster_signals("XYZ", [Insider("Ann", "2026-01-05", 5000)])[0].metadata["insiders"] == 1


def test_insider_strategy_counts_requests_and_skips_failures():
    class FD:
        def get_insider_trades(self, ticker, end_date, *, start_date=None, limit=1000):
            if ticker == "BAD":
                raise RuntimeError("404")
            assert start_date < end_date
            return [Insider("Ann", "2026-03-01", 4000), Insider("Bob", "2026-03-10", 4000)]

    data = _data({}, fd=FD())
    ticks = []
    signals = InsiderClusterStrategy(progress=ticks.append).generate_signals(["OK", "BAD"], data)
    assert ticks == [0, 1] and [s.ticker for s in signals] == ["OK"] and data.fd_requests == {"insider_trades": 2}
    with pytest.raises(ValueError, match="Financial Datasets"):
        InsiderClusterStrategy().generate_signals(["OK"], _data({}))


# -------------------------------------------------------------------- committee

def test_rebalance_dates_leave_room_for_the_last_holding_period():
    dates = rebalance_dates(today=TODAY, history_days=365, step_trading_days=63)
    assert dates == sorted(dates) and len(dates) == 4
    assert date.fromisoformat(dates[-1]) <= TODAY - timedelta(days=91)
    assert date.fromisoformat(dates[0]) >= TODAY - timedelta(days=365)


def test_committee_strategy_buys_top_consensus_with_lagged_fundamentals(monkeypatch):
    from types import SimpleNamespace

    calls = []

    class Snap:
        def __init__(self, t):
            self.ticker, self.requests, self.as_of = t, {"financial_metrics": 2, "line_items": 2}, ""

    def fake_run_committee(tickers, client, *, personas=None, as_of=None, snapshots=None, max_workers=4, exclude_needs=(), progress=None):
        calls.append((tuple(tickers), as_of, tuple(sorted(snapshots or {})), tuple(exclude_needs)))
        verdict = lambda t, c, agree: SimpleNamespace(ticker=t, consensus=c, agreement=agree, bullish=5, bearish=1, neutral=2, abstained=5, voters=8, data_gaps=[], signals=[])  # noqa: E731
        verdicts = [verdict("AAA", 0.6, 0.8), verdict("BBB", 0.3, 0.4), verdict("CCC", -0.5, 0.9)]
        snaps = {t: Snap(t) for t in tickers if t not in (snapshots or {})}
        return SimpleNamespace(verdicts=verdicts, snapshots={**(snapshots or {}), **snaps}, errors={"CCC": "boom"})

    import v2.personas.committee as committee_mod
    monkeypatch.setattr(committee_mod, "run_committee", fake_run_committee)

    class Store:
        def __init__(self):
            self.saved = []

        def cached_snapshot(self, ticker, as_of, *, max_age_hours=24.0):
            return Snap(ticker) if ticker == "BBB" else None      # BBB already cached for every date

        def save_snapshot(self, snap):
            self.saved.append(snap.ticker)

    store = Store()
    data = _data({}, fd=object())
    ticks = []
    strat = CommitteeStrategy(holding_days=63, history_days=365, top_n=2, min_consensus=0.2, min_agreement=0.5, store=store, progress=ticks.append)
    signals = strat.generate_signals(["AAA", "BBB", "CCC"], data)

    assert len(strat.dates) == 4 and ticks == [3, 6, 9, 12]
    assert calls and all(c[2] == ("BBB",) and c[3] == ("news", "insiders") for c in calls)
    # fundamentals are asked for 45 days before the signal date; the entry is the day after the signal
    signal_date, as_of = signals[0].metadata["signal_date"], signals[0].metadata["as_of"]
    assert date.fromisoformat(signal_date) - date.fromisoformat(as_of) == timedelta(days=45)
    assert signals[0].entry_date == (date.fromisoformat(signal_date) + timedelta(days=1)).isoformat()
    # BBB fails the agreement floor, CCC is bearish → only AAA each period
    assert {s.ticker for s in signals} == {"AAA"} and len(signals) == 4
    # paid requests counted only for freshly fetched snapshots (AAA, CCC × 4 dates); cached BBB is free and not re-saved
    assert data.fd_requests == {"financial_metrics": 16, "line_items": 16} and sorted(set(store.saved)) == ["AAA", "CCC"]
    assert any(k.startswith("CCC@") for k in strat.errors)
    # the per-period log explains every date: who voted, who was picked
    assert len(strat.periods) == 4 and strat.periods[0]["picked"] == ["AAA"]
    row = {v["ticker"]: v for v in strat.periods[0]["verdicts"]}
    assert row["AAA"]["picked"] and not row["BBB"]["picked"] and row["CCC"]["consensus"] == -0.5 and row["AAA"]["voters"] == 8


def test_committee_strategy_stops_when_a_date_fails_wholesale(monkeypatch):
    """Credits ran out mid-run once: every later date failed and was still paid for in time. Now it stops."""
    from types import SimpleNamespace

    seen = []

    def fake_run_committee(tickers, client, *, personas=None, as_of=None, snapshots=None, max_workers=4, exclude_needs=(), progress=None):
        seen.append(as_of)
        broken = len(seen) >= 2  # first date fine, second date the provider rejects everything
        gaps = ["metrics_ttm: RuntimeError: HTTP 402 for /financial-metrics/: payment required", "line_items_ttm: RuntimeError: HTTP 402"] if broken else []
        sig = SimpleNamespace(abstained=broken, reasoning="abstain · missing inputs: ttm metrics (metrics_ttm: RuntimeError: HTTP 402 for /financial-metrics/: payment required)")
        mk = lambda t: SimpleNamespace(ticker=t, consensus=0.0 if broken else 0.5, agreement=0.0 if broken else 0.8, bullish=0 if broken else 6, bearish=0, neutral=0 if broken else 2,  # noqa: E731
                                       abstained=13 if broken else 5, voters=0 if broken else 8, data_gaps=gaps, signals=[sig, sig])
        snaps = {t: SimpleNamespace(ticker=t, requests={} if broken else {"financial_metrics": 2}, gaps=gaps) for t in tickers}
        return SimpleNamespace(verdicts=[mk(t) for t in tickers], snapshots=snaps, errors={})

    import v2.personas.committee as committee_mod
    monkeypatch.setattr(committee_mod, "run_committee", fake_run_committee)
    strat = CommitteeStrategy(holding_days=63, history_days=365, top_n=2)
    signals = strat.generate_signals(["AAA", "BBB", "CCC"], _data({}, fd=object()))
    assert len(strat.dates) == 4 and len(seen) == 2                      # stopped after the failing date
    assert strat.aborted and strat.aborted["failed"] == 3 and "HTTP 402" in strat.aborted["reason"] and len(strat.aborted["remaining_dates"]) == 2
    assert len(signals) == 2 and len(strat.periods) == 2                  # the good date's picks are kept
    assert strat.periods[1]["verdicts"][0]["reasons"] == ["ttm metrics (metrics_ttm: RuntimeError: HTTP 402 for /financial-metrics/: payment required) × 2"]
    assert strat.periods[0]["verdicts"][0]["reasons"] == []


def test_committee_strategy_full_mode_and_missing_client():
    with pytest.raises(ValueError, match="Financial Datasets"):
        CommitteeStrategy().generate_signals(["AAA"], _data({}))


def test_price_cache_keeps_only_date_and_close_and_slices_by_bisect():
    """A 10-year S&P 500 run holds ~2M bars; storing full provider rows would swap a small VPS."""
    from datetime import date, timedelta

    from v2.backtesting.strategies import Bar, PriceCache

    class Row:
        def __init__(self, t, c):
            self.time, self.close, self.open, self.high, self.low, self.volume = t, c, c, c, c, 1_000_000

    class Src:
        def get_prices(self, ticker, start, end):
            d, out, i = date.fromisoformat(start), [], 0
            while d <= date.fromisoformat(end):
                if d.weekday() < 5:
                    out.append(Row(d.isoformat() + "T00:00:00Z", 100 + i)); i += 1
                d += timedelta(days=1)
            return out

    cache = PriceCache(Src(), today=date(2026, 3, 31))
    rows = cache.get_prices("AAA", "2026-03-02", "2026-03-06")
    assert [type(r) for r in rows] == [Bar] * 5 and rows[0].time == "2026-03-02" and rows[-1].time == "2026-03-06"
    assert not hasattr(rows[0], "volume")                                      # only what consumers read
    assert cache.get_prices("AAA", "2026-03-07", "2026-03-08") == []            # weekend → empty slice
    assert [d for d, _ in cache.closes("AAA", "2026-03-27")] == ["2026-03-27", "2026-03-30", "2026-03-31"]
    assert cache.requests == 1                                                 # one fetch covered every call above
    cache.get_prices("AAA", "2026-01-05", "2026-01-06")                        # earlier start → one more fetch, merged in order
    assert cache.requests == 2 and cache._dates["AAA"] == sorted(cache._dates["AAA"]) and len(set(cache._dates["AAA"])) == len(cache._dates["AAA"])
