"""Three more strategies for the backtest engine, plus a price cache.

All three speak the same contract as :class:`v2.backtesting.strategy.Strategy`
— produce ``TradeSignal`` objects, let the engine fill them — but they take
their data through an explicit :class:`BacktestData` bundle instead of a
bare FD client, so the price feed (yfinance, free) can be separated from
the event / fundamentals feed (Financial Datasets, paid per request).

* :class:`MomentumStrategy` — classic 12-1 price momentum with a periodic
  rebalance, optionally restricted to names near their 52-week high.
  Price-only, so it is free under yfinance.
* :class:`InsiderClusterStrategy` — several distinct insiders buying inside a
  short window (a "cluster") is the entry. One FD insider-trades request per
  ticker.
* :class:`CommitteeStrategy` — the 13 investor personas vote at each
  historical rebalance date; the top consensus names are bought. This is the
  expensive one: every (ticker, date) snapshot is ~4 FD requests in lean mode.

Nothing here looks at prices after the signal date: a signal computed from
the close of day *t* is dated *t + 1* so the engine fills it at the next
trading day's close.
"""

from __future__ import annotations

import logging
import re
from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Callable, Iterable, NamedTuple

from v2.backtesting.models import TradeSignal
from v2.backtesting.strategy import Strategy

logger = logging.getLogger(__name__)

Progress = Callable[[int], None]

STRATEGY_KEYS = ("pead", "momentum", "insider", "committee")


# --------------------------------------------------------------------------- helpers

def _iso(value: Any) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)[:10]


def _d(value: Any) -> date:
    return date.fromisoformat(_iso(value))


def _field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if f == f else None


# ----------------------------------------------------------------------- price cache

class Bar(NamedTuple):
    """What the cache keeps of a provider row: the date and the close.

    Every consumer (engine fills, momentum ranking, the event study) reads only
    ``time`` and ``close``; a full pydantic row is ~5× the memory, which is the
    difference between a 10-year S&P 500 run fitting in RAM or swapping.
    """

    time: str
    close: float | None


class PriceCache:
    """One wide fetch per ticker, then serve every sub-range from memory.

    The engine asks for a small window around every signal; without this a
    momentum backtest would make one provider call per trade. ``chunk_days``
    splits the wide fetch into consecutive windows for providers that cap the
    number of bars per request (Financial Datasets returns ~100), and every
    provider call is counted in ``requests`` so the run can be priced.
    Rows are stored as :class:`Bar` (date + close), sorted once per ticker.
    """

    def __init__(self, source: Any, *, chunk_days: int | None = None, today: date | None = None) -> None:
        self._source = source
        self._chunk = chunk_days
        self._today = today or date.today()
        self._dates: dict[str, list[str]] = {}         # ticker -> dates ascending
        self._closes: dict[str, list[float | None]] = {}  # ticker -> closes aligned with _dates
        self._covered: dict[str, date] = {}            # ticker -> earliest date fetched
        self.requests = 0
        self.failed: dict[str, str] = {}

    def _fetch(self, ticker: str, start: date, end: date) -> list[Any]:
        if self._chunk is None:
            rows = list(self._source.get_prices(ticker, start.isoformat(), end.isoformat()) or [])
            self.requests += 1  # counted after the call: a rejected request is not billed
            return rows
        out: list[Any] = []
        cursor = start
        while cursor <= end:
            stop = min(cursor + timedelta(days=self._chunk - 1), end)
            out.extend(self._source.get_prices(ticker, cursor.isoformat(), stop.isoformat()) or [])
            self.requests += 1
            cursor = stop + timedelta(days=1)
        return out

    def warm(self, ticker: str, start: Any) -> None:
        """Make sure ``[start, today]`` is in memory for ``ticker``."""
        start_d = _d(start)
        have = self._covered.get(ticker)
        if have is not None and have <= start_d:
            return
        end = self._today
        if have is not None:
            end = have - timedelta(days=1)
        try:
            rows = self._fetch(ticker, start_d, end)
        except Exception as exc:  # noqa: BLE001 — a dead ticker must not kill the run
            logger.warning("prices %s failed: %s", ticker, exc)
            self.failed[ticker] = f"{type(exc).__name__}: {str(exc)[:120]}"
            rows = []
        merged = dict(zip(self._dates.get(ticker, []), self._closes.get(ticker, [])))
        for p in rows:
            t = _iso(_field(p, "time") or _field(p, "date") or "")
            if t:
                merged[t] = _num(_field(p, "close"))
        dates = sorted(merged)
        self._dates[ticker], self._closes[ticker] = dates, [merged[t] for t in dates]
        self._covered[ticker] = start_d

    def get_prices(self, ticker: str, start: Any, end: Any) -> list[Bar]:
        self.warm(ticker, start)
        s, e = _iso(start), _iso(end)
        dates, closes = self._dates.get(ticker, []), self._closes.get(ticker, [])
        lo, hi = bisect_left(dates, s), bisect_right(dates, e)
        return [Bar(t, c) for t, c in zip(dates[lo:hi], closes[lo:hi])]

    def series(self, ticker: str, start: Any) -> tuple[list[str], list[float]]:
        """``(dates, closes)`` ascending from ``start`` to today, bad rows dropped.

        Two parallel lists rather than one tuple per bar: for a 10-year index
        run the strategy's working set is otherwise several times the cache.
        """
        self.warm(ticker, start)
        dates, closes = self._dates.get(ticker, []), self._closes.get(ticker, [])
        lo = bisect_left(dates, _iso(start))
        out_d: list[str] = []
        out_c: list[float] = []
        for t, c in zip(dates[lo:], closes[lo:]):
            if c is not None and c > 0:
                out_d.append(t)
                out_c.append(c)
        return out_d, out_c

    def closes(self, ticker: str, start: Any) -> list[tuple[str, float]]:
        """``(date, close)`` ascending from ``start`` to today, skipping bad rows."""
        dates, closes = self.series(ticker, start)
        return list(zip(dates, closes))


@dataclass
class BacktestData:
    """What a strategy (and the engine) may read from.

    ``prices`` serves daily bars (usually a :class:`PriceCache`); ``fd`` is the
    persona-style data client for insider trades / fundamentals (``None`` when
    the run is price-only); ``raw`` is the production FD client for endpoints
    the persona adapter does not cover (earnings history for PEAD).
    ``fd_requests`` accumulates paid requests by endpoint.
    """

    prices: Any
    fd: Any = None
    raw: Any = None
    fd_requests: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    today: date = field(default_factory=date.today)
    progress: Progress | None = None
    _earnings_calls: int = 0

    def get_prices(self, ticker: str, start: Any, end: Any) -> list[Any]:
        return self.prices.get_prices(ticker, start, end)

    def get_earnings_history(self, ticker: str, limit: int = 12) -> list[Any]:
        """PEAD's feed: one paid request per ticker, reported as progress."""
        if self.raw is None:
            raise ValueError("PEAD needs the Financial Datasets client")
        if self.progress:
            self.progress(self._earnings_calls)
        self._earnings_calls += 1
        self.count("earnings")
        return self.raw.get_earnings_history(ticker, limit=limit)

    def count(self, endpoint: str, n: int = 1) -> None:
        self.fd_requests[endpoint] = self.fd_requests.get(endpoint, 0) + n


def rebalance_dates(*, today: date, history_days: int, step_trading_days: int, lag_days: int = 0) -> list[str]:
    """Signal dates every ``step_trading_days`` (≈ calendar) back over ``history_days``.

    The most recent date leaves room for one full holding period so the last
    trade can close before today; ``lag_days`` shifts everything earlier.
    """
    step = max(1, round(step_trading_days * 365 / 252))
    first_allowed = today - timedelta(days=history_days)
    out: list[date] = []
    cursor = today - timedelta(days=step + lag_days)
    while cursor >= first_allowed:
        out.append(cursor)
        cursor -= timedelta(days=step)
    return [d.isoformat() for d in sorted(out)]


# --------------------------------------------------------------------------- momentum

class MomentumStrategy(Strategy):
    """12-1 momentum: rank by ``close[t-skip] / close[t-lookback] - 1``, buy the top N.

    Every ``holding_days`` trading days the ranking is redone and a fresh set
    of long signals is emitted; each position is held for exactly one period.
    ``near_high_pct`` (e.g. 0.05) additionally requires the close to be within
    that fraction of the trailing 52-week high — the breakout variant.
    """

    def __init__(
        self,
        *,
        lookback_days: int = 252,
        skip_days: int = 21,
        holding_days: int = 21,
        top_n: int = 5,
        history_days: int = 730,
        near_high_pct: float | None = None,
        min_momentum: float = 0.0,
        progress: Progress | None = None,
        universe_at: Callable[[str], list[str]] | None = None,
    ) -> None:
        if skip_days >= lookback_days:
            raise ValueError("skip_days must be smaller than lookback_days")
        self.lookback, self.skip, self.holding = lookback_days, skip_days, holding_days
        self.top_n, self.history_days = top_n, history_days
        self.near_high, self.min_momentum = near_high_pct, min_momentum
        self._progress = progress
        #: date → constituents on that date (point-in-time universe); None = rank every ticker given
        self.universe_at = universe_at
        #: one entry per rebalance date: pool size, how many had enough history, what was bought
        self.periods: list[dict[str, Any]] = []
        #: tickers the strategy never got a usable price series for
        self.no_data: list[str] = []

    @property
    def name(self) -> str:
        return "momentum"

    def generate_signals(self, tickers: list[str], data: Any) -> list[TradeSignal]:
        data = _as_data(data)
        # history + lookback in calendar days, with slack for holidays
        start = data.today - timedelta(days=self.history_days + int(self.lookback * 1.6) + 10)
        # ticker -> (dates, closes), two parallel lists sharing the cache's date strings
        series: dict[str, tuple[list[str], list[float]]] = {}
        self.no_data = []
        for i, t in enumerate(tickers):
            if self._progress:
                self._progress(i)
            dates, closes = _series(data.prices, t, start)
            if len(dates) > self.lookback:
                series[t] = (dates, closes)
            else:
                self.no_data.append(t)
        if not series:
            return []

        # Trading calendar = union of dates; rebalance every `holding` trading days
        calendar = sorted({d for ds, _ in series.values() for d in ds})
        first_signal = (data.today - timedelta(days=self.history_days)).isoformat()
        signals: list[TradeSignal] = []
        step = max(1, self.holding)
        start_i = next((i for i, d in enumerate(calendar) if d >= first_signal), len(calendar))
        self.periods = []
        for ci in range(start_i, len(calendar) - 1, step):
            d = calendar[ci]
            allowed = set(self.universe_at(d)) if self.universe_at else None
            scored: list[tuple[float, str, float | None]] = []
            ranked = 0
            for t, (dates, closes) in series.items():
                if allowed is not None and t not in allowed:
                    continue
                i = bisect_left(dates, d)
                if i >= len(dates) or dates[i] != d or i < self.lookback:
                    continue  # no bar on this date, or not enough history yet
                now = closes[i - self.skip] if self.skip else closes[i]
                then = closes[i - self.lookback]
                if then <= 0:
                    continue
                mom = now / then - 1
                ranked += 1
                from_high = None
                if self.near_high is not None:
                    high = max(closes[max(0, i - 251): i + 1])
                    from_high = closes[i] / high - 1
                    if from_high < -self.near_high:
                        continue
                if mom > self.min_momentum:
                    scored.append((mom, t, from_high))
            scored.sort(reverse=True)
            entry = (date.fromisoformat(d) + timedelta(days=1)).isoformat()
            self.periods.append({"signal_date": d, "pool": len(allowed) if allowed is not None else len(tickers),
                                 "priced": len(allowed & set(series)) if allowed is not None else len(series), "ranked": ranked,
                                 "picked": [t for _, t, _ in scored[: self.top_n]]})
            for mom, t, from_high in scored[: self.top_n]:
                meta: dict[str, Any] = {"signal_date": d, "momentum": round(mom, 4)}
                if from_high is not None:
                    meta["pct_from_52w_high"] = round(from_high, 4)
                signals.append(TradeSignal(ticker=t, direction="long", entry_date=entry, holding_days=self.holding, metadata=meta))
        return signals


# ---------------------------------------------------------------------------- insiders

def _insider_side(trade: Any) -> str | None:
    kind = _field(trade, "transaction_type")
    if kind:
        kind = str(kind).lower()
        if kind in ("buy", "purchase", "p", "p-purchase"):
            return "buy"
        if kind in ("sell", "sale", "s", "s-sale"):
            return "sell"
        return None
    shares = _num(_field(trade, "transaction_shares"))
    if shares is None or shares == 0:
        return None
    return "buy" if shares > 0 else "sell"


def _trade_value(trade: Any) -> float:
    value = _num(_field(trade, "transaction_value"))
    if value is not None:
        return abs(value)
    shares = _num(_field(trade, "transaction_shares")) or 0.0
    price = _num(_field(trade, "transaction_price_per_share")) or 0.0
    return abs(shares * price)


class InsiderClusterStrategy(Strategy):
    """Buy when ``min_insiders`` distinct insiders bought within ``window_days``.

    Open-market purchases only (positive share count / a buy type). The
    cluster is dated by the filing that completes it; one signal per ticker
    per ``holding_days`` window so a long buying streak is not re-entered
    every day. ``min_value_usd`` is the cluster's total purchase value.
    """

    def __init__(
        self,
        *,
        window_days: int = 30,
        min_insiders: int = 2,
        min_value_usd: float = 100_000.0,
        holding_days: int = 63,
        history_days: int = 730,
        progress: Progress | None = None,
    ) -> None:
        self.window, self.min_insiders, self.min_value = window_days, min_insiders, min_value_usd
        self.holding, self.history_days = holding_days, history_days
        self._progress = progress

    @property
    def name(self) -> str:
        return "insider"

    def generate_signals(self, tickers: list[str], data: Any) -> list[TradeSignal]:
        data = _as_data(data)
        if data.fd is None:
            raise ValueError("insider strategy needs the Financial Datasets client")
        end = data.today.isoformat()
        start = (data.today - timedelta(days=self.history_days + self.window)).isoformat()
        signals: list[TradeSignal] = []
        for i, t in enumerate(tickers):
            if self._progress:
                self._progress(i)
            data.count("insider_trades")
            try:
                rows = data.fd.get_insider_trades(t, end, start_date=start, limit=1000) or []
            except Exception as exc:  # noqa: BLE001
                logger.warning("insider trades %s failed: %s", t, exc)
                continue
            signals.extend(self.cluster_signals(t, rows))
        return signals

    def cluster_signals(self, ticker: str, rows: Iterable[Any]) -> list[TradeSignal]:
        buys = []
        for r in rows:
            if _insider_side(r) != "buy":
                continue
            when = _field(r, "filing_date") or _field(r, "transaction_date")
            if not when:
                continue
            name = str(_field(r, "name") or _field(r, "insider_name") or _field(r, "title") or "?").strip().lower()
            buys.append((_iso(when), name, _trade_value(r)))
        buys.sort()
        signals: list[TradeSignal] = []
        blocked_until: date | None = None
        for j, (when, _, _) in enumerate(buys):
            day = date.fromisoformat(when)
            if blocked_until and day <= blocked_until:
                continue
            lo = (day - timedelta(days=self.window)).isoformat()
            window = [b for b in buys[: j + 1] if b[0] >= lo]
            names = {b[1] for b in window}
            value = sum(b[2] for b in window)
            if len(names) >= self.min_insiders and value >= self.min_value:
                signals.append(TradeSignal(
                    ticker=ticker, direction="long", entry_date=(day + timedelta(days=1)).isoformat(), holding_days=self.holding,
                    metadata={"signal_date": when, "insiders": len(names), "cluster_value_usd": round(value, 2), "window_days": self.window},
                ))
                blocked_until = day + timedelta(days=round(self.holding * 365 / 252))
        return signals


# --------------------------------------------------------------------------- committee

class CommitteeStrategy(Strategy):
    """Let the investor personas vote at every rebalance date; buy the top N.

    Fundamentals are requested as of ``signal_date - filing_lag_days`` so a
    quarter that had not been filed yet on the signal date is not used (FD
    filters by report period, not filing date). Snapshots go through
    ``store`` when given, so re-running the same dates costs nothing.
    """

    def __init__(
        self,
        *,
        holding_days: int = 63,
        history_days: int = 730,
        top_n: int = 5,
        min_consensus: float = 0.2,
        min_agreement: float = 0.5,
        personas: Iterable[str] | None = None,
        lean: bool = True,
        filing_lag_days: int = 45,
        store: Any = None,
        progress: Progress | None = None,
        max_workers: int = 4,
    ) -> None:
        self.holding, self.history_days, self.top_n = holding_days, history_days, top_n
        self.min_consensus, self.min_agreement = min_consensus, min_agreement
        self.personas = list(personas) if personas else None
        self.lean, self.lag, self.store = lean, filing_lag_days, store
        self._progress, self._workers = progress, max_workers
        self.dates: list[str] = []
        self.errors: dict[str, str] = {}
        #: one entry per rebalance date: every ticker's vote and whether it was bought
        self.periods: list[dict[str, Any]] = []
        #: set when a rebalance date's fetches failed wholesale and the run stopped early
        self.aborted: dict[str, Any] | None = None

    @property
    def name(self) -> str:
        return "committee"

    def plan(self, today: date) -> list[str]:
        self.dates = rebalance_dates(today=today, history_days=self.history_days, step_trading_days=self.holding)
        return self.dates

    def generate_signals(self, tickers: list[str], data: Any) -> list[TradeSignal]:
        from v2.personas.committee import run_committee

        data = _as_data(data)
        if data.fd is None:
            raise ValueError("committee strategy needs the Financial Datasets client")
        dates = self.plan(data.today)
        exclude = ("news", "insiders") if self.lean else ()
        signals: list[TradeSignal] = []
        done = 0
        for signal_date in dates:
            as_of = (date.fromisoformat(signal_date) - timedelta(days=self.lag)).isoformat()
            cached = {}
            if self.store is not None:
                for t in tickers:
                    snap = self.store.cached_snapshot(t, as_of, max_age_hours=24 * 3650)
                    if snap is not None:
                        cached[t] = snap
            result = run_committee(tickers, data.fd, personas=self.personas, as_of=as_of, snapshots=cached,
                                   max_workers=self._workers, exclude_needs=exclude)
            for t, snap in result.snapshots.items():
                if t not in cached:
                    for endpoint, n in snap.requests.items():
                        data.count(endpoint, n)
                    if self.store is not None:
                        self.store.save_snapshot(snap)
            for t, err in result.errors.items():
                self.errors[f"{t}@{as_of}"] = err
            picked = [v for v in result.verdicts if v.voters > 0 and v.consensus >= self.min_consensus and v.agreement >= self.min_agreement]
            chosen = {v.ticker for v in picked[: self.top_n]}
            self.periods.append({
                "signal_date": signal_date, "as_of": as_of, "picked": sorted(chosen),
                "verdicts": [{"ticker": v.ticker, "consensus": round(v.consensus, 3), "agreement": round(v.agreement, 2), "voters": v.voters,
                              "abstained": v.abstained, "bullish": v.bullish, "bearish": v.bearish, "neutral": v.neutral,
                              "gaps": len(v.data_gaps), "reasons": abstain_reasons(v), "picked": v.ticker in chosen} for v in result.verdicts],
                "missing": sorted(set(tickers) - {v.ticker for v in result.verdicts}),
            })
            failed = [v for v in result.verdicts if core_gaps(v.data_gaps)] + [t for t in tickers if t not in result.snapshots]
            if len(tickers) >= 2 and len(failed) > len(tickers) / 2:
                sample = next((g for v in result.verdicts for g in v.data_gaps if core_gaps([g])), None) or next(iter(result.errors.values()), "")
                self.aborted = {"signal_date": signal_date, "as_of": as_of, "failed": len(failed), "of": len(tickers),
                                "reason": (sample or "provider requests failed")[:200], "remaining_dates": dates[dates.index(signal_date) + 1:]}
                logger.warning("committee backtest stopped at %s: %d/%d tickers without core data (%s)", signal_date, len(failed), len(tickers), sample)
                break
            entry = (date.fromisoformat(signal_date) + timedelta(days=1)).isoformat()
            for v in picked[: self.top_n]:
                signals.append(TradeSignal(
                    ticker=v.ticker, direction="long", entry_date=entry, holding_days=self.holding,
                    metadata={"signal_date": signal_date, "as_of": as_of, "consensus": round(v.consensus, 3), "agreement": round(v.agreement, 2),
                              "bullish": v.bullish, "bearish": v.bearish, "neutral": v.neutral, "abstained": v.abstained},
                ))
            done += len(tickers)
            if self._progress:
                self._progress(done)
        return signals


def core_gaps(gaps: Iterable[str]) -> list[str]:
    """Gap messages about fundamentals (a snapshot without them is a failed fetch)."""
    return [g for g in gaps if g.startswith(("metrics_", "line_items_"))]


def abstain_reasons(verdict: Any, limit: int = 2) -> list[str]:
    """The most common abstention notes across the personas, short enough for a table cell."""
    counts: dict[str, int] = {}
    for sig in getattr(verdict, "signals", []) or []:
        if not getattr(sig, "abstained", False):
            continue
        note = str(getattr(sig, "reasoning", "") or "")
        note = note[len("abstain · "):] if note.startswith("abstain · ") else note
        note = note.replace("missing inputs: ", "")
        # keep the provider's error but not the whole URL/traceback
        note = re.sub(r"\s+", " ", note)[:120]
        if note:
            counts[note] = counts.get(note, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])[:limit]
    return [f"{n} × {c}" if c > 1 else n for n, c in ranked]


def _series(prices: Any, ticker: str, start: Any) -> tuple[list[str], list[float]]:
    """``(dates, closes)`` from a :class:`PriceCache`, or built from ``closes()`` on anything else."""
    if hasattr(prices, "series"):
        return prices.series(ticker, start)
    pairs = prices.closes(ticker, start)
    return [d for d, _ in pairs], [c for _, c in pairs]


def _as_data(data: Any) -> BacktestData:
    if isinstance(data, BacktestData):
        return data
    # A bare client (tests, CLI): prices and everything else from the same object.
    return BacktestData(prices=PriceCache(data), fd=data, raw=data)
