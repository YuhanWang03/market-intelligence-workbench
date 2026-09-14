"""Generic backtesting engine.

Simulates trade execution from strategy-generated signals.
Handles position sizing, price lookup, equity curve, and metrics.
The engine is strategy-agnostic — it only sees TradeSignals.

Usage:
    from v2.data import FDClient
    from v2.backtesting import BacktestEngine, PEADStrategy

    with FDClient() as fd:
        strategy = PEADStrategy(holding_days=5)
        engine = BacktestEngine(capital=100_000, per_trade=10_000)
        result = engine.run(strategy, ["AAPL", "MSFT"], fd)
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta

import numpy as np

from v2.backtesting.models import (
    BacktestResult,
    PerformanceMetrics,
    Trade,
    TradeSignal,
)
from v2.backtesting.strategy import Strategy
from v2.data.client import FDClient

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Strategy-agnostic backtesting engine.

    Given a list of TradeSignals, the engine:
    1. Fetches prices for each signal's ticker + date range.
    2. Fills each signal at the next available trading day's close.
    3. Exits after the signal's holding_days trading days.
    4. Computes position-level P&L based on equal-dollar sizing.
    5. Builds an equity curve and performance metrics.
    """

    def __init__(
        self,
        *,
        capital: float = 100_000.0,
        per_trade: float = 10_000.0,
        cost_bps: float = 0.0,
    ) -> None:
        self._capital = capital
        self._per_trade = per_trade
        #: one-way transaction cost in basis points, charged on entry and exit
        self._cost_bps = cost_bps

    def run(
        self,
        strategy: Strategy,
        tickers: list[str],
        fd_client: FDClient,
    ) -> BacktestResult:
        """Run a full backtest: generate signals, then execute them.

        Args:
            strategy:  Strategy instance that produces TradeSignals.
            tickers:   Stock symbols to pass to the strategy.
            fd_client: Data client for price lookups.
        """
        signals = strategy.generate_signals(tickers, fd_client)
        return self.run_signals(signals, fd_client)

    def run_signals(
        self,
        signals: list[TradeSignal],
        fd_client: FDClient,
    ) -> BacktestResult:
        """Execute a list of pre-built signals.

        Useful for testing, manual signal lists, or compositing
        signals from multiple strategies before execution.
        """
        if not signals:
            return BacktestResult()

        # Fill each signal into a Trade (or skip if prices unavailable)
        trades: list[Trade] = []
        for signal in signals:
            trade = self._fill_signal(signal, fd_client)
            if trade is not None:
                trades.append(trade)

        if not trades:
            return BacktestResult()

        # Sort chronologically, build equity curve, compute stats
        trades.sort(key=lambda t: t.entry_date)
        equity_curve = self._build_equity_curve(trades)
        metrics = self._compute_metrics(trades, equity_curve)

        return BacktestResult(
            trades=trades,
            metrics=metrics,
            equity_curve=equity_curve,
        )

    # ------------------------------------------------------------------
    # Signal -> Trade conversion
    # ------------------------------------------------------------------

    def _fill_signal(
        self,
        signal: TradeSignal,
        fd_client: FDClient,
    ) -> Trade | None:
        """Convert a TradeSignal into a filled Trade using market prices."""
        entry = _parse_date(signal.entry_date)

        # Fetch prices around the signal's entry window
        price_start = (entry - timedelta(days=5)).isoformat()
        price_end = (entry + timedelta(days=signal.holding_days * 2 + 10)).isoformat()

        # Don't request future dates
        today = date.today()
        if _parse_date(price_end) > today:
            price_end = today.isoformat()

        prices = fd_client.get_prices(signal.ticker, price_start, price_end)
        if not prices:
            return None

        # Build date -> close lookup
        price_map = {p.time[:10]: p.close for p in prices}
        trading_days = sorted(price_map.keys())

        # Entry: first trading day on or after the signal's desired date
        entry_date = _find_next_trading_day(signal.entry_date, trading_days)
        if entry_date is None:
            return None

        # Exit: holding_days trading days after entry
        entry_idx = trading_days.index(entry_date)
        exit_idx = entry_idx + signal.holding_days
        if exit_idx >= len(trading_days):
            return None
        exit_date = trading_days[exit_idx]

        entry_price = price_map[entry_date]
        exit_price = price_map[exit_date]

        # Equal-dollar position sizing
        shares = self._per_trade / entry_price

        # P&L depends on direction
        if signal.direction == "long":
            pnl = shares * (exit_price - entry_price)
            return_pct = (exit_price - entry_price) / entry_price
        else:
            pnl = shares * (entry_price - exit_price)
            return_pct = (entry_price - exit_price) / entry_price
        # Round-trip transaction cost (entry + exit), on the dollars deployed
        round_trip = 2 * self._cost_bps / 10_000
        if round_trip:
            pnl -= self._per_trade * round_trip
            return_pct -= round_trip

        return Trade(
            ticker=signal.ticker,
            direction=signal.direction,
            entry_date=entry_date,
            exit_date=exit_date,
            entry_price=entry_price,
            exit_price=exit_price,
            shares=round(shares, 4),
            pnl=round(pnl, 2),
            return_pct=round(return_pct, 6),
            holding_days=signal.holding_days,
            metadata=signal.metadata,
        )

    # ------------------------------------------------------------------
    # Equity curve
    # ------------------------------------------------------------------

    def _build_equity_curve(self, trades: list[Trade]) -> list[float]:
        """Portfolio value after each rebalance period (trades sharing an entry date)."""
        equity = self._capital
        curve = [equity]
        for _, pnl in _period_pnl(trades):
            equity += pnl
            curve.append(round(equity, 2))
        return curve

    # ------------------------------------------------------------------
    # Performance metrics
    # ------------------------------------------------------------------

    def _compute_metrics(
        self,
        trades: list[Trade],
        equity_curve: list[float],
    ) -> PerformanceMetrics:
        """Compute Sharpe, drawdown, win rate, and other stats."""
        returns = [t.return_pct for t in trades]
        n = len(returns)

        # Total return
        final_equity = equity_curve[-1]
        total_return_pct = (final_equity - self._capital) / self._capital

        # Annualized return
        first_entry = _parse_date(trades[0].entry_date)
        last_exit = _parse_date(trades[-1].exit_date)
        calendar_days = (last_exit - first_entry).days
        years = max(calendar_days / 365.25, 0.01)
        annualized = (1 + total_return_pct) ** (1 / years) - 1

        # Trade-level Sharpe (kept for reference): treats every trade as independent,
        # which overstates it when many positions are opened on the same day.
        arr = np.array(returns)
        avg = float(arr.mean())
        std = float(arr.std(ddof=1)) if n > 1 else 1.0
        trades_per_year = n / years if years > 0 else n
        sharpe_trades = (avg / std) * np.sqrt(trades_per_year) if std > 0 else 0.0

        # Portfolio Sharpe: one observation per rebalance period (trades sharing an
        # entry date, equal-dollar → the period return is their mean return).
        period_returns = np.array([r for _, r in _period_returns(trades)])
        n_periods = len(period_returns)
        p_avg = float(period_returns.mean())
        p_std = float(period_returns.std(ddof=1)) if n_periods > 1 else 0.0
        periods_per_year = n_periods / years if years > 0 else n_periods
        sharpe = (p_avg / p_std) * np.sqrt(periods_per_year) if p_std > 0 else 0.0

        # Max drawdown from equity curve
        peak = equity_curve[0]
        max_dd = 0.0
        for val in equity_curve:
            if val > peak:
                peak = val
            dd = (peak - val) / peak
            if dd > max_dd:
                max_dd = dd

        # Win rate
        wins = sum(1 for r in returns if r > 0)

        return PerformanceMetrics(
            total_return_pct=round(total_return_pct, 6),
            annualized_return_pct=round(annualized, 6),
            sharpe_ratio=round(sharpe, 4),
            max_drawdown_pct=round(max_dd, 6),
            win_rate=round(wins / n, 4) if n > 0 else 0.0,
            n_trades=n,
            n_long=sum(1 for t in trades if t.direction == "long"),
            n_short=sum(1 for t in trades if t.direction == "short"),
            avg_return_pct=round(avg, 6),
            avg_holding_days=round(sum(t.holding_days for t in trades) / n, 1),
            n_periods=n_periods,
            sharpe_trade_level=round(float(sharpe_trades), 4),
            cost_bps=self._cost_bps,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_date(s: str) -> date:
    return datetime.strptime(s[:10], "%Y-%m-%d").date()


def _period_pnl(trades: list[Trade]) -> list[tuple[str, float]]:
    """``(entry_date, total P&L)`` per rebalance period, chronological."""
    totals: dict[str, float] = {}
    for t in trades:
        totals[t.entry_date] = totals.get(t.entry_date, 0.0) + t.pnl
    return sorted(totals.items())


def _period_returns(trades: list[Trade]) -> list[tuple[str, float]]:
    """``(entry_date, mean trade return)`` per period — the equal-dollar portfolio return."""
    groups: dict[str, list[float]] = {}
    for t in trades:
        groups.setdefault(t.entry_date, []).append(t.return_pct)
    return sorted((d, float(np.mean(r))) for d, r in groups.items())


def yearly_breakdown(trades: list[Trade], capital: float) -> list[dict]:
    """One row per calendar year, trades grouped by the year they were *entered*.

    ``return_pct`` is that year's P&L over the equity at the start of the year
    (the engine sizes every trade at a fixed dollar amount, so equity is
    additive). ``start`` / ``end`` are the first entry and the last exit of the
    year's trades — the span a benchmark should be measured over; a December
    entry exits in January, so consecutive spans overlap by one period.
    """
    if not trades:
        return []
    equity = float(capital)
    years: dict[str, dict] = {}
    for entry_date, pnl in _period_pnl(trades):
        year = entry_date[:4]
        row = years.setdefault(year, {"year": year, "periods": 0, "trades": 0, "pnl": 0.0, "start_equity": equity,
                                      "start": entry_date, "end": entry_date, "wins": 0})
        row["periods"] += 1
        row["pnl"] += pnl
        equity += pnl
    for t in trades:
        row = years[t.entry_date[:4]]
        row["trades"] += 1
        row["wins"] += 1 if t.return_pct > 0 else 0
        if t.exit_date > row["end"]:
            row["end"] = t.exit_date
    out = []
    for year in sorted(years):
        r = years[year]
        out.append({"year": year, "periods": r["periods"], "trades": r["trades"], "start": r["start"], "end": r["end"],
                    "pnl": round(r["pnl"], 2), "start_equity": round(r["start_equity"], 2),
                    "return_pct": round(r["pnl"] / r["start_equity"], 6) if r["start_equity"] > 0 else None,
                    "win_rate": round(r["wins"] / r["trades"], 4) if r["trades"] else None})
    return out


def _find_next_trading_day(
    date_str: str,
    trading_days: list[str],
) -> str | None:
    """Find the first trading day on or after date_str."""
    for td in trading_days:
        if td >= date_str:
            return td
    return None
