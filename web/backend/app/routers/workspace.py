"""Unified endpoints consumed by the Core / Research / Lab workbench.

The existing dashboard endpoints remain unchanged.  This router exposes the
remaining production state (push feed, watchlist, price alerts) and gives the
already-existing offline research engines a small, validated HTTP surface.
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
import sqlite3
import threading
from v2.usage_context import ContextThread
import uuid
from datetime import datetime, timedelta, timezone
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, field_validator

from app.auth import require_owner
from app.config import SETTINGS
from app.fd_pricing import cost as fd_cost
from app.screening import CRITERIA, DEFAULT_RULES, Rule, screen as lab_screen
from app.lab_store import LabRunStore
from app.sources import BIG_LIMIT, INDEX_UNIVERSES, MAX_TICKERS, Universe, normalize_tickers, resolve_universe
from v2.archive.store import recent_trading_day_cutoff_iso

router = APIRouter(prefix="/api", tags=["workspace"], dependencies=[Depends(require_owner)])

_LAB_TIMEOUT_SECONDS = 240
_LAB_STORE: LabRunStore | None = None


def _lab_store() -> LabRunStore:
    global _LAB_STORE
    if _LAB_STORE is None:
        _LAB_STORE = LabRunStore()
    return _LAB_STORE


def _summarize(kind: str, result: dict) -> dict:
    """Compact, list-friendly view of one run; the full result is stored alongside."""
    summary: dict = {"tickers": result.get("tickers", [])}
    if kind == "backtest" and result.get("kind") == "sweep":
        rows = [r for r in (result.get("rows") or []) if r.get("sharpe_ratio") is not None]
        best = max(rows, key=lambda r: r["sharpe_ratio"], default=None)
        summary.update({"strategy": "momentum", "sweep": True, "universe": result.get("universe"), "n_combos": len(result.get("rows") or []),
                        "data_source": result.get("data_source"), "fd_cost_usd": result.get("fd_cost_usd"),
                        "n_trades": best["n_trades"] if best else 0, "total_return_pct": best["total_return_pct"] if best else None,
                        "sharpe_ratio": best["sharpe_ratio"] if best else None, "max_drawdown_pct": best["max_drawdown_pct"] if best else None,
                        "excess_return_pct": best["excess_return_pct"] if best else None,
                        "best": {k: best[k] for k in ("top_n", "holding_days", "near_high_pct")} if best else None})
    elif kind == "backtest":
        m = result.get("metrics") or {}
        summary.update({"strategy": result.get("strategy"), "universe": result.get("universe"), "n_trades": m.get("n_trades", 0),
                        "total_return_pct": m.get("total_return_pct"), "sharpe_ratio": m.get("sharpe_ratio"), "max_drawdown_pct": m.get("max_drawdown_pct"),
                        "data_source": result.get("data_source"), "fd_cost_usd": result.get("fd_cost_usd"), "excess_return_pct": result.get("excess_return_pct")})
    elif kind == "event_study":
        summary.update({"universe": result.get("universe"), "n_events": len(result.get("events") or []), "n_groups": len(result.get("aggregates") or []),
                        "data_source": result.get("data_source"), "fd_cost_usd": result.get("fd_cost_usd")})
    elif kind == "screening":
        summary.update({"universe": result.get("universe"), "universe_size": result.get("universe_size"), "n_candidates": len(result.get("candidates") or []),
                        "candidates": [c.get("ticker") for c in (result.get("candidates") or [])][:20],
                        "data_source": result.get("data_source"), "fd_cost_usd": result.get("fd_cost_usd")})
    elif kind == "committee":
        verdicts = result.get("verdicts") or []
        summary.update({"run_id": result.get("run_id"), "source": result.get("source"), "n_tickers": len(verdicts), "fd_cost_usd": result.get("fd_cost_usd"),
                        "tickers": [v.get("ticker") for v in verdicts], "top": [t.get("ticker") for t in (result.get("top") or [])[:5]],
                        "stances": {k: sum(1 for v in verdicts if v.get("stance") == k) for k in ("bullish", "bearish", "neutral", "abstain")}})
    elif kind == "backfill":
        summary.update({"checked": result.get("checked"), "filled": result.get("filled")})
    return summary


def _remember_run(kind: str, result: dict, params: dict | None = None) -> str:
    """Persist a run and stamp its id onto the result."""
    run_id = _lab_store().save(kind, params=params, summary=_summarize(kind, result), result=result)
    result["lab_run_id"] = run_id
    return run_id


def _normalize_tickers(values: list[str], *, limit: int = MAX_TICKERS) -> list[str]:
    return normalize_tickers(values, limit=limit)


@router.get("/activity")
async def activity(
    days: int = Query(2, ge=1, le=30),
    limit: int = Query(100, ge=1, le=200),
    realtime_only: bool = Query(False),
) -> dict:
    """Recent archived pushes, used as the real Core alert feed."""
    db_path = SETTINGS.archive_db_path
    if not db_path.exists():
        return {"items": [], "warning": "archive.db not found"}

    def _fetch() -> list[dict]:
        cutoff = (
            recent_trading_day_cutoff_iso(2)
            if realtime_only
            else (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        )
        conn = sqlite3.connect(str(db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        try:
            columns = {row[1] for row in conn.execute("PRAGMA table_info(pushes)")}
            if not columns:
                return []
            optional = {
                "title": "title" if "title" in columns else "NULL AS title",
                "priority_tier": "priority_tier" if "priority_tier" in columns else "NULL AS priority_tier",
                "importance_score": "importance_score" if "importance_score" in columns else "NULL AS importance_score",
            }
            realtime_clause = """
                AND (agent IN ('intraday_anomaly', 'alert', 'anomaly')
                     OR msg_type = 'intraday_anomaly')
            """ if realtime_only else ""
            sql = f"""
                SELECT id, ts, agent, msg_type, tickers,
                       substr(COALESCE(text_html, ''), 1, 1000) AS preview,
                       {optional['title']}, {optional['priority_tier']},
                       {optional['importance_score']}
                FROM pushes
                WHERE ts >= ?
                {realtime_clause}
                ORDER BY ts DESC
                LIMIT ?
            """
            rows = conn.execute(sql, (cutoff, limit)).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    try:
        return {"items": await run_in_threadpool(_fetch)}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/costs")
async def query_costs(
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    recent_filter: Literal['all', 'data', 'llm', 'search', 'pending'] = Query('all', alias='filter'),
) -> dict:
    """Estimated spend for successful, uncached paid data requests."""
    from v2.data.usage_ledger import report as cost_report

    return await run_in_threadpool(cost_report, limit, recent_filter, offset, 24)


@router.post('/costs/refresh')
async def refresh_costs(
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    recent_filter: Literal['all', 'data', 'llm', 'search', 'pending'] = Query('all', alias='filter'),
) -> dict:
    """Refresh official prices, reconcile eligible usage, then return one report."""
    from v2.data.price_sync import sync_prices
    from v2.data.usage_ledger import reconcile_tavily_flat_rate, report as cost_report, rollup_and_prune

    retention = await run_in_threadpool(rollup_and_prune, 24)
    price_sync = await run_in_threadpool(sync_prices, force=True)
    tavily_reconciliation = await run_in_threadpool(reconcile_tavily_flat_rate)
    result = await run_in_threadpool(cost_report, limit, recent_filter, offset, 24)
    result['refresh_result'] = {
        'price_sync': price_sync,
        'tavily_reconciliation': tavily_reconciliation,
        'retention': retention,
    }
    return result


@router.post('/costs/prices')
async def add_cost_price(payload: dict) -> dict:
    from v2.data.usage_ledger import add_price
    try:
        return await run_in_threadpool(add_price, payload)
    except (ValueError, KeyError, TypeError, OverflowError):
        raise HTTPException(400, '价格配置无效：请检查模型、币种（CNY／USD）、单价、生效时间、复核期限及来源说明')


@router.post('/costs/prices/sync')
async def sync_official_prices() -> dict:
    from v2.data.price_sync import sync_prices
    from v2.data.usage_ledger import reconcile_pending, reconcile_tavily_flat_rate

    result = await run_in_threadpool(sync_prices, force=True)
    if result.get('status') == 'ok':
        result['reconciliation'] = await run_in_threadpool(reconcile_pending)
    result['tavily_reconciliation'] = await run_in_threadpool(reconcile_tavily_flat_rate)
    return result


@router.get('/costs/deepseek-balance')
async def cost_balance() -> dict:
    from v2.data.provider_balance import deepseek_balance
    return await run_in_threadpool(deepseek_balance)


@router.post('/costs/tavily/sync')
async def sync_tavily_quota() -> dict:
    from v2.data.billing_rules import sync_tavily
    return await run_in_threadpool(sync_tavily)


@router.post('/costs/tavily/calibrate')
async def calibrate_tavily(payload: dict) -> dict:
    from v2.data.billing_rules import save_quota
    from v2.data.usage_ledger import now_iso
    try:
        if payload.get('confirmed') is not True:
            raise ValueError('需要确认账户用量')
        return await run_in_threadpool(save_quota, payload['used'], '用户手动校准账户本月用量', now_iso())
    except (ValueError, KeyError, TypeError, OverflowError):
        raise HTTPException(400, '请确认并填写本月账户实际已用 credits（非本项目记录数）')


@router.post('/costs/model-mappings')
async def add_model_mapping(payload: dict) -> dict:
    from v2.data.billing_rules import configure_alias
    from v2.data.usage_ledger import reconcile_pending
    try:
        mapping = await run_in_threadpool(configure_alias, payload)
    except (ValueError, KeyError, TypeError, OverflowError):
        raise HTTPException(400, '模型映射无效：请确认名称、目标价格、有效时间及依据')
    result = await run_in_threadpool(reconcile_pending)
    return {**mapping, **result}


@router.post('/costs/reconcile')
async def reconcile_costs() -> dict:
    from v2.data.usage_ledger import reconcile_pending
    return await run_in_threadpool(reconcile_pending)


@router.get("/monitoring/universe")
async def monitoring_universe() -> dict:
    """The production ticker pool scanned by the minute-level streamer."""
    from v2.screening.universe import TECH_30
    from v2.bot.state import intraday_universe_list

    intraday = await run_in_threadpool(intraday_universe_list)
    return {
        "intraday": intraday,
        "source": "TECH_30" if intraday == list(TECH_30) else "TECH_30 + 自定义",
        "scan_interval_seconds": 60,
        "price_pct_threshold": 0.03,
        "volume_pace_threshold": 2.5,
    }


class MonitoringTickerInput(BaseModel):
    ticker: str


@router.post("/monitoring/universe")
async def add_monitoring_ticker(body: MonitoringTickerInput) -> dict:
    from v2.bot.state import intraday_universe_add

    try:
        added = await run_in_threadpool(intraday_universe_add, body.ticker)
        return {"added": added, **await monitoring_universe()}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/monitoring/universe/{ticker}")
async def remove_monitoring_ticker(ticker: str) -> dict:
    from v2.bot.state import intraday_universe_remove

    try:
        removed = await run_in_threadpool(intraday_universe_remove, ticker)
        return {"removed": removed, **await monitoring_universe()}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


class WatchlistInput(BaseModel):
    ticker: str
    note: str = Field(default="", max_length=200)


@router.get("/watchlist")
async def watchlist() -> dict:
    from v2.bot.state import watchlist_list

    return {"items": await run_in_threadpool(watchlist_list)}


@router.post("/watchlist")
async def add_watchlist(body: WatchlistInput) -> dict:
    from v2.bot.state import watchlist_add, watchlist_list

    try:
        added = await run_in_threadpool(watchlist_add, body.ticker, body.note)
        items = await run_in_threadpool(watchlist_list)
        return {"added": added, "items": items}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/watchlist/{ticker}")
async def remove_watchlist(ticker: str) -> dict:
    from v2.bot.state import watchlist_list, watchlist_remove

    removed = await run_in_threadpool(watchlist_remove, ticker)
    return {"removed": removed, "items": await run_in_threadpool(watchlist_list)}


class PriceAlertInput(BaseModel):
    ticker: str
    direction: Literal["above", "below"]
    target_price: float = Field(gt=0)


@router.get("/price-alerts")
async def price_alerts(include_fired: bool = False) -> dict:
    from v2.bot.state import alert_list

    return {"items": await run_in_threadpool(alert_list, include_fired)}


@router.post("/price-alerts")
async def add_price_alert(body: PriceAlertInput) -> dict:
    from v2.bot.state import alert_add, alert_list

    try:
        alert_id = await run_in_threadpool(alert_add, body.ticker, body.direction, body.target_price)
        return {"id": alert_id, "items": await run_in_threadpool(alert_list, False)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/price-alerts/{alert_id}")
async def remove_price_alert(alert_id: int) -> dict:
    from v2.bot.state import alert_list, alert_remove

    removed = await run_in_threadpool(alert_remove, alert_id)
    return {"removed": removed, "items": await run_in_threadpool(alert_list, False)}


class BacktestInput(BaseModel):
    universe: Universe = "custom"
    #: up to BIG_LIMIT for the free momentum strategy; paid strategies are checked against MAX_TICKERS in _check_backtest_size
    tickers: list[str] = Field(default_factory=lambda: ["AAPL", "MSFT", "NVDA"], max_length=BIG_LIMIT)
    strategy: Literal["pead", "momentum", "insider", "committee"] = "pead"
    #: where daily prices come from; events / fundamentals are always Financial Datasets
    data_source: Literal["yfinance", "fd"] = "yfinance"
    holding_days: int = Field(default=5, ge=1, le=252)
    capital: float = Field(default=100_000, gt=0, le=100_000_000)
    per_trade: float = Field(default=10_000, gt=0, le=10_000_000)
    #: one-way transaction cost, basis points (10 = 0.1 % per side)
    cost_bps: float = Field(default=10, ge=0, le=200)
    # pead
    earnings_limit: int = Field(default=8, ge=1, le=20)
    # momentum / insider / committee: how far back signals are generated
    history_days: int = Field(default=730, ge=60, le=3650)
    top_n: int = Field(default=5, ge=1, le=60)
    # momentum
    lookback_days: int = Field(default=252, ge=20, le=504)
    skip_days: int = Field(default=21, ge=0, le=120)
    near_high_pct: float | None = Field(default=None, ge=0, le=1)
    # insider cluster
    window_days: int = Field(default=30, ge=1, le=180)
    min_insiders: int = Field(default=2, ge=1, le=20)
    min_value_usd: float = Field(default=100_000, ge=0)
    # committee
    min_consensus: float = Field(default=0.2, ge=-1, le=1)
    min_agreement: float = Field(default=0.5, ge=0, le=1)
    personas: list[str] | None = Field(default=None, max_length=20)
    lean: bool = True
    filing_lag_days: int = Field(default=45, ge=0, le=120)

    def params(self) -> dict:
        common = {"holding_days": self.holding_days, "capital": self.capital, "per_trade": self.per_trade, "cost_bps": self.cost_bps, "data_source": self.data_source}
        extra = {
            "pead": {"earnings_limit": self.earnings_limit},
            "momentum": {"history_days": self.history_days, "top_n": self.top_n, "lookback_days": self.lookback_days, "skip_days": self.skip_days, "near_high_pct": self.near_high_pct},
            "insider": {"history_days": self.history_days, "window_days": self.window_days, "min_insiders": self.min_insiders, "min_value_usd": self.min_value_usd},
            "committee": {"history_days": self.history_days, "top_n": self.top_n, "min_consensus": self.min_consensus, "min_agreement": self.min_agreement,
                          "personas": self.personas, "lean": self.lean, "filing_lag_days": self.filing_lag_days},
        }[self.strategy]
        return {**common, **extra}


#: FD's /prices endpoint returns at most ~100 bars per request
FD_PRICE_CHUNK_DAYS = 90


@contextmanager
def _data_bundle(data_source: str, *, needs_fd: bool, persona_client: bool = False):
    """Price feed per ``data_source``; the FD client only when the run needs it.

    Shared by the backtest and the event study: prices come from yfinance (free)
    or FD (paid, chunked), events / fundamentals always from FD via ``raw``.
    """
    from v2.backtesting.strategies import BacktestData, PriceCache

    raw = None
    if needs_fd or data_source == "fd":
        from v2.data import CachedFDClient

        raw = CachedFDClient()
        raw.__enter__()
    try:
        if data_source == "fd":
            from v2.data.price_source import FDPriceSource

            prices = PriceCache(FDPriceSource(raw), chunk_days=FD_PRICE_CHUNK_DAYS)
        else:
            from v2.data.price_source import YFinancePriceSource

            prices = PriceCache(YFinancePriceSource())
        fd = None
        if raw is not None and persona_client:
            from v2.personas.data import adapt_client

            fd = adapt_client(raw)
        yield BacktestData(prices=prices, fd=fd, raw=raw)
    finally:
        if raw is not None:
            raw.__exit__(None, None, None)


def _backtest_data(body: BacktestInput):
    return _data_bundle(body.data_source, needs_fd=body.strategy != "momentum", persona_client=body.strategy in ("insider", "committee"))


def _fd_bill(data, data_source: str) -> dict[str, int]:
    """Paid requests by endpoint, including FD price chunks when FD served prices."""
    counts = dict(data.fd_requests)
    if data_source == "fd" and data.prices.requests:
        counts["prices"] = counts.get("prices", 0) + data.prices.requests
    return counts


def _build_strategy(body: BacktestInput, on_tick=None):
    from v2.backtesting import CommitteeStrategy, InsiderClusterStrategy, MomentumStrategy, PEADStrategy

    if body.strategy == "pead":
        return PEADStrategy(earnings_limit=body.earnings_limit, holding_days=body.holding_days)
    if body.strategy == "momentum":
        universe_at = None
        if body.universe in INDEX_UNIVERSES:
            from v2.screening.universes import membership_lookup

            universe_at = membership_lookup(body.universe)  # None when no change history is stored
        return MomentumStrategy(lookback_days=body.lookback_days, skip_days=body.skip_days, holding_days=body.holding_days, top_n=body.top_n,
                                history_days=body.history_days, near_high_pct=body.near_high_pct, progress=on_tick, universe_at=universe_at)
    if body.strategy == "insider":
        return InsiderClusterStrategy(window_days=body.window_days, min_insiders=body.min_insiders, min_value_usd=body.min_value_usd,
                                      holding_days=body.holding_days, history_days=body.history_days, progress=on_tick)
    from app.routers.committee import _store  # lazy: committee imports this module

    return CommitteeStrategy(holding_days=body.holding_days, history_days=body.history_days, top_n=body.top_n, min_consensus=body.min_consensus,
                             min_agreement=body.min_agreement, personas=body.personas, lean=body.lean, filing_lag_days=body.filing_lag_days,
                             store=_store(), progress=on_tick)


def _backtest_limit(body: BacktestInput) -> int:
    """Price-only momentum is free, so it may run over a whole index; paid strategies keep the small cap."""
    return BIG_LIMIT if body.strategy == "momentum" else MAX_TICKERS


def _check_backtest_size(body: BacktestInput) -> None:
    """A custom list over the strategy's cap is refused up front instead of silently truncated."""
    if body.universe == "custom":
        n = len(normalize_tickers(body.tickers, limit=BIG_LIMIT))
        limit = _backtest_limit(body)
        if n > limit:
            raise ValueError(f"custom list has {n} tickers; the {body.strategy} strategy accepts at most {limit} (only momentum may exceed {MAX_TICKERS})")


def _price_return(data, ticker: str, start: str, end: str) -> float | None:
    """Close-to-close return of ``ticker`` over ``[start, end]``; None without at least two bars."""
    rows = data.get_prices(ticker, start, end) or []
    closes = []
    for r in rows:
        close = r.get("close") if isinstance(r, dict) else getattr(r, "close", None)
        try:
            close = float(close)
        except (TypeError, ValueError):
            continue
        if close > 0:
            closes.append(close)
    if len(closes) < 2:
        return None
    return closes[-1] / closes[0] - 1


def _benchmark(data, trades, ticker: str = "SPY") -> dict | None:
    """Buy-and-hold return of ``ticker`` from the first entry to the last exit, for comparison."""
    if not trades:
        return None
    start = min(t.entry_date for t in trades)
    end = max(t.exit_date for t in trades)
    total = _price_return(data, ticker, start, end)
    if total is None:
        return None
    days = (datetime.fromisoformat(end) - datetime.fromisoformat(start)).days
    years = days / 365.25
    annualized = (1 + total) ** (1 / years) - 1 if years >= 0.1 else None
    return {"ticker": ticker, "start": start, "end": end, "total_return_pct": round(total, 6),
            "annualized_return_pct": round(annualized, 6) if annualized is not None else None}


def _yearly(data, trades, capital: float, ticker: str = "SPY", deployed_usd: float | None = None) -> list[dict]:
    """Calendar-year rows: strategy return vs. buy-and-hold ``ticker`` over the same span, and the excess.

    With ``deployed_usd`` each row also carries the year's P&L on the deployed
    amount (``return_on_deployed_pct``) and its excess, undoing the dilution /
    leverage of ``per_trade × positions ≠ capital``.
    """
    from v2.backtesting import yearly_breakdown

    rows = yearly_breakdown(trades, capital)
    for row in rows:
        bench = _price_return(data, ticker, row["start"], row["end"])
        row["benchmark_pct"] = round(bench, 6) if bench is not None else None
        row["excess_pct"] = round(row["return_pct"] - bench, 6) if bench is not None and row["return_pct"] is not None else None
        if deployed_usd:
            on_dep = row["pnl"] / deployed_usd
            row["return_on_deployed_pct"] = round(on_dep, 6)
            row["excess_on_deployed_pct"] = round(on_dep - bench, 6) if bench is not None else None
    return rows


def _deployment(trades, capital: float, per_trade: float, benchmark: dict | None) -> dict | None:
    """How much of ``capital`` the strategy actually put to work, and the same run measured on that amount.

    The engine sizes every position at ``per_trade`` and never checks capital, so
    5 positions × $10k on $100k leaves half the money idle (returns on capital are
    diluted) while 30 × $10k is 3× leverage (inflated). ``on_deployed`` restates
    return, annualized return, drawdown and excess over the benchmark on the peak
    deployed amount, which is the fair comparison against a fully invested SPY.
    """
    from v2.backtesting.engine import _period_pnl

    if not trades:
        return None
    per_period: dict[str, int] = {}
    for t in trades:
        per_period[t.entry_date] = per_period.get(t.entry_date, 0) + 1
    positions = max(per_period.values())
    deployed = positions * per_trade
    if deployed <= 0:
        return None
    equity, peak, max_dd = deployed, deployed, 0.0
    for _, pnl in _period_pnl(trades):
        equity += pnl
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak if peak > 0 else 0.0)
    total = (equity - deployed) / deployed
    start, end = min(t.entry_date for t in trades), max(t.exit_date for t in trades)
    years = (datetime.fromisoformat(end) - datetime.fromisoformat(start)).days / 365.25
    annualized = (1 + total) ** (1 / years) - 1 if years >= 0.1 and total > -1 else None
    return {"positions_per_period": positions, "per_trade": per_trade, "deployed_usd": round(deployed, 2), "capital": capital,
            "utilization": round(deployed / capital, 4),
            "on_deployed": {"total_return_pct": round(total, 6), "annualized_return_pct": round(annualized, 6) if annualized is not None else None,
                            "max_drawdown_pct": round(max_dd, 6),
                            "excess_return_pct": round(total - benchmark["total_return_pct"], 6) if benchmark else None}}


def _backtest_total(body: BacktestInput, n_tickers: int) -> int:
    """Progress units: tickers, or tickers × rebalance dates for the committee."""
    if body.strategy != "committee":
        return n_tickers
    from v2.backtesting.strategies import rebalance_dates

    return n_tickers * max(1, len(rebalance_dates(today=datetime.now(timezone.utc).date(), history_days=body.history_days, step_trading_days=body.holding_days)))


def _backtest_universe(body: BacktestInput, holding_steps: list[int] | None = None) -> tuple[list[str], dict]:
    """Tickers to load prices for, plus how the universe was built.

    For an index pool with stored change history the list is the union of the
    constituents on every rebalance date over the history window, so names that
    have since left the index are still ranked in the periods they belonged to
    (prices permitting). The strategy filters per date with ``universe_at``.
    ``holding_steps`` widens the union to several rebalance grids (the sweep).
    """
    tickers, meta = resolve_universe(body.universe, body.tickers, limit=_backtest_limit(body))
    info: dict = {"point_in_time": False, "changes": 0}
    if body.strategy == "momentum" and body.universe in INDEX_UNIVERSES:
        from v2.backtesting.strategies import rebalance_dates
        from v2.screening.universes import load_changes, members_at, membership_mode

        changes = load_changes(body.universe)
        mode = membership_mode(body.universe)
        info.update({"changes": len(changes), "mode": mode})
        if mode != "none":
            today = datetime.now(timezone.utc).date()
            union: dict[str, None] = dict.fromkeys(tickers)
            dates = {today.isoformat()}
            for step in holding_steps or [body.holding_days]:
                dates.update(rebalance_dates(today=today, history_days=body.history_days, step_trading_days=step))
            for d in sorted(dates):
                for t in members_at(body.universe, d)[0]:
                    union.setdefault(t, None)
            tickers = list(union)[:BIG_LIMIT + 100]
            info.update({"point_in_time": True, "history_from": changes[-1]["date"] if changes else None, "tickers_incl_former": len(tickers)})
    return tickers, {**meta, "membership": info}


def _run_backtest(body: BacktestInput, on_tick=None) -> dict:
    from v2.backtesting import BacktestEngine

    _check_backtest_size(body)
    tickers, meta = _backtest_universe(body)
    with _backtest_data(body) as data:
        data.progress = on_tick
        strategy = _build_strategy(body, on_tick)
        result = BacktestEngine(capital=body.capital, per_trade=body.per_trade, cost_bps=body.cost_bps).run(strategy, tickers, data)
        benchmark = _benchmark(data, result.trades)
        deployment = _deployment(result.trades, body.capital, body.per_trade, benchmark)
        yearly = _yearly(data, result.trades, body.capital, deployed_usd=deployment["deployed_usd"] if deployment else None)
        fd_requests = _fd_bill(data, body.data_source)
        notes = {"price_failures": dict(data.prices.failed), "errors": dict(getattr(strategy, "errors", {}) or {}),
                 "rebalance_dates": list(getattr(strategy, "dates", []) or []), "periods": list(getattr(strategy, "periods", []) or []),
                 "aborted": getattr(strategy, "aborted", None), "no_data": list(getattr(strategy, "no_data", []) or []),
                 "membership": meta.get("membership")}
    excess = None
    if benchmark and result.metrics:
        excess = round(result.metrics.total_return_pct - benchmark["total_return_pct"], 6)
    return {"kind": "backtest", "strategy": body.strategy, "data_source": body.data_source, "universe": meta["universe"], "universe_as_of": meta.get("as_of"),
            "tickers": tickers, "params": body.params(), "fd_requests": fd_requests, "fd_cost_usd": fd_cost(fd_requests), "notes": notes,
            "benchmark": benchmark, "excess_return_pct": excess, "yearly": yearly, "deployment": deployment, **result.model_dump()}


# ------------------------------------------------------------------ momentum parameter sweep

class SweepInput(BaseModel):
    """Grid of momentum variants evaluated on one shared price load.

    Every combination sees the same universe (point-in-time when history is
    stored), the same history window and costs; only ``top_n``,
    ``holding_days`` and the 52-week-high filter vary. Each combination is
    fully invested — ``per_trade = capital / top_n`` — so a 30-name variant is
    not silently 3× levered against a 10-name one.
    """

    universe: Universe = "sp500"
    tickers: list[str] = Field(default_factory=list, max_length=BIG_LIMIT)
    data_source: Literal["yfinance", "fd"] = "yfinance"
    history_days: int = Field(default=1825, ge=60, le=3650)
    lookback_days: int = Field(default=252, ge=20, le=504)
    skip_days: int = Field(default=21, ge=0, le=120)
    capital: float = Field(default=100_000, gt=0, le=100_000_000)
    cost_bps: float = Field(default=10, ge=0, le=200)
    top_ns: list[int] = Field(default=[10, 20, 30], min_length=1, max_length=6)
    holding_days_list: list[int] = Field(default=[21, 42, 63], min_length=1, max_length=6)
    #: None = no filter; 0.10 = only names within 10 % of their 52-week high
    near_high_pcts: list[float | None] = Field(default=[None, 0.10], min_length=1, max_length=4)

    @field_validator("top_ns", "holding_days_list")
    @classmethod
    def _positive_grid(cls, values: list[int]) -> list[int]:
        out = sorted({int(v) for v in values})
        if any(v < 1 or v > 252 for v in out):
            raise ValueError("grid values must be between 1 and 252")
        return out

    @field_validator("near_high_pcts")
    @classmethod
    def _near_high_grid(cls, values: list[float | None]) -> list[float | None]:
        out: list[float | None] = []
        for v in values:
            if v is not None and not 0 <= v <= 1:
                raise ValueError("near_high_pct must be between 0 and 1")
            if v not in out:
                out.append(v)
        return out

    def combos(self) -> list[dict]:
        return [{"top_n": n, "holding_days": h, "near_high_pct": nh}
                for nh in self.near_high_pcts for h in self.holding_days_list for n in self.top_ns]

    def as_backtest(self, top_n: int | None = None, holding_days: int | None = None, near_high_pct: float | None = None) -> BacktestInput:
        n = top_n or self.top_ns[0]
        return BacktestInput(universe=self.universe, tickers=self.tickers or ["AAPL"], strategy="momentum", data_source=self.data_source,
                             holding_days=holding_days or self.holding_days_list[0], capital=self.capital, per_trade=self.capital / n, cost_bps=self.cost_bps,
                             history_days=self.history_days, top_n=top_n or self.top_ns[0], lookback_days=self.lookback_days, skip_days=self.skip_days,
                             near_high_pct=near_high_pct)

    def params(self) -> dict:
        return {"history_days": self.history_days, "lookback_days": self.lookback_days, "skip_days": self.skip_days, "capital": self.capital,
                "sizing": "capital / top_n", "cost_bps": self.cost_bps, "data_source": self.data_source,
                "grid": {"top_ns": self.top_ns, "holding_days_list": self.holding_days_list, "near_high_pcts": self.near_high_pcts}}


def _sweep_universe(body: SweepInput) -> tuple[list[str], dict]:
    base = body.as_backtest()
    _check_backtest_size(base)
    return _backtest_universe(base, holding_steps=body.holding_days_list)


def _run_sweep(body: SweepInput, on_tick=None) -> dict:
    """Load every price series once, then run the whole grid against the in-memory cache."""
    from v2.backtesting import BacktestEngine

    tickers, meta = _sweep_universe(body)
    combos = body.combos()
    tick = on_tick or (lambda i: None)
    with _data_bundle(body.data_source, needs_fd=False) as data:
        # Same start MomentumStrategy computes, so the strategies below never fetch again.
        start = data.today - timedelta(days=body.history_days + int(body.lookback_days * 1.6) + 10)
        for i, t in enumerate(tickers):
            tick(i)
            data.prices.warm(t, start)
        data.prices.warm("SPY", start)
        rows = []
        no_data: list[str] = []
        for k, combo in enumerate(combos):
            bt = body.as_backtest(**combo)
            strategy = _build_strategy(bt)
            result = BacktestEngine(capital=bt.capital, per_trade=bt.per_trade, cost_bps=bt.cost_bps).run(strategy, tickers, data)
            benchmark = _benchmark(data, result.trades)
            m = result.metrics
            no_data = list(strategy.no_data)
            row = {**combo, "per_trade": round(bt.per_trade, 2), "n_trades": m.n_trades if m else 0, "n_periods": m.n_periods if m else 0,
                   "total_return_pct": m.total_return_pct if m else None, "annualized_return_pct": m.annualized_return_pct if m else None,
                   "sharpe_ratio": m.sharpe_ratio if m else None, "max_drawdown_pct": m.max_drawdown_pct if m else None,
                   "win_rate": m.win_rate if m else None, "avg_return_pct": m.avg_return_pct if m else None,
                   "benchmark_pct": benchmark["total_return_pct"] if benchmark else None,
                   "excess_return_pct": round(m.total_return_pct - benchmark["total_return_pct"], 6) if m and benchmark else None,
                   "start": benchmark["start"] if benchmark else None, "end": benchmark["end"] if benchmark else None}
            rows.append(row)
            tick(len(tickers) + k + 1)
        fd_requests = _fd_bill(data, body.data_source)
        notes = {"price_failures": dict(data.prices.failed), "membership": meta.get("membership"), "no_data": no_data}
    return {"kind": "sweep", "strategy": "momentum", "data_source": body.data_source, "universe": meta["universe"], "universe_as_of": meta.get("as_of"),
            "tickers": tickers, "params": body.params(), "fd_requests": fd_requests, "fd_cost_usd": fd_cost(fd_requests), "notes": notes, "rows": rows}


class EventStudyInput(BaseModel):
    universe: Universe = "custom"
    tickers: list[str] = Field(default_factory=lambda: ["AAPL", "MSFT", "NVDA"], max_length=MAX_TICKERS)
    #: where daily prices (stock and SPY) come from; earnings history is always Financial Datasets
    data_source: Literal["yfinance", "fd"] = "yfinance"
    earnings_limit: int = Field(default=8, ge=1, le=20)
    n_bootstrap: int = Field(default=2000, ge=100, le=10_000)
    require_eps_surprise: bool = True
    #: one event per (ticker, report period): the 8-K and the later 10-Q/10-K are the same announcement
    dedupe: bool = True
    #: "surprise" → ALL / BEAT / MISS / MEET; "reaction" → terciles of the 2-day reaction; "source" → by filing type
    group_by: Literal["surprise", "reaction", "source"] = "surprise"


def _run_event_study(body: EventStudyInput) -> dict:
    from v2.event_study import compute_car

    tickers, meta = resolve_universe(body.universe, body.tickers, limit=20)
    # The bundle answers get_prices (chosen feed) and get_earnings_history (FD, counted), which is all compute_car reads.
    with _data_bundle(body.data_source, needs_fd=True) as data:
        result = compute_car(
            tickers,
            data,
            earnings_limit=body.earnings_limit,
            n_bootstrap=body.n_bootstrap,
            require_eps_surprise=body.require_eps_surprise,
            dedupe=body.dedupe,
            group_by=body.group_by,
        )
        fd_requests = _fd_bill(data, body.data_source)
        price_failures = dict(data.prices.failed)
    return {"kind": "event_study", "universe": meta["universe"], "tickers": tickers, "data_source": body.data_source,
            "params": {"earnings_limit": body.earnings_limit, "n_bootstrap": body.n_bootstrap, "require_eps_surprise": body.require_eps_surprise,
                       "data_source": body.data_source, "dedupe": body.dedupe, "group_by": body.group_by},
            "fd_requests": fd_requests, "fd_cost_usd": fd_cost(fd_requests), "price_failures": price_failures, **result.model_dump()}


class ScreeningInput(BaseModel):
    universe: Universe = "tech30"
    tickers: list[str] = Field(default_factory=list, max_length=MAX_TICKERS)
    #: where the four screening inputs come from. yfinance is free and covers
    #: market cap / revenue growth / gross margin; FD bills per request.
    data_source: Literal["yfinance", "fd"] = "yfinance"
    #: enrich candidates with Wall-Street earnings estimates (one FD request per candidate)
    with_earnings: bool = False
    #: pick-and-mix criteria; omitted → DEFAULT_RULES (the old five-threshold screen minus the cap ceiling)
    rules: list[Rule] | None = Field(default=None, max_length=24)
    # legacy thresholds, still accepted; folded into rules when `rules` is omitted
    market_cap_min: float | None = Field(default=None, ge=0)
    market_cap_max: float | None = Field(default=None, gt=0)
    revenue_growth_min: float | None = Field(default=None, ge=-1, le=10)
    gross_margin_min: float | None = Field(default=None, ge=-1, le=1)
    volatility_max: float | None = Field(default=None, gt=0, le=10)

    def effective_rules(self) -> list[Rule]:
        if self.rules is not None:
            unknown = [r.field for r in self.rules if r.field not in CRITERIA]
            if unknown:
                raise ValueError(f"unknown screening field: {', '.join(unknown)}")
            return list(self.rules)
        legacy = [("market_cap", "gte", self.market_cap_min), ("market_cap", "lte", self.market_cap_max), ("revenue_growth", "gte", self.revenue_growth_min),
                  ("gross_margin", "gte", self.gross_margin_min), ("volatility", "lte", self.volatility_max)]
        picked = [Rule(field=f, op=o, value=v) for f, o, v in legacy if v is not None]
        return picked or list(DEFAULT_RULES)


class _ScreenData:
    """Metrics from one client, earnings (optional) from another, both tolerant.

    Wraps the screener's data dependency so that a ticker the provider does not
    cover is skipped instead of aborting the run, and so that FD requests can be
    counted for the cost line. ``metrics_is_fd`` says whether metrics calls bill.
    """

    def __init__(self, metrics_client, *, metrics_is_fd: bool, earnings_client=None):
        self._metrics = metrics_client
        self._earnings = earnings_client
        self._metrics_is_fd = metrics_is_fd
        self.skipped: dict[str, str] = {}
        self.fd_requests: dict[str, int] = {}

    def _count(self, endpoint: str) -> None:
        self.fd_requests[endpoint] = self.fd_requests.get(endpoint, 0) + 1

    def get_financial_metrics(self, ticker, end_date, limit=1, **kwargs):
        if self._metrics_is_fd:
            self._count("financial_metrics")
        try:
            return self._metrics.get_financial_metrics(ticker, end_date, limit=limit, **kwargs)
        except Exception as exc:  # noqa: BLE001 — provider miss for one ticker
            self.skipped.setdefault(str(ticker), f"metrics: {type(exc).__name__}: {str(exc)[:120]}")
            return []

    def get_earnings(self, ticker):
        if self._earnings is None:
            return None
        self._count("earnings")
        try:
            return self._earnings.get_earnings(ticker)
        except Exception as exc:  # noqa: BLE001
            self.skipped.setdefault(str(ticker), f"earnings: {type(exc).__name__}: {str(exc)[:120]}")
            return None

    def __getattr__(self, name):  # anything else (misses, stats, ...) comes from the metrics client
        return getattr(self._metrics, name)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        for client in (self._metrics, self._earnings):
            close = getattr(client, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # noqa: BLE001
                    pass
        return False


def _screen_clients(body: "ScreeningInput") -> _ScreenData:
    """yfinance (free) or FD for metrics; FD for earnings only when asked."""
    from v2.data import CachedFDClient

    fd = CachedFDClient() if (body.data_source == "fd" or body.with_earnings) else None
    if body.data_source == "yfinance":
        try:
            from v2.data.yfinance_client import YFinanceClient
        except Exception as exc:  # noqa: BLE001 — fall back to FD rather than fail the screen
            raise RuntimeError(f"yfinance client unavailable ({type(exc).__name__}); choose data_source=fd") from exc
        return _ScreenData(YFinanceClient(), metrics_is_fd=False, earnings_client=fd if body.with_earnings else None)
    return _ScreenData(fd, metrics_is_fd=True, earnings_client=fd if body.with_earnings else None)


def _run_screening(body: ScreeningInput, on_tick=None) -> dict:
    from v2.data.price_source import default_price_source

    tickers, meta = resolve_universe(body.universe, body.tickers, limit=BIG_LIMIT)
    rules = body.effective_rules()
    with _screen_clients(body) as client:
        result = lab_screen(tickers, client, default_price_source(), rules, on_tick=on_tick, with_earnings=body.with_earnings)
        skipped, fd_requests = dict(client.skipped), dict(client.fd_requests)
    return {"kind": "screening", "universe": meta["universe"], "universe_as_of": meta.get("as_of"), "tickers": tickers,
            "data_source": body.data_source, "with_earnings": body.with_earnings,
            "fd_requests": fd_requests, "fd_cost_usd": fd_cost(fd_requests), "skipped": skipped, **result}


#: screens bigger than this run as a background job (nginx cuts requests at 90 s)
SCREEN_SYNC_MAX = 40
#: backtests with more progress units than this run as a background job
BACKTEST_SYNC_MAX = 30
_JOBS: dict[str, dict] = {}
_JOBS_LOCK = threading.Lock()


def _run_job(job_id: str, kind: str, body, fn) -> None:
    def tick(i: int) -> None:
        with _JOBS_LOCK:
            _JOBS[job_id]["done"] = i

    try:
        result = fn(body, on_tick=tick)
        _remember_run(kind, result, body.model_dump())
        with _JOBS_LOCK:
            _JOBS[job_id].update({"status": "completed", "done": _JOBS[job_id]["total"], "result": result})
    except Exception as exc:  # noqa: BLE001
        with _JOBS_LOCK:
            _JOBS[job_id].update({"status": "failed", "error": f"{type(exc).__name__}: {str(exc)[:200]}"})


def _running_job(kind: str) -> dict | None:
    """The job of this kind still running, if any — heavy jobs run one at a time on a small box."""
    with _JOBS_LOCK:
        for job in _JOBS.values():
            if job["kind"] == f"{kind}_job" and job["status"] == "running":
                return dict(job)
    return None


def _refuse_if_busy(kind: str) -> None:
    job = _running_job(kind)
    if job:
        raise HTTPException(status_code=409, detail=f"已有一个回测任务在运行（{job['done']} / {job['total']}），等它结束再启动新的；同时跑两个会把内存用完。")


def _start_job(kind: str, body, total: int, fn) -> dict:
    job_id = uuid.uuid4().hex[:12]
    with _JOBS_LOCK:
        if len(_JOBS) > 50:  # keep the in-memory table small
            for old in sorted(_JOBS, key=lambda k: _JOBS[k]["started_at"])[:25]:
                _JOBS.pop(old, None)
        _JOBS[job_id] = {"job_id": job_id, "kind": f"{kind}_job", "status": "running", "done": 0, "total": total,
                         "universe": body.universe, "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    ContextThread(target=_run_job, args=(job_id, kind, body, fn), name=f"{kind}-{job_id}", daemon=True).start()
    return _job_view(job_id)


def _job_view(job_id: str) -> dict:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        return dict(job) if job else {}


async def _lab_call(fn, body) -> dict:
    try:
        return await asyncio.wait_for(run_in_threadpool(fn, body), timeout=_LAB_TIMEOUT_SECONDS)
    except asyncio.TimeoutError as exc:
        raise HTTPException(status_code=504, detail="lab run timed out") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/lab/backtest")
async def run_backtest(body: BacktestInput, background: bool | None = None) -> dict:
    """Quick runs answer inline; the committee strategy (or ?background=true) returns a job to poll."""
    try:
        _check_backtest_size(body)
        tickers, _ = await run_in_threadpool(_backtest_universe, body)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    total = _backtest_total(body, len(tickers))
    if background or (background is None and (body.strategy == "committee" or total > BACKTEST_SYNC_MAX)):
        _refuse_if_busy("backtest")
        return _start_job("backtest", body, total, _run_backtest)
    result = await _lab_call(_run_backtest, body)
    _remember_run("backtest", result, body.model_dump())
    return result


@router.post("/lab/backtest/sweep")
async def run_backtest_sweep(body: SweepInput) -> dict:
    """Momentum parameter grid on one price load; always a job (poll /lab/backtest/jobs/{id})."""
    try:
        tickers, _ = await run_in_threadpool(_sweep_universe, body)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _refuse_if_busy("backtest")
    return _start_job("backtest", body, len(tickers) + len(body.combos()), _run_sweep)


@router.get("/lab/backtest/jobs/{job_id}")
async def backtest_job(job_id: str) -> dict:
    job = _job_view(job_id)
    if not job or job.get("kind") != "backtest_job":
        raise HTTPException(status_code=404, detail="backtest job not found")
    return job


@router.post("/lab/event-study")
async def run_event_study(body: EventStudyInput) -> dict:
    result = await _lab_call(_run_event_study, body)
    _remember_run("event_study", result, body.model_dump())
    return result


@router.post("/lab/screening")
async def run_screening(body: ScreeningInput, background: bool | None = None) -> dict:
    """Small universes answer inline; large ones (or ?background=true) return a job to poll."""
    try:
        tickers, _ = await run_in_threadpool(resolve_universe, body.universe, body.tickers, limit=BIG_LIMIT)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if background or (background is None and len(tickers) > SCREEN_SYNC_MAX):
        return _start_job("screening", body, len(tickers), _run_screening)
    result = await _lab_call(_run_screening, body)
    _remember_run("screening", result, body.model_dump())
    return result


@router.get("/lab/screening/jobs/{job_id}")
async def screening_job(job_id: str) -> dict:
    job = _job_view(job_id)
    if not job or job.get("kind") != "screening_job":
        raise HTTPException(status_code=404, detail="screening job not found")
    return job


@router.get("/lab/screening/criteria")
async def screening_criteria() -> dict:
    """Fields a screening rule may use, with labels / units, and the default rule set."""
    return {"kind": "criteria", "items": CRITERIA, "defaults": [r.model_dump() for r in DEFAULT_RULES]}


@router.get("/lab/universes")
async def universes() -> dict:
    """Named universes with sizes and snapshot dates (index lists refreshable on the VPS)."""
    from v2.screening.universe import TECH_30
    from v2.screening.universes import universe_status

    items = {"tech30": {"size": len(TECH_30), "as_of": None, "label": "TECH_30 监控池"}}
    items.update(universe_status())
    return {"kind": "universes", "items": items}


@router.get("/lab/signals")
async def signal_candidates() -> dict:
    """Current deterministic thresholds behind production anomaly signals."""
    from v2.monitoring.models import MonitorConfig

    monitor = MonitorConfig()
    return {
        "kind": "signals",
        "monitoring": monitor.model_dump(),
        "intraday": {
            "price_pct_threshold": 0.03,
            "volume_pace_threshold": 2.5,
            "sector_gap_pp": 0.015,
            "cooldown_minutes": 30,
        },
        "note": "read-only snapshot; changing Lab UI does not mutate production thresholds",
    }


@router.get("/lab/runs")
async def lab_runs(limit: int = Query(50, ge=1, le=200), kind: str | None = None) -> dict:
    return {"kind": "runs", "items": _lab_store().list(limit=limit, kind=kind), "counts": _lab_store().counts()}


@router.get("/lab/runs/{run_id}")
async def lab_run_detail(run_id: str) -> dict:
    row = _lab_store().get(run_id)
    if row is None:
        raise HTTPException(status_code=404, detail="lab run not found")
    return row


@router.delete("/lab/runs/{run_id}")
async def lab_run_delete(run_id: str) -> dict:
    if not _lab_store().delete(run_id):
        raise HTTPException(status_code=404, detail="lab run not found")
    return {"deleted": run_id, "counts": _lab_store().counts()}


class CleanupInput(BaseModel):
    older_than_days: int = Field(default=30, ge=1, le=3650)


@router.post("/lab/runs/cleanup")
async def lab_runs_cleanup(body: CleanupInput | None = None) -> dict:
    """Drop every run older than ``older_than_days`` — results carry full trade lists, so the log grows fast."""
    days = (body or CleanupInput()).older_than_days
    deleted = _lab_store().delete_older_than(days)
    return {"deleted": deleted, "older_than_days": days, "counts": _lab_store().counts()}
