"""Screening orchestration: universe -> data -> filter -> candidates."""

from __future__ import annotations

import logging
from datetime import date, timedelta

import numpy as np

from v2.data.client import FDClient
from v2.data.price_source import PriceSource, default_price_source
from v2.screening.delta_fetcher import enrich_with_delta
from v2.screening.filters import passes_filter
from v2.screening.models import FilterConfig, ScreenCandidate, ScreenResult
from v2.universe import SECTOR_ETFS, sector_etf_for

logger = logging.getLogger(__name__)

_MIN_PRICE_HISTORY = 60  # trading days needed for a meaningful vol estimate
_TRADING_DAYS_PER_YEAR = 252


def run_screening(
    tickers: list[str],
    fd_client: FDClient,
    config: FilterConfig,
    *,
    price_source: PriceSource | None = None,
) -> ScreenResult:
    """Scan *tickers*, fetch metrics + prices, return candidates that pass.

    ``price_source`` is the daily-OHLCV source (Phase 4.5-mini); defaults
    to :func:`v2.data.price_source.default_price_source` which returns
    a real-time EOD yfinance client. Tests / backtest inject an
    :class:`FDPriceSource` to reproduce historical snapshots.
    """
    if price_source is None:
        price_source = default_price_source()

    today = date.today()
    today_str = today.isoformat()
    # Pull ~400 calendar days to ensure we get >=252 trading days
    history_start = (today - timedelta(days=400)).isoformat()

    candidates: list[ScreenCandidate] = []
    tavily_calls = 0

    # If wrapped with CachedFDClient, query its real-miss counter at start/end so
    # the displayed fd_calls reflects actual API hits, not invocation counts.
    initial_misses = getattr(fd_client, "misses", None)
    invocation_count = 0  # used when there's no cache to query

    for ticker in tickers:
        candidate = build_candidate(
            ticker, fd_client, today_str, history_start,
            price_source=price_source,
        )
        invocation_count += 2
        if candidate is None:
            continue
        if passes_filter(candidate, config):
            candidates.append(candidate)

    # Optimization: only fetch earnings for tickers that passed the filter.
    # Cuts earnings FD calls from N (universe size) to len(candidates) — typically
    # a 60-70% saving.
    for c in candidates:
        enrich_with_earnings(c, fd_client)
        invocation_count += 1

    # 改进 ②: tags computation happens after earnings enrichment, so the
    # "业绩超预期" tag can fire when surprise_pct is known.
    for c in candidates:
        c.tags = _compute_tags(c)

    # ETF benchmarking — populate sector relative-strength context for
    # the narrator's template fill. Bounded extra cost: |SECTOR_ETFS| price
    # calls (through the injected price_source).
    _enrich_with_sector_strength(
        candidates, today_str, history_start,
        price_source=price_source,
    )

    # 改进 ①: Enrich passing candidates with dynamic delta context for narrator
    if candidates:
        tavily_calls = enrich_with_delta(candidates)

    # Real FD calls = cache misses (if cached client). Otherwise fall back to
    # invocation count.
    if initial_misses is not None:
        fd_calls = int(fd_client.misses) - int(initial_misses)
    else:
        fd_calls = invocation_count

    return ScreenResult(
        date=today_str,
        universe_size=len(tickers),
        candidates=candidates,
        fd_calls=fd_calls,
        tavily_calls=tavily_calls,
        api_calls=fd_calls + tavily_calls,   # legacy aggregate
    )


def build_candidate(
    ticker: str,
    fd: FDClient,
    today: str,
    history_start: str,
    *,
    fallback=None,
    price_source: PriceSource | None = None,
) -> ScreenCandidate | None:
    """Fetch metrics + prices + earnings and build a snapshot, or None if data unusable.

    ``price_source`` is the daily-OHLCV provider; defaults to
    :func:`default_price_source` (yfinance). ``fd`` still handles
    metrics + (in :func:`enrich_with_earnings`) earnings. The
    ``fallback`` argument is the metrics-side fallback (e.g.
    :class:`YFinanceClient` for foreign ADRs FD doesn't cover) — it
    falls back on metrics only; prices already come from the
    injected ``price_source``.
    """
    if price_source is None:
        price_source = default_price_source()

    metrics_list = fd.get_financial_metrics(ticker, today, limit=1)
    prices = price_source.get_prices(ticker, history_start, today)

    # Fallback path — try the secondary source for metrics-side gap.
    # Prices come from the injected price_source so we don't repeat the
    # fallback round-trip here.
    if fallback is not None:
        if not metrics_list:
            metrics_list = fallback.get_financial_metrics(ticker, today, limit=1)

    if not metrics_list or len(prices) < _MIN_PRICE_HISTORY:
        return None

    m = metrics_list[0]

    closes = np.array([p.close for p in prices], dtype=float)
    latest = float(closes[-1])
    prev = float(closes[-2])
    price_change = (latest - prev) / prev if prev > 0 else None

    # 1-week return (5 trading days back) — for 改进 ① peer-diff signal
    return_1w: float | None = None
    if len(closes) >= 6 and closes[-6] > 0:
        return_1w = float((closes[-1] - closes[-6]) / closes[-6])

    # 52-week high — for the "市值突破" tag (近期创新高)
    last_252_closes = closes[-_MIN_PRICE_HISTORY:]
    high_52w = float(last_252_closes.max())

    # Annualized vol of daily log returns (winsorized).
    log_returns = np.diff(np.log(closes))
    log_returns = np.clip(log_returns, -0.25, 0.25)
    volatility = float(log_returns.std(ddof=1) * np.sqrt(_TRADING_DAYS_PER_YEAR))

    # Note: earnings are intentionally NOT fetched here — see enrich_with_earnings()
    # below. We only want to spend FD calls on tickers that pass the basic filter.

    return ScreenCandidate(
        ticker=ticker,
        price=latest,
        price_change=price_change,
        market_cap=m.market_cap,
        revenue_growth=m.revenue_growth,
        gross_margin=m.gross_margin,
        volatility=volatility,
        high_52w=high_52w,
        return_1w=return_1w,
    )


def enrich_with_earnings(candidate: ScreenCandidate, fd: FDClient) -> None:
    """In-place: fetch Wall-Street earnings estimate vs actual (改进 ④).

    Called only for candidates that already passed the basic filter, so we
    don't waste an FD call on tickers we'll discard anyway.
    """
    try:
        earnings = fd.get_earnings(candidate.ticker)
    except Exception:
        return

    if not earnings or not earnings.quarterly:
        return

    q = earnings.quarterly
    if q.revenue and q.estimated_revenue and q.estimated_revenue > 0:
        candidate.revenue_actual = float(q.revenue)
        candidate.revenue_estimate = float(q.estimated_revenue)
        candidate.revenue_surprise_pct = (
            candidate.revenue_actual - candidate.revenue_estimate
        ) / candidate.revenue_estimate

    if q.earnings_per_share is not None and q.estimated_earnings_per_share:
        est = float(q.estimated_earnings_per_share)
        if est != 0:
            candidate.eps_actual = float(q.earnings_per_share)
            candidate.eps_estimate = est
            candidate.eps_surprise_pct = (candidate.eps_actual - est) / abs(est)


def _enrich_with_sector_strength(
    candidates: list[ScreenCandidate],
    today: str,
    history_start: str,
    *,
    price_source: PriceSource,
) -> None:
    """Populate sector_etf / sector_return_1w / sector_diff_1w_pp in place.

    Fetches each unique sector ETF's price series exactly once, then computes
    a 1-week return for each candidate's sector benchmark.
    """
    needed_etfs = {sector_etf_for(c.ticker) for c in candidates}
    etf_1w_returns: dict[str, float] = {}

    for etf in needed_etfs:
        try:
            prices = price_source.get_prices(etf, history_start, today)
        except Exception as exc:
            logger.warning("Sector ETF %s fetch failed: %s", etf, exc)
            continue
        if not prices or len(prices) < 6:
            continue
        last = prices[-1].close
        base = prices[-6].close
        if base > 0:
            etf_1w_returns[etf] = (last - base) / base

    for c in candidates:
        etf = sector_etf_for(c.ticker)
        c.sector_etf = etf
        sec_ret = etf_1w_returns.get(etf)
        c.sector_return_1w = sec_ret
        if sec_ret is not None and c.return_1w is not None:
            c.sector_diff_1w_pp = c.return_1w - sec_ret


# ---------------------------------------------------------------------------
# 改进 ②: Hard tag computation
# ---------------------------------------------------------------------------

def _compute_tags(c: ScreenCandidate) -> list[str]:
    """Translate raw metrics into human-readable hard tags for the card header.

    Tags help the reader instantly grok *why* this ticker made it through the
    screen — no need to mentally cross-reference the metrics line below.
    """
    tags: list[str] = []

    # Growth
    if c.revenue_growth is not None and c.revenue_growth >= 0.15:
        tags.append("高成长")

    # Margin (tiered)
    if c.gross_margin is not None:
        if c.gross_margin >= 0.80:
            tags.append("超高毛利")
        elif c.gross_margin >= 0.70:
            tags.append("高毛利")

    # Size
    if c.market_cap is not None:
        if c.market_cap < 50e9:
            tags.append("小盘")
        elif c.market_cap < 500e9:
            tags.append("中盘")
        elif c.market_cap >= 1e12:
            tags.append("超大盘")

    # Volatility extremes
    if c.volatility is not None:
        if c.volatility >= 0.50:
            tags.append("高波动")
        elif c.volatility <= 0.30:
            tags.append("低波动")

    # Price near 52-week high → momentum / market-cap breakout proxy
    if c.high_52w and c.high_52w > 0 and c.price / c.high_52w >= 0.95:
        tags.append("市值突破")

    # Earnings surprise (改进 ④ source)
    rev_beat = (c.revenue_surprise_pct or 0) >= 0.05
    eps_beat = (c.eps_surprise_pct or 0) >= 0.05
    if rev_beat or eps_beat:
        tags.append("业绩超预期")

    return tags
