"""End-to-end orchestration for lateral expansion (玩法 ③).

Flow:
    1. LLM discovers raw neighbor candidates for each seed (4 categories).
    2. Aggregate by ticker — same ticker may appear under multiple seeds/categories.
    3. Verify each unique ticker exists via FD company_facts.
    4. For real + new tickers, build a ScreenCandidate and run the screening filter.
    5. For filter-passers, generate bull/bear narration (reusing screening.narrate).
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

from v2.data.client import FDClient
from v2.data.price_source import PriceSource, default_price_source
from v2.data.yfinance_client import KNOWN_ADRS, YFinanceClient
from v2.lateral.discover import discover
from v2.lateral.models import LateralResult, Neighbor
from v2.lateral.verify import verify, verify_relation
from v2.screening import (
    FilterConfig,
    ScreenCandidate,
    build_candidate,
    narrate,
    passes_filter,
)

logger = logging.getLogger(__name__)


class _CountedProvider:
    """Count logical provider method attempts, including failed calls.

    This is not a count of underlying HTTP retries or billable requests.
    """
    def __init__(self, provider, counts, name):
        self.provider, self.counts, self.name = provider, counts, name

    def __getattr__(self, name):
        method = getattr(self.provider, name)
        if not callable(method) or not name.startswith('get_'):
            return method
        def counted(*args, **kwargs):
            key = self.name + '.' + name
            self.counts[key] = self.counts.get(key, 0) + 1
            return method(*args, **kwargs)
        return counted


def run_lateral_expansion(
    seeds: list[str],
    universe: set[str],
    fd_client: FDClient,
    filter_config: FilterConfig,
    *,
    price_source: PriceSource | None = None,
) -> LateralResult:
    """Run the full pipeline for one expansion pass.

    ``price_source`` provides daily OHLCV for the candidate build pass
    (Phase 4.5-mini default: yfinance real-time EOD). The non-price
    FD endpoints inside :func:`v2.screening.build_candidate` —
    ``get_financial_metrics`` / ``get_earnings`` — return quarterly
    data with their own 45-90 day reporting lag, so a 3-day buffer is
    moot there and ``date.today()`` is correct.
    """
    if price_source is None:
        price_source = default_price_source()

    today = date.today()
    today_str = today.isoformat()
    history_start = (today - timedelta(days=400)).isoformat()
    counts, warnings, errors = {}, [], []
    fd_client = _CountedProvider(fd_client, counts, 'Financial Datasets')
    price_source = _CountedProvider(price_source, counts, 'Prices')

    def record_failure(neighbor, stage, exc):
        # SDK exception text can contain request metadata; never expose it.
        kind = str(getattr(exc, 'error_type', type(exc).__name__))
        errors.append({'ticker': neighbor.ticker, 'stage': stage, 'type': kind})
        neighbor.failed_reason = f'{stage}失败（{kind}）'
        warnings.append(f'{neighbor.ticker}：{stage}失败（{kind}）；其他候选继续处理。')

    # Step 1: LLM discovery
    logger.info("Discovering neighbors for %d seeds...", len(seeds))
    pairs, tokens = discover(seeds)
    if not pairs:
        warnings.append('候选生成未返回有效结果或生成失败；不表示该公司没有产业关系。')

    # Step 2: Aggregate by ticker — collect all (seed, category) labels
    by_ticker: dict[str, Neighbor] = {}
    for ticker, label in pairs:
        if ticker not in by_ticker:
            by_ticker[ticker] = Neighbor(ticker=ticker)
        by_ticker[ticker].labels.append(label)
    neighbors = list(by_ticker.values())
    logger.info("Aggregated %d unique tickers from %d raw suggestions",
                len(neighbors), len(pairs))

    # Step 3: Verify existence
    for n in neighbors:
        try:
            verify(n, fd_client, universe)
        except Exception as exc:
            record_failure(n, '公司身份核查', exc)

    # Step 3.5 (Resume polish): Tavily-verify the seed-neighbor relationship.
    # Check confirmed companies, including those already in our universe.
    # Each relationship label may require its own search.
    tavily_calls = 0
    for n in neighbors:
        if n.exists:
            try:
                tavily_calls += verify_relation(n)
            except Exception as exc:
                record_failure(n, '关系搜索', exc)
    logger.info("Tavily relation checks: %d calls, %d verified",
                tavily_calls,
                sum(1 for n in neighbors if n.relation_verified))

    # Step 4: Hard-filter the real + new tickers
    new_real = [n for n in neighbors if n.exists and not n.already_in_universe]
    logger.info("%d new real candidates to screen", len(new_real))

    yf_client = _CountedProvider(YFinanceClient(), counts, 'Yahoo fallback')

    for n in new_real:
        use_fallback = yf_client if n.ticker in KNOWN_ADRS else None
        try:
            candidate = build_candidate(
                n.ticker, fd_client, today_str, history_start,
                fallback=use_fallback,
                price_source=price_source,
            )
        except Exception as exc:
            record_failure(n, '财务筛选', exc)
            continue  # Keep exists, labels and search evidence already collected.
        if candidate is None:
            n.failed_reason = "数据不足"
            warnings.append(f'{n.ticker}：财务筛选数据不足；已发现的关系保留。')
            continue
        n.candidate = candidate
        try:
            if passes_filter(candidate, filter_config):
                n.passed_filter = True
            else:
                n.failed_reason = _explain_failure(candidate, filter_config)
        except Exception as exc:
            record_failure(n, '财务筛选', exc)

    # Step 5: Narrate passers (reuse screening narrator)
    passers = [n for n in new_real if n.passed_filter and n.candidate is not None]
    # Emit a single transform/filter summary so the dashboard's "筛选" pill
    # has an event to highlight (the per-neighbor verify_relation + passes_filter
    # checks themselves are Python-only loops with no individual emit).
    from v2.observability import emit as _emit
    _emit(
        "transform", op="filter",
        candidates=len(new_real),
        passed=len(passers),
        filtered_out=len(new_real) - len(passers),
    )
    if passers:
        logger.info("Narrating %d passers with DeepSeek...", len(passers))
        try:
            narrations, narr_tokens = narrate([n.candidate for n in passers])
        except Exception:
            narrations, narr_tokens = {}, 0
            warnings.append('多空解读生成失败；已发现的关系保留。')
        tokens += narr_tokens
        for n in passers:
            note = narrations.get(n.ticker, {})
            n.bull = note.get("bull", "") or ""
            n.bear = note.get("bear", "") or ""

    return LateralResult(
        date=today_str,
        seeds=seeds,
        neighbors=neighbors,
        llm_tokens=tokens,
        api_calls=sum(counts.values()),
        tavily_calls=tavily_calls,
        warnings=warnings,
        candidate_errors=errors,
        api_call_counts=counts,
    )


def _explain_failure(c: ScreenCandidate, cfg: FilterConfig) -> str:
    """Pinpoint which filter rule the candidate failed (first failure wins)."""
    if c.market_cap is None:
        return "市值数据缺失"
    if c.market_cap < cfg.market_cap_min:
        return f"市值 ${c.market_cap / 1e9:.1f}B < ${cfg.market_cap_min / 1e9:.0f}B"
    if c.market_cap > cfg.market_cap_max:
        return f"市值 ${c.market_cap / 1e12:.1f}T > ${cfg.market_cap_max / 1e12:.0f}T"
    if c.revenue_growth is None:
        return "营收数据缺失"
    if c.revenue_growth < cfg.revenue_growth_min:
        return f"营收 {c.revenue_growth:+.1%} < {cfg.revenue_growth_min:.1%}"
    if c.gross_margin is None:
        return "毛利数据缺失"
    if c.gross_margin < cfg.gross_margin_min:
        return f"毛利 {c.gross_margin:.1%} < {cfg.gross_margin_min:.1%}"
    if c.volatility is None:
        return "波动率数据缺失"
    if c.volatility > cfg.volatility_max:
        return f"波动 {c.volatility:.1%} > {cfg.volatility_max:.1%}"
    return "未通过"
