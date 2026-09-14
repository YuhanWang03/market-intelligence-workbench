"""Shared-data, deterministic stock research services and aggregator.

The engine intentionally has no dependency on chat, prompts, or an LLM.  Every
score is produced from documented metric rules; narrative text is a templated
explanation of those results.
"""

from __future__ import annotations

import math
import re
import time
from concurrent.futures import as_completed
from v2.usage_context import ContextExecutor as ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from statistics import median
from typing import Any, Callable

from v2.data import CachedFDClient
from v2.data.price_source import default_price_source
from v2.research.cache import CACHE_POLICY, ResearchCache
from v2.research.depth import EXPECTATIONS_CAPABILITY, classify_catalysts, sanitize_error
from v2.research.intelligence import build_intelligence
from v2.research.provider_health import ProviderHealthService
from v2.research.services import ResearchServices
from v2.research.store import ENGINE_VERSION, FEATURE_FREEZE

Progress = Callable[[str, str], None]
MODULES = ("fundamental", "valuation", "earnings", "expectations", "institutional", "fund_flow", "technical", "catalyst", "sec", "macro", "supply_chain", "risk")
MODULE_DEPENDENCIES = {
    "expectations": ("earnings",),
    "catalyst": ("earnings", "sec", "macro"),
    "risk": ("fundamental", "valuation", "sec", "institutional", "macro", "supply_chain", "catalyst"),
    "aggregate": MODULES,
}


def resolve_modules(requested: list[str] | tuple[str, ...] | None) -> list[str]:
    wanted = set(MODULES if not requested else requested)
    invalid = wanted.difference(MODULES)
    if invalid:
        raise ValueError(f"unknown research module: {', '.join(sorted(invalid))}")
    queue = list(wanted)
    while queue:
        current = queue.pop()
        for dependency in MODULE_DEPENDENCIES.get(current, ()):
            if dependency != "aggregate" and dependency not in wanted:
                wanted.add(dependency)
                queue.append(dependency)
    return [name for name in MODULES if name in wanted]


def normalize_ticker(raw: str) -> str:
    ticker = raw.strip().upper()
    if not re.fullmatch(r"[A-Z][A-Z0-9.-]{0,7}", ticker):
        raise ValueError("invalid ticker")
    return ticker


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _clamp(value: float) -> int:
    return round(max(0.0, min(100.0, value)))


def _ratio_score(value: float | None, low: float, high: float) -> float | None:
    if value is None:
        return None
    if high == low:
        return 50.0
    return max(0.0, min(100.0, (value - low) / (high - low) * 100.0))


def _weighted(parts: list[tuple[float | None, float]]) -> int | None:
    available = [(score, weight) for score, weight in parts if score is not None]
    if not available:
        return None
    return _clamp(sum(score * weight for score, weight in available) / sum(weight for _, weight in available))


def _pct(value: float | None) -> str:
    return "N/A" if value is None else f"{value * 100:+.1f}%"


def _money(value: float | None) -> str:
    if value is None:
        return "N/A"
    sign = "-" if value < 0 else ""
    value = abs(value)
    for unit, divisor in (("T", 1e12), ("B", 1e9), ("M", 1e6)):
        if value >= divisor:
            return f"{sign}${value / divisor:.2f}{unit}"
    return f"{sign}${value:,.0f}"


def _cmf(highs: list[float], lows: list[float], closes: list[float], volumes: list[float], window: int = 20) -> float | None:
    """Dependency-free Chaikin Money Flow for the research service."""
    if len(closes) < window:
        return None
    flow = 0.0
    total_volume = 0.0
    for high, low, close, volume in zip(highs[-window:], lows[-window:], closes[-window:], volumes[-window:]):
        spread = high - low
        multiplier = 0.0 if spread <= 0 else ((close - low) - (high - close)) / spread
        flow += multiplier * volume
        total_volume += volume
    return flow / total_volume if total_volume > 0 else None


def _source(source_id: str, provider: str, kind: str, title: str, url: str | None = None, published_at: str | None = None) -> dict:
    return {"id": source_id, "provider": provider, "type": kind, "title": title, "url": url, "published_at": published_at, "fetched_at": _now()}


@dataclass
class StockResearchDataset:
    ticker: str
    company: Any = None
    metrics: list[Any] = field(default_factory=list)
    earnings: list[Any] = field(default_factory=list)
    insiders: list[Any] = field(default_factory=list)
    news: list[Any] = field(default_factory=list)
    prices: list[Any] = field(default_factory=list)
    upcoming_earnings: Any = None
    expectations: dict = field(default_factory=dict)
    filings: list[dict] = field(default_factory=list)
    sec_depth: dict = field(default_factory=dict)
    institutional: dict = field(default_factory=dict)
    macro: dict = field(default_factory=dict)
    supply_chain_raw: dict = field(default_factory=dict)
    sources: list[dict] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)
    provider_error_types: dict[str, str] = field(default_factory=dict)
    fetched_at: str = field(default_factory=_now)


class ResearchEngine:
    def __init__(self, fd: Any = None, prices: Any = None, cache: ResearchCache | None = None, services: ResearchServices | None = None) -> None:
        self.fd = fd or CachedFDClient()
        self.prices = prices or default_price_source()
        self.cache = cache or ResearchCache()
        self.services = services or ResearchServices.defaults()
        if getattr(self.services, "news", None) is None:
            from v2.research.services import NewsDataService
            self.services.news = NewsDataService(self.cache.store)
        else:
            self.services.news.store = self.cache.store
        if getattr(self.services, "technical", None) is None:
            from v2.research.services import TechnicalAnalysisService
            self.services.technical = TechnicalAnalysisService()
        if hasattr(self.services.supply_chain, "store"):
            self.services.supply_chain.store = self.cache.store

    def run(self, raw_ticker: str, *, modules: list[str] | None = None, force_refresh: bool = False, refresh: bool | None = None, progress: Progress | None = None, run_id: str | None = None) -> dict:
        ticker = normalize_ticker(raw_ticker)
        force_refresh = force_refresh or bool(refresh)
        required = resolve_modules(modules)
        builders = {
            "fundamental": self._fundamental,
            "valuation": self._valuation,
            "earnings": self._earnings,
            "expectations": self._expectations,
            "institutional": self._institutional,
            "fund_flow": self._fund_flow,
            "technical": self._technical,
            "sec": self._sec,
            "macro": self._macro,
            "supply_chain": self._supply_chain,
        }
        latest = self.cache.store.latest(ticker) or {}
        results: dict[str, dict] = dict(latest.get("modules", {}))
        pending: list[str] = []
        for name in required:
            cached = None if force_refresh else self.cache.get_module(ticker, name)
            if cached:
                results[name] = cached
                if progress:
                    progress(name, "CACHED")
            else:
                pending.append(name)
        dataset = self._collect(ticker, progress, set(pending), force_refresh) if pending else StockResearchDataset(ticker=ticker)

        # DAG stage 1: independent research modules consume the same normalized
        # dataset and can execute concurrently.  Each future has its own failure
        # boundary, so one provider outage never cancels sibling modules.
        base_pending = [name for name in pending if name in builders]
        with ThreadPoolExecutor(max_workers=max(1, min(8, len(base_pending)))) as executor:
            futures = {}
            for name in base_pending:
                builder = builders[name]
                if progress:
                    progress(name, "RUNNING")
                futures[executor.submit(builder, dataset)] = name
            for future in as_completed(futures):
                name = futures[future]
                try:
                    results[name] = self._finalize_module(future.result(), dataset)
                except Exception as exc:
                    results[name] = self._finalize_module(self._module(ticker, name, "PARTIAL_ERROR", "模块暂时不可用", error=str(exc)), dataset)
                self.cache.put_module(ticker, name, results[name])
                if progress:
                    progress(name, results[name]["status"])

        # DAG stage 2: catalysts run only after earnings/SEC/macro settle; news
        # is already normalized once in the shared dataset.
        if "catalyst" in pending:
            if progress:
                progress("catalyst", "RUNNING")
            try:
                results["catalyst"] = self._finalize_module(self._catalyst(dataset, results), dataset)
            except Exception as exc:
                results["catalyst"] = self._finalize_module(self._module(ticker, "catalyst", "PARTIAL_ERROR", "催化剂模块暂时不可用", error=str(exc)), dataset)
            self.cache.put_module(ticker, "catalyst", results["catalyst"])
            if progress:
                progress("catalyst", results["catalyst"]["status"])

        # DAG stage 3: risk consumes all evidence-bearing modules.
        if "risk" in pending:
            if progress:
                progress("risk", "RUNNING")
            try:
                results["risk"] = self._finalize_module(self._risk(dataset, results), dataset)
            except Exception as exc:
                results["risk"] = self._finalize_module(self._module(ticker, "risk", "PARTIAL_ERROR", "风险模块暂时不可用", error=str(exc)), dataset)
            self.cache.put_module(ticker, "risk", results["risk"])
            if progress:
                progress("risk", results["risk"]["status"])
        for name in MODULES:
            if name not in results:
                results[name] = self._finalize_module(self._module(ticker, name, "SKIPPED", "本次未请求该模块。"), dataset)
        provider_health = ProviderHealthService().check_all() if isinstance(self.fd, CachedFDClient) else []
        result = self._aggregate(dataset, results, provider_health)
        result.setdefault("production_diagnostics", {})["raw_data_cache"] = {
            "hits": int(getattr(self.fd, "hits", 0)),
            "misses": int(getattr(self.fd, "misses", 0)),
        }
        scoped_statuses = [results[name]["status"] for name in required]
        core_statuses = [results[name]["status"] for name in ("fundamental", "valuation", "earnings") if name in required]
        core_unavailable = [
            results[name]["status"] == "FAILED" or (
                results[name]["status"] == "PARTIAL_DATA"
                and not results[name].get("data_sources_used")
            )
            for name in ("fundamental", "valuation", "earnings") if name in required
        ]
        if len(core_unavailable) == 3 and all(core_unavailable):
            result["status"] = "FAILED"
        elif scoped_statuses and all(status == "FAILED" for status in scoped_statuses):
            result["status"] = "FAILED"
        elif any(status in {"FAILED", "PARTIAL_ERROR"} for status in scoped_statuses):
            result["status"] = "PARTIAL_ERROR"
        elif any(status == "PARTIAL_DATA" for status in scoped_statuses):
            result["status"] = "PARTIAL_DATA"
        else:
            result["status"] = "COMPLETED"
        result["run_id"] = run_id or result["run_id"]
        result["from_cache"] = not pending
        result["requested_modules"] = list(modules or MODULES)
        result["resolved_modules"] = required
        self.cache.store.save_snapshot(result["run_id"], result)
        return result

    def _safe(self, dataset: StockResearchDataset, key: str, fn: Callable[[], Any], default: Any) -> Any:
        for attempt in range(3):
            try:
                value = fn()
                if value is None or (key in {"company", "metrics", "earnings", "prices"} and not value):
                    dataset.errors[key] = f"{key} provider returned no data"
                    dataset.provider_error_types[key] = "EMPTY_DATA"
                    return default
                return value
            except Exception as exc:
                error_type = str(getattr(exc, "error_type", "PROVIDER_ERROR"))
                retryable = error_type in {"TIMEOUT", "UNREACHABLE", "DEGRADED"}
                if retryable and attempt < 2:
                    time.sleep(.25 * (4 ** attempt))
                    continue
                dataset.errors[key] = sanitize_error(exc)
                dataset.provider_error_types[key] = error_type
                return default
        return default

    def _collect(self, ticker: str, progress: Progress | None, required: set[str] | None = None, force_refresh: bool = False) -> StockResearchDataset:
        d = StockResearchDataset(ticker=ticker)
        required = required or set(MODULES)
        today = date.today()
        start = today - timedelta(days=550)
        if progress:
            progress("data", "RUNNING")
        if required & {"fundamental", "valuation", "risk"}:
            d.company = self._safe(d, "company", lambda: self.fd.get_company_facts(ticker), None)
            d.metrics = self._safe(d, "metrics", lambda: self.fd.get_financial_metrics(ticker, today.isoformat(), period="ttm", limit=20), [])
        if required & {"earnings", "expectations", "sec", "catalyst", "risk"}:
            d.earnings = self._safe(d, "earnings", lambda: self.fd.get_earnings_history(ticker, limit=32), [])
        if required & {"institutional", "risk", "fund_flow"}:
            d.insiders = self._safe(d, "insiders", lambda: self.fd.get_insider_trades(ticker, today.isoformat(), (today - timedelta(days=90)).isoformat(), limit=250), [])
        if "catalyst" in required:
            news_result = self.services.news.collect(ticker, self.fd, force_refresh=force_refresh)
            d.news = news_result.get("items", [])
            d.expectations["news_diagnostics"] = news_result.get("diagnostics", {})
        if required & {"fund_flow", "technical"}:
            d.prices = self._safe(d, "prices", lambda: self.prices.get_prices(ticker, start.isoformat(), today.isoformat()), [])
        if required & {"earnings", "expectations", "catalyst"}:
            try:
                from v2.earnings.calendar import get_upcoming
                d.upcoming_earnings = self._safe(d, "earnings_calendar", lambda: get_upcoming(ticker), None)
            except Exception as exc:
                d.errors["earnings_calendar"] = str(exc)

        # Shared services normalize the reliable low-level implementations that
        # previously sat behind bot responders.  They run once per Research Run
        # and populate StockResearchDataset for all downstream modules.
        service_calls = {}
        if required & {"earnings", "expectations", "sec", "catalyst"}:
            service_calls["earnings_sec"] = lambda: self.services.earnings_sec.collect(ticker, d.earnings, d.upcoming_earnings)
        if required & {"institutional", "risk"}:
            service_calls["institutional"] = lambda: self.services.institutional.collect(ticker, d.insiders)
        if required & {"macro", "catalyst", "risk"}:
            service_calls["macro"] = lambda: self.services.macro.collect(today)
        if required & {"supply_chain", "risk"}:
            service_calls["supply_chain"] = lambda: self.services.supply_chain.collect(ticker, self.fd)
        service_results: dict[str, dict] = {}
        with ThreadPoolExecutor(max_workers=max(1, min(4, len(service_calls)))) as executor:
            futures = {executor.submit(call): name for name, call in service_calls.items()}
            for future in as_completed(futures):
                name = futures[future]
                try:
                    service_results[name] = future.result() or {}
                except Exception as exc:
                    d.errors[name] = sanitize_error(exc)
                    d.provider_error_types[name] = str(getattr(exc, "error_type", "PROVIDER_ERROR"))
                    service_results[name] = {}
        earnings_sec = service_results.get("earnings_sec", {})
        news_diagnostics = d.expectations.get("news_diagnostics")
        d.expectations = earnings_sec.get("expectations", {})
        if news_diagnostics:
            d.expectations["news_diagnostics"] = news_diagnostics
        d.filings = earnings_sec.get("filings", [])
        d.sec_depth = earnings_sec.get("sec_depth", {})
        d.institutional = service_results.get("institutional", {})
        d.macro = service_results.get("macro", {})
        d.supply_chain_raw = service_results.get("supply_chain", {})
        for service_name, payload in service_results.items():
            for warning in payload.get("warnings", []) if isinstance(payload, dict) else []:
                d.errors.setdefault(service_name, str(warning))
        if d.company:
            d.sources.append(_source("fd_company", "Financial Datasets", "company_facts", f"{ticker} company facts", getattr(d.company, "sec_filings_url", None)))
        if d.metrics:
            d.sources.append(_source("fd_metrics", "Financial Datasets", "financial_metrics", f"{ticker} financial metrics", published_at=getattr(d.metrics[0], "report_period", None)))
        if d.earnings:
            latest = d.earnings[0]
            d.sources.append(_source("fd_earnings", "Financial Datasets", "earnings", f"{ticker} earnings history", getattr(latest, "filing_url", None), getattr(latest, "filing_date", None)))
        if d.prices:
            d.sources.append(_source("yf_prices", "Yahoo Finance", "ohlcv", f"{ticker} split-unadjusted OHLCV", published_at=d.prices[-1].time))
        if d.insiders:
            d.sources.append(_source("fd_insiders", "Financial Datasets", "insider_trades", f"{ticker} insider transactions"))
        if d.news:
            providers = {item.get("provider", "Financial Datasets") if isinstance(item, dict) else "Financial Datasets" for item in d.news}
            for provider in providers:
                source_id = "tavily_news" if provider == "Tavily" else "fd_news"
                d.sources.append(_source(source_id, provider, "company_news", f"{ticker} recent company news"))
        if d.upcoming_earnings:
            d.sources.append(_source("yf_calendar", "Yahoo Finance", "earnings_calendar", f"{ticker} upcoming earnings", published_at=d.upcoming_earnings.release_date))
        if any(item.get("source") == "SEC EDGAR" for item in d.filings):
            d.sources.append(_source("sec_filings", "SEC EDGAR", "filings", f"{ticker} recent 10-K, 10-Q and 8-K filings", getattr(d.company, "sec_filings_url", None) if d.company else None))
        if d.institutional.get("institutional_holdings"):
            d.sources.append(_source("local_13f", "SEC 13F archive", "institutional_holdings", f"{ticker} holdings across tracked managers"))
        if d.institutional.get("etf_exposure"):
            d.sources.append(_source("local_etf", "ARK holdings archive", "etf_exposure", f"{ticker} latest ARK ETF exposure"))
        if d.macro.get("snapshot"):
            d.sources.append(_source("macro_snapshot", "FRED + Yahoo Finance", "macro_snapshot", "US macro and market snapshot", published_at=today.isoformat()))
        if d.supply_chain_raw.get("neighbors"):
            d.sources.append(_source("supply_chain", "DeepSeek + Financial Datasets + Yahoo Finance + Tavily", "supply_chain", f"{ticker} validated company relationships"))
        if progress:
            progress("data", "PARTIAL_ERROR" if d.errors else "COMPLETED")
        return d

    def _module(self, ticker: str, name: str, status: str, summary: str, *, score: int | None = None, metrics: dict | None = None, findings: list | None = None, risks: list | None = None, sources: list | None = None, confidence: float = 0.0, details: dict | None = None, error: str | None = None) -> dict:
        return {"status": status, "ticker": ticker, "module": name, "summary": summary, "score": score, "metrics": metrics or {}, "findings": findings or [], "risks": risks or [], "sources": sources or [], "confidence": confidence, "generated_at": _now(), "details": details or {}, "error": error}

    def _finalize_module(self, result: dict, dataset: StockResearchDataset) -> dict:
        result = dict(result)
        status = result.get("status", "FAILED")
        if status == "SKIPPED":
            result.update({"engine_version": ENGINE_VERSION, "completeness": 0.0, "missing_fields": [], "warnings": [], "errors": [], "provider_errors": [], "data_sources_used": [], "source_count": 0, "verified_source_count": 0, "cache_hit": False, "cache_expires_at": None, "source_freshness": {}})
            return result
        if status == "PARTIAL":
            status = "PARTIAL_DATA"
        metrics = result.get("metrics") or {}
        missing = list(result.get("missing_fields") or [key for key, value in metrics.items() if value is None])
        provider_errors = list(result.get("provider_errors") or [])
        error_dependencies = {
            "fundamental": {"company", "metrics", "earnings"}, "valuation": {"metrics"},
            "earnings": {"earnings"}, "expectations": {"earnings", "earnings_calendar", "earnings_sec"},
            "institutional": {"insiders", "institutional"}, "fund_flow": {"prices", "insiders"},
            "technical": {"prices"}, "catalyst": {"news", "earnings", "earnings_sec", "macro"},
            "sec": {"earnings_sec"}, "macro": {"macro"}, "supply_chain": {"supply_chain"},
            "risk": set(dataset.errors),
        }
        for key, message in dataset.errors.items():
            if key in error_dependencies.get(result.get("module", ""), set()):
                error_type = dataset.provider_error_types.get(key, "PROVIDER_ERROR")
                provider_errors.append({"provider": key, "type": error_type, "message": message, "retryable": error_type not in {"AUTH_ERROR", "EMPTY_DATA"}})
        if result.get("error"):
            dependency_error_types = {str(item.get("type") or "PROVIDER_ERROR") for item in provider_errors}
            if dependency_error_types == {"EMPTY_DATA"}:
                # Do not duplicate a provider's empty response as a code
                # execution failure merely because a legacy builder mirrors
                # the same condition through its `error` field.
                status = "PARTIAL_DATA"
            else:
                provider_errors.append({"provider": result.get("module"), "type": "EXECUTION_ERROR", "message": str(result["error"]), "retryable": True})
                if status != "FAILED":
                    status = "PARTIAL_ERROR"
        elif provider_errors:
            error_types = {str(item.get("type") or "PROVIDER_ERROR") for item in provider_errors}
            if error_types == {"EMPTY_DATA"}:
                status = "PARTIAL_DATA"
            else:
                # Authentication and transport failures degrade every module,
                # including core modules. They must never masquerade as data
                # incompleteness.
                status = "PARTIAL_ERROR"
        elif status == "FAILED" and result.get("module") not in {"fundamental", "valuation", "earnings"}:
            # Optional modules with a successful code path but no provider data
            # are data-incomplete, not execution failures.
            status = "PARTIAL_DATA"
        required_count = len(metrics)
        completeness = 1.0 if not required_count else round((required_count - len(missing)) / required_count, 4)
        verified_source_count = len(result.get("sources", []))
        if result.get("module") == "supply_chain":
            completeness = round(float(metrics.get("verified_relationships", 0)) / float(metrics["relationships"]), 4) if metrics.get("relationships") else 0.0
            verified_source_count = int(metrics.get("verified_relationships", 0) or 0)
        if status == "COMPLETED" and missing:
            status = "PARTIAL_DATA"
        result.update({"status": status, "engine_version": ENGINE_VERSION, "completeness": max(0.0, completeness), "missing_fields": missing, "warnings": result.get("warnings", result.get("details", {}).get("warnings", [])) or [], "errors": [str(result["error"])] if result.get("error") else [], "provider_errors": provider_errors, "data_sources_used": list(result.get("sources", [])), "source_count": len(result.get("sources", [])), "verified_source_count": verified_source_count, "cache_hit": bool(result.get("cache_hit", False)), "cache_expires_at": result.get("cache_expires_at"), "source_freshness": result.get("source_freshness", {source: result.get("generated_at") for source in result.get("sources", [])})})
        return result

    @staticmethod
    def _quarters(d: StockResearchDataset) -> list[Any]:
        priority = {"10-Q": 3, "8-K": 2, "10-K": 1, "20-F": 1}
        by_period: dict[str, Any] = {}
        for record in d.earnings:
            if not getattr(record, "quarterly", None):
                continue
            period = record.report_period
            if period not in by_period or priority.get(record.source_type, 0) > priority.get(by_period[period].source_type, 0):
                by_period[period] = record
        return sorted(by_period.values(), key=lambda row: row.report_period, reverse=True)[:8]

    def _fundamental(self, d: StockResearchDataset) -> dict:
        latest = d.metrics[0] if d.metrics else None
        quarters = self._quarters(d)
        if not latest and not quarters and not d.company:
            return self._module(d.ticker, "fundamental", "FAILED", "Financial data temporarily unavailable.", error=d.errors.get("metrics") or d.errors.get("earnings"))
        qrows = []
        for record in reversed(quarters):
            q = record.quarterly
            revenue = _finite(q.revenue)
            qrows.append({"period": record.report_period, "revenue": revenue, "revenue_yoy": _finite(q.revenue_chg), "eps": _finite(q.earnings_per_share), "net_income": _finite(q.net_income), "net_income_growth": _finite(q.net_income_chg), "free_cash_flow": _finite(q.free_cash_flow), "free_cash_flow_growth": _finite(q.free_cash_flow_chg), "operating_cash_flow": _finite(q.net_cash_flow_from_operations), "capital_expenditure": _finite(q.capital_expenditure), "stock_based_compensation": _finite(getattr(q, "stock_based_compensation", None)), "share_repurchases": _finite(getattr(q, "share_repurchases", None)), "dividends_paid": _finite(getattr(q, "dividends_paid", None)), "research_and_development": _finite(getattr(q, "research_and_development", None)), "gross_margin": (revenue and _finite(q.gross_profit) is not None) and _finite(q.gross_profit) / revenue, "operating_margin": (revenue and _finite(q.operating_income) is not None) and _finite(q.operating_income) / revenue, "net_margin": (revenue and _finite(q.net_income) is not None) and _finite(q.net_income) / revenue, "fcf_margin": (revenue and _finite(q.free_cash_flow) is not None) and _finite(q.free_cash_flow) / revenue, "shares": _finite(q.weighted_average_shares_diluted or q.weighted_average_shares)})
        growth = _finite(getattr(latest, "revenue_growth", None))
        profitability = _weighted([(_ratio_score(_finite(getattr(latest, "gross_margin", None)), .2, .7), .2), (_ratio_score(_finite(getattr(latest, "operating_margin", None)), 0, .35), .2), (_ratio_score(_finite(getattr(latest, "net_margin", None)), 0, .3), .15), (_ratio_score(_finite(getattr(latest, "return_on_invested_capital", None)), .05, .3), .3), (_ratio_score(_finite(getattr(latest, "free_cash_flow_yield", None)), 0, .08), .15)])
        growth_score = _weighted([(_ratio_score(growth, -.1, .35), .45), (_ratio_score(_finite(getattr(latest, "earnings_growth", None)), -.1, .4), .35), (_ratio_score(_finite(getattr(latest, "free_cash_flow_growth", None)), -.2, .4), .2)])
        current_ratio = _finite(getattr(latest, "current_ratio", None))
        interest = _finite(getattr(latest, "interest_coverage", None))
        debt_equity = _finite(getattr(latest, "debt_to_equity", None))
        health_score = _weighted([(_ratio_score(current_ratio, .7, 2.0), .35), (_ratio_score(interest, 1.0, 10.0), .35), (None if debt_equity is None else 100 - (_ratio_score(debt_equity, .2, 2.0) or 0), .3)])
        score = _weighted([(growth_score, .35), (profitability, .4), (health_score, .25)])
        trend = "Insufficient Evidence"
        recent_growth = [row["revenue_yoy"] for row in qrows if row["revenue_yoy"] is not None]
        if recent_growth:
            trend = "Contracting" if recent_growth[-1] < 0 else "Accelerating" if len(recent_growth) > 1 and recent_growth[-1] > recent_growth[-2] + .03 else "Decelerating" if len(recent_growth) > 1 and recent_growth[-1] < recent_growth[-2] - .03 else "Strong Growth" if recent_growth[-1] >= .15 else "Stable"
        newest = qrows[-1] if qrows else {}
        oldest = qrows[0] if len(qrows) >= 4 else {}
        share_change = None
        if newest.get("shares") and oldest.get("shares"):
            share_change = newest["shares"] / oldest["shares"] - 1
        balance = "Strong" if (health_score or 0) >= 70 else "Weak" if health_score is not None and health_score < 40 else "Normal"
        findings = [{"title": "增长趋势", "description": trend, "confidence": .85 if recent_growth else .25, "source_ids": ["fd_metrics", "fd_earnings"]}, {"title": "资产负债表", "description": balance, "confidence": .75 if health_score is not None else .2, "source_ids": ["fd_metrics"]}]
        if share_change is not None and share_change > .08 and growth and growth > .1:
            findings.append({"title": "稀释提示", "description": "利润或收入增长较强，但股份数量增长抵消了部分每股价值增长。", "confidence": .8, "source_ids": ["fd_earnings"]})
        profile = {"description": None, "business_model": None, "core_products": [], "primary_markets": [], "segments": [], **d.sec_depth.get("company_profile", {})}
        capital_allocation = {
            "stock_based_compensation": newest.get("stock_based_compensation"),
            "buyback": newest.get("share_repurchases"),
            "dividend": newest.get("dividends_paid"),
            "research_and_development": newest.get("research_and_development"),
            "capital_expenditure": newest.get("capital_expenditure"),
            "assessment": "DILUTIVE" if share_change is not None and share_change > .03 else "SHAREHOLDER_RETURN" if (newest.get("share_repurchases") or newest.get("dividends_paid")) else "INSUFFICIENT_EVIDENCE",
        }
        return self._module(d.ticker, "fundamental", "COMPLETED" if latest and quarters else "PARTIAL", f"基本面评分 {score if score is not None else 'N/A'}；增长状态：{trend}。", score=score, metrics={"market_cap": _finite(getattr(latest, "market_cap", None)), "revenue_growth": growth, "eps_growth": _finite(getattr(latest, "earnings_per_share_growth", None)), "gross_margin": _finite(getattr(latest, "gross_margin", None)), "operating_margin": _finite(getattr(latest, "operating_margin", None)), "net_margin": _finite(getattr(latest, "net_margin", None)), "roic": _finite(getattr(latest, "return_on_invested_capital", None)), "roe": _finite(getattr(latest, "return_on_equity", None)), "roa": _finite(getattr(latest, "return_on_assets", None)), "current_ratio": current_ratio, "interest_coverage": interest, "debt_to_equity": debt_equity, "share_count_change": share_change, "growth_score": growth_score, "profitability_score": profitability, "financial_health_score": health_score}, findings=findings, sources=[s["id"] for s in d.sources if s["id"] in {"fd_company", "fd_metrics", "fd_earnings", "sec_filings"}], confidence=.85 if latest and len(quarters) >= 4 else .55, details={"company": {"name": getattr(d.company, "name", None), "ticker": d.ticker, "sector": getattr(d.company, "sector", None), "industry": getattr(d.company, "industry", None), "exchange": getattr(d.company, "exchange", None), "location": getattr(d.company, "location", None), **profile}, "quarterly_trends": qrows, "growth_trend": trend, "financial_health": {"balance_sheet": balance, "debt_risk": "Low" if (health_score or 0) >= 70 else "High" if health_score is not None and health_score < 40 else "Medium", "cash_generation": "Strong" if (_finite(getattr(latest, "free_cash_flow_growth", None)) or 0) > .1 else "Weak" if (_finite(getattr(latest, "free_cash_flow_growth", None)) or 0) < 0 else "Medium"}, "shareholder_value": {"shares_outstanding": newest.get("shares"), "share_count_change": share_change, **capital_allocation}, "capital_allocation": capital_allocation, "management": {"status": "Insufficient Evidence"}})

    def _valuation(self, d: StockResearchDataset) -> dict:
        rows = d.metrics
        if not rows:
            return self._module(d.ticker, "valuation", "FAILED", "Financial data temporarily unavailable.", error=d.errors.get("metrics"))
        latest = rows[0]
        fields = {"pe_ttm": "price_to_earnings_ratio", "forward_pe": "forward_price_to_earnings_ratio", "peg": "peg_ratio", "price_to_sales": "price_to_sales_ratio", "ev_sales": "enterprise_value_to_revenue_ratio", "ev_ebitda": "enterprise_value_to_ebitda_ratio", "fcf_yield": "free_cash_flow_yield"}
        current = {key: _finite(getattr(latest, attr, None)) for key, attr in fields.items()}
        for key in ("pe_ttm", "forward_pe", "ev_ebitda"):
            if current[key] is not None and current[key] <= 0:
                current[key] = None
        history = {}
        for key, attr in fields.items():
            values = [value for row in rows if (value := _finite(getattr(row, attr, None))) is not None and ("pe" not in key or value > 0)]
            history[key] = {"current": current[key], "available_periods": len(values), "median": median(values) if values else None, "percentile": (sum(value <= current[key] for value in values) / len(values)) if values and current[key] is not None else None}
        pe = current["pe_ttm"]
        fcf = current["fcf_yield"]
        valuation_score = _weighted([(None if pe is None else max(0, min(100, 110 - pe * 2)), .55), (_ratio_score(fcf, 0, .08), .45)])
        med = history["pe_ttm"]["median"]
        summary = "有效估值指标不足。" if valuation_score is None else ("当前估值低于可用历史中位数。" if pe is not None and med is not None and pe < med else "当前估值不低于可用历史中位数。")
        preferences = self.cache.store.peer_preferences(d.ticker)
        base_peers = _PEERS.get(d.ticker, _SECTOR_PEERS.get(str(getattr(d.company, "sector", "")).lower(), []))
        peers = [peer for peer in dict.fromkeys([*base_peers, *preferences["added"]]) if peer not in preferences["removed"] and peer != d.ticker][:6]
        peer_rows: list[dict] = []
        if isinstance(self.fd, CachedFDClient):
            for peer in peers:
                facts = self._safe(d, f"peer_{peer}_company", lambda peer=peer: self.fd.get_company_facts(peer), None)
                metrics = self._safe(d, f"peer_{peer}_metrics", lambda peer=peer: self.fd.get_financial_metrics(peer, date.today().isoformat(), period="ttm", limit=1), [])
                if not facts or not metrics:
                    continue
                row = metrics[0]
                peer_rows.append({"ticker": peer, "name": getattr(facts, "name", None), "sector": getattr(facts, "sector", None), "industry": getattr(facts, "industry", None), "market_cap": _finite(getattr(row, "market_cap", None)), "pe_ttm": _finite(getattr(row, "price_to_earnings_ratio", None)), "ev_ebitda": _finite(getattr(row, "enterprise_value_to_ebitda_ratio", None)), "revenue_growth": _finite(getattr(row, "revenue_growth", None)), "operating_margin": _finite(getattr(row, "operating_margin", None)), "roic": _finite(getattr(row, "return_on_invested_capital", None)), "validated": True})
        peer_note = f"{len(peer_rows)}/{len(peers)} 个同行通过公司与财务数据验证；用户可增删候选。"
        return self._module(d.ticker, "valuation", "COMPLETED" if peer_rows else "PARTIAL", summary, score=valuation_score, metrics=current, findings=[{"title": "历史估值位置", "description": summary, "confidence": .7, "source_ids": ["fd_metrics"]}], sources=["fd_metrics"], confidence=.82 if peer_rows else .7, details={"history": history, "peer_set": peers, "peer_comparison": peer_rows, "peer_preferences": preferences, "peer_note": peer_note})

    def _earnings(self, d: StockResearchDataset) -> dict:
        quarters = self._quarters(d)
        if not quarters:
            return self._module(d.ticker, "earnings", "FAILED", "暂无可靠财报历史。", metrics={"latest_eps_surprise": None, "latest_revenue_surprise": None, "beat_rate": None}, error=d.errors.get("earnings"))
        rows = []
        for record in quarters:
            q = record.quarterly
            eps_a, eps_e = _finite(q.earnings_per_share), _finite(q.estimated_earnings_per_share)
            rev_a, rev_e = _finite(q.revenue), _finite(q.estimated_revenue)
            rows.append({"period": record.report_period, "filing_date": record.filing_date, "source_type": record.source_type, "filing_url": record.filing_url, "eps_actual": eps_a, "eps_estimate": eps_e, "eps_surprise": None if eps_a is None or not eps_e else (eps_a - eps_e) / abs(eps_e), "revenue_actual": rev_a, "revenue_estimate": rev_e, "revenue_surprise": None if rev_a is None or not rev_e else (rev_a - rev_e) / abs(rev_e)})
        surprises = [row["eps_surprise"] for row in rows if row["eps_surprise"] is not None]
        score = _weighted([(_ratio_score(sum(surprises[:4]) / min(4, len(surprises)), -.1, .1) if surprises else None, .7), (_ratio_score(sum(value > 0 for value in surprises[:4]) / min(4, len(surprises)), 0, 1) if surprises else None, .3)])
        return self._module(d.ticker, "earnings", "COMPLETED", f"最近 {len(rows)} 个季度中有 {sum(value > 0 for value in surprises)} 次 EPS 超预期。" if surprises else "有财报数据，但一致预期字段不足。", score=score, metrics={"latest_eps_surprise": rows[0]["eps_surprise"], "latest_revenue_surprise": rows[0]["revenue_surprise"], "beat_rate": sum(value > 0 for value in surprises) / len(surprises) if surprises else None}, sources=["fd_earnings"], confidence=.88, details={"history": rows})

    def _expectations(self, d: StockResearchDataset) -> dict:
        earnings = self._earnings(d)
        upcoming = d.expectations.get("upcoming_earnings") or (asdict(d.upcoming_earnings) if d.upcoming_earnings else None)
        if d.expectations.get("consensus") is not None:
            upcoming = {**(upcoming or {}), **d.expectations["consensus"]}
        surprise = earnings["metrics"].get("latest_eps_surprise")
        beat_rate = earnings["metrics"].get("beat_rate")
        score = _weighted([(_ratio_score(surprise, -.1, .1), .55), (_ratio_score(beat_rate, 0, 1), .45)])
        trend = "Positive" if score is not None and score >= 65 else "Negative" if score is not None and score < 40 else "Neutral" if score is not None else "Insufficient Evidence"
        status = "PARTIAL"  # provider does not expose a 30/60/90-day revision history
        return self._module(d.ticker, "expectations", status, f"预期动量：{trend}。历史修正序列若数据源未提供则不推断。", score=score, metrics={"earnings_momentum": score, "revision_trend": None, "forward_eps": upcoming.get("eps_estimate") if upcoming else None, "revenue_consensus": upcoming.get("revenue_estimate") if upcoming else None}, findings=[{"title": "Estimate Revision", "description": "Insufficient Evidence" if not upcoming else "当前仅有最新一致预期，缺少可比的 30/60/90 日快照。", "confidence": .3, "source_ids": ["yf_calendar"] if upcoming else []}], sources=[key for key in ("fd_earnings", "yf_calendar") if any(s["id"] == key for s in d.sources)], confidence=.55, details={"upcoming_earnings": upcoming, "revision_history": d.expectations.get("revision_history", []), "capability_matrix": EXPECTATIONS_CAPABILITY, "guidance": d.sec_depth.get("guidance", [])})

    def _institutional(self, d: StockResearchDataset) -> dict:
        buys = sells = 0.0
        for trade in d.insiders:
            value = abs(_finite(trade.transaction_value) or ((_finite(trade.transaction_shares) or 0) * (_finite(trade.transaction_price_per_share) or 0)))
            kind = (trade.transaction_type or "").lower()
            if "purchase" in kind or kind in {"p", "buy"}:
                buys += value
            elif "sale" in kind or kind in {"s", "sell"}:
                sells += value
        score = None if not d.insiders else _clamp(50 + 50 * ((buys - sells) / max(buys + sells, 1)))
        holdings = d.institutional.get("institutional_holdings", [])
        changes = d.institutional.get("13f_changes", [])
        etf_exposure = d.institutional.get("etf_exposure", [])
        sources = []
        if d.insiders:
            sources.append("fd_insiders")
        if holdings:
            sources.append("local_13f")
        if etf_exposure:
            sources.append("local_etf")
        available = bool(d.insiders or holdings or etf_exposure)
        summary = f"读取 {len(holdings)} 家跟踪机构持仓、{len(changes)} 条 13F 变化、{len(etf_exposure)} 条 ETF 暴露和 {len(d.insiders)} 条内部人交易。" if available else "暂无可用机构、ETF或内部人记录。"
        status = "COMPLETED" if holdings and d.insiders else "PARTIAL" if available else "FAILED"
        return self._module(d.ticker, "institutional", status, summary, score=score, metrics={"insider_buy_value_90d": buys, "insider_sell_value_90d": sells, "institutional_holders": len(holdings), "13f_changes": len(changes), "etf_exposure_count": len(etf_exposure), "institutional_flow": None, "etf_flow": None, "institutional_flow_available": False, "etf_flow_available": False}, risks=[{"title": "内部人净卖出", "description": _money(sells - buys), "confidence": .8, "source_ids": ["fd_insiders"]}] if sells > buys else [], sources=sources, confidence=.78 if holdings and d.insiders else .55 if available else .1, details={"institutional_holdings": holdings, "13f_changes": changes, "insider_transactions": d.institutional.get("insider_transactions", []), "ownership_changes": d.institutional.get("ownership_changes", []), "etf_exposure": etf_exposure, "semantics": {"13f": "quarterly disclosed positions, not real-time flow", "etf_exposure": "latest holdings snapshot, not ETF fund flow", "insider_transactions": "reported transactions, not institutional flow"}, "warnings": d.institutional.get("warnings", [])})

    def _fund_flow(self, d: StockResearchDataset) -> dict:
        from v2.research.moneyflow import build_flow_analysis
        prices = d.prices
        if len(prices) < 20:
            return self._module(d.ticker, "fund_flow", "FAILED", "OHLCV 数据不足，无法计算资金流。", error=d.errors.get("prices"), details={"flow_analysis": build_flow_analysis(d.ticker, prices)})
        volumes = [_finite(p.volume) or 0 for p in prices]
        closes = [_finite(p.close) or 0 for p in prices]
        cmf = _cmf([p.high for p in prices], [p.low for p in prices], closes, volumes, 20)
        vol_ma = sum(volumes[-20:]) / 20
        rvol = volumes[-1] / vol_ma if vol_ma else None
        obv = 0.0
        ad = 0.0
        for index, p in enumerate(prices):
            if index:
                obv += volumes[index] if closes[index] > closes[index - 1] else -volumes[index] if closes[index] < closes[index - 1] else 0
            spread = p.high - p.low
            ad += (0 if spread <= 0 else ((p.close - p.low) - (p.high - p.close)) / spread) * p.volume
        score = _weighted([(_ratio_score(cmf, -.25, .25), .65), (_ratio_score(rvol, .5, 2), .35)])
        state = "净流入" if cmf is not None and cmf > .05 else "净流出" if cmf is not None and cmf < -.05 else "中性"
        return self._module(d.ticker, "fund_flow", "COMPLETED", f"CMF20 显示{state}；相对成交量 {_finite(rvol) or 0:.2f}×。", score=score, metrics={"volume": volumes[-1], "relative_volume": rvol, "cmf20": cmf, "obv": obv, "accumulation_distribution": ad, "institutional_flow": None, "etf_flow": None, "insider_flow": self._institutional(d)["metrics"]}, sources=["yf_prices"] + (["fd_insiders"] if d.insiders else []), confidence=.82, details={"flow_analysis": build_flow_analysis(d.ticker, prices)})

    def _technical(self, d: StockResearchDataset) -> dict:
        prices = d.prices
        if len(prices) < 20:
            return self._module(d.ticker, "technical", "FAILED", "OHLCV 数据不足。")
        analysis = self.services.technical.analyze(prices, "1d")
        current = analysis.get("currentPrice") or analysis.get("current_price") or prices[-1].close
        sma20 = analysis.get("sma20") or analysis.get("SMA20")
        sma50 = analysis.get("sma50") or analysis.get("SMA50")
        sma200 = analysis.get("sma200") or analysis.get("SMA200")
        score = _weighted([(100 if current > sma20 else 20, .4), (None if sma50 is None else (100 if current > sma50 else 20), .3), (None if sma200 is None else (100 if current > sma200 else 20), .3)])
        trend = "Bullish" if score and score >= 70 else "Bearish" if score is not None and score < 40 else "Mixed"
        metrics = {"current_price": current, "sma20": sma20, "sma50": sma50, "sma200": sma200, "ema": analysis.get("EMA20") or analysis.get("ema20"), "rsi": analysis.get("RSI14") or analysis.get("rsi14"), "macd": analysis.get("MACD") or analysis.get("macd"), "atr": analysis.get("ATR14") or analysis.get("atr14"), "cmf": analysis.get("CMF20") or analysis.get("cmf20"), "obv": analysis.get("OBV") or analysis.get("obv"), "relative_volume": analysis.get("volumeRatio") or analysis.get("volume_ratio"), "support": analysis.get("supports", []), "resistance": analysis.get("resistances", []), "breakout_state": analysis.get("breakoutStatus"), "trend_state": analysis.get("trend") or trend, "market_regime": analysis.get("marketRegime"), "timeframe": analysis.get("timeframe", "1d")}
        return self._module(d.ticker, "technical", "COMPLETED", f"日线趋势：{trend}。", score=score, metrics=metrics, sources=["yf_prices"], confidence=.9, details={"analysis": analysis})

    def _catalyst(self, d: StockResearchDataset, modules: dict[str, dict] | None = None) -> dict:
        items = []
        if d.upcoming_earnings:
            items.append({"title": "季度财报", "event_date": d.upcoming_earnings.release_date, "catalyst_type": "earnings", "source_kind": "EVENT", "direction": "neutral", "impact": "Very High", "description": "下一次已知财报窗口。", "source_id": "yf_calendar", "confidence": .9})
        for index, news in enumerate(d.news[:5]):
            title = news.get("title") if isinstance(news, dict) else news.title
            event_date = news.get("date") if isinstance(news, dict) else news.date
            url = news.get("url") if isinstance(news, dict) else news.url
            provider = news.get("provider", "Financial Datasets") if isinstance(news, dict) else "Financial Datasets"
            items.append({"title": title, "event_date": event_date, "catalyst_type": "company_news", "source_kind": "NEWS", "direction": "neutral", "impact": "Unscored", "description": "近期新闻事实；未使用语言模型推断方向。", "source_id": "tavily_news" if provider == "Tavily" else "fd_news", "source_url": url, "confidence": .65})
        for event in d.macro.get("upcoming_events", [])[:5]:
            items.append({"title": event.get("title") or event.get("release_type"), "event_date": event.get("event_date"), "catalyst_type": "macro", "source_kind": "EVENT", "direction": "neutral", "impact": "High" if event.get("release_type") in {"FOMC", "CPI", "NFP"} else "Medium", "description": "已公布日程的美国宏观事件。", "source_id": "macro_snapshot", "confidence": .9})
        for row in d.sec_depth.get("guidance", [])[:4]:
            items.append({"title": f"Management guidance: {row.get('status')}", "event_date": row.get("filing_date"), "catalyst_type": "guidance", "source_kind": "CATALYST", "direction": "negative" if row.get("status") in {"LOWERED", "WITHDRAWN"} else "positive" if row.get("status") == "RAISED" else "neutral", "impact": "High", "description": row.get("evidence_text"), "source_id": "sec_filings", "source_url": row.get("source_url"), "confidence": row.get("confidence", .65)})
        items = classify_catalysts(items)
        status = "COMPLETED" if items else "PARTIAL"
        score = 55 if d.upcoming_earnings else None
        diagnostics = d.expectations.get("news_diagnostics", {})
        attempts = diagnostics.get("provider_attempts", [])
        fallback_recovered = any(attempt.get("provider") != "Financial Datasets" and int(attempt.get("filtered_count") or 0) > 0 for attempt in attempts)
        provider_errors = [{"provider": attempt.get("provider"), "type": attempt.get("error_type") or "PROVIDER_ERROR", "message": attempt.get("error"),
                            "retryable": (attempt.get("error_type") or "PROVIDER_ERROR") not in {"AUTH_ERROR", "EMPTY_DATA"}}
                           for attempt in attempts if attempt.get("error") and not (fallback_recovered and attempt.get("provider") == "Financial Datasets")]
        result = self._module(d.ticker, "catalyst", status, f"识别到 {len(items)} 条去重记录，并区分新闻、事件与催化剂。" if items else "暂无可靠催化剂数据。", score=score, metrics={"known_events": len(items), "news_count": sum(i["item_type"] == "NEWS" for i in items), "event_count": sum(i["item_type"] == "EVENT" for i in items), "catalyst_count": sum(i["item_type"] == "CATALYST" for i in items)}, sources=list({item["source_id"] for item in items}), confidence=.7 if items else .2, details={"timeline": items, "news_diagnostics": diagnostics})
        result["provider_errors"] = provider_errors
        return result

    def _sec(self, d: StockResearchDataset) -> dict:
        filings = d.filings
        direct = [item for item in filings if item.get("source") == "SEC EDGAR"]
        status = "COMPLETED" if direct else "PARTIAL" if filings else "FAILED"
        sources = (["sec_filings"] if direct else []) + (["fd_earnings"] if filings else [])
        findings = d.sec_depth.get("findings", [])
        risk_changes = d.sec_depth.get("risk_factor_changes", [])
        parsed_forms = d.sec_depth.get("parsed_forms", [])
        return self._module(d.ticker, "sec", status, f"已索引 {len(filings)} 份文件并解析 {len(findings)} 条发现。", metrics={"filing_count": len(filings), "direct_sec_count": len(direct), "finding_count": len(findings), "risk_change_count": len(risk_changes)}, findings=findings, sources=list(dict.fromkeys(sources)), confidence=.9 if findings else .7 if direct else .5 if filings else .1, details={"filings": filings, "sec_findings": findings, "industry_evidence": d.sec_depth.get("industry_evidence", []), "risk_factor_changes": risk_changes, "guidance": d.sec_depth.get("guidance", []), "parsed_forms": parsed_forms, "status_note": "已完成正文解析。" if findings else "已获得文件元数据；正文暂未成功解析。"})

    def _macro(self, d: StockResearchDataset) -> dict:
        snapshot = d.macro.get("snapshot") or {}
        events = d.macro.get("upcoming_events", [])
        if not snapshot and not events:
            return self._module(d.ticker, "macro", "FAILED", "FRED/Yahoo 宏观数据暂时不可用。", metrics={}, confidence=.1, details={"snapshot": None, "upcoming_events": [], "warnings": d.macro.get("warnings", [])})
        metrics = {key: snapshot.get(key) for key in ("vix", "vix_pct_change_1d", "dxy", "wti_crude", "gold", "fed_funds_upper", "fed_funds_lower", "dgs2", "dgs10", "t10y2y")}
        status = "COMPLETED" if snapshot and not d.macro.get("warnings") else "PARTIAL"
        return self._module(d.ticker, "macro", status, f"已读取宏观快照和未来 {len(events)} 个已知事件。", metrics=metrics, findings=[{"title": "市场状态", "description": "宏观数值来自 FRED 与 Yahoo Finance；不由语言模型生成。", "confidence": .9, "source_ids": ["macro_snapshot"]}], sources=["macro_snapshot"] if snapshot else [], confidence=.85 if snapshot else .55, details={"snapshot": snapshot or None, "upcoming_events": events, "warnings": d.macro.get("warnings", [])})

    def _supply_chain(self, d: StockResearchDataset) -> dict:
        raw = d.supply_chain_raw
        relationships = []
        for neighbor in raw.get("neighbors", []) if isinstance(raw, dict) else []:
            if not neighbor.get("exists"):
                continue
            for label in neighbor.get("labels", []):
                if str(label.get("seed", "")).upper() != d.ticker:
                    continue
                relationships.append({
                    "source_company": d.ticker,
                    "target_company": neighbor.get("ticker"),
                    "relationship_type": label.get("category"),
                    "description": label.get("reason"),
                    "source": label.get("evidence_url") or neighbor.get("relation_evidence_url"),
                    "evidence_status": label.get("evidence_status", "UNCHECKED"),
                    "evidence_text": label.get("evidence_text", ""),
                    "confidence": .9 if neighbor.get("relation_verified") else .45,
                    "verified": bool(neighbor.get("relation_verified")),
                    "updated_at": raw.get("date") or _now(),
                })
        category_to_type = {"supplier": "SUPPLIER", "customer": "CUSTOMER", "smaller_peer": "COMPETITOR", "beneficiary": "BENEFICIARY", "partner": "PARTNER", "platform": "PLATFORM", "substitute": "SUBSTITUTE"}
        persisted = { (item["target_ticker"], item["relationship_type"]): item for item in self.cache.store.relationships(d.ticker) }
        for item in relationships:
            saved = persisted.get((item["target_company"], category_to_type.get(str(item["relationship_type"]).lower(), str(item["relationship_type"]).upper())))
            if saved:
                item["id"] = saved["id"]
                item["status"] = saved["status"]
        verified = sum(item["verified"] for item in relationships)
        status = "COMPLETED" if relationships and verified == len(relationships) else "PARTIAL" if relationships else "FAILED"
        if raw.get('candidate_errors'):
            status = 'PARTIAL_ERROR'
        summary = f"识别 {len(relationships)} 条关系，其中 {verified} 条通过外部证据验证。" if relationships else "本次未获得可展示的产业关系。"
        return self._module(d.ticker, "supply_chain", status, summary, metrics={"relationships": len(relationships), "verified_relationships": verified, "llm_tokens": raw.get("llm_tokens", 0), "api_calls": raw.get("api_calls", 0), "tavily_calls": raw.get("tavily_calls", 0)}, sources=["supply_chain"] if relationships else [], confidence=.75 if verified else .4 if relationships else .1, details={"relationships": relationships, "warnings": raw.get("warnings", []), "candidate_errors": raw.get("candidate_errors", []), "api_call_counts": raw.get("api_call_counts", {}), "call_count_note": "调用数为逻辑接口尝试次数（含失败），不等于底层 HTTP 重试或计费次数。"})

    def _risk(self, d: StockResearchDataset, modules: dict[str, dict]) -> dict:
        risks: list[dict] = []
        valuation, fundamental, institutional = modules["valuation"], modules["fundamental"], modules["institutional"]

        def add(name: str, level: str = "UNKNOWN", reason: str = "Insufficient Evidence", evidence: list[str] | None = None, confidence: float = .1, trend: str = "UNKNOWN") -> None:
            risks.append({"name": name, "level": level, "trend": trend, "reason": reason, "recent_change": trend, "evidence": evidence or [], "confidence": confidence, "updated_at": _now()})

        pe = _finite(valuation.get("metrics", {}).get("pe_ttm"))
        add("VALUATION", "HIGH" if pe is not None and pe >= 45 else "MEDIUM" if pe is not None and pe >= 25 else "LOW" if pe is not None else "UNKNOWN", f"TTM P/E 为 {pe:.1f}x。" if pe is not None else "Insufficient Evidence", ["fd_metrics"] if pe is not None else [], .86 if pe is not None else .1)
        health = _finite(fundamental.get("metrics", {}).get("financial_health_score"))
        add("FINANCIAL", "LOW" if health is not None and health >= 70 else "HIGH" if health is not None and health < 40 else "MEDIUM" if health is not None else "UNKNOWN", "依据流动性、利息保障和杠杆指标的确定性规则。" if health is not None else "Insufficient Evidence", ["fd_metrics"] if health is not None else [], .8 if health is not None else .1)
        risk_changes = modules["sec"].get("details", {}).get("risk_factor_changes", [])
        accounting_hits = [item for item in modules["sec"].get("details", {}).get("sec_findings", []) if any(word in str(item).lower() for word in ("restatement", "material weakness", "going concern"))]
        add("ACCOUNTING", "HIGH" if accounting_hits else "LOW" if modules["sec"].get("details", {}).get("parsed_forms") else "UNKNOWN", f"检测到 {len(accounting_hits)} 条会计或内控风险证据。" if accounting_hits else "已解析文件未发现明确会计高风险关键词。" if modules["sec"].get("details", {}).get("parsed_forms") else "Insufficient Evidence", ["sec_filings"] if modules["sec"].get("details", {}).get("parsed_forms") else [], .76 if modules["sec"].get("details", {}).get("parsed_forms") else .1, "RISING" if accounting_hits else "STABLE" if modules["sec"].get("details", {}).get("parsed_forms") else "UNKNOWN")
        relationships = modules["supply_chain"].get("details", {}).get("relationships", [])
        suppliers = [r for r in relationships if r.get("relationship_type") == "supplier"]
        customers = [r for r in relationships if r.get("relationship_type") == "customer"]
        add("SUPPLY_CHAIN", "MEDIUM" if relationships and any(not r.get("verified") for r in relationships) else "LOW" if relationships else "UNKNOWN", f"{len(relationships)} 条关系，{sum(bool(r.get('verified')) for r in relationships)} 条已验证。" if relationships else "Insufficient Evidence", ["supply_chain"] if relationships else [], .72 if relationships else .1)
        add("CUSTOMER_CONCENTRATION", "MEDIUM" if len(customers) == 1 else "LOW" if len(customers) >= 2 else "UNKNOWN", f"当前有 {len(customers)} 条可识别客户关系；该值不是收入集中度百分比。" if customers else "Insufficient Evidence", ["supply_chain"] if customers else [], .58 if customers else .1)
        guidance = modules["sec"].get("details", {}).get("guidance", [])
        lowered = [g for g in guidance if g.get("status") in {"LOWERED", "WITHDRAWN"}]
        add("MANAGEMENT", "HIGH" if lowered else "LOW" if guidance else "UNKNOWN", f"识别到 {len(lowered)} 条下调或撤回指引。" if lowered else "存在管理层指引证据，未检测到下调或撤回。" if guidance else "Insufficient Evidence", ["sec_filings"] if guidance else [], .68 if guidance else .1, "RISING" if lowered else "STABLE" if guidance else "UNKNOWN")
        vix = _finite(modules["macro"].get("metrics", {}).get("vix"))
        add("MACRO", "HIGH" if vix is not None and vix >= 30 else "MEDIUM" if vix is not None and vix >= 20 else "LOW" if vix is not None else "UNKNOWN", f"VIX 当前为 {vix:.2f}。" if vix is not None else "Insufficient Evidence", ["macro_snapshot"] if vix is not None else [], .82 if vix is not None else .1)
        catalysts = modules["catalyst"].get("details", {}).get("timeline", [])
        negative = [c for c in catalysts if c.get("direction") == "negative"]
        add("EVENT", "HIGH" if negative else "MEDIUM" if catalysts else "UNKNOWN", f"未来/近期事件 {len(catalysts)} 条，其中负向 {len(negative)} 条。" if catalysts else "Insufficient Evidence", sorted({c.get("source_id") for c in catalysts if c.get("source_id")}), .68 if catalysts else .1, "RISING" if negative else "STABLE" if catalysts else "UNKNOWN")
        known = [r for r in risks if r["level"] != "UNKNOWN"]
        score = _clamp(sum({"LOW": 85, "MEDIUM": 55, "HIGH": 20}[r["level"]] for r in known) / len(known)) if known else None
        return self._module(d.ticker, "risk", "COMPLETED" if len(known) >= 4 else "PARTIAL", f"8 类风险中 {len(known)} 类有证据；缺失项明确标记 UNKNOWN。", score=score, risks=risks, sources=sorted({source for risk in risks for source in risk["evidence"]}), confidence=.78 if len(known) >= 4 else .45, details={"radar": risks, "risk_factor_changes": risk_changes})

    def _aggregate(self, d: StockResearchDataset, modules: dict[str, dict], provider_health: list[dict] | None = None) -> dict:
        core = [modules[name] for name in ("fundamental", "valuation", "earnings")]
        if all(item["status"] in {"FAILED", "SKIPPED"} for item in core):
            status = "FAILED"
        elif any(item["status"] in {"FAILED", "PARTIAL_ERROR"} for item in modules.values()):
            status = "PARTIAL_ERROR"
        elif any(item["status"] == "PARTIAL_DATA" for item in modules.values()):
            status = "PARTIAL_DATA"
        else:
            status = "COMPLETED"
        f = modules["fundamental"]
        v = modules["valuation"]
        e = modules["expectations"]
        risk = modules["risk"]
        company = f["details"].get("company", {})
        run_id = f"research-{d.ticker}-{int(datetime.now(timezone.utc).timestamp())}"
        intelligence = build_intelligence(d.ticker, modules, company, sources=d.sources, snapshot_id=run_id, provider_health=provider_health or [])
        scores = {"fundamental": f["score"], "growth": f["metrics"].get("growth_score"), "profitability": f["metrics"].get("profitability_score"), "financial_health": f["metrics"].get("financial_health_score"), "valuation": v["score"], "earnings_momentum": e["score"], "institutional": modules["institutional"]["score"], "technical": modules["technical"]["score"], "catalyst": modules["catalyst"]["score"], "sector_aware": intelligence["scoring_profile"].get("overall")}
        known_risks = [item for item in risk["details"].get("radar", []) if item["level"] != "UNKNOWN"][:5]
        high_count = sum(item["level"] == "HIGH" for item in known_risks)
        risk_level = "High" if high_count >= 2 else "Medium" if high_count or any(item["level"] == "MEDIUM" for item in known_risks) else "Low" if known_risks else "Unknown"
        return {"run_id": f"research-{d.ticker}-{int(datetime.now(timezone.utc).timestamp())}", "engine_version": ENGINE_VERSION,
                "research_engine_feature_freeze": FEATURE_FREEZE, "release_status": "RESEARCH ENGINE V1.0 FEATURE FREEZE",
                "status": status, "ticker": d.ticker, "from_cache": False, "generated_at": _now(), "dataset": {"ticker": d.ticker, "fetched_at": d.fetched_at, "errors": d.errors, "available": {"company": bool(d.company), "metrics": len(d.metrics), "earnings": len(d.earnings), "expectations": bool(d.expectations), "filings": len(d.filings), "institutional_holdings": len(d.institutional.get("institutional_holdings", [])), "insiders": len(d.insiders), "news": len(d.news), "macro": bool(d.macro.get("snapshot")), "supply_chain_relationships": len(modules["supply_chain"].get("details", {}).get("relationships", [])), "prices": len(d.prices)}}, "module_status": {name: result["status"] for name, result in modules.items()}, "modules": modules, "scores": scores, "risk_level": risk_level, "investment_thesis": intelligence["core_thesis"], "core_thesis": intelligence["core_thesis"], "why_now": intelligence["why_now"], "bull_case": intelligence["scenarios"]["BULL"]["narrative"], "base_case": intelligence["scenarios"]["BASE"]["narrative"], "bear_case": intelligence["scenarios"]["BEAR"]["narrative"], "key_catalysts": intelligence["key_catalysts"], "key_risks": intelligence["key_risks_v2"] or known_risks, "thesis_invalidation": intelligence["thesis_invalidation"], "sources": d.sources, **intelligence}


_PEERS = {
    "NVDA": ["AMD", "AVGO", "MRVL", "INTC", "QCOM"], "AMD": ["NVDA", "AVGO", "INTC"],
    "AAPL": ["MSFT", "GOOGL", "DELL", "HPQ"], "JPM": ["BAC", "WFC", "C", "GS"],
    "XOM": ["CVX", "COP", "EOG", "SHEL"], "TSLA": ["GM", "F", "RIVN", "TM"],
}
_SECTOR_PEERS = {
    "technology": ["MSFT", "AAPL", "GOOGL", "ORCL"],
    "financial services": ["JPM", "BAC", "WFC", "C"],
    "energy": ["XOM", "CVX", "COP", "EOG"],
    "consumer cyclical": ["TSLA", "GM", "F", "AMZN"],
}
