"""Engine-shaped fixtures for the benchmark, synthesised offline.

The Research Engine is fetch + intelligence.  Fetching needs the
production-only ``v2.data`` package and provider keys; the intelligence layer
(``v2.research.intelligence.build_intelligence``) is pure and turns module
data into the evidence index, findings, confidence and limitations that the
V2 research adapter consumes.  Feeding it deterministic module profiles
therefore yields envelopes with production structure and ids, without a
network.  Market envelopes come from the real market adapter over a
deterministic price series.

Numbers are deliberately distinctive (V1's answer-key rule), so a later
answer key can quote them without a ``10%`` matching by accident.
"""

from __future__ import annotations

import hashlib
from datetime import date, datetime, timedelta
from functools import lru_cache
from types import SimpleNamespace
from typing import Any
from zoneinfo import ZoneInfo

from v2.agent_v2.adapters.market import register_market_capabilities
from v2.agent_v2.adapters.research import _FOCUS_MODULES, _envelope
from v2.agent_v2.catalog import default_catalog
from v2.agent_v2.execution import CapabilityRegistry, ExecutionContext
from v2.agent_v2.models import BudgetClass, NormalizedRequest, PlanTask, ToolEnvelope

_ET = ZoneInfo("America/New_York")
GENERATED_AT = "2026-09-05T00:00:00Z"
AS_OF = date(2026, 9, 4)

#: ticker -> sector, industry, revenue growth, gross margin, ROIC, P/E, EPS surprise,
#: insider sell value (90d), trend, day move, volume ratio, sector ETF
PROFILES: dict[str, tuple[Any, ...]] = {
    "NVDA": ("Technology", "Semiconductors", 0.553, 0.731, 0.746, 48.3, 0.056, 1.42e7, "Bullish", 0.0385, 1.42, "SMH"),
    "AMD": ("Technology", "Semiconductors", 0.184, 0.512, 0.078, 37.9, 0.029, 1.6e6, "Bearish", -0.0132, 1.08, "SMH"),
    "SMCI": ("Technology", "Computer Hardware", 0.427, 0.113, 0.219, 15.4, -0.236, 0.0, "Bearish", -0.054, 2.63, "SMH"),
    "AVGO": ("Technology", "Semiconductors", 0.221, 0.634, 0.164, 33.7, 0.03, 5.95e6, "Bullish", 0.0216, 1.31, "SMH"),
    "ARM": ("Technology", "Semiconductors", 0.312, 0.957, 0.091, 88.2, 0.083, 0.0, "Bullish", 0.0742, 3.41, "SMH"),
    "MSFT": ("Technology", "Software - Infrastructure", 0.152, 0.692, 0.281, 34.6, 0.041, 0.0, "Bullish", 0.0114, 1.02, "XLK"),
    "CRWD": ("Technology", "Software - Infrastructure", 0.289, 0.752, 0.052, 92.7, 0.062, 1.097e7, "Bearish", -0.021, 1.24, "XLK"),
    "PLTR": ("Technology", "Software - Infrastructure", 0.271, 0.803, 0.113, 214.5, 0.056, 2.3e6, "Bearish", -0.0518, 2.94, "XLK"),
    "AAPL": ("Technology", "Consumer Electronics", 0.081, 0.463, 0.482, 31.2, 0.024, 9.19e6, "Bullish", 0.0086, 0.93, "XLK"),
    "GOOGL": ("Communication Services", "Internet Content & Information", 0.127, 0.581, 0.263, 22.8, 0.047, 0.0, "Bullish", 0.0042, 0.81, "XLK"),
    "TSLA": ("Consumer Cyclical", "Auto Manufacturers", -0.043, 0.176, 0.061, 62.4, -0.018, 0.0, "Bearish", -0.0274, 1.63, "SPY"),
}
DEFAULT_PROFILE = ("Technology", "Software - Application", 0.104, 0.55, 0.12, 28.0, 0.01, 0.0, "Bullish", 0.005, 1.0, "SPY")

SOURCES = [
    {"id": "fd_metrics", "provider": "Financial Datasets", "title": "Financial Datasets fundamentals", "url": "", "published_at": "2026-06-30", "fetched_at": GENERATED_AT},
    {"id": "fd_earnings", "provider": "Financial Datasets", "title": "Financial Datasets earnings history", "url": "", "published_at": "2026-08-27", "fetched_at": GENERATED_AT},
    {"id": "fd_insiders", "provider": "Financial Datasets", "title": "Form 4 insider transactions", "url": "", "published_at": "2026-08-22", "fetched_at": GENERATED_AT},
    {"id": "yf_prices", "provider": "Yahoo Finance", "title": "Daily prices", "url": "", "published_at": "2026-09-04", "fetched_at": GENERATED_AT},
    {"id": "sec_filings", "provider": "SEC EDGAR", "title": "Recent 8-K and 10-Q filings", "url": "https://www.sec.gov/", "published_at": "2026-08-30", "fetched_at": GENERATED_AT},
    {"id": "supply_chain", "provider": "Supply chain graph", "title": "Verified supplier/customer relationships", "url": "", "published_at": "2026-07-15", "fetched_at": GENERATED_AT},
    {"id": "yf_calendar", "provider": "Yahoo Finance", "title": "Earnings calendar", "url": "", "published_at": "2026-09-04", "fetched_at": GENERATED_AT},
]


def _module(status: str = "COMPLETED", confidence: float = 0.9, completeness: float = 1.0, *, metrics: dict | None = None, details: dict | None = None, sources: list[str] | None = None, score: int = 70) -> dict[str, Any]:
    sources = sources or []
    return {
        "status": status,
        "confidence": confidence,
        "completeness": completeness,
        "metrics": metrics or {},
        "details": details or {},
        "sources": sources,
        "data_sources_used": sources,
        "score": score,
        "source_count": len(sources),
        "verified_source_count": len(sources),
        "missing_fields": [],
        "generated_at": GENERATED_AT,
    }


def modules_for(ticker: str) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Deterministic module data for one ticker, in the engine's own shape."""

    sector, industry, growth, margin, roic, pe, surprise, insider_sell, trend, _, _, _ = PROFILES.get(ticker, DEFAULT_PROFILE)
    semis = industry == "Semiconductors"
    relationships = [{"target_company": "TSM", "relationship_type": "supplier", "verified": True}] if semis else []
    if ticker == "NVDA":
        relationships += [{"target_company": "ASML", "relationship_type": "supplier", "verified": True}, {"target_company": "MSFT", "relationship_type": "customer", "verified": True}]
    radar = [{"name": "SUPPLY_CHAIN", "level": "HIGH", "reason": "Foundry concentration and export restrictions", "evidence": ["supply_chain", "sec_filings"], "confidence": 0.85, "trend": "RISING"}] if semis else []
    radar.append({"name": "VALUATION", "level": "HIGH" if pe >= 40 else "MEDIUM", "reason": f"P/E {pe:.1f}", "evidence": ["fd_metrics"], "confidence": 0.85, "trend": "STABLE"})
    guidance = [{"guidance_type": "FORMAL_GUIDANCE", "status": "RAISED" if growth > 0.2 else "UNCHANGED", "evidence_text": f"{ticker} guided next-quarter revenue growth of {growth * 100:.1f}%."}]
    modules = {
        "fundamental": _module(metrics={"revenue_growth": growth, "gross_margin": margin, "roic": roic, "roe": round(roic * 0.8 + 0.05, 3), "growth_score": min(95, int(50 + growth * 100)), "profitability_score": int(margin * 100), "financial_health_score": 65}, details={"capital_allocation": {"assessment": "SHAREHOLDER_RETURN" if ticker in {"AAPL", "MSFT"} else "GROWTH_INVESTMENT", "research_and_development": 1, "stock_based_compensation": 1}}, sources=["fd_metrics"], score=int(55 + growth * 60)),
        "valuation": _module(metrics={"pe_ttm": pe, "ps_ttm": round(pe / 4.7, 2)}, details={"peer_comparison": [{"ticker": "MSFT", "sector": "Technology", "industry": "Software - Infrastructure", "revenue_growth": 0.152, "operating_margin": 0.447}, {"ticker": "AMD", "sector": "Technology", "industry": "Semiconductors", "revenue_growth": 0.184, "operating_margin": 0.094}]}, sources=["fd_metrics"], score=max(20, int(90 - pe))),
        "earnings": _module(metrics={"latest_eps_surprise": surprise, "next_earnings_date": "2026-11-19" if ticker == "NVDA" else "2026-10-21"}, sources=["fd_earnings"], score=int(60 + surprise * 300)),
        "expectations": _module("PARTIAL", 0.4, 0.5, metrics={"revision_trend": None}, sources=["fd_earnings"], score=50),
        "institutional": _module(metrics={"insider_buy_value_90d": 0, "insider_sell_value_90d": insider_sell, "institutional_ownership": round(0.55 + (hash(ticker) % 30) / 100, 2)}, sources=["fd_insiders"], score=55),
        "technical": _module(metrics={"trend_state": trend, "timeframe": "1d", "rsi_14": 63.7 if trend == "Bullish" else 38.9}, sources=["yf_prices"], score=70 if trend == "Bullish" else 40),
        "catalyst": _module(details={"timeline": [{"title": f"{ticker} earnings", "description": "Scheduled earnings event", "direction": "neutral", "impact": "Very High", "item_type": "EVENT", "source_id": "yf_calendar"}]}, sources=["yf_calendar"], score=60),
        "macro": _module(metrics={"vix": 18.4, "wti_crude": 71.6}, sources=["macro_snapshot"], score=60),
        "sec": _module(details={"guidance": guidance, "risk_factor_changes": []}, sources=["sec_filings"], score=65),
        "supply_chain": _module(details={"relationships": relationships}, sources=["supply_chain"] if relationships else [], completeness=1.0 if relationships else 0.2, score=65),
        "risk": _module(details={"radar": radar}, sources=["fd_metrics", "sec_filings"], score=max(20, 90 - int(pe))),
    }
    company = {"name": ticker, "ticker": ticker, "sector": sector, "industry": industry}
    return modules, company


def _build_intelligence():
    from v2.agent_v2.eval._data_shim import install

    install()
    from v2.research.intelligence import build_intelligence

    return build_intelligence


@lru_cache(maxsize=256)
def synthesize_research_result(ticker: str, focus: str = "overview") -> dict[str, Any]:
    """An engine-shaped result dict for one ticker and focus, built offline."""

    ticker = ticker.upper()
    modules, company = modules_for(ticker)
    requested = _FOCUS_MODULES.get(focus) or list(modules)
    selected = {name: modules[name] for name in requested if name in modules}
    run_id = f"synthetic-{ticker.lower()}-{focus}-{hashlib.sha1(ticker.encode()).hexdigest()[:6]}"
    result = _build_intelligence()(ticker, selected, company, sources=SOURCES, snapshot_id=run_id)
    levels = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
    radar = modules["risk"]["details"]["radar"] if "risk" in selected else []
    risk_level = max((row["level"] for row in radar), key=lambda level: levels.get(level, 0), default="MEDIUM")
    return {
        **result,
        "ticker": ticker,
        "status": "COMPLETED" if all(module["status"] == "COMPLETED" for module in selected.values()) else "PARTIAL",
        "run_id": run_id,
        "generated_at": GENERATED_AT,
        "from_cache": False,
        "scores": {name: module["score"] for name, module in selected.items()},
        "risk_level": risk_level,
        "requested_modules": list(selected),
        "module_status": {name: module["status"] for name, module in selected.items()},
        "modules": selected,
        "sources": SOURCES,
    }


def synthesize_research_envelope(ticker: str, focus: str = "overview") -> ToolEnvelope:
    return _envelope(synthesize_research_result(ticker, focus), "research.stock")


class DeterministicPrices:
    """A price series seeded by ticker so the market adapter has 430 days of history."""

    def get_prices(self, ticker: str, start: str, end: str):
        seed = int(hashlib.sha1(ticker.upper().encode()).hexdigest(), 16)
        base = 40.0 + seed % 400
        _, _, _, _, _, _, _, _, trend, day_move, volume_ratio, _ = PROFILES.get(ticker.upper(), DEFAULT_PROFILE)
        drift = 0.0012 if trend == "Bullish" else -0.0008
        rows = []
        price = base
        first = AS_OF - timedelta(days=430)
        days = 0
        current = first
        while current <= AS_OF:
            if current.weekday() < 5:
                wobble = ((seed >> (days % 20)) % 7 - 3) / 1000
                price = price * (1 + drift + wobble)
                if current == AS_OF:
                    price = rows[-1].close * (1 + day_move) if rows else price
                volume = 1_000_000 + (seed % 900_000) + days * 500
                if current == AS_OF:
                    volume = int((1_000_000 + (seed % 900_000)) * volume_ratio)
                rows.append(SimpleNamespace(time=current.isoformat(), close=round(price, 2), volume=volume))
                days += 1
            current += timedelta(days=1)
        return rows


def synthesize_anomaly(ticker: str):
    """A move-attribution result in the shape ``market.explain_move`` consumes."""

    ticker = ticker.upper()
    _, _, _, _, _, _, _, _, _, day_move, volume_ratio, sector_etf = PROFILES.get(ticker, DEFAULT_PROFILE)
    series = DeterministicPrices().get_prices(ticker, "", "")
    price = float(series[-1].close)  # the same close the performance adapter reports
    reasons = []
    if abs(day_move) >= 0.05:
        reasons.append(SimpleNamespace(text=f"{ticker} 披露了一项重大合同", confidence="高", note="权威媒体同日报道"))
    reasons.append(SimpleNamespace(text="行业轮动", confidence="中", note=""))
    reasons.append(SimpleNamespace(text="期权市场波动", confidence="低", note="缺少直接证据"))
    sector_return = 0.029 if sector_etf == "SMH" else 0.006 if sector_etf == "XLK" else 0.004
    return SimpleNamespace(
        date=AS_OF.isoformat(),
        price=price,
        price_change_pct=day_move,
        volume_today=int(1_500_000 * volume_ratio),
        volume_avg_30d=1_500_000.0,
        volume_ratio=volume_ratio,
        sector_etf=sector_etf,
        sector_return_1d=sector_return,
        relative_1d_pp=round(day_move - sector_return, 4),
        contrarian=(day_move < 0) != (sector_return < 0),
        reasons=reasons,
        sources=[{"title": "Same-day report", "url": "https://example.test/report"}],
        next_steps=[],
        filtered_count=2,
    )


def synthesize_market_registry(registry: CapabilityRegistry | None = None) -> CapabilityRegistry:
    """Register the real market adapters over deterministic sources (closed session)."""

    registry = registry or CapabilityRegistry(default_catalog())
    now = datetime(2026, 9, 4, 18, 0, tzinfo=_ET)
    register_market_capabilities(registry, price_source_factory=DeterministicPrices, move_provider=synthesize_anomaly, now_factory=lambda: now)
    return registry


def synthesize_market_envelope(capability: str, ticker: str) -> ToolEnvelope:
    registry = synthesize_market_registry()
    context = ExecutionContext("synthetic", NormalizedRequest("q", "q"), BudgetClass.FOCUSED)
    return registry.execute(PlanTask("m", capability, {"ticker": ticker.upper()}), context)
