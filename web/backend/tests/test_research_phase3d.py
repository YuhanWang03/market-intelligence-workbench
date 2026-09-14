from __future__ import annotations

from copy import deepcopy

from v2.research.depth import _industry_evidence_from_text
from v2.research.industry import build_industry_metrics, profile_for
from v2.research.intelligence import build_intelligence


def module(*, metrics=None, details=None, sources=None, score=70, completeness=1.0, confidence=.9):
    sources = sources or []
    return {"status": "COMPLETED", "metrics": metrics or {}, "details": details or {}, "sources": sources,
            "data_sources_used": sources, "score": score, "completeness": completeness, "confidence": confidence,
            "source_count": len(sources), "verified_source_count": len(sources), "missing_fields": []}


def modules() -> dict:
    return {
        "fundamental": module(metrics={"revenue_growth": .12, "gross_margin": .42, "operating_margin": .20, "roic": .16, "roe": .17,
                                               "growth_score": 72, "profitability_score": 74, "financial_health_score": 68},
                              details={"capital_allocation": {}, "quarterly_trends": [{"capital_expenditure": 2_000_000_000, "free_cash_flow": 4_000_000_000}]}, sources=["fd_metrics", "fd_earnings"]),
        "valuation": module(metrics={"pe_ttm": 32}, details={"peer_comparison": []}, sources=["fd_metrics"]),
        "earnings": module(metrics={"latest_eps_surprise": .04}, sources=["fd_earnings"]),
        "expectations": module(metrics={"revision_trend": None}, completeness=.25, confidence=.4),
        "institutional": module(metrics={"insider_buy_value_90d": 0, "insider_sell_value_90d": 1}, sources=["fd_insiders"]),
        "fund_flow": module(metrics={}, sources=["yf_prices"]),
        "technical": module(metrics={"trend_state": "Bullish", "timeframe": "1d"}, sources=["yf_prices"]),
        "catalyst": module(details={"timeline": [{"title": "Quarterly earnings", "description": "Known scheduled event", "direction": "neutral", "impact": "High", "item_type": "EVENT", "source_id": "sec_filings", "confidence": .8}]}, sources=["sec_filings"]),
        "sec": module(metrics={"filing_count": 2}, details={"filings": [{"form": "10-Q"}], "sec_findings": []}, sources=["sec_filings"]),
        "macro": module(metrics={"vix": 18, "wti_crude": 71}, sources=["macro_snapshot"]),
        "supply_chain": module(metrics={"relationships": 0}, details={"relationships": []}),
        "risk": module(details={"radar": [{"name": "VALUATION", "level": "HIGH", "reason": "Valuation leaves less room for execution error.", "evidence": ["fd_metrics"], "confidence": .8, "trend": "STABLE"}]}, sources=["fd_metrics"]),
    }


def test_profile_regression_set():
    expected = {
        "AAPL": "TECHNOLOGY", "NVDA": "TECHNOLOGY", "MSFT": "TECHNOLOGY",
        "JPM": "FINANCIALS", "BAC": "FINANCIALS", "XOM": "ENERGY", "CVX": "ENERGY",
        "TSLA": "AUTOMOTIVE", "GM": "AUTOMOTIVE", "F": "AUTOMOTIVE",
        "WMT": "GENERAL", "KO": "GENERAL", "CAT": "GENERAL",
    }
    descriptors = {
        "TECHNOLOGY": ("Technology", "Software"), "FINANCIALS": ("Financial Services", "Banks"),
        "ENERGY": ("Energy", "Oil & Gas"), "AUTOMOTIVE": ("Consumer Cyclical", "Auto Manufacturers"),
        "GENERAL": ("Industrials", "Industrial Distribution"),
    }
    for ticker, profile in expected.items():
        sector, industry = descriptors[profile]
        assert profile_for(ticker, sector, industry) == profile


def test_financials_industry_metrics_are_separate_and_no_generic_leverage():
    data = modules()
    data["sec"]["details"]["sec_findings"] = [{"evidence_text": "Net interest margin was 2.91%. The CET1 capital ratio was 15.3%. The efficiency ratio was 54.0%."}]
    result = build_intelligence("JPM", data, {"name": "JPMorgan Chase", "sector": "Financial Services", "industry": "Banks"})
    industry = result["industry_metrics"]
    assert industry["profile"] == "FINANCIALS"
    assert industry["metrics"]["net_interest_margin"]["value"] == .0291
    assert industry["metrics"]["cet1"]["value"] == .153
    assert "current_ratio" not in industry["metrics"]
    assert "debt_to_equity" in result["scoring_profile"]["excluded_metrics"]


def test_energy_metrics_link_macro_and_keep_breakeven_na():
    result = build_intelligence("XOM", modules(), {"name": "Exxon Mobil", "sector": "Energy", "industry": "Oil & Gas Integrated"})
    industry = result["industry_metrics"]
    assert industry["profile"] == "ENERGY"
    assert industry["metrics"]["free_cash_flow"]["value"] == 4_000_000_000
    assert industry["metrics"]["wti_crude"]["value"] == 71
    assert industry["metrics"]["breakeven"]["value"] is None


def test_automotive_profile_dynamic_energy_requirement():
    tsla = build_industry_metrics("TSLA", "AUTOMOTIVE", modules())
    gm = build_industry_metrics("GM", "AUTOMOTIVE", modules())
    assert "energy_revenue" not in tsla["required_metrics"]
    assert "energy_revenue" in gm["optional_metrics"]
    assert tsla["required_metrics"] == gm["required_metrics"]


def test_thesis_v3_debate_tension_and_factuality_are_evidence_bound():
    result = build_intelligence("TSLA", deepcopy(modules()), {"name": "Tesla", "sector": "Consumer Cyclical", "industry": "Auto Manufacturers"})
    assert result["industry_context"]["profile"] == "AUTOMOTIVE"
    assert result["main_tension"]["statement"]
    debate = result["investment_debate"]
    assert len(debate["what_bulls_believe"]) <= 5
    assert len(debate["what_bears_believe"]) <= 5
    assert len(debate["what_matters_most"]) <= 5
    narrative = result["investment_thesis_v3"]
    assert narrative["factuality"]["status"] == "PASSED"
    assert narrative["llm_used"] is False
    assert narrative["research_confidence"]["score"] == result["research_confidence"]["score"]
    assert result["confidence_contributors"]


def test_sec_industry_evidence_retains_only_quantified_driver_sentences():
    text = (
        "General strategy remains unchanged. "
        "Net interest margin was 2.91% for the quarter. "
        "Management discussed vehicle deliveries without providing a number. "
        "Automotive revenues were $21.4 billion during the period."
    )
    rows = _industry_evidence_from_text(text, "2026-06-30", "https://www.sec.gov/example")
    assert [row["evidence_text"] for row in rows] == [
        "Net interest margin was 2.91% for the quarter.",
        "Automotive revenues were $21.4 billion during the period.",
    ]
