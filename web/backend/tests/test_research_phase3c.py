from copy import deepcopy

from v2.research.depth import _guidance_from_text
from v2.research.intelligence import build_intelligence
from v2.research.store import ResearchStore
from v2.research.provider_health import ProviderHealthService, classify_provider_error
from v2.research.engine import ResearchEngine, StockResearchDataset
from v2.data.client import FDClient, ProviderRequestError


def module(status="COMPLETED", confidence=.9, completeness=1.0, *, metrics=None, details=None, sources=None, score=70):
    return {"status": status, "confidence": confidence, "completeness": completeness, "metrics": metrics or {},
            "details": details or {}, "sources": sources or [], "data_sources_used": sources or [], "score": score,
            "source_count": len(sources or []), "verified_source_count": len(sources or []), "missing_fields": []}


def modules_for(ticker):
    profiles = {
        "AAPL": ("Technology", "Consumer Electronics", .08, .46, .48, 31),
        "NVDA": ("Technology", "Semiconductors", .55, .73, .75, 48),
        "JPM": ("Financial Services", "Banks - Diversified", .06, None, .17, 14),
        "XOM": ("Energy", "Oil & Gas Integrated", .03, .34, .12, 16),
        "TSLA": ("Consumer Cyclical", "Auto Manufacturers", -.04, .18, .06, 62),
    }
    sector, industry, growth, margin, roic, pe = profiles[ticker]
    return {
        "fundamental": module(metrics={"revenue_growth": growth, "gross_margin": margin, "roic": roic, "roe": .18, "growth_score": 70, "profitability_score": 75, "financial_health_score": 65}, details={"capital_allocation": {"assessment": "SHAREHOLDER_RETURN" if ticker in {"AAPL", "XOM"} else "INSUFFICIENT_EVIDENCE", "research_and_development": 1, "stock_based_compensation": 1}}, sources=["fd_metrics"]),
        "valuation": module(metrics={"pe_ttm": pe}, details={"peer_comparison": [{"ticker": "MSFT", "sector": "Technology", "industry": "Software", "revenue_growth": .1, "operating_margin": .4}, {"ticker": "HPQ", "sector": "Technology", "industry": "Computer Hardware", "revenue_growth": .02, "operating_margin": .08}]}, sources=["fd_metrics"]),
        "earnings": module(metrics={"latest_eps_surprise": .05}, sources=["fd_earnings"]),
        "expectations": module("PARTIAL", .4, .5, metrics={"revision_trend": None}, sources=["fd_earnings"]),
        "institutional": module(metrics={"insider_buy_value_90d": 0, "insider_sell_value_90d": 1}, sources=["fd_insiders"]),
        "technical": module(metrics={"trend_state": "Bearish" if ticker == "TSLA" else "Bullish", "timeframe": "1d"}, sources=["yf_prices"]),
        "catalyst": module(details={"timeline": [{"title": f"{ticker} earnings", "description": "Scheduled earnings event", "direction": "neutral", "impact": "Very High", "item_type": "EVENT", "source_id": "yf_calendar"}]}, sources=["yf_calendar"]),
        "macro": module(metrics={"vix": 18, "wti_crude": 76}, sources=["macro_snapshot"]),
        "sec": module(details={"guidance": [], "risk_factor_changes": []}, sources=["sec_filings"]),
        "supply_chain": module(details={"relationships": ([{"target_company": "TSM", "relationship_type": "supplier", "verified": True}] if ticker == "NVDA" else [])}, sources=["supply_chain"] if ticker == "NVDA" else []),
        "risk": module(details={"radar": ([{"name": "SUPPLY_CHAIN", "level": "HIGH", "reason": "Foundry concentration and export restrictions", "evidence": ["supply_chain", "sec_filings"], "confidence": .85, "trend": "RISING"}] if ticker == "NVDA" else [{"name": "VALUATION", "level": "HIGH" if pe >= 40 else "MEDIUM", "reason": f"P/E {pe}", "evidence": ["fd_metrics"], "confidence": .85, "trend": "STABLE"}])}, sources=["fd_metrics", "sec_filings"]),
    }, {"name": ticker, "ticker": ticker, "sector": sector, "industry": industry}


def test_five_company_theses_and_sector_profiles_are_specific():
    results = {}
    for ticker in ("AAPL", "NVDA", "JPM", "XOM", "TSLA"):
        modules, company = modules_for(ticker)
        results[ticker] = build_intelligence(ticker, modules, company)
    assert len({item["core_thesis"] for item in results.values()}) == 5
    assert results["JPM"]["scoring_profile"]["profile"] == "FINANCIALS"
    assert "debt_to_equity" in results["JPM"]["scoring_profile"]["excluded_metrics"]
    assert results["XOM"]["scoring_profile"]["profile"] == "ENERGY"
    assert results["AAPL"]["scoring_profile"]["profile"] == "TECHNOLOGY"
    for result in results.values():
        assert result["scenarios"]["BULL"]["narrative"].startswith(result["research_findings"][0]["ticker"])
        assert all(finding["evidence_ids"] for finding in result["research_findings"])
    nvda_risks = [f for f in results["NVDA"]["research_findings"] if f["category"] == "risk"]
    assert any("export restrictions" in f["claim"] for f in nvda_risks)
    assert any("supply_chain" in f["source_ids"] for f in nvda_risks)


def test_peer_relevance_guidance_and_missing_expectations():
    modules, company = modules_for("AAPL")
    result = build_intelligence("AAPL", modules, company)
    peers = {row["ticker"]: row for row in result["peer_relevance"]}
    assert peers["MSFT"]["relevance_score"] >= peers["HPQ"]["relevance_score"]
    missing = [f for f in result["research_findings"] if f["title"] == "Expectations history unavailable"][0]
    assert missing["direction"] == "NEUTRAL" and missing["confidence"] <= .3
    rows = _guidance_from_text("We expect revenue between $10 billion and $11 billion in fiscal 2027. We believe our culture is strong.", "2026-09-01", "sec")
    assert rows[0]["guidance_type"] in {"FORMAL_GUIDANCE", "QUANTITATIVE_OUTLOOK"}
    assert rows[0]["value"] is None and rows[0]["range"] is None
    assert rows[1]["guidance_type"] == "MANAGEMENT_COMMENTARY"


def test_distinct_guidance_claims_receive_distinct_evidence_ids():
    modules, company = modules_for("NVDA")
    modules["sec"]["details"]["guidance"] = [
        {"guidance_type": "MANAGEMENT_COMMENTARY", "status": "UNCHANGED", "evidence_text": "First filing excerpt."},
        {"guidance_type": "MANAGEMENT_COMMENTARY", "status": "UNCHANGED", "evidence_text": "Second filing excerpt."},
    ]
    result = build_intelligence("NVDA", modules, company)
    guidance = [row for row in result["research_findings"] if row["module"] == "sec" and row["category"] == "guidance"]
    evidence_ids = [row["evidence_ids"][0] for row in guidance]
    assert len(guidance) == 2
    assert len(set(evidence_ids)) == 2
    assert set(evidence_ids) <= {row["id"] for row in result["evidence_index"]}


def test_low_completeness_conflicts_and_traceability():
    modules, company = modules_for("TSLA")
    modules["fundamental"]["completeness"] = .15
    modules["valuation"]["completeness"] = .15
    modules["technical"]["metrics"]["trend_state"] = "Bullish"
    result = build_intelligence("TSLA", modules, company)
    assert result["research_confidence"]["score"] < 75
    assert result["quality_gate"]["status"] == "LIMITED_RESEARCH_THESIS"
    assert result["conflicts"]
    evidence_ids = {row["id"] for row in result["evidence_index"]}
    assert all(set(finding["evidence_ids"]) <= evidence_ids for finding in result["research_findings"])


def test_what_changed_v2_detects_new_driver_and_risk(tmp_path):
    store = ResearchStore(tmp_path / "research.db")
    modules, company = modules_for("AAPL")
    old = {"run_id": "old", "ticker": "AAPL", "status": "COMPLETED", "generated_at": "2026-09-01T00:00:00+00:00", "modules": modules, **build_intelligence("AAPL", deepcopy(modules), company)}
    changed = deepcopy(modules)
    changed["risk"]["details"]["radar"].append({"name": "REGULATORY", "level": "HIGH", "reason": "New filing disclosure", "evidence": ["sec_filings"], "confidence": .8, "trend": "RISING"})
    changed["fundamental"]["metrics"]["revenue_growth"] = -.1
    new = {"run_id": "new", "ticker": "AAPL", "status": "COMPLETED", "generated_at": "2026-09-02T00:00:00+00:00", "modules": changed, **build_intelligence("AAPL", changed, company)}
    store.save_snapshot("old", old)
    store.save_snapshot("new", new)
    comparison = store.compare_latest("AAPL")
    assert comparison["driver_changes"]["negative"]["added"]
    assert comparison["risk_changes"]["findings"]["added"]


def test_provider_error_semantics_and_secret_safe_health(monkeypatch):
    class Response:
        status_code = 401
        content = b"denied"
        text = "denied"
    monkeypatch.setattr("requests.request", lambda *args, **kwargs: Response())
    monkeypatch.setenv("FINANCIAL_DATASETS_API_KEY", "never-return-this-secret")
    row = ProviderHealthService().check("Financial Datasets")
    assert row["status"] == "AUTH_ERROR" and row["configured"] and row["reachable"]
    assert "never-return-this-secret" not in str(row)
    client = FDClient(api_key="never-return-this-secret")
    monkeypatch.setattr(client._session, "request", lambda *args, **kwargs: Response())
    try:
        client.get_company_facts("AAPL")
        assert False, "AUTH_ERROR must not silently become empty data"
    except ProviderRequestError as exc:
        assert exc.error_type == "AUTH_ERROR" and "never-return-this-secret" not in str(exc)
    assert classify_provider_error(status_code=429) == "RATE_LIMITED"


def test_empty_data_semantics_for_missing_ticker(monkeypatch):
    class Response:
        status_code = 404
        content = b""
        text = ""

    client = FDClient(api_key="never-return-this-secret")
    monkeypatch.setattr(client._session, "request", lambda *args, **kwargs: Response())
    try:
        client.get_company_facts("NOTREAL")
        assert False, "404 must retain EMPTY_DATA semantics"
    except ProviderRequestError as exc:
        assert exc.error_type == "EMPTY_DATA"

    engine = object.__new__(ResearchEngine)
    dataset = StockResearchDataset(ticker="JPM")
    dataset.errors["earnings"] = "earnings provider returned no data"
    dataset.provider_error_types["earnings"] = "EMPTY_DATA"
    result = engine._finalize_module(engine._module("JPM", "earnings", "FAILED", "no history", metrics={"latest_eps_surprise": None}, error=dataset.errors["earnings"]), dataset)
    assert result["status"] == "PARTIAL_DATA"
    assert result["completeness"] == 0.0
    assert [item["type"] for item in result["provider_errors"]] == ["EMPTY_DATA"]


def test_run_metadata_prefers_typed_provider_error(tmp_path):
    store = ResearchStore(tmp_path / "research.db")
    store.create_run("empty-run", "JPM", ["earnings"], "FULL")
    store.update_module("empty-run", "earnings", "PARTIAL_DATA", {
        "error": "earnings provider returned no data",
        "errors": ["earnings provider returned no data"],
        "provider_errors": [{"provider": "earnings", "type": "EMPTY_DATA", "message": "earnings provider returned no data"}],
    })
    row = store.get_run("empty-run")["module_runs"][0]
    assert row["error_type"] == "EMPTY_DATA"


def test_provider_health_penalty_provenance_and_capability_audits():
    modules, company = modules_for("AAPL")
    sources = [{"id": "fd_metrics", "provider": "Financial Datasets", "published_at": "2026-06-30", "fetched_at": "2026-09-05T00:00:00Z"}]
    result = build_intelligence("AAPL", modules, company, sources=sources, snapshot_id="run-1", provider_health=[{"provider": "Financial Datasets", "status": "AUTH_ERROR"}])
    assert result["research_confidence"]["modules"]["fundamental"]["score"] <= 34
    evidence = next(row for row in result["evidence_index"] if "fd_metrics" in row["source_ids"])
    assert evidence["data_period"] == "2026-06-30" and evidence["fetched_at"] == "2026-09-05T00:00:00Z"
    assert evidence["snapshot_id"] == "run-1" and evidence["provider"] == ["Financial Datasets"]
    jpm_modules, jpm_company = modules_for("JPM")
    xom_modules, xom_company = modules_for("XOM")
    assert build_intelligence("JPM", jpm_modules, jpm_company)["data_capability_audit"]["matrix"]
    assert build_intelligence("XOM", xom_modules, xom_company)["data_capability_audit"]["matrix"]
