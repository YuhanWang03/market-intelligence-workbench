from __future__ import annotations

from v2.research.industry import build_industry_metrics, classify_industry_risk
from v2.research.intelligence import detect_generic_fallback, ga_coverage_gate, research_support_tier
import pytest

from v2.research.narrative import build_main_tension


def _modules(texts: list[str], *, roe: float | None = None) -> dict:
    return {
        "fundamental": {
            "metrics": {"roe": roe},
            "details": {"quarterly_trends": [{}]},
        },
        "sec": {
            "details": {
                "sec_findings": [],
                "industry_evidence": [
                    {"evidence_text": text, "filing_date": "2026-06-30", "source_url": "https://sec.example/filing"}
                    for text in texts
                ],
            }
        },
        "macro": {"metrics": {}},
    }


def test_energy_semantic_extraction_and_unit_normalization():
    modules = _modules([
        "Production of 4.5 million oil-equivalent barrels per day was reported for the quarter.",
        "Crude oil production was 2.6 million barrels per day.",
        "Natural gas production was 8.1 billion cubic feet per day.",
        "Upstream earnings were $5.4 billion.",
        "Downstream earnings were $1.2 billion.",
        "Cash capex was $6.8 billion.",
        "Adjusted free cash flow was $12.0 billion.",
    ])
    result = build_industry_metrics("ANY", "ENERGY", modules)
    assert result["available_count"] == 5
    assert result["optional_available_count"] >= 1
    assert result["metrics"]["production"]["value"] == 4_500_000
    assert result["metrics"]["production"]["unit"] == "BOE/day"
    assert result["metrics"]["gas_production"]["value"] == 8_100
    assert result["metrics"]["gas_production"]["unit"] == "MMcf/day"
    assert result["metrics"]["capex"]["value"] == 6_800_000_000
    assert result["metrics"]["breakeven"]["value"] is None


def test_financial_aliases_cover_bank_metrics_without_ticker_rules():
    modules = _modules([
        "Net interest yield was 2.10% for the quarter.",
        "The Common Equity Tier 1 capital ratio was 11.9%.",
        "Loans and leases increased 3.0% year over year.",
        "Deposits increased 2.0% year over year.",
        "Provision for credit losses was $1.5 billion.",
        "Net charge-offs were 0.62% of average loans.",
        "The nonperforming loans ratio was 0.50%.",
        "The efficiency ratio was 64.0%.",
    ], roe=.13)
    result = build_industry_metrics("ANY", "FINANCIALS", modules)
    assert result["available_count"] == 6
    assert result["optional_available_count"] == 3
    assert result["metrics"]["net_interest_margin"]["value"] == .021
    assert result["metrics"]["cet1"]["value"] == pytest.approx(.119)
    assert result["metrics"]["provision_credit_losses"]["value"] == 1_500_000_000


def test_automotive_table_units_and_energy_growth_are_not_ticker_specific():
    modules = _modules([
        "Wholesale Units (000) 742 710 for the current and prior periods.",
        "Automotive revenues were $20.4 billion.",
        "Automotive gross margin was 18.2%.",
        "Inventory was $14.0 billion.",
        "Capital spending was $2.4 billion.",
        "Company adjusted free cash flow was $2.1 billion.",
        "Automotive regulatory credits revenue was $442 million.",
        "Energy generation and storage revenues were $3.1 billion.",
        "Energy generation and storage revenue increased 13% year over year.",
    ])
    result = build_industry_metrics("ANY", "AUTOMOTIVE", modules)
    assert result["available_count"] == 6
    assert result["optional_available_count"] >= 3
    assert result["metrics"]["deliveries"]["value"] == 742_000
    assert result["metrics"]["energy_growth"]["value"] == .13
    assert result["metrics"]["asp"]["value"] is None


def test_energy_company_production_wins_and_gas_rejects_boe():
    modules = _modules([
        "The Company's share of net production was about 68 thousand barrels per day in one project.",
        "Its share of combined oil and gas production was approximately 320 thousand oil-equivalent barrels per day.",
        "In 2025, Upstream production averaged 4.7 million oil-equivalent barrels per day.",
    ])
    result = build_industry_metrics("ANY", "ENERGY", modules)
    assert result["metrics"]["production"]["value"] == 4_700_000
    assert result["metrics"]["gas_production"]["value"] is None


def test_energy_operational_table_uses_only_explicit_totals_and_units():
    modules = _modules([
        "Upstream Operational Results | Net production of crude oil, natural gas liquids, bitumen and synthetic oil (thousands of barrels daily) | United States | 1,653 | Canada | 922 | Total | 3,310 | Natural gas production (millions of cubic feet daily) | United States | 3,100 | Non-U.S. | 4,250 | Total | 7,350 |",
    ])
    result = build_industry_metrics("ANY", "ENERGY", modules)
    assert result["metrics"]["oil_production"]["value"] == 3_310_000
    assert result["metrics"]["gas_production"]["value"] == 7_350


def test_xom_style_total_table_is_company_level_and_market_supply_is_rejected():
    modules = _modules([
        "Oil production declines at 15 percent; supplies would fall from 100 million barrels per day.",
        "Total natural gas production available for sale | | | 8,442 | | | 8,078 | "
        "(thousands of oil-equivalent barrels daily) | Oil-equivalent production | | | 4,736 | | | 4,333 |",
    ])
    result = build_industry_metrics("ANY", "ENERGY", modules)
    assert result["metrics"]["production"]["value"] == 4_736_000
    assert result["metrics"]["gas_production"]["value"] == 8_442
    assert result["metrics"]["oil_production"]["value"] is None


def test_issuer_level_vehicle_volume_is_distinct_from_segment_volume():
    gm = build_industry_metrics("ANY", "AUTOMOTIVE", _modules([
        "The following table summarizes wholesale vehicle sales by our Automotive operations (vehicles in thousands): "
        "| 2025 | 2024 | GMNA | 3,296 | 3,464 | GMI | 503 | 547 | Total | 3,799 | 100.0 | % | 4,010 | 100.0 | % |"
    ]))
    ford = build_industry_metrics("ANY", "AUTOMOTIVE", _modules([
        "Ford Blue Segment | Key Metrics Wholesale Units (000) | 2025 | 2,728 | 2024 | 2,862 |"
    ]))
    assert gm["metrics"]["deliveries"]["value"] == 3_799_000
    assert gm["metrics"]["deliveries"]["normalized_name"] == "vehicle_volume"
    assert "wholesale" in gm["metrics"]["deliveries"]["definition"].lower()
    assert ford["metrics"]["deliveries"]["value"] is None


def test_company_vehicle_total_outranks_regional_sales_and_inventory_requires_units():
    result = build_industry_metrics("ANY", "AUTOMOTIVE", _modules([
        "Our total vehicle sales in one region were 0.9 million units.",
        "The following table summarizes wholesale vehicle sales by our Automotive operations (vehicles in thousands): "
        "| 2025 | 2024 | GMNA | 3,296 | 3,464 | GMI | 503 | 547 | Total | 3,799 | 100.0 | % | 4,010 | 100.0 | % |",
        "Inventories | Total inventories | | $ | 15,950 | | $ | 14,467 |",
        "Automotive net sales and revenue | | $ | 92 | | 91 |",
    ]))
    assert result["metrics"]["deliveries"]["value"] == 3_799_000
    assert result["metrics"]["inventory"]["value"] is None
    assert result["metrics"]["automotive_revenue"]["value"] is None


def test_automotive_change_amounts_are_not_reported_levels():
    modules = _modules([
        "Automotive sales revenue increased $6.77 billion, or 24%, due to an 18% increase in cash deliveries.",
        "Cost of automotive sales revenue benefits from manufacturing credits earned, amounting to $565 million.",
        "Average selling price increased while energy revenue increased $350 million, or 13%.",
        "We recorded inventory write-downs of $100 million.",
        "Automotive Regulatory Credits performance obligations totaled $287 million.",
    ])
    result = build_industry_metrics("ANY", "AUTOMOTIVE", modules)
    for name in ("deliveries", "automotive_revenue", "asp", "inventory", "regulatory_credits", "energy_revenue"):
        assert result["metrics"][name]["value"] is None


def test_automotive_flattened_tables_extract_current_reported_levels():
    modules = _modules([
        "Revenue by source (in millions): | Automotive sales | | $ | 20,006 | | $ | 15,787 | Automotive regulatory credits | | | 146 | | | 439 | Energy generation and storage sales | | | 2,998 | | | 2,646 |",
        "Inventories (in millions): Raw materials | 6,468 | Work in process | 1,764 | Finished goods | 5,929 | Total | $ | 13,752 | Finished goods inventory includes products in transit.",
        "Gross margin for total automotive decreased from 18.4% to 17.8%.",
    ])
    result = build_industry_metrics("ANY", "AUTOMOTIVE", modules)
    assert result["metrics"]["automotive_revenue"]["value"] == 20_006_000_000
    assert result["metrics"]["regulatory_credits"]["value"] == 146_000_000
    assert result["metrics"]["energy_revenue"]["value"] == 2_998_000_000
    assert result["metrics"]["inventory"]["value"] == 13_752_000_000
    assert result["metrics"]["automotive_gross_margin"]["value"] == pytest.approx(.178)


def test_industry_risk_relevance_demotes_unsubstantiated_bank_supply_chain():
    credit = classify_industry_risk("FINANCIALS", "CREDIT risk", "Net charge-offs increased", confidence=.8, has_evidence=True)
    supply = classify_industry_risk("FINANCIALS", "SUPPLY_CHAIN risk", "Third-party relationship count", confidence=.72, has_evidence=True)
    assert credit["industry_relevance"] > supply["industry_relevance"]
    assert supply["native_to_profile"] is False


def test_main_tension_skips_ineligible_opposing_driver():
    drivers = {
        "positive": [{"id": "p", "title": "CET1", "importance": "HIGH", "confidence": .8, "industry_relevance": .95,
                      "source_backed": True, "findings": ["fp"], "evidence": ["ep"]}],
        "negative": [{"id": "n", "title": "Supply chain", "importance": "HIGH", "confidence": .8, "industry_relevance": .25,
                      "source_backed": True, "findings": ["fn"], "evidence": ["en"]}],
    }
    result = build_main_tension("BANK", drivers, [])
    assert result["statement"] == "No high-confidence opposing driver identified."
    assert result["negative_driver_id"] is None


def test_generic_fallback_and_ga_gate_are_explicit():
    industry = {"available_count": 1, "required_count": 9}
    findings = [{"category": "growth", "evidence_ids": ["e1"]}]
    fallback = detect_generic_fallback("FINANCIALS", industry, findings)
    gate = ga_coverage_gate("FINANCIALS", industry, findings, [{"id": "e1", "verified": True}])
    assert fallback["status"] == "INDUSTRY_COVERAGE_LIMITED"
    assert gate["status"] == "LIMITED"


def test_support_tier_and_gate_v2_treat_limited_as_publishable():
    industry = {"required_coverage": .50, "optional_coverage": .20, "available_count": 3, "required_count": 6}
    findings = [{"industry_specific": True, "evidence_ids": ["e1"]}]
    evidence = [{"id": "e1", "verified": True}]
    modules = {name: {"status": "COMPLETED", "metrics": {"ok": 1}} for name in ("fundamental", "valuation", "sec")}
    confidence = {"modules": {name: {"score": 70} for name in ("fundamental", "valuation", "sec")}}
    support = research_support_tier("AUTOMOTIVE", industry, findings, evidence, modules, confidence)
    gate = ga_coverage_gate("AUTOMOTIVE", industry, findings, evidence, modules, confidence, support)
    assert support["tier"] == "LIMITED"
    assert gate["status"] == "LIMITED"
    assert gate["publishable"] is True
