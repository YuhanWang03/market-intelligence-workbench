"""Current no-fabrication industry data capability audits."""

JPM_CAPABILITY_MATRIX = [
    {"metric": "NIM", "financial_datasets": "NOT_IN_CURRENT_MODEL", "sec_xbrl": "POSSIBLE_TAG_VARIATION", "filing_text": "AVAILABLE", "reliability": "MEDIUM", "recommended_source": "SEC XBRL + earnings release cross-check"},
    {"metric": "CET1", "financial_datasets": "NOT_IN_CURRENT_MODEL", "sec_xbrl": "LIMITED", "filing_text": "AVAILABLE", "reliability": "HIGH", "recommended_source": "10-Q capital table / earnings release"},
    {"metric": "Loan Growth", "financial_datasets": "NOT_IN_CURRENT_MODEL", "sec_xbrl": "AVAILABLE_DERIVED", "filing_text": "AVAILABLE", "reliability": "HIGH", "recommended_source": "SEC XBRL period comparison"},
    {"metric": "Deposit Growth", "financial_datasets": "NOT_IN_CURRENT_MODEL", "sec_xbrl": "AVAILABLE_DERIVED", "filing_text": "AVAILABLE", "reliability": "HIGH", "recommended_source": "SEC XBRL period comparison"},
    {"metric": "Provision for Credit Losses", "financial_datasets": "NOT_IN_CURRENT_MODEL", "sec_xbrl": "AVAILABLE", "filing_text": "AVAILABLE", "reliability": "HIGH", "recommended_source": "SEC XBRL"},
    {"metric": "Net Charge-Off", "financial_datasets": "NOT_IN_CURRENT_MODEL", "sec_xbrl": "POSSIBLE_TAG_VARIATION", "filing_text": "AVAILABLE", "reliability": "MEDIUM", "recommended_source": "10-Q credit-quality table"},
    {"metric": "Credit Quality", "financial_datasets": "NOT_IN_CURRENT_MODEL", "sec_xbrl": "DERIVED_MULTI_TAG", "filing_text": "AVAILABLE", "reliability": "MEDIUM", "recommended_source": "10-Q credit-quality table"},
    {"metric": "Efficiency Ratio", "financial_datasets": "NOT_IN_CURRENT_MODEL", "sec_xbrl": "DERIVED", "filing_text": "AVAILABLE", "reliability": "MEDIUM", "recommended_source": "earnings release, validate against SEC"},
]

XOM_CAPABILITY_MATRIX = [
    {"metric": "Production", "financial_datasets": "NOT_IN_CURRENT_MODEL", "sec_xbrl": "LIMITED", "filing_text": "AVAILABLE", "reliability": "HIGH", "recommended_source": "10-Q/10-K production table"},
    {"metric": "Oil Production", "financial_datasets": "NOT_IN_CURRENT_MODEL", "sec_xbrl": "LIMITED", "filing_text": "AVAILABLE", "reliability": "HIGH", "recommended_source": "10-Q/10-K production table"},
    {"metric": "Gas Production", "financial_datasets": "NOT_IN_CURRENT_MODEL", "sec_xbrl": "LIMITED", "filing_text": "AVAILABLE", "reliability": "HIGH", "recommended_source": "10-Q/10-K production table"},
    {"metric": "Realized Oil Price", "financial_datasets": "NOT_IN_CURRENT_MODEL", "sec_xbrl": "LIMITED", "filing_text": "AVAILABLE", "reliability": "MEDIUM", "recommended_source": "earnings supplement"},
    {"metric": "Realized Gas Price", "financial_datasets": "NOT_IN_CURRENT_MODEL", "sec_xbrl": "LIMITED", "filing_text": "AVAILABLE", "reliability": "MEDIUM", "recommended_source": "earnings supplement"},
    {"metric": "Upstream/Downstream/Chemical Earnings", "financial_datasets": "NOT_IN_CURRENT_MODEL", "sec_xbrl": "AVAILABLE_SEGMENTS", "filing_text": "AVAILABLE", "reliability": "HIGH", "recommended_source": "SEC segment facts"},
    {"metric": "CapEx", "financial_datasets": "AVAILABLE", "sec_xbrl": "AVAILABLE", "filing_text": "AVAILABLE", "reliability": "HIGH", "recommended_source": "Financial Datasets, validate against SEC"},
    {"metric": "FCF", "financial_datasets": "AVAILABLE", "sec_xbrl": "DERIVED", "filing_text": "AVAILABLE", "reliability": "HIGH", "recommended_source": "Financial Datasets + SEC cash flow"},
    {"metric": "Reserve / Production", "financial_datasets": "NOT_IN_CURRENT_MODEL", "sec_xbrl": "LIMITED", "filing_text": "AVAILABLE_ANNUALLY", "reliability": "MEDIUM", "recommended_source": "10-K reserve disclosure"},
    {"metric": "Breakeven", "financial_datasets": "UNAVAILABLE", "sec_xbrl": "UNAVAILABLE", "filing_text": "NOT_STANDARDIZED", "reliability": "LOW", "recommended_source": "Keep N/A unless explicitly disclosed"},
]

AUTOMOTIVE_PROFILE_RECOMMENDATION = {
    "recommended": True,
    "phase": "IMPLEMENTED_PHASE_3D",
    "reason": "TSLA combines automotive manufacturing and energy businesses; GENERAL misses delivery, pricing and automotive-margin economics.",
    "required_metrics": ["deliveries", "automotive_revenue", "automotive_margin", "inventory", "capex", "free_cash_flow"],
    "optional_metrics": ["asp", "regulatory_credits", "energy_revenue", "energy_growth", "segment_margin"],
}


def capability_audit(ticker: str) -> dict | None:
    if ticker == "JPM":
        return {"company": "JPM", "profile": "FINANCIALS", "matrix": JPM_CAPABILITY_MATRIX, "integrated_now": [], "policy": "Unparsed values remain N/A; no LLM numeric extraction."}
    if ticker == "XOM":
        return {"company": "XOM", "profile": "ENERGY", "matrix": XOM_CAPABILITY_MATRIX, "integrated_now": ["CapEx", "FCF"], "policy": "Unparsed values remain N/A; no LLM numeric extraction."}
    if ticker == "TSLA":
        return {"company": "TSLA", "profile": "AUTOMOTIVE", "profile_review": AUTOMOTIVE_PROFILE_RECOMMENDATION,
                "policy": "Only structured SEC/FD values are scored; missing automotive metrics remain N/A."}
    return None
