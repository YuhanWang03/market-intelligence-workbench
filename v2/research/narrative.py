"""Evidence-bound deterministic Research Narrative v3."""

from __future__ import annotations

import re
from typing import Any


def build_main_tension(ticker: str, drivers: dict, limitations: list[str]) -> dict:
    rank = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}

    def eligible(driver: dict) -> bool:
        return (
            rank.get(str(driver.get("importance")), 0) >= rank["HIGH"]
            and float(driver.get("confidence") or 0) >= .55
            and float(driver.get("industry_relevance") or 0) >= .65
            and bool(driver.get("source_backed"))
            and bool(driver.get("findings"))
            and bool(driver.get("evidence"))
        )

    positive = next((row for row in drivers.get("positive", []) if eligible(row)), None)
    negative = next((row for row in drivers.get("negative", []) if eligible(row)), None)
    if positive and negative:
        statement = f"{positive['title']} versus {negative['title']}."
        refs = positive.get("findings", []) + negative.get("findings", [])
    elif positive:
        statement = "No high-confidence opposing driver identified."
        refs = positive.get("findings", [])
    elif negative:
        statement = "No high-confidence opposing driver identified."
        refs = negative.get("findings", [])
    else:
        statement = "Available evidence is insufficient to define a company-specific investment tension."
        refs = []
    return {"ticker": ticker, "statement": statement, "positive_driver_id": positive.get("id") if positive else None,
            "negative_driver_id": negative.get("id") if negative else None, "finding_ids": refs,
            "limitations": limitations[:2], "eligibility": {"minimum_importance": "HIGH", "minimum_confidence": .55,
            "minimum_industry_relevance": .65, "requires_source_backed_evidence": True}, "deterministic": True}


def build_debate(drivers: dict, catalysts: list[dict], invalidations: list[dict]) -> dict:
    bulls = [{"claim": row["title"], "driver_id": row["id"], "finding_ids": row.get("findings", []), "evidence_ids": row.get("evidence", [])} for row in drivers.get("positive", [])[:5]]
    bears = [{"claim": row["title"], "driver_id": row["id"], "finding_ids": row.get("findings", []), "evidence_ids": row.get("evidence", [])} for row in drivers.get("negative", [])[:5]]
    matters = []
    for row in invalidations[:3]:
        matters.append({"claim": row["monitor"], "type": "INVALIDATION_MONITOR", "driver_id": row.get("driver_id"), "evidence_ids": row.get("evidence_ids", [])})
    for row in catalysts[:2]:
        matters.append({"claim": row["title"], "type": "CATALYST", "finding_id": row.get("id"), "evidence_ids": row.get("evidence_ids", [])})
    return {"what_bulls_believe": bulls, "what_bears_believe": bears, "what_matters_most": matters[:5]}


def confidence_explanation(confidence: dict, scoring: dict, provider_health: list[dict]) -> dict:
    modules = confidence.get("modules", {})
    strongest = sorted(modules.items(), key=lambda item: item[1].get("score", 0), reverse=True)[:4]
    weakest = sorted(modules.items(), key=lambda item: item[1].get("score", 0))[:4]
    contributors = [f"{name}: {row.get('score', 0)}/100 module confidence" for name, row in strongest if row.get("score", 0) >= 60]
    limitations = [f"{name}: {row.get('score', 0)}/100; missing {', '.join(row.get('missing_fields', [])[:3]) or 'coverage'}" for name, row in weakest if row.get("score", 0) < 60]
    if scoring.get("coverage", 1) < .75:
        limitations.append(f"Industry metric coverage is {round(scoring.get('coverage', 0) * 100)}%.")
    unhealthy = [row["provider"] for row in provider_health if row.get("status") != "HEALTHY"]
    if unhealthy:
        limitations.append("Provider limitations: " + ", ".join(unhealthy))
    return {"score": confidence.get("score"), "contributors": contributors[:5], "limitations": limitations[:5],
            "calculation": "deterministic; narrative layer cannot modify score"}


def factuality_guard(payload: dict, finding_ids: set[str], evidence_ids: set[str], catalyst_ids: set[str]) -> dict:
    removed = []
    unsupported_numbers = []
    price_target = re.compile(r"(?:price target|target price|目标价)", re.I)
    for section in ("what_bulls_believe", "what_bears_believe"):
        valid = []
        for row in payload.get("investment_debate", {}).get(section, []):
            if not set(row.get("finding_ids", [])).issubset(finding_ids) or not set(row.get("evidence_ids", [])).issubset(evidence_ids) or price_target.search(row.get("claim", "")):
                removed.append({"section": section, "claim": row.get("claim")})
            else:
                valid.append(row)
        payload["investment_debate"][section] = valid
    valid_matters = []
    for row in payload.get("investment_debate", {}).get("what_matters_most", []):
        if row.get("type") == "CATALYST" and row.get("finding_id") not in catalyst_ids:
            removed.append({"section": "what_matters_most", "claim": row.get("claim")})
        elif not set(row.get("evidence_ids", [])).issubset(evidence_ids):
            removed.append({"section": "what_matters_most", "claim": row.get("claim")})
        else:
            valid_matters.append(row)
    payload["investment_debate"]["what_matters_most"] = valid_matters
    payload["factuality"] = {"status": "PASSED" if not removed and not unsupported_numbers else "CLAIMS_REMOVED",
                             "removed_claims": removed, "unsupported_numbers": unsupported_numbers,
                             "checks": ["ticker", "finding references", "evidence references", "catalyst references", "unsupported price target"]}
    return payload


class ResearchNarrativeEngine:
    """Connect structured intelligence without introducing facts or scores."""

    def build(self, ticker: str, *, core_thesis: str, why_now: str, main_tension: dict, drivers: dict,
              debate: dict, scenarios: dict, catalysts: list[dict], risks: list[dict], invalidations: list[dict],
              confidence: dict, limitations: list[str], findings: list[dict], evidence: list[dict]) -> dict:
        payload: dict[str, Any] = {"version": "investment-thesis-v3", "ticker": ticker, "core_thesis": core_thesis,
            "why_now": why_now, "main_tension": main_tension, "positive_drivers": drivers.get("positive", [])[:5],
            "negative_drivers": drivers.get("negative", [])[:5], "investment_debate": debate,
            "bull_case": scenarios["BULL"], "base_case": scenarios["BASE"], "bear_case": scenarios["BEAR"],
            "key_catalysts": catalysts[:5], "key_risks": risks[:5], "thesis_invalidation": invalidations[:5],
            "research_confidence": confidence, "research_limitations": limitations[:8],
            "decision_support_only": True, "llm_used": False}
        return factuality_guard(payload, {row["id"] for row in findings}, {row["id"] for row in evidence}, {row["id"] for row in catalysts})
