"""Deterministic industry metrics, scoring and findings for Research V1.0.

This module only transforms already-normalized company, financial and SEC
evidence. Missing values remain ``None``; no language model is involved.
"""

from __future__ import annotations

import math
import re
from typing import Any


AUTOMOTIVE_TICKERS = {"TSLA", "GM", "F", "RIVN", "LCID", "STLA", "TM", "HMC"}

DRIVER_TAXONOMY = {
    "FINANCIALS": ["PROFITABILITY", "CAPITAL_STRENGTH", "CREDIT_QUALITY", "LOAN_GROWTH", "DEPOSIT_GROWTH", "RATE_SENSITIVITY", "EFFICIENCY"],
    "ENERGY": ["PRODUCTION", "REALIZED_PRICE", "UPSTREAM", "DOWNSTREAM", "CAPEX", "FCF", "COMMODITY_EXPOSURE", "RESERVES"],
    "AUTOMOTIVE": ["DELIVERIES", "PRICING", "AUTOMOTIVE_MARGIN", "INVENTORY", "CAPACITY", "CAPEX", "FCF", "REGULATORY", "ENERGY_BUSINESS"],
    "TECHNOLOGY": ["GROWTH", "PROFITABILITY", "CAPITAL_EFFICIENCY", "R_AND_D", "SBC", "VALUATION"],
    "GENERAL": ["GROWTH", "PROFITABILITY", "CAPITAL_EFFICIENCY", "FINANCIAL_HEALTH", "VALUATION"],
}

RISK_TAXONOMY = {
    "FINANCIALS": ["CREDIT", "CAPITAL", "LIQUIDITY", "RATE_SENSITIVITY", "DEPOSIT", "REGULATORY", "LEGAL", "COMMERCIAL_REAL_ESTATE", "CYBERSECURITY"],
    "ENERGY": ["COMMODITY", "PRODUCTION", "RESERVE", "CAPEX", "REGULATORY", "GEOPOLITICAL", "OPERATIONAL", "ENVIRONMENTAL"],
    "AUTOMOTIVE": ["DEMAND", "PRICING", "MARGIN", "INVENTORY", "SUPPLY_CHAIN", "REGULATORY", "PRODUCT", "CAPACITY", "COMPETITION"],
    "TECHNOLOGY": ["VALUATION", "PRODUCT", "COMPETITION", "REGULATORY", "SUPPLY_CHAIN", "CUSTOMER_CONCENTRATION", "CYBERSECURITY", "SBC"],
    "GENERAL": ["FINANCIAL", "VALUATION", "OPERATIONAL", "MANAGEMENT", "EVENT", "SUPPLY_CHAIN", "REGULATORY", "LEGAL"],
}

# GA coverage is based on metrics that are broadly applicable to the profile.
# Company-specific disclosures remain useful, but cannot lower another
# company's required coverage merely because the metric is not applicable.
INDUSTRY_METRIC_REQUIREMENTS = {
    "FINANCIALS": {
        "required": ["roe", "net_interest_margin", "cet1", "loan_growth", "deposit_growth", "credit_quality"],
        "optional": ["provision_credit_losses", "net_charge_off_rate", "efficiency_ratio"],
    },
    "ENERGY": {
        "required": ["production", "upstream_earnings", "downstream_earnings", "capex", "free_cash_flow"],
        "optional": ["oil_production", "gas_production", "realized_oil_price", "realized_gas_price",
                     "chemical_earnings", "reserve_production_ratio", "breakeven", "wti_crude", "natural_gas"],
    },
    "AUTOMOTIVE": {
        "required": ["deliveries", "automotive_revenue", "automotive_margin", "inventory", "capex", "free_cash_flow"],
        "optional": ["asp", "regulatory_credits", "energy_revenue", "energy_growth", "segment_margin"],
    },
    "TECHNOLOGY": {
        "required": ["revenue_growth", "gross_margin", "operating_margin", "roic"],
        "optional": ["free_cash_flow", "research_and_development", "stock_based_compensation"],
    },
    "GENERAL": {
        "required": ["revenue_growth", "gross_margin", "operating_margin", "roic"],
        "optional": ["free_cash_flow"],
    },
}

_NORMALIZED_METRIC_NAMES = {"deliveries": "vehicle_volume", "automotive_gross_margin": "automotive_margin"}

_METRIC_DEFINITIONS = {
    "deliveries": "Company-level vehicle volume as defined by the issuer; wholesale volume, retail sales and customer deliveries are not assumed equivalent.",
    "automotive_revenue": "Company-reported automotive net sales or automotive revenue for the disclosed period.",
    "automotive_margin": "Company-level automotive margin using the issuer's disclosed gross-margin or EBIT-margin definition.",
    "automotive_gross_margin": "Company-level automotive margin using the issuer's disclosed gross-margin definition.",
    "segment_margin": "Issuer-reported automotive segment margin; it is not treated as consolidated automotive margin.",
    "production": "Company-level oil-equivalent production for the disclosed period.",
    "oil_production": "Company-level oil or liquids production; market supply and project-only production are excluded.",
    "gas_production": "Company-level natural-gas production available for sale.",
}

_RISK_ALIASES = {
    "CREDIT": ("credit", "charge-off", "charge off", "nonperforming", "loan loss"),
    "CAPITAL": ("cet1", "capital adequacy", "capital ratio", "capital requirement"),
    "LIQUIDITY": ("liquidity", "funding", "cash"),
    "RATE_SENSITIVITY": ("interest rate", "net interest", "rate sensitivity"),
    "DEPOSIT": ("deposit",), "COMMERCIAL_REAL_ESTATE": ("commercial real estate", "cre"),
    "COMMODITY": ("commodity", "oil price", "gas price", "realized price"),
    "PRODUCTION": ("production", "output"), "RESERVE": ("reserve",), "CAPEX": ("capex", "capital expenditure"),
    "GEOPOLITICAL": ("geopolitical", "sanction", "war", "middle east"),
    "ENVIRONMENTAL": ("environment", "emission", "climate", "spill"),
    "DEMAND": ("demand", "deliver", "vehicle sales", "wholesale"), "PRICING": ("pricing", "price cut", "asp"),
    "MARGIN": ("margin", "profitability"), "INVENTORY": ("inventory",), "CAPACITY": ("capacity", "factory", "plant"),
    "PRODUCT": ("product", "recall", "quality", "launch"), "COMPETITION": ("competition", "competitive"),
    "CUSTOMER_CONCENTRATION": ("customer concentration", "single customer"), "SBC": ("stock-based compensation", "sbc", "dilution"),
    "CYBERSECURITY": ("cyber", "data breach"), "REGULATORY": ("regulatory", "regulation", "compliance"),
    "LEGAL": ("legal", "litigation", "lawsuit", "investigation"), "SUPPLY_CHAIN": ("supply chain", "supplier"),
    "VALUATION": ("valuation", "p/e", "multiple"), "OPERATIONAL": ("operational", "execution", "downtime"),
    "MANAGEMENT": ("management", "guidance"), "EVENT": ("event", "catalyst"), "FINANCIAL": ("financial", "leverage", "debt"),
}


def classify_industry_risk(profile: str, name: str, reason: str, *, confidence: float = 0.0,
                           has_evidence: bool = False) -> dict:
    """Map a generic risk into the profile taxonomy without hiding cross-sector risk."""
    profile = profile if profile in RISK_TAXONOMY else "GENERAL"
    raw = f"{name} {reason}".lower().replace("_", " ")
    matched = next((kind for kind, words in _RISK_ALIASES.items() if any(word in raw for word in words)), str(name).upper())
    allowed = RISK_TAXONOMY[profile]
    if matched in allowed:
        relevance = .95
    elif profile == "FINANCIALS" and matched in {"FINANCIAL", "ACCOUNTING", "MANAGEMENT"}:
        matched = "CAPITAL" if matched == "FINANCIAL" else "REGULATORY"
        relevance = .78
    elif profile == "ENERGY" and matched in {"FINANCIAL", "MANAGEMENT", "SUPPLY_CHAIN"}:
        matched = "CAPEX" if matched == "FINANCIAL" else "OPERATIONAL"
        relevance = .68 if matched == "CAPEX" else .55
    elif profile == "AUTOMOTIVE" and matched in {"FINANCIAL", "MANAGEMENT", "EVENT"}:
        matched = "MARGIN" if matched == "FINANCIAL" else "PRODUCT"
        relevance = .65
    elif profile == "TECHNOLOGY" and matched in {"FINANCIAL", "MANAGEMENT", "EVENT"}:
        matched = "VALUATION" if matched == "FINANCIAL" else "PRODUCT"
        relevance = .62
    else:
        # Cross-industry risks remain available, but need unusually strong,
        # sourced evidence before they can outrank native sector risks.
        relevance = .25
    if has_evidence and confidence >= .8:
        relevance = max(relevance, .55)
    return {"risk_type": matched, "industry_relevance": round(relevance, 3),
            "taxonomy_profile": profile, "native_to_profile": matched in allowed}


def profile_for(ticker: str, sector: str | None, industry: str | None) -> str:
    ticker = ticker.upper()
    text = f"{sector or ''} {industry or ''}".lower()
    if ticker in AUTOMOTIVE_TICKERS or any(word in text for word in ("auto manufacturer", "automotive", "motor vehicle")):
        return "AUTOMOTIVE"
    if any(word in text for word in ("financial", "bank", "insurance", "credit")):
        return "FINANCIALS"
    if any(word in text for word in ("energy", "oil", "gas", "petroleum")):
        return "ENERGY"
    if any(word in text for word in ("technology", "semiconductor", "software", "hardware")):
        return "TECHNOLOGY"
    return "GENERAL"


def _num(value: Any) -> float | None:
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def _sec_rows(modules: dict[str, dict]) -> list[dict]:
    details = modules.get("sec", {}).get("details", {})
    rows = list(details.get("sec_findings", [])) + list(details.get("industry_evidence", []))
    normalized = []
    seen = set()
    for row in rows:
        text = re.sub(r"\s+", " ", str(row.get("evidence_text") or row.get("summary") or "")).strip()
        key = re.sub(r"\W+", "", text.lower())[:400]
        if not text or key in seen:
            continue
        seen.add(key)
        normalized.append({"text": text, "filing_date": str(row.get("filing_date") or ""),
                           "source_url": row.get("source_url")})
    return sorted(normalized, key=lambda row: row["filing_date"], reverse=True)


def _signed_number(raw: str) -> float:
    value = raw.strip().replace("$", "").replace(",", "")
    words = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
             "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
             "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
             "eighteen": 18, "nineteen": 19, "twenty": 20}
    if value.lower() in words:
        return float(words[value.lower()])
    negative = value.startswith("(") and value.endswith(")")
    value = value.strip("()")
    return -float(value) if negative else float(value)


def _scale_multiplier(suffix: str, context: str = "") -> float:
    value = suffix.lower().strip()
    if value in {"billion", "bn", "b"}:
        return 1_000_000_000
    if value in {"million", "mm", "m"}:
        return 1_000_000
    if value in {"thousand", "000"}:
        return 1_000
    lower = context.lower()
    if re.search(r"(?:\$m|millions? of dollars|in millions)", lower):
        return 1_000_000
    if re.search(r"(?:\$b|billions? of dollars|in billions)", lower):
        return 1_000_000_000
    if re.search(r"(?:units?|vehicles?)\s*\(000\)", lower):
        return 1_000
    return 1


def _extract(rows: list[dict], labels: tuple[str, ...], kind: str = "percent", unit: str | None = None,
             *, allow_change: bool = False, prefer: tuple[str, ...] = (), reject: tuple[str, ...] = ()) -> dict | None:
    label_re = re.compile("(?:" + "|".join(labels) + ")", re.I)
    percent_re = re.compile(r"(\(?-?\d[\d,]*(?:\.\d+)?\)?|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)\s*(?:%|percent\b)", re.I)
    money_re = re.compile(r"\$\s*(\(?-?\d[\d,]*(?:\.\d+)?\)?)\s*(billion|million|thousand|bn|mm|m|b)?\b", re.I)
    number_re = re.compile(r"(\(?-?\d[\d,]*(?:\.\d+)?\)?)\s*(million|thousand|billion|bn|mm|m|b)?\b", re.I)
    production_re = re.compile(
        r"(\(?-?\d[\d,]*(?:\.\d+)?\)?)\s*(billion|million|thousand)?\s*"
        r"(oil[- ]equivalent barrels?|barrels?|boe|mboe|mmboe|cubic feet|mcf|mmcf|bcf)\s*(?:per\s*day|/\s*(?:day|d))",
        re.I,
    )
    candidates: list[tuple[float, dict]] = []
    for row_index, row in enumerate(rows):
        text = row["text"]
        for label_match in label_re.finditer(text):
            context = text[max(0, label_match.start() - 180): min(len(text), label_match.end() + 300)]
            after = text[label_match.end(): min(len(text), label_match.end() + 240)]
            if any(re.search(pattern, context, re.I) for pattern in reject):
                continue
            # A change amount is not the reported level.  Growth metrics opt in
            # explicitly; stock metrics wait for an actual value/table cell.
            if not allow_change and re.match(r"\s*(?:increased|decreased|grew|declined|rose|fell)(?:\s+by)?\b", after, re.I):
                continue
            if kind == "percent":
                match = percent_re.search(after)
                if not match:
                    continue
                value, canonical = _signed_number(match.group(1)) / 100, "ratio"
            elif kind == "money":
                match = money_re.search(after)
                if not match:
                    continue
                value = _signed_number(match.group(1)) * _scale_multiplier(match.group(2) or "", context)
                canonical = "USD"
            elif kind == "production":
                match = production_re.search(after)
                if not match:
                    continue
                value = _signed_number(match.group(1))
                scale, raw_unit = (match.group(2) or "").lower(), match.group(3).lower()
                if unit == "MMcf/day":
                    if raw_unit not in {"cubic feet", "mcf", "mmcf", "bcf"}:
                        continue
                    if raw_unit == "bcf":
                        value *= 1_000
                    elif raw_unit == "mmcf":
                        pass
                    elif raw_unit == "mcf":
                        value *= 1_000 if scale == "billion" else 1 if scale == "million" else .001 if scale == "thousand" else .001
                    elif raw_unit == "cubic feet":
                        value *= 1_000 if scale == "billion" else 1 if scale == "million" else .001 if scale == "thousand" else .000001
                    canonical = "MMcf/day"
                else:
                    if raw_unit in {"cubic feet", "mcf", "mmcf", "bcf"}:
                        continue
                    value *= 1_000_000 if scale == "million" or raw_unit == "mmboe" else 1_000 if scale == "thousand" or raw_unit == "mboe" else 1
                    canonical = "BOE/day"
            else:
                clean_after = re.sub(r"^\s*\(000\)", " ", after, flags=re.I)
                match = number_re.search(clean_after)
                if not match:
                    continue
                vicinity = clean_after[max(0, match.start() - 16):match.end() + 28]
                if kind == "quantity" and (
                    "$" in vicinity or re.search(r"%|percent|\b(?:19|20)\d{2}\b", vicinity, re.I)
                    or not (match.group(2) or re.search(r"units?|vehicles?|deliveries|wholesales?", context, re.I))
                ):
                    continue
                value = _signed_number(match.group(1)) * _scale_multiplier(match.group(2) or "", context)
                canonical = unit or "count"
            score = 2.0 / (row_index + 1)
            score += 4 * sum(bool(re.search(pattern, context, re.I)) for pattern in prefer)
            candidates.append((score, {"value": value, "unit": canonical, "raw": context.strip()[:420],
                                      "filing_date": row.get("filing_date"), "source_url": row.get("source_url")}))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def _extract_table_value(rows: list[dict], labels: tuple[str, ...], kind: str, unit: str,
                         *, reject: tuple[str, ...] = (), require: tuple[str, ...] = (), scale_money: bool = True) -> dict | None:
    """Read the first current-period value following a row label in a flattened SEC table."""
    label_re = re.compile("(?:" + "|".join(labels) + ")", re.I)
    for row in rows:
        text = row["text"]
        for label in label_re.finditer(text):
            context = text[max(0, label.start() - 220):min(len(text), label.end() + 420)]
            if any(re.search(pattern, context, re.I) for pattern in reject):
                continue
            if require and not any(re.search(pattern, text, re.I) for pattern in require):
                continue
            after = text[label.end():label.end() + 160]
            if kind == "quantity":
                year_values = [(int(year), raw) for year, raw in re.findall(
                    r"\b(20\d{2})\s*(?:\|\s*)+(\(?-?\d[\d,]*(?:\.\d+)?\)?)", after)]
                if year_values:
                    newest = max(year for year, _ in year_values)
                    raw = next(raw for year, raw in year_values if year == newest)
                    value = _signed_number(raw) * (1_000 if re.search(r"units?\s*\(000\)|vehicles? in thousands", text, re.I) else 1)
                    return {"value": value, "unit": unit, "raw": context.strip()[:420],
                            "filing_date": row.get("filing_date"), "source_url": row.get("source_url")}
            if kind == "money":
                match = re.search(r"(?:\|\s*)*\$?\s*(?:\|\s*)+(\(?-?\d[\d,]*(?:\.\d+)?\)?)", after)
            elif kind == "percent":
                match = re.search(r"(?:\|\s*)+(\(?-?\d[\d,]*(?:\.\d+)?\)?)\s*(?:\|\s*)*(?:%|percent)?", after, re.I)
            else:
                match = re.search(r"(?:\|\s*)+(\(?-?\d[\d,]*(?:\.\d+)?\)?)", after)
            if not match:
                continue
            value = _signed_number(match.group(1))
            if kind == "money":
                value *= _scale_multiplier("", text) if scale_money else 1
            elif kind == "percent":
                value /= 100
            elif re.search(r"units?\s*\(000\)", context, re.I):
                value *= 1_000
            return {"value": value, "unit": unit, "raw": context.strip()[:420],
                    "filing_date": row.get("filing_date"), "source_url": row.get("source_url")}
    return None


def _extract_transition_percent(rows: list[dict], labels: tuple[str, ...]) -> dict | None:
    label_re = re.compile("(?:" + "|".join(labels) + ")", re.I)
    for row in rows:
        match_label = label_re.search(row["text"])
        if not match_label:
            continue
        context = row["text"][max(0, match_label.start() - 80):match_label.end() + 260]
        match = re.search(r"(?:increased|decreased)\s+from\s+\d+(?:\.\d+)?%\s+to\s+(\d+(?:\.\d+)?)%", context, re.I)
        if match:
            return {"value": float(match.group(1)) / 100, "unit": "ratio", "raw": context.strip()[:420],
                    "filing_date": row.get("filing_date"), "source_url": row.get("source_url")}
    return None


def _extract_inventory_table(rows: list[dict]) -> dict | None:
    for row in rows:
        text = row["text"]
        if not re.search(r"finished goods(?: inventory)?", text, re.I):
            continue
        matches = list(re.finditer(r"\bTotal\s*(?:\|\s*)+\$?\s*(?:\|\s*)*(\d[\d,]*(?:\.\d+)?)", text, re.I))
        anchor = re.search(r"finished goods inventory includes", text, re.I)
        eligible = [match for match in matches if not anchor or match.start() < anchor.start()]
        if not eligible:
            continue
        match = eligible[-1]
        context = text[max(0, match.start() - 260):min(len(text), (anchor.end() if anchor else match.end() + 220))]
        value = _signed_number(match.group(1)) * _scale_multiplier("", text)
        return {"value": value, "unit": "USD", "raw": context.strip()[:420],
                "filing_date": row.get("filing_date"), "source_url": row.get("source_url")}
    return None


def _extract_preceding_total(rows: list[dict], headings: tuple[str, ...]) -> dict | None:
    heading_re = re.compile("(?:" + "|".join(headings) + ")", re.I)
    for row in rows:
        heading = heading_re.search(row["text"])
        if not heading:
            continue
        before = row["text"][max(0, heading.start() - 600):heading.start()]
        totals = list(re.finditer(r"\bTotal\s*(?:\|\s*)+(?:\$\s*(?:\|\s*)*)?(\(?-?\d[\d,]*(?:\.\d+)?\)?)", before, re.I))
        if not totals or not re.search(r"millions? of dollars|dollars in millions|\$m", row["text"], re.I):
            continue
        match = totals[-1]
        context = row["text"][max(0, match.start() - 220):min(len(row["text"]), heading.end() + 100)]
        return {"value": _signed_number(match.group(1)) * 1_000_000, "unit": "USD", "raw": context.strip()[:420],
                "filing_date": row.get("filing_date"), "source_url": row.get("source_url")}
    return None


def _extract_production_table_total(rows: list[dict], label: str, next_label: str | None,
                                    unit_pattern: str, multiplier: float, unit: str) -> dict | None:
    label_re = re.compile(label, re.I)
    for row in rows:
        start_match = label_re.search(row["text"])
        if not start_match:
            continue
        end_match = re.search(next_label, row["text"][start_match.end():], re.I) if next_label else None
        end = start_match.end() + end_match.start() if end_match else min(len(row["text"]), start_match.end() + 3500)
        section = row["text"][start_match.start():end]
        if not re.search(unit_pattern, section, re.I):
            continue
        totals = list(re.finditer(r"\bTotal\s*(?:\|\s*)+(\(?-?\d[\d,]*(?:\.\d+)?\)?)", section, re.I))
        if not totals:
            continue
        match = totals[0]
        return {"value": _signed_number(match.group(1)) * multiplier, "unit": unit,
                "raw": section[max(0, match.start() - 300):match.end() + 160].strip()[:420],
                "filing_date": row.get("filing_date"), "source_url": row.get("source_url")}
    return None


def _extract_labeled_table_value(rows: list[dict], labels: tuple[str, ...], *, unit: str,
                                 multiplier: float, required_context: tuple[str, ...] = ()) -> dict | None:
    """Extract a company-level table row only when its unit context is explicit."""
    label_re = re.compile("(?:" + "|".join(labels) + ")", re.I)
    for row in rows:
        text = row["text"]
        for label in label_re.finditer(text):
            vicinity = text[max(0, label.start() - 1200):min(len(text), label.end() + 1200)]
            if required_context and not all(re.search(pattern, vicinity, re.I) for pattern in required_context):
                continue
            after = text[label.end():label.end() + 180]
            match = re.search(r"(?:\|\s*)+(?:\$\s*(?:\|\s*)*)?(\(?-?\d[\d,]*(?:\.\d+)?\)?)", after)
            if match:
                return {"value": _signed_number(match.group(1)) * multiplier, "unit": unit,
                        "raw": vicinity.strip()[:420], "source_label": label.group(0).strip(),
                        "filing_date": row.get("filing_date"), "source_url": row.get("source_url")}
    return None


def _extract_company_vehicle_volume(rows: list[dict]) -> dict | None:
    """Read issuer-level vehicle volume without promoting a single segment."""
    # Prefer an issuer-level total table across the entire corpus before any
    # narrative sentence, which may describe only one geography.
    for row in rows:
        text = row["text"]
        anchor = re.search(r"table summarizes wholesale vehicle sales by (?:our|the) automotive operations", text, re.I)
        if anchor:
            section = text[anchor.start():anchor.start() + 2200]
            total = re.search(r"\bTotal\s*(?:\|\s*)+(\d[\d,]*(?:\.\d+)?)\s*(?:\|\s*)+100(?:\.0)?\s*(?:\|\s*)*%", section, re.I)
            if total and re.search(r"vehicles in thousands", section, re.I):
                return {"value": _signed_number(total.group(1)) * 1_000, "unit": "vehicles",
                        "raw": section[:420], "source_label": "Wholesale vehicle sales by Automotive operations — Total",
                        "filing_date": row.get("filing_date"), "source_url": row.get("source_url")}
    for row in rows:
        text = row["text"]
        for pattern, label in (
            (r"total company wholesales? (?:were|totaled)\s+(\d[\d,.]*)\s*(million|thousand)?\s+(?:units|vehicles)", "Total company wholesales"),
        ):
            match = re.search(pattern, text, re.I)
            if match:
                vicinity = text[max(0, match.start() - 180):match.end() + 220]
                return {"value": _signed_number(match.group(1)) * _scale_multiplier(match.group(2) or ""),
                        "unit": "vehicles", "raw": vicinity.strip()[:420], "source_label": label,
                        "filing_date": row.get("filing_date"), "source_url": row.get("source_url")}
        wholesale = re.search(r"wholesale units\s*\(000\)", text, re.I)
        if wholesale:
            vicinity = text[max(0, wholesale.start() - 320):wholesale.end() + 420]
            if not re.search(r"\bsegment\b", vicinity, re.I):
                value = re.search(r"(?:\|\s*)*(\d[\d,]*(?:\.\d+)?)", text[wholesale.end():wholesale.end() + 120])
                if value:
                    return {"value": _signed_number(value.group(1)) * 1_000, "unit": "vehicles",
                            "raw": vicinity.strip()[:420], "source_label": "Wholesale Units",
                            "filing_date": row.get("filing_date"), "source_url": row.get("source_url")}
    return None


def _metric(value: Any, unit: str, sources: list[str], method: str, raw: str | None = None,
            filing_date: str | None = None, source_url: str | None = None) -> dict:
    value = _num(value)
    return {"value": value, "unit": unit, "available": value is not None, "sources": sources if value is not None else [],
            "method": method if value is not None else "N/A", "raw_evidence": raw,
            "filing_date": filing_date if value is not None else None, "source_url": source_url if value is not None else None}


def _source_label(name: str, raw: str | None) -> str:
    text = raw or ""
    candidates = {
        "deliveries": ((r"wholesale vehicle sales", "Wholesale Vehicle Sales"),
                       (r"wholesale (?:units|volume)", "Wholesale Volume"),
                       (r"total vehicle sales", "Vehicle Sales"), (r"deliver", "Deliveries")),
        "automotive_margin": ((r"adjusted EBIT margin", "Adjusted EBIT Margin"),
                              (r"EBIT margin", "EBIT Margin"), (r"gross margin", "Automotive Gross Margin")),
        "automotive_gross_margin": ((r"gross margin", "Automotive Gross Margin"),),
        "segment_margin": ((r"Ford Blue", "Ford Blue EBIT Margin"), (r"GMNA", "GMNA EBIT-Adjusted Margin"),
                           (r"segment", "Automotive Segment Margin")),
    }.get(name, ())
    return next((label for pattern, label in candidates if re.search(pattern, text, re.I)), name.replace("_", " ").title())


def _enrich_metric_semantics(profile: str, name: str, item: dict, period: str | None) -> None:
    source = (item.get("sources") or [None])[0]
    normalized = _NORMALIZED_METRIC_NAMES.get(name, name)
    source_label = _source_label(name, item.get("raw_evidence"))
    definition = _METRIC_DEFINITIONS.get(name, f"Issuer/provider reported {source_label}; no value is estimated when unavailable.")
    if name == "deliveries" and source_label == "Wholesale Volume":
        definition = "Issuer-reported wholesale vehicle volume; this is not re-labeled as retail sales or customer deliveries."
    item.update({"metric_name": name, "normalized_name": normalized, "period": item.get("filing_date") or period,
                 "source": source, "source_label": source_label, "definition": definition,
                 "source_definition": definition,
                 "confidence": .90 if source == "fd_earnings" else .88 if source == "sec_filings" and item.get("raw_evidence") else .85 if source else 0.0})


def _latest_quarter(modules: dict[str, dict]) -> dict:
    rows = modules.get("fundamental", {}).get("details", {}).get("quarterly_trends", [])
    return rows[0] if rows else {}


def build_industry_metrics(ticker: str, profile: str, modules: dict[str, dict]) -> dict:
    ticker = ticker.upper()
    fm = modules.get("fundamental", {}).get("metrics", {})
    latest = _latest_quarter(modules)
    sec_rows = _sec_rows(modules)
    metrics: dict[str, dict] = {}
    required: list[str] = []
    optional: list[str] = []

    def sec_metric(name: str, labels: tuple[str, ...], kind: str, unit: str, required_metric: bool = True,
                   *, allow_change: bool = False, prefer: tuple[str, ...] = (), reject: tuple[str, ...] = ()) -> None:
        parsed = _extract(sec_rows, labels, kind, unit, allow_change=allow_change, prefer=prefer, reject=reject)
        metrics[name] = _metric(parsed["value"] if parsed else None, parsed["unit"] if parsed else unit,
                                ["sec_filings"], "deterministic SEC semantic extraction with unit normalization",
                                parsed["raw"] if parsed else None, parsed.get("filing_date") if parsed else None,
                                parsed.get("source_url") if parsed else None)
        (required if required_metric else optional).append(name)

    if profile == "FINANCIALS":
        metrics["roe"] = _metric(fm.get("roe"), "ratio", ["fd_metrics"], "Financial Datasets normalized metric")
        required.append("roe")
        sec_metric("net_interest_margin", (r"net interest (?:margin|yield)", r"\bNIM\b"), "percent", "ratio")
        parsed_cet1 = _extract(sec_rows, (r"(?:standardized and advanced )?CET1(?: capital)? ratios? (?:were|was|of)",
                                           r"common equity tier 1 capital ratio (?:was|of)"), "percent", "ratio",
                                reject=(r"minimum|required|requirement|estimated impact",))
        metrics["cet1"] = _metric(parsed_cet1 and parsed_cet1["value"], "ratio", ["sec_filings"] if parsed_cet1 else [],
                                  "deterministic SEC semantic extraction with regulatory-threshold exclusion",
                                  parsed_cet1 and parsed_cet1["raw"], parsed_cet1 and parsed_cet1.get("filing_date"), parsed_cet1 and parsed_cet1.get("source_url"))
        required.append("cet1")
        sec_metric("loan_growth", (r"loan growth", r"(?:firmwide )?average loans?(?: and leases)?(?: of \$?[\d,.]+ (?:trillion|billion|million))? (?:were )?(?:up|down|grew|increased|decreased|declined)", r"loans? and leases (?:grew|increased|decreased|declined)"), "percent", "ratio", allow_change=True)
        sec_metric("deposit_growth", (r"deposit growth", r"(?:firmwide )?average deposits?(?: of \$?[\d,.]+ (?:trillion|billion|million))? (?:were )?(?:up|down|grew|increased|decreased|declined)", r"deposits? (?:grew|increased|decreased|declined)"), "percent", "ratio", allow_change=True)
        sec_metric("provision_credit_losses", (r"provision for (?:credit|loan) losses (?:was|were|totaled)",
                                                 r"provision for (?:credit|loan) losses (?:increased|decreased).*?\bto"), "money", "USD", False)
        sec_metric("net_charge_off_rate", (r"net charge[- ]off(?:s| rate| ratio)*", r"\bNCO(?:s| rate| ratio)?\b"), "percent", "ratio", False)
        sec_metric("credit_quality", (r"nonperforming (?:assets?|loans?) (?:ratio|rate)", r"nonaccrual loans? (?:ratio|rate)"), "percent", "ratio")
        efficiency = _extract(sec_rows, (r"efficiency ratio (?:was|of)",), "percent", "ratio") or _extract_table_value(sec_rows, (r"efficiency ratio",), "percent", "ratio")
        metrics["efficiency_ratio"] = _metric(efficiency and efficiency["value"], "ratio", ["sec_filings"] if efficiency else [],
                                              "deterministic SEC table/semantic extraction", efficiency and efficiency["raw"],
                                              efficiency and efficiency.get("filing_date"), efficiency and efficiency.get("source_url"))
        optional.append("efficiency_ratio")
    elif profile == "ENERGY":
        parsed_production = _extract(sec_rows, (r"(?:upstream|worldwide|company|total) (?:net )?(?:oil[- ]equivalent )?production(?: volumes?)?(?: averaged| was| of)?", r"\bproduction of"),
                                     "production", "BOE/day", prefer=(r"upstream production averaged", r"worldwide net oil[- ]equivalent production"),
                                     reject=(r"share of|project|field|reservoir|country|canada|kazakhstan",))
        metrics["production"] = _metric(parsed_production and parsed_production["value"], "BOE/day", ["sec_filings"] if parsed_production else [],
                                        "deterministic company-level SEC production extraction", parsed_production and parsed_production["raw"],
                                        parsed_production and parsed_production.get("filing_date"), parsed_production and parsed_production.get("source_url"))
        required.append("production")
        table_production = _extract_labeled_table_value(
            sec_rows, (r"oil-equivalent production",), unit="BOE/day", multiplier=1_000,
            required_context=(r"thousands of oil-equivalent barrels daily",))
        if table_production:
            metrics["production"] = _metric(table_production["value"], "BOE/day", ["sec_filings"],
                                            "deterministic SEC company-total production table extraction",
                                            table_production["raw"], table_production.get("filing_date"),
                                            table_production.get("source_url"))
        oil_production = _extract_production_table_total(
            sec_rows, r"net production of crude oil, natural gas liquids, bitumen and\s*synthetic oil",
            r"natural gas production", r"thousands of barrels daily", 1_000, "BOE/day") or _extract(
            sec_rows, (r"(?:total|worldwide) (?:crude )?oil production", r"(?:total|worldwide) liquids production"),
            "production", "BOE/day", reject=(r"share of|project|field|basin|reservoir|country",))
        gas_production = _extract_labeled_table_value(
            sec_rows, (r"total natural gas production available for sale",), unit="MMcf/day", multiplier=1,
            required_context=(r"oil-equivalent production",)) or _extract_production_table_total(
            sec_rows, r"natural gas production", None, r"millions of cubic feet daily", 1, "MMcf/day") or _extract(
            sec_rows, (r"(?:total |worldwide )?natural gas production", r"(?:total |worldwide )?gas production"),
            "production", "MMcf/day", reject=(r"share of|project|field|basin|reservoir|capacity",))
        for name, parsed, unit_name in (("oil_production", oil_production, "BOE/day"), ("gas_production", gas_production, "MMcf/day")):
            metrics[name] = _metric(parsed and parsed["value"], unit_name, ["sec_filings"] if parsed else [],
                                    "deterministic SEC production table total extraction", parsed and parsed["raw"],
                                    parsed and parsed.get("filing_date"), parsed and parsed.get("source_url"))
            optional.append(name)
        oil_price = _extract_table_value(sec_rows, (r"(?:crude|liquids?) realizations?",), "money", "USD/bbl", require=(r"\$/\s*BBL",), scale_money=False) or _extract(
            sec_rows, (r"realized (?:oil|crude|liquids?) (?:price|prices) (?:was|were|of)",), "money", "USD/bbl", reject=(r"increased|decreased",))
        gas_price = _extract_table_value(sec_rows, (r"natural gas realizations?",), "money", "USD/Mcf", require=(r"\$/\s*MCF",), scale_money=False) or _extract(
            sec_rows, (r"realized (?:natural )?gas (?:price|prices) (?:was|were|of)",), "money", "USD/Mcf", reject=(r"increased|decreased",))
        for name, parsed, unit_name in (("realized_oil_price", oil_price, "USD/bbl"), ("realized_gas_price", gas_price, "USD/Mcf")):
            metrics[name] = _metric(parsed and parsed["value"], unit_name, ["sec_filings"] if parsed else [],
                                    "deterministic SEC realization table extraction", parsed and parsed["raw"],
                                    parsed and parsed.get("filing_date"), parsed and parsed.get("source_url"))
            optional.append(name)
        upstream = _extract(sec_rows, (r"upstream (?:earnings|income|profit).{0,80}?(?:were|was|totaled)",), "money", "USD", reject=(r"driver analysis|increased earnings|decreased earnings",)) or _extract_preceding_total(sec_rows, (r"upstream earnings driver analysis",))
        downstream = _extract(sec_rows, (r"downstream (?:earnings|income|profit).{0,80}?(?:were|was|totaled)", r"energy products earnings.{0,80}?(?:were|was|totaled)"), "money", "USD", reject=(r"driver analysis|increased earnings|decreased earnings",)) or _extract_preceding_total(sec_rows, (r"(?:downstream|energy products) earnings driver analysis",))
        for name, parsed in (("upstream_earnings", upstream), ("downstream_earnings", downstream)):
            metrics[name] = _metric(parsed and parsed["value"], "USD", ["sec_filings"] if parsed else [],
                                    "deterministic SEC reported earnings/table-total extraction", parsed and parsed["raw"],
                                    parsed and parsed.get("filing_date"), parsed and parsed.get("source_url"))
            required.append(name)
        sec_metric("chemical_earnings", (r"chemical earnings", r"chemical products earnings"), "money", "USD", False)
        capex = _extract(sec_rows, (r"cash capex", r"capital expenditures?", r"capital spending"), "money", "USD") if latest.get("capital_expenditure") is None else None
        fcf = _extract(sec_rows, (r"(?:adjusted )?free cash flow",), "money", "USD") if latest.get("free_cash_flow") is None else None
        capex_value = abs(float(latest["capital_expenditure"])) if latest.get("capital_expenditure") is not None else capex and abs(capex["value"])
        metrics["capex"] = _metric(capex_value, "USD", ["fd_earnings"] if latest.get("capital_expenditure") is not None else (["sec_filings"] if capex else []), "Financial Datasets quarterly cash flow (cash outflow normalized positive)" if latest.get("capital_expenditure") is not None else "deterministic SEC semantic extraction with unit normalization", capex and capex["raw"], capex and capex.get("filing_date"), capex and capex.get("source_url"))
        metrics["free_cash_flow"] = _metric(latest.get("free_cash_flow") if latest.get("free_cash_flow") is not None else fcf and fcf["value"], "USD", ["fd_earnings"] if latest.get("free_cash_flow") is not None else (["sec_filings"] if fcf else []), "Financial Datasets quarterly cash flow" if latest.get("free_cash_flow") is not None else "deterministic SEC semantic extraction with unit normalization", fcf and fcf["raw"], fcf and fcf.get("filing_date"), fcf and fcf.get("source_url"))
        required.extend(["capex", "free_cash_flow"])
        sec_metric("reserve_production_ratio", (r"reserve[- ]to[- ]production ratio",), "quantity", "years", False)
        metrics["breakeven"] = _metric(None, "USD/bbl", [], "N/A")
        optional.append("breakeven")
        macro = modules.get("macro", {}).get("metrics", {})
        metrics["wti_crude"] = _metric(macro.get("wti_crude"), "USD/bbl", ["macro_snapshot"], "FRED/Yahoo macro snapshot")
        metrics["natural_gas"] = _metric(macro.get("natural_gas"), "USD/MMBtu", ["macro_snapshot"], "FRED/Yahoo macro snapshot")
        optional.extend(["wti_crude", "natural_gas"])
    elif profile == "AUTOMOTIVE":
        deliveries = _extract_company_vehicle_volume(sec_rows) or _extract(
            sec_rows, (r"(?:company|worldwide) total vehicle sales (?:were|totaled)", r"total vehicle deliveries (?:were|totaled)"),
            "quantity", "vehicles", reject=(r"revenue|changeover|increase of|decrease of|segment",))
        metrics["deliveries"] = _metric(deliveries and deliveries["value"], "vehicles", ["sec_filings"] if deliveries else [],
                                        "deterministic SEC unit/table extraction", deliveries and deliveries["raw"],
                                        deliveries and deliveries.get("filing_date"), deliveries and deliveries.get("source_url"))
        required.append("deliveries")
        auto_revenue = _extract(sec_rows, (r"automotive (?:net sales and |sales )?revenues? (?:was|were|totaled)",), "money", "USD", reject=(r"increased|decreased",)) or _extract_table_value(
            sec_rows, (r"automotive (?:net sales and revenue|sales)",), "money", "USD", reject=(r"cost of automotive",),
            require=(r"in millions|dollars in millions|dollars in billions|\$m\b|\$bn\b",))
        metrics["automotive_revenue"] = _metric(auto_revenue and auto_revenue["value"], "USD", ["sec_filings"] if auto_revenue else [],
                                                "deterministic SEC reported-level extraction", auto_revenue and auto_revenue["raw"],
                                                auto_revenue and auto_revenue.get("filing_date"), auto_revenue and auto_revenue.get("source_url"))
        required.append("automotive_revenue")
        auto_margin = _extract_transition_percent(sec_rows, (r"gross margin for total automotive", r"automotive gross margin")) or _extract(
            sec_rows, (r"automotive gross margin (?:was|of)",), "percent", "ratio") or _extract_table_value(sec_rows, (r"automotive gross margin",), "percent", "ratio")
        metrics["automotive_margin"] = _metric(auto_margin and auto_margin["value"], "ratio", ["sec_filings"] if auto_margin else [],
                                                     "deterministic SEC reported-margin extraction", auto_margin and auto_margin["raw"],
                                                     auto_margin and auto_margin.get("filing_date"), auto_margin and auto_margin.get("source_url"))
        metrics["automotive_gross_margin"] = dict(metrics["automotive_margin"])
        required.append("automotive_margin")
        sec_metric("asp", (r"average selling price (?:was|of)", r"\bASP (?:was|of)"), "money", "USD", False, reject=(r"increase|decrease|higher|lower",))
        inventory = _extract_inventory_table(sec_rows) or _extract(sec_rows, (r"(?:total )?inventor(?:y|ies) (?:was|were|of)",), "money", "USD", reject=(r"write[- ]downs?|adjustments?",)) or _extract_table_value(
            sec_rows, (r"total inventor(?:y|ies)",), "money", "USD", reject=(r"write[- ]downs?|adjustments?",),
            require=(r"in millions|dollars in millions|\$m\b",))
        metrics["inventory"] = _metric(inventory and inventory["value"], "USD", ["sec_filings"] if inventory else [],
                                       "deterministic SEC reported inventory level", inventory and inventory["raw"],
                                       inventory and inventory.get("filing_date"), inventory and inventory.get("source_url"))
        required.append("inventory")
        regulatory = _extract(sec_rows, (r"automotive regulatory credits? revenue (?:was|were|totaled)",), "money", "USD", reject=(r"increased|decreased|performance obligations",)) or _extract_table_value(
            sec_rows, (r"automotive regulatory credits",), "money", "USD", reject=(r"performance obligations|cost of",))
        metrics["regulatory_credits"] = _metric(regulatory and regulatory["value"], "USD", ["sec_filings"] if regulatory else [],
                                                "deterministic SEC reported regulatory-credit revenue", regulatory and regulatory["raw"],
                                                regulatory and regulatory.get("filing_date"), regulatory and regulatory.get("source_url"))
        optional.append("regulatory_credits")
        capex = _extract(sec_rows, (r"capital expenditures?", r"capital spending", r"cash capex"), "money", "USD") if latest.get("capital_expenditure") is None else None
        fcf = _extract(sec_rows, (r"(?:company )?(?:adjusted )?free cash flow",), "money", "USD") if latest.get("free_cash_flow") is None else None
        capex_value = abs(float(latest["capital_expenditure"])) if latest.get("capital_expenditure") is not None else capex and abs(capex["value"])
        metrics["capex"] = _metric(capex_value, "USD", ["fd_earnings"] if latest.get("capital_expenditure") is not None else (["sec_filings"] if capex else []), "Financial Datasets quarterly cash flow (cash outflow normalized positive)" if latest.get("capital_expenditure") is not None else "deterministic SEC semantic extraction with unit normalization", capex and capex["raw"], capex and capex.get("filing_date"), capex and capex.get("source_url"))
        metrics["free_cash_flow"] = _metric(latest.get("free_cash_flow") if latest.get("free_cash_flow") is not None else fcf and fcf["value"], "USD", ["fd_earnings"] if latest.get("free_cash_flow") is not None else (["sec_filings"] if fcf else []), "Financial Datasets quarterly cash flow" if latest.get("free_cash_flow") is not None else "deterministic SEC semantic extraction with unit normalization", fcf and fcf["raw"], fcf and fcf.get("filing_date"), fcf and fcf.get("source_url"))
        required.extend(["capex", "free_cash_flow"])
        energy_revenue = _extract(sec_rows, (r"energy generation and storage revenues? (?:was|were|totaled)", r"energy revenues? (?:was|were|totaled)"), "money", "USD", reject=(r"increased|decreased|cost of|credits? earned",)) or _extract_table_value(
            sec_rows, (r"energy generation and storage sales",), "money", "USD", reject=(r"cost of",))
        metrics["energy_revenue"] = _metric(energy_revenue and energy_revenue["value"], "USD", ["sec_filings"] if energy_revenue else [],
                                            "deterministic SEC reported-level extraction", energy_revenue and energy_revenue["raw"],
                                            energy_revenue and energy_revenue.get("filing_date"), energy_revenue and energy_revenue.get("source_url"))
        optional.append("energy_revenue")
        sec_metric("energy_growth", (r"energy (?:generation and storage )?revenue growth", r"energy (?:generation and storage )?revenues? (?:grew|increased|decreased|declined)"), "percent", "ratio", False, allow_change=True)
        segment_margin = _extract(sec_rows, (r"(?:company|automotive segment) adjusted EBIT margin (?:was|of)",), "percent", "ratio")
        metrics["segment_margin"] = _metric(segment_margin and segment_margin["value"], "ratio", ["sec_filings"] if segment_margin else [],
                                            "deterministic issuer-defined segment margin extraction", segment_margin and segment_margin["raw"],
                                            segment_margin and segment_margin.get("filing_date"), segment_margin and segment_margin.get("source_url"))
        optional.append("segment_margin")
    else:
        for name in ("revenue_growth", "gross_margin", "operating_margin", "roic"):
            metrics[name] = _metric(fm.get(name), "ratio", ["fd_metrics"], "Financial Datasets normalized metric")
            required.append(name)
        metrics["free_cash_flow"] = _metric(latest.get("free_cash_flow"), "USD", ["fd_earnings"], "Financial Datasets quarterly cash flow")
        optional.append("free_cash_flow")
        if profile == "TECHNOLOGY":
            for name in ("research_and_development", "stock_based_compensation"):
                metrics[name] = _metric(latest.get(name), "USD", ["fd_earnings"], "Financial Datasets quarterly statement")
                optional.append(name)

    configured = INDUSTRY_METRIC_REQUIREMENTS.get(profile, {"required": required, "optional": optional})
    required = list(configured["required"])
    optional = list(configured["optional"])
    period = str(latest.get("period") or "") or None
    for name, item in metrics.items():
        _enrich_metric_semantics(profile, name, item, period)
    available = [name for name in required if metrics.get(name, {}).get("available")]
    optional_available = [name for name in optional if metrics.get(name, {}).get("available")]
    return {"profile": profile, "metrics": metrics, "required_metrics": required, "optional_metrics": optional,
            "available_metrics": available, "available_count": len(available), "required_count": len(required),
            "optional_available_metrics": optional_available, "optional_available_count": len(optional_available),
            "optional_count": len(optional), "required_coverage": round(len(available) / max(1, len(required)), 3),
            "optional_coverage": round(len(optional_available) / max(1, len(optional)), 3),
            "completeness": round(len(available) / max(1, len(required)), 3),
            "sources": sorted({source for item in metrics.values() for source in item.get("sources", [])}),
            "driver_taxonomy": DRIVER_TAXONOMY[profile], "risk_taxonomy": RISK_TAXONOMY[profile], "no_estimation": True}


def industry_score(industry: dict) -> dict:
    profile, metrics = industry["profile"], industry["metrics"]
    def value(name: str) -> float | None:
        return _num(metrics.get(name, {}).get("value"))
    rules: dict[str, tuple[float | None, float, float]] = {}
    if profile == "FINANCIALS":
        rules = {"PROFITABILITY": (value("roe"), .08, .18), "CAPITAL_STRENGTH": (value("cet1"), .08, .14),
                 "CREDIT_QUALITY": (value("net_charge_off_rate"), .04, .005), "LOAN_GROWTH": (value("loan_growth"), -.03, .12),
                 "DEPOSIT_GROWTH": (value("deposit_growth"), -.03, .12), "RATE_SENSITIVITY": (value("net_interest_margin"), .015, .04),
                 "EFFICIENCY": (value("efficiency_ratio"), .75, .45)}
    elif profile == "ENERGY":
        rules = {"PRODUCTION": (value("production"), None, None), "REALIZED_PRICE": (value("realized_oil_price"), None, None),
                 "UPSTREAM": (value("upstream_earnings"), None, None), "DOWNSTREAM": (value("downstream_earnings"), None, None),
                 "CAPEX": (value("capex"), None, None), "FCF": (value("free_cash_flow"), 0, None),
                 "COMMODITY_EXPOSURE": (value("wti_crude"), None, None), "RESERVES": (value("reserve_production_ratio"), None, None)}
    elif profile == "AUTOMOTIVE":
        rules = {"DELIVERIES": (value("deliveries"), None, None), "PRICING": (value("asp"), None, None),
                 "AUTOMOTIVE_MARGIN": (value("automotive_margin"), .10, .25), "INVENTORY": (value("inventory"), None, None),
                 "CAPEX": (value("capex"), None, None), "FCF": (value("free_cash_flow"), 0, None),
                 "REGULATORY": (value("regulatory_credits"), None, None), "ENERGY_BUSINESS": (value("energy_growth"), 0, .30)}
    scores = {}
    for name, (raw, low, high) in rules.items():
        if raw is None or (low is None and high is None):
            scores[name] = None
        elif high is None:
            scores[name] = 75 if raw > low else 30
        elif low > high:
            scores[name] = max(0, min(100, round(100 * (low - raw) / max(abs(low - high), 1e-9))))
        else:
            scores[name] = max(0, min(100, round(100 * (raw - low) / max(high - low, 1e-9))))
    available = [score for score in scores.values() if score is not None]
    return {"profile": profile, "overall": round(sum(available) / len(available)) if available else None,
            "components": scores, "coverage": industry["completeness"],
            "scoring_confidence": round(industry["completeness"] * .9, 3)}


def industry_findings(ticker: str, industry: dict) -> list[dict]:
    profile, metrics = industry["profile"], industry["metrics"]
    definitions = {
        "FINANCIALS": [("roe", "PROFITABILITY", .12, None), ("cet1", "CAPITAL_STRENGTH", .12, None), ("loan_growth", "LOAN_GROWTH", 0, None), ("deposit_growth", "DEPOSIT_GROWTH", 0, None), ("net_charge_off_rate", "CREDIT_QUALITY", None, .02), ("efficiency_ratio", "EFFICIENCY", None, .60)],
        "ENERGY": [("free_cash_flow", "FCF", 0, None), ("production", "PRODUCTION", None, None), ("realized_oil_price", "REALIZED_PRICE", None, None), ("upstream_earnings", "UPSTREAM", 0, None), ("downstream_earnings", "DOWNSTREAM", 0, None)],
        "AUTOMOTIVE": [("automotive_margin", "AUTOMOTIVE_MARGIN", .18, None), ("deliveries", "DELIVERIES", None, None), ("free_cash_flow", "FCF", 0, None), ("energy_growth", "ENERGY_BUSINESS", 0, None), ("inventory", "INVENTORY", None, None)],
    }.get(profile, [])
    rows = []
    for metric_name, driver_type, positive_floor, positive_ceiling in definitions:
        item = metrics.get(metric_name, {})
        value = _num(item.get("value"))
        if value is None:
            continue
        if positive_floor is None and positive_ceiling is None:
            direction = "NEUTRAL"
        elif positive_ceiling is not None:
            direction = "POSITIVE" if value <= positive_ceiling else "NEGATIVE"
        else:
            direction = "POSITIVE" if value >= positive_floor else "NEGATIVE"
        rows.append({"ticker": ticker, "category": "industry", "driver_type": driver_type, "metric": metric_name,
                     "title": metric_name.replace("_", " ").title(), "claim": f"{metric_name} is {value:g} {item.get('unit') or ''} on the latest available reported basis.",
                     "direction": direction, "importance": "CRITICAL", "confidence": .88,
                     "source_ids": item.get("sources", []), "metrics": {metric_name: value,
                     "metric_definition": item.get("definition"), "source_definition": item.get("source_definition"),
                     "period": item.get("period")}})
    return rows
