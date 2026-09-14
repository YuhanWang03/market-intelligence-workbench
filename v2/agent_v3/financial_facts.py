"""Original SEC XBRL facts, explicit duration joins and reproducible ratios.

No language inference: taxonomy tags and duration ranges are data contracts.
Never combine quarterly income with year-to-date cash flows or future filings.
"""
from datetime import date
from decimal import Decimal, InvalidOperation
from functools import lru_cache
import hashlib
import time

from v2.agent_v2.models import EvidenceItem


TAGS = {
    "revenue": ("RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet"),
    "gross_profit": ("GrossProfit",),
    "operating_income": ("OperatingIncomeLoss",),
    "operating_cash_flow": ("NetCashProvidedByUsedInOperatingActivities",),
    "capex": ("PaymentsToAcquirePropertyPlantAndEquipment",),
}


@lru_cache(maxsize=128)
def _fetch(ticker, cache_bucket):
    import os
    import httpx
    from edgar import Company, get_identity
    from v2.sec.client import _ensure_identity, _throttle
    _ensure_identity()
    _throttle()
    cik = int(Company(ticker).cik)
    _throttle()
    response = httpx.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json",
                         headers={"User-Agent": os.environ.get("SEC_USER_AGENT") or get_identity()}, timeout=12)
    response.raise_for_status()
    payload = response.json()
    if int(payload.get("cik", -1)) != cik:
        raise ValueError("SEC company identity mismatch")
    return payload


def load_facts(ticker):
    return _fetch(ticker, int(time.time() // 3600))


def normalized_facts(payload, as_of):
    """Preserve conflicting tags; select latest available restatement per tag."""
    cutoff = date.fromisoformat(as_of)
    selected = {}
    for metric, tags in TAGS.items():
        for tag in tags:
            for unit, rows in payload.get("facts", {}).get("us-gaap", {}).get(tag, {}).get("units", {}).items():
                if unit != "USD":
                    continue
                for row in rows:
                    try:
                        start, end, filed = (date.fromisoformat(row[key]) for key in ("start", "end", "filed"))
                        value = Decimal(str(row["val"]))
                        duration = (end - start).days + 1
                        window = "quarter" if 75 <= duration <= 105 else "annual" if 350 <= duration <= 380 else ""
                        if not value.is_finite() or not window or end > cutoff or filed > cutoff or filed < end:
                            continue
                        if row.get("form") not in {"10-K", "10-Q", "10-K/A", "10-Q/A"} or not row.get("accn"):
                            continue
                    except (KeyError, ValueError, TypeError, InvalidOperation):
                        continue
                    key = (metric, tag, str(start), str(end), unit)
                    candidate = {**row, "metric": metric, "tag": tag, "unit": unit, "value": value, "window": window}
                    current = selected.get(key)
                    if current is None or (row["filed"], row["accn"]) > (current["filed"], current["accn"]):
                        selected[key] = candidate
    groups = {}
    for row in selected.values():
        groups.setdefault((row["start"], row["end"], row["unit"], row["window"]), {}).setdefault(row["metric"], []).append(row)
    return groups


def financial_evidence(payload, ticker, as_of):
    groups = normalized_facts(payload, as_of)
    output, diagnostics = [], []
    cik = int(payload["cik"])
    # Two latest periods of each type; quarterly and annual remain distinct.
    keys = []
    for window in ("quarter", "annual"):
        keys.extend(sorted((key for key in groups if key[3] == window), key=lambda key: key[1], reverse=True)[:2])
    for start, end, unit, window in keys:
        inputs = {}
        for metric, rows in groups[(start, end, unit, window)].items():
            if len({row["value"] for row in rows}) != 1:
                diagnostics.append({"metric": metric, "period_start": start, "period_end": end, "reason": "conflicting_taxonomy_values"})
                continue
            inputs[metric] = max(rows, key=lambda row: (row["filed"], row["accn"]))
        formulas = {
            "gross_margin": ("gross_profit", "revenue", "divide"),
            "operating_margin": ("operating_income", "revenue", "divide"),
            "free_cash_flow": ("operating_cash_flow", "capex", "subtract"),
        }
        for metric, (left, right, operation) in formulas.items():
            if left not in inputs or right not in inputs:
                diagnostics.append({"metric": metric, "period_end": end, "reason": "missing_components"})
                continue
            a, b = inputs[left], inputs[right]
            # Restated and original components from different filings are not joined.
            if a["accn"] != b["accn"] or (operation == "divide" and b["value"] <= 0) or (operation == "subtract" and b["value"] < 0):
                diagnostics.append({"metric": metric, "period_end": end, "reason": "incompatible_components"})
                continue
            value = a["value"] / b["value"] if operation == "divide" else a["value"] - b["value"]
            result_unit = "ratio" if operation == "divide" else unit
            url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{a['accn'].replace('-', '')}/{a['accn']}-index.html"
            identity = hashlib.sha256(f"{ticker}|{metric}|{start}|{end}|{a['accn']}".encode()).hexdigest()[:16]
            label = {"gross_margin": "毛利率", "operating_margin": "营业利润率", "free_cash_flow": "自由现金流（经营现金流减购建固定资产支出）"}[metric]
            display = f"{value * 100:.2f}%" if result_unit == "ratio" else f"{value:,.2f} USD"
            components = [{k: row[k] for k in ("tag", "start", "end", "filed", "accn", "unit", "val")} for row in (a, b)]
            output.append(EvidenceItem(f"sec-fin-{identity}", ticker,
                f"{ticker} {start} 至 {end} {label}：{display}；按同份 SEC 申报的 US-GAAP 原始项目计算。",
                metric=metric, value=float(value), unit=result_unit, period=f"{start}/{end}", as_of=a["filed"],
                source_id=a["accn"], source_title="SEC XBRL 原始财报项目计算", source_url=url,
                metadata={"period_start": start, "period_end": end, "measurement_window": window,
                          "accounting_basis": "US-GAAP components", "formula": f"{left} {operation} {right}",
                          "components": components, "original_financial_fact": True,
                          "metrics": {metric: float(value)}, "verified": True}))
    return output, diagnostics


def enrich_financials(envelope, context, loader=load_facts):
    from dataclasses import replace
    run = getattr(context, "v3_run", None)
    evidence, diagnostics = [], []
    for ticker in dict.fromkeys(envelope.subject.split(",")):
        if not ticker:
            continue
        try:
            if run:
                run.check(reserve=18)
            items, gaps = financial_evidence(loader(ticker), ticker, date.today().isoformat())
            evidence.extend(items)
            diagnostics.append({"ticker": ticker, "status": "available" if items else "no_usable_facts", "gaps": gaps})
        except Exception as exc:
            diagnostics.append({"ticker": ticker, "status": "unavailable", "error_type": type(exc).__name__})
    reconciliations = []
    for old in envelope.evidence:
        for new in evidence:
            if old.entity != new.entity or old.metric != new.metric:
                continue
            same_basis = (old.unit == new.unit and old.metadata.get('period_start') == new.metadata['period_start']
                          and old.metadata.get('period_end') == new.metadata['period_end']
                          and old.metadata.get('accounting_basis') == new.metadata['accounting_basis'])
            status = 'not_comparable'
            if same_basis and isinstance(old.value, (int, float)) and not isinstance(old.value, bool):
                status = 'matched' if abs(old.value - new.value) <= max(abs(new.value) * .001, .000001) else 'conflict'
            reconciliations.append({'provider_evidence': old.id, 'original_evidence': new.id, 'status': status})
    limits = list(envelope.limitations)
    if evidence:
        limits.append('已补充 SEC 原始财报计算项；原供应商指标若口径不明，不能与补充项视为同期间数据。不同公司财年起止日不同，不构成严格同期比较。')
    if any(row['status'] != 'available' or row.get('gaps') for row in diagnostics):
        limits.append('原始财报补充仍有缺口，读取失败、组成项目缺失和口径冲突已分开记录；未补出项目不能估填。')
    return replace(envelope, evidence=[*envelope.evidence, *evidence], limitations=limits,
                   metadata={**envelope.metadata, "financial_recovery": diagnostics, 'financial_reconciliation': reconciliations})
