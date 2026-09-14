"""Dated history for one stock: SEC filings from EDGAR and the monitor's anomaly memory.

Both answer "what happened around these dates" for a drawdown question;
neither is a substitute for reading the filing or the news.
"""

from __future__ import annotations

import hashlib
from datetime import date, timedelta
from typing import Any, Callable

from v2.agent_v2.execution import CapabilityRegistry, ExecutionContext
from v2.agent_v2.models import EvidenceItem, ResultStatus, ToolEnvelope


def _default_filings(ticker: str, form: str, since: str, until: str) -> list[Any]:
    from v2.bot.responders import _build_sec_filing_for_view
    from v2.sec import client as sec_client

    rows = []
    for raw in sec_client.get_recent_filings(ticker, form, since, until) or []:
        filing = _build_sec_filing_for_view(raw, ticker, form=form)
        if filing is not None:
            rows.append(filing)
    return rows


def _default_anomalies(ticker: str, query: str, lookback_days: int) -> list[Any]:
    from v2.memory import AnomalyMemory

    return AnomalyMemory().recall(ticker, query, lookback_days=lookback_days, n_results=6)


def _evidence_id(kind: str, ticker: str, key: str) -> str:
    return f"evidence-{kind}-{hashlib.sha1(f'{kind}|{ticker}|{key}'.encode('utf-8')).hexdigest()[:16]}"


_FORM_LABELS = {"4": "Form 4（内幕交易）", "3": "Form 3（内幕人初始持股）", "144": "Form 144（拟出售通知）", "13D": "13D", "13G": "13G"}


def _form_label(form: str) -> str:
    return _FORM_LABELS.get(str(form).upper(), str(form))


def filings_envelope(ticker: str, context: ExecutionContext, fetch: Callable[[str, str, str, str], list[Any]], *, since: str = "", until: str = "", forms: list[str] | None = None, today: date | None = None) -> ToolEnvelope:
    current = today or date.today()
    until = until or current.isoformat()
    since = since or (current - timedelta(days=365)).isoformat()
    explicit = [str(form) for form in (forms or []) if form]
    forms = explicit or ["8-K"]
    rows: list[Any] = []
    errors: list[str] = []

    def fetch_form(form: str) -> None:
        try:
            rows.extend(fetch(ticker, form, since, until))
        except Exception as exc:  # noqa: BLE001 — one form failing is data, not a crash
            errors.append(f"{form}: {type(exc).__name__}: {str(exc)[:120]}")

    for form in forms:
        fetch_form(form)
    # A foreign private issuer (ARM, TSM, BABA) files 6-K instead of 8-K;
    # with no explicit form list and no 8-K, look there before saying "none".
    if not explicit and not rows and "8-K" in forms:
        forms = [*forms, "6-K"]
        fetch_form("6-K")
    rows.sort(key=lambda row: str(getattr(row, "filing_date", "")), reverse=True)
    rows = rows[:12]
    evidence: list[EvidenceItem] = []
    for row in rows:
        filing_date = str(getattr(row, "filing_date", "") or "")[:10]
        form = str(getattr(row, "form", "") or "")
        accession = str(getattr(row, "accession_number", "") or "")
        cik = str(getattr(row, "cik", "") or "").lstrip("0")
        url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession.replace('-', '')}/" if cik and accession else ""
        evidence.append(
            EvidenceItem(
                id=_evidence_id("filing", ticker, accession or f"{form}:{filing_date}"),
                entity=ticker,
                claim=f"{ticker} 于 {filing_date} 向 SEC 提交了 {_form_label(form)}（{accession}）。",
                as_of=filing_date,
                source_id="sec_edgar",
                source_title=f"{ticker} {_form_label(form)} {filing_date}",
                source_url=url,
                producer_run_id=context.run_id,
                metadata={"evidence_scope": "filing", "date": filing_date, "form": form, "accession": accession},
            )
        )
    label = "、".join(forms)
    if not evidence:
        note = f"{ticker} {since} 至 {until} 未查到 {label} 申报。"
        evidence.append(EvidenceItem(id=_evidence_id("filing-none", ticker, f"{since}:{until}"), entity=ticker, claim=note, source_id="sec_edgar", source_title="SEC EDGAR", producer_run_id=context.run_id, metadata={"citation_kind": "limitations", "verified": True}))
    status = ResultStatus.FAILED if errors and not rows else ResultStatus.PARTIAL_ERROR if errors else ResultStatus.COMPLETED
    narrative = f"{ticker} {since} 至 {until} 的 {label} 申报：" + ("；".join(f"{item.metadata['date']} {item.metadata['form']}[{item.id}]" for item in evidence if item.metadata.get("evidence_scope") == "filing") or f"未查到[{evidence[0].id}]") + "。"
    return ToolEnvelope(
        "filings.recent",
        status,
        subject=ticker,
        as_of=until,
        summary=f"{ticker} {since} 至 {until} 共 {len(rows)} 份 {label} 申报。",
        metrics={"count": len(rows), "since": since, "until": until, "dates": [item.metadata["date"] for item in evidence if item.metadata.get("evidence_scope") == "filing"]},
        evidence=evidence,
        limitations=["此处只列日期和表格类型；申报内容由申报阅读者按下跌日读取。"] if rows else [],
        errors=errors,
        metadata={"narrative": narrative, "dates": [item.metadata["date"] for item in evidence if item.metadata.get("evidence_scope") == "filing"]},
    )


def anomaly_history_envelope(ticker: str, context: ExecutionContext, recall: Callable[[str, str, int], list[Any]], *, lookback_days: int = 90, query: str = "") -> ToolEnvelope:
    query = query or f"{ticker} 大跌 下跌 异动"
    try:
        rows = list(recall(ticker, query, int(lookback_days)))
    except Exception as exc:  # noqa: BLE001 — the memory store is optional infrastructure
        return ToolEnvelope("market.anomaly_history", ResultStatus.FAILED, subject=ticker, errors=[f"anomaly memory unavailable: {type(exc).__name__}: {str(exc)[:120]}"])
    rows.sort(key=lambda row: str(getattr(row, "date", "")), reverse=True)
    evidence: list[EvidenceItem] = []
    for row in rows:
        if len(evidence) >= 6:
            break
        day = str(getattr(row, "date", "") or "")[:10]
        flags = str(getattr(row, "flags", "") or "")
        doc = " ".join(str(getattr(row, "doc", "") or "").split())[:240]
        if not flags and doc.strip("。 .").upper() in {"", ticker.upper()}:
            continue  # a record with neither flags nor content says nothing
        evidence.append(
            EvidenceItem(
                id=_evidence_id("anomaly", ticker, f"{day}:{flags}"),
                entity=ticker,
                claim=f"{ticker} {day} 盯盘记录：{flags or '无标志'}；{doc}" if doc else f"{ticker} {day} 盯盘记录：{flags or '无标志'}。",
                as_of=day,
                source_id="anomaly_memory",
                source_title=f"{ticker} anomaly {day}",
                producer_run_id=context.run_id,
                metadata={"evidence_scope": "anomaly", "date": day, "flags": flags},
            )
        )
    if not evidence:
        note = f"{ticker} 近 {int(lookback_days)} 天盯盘系统没有记录到异动。"
        evidence.append(EvidenceItem(id=_evidence_id("anomaly-none", ticker, str(lookback_days)), entity=ticker, claim=note, source_id="anomaly_memory", source_title="Anomaly memory", producer_run_id=context.run_id, metadata={"citation_kind": "limitations", "verified": True}))
    narrative = f"{ticker} 盯盘记录：" + ("；".join(f"{item.metadata['date']}（{item.metadata['flags'] or '无标志'}）[{item.id}]" for item in evidence if item.metadata.get("evidence_scope") == "anomaly") or f"近 {int(lookback_days)} 天无记录[{evidence[0].id}]") + "。"
    return ToolEnvelope(
        "market.anomaly_history",
        ResultStatus.COMPLETED,
        subject=ticker,
        summary=f"{ticker} 近 {int(lookback_days)} 天盯盘记录 {len(rows)} 条。",
        metrics={"count": len(rows), "dates": [item.metadata["date"] for item in evidence if item.metadata.get("evidence_scope") == "anomaly"]},
        evidence=evidence,
        metadata={"narrative": narrative, "dates": [item.metadata["date"] for item in evidence if item.metadata.get("evidence_scope") == "anomaly"]},
    )


def register_history_capabilities(
    registry: CapabilityRegistry,
    *,
    filings_fetch: Callable[[str, str, str, str], list[Any]] = _default_filings,
    anomaly_recall: Callable[[str, str, int], list[Any]] = _default_anomalies,
    today_factory: Callable[[], date] = date.today,
) -> None:
    def filings(arguments: dict[str, Any], context: ExecutionContext) -> ToolEnvelope:
        return filings_envelope(
            str(arguments.get("ticker") or "").upper(),
            context,
            filings_fetch,
            since=str(arguments.get("since") or ""),
            until=str(arguments.get("until") or ""),
            forms=[str(value) for value in (arguments.get("forms") or [])],
            today=today_factory(),
        )

    def anomalies(arguments: dict[str, Any], context: ExecutionContext) -> ToolEnvelope:
        return anomaly_history_envelope(
            str(arguments.get("ticker") or "").upper(),
            context,
            anomaly_recall,
            lookback_days=int(arguments.get("lookback_days") or 90),
            query=str(arguments.get("query") or ""),
        )

    registry.register("filings.recent", filings)
    registry.register("market.anomaly_history", anomalies)
