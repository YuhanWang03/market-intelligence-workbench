"""Research Engine adapters with compact, evidence-preserving output."""

from __future__ import annotations

import hashlib
import json
import logging
from v2.usage_context import ContextExecutor as ThreadPoolExecutor
from dataclasses import replace
from typing import Any, Callable

from v2.agent_v2.execution import CapabilityRegistry, ExecutionContext
from v2.agent_v2.models import EvidenceItem, ResultStatus, ToolEnvelope

logger = logging.getLogger(__name__)

_FOCUS_MODULES = {
    "overview": ["fundamental", "valuation", "earnings", "risk"],
    "fundamentals": ["fundamental"],
    "valuation": ["valuation"],
    "earnings": ["earnings", "expectations", "sec"],
    "market": ["technical", "fund_flow"],
    "ownership": ["institutional"],
    "catalysts": ["catalyst"],
    "filings": ["sec"],
    "supply_chain": ["supply_chain"],
    # The engine's own "risk" module depends on supply_chain, whose neighbour
    # discovery (Tavily) took a risk question to 150 s; the risk answer is
    # written from these modules instead.
    "risk": ["fundamental", "valuation", "sec", "macro", "catalyst"],
    "full": None,
}


def _engine_factory():
    from v2.research import ResearchEngine

    return ResearchEngine()


def _store_factory():
    from v2.research.store import ResearchStore

    return ResearchStore()


def _status(raw: str, cache_hit: bool) -> ResultStatus:
    if cache_hit:
        return ResultStatus.CACHED
    value = (raw or "").upper()
    if value == "COMPLETED":
        return ResultStatus.COMPLETED
    if value in {"PARTIAL", "PARTIAL_DATA"}:
        return ResultStatus.PARTIAL_DATA
    if value == "PARTIAL_ERROR":
        return ResultStatus.PARTIAL_ERROR
    return ResultStatus.FAILED


def _evidence(result: dict[str, Any]) -> list[EvidenceItem]:
    ticker = str(result.get("ticker") or "")
    run_id = str(result.get("run_id") or "")
    sources = {str(row.get("id")): row for row in result.get("sources", [])}
    items: list[EvidenceItem] = []
    used: dict[str, EvidenceItem] = {}
    for row in result.get("evidence_index", [])[:40]:
        source_ids = [str(value) for value in row.get("source_ids", []) if value]
        source = next((sources[value] for value in source_ids if value in sources), {})
        metrics = row.get("metrics") or {}
        metric = next(iter(metrics), "")
        item = EvidenceItem(
            id=str(row.get("id") or f"{run_id}-evidence-{len(items) + 1}"),
            entity=str(row.get("ticker") or ticker),
            claim=str(row.get("claim") or ""),
            metric=metric,
            value=metrics.get(metric) if metric else None,
            unit=str(row.get("unit") or ""),
            period=str(row.get("data_period") or ""),
            as_of=str(row.get("published_at") or row.get("fetched_at") or result.get("generated_at") or ""),
            source_id=source_ids[0] if source_ids else "",
            source_title=str(source.get("title") or ""),
            source_url=str(source.get("url") or ""),
            confidence=None,
            producer_run_id=run_id,
            metadata={"module": row.get("module"), "verified": bool(row.get("verified")), "metrics": metrics,
                      **{key: row[key] for key in ("period_start", "period_end", "accounting_basis", "measurement_window") if row.get(key)}},
        )
        current = used.get(item.id)
        if current == item:
            continue
        if current is not None:
            original_id = item.id
            digest = hashlib.sha1(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:8]
            candidate_id = f"{original_id}-{digest}"
            item = replace(
                item,
                id=candidate_id,
                metadata={**item.metadata, "original_evidence_id": original_id, "collision_disambiguated": True},
            )
            if used.get(candidate_id) == item:
                continue
            sequence = 2
            while candidate_id in used:
                candidate_id = f"{original_id}-{digest}-{sequence}"
                sequence += 1
            if candidate_id != item.id:
                item = replace(item, id=candidate_id)
        used[item.id] = item
        items.append(item)
    return items


def _limitations(result: dict[str, Any]) -> list[str]:
    limitations = list(result.get("confidence_limitations") or [])
    diagnostics = result.get("production_diagnostics", {}).get("modules", {})
    for name, row in diagnostics.items():
        if row.get("status") in {"FAILED", "PARTIAL", "PARTIAL_DATA", "PARTIAL_ERROR"}:
            limitations.append(f"{name}: {row.get('status')}")
    return list(dict.fromkeys(str(value) for value in limitations if value))[:12]


def _derived_evidence(result: dict[str, Any], limitations: list[str]) -> list[EvidenceItem]:
    """Make engine-derived metrics and limitations citeable like source evidence."""

    ticker = str(result.get("ticker") or "")
    run_id = str(result.get("run_id") or "")
    generated_at = str(result.get("generated_at") or "")
    diagnostics = result.get("production_diagnostics", {}).get("modules", {}) or {}
    module_diagnostics = {
        str(name): {
            "status": row.get("status"),
            "completeness": row.get("completeness"),
            "missing_fields": list(row.get("missing_fields") or [])[:8],
        }
        for name, row in diagnostics.items()
        if isinstance(row, dict)
    }
    metrics = {
        "scores": result.get("scores") or {},
        "risk_level": result.get("risk_level"),
        "confidence": result.get("research_confidence") or {},
        "module_diagnostics": module_diagnostics,
    }
    items: list[EvidenceItem] = []
    if any(value for value in metrics.values()):
        encoded = json.dumps(metrics, ensure_ascii=False, sort_keys=True, default=str)
        digest = hashlib.sha1(f"{ticker}|{run_id}|metrics|{encoded}".encode("utf-8")).hexdigest()[:16]
        items.append(
            EvidenceItem(
                id=f"evidence-research-metrics-{digest}",
                entity=ticker,
                claim=f"{ticker} Research Engine 派生评分与风险指标：{encoded}",
                as_of=generated_at,
                source_id="research_engine",
                source_title="Research Engine derived metrics",
                producer_run_id=run_id,
                metadata={"citation_kind": "metrics", "metrics": metrics, "verified": True},
            )
        )
    if limitations:
        encoded = json.dumps(limitations, ensure_ascii=False, default=str)
        completeness_notes = [
            f"{name} 数据完整度 {float(row['completeness']):.1%}"
            for name, row in module_diagnostics.items()
            if isinstance(row.get("completeness"), (int, float))
        ]
        digest = hashlib.sha1(f"{ticker}|{run_id}|limitations|{encoded}".encode("utf-8")).hexdigest()[:16]
        items.append(
            EvidenceItem(
                id=f"evidence-research-limitations-{digest}",
                entity=ticker,
                claim=f"{ticker} Research Engine 已知数据限制：{'；'.join(limitations)}" + (f"；模块诊断：{'；'.join(completeness_notes)}" if completeness_notes else ""),
                as_of=generated_at,
                source_id="research_engine",
                source_title="Research Engine data limitations",
                producer_run_id=run_id,
                metadata={"citation_kind": "limitations", "limitations": limitations, "module_diagnostics": module_diagnostics, "verified": True},
            )
        )
    return items


#: Modules a stock's numbers come from; when they fail the engine's "completed" is a hollow run.
_CORE_MODULES = ("fundamental", "valuation")


def _core_failures(result: dict[str, Any]) -> list[str]:
    diagnostics = result.get("production_diagnostics", {}).get("modules", {}) or {}
    return [name for name in _CORE_MODULES if str((diagnostics.get(name) or {}).get("status") or "").upper() == "FAILED"]


def _module_metrics(result: dict[str, Any]) -> dict[str, dict[str, float]]:
    """The numeric metrics of the valuation, fundamental and earnings modules, for side-by-side rows."""

    modules = result.get("modules") or {}
    out: dict[str, dict[str, float]] = {}
    for name in _TABLE_MODULES:
        module = modules.get(name) if isinstance(modules, dict) else None
        metrics = module.get("metrics") if isinstance(module, dict) else None
        if not isinstance(metrics, dict):
            continue
        numeric = {str(key): float(value) for key, value in metrics.items() if isinstance(value, (int, float)) and not isinstance(value, bool) and key in _COMPARABLE_METRICS}
        if numeric:
            out[name] = numeric
    return out


def _envelope(result: dict[str, Any], capability: str) -> ToolEnvelope:
    findings = list(result.get("research_findings") or [])[:12]
    limitations = _limitations(result)
    status = _status(str(result.get("status") or ""), bool(result.get("from_cache")))
    core_failures = _core_failures(result)
    if core_failures and status == ResultStatus.COMPLETED:
        # A compare with both valuation modules failed read "completed" and
        # scored 50 on nothing; the run is partial and the answer must say so.
        status = ResultStatus.PARTIAL_DATA
        limitations.append("核心模块失败（" + "、".join(core_failures) + "）：估值和基本面数字缺失，数据源可能临时不可用，无法做同口径比较")
    derived = _derived_evidence(result, limitations)
    source_evidence = _evidence(result)
    return ToolEnvelope(
        capability=capability,
        status=status,
        subject=str(result.get("ticker") or ""),
        as_of=str(result.get("generated_at") or ""),
        summary=str(result.get("core_thesis") or result.get("investment_thesis") or ""),
        metrics={"scores": result.get("scores", {}), "risk_level": result.get("risk_level"), "confidence": result.get("research_confidence", {})},
        findings=findings,
        evidence=[*source_evidence[: 40 - len(derived)], *derived],
        limitations=limitations,
        run_id=str(result.get("run_id") or ""),
        cache_hit=bool(result.get("from_cache")),
        metadata={"requested_modules": result.get("requested_modules", []), "module_status": result.get("module_status", {}), "module_metrics": _module_metrics(result)},
    )


#: Metric names worth a side-by-side row, with the label the row uses.
_COMPARABLE_METRICS = {
    "pe_ttm": "市盈率（TTM）", "pe_ratio": "市盈率（TTM）", "trailing_pe": "市盈率（TTM）", "forward_pe": "前瞻市盈率", "peg": "PEG", "peg_ratio": "PEG",
    "price_to_sales": "市销率", "ps_ratio": "市销率", "ev_sales": "EV/销售额", "ev_ebitda": "EV/EBITDA", "fcf_yield": "自由现金流收益率",
    "revenue_growth": "营收增速", "revenue_growth_yoy": "营收增速", "eps_growth": "每股收益增速", "earnings_growth": "盈利增速",
    "gross_margin": "毛利率", "operating_margin": "营业利润率", "net_margin": "净利率", "roic": "ROIC", "roe": "ROE", "free_cash_flow_margin": "自由现金流利润率",
    "latest_eps_surprise": "最新每股收益超预期", "eps_surprise_pct": "最新每股收益超预期", "eps_surprise": "最新每股收益超预期", "beat_rate": "财报超预期比例", "debt_to_equity": "负债权益比",
}
#: Engine modules whose own metrics dict is worth reading for the rows (evidence rows carry only a few of them).
_TABLE_MODULES = ("valuation", "fundamental", "earnings")


def _format_metric(name: str, value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if any(word in name for word in ("growth", "margin", "roic", "roe", "surprise")) and abs(number) <= 5:
        return f"{number * 100:.1f}%"
    if any(word in name for word in ("growth", "margin", "roic", "roe", "surprise")):
        return f"{number:.1f}%"
    return f"{number:.1f}"


def _comparison_table(envelopes: list[ToolEnvelope]) -> list[EvidenceItem]:
    """Side-by-side rows for the metrics two or more of the compared stocks carry, one citeable item per metric.

    A compare answered from each stock's own evidence items reads as two
    monologues; the rows give the synthesizer the same figure for every
    ticker under one id, so "同口径比较" is a citation, not a reconstruction.
    """

    values: dict[str, dict[str, tuple[str, str]]] = {}
    for envelope in envelopes:
        for item in envelope.evidence:
            metrics = item.metadata.get("metrics") if isinstance(item.metadata, dict) else None
            for name, value in (metrics or {}).items():
                label = _COMPARABLE_METRICS.get(str(name))
                if label is None or value is None or not isinstance(value, (int, float)) or isinstance(value, bool):
                    continue
                values.setdefault(label, {}).setdefault(envelope.subject, (_format_metric(str(name), value), item.id))
        # The modules' own metrics (forward P/E, EV/EBITDA, beat rate) are not evidence rows; the limitations item stands for the module.
        module_metrics = envelope.metadata.get("module_metrics") if isinstance(envelope.metadata, dict) else None
        anchor = next((item.id for item in envelope.evidence if item.metadata.get("citation_kind") == "metrics"), envelope.evidence[0].id if envelope.evidence else "")
        for module_name, metrics in (module_metrics or {}).items():
            for name, value in (metrics or {}).items():
                label = _COMPARABLE_METRICS.get(str(name))
                if label is None or value is None or not isinstance(value, (int, float)) or isinstance(value, bool):
                    continue
                values.setdefault(label, {}).setdefault(envelope.subject, (_format_metric(str(name), value), anchor))
    rows: list[EvidenceItem] = []
    tickers = [envelope.subject for envelope in envelopes]
    for label, per_ticker in values.items():
        if len(per_ticker) < 2:
            continue
        cells = [f"{ticker} {per_ticker[ticker][0]}" if ticker in per_ticker else f"{ticker} 缺失" for ticker in tickers]
        rows.append(
            EvidenceItem(
                id="compare-" + hashlib.sha256((label + "|" + "|".join(cells)).encode("utf-8")).hexdigest()[:12],
                entity=",".join(tickers),
                claim=f"同口径对照 {label}：" + "、".join(cells) + "。",
                metric=label,
                source_id="research_engine",
                source_title="Research Engine 同口径对照",
                metadata={"citation_kind": "metrics", "comparison": True, "verified": True, "from": [per_ticker[t][1] for t in tickers if t in per_ticker]},
            )
        )
    return rows[:8]


def register_research_capabilities(
    registry: CapabilityRegistry,
    *,
    engine_factory: Callable[[], Any] = _engine_factory,
    store_factory: Callable[[], Any] = _store_factory,
) -> None:
    def stock(arguments: dict[str, Any], context: ExecutionContext) -> ToolEnvelope:
        ticker = str(arguments.get("ticker") or "").upper()
        focus = str(arguments.get("focus") or "overview")
        modules = _FOCUS_MODULES.get(focus, _FOCUS_MODULES["overview"])
        result = engine_factory().run(ticker, modules=modules)
        return _envelope(result, "research.stock")

    def compare(arguments: dict[str, Any], context: ExecutionContext) -> ToolEnvelope:
        tickers = list(dict.fromkeys(str(value).upper() for value in arguments.get("tickers", [])))[:4]
        dimensions = arguments.get("dimensions") or ["overview"]
        focus = str(dimensions[0]) if dimensions else "overview"
        modules = _FOCUS_MODULES.get(focus, _FOCUS_MODULES["overview"])

        def research(ticker: str) -> ToolEnvelope | Exception:
            # One ticker's engine failing (a new listing with no data, a
            # provider error) must not take the other's research with it.
            try:
                return _envelope(engine_factory().run(ticker, modules=modules), "research.stock")
            except Exception as exc:  # noqa: BLE001 — reported per ticker below
                logger.warning("research.compare: %s failed: %s: %s", ticker, type(exc).__name__, exc)
                return exc

        with ThreadPoolExecutor(max_workers=max(1, len(tickers))) as pool:
            rows = list(pool.map(research, tickers))
        envelopes = [row for row in rows if isinstance(row, ToolEnvelope)]
        failed = [(ticker, row) for ticker, row in zip(tickers, rows) if not isinstance(row, ToolEnvelope)]
        if not envelopes:
            status = ResultStatus.FAILED
        elif all(item.ok for item in envelopes) and not failed:
            status = ResultStatus.COMPLETED
        else:
            status = ResultStatus.PARTIAL_ERROR
        summaries = [f"{item.subject}: {item.summary}" for item in envelopes if item.summary]
        table = _comparison_table(envelopes)
        return ToolEnvelope(
            "research.compare",
            status,
            subject=",".join(tickers),
            summary="\n".join(summaries),
            findings=[finding for item in envelopes for finding in item.findings],
            evidence=[*table, *(evidence for item in envelopes for evidence in item.evidence)],
            limitations=[value for item in envelopes for value in item.limitations] + [f"{ticker} 的研究未完成（{type(exc).__name__}），比较只覆盖其余股票" for ticker, exc in failed],
            errors=[f"{ticker}: {type(exc).__name__}: {str(exc)[:200]}" for ticker, exc in failed],
            metadata={"dimensions": list(dimensions), "research_run_ids": [item.run_id for item in envelopes], "failed_tickers": [ticker for ticker, _exc in failed]},
        )

    def changes(arguments: dict[str, Any], context: ExecutionContext) -> ToolEnvelope:
        ticker = str(arguments.get("ticker") or "").upper()
        result = store_factory().compare_latest(ticker)
        if result is None:
            return ToolEnvelope("research.changes", ResultStatus.PARTIAL_DATA, subject=ticker, limitations=["at least two research snapshots are required"])
        summary = f"{ticker} 的最近两次研究快照已完成比较。"
        evidence = EvidenceItem(
            id=f"research-change-{ticker}-{result.get('new_run_id') or 'latest'}",
            entity=ticker,
            claim=summary,
            source_id="research_store",
            producer_run_id=str(result.get("new_run_id") or ""),
            metadata=result,
        )
        return ToolEnvelope("research.changes", ResultStatus.COMPLETED, subject=ticker, summary=summary, evidence=[evidence], metadata=result)

    registry.register("research.stock", stock)
    registry.register("research.compare", compare)
    registry.register("research.changes", changes)
