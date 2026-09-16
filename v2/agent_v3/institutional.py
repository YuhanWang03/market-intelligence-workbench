"""Latest 13F portfolio for a named manager, as field-level evidence.

V2 answers this capability with the Telegram card (one text blob per answer).
V3 reads the same 13F data — the tracked filings in ``data/edgar.db`` first,
EDGAR when the manager has nothing tracked — and emits one evidence item per
fact: the filing summary, each of the top positions, and each significant
quarter-over-quarter change, so the verifier can trace every number.
"""
from __future__ import annotations

import re
import sqlite3
from typing import Any

from v2.agent_v2.catalog import CapabilityCatalog, CapabilitySpec
from v2.agent_v2.models import EvidenceItem, ResultStatus, ToolEnvelope
from v2.institutional.managers import MANAGERS

CAPABILITY = "institutional.manager_portfolio"
SOURCE_ID = "sec_13f_hr"
DEFAULT_TOP = 10
MAX_CHANGES = 8

_BY_CIK = dict(MANAGERS)

#: Alias (already normalised: lowercase, no spaces or punctuation) → CIK.
#: Chinese names are how the owner asks; the English ones match V2's bot.
MANAGER_ALIASES: dict[str, str] = {
    **{alias: "1067983" for alias in ("brk", "berkshire", "berkshirehathaway", "buffett", "warrenbuffett", "巴菲特", "沃伦巴菲特", "伯克希尔", "伯克夏", "伯克希尔哈撒韦")},
    **{alias: "1649339" for alias in ("burry", "michaelburry", "scion", "scionasset", "伯里", "迈克尔伯里", "大空头")},
    **{alias: "1336528" for alias in ("ackman", "billackman", "pershing", "pershingsquare", "阿克曼", "潘兴广场")},
    **{alias: "1079114" for alias in ("einhorn", "davideinhorn", "greenlight", "爱因霍恩", "绿光")},
    **{alias: "1037389" for alias in ("renaissance", "rentech", "simons", "jimsimons", "文艺复兴", "西蒙斯", "大奖章")},
    **{alias: "1179392" for alias in ("twosigma", "两西格玛", "双西格玛")},
    **{alias: "1009207" for alias in ("deshaw", "shaw", "德劭")},
    **{alias: "1423053" for alias in ("citadel", "griffin", "kengriffin", "城堡", "肯格里芬")},
    **{alias: "1135730" for alias in ("coatue", "laffont", "高途资本", "coatue资本")},
    **{alias: "1697748" for alias in ("ark", "arkinvest", "cathiewood", "wood", "木头姐", "凯西伍德", "凯瑟琳伍德", "方舟")},
}

_STRIP = re.compile(r"[\s\.\-_,&'’\"()（）·]+")


def _normalise(text: str) -> str:
    return _STRIP.sub("", str(text or "")).lower()


def resolve_manager(text: str) -> tuple[str, str] | None:
    """(CIK, display name) for a manager mentioned by name, alias or Chinese name; None when unknown."""
    key = _normalise(text)
    if not key:
        return None
    cik = MANAGER_ALIASES.get(key)
    if cik is None:
        for name_cik, name in MANAGERS:
            if key == _normalise(name):
                cik = name_cik
                break
    if cik is None and len(key) >= 3:
        # "伯克希尔最近" / "buffett's berkshire": the alias inside the phrase, longest first.
        for alias, alias_cik in sorted(MANAGER_ALIASES.items(), key=lambda item: -len(item[0])):
            if len(alias) >= 3 and (alias in key or key in alias):
                cik = alias_cik
                break
    return (cik, _BY_CIK[cik]) if cik else None


def supported_managers() -> list[str]:
    return [name for _, name in MANAGERS]


def tracked_filings(cik: str, n_filings: int = 2, db_path=None) -> list[tuple[dict, list[dict]]]:
    """The newest ``n_filings`` tracked 13F-HR filings for a CIK, newest first: (filing row, position rows)."""
    from v2.institutional import tracker

    path = db_path or tracker._DB_PATH
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        if not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='filings'").fetchone():
            return []
        filings = [dict(row) for row in conn.execute("SELECT * FROM filings WHERE cik=? ORDER BY period_of_report DESC, filing_date DESC LIMIT ?", (cik, n_filings))]
        return [(filing, [dict(row) for row in conn.execute("SELECT * FROM positions WHERE accession=?", (filing["accession"],))]) for filing in filings]
    finally:
        conn.close()


def live_filings(cik: str, name: str, n_filings: int = 2) -> list[tuple[dict, list[dict]]]:
    """EDGAR fetch through the V2 client, converted to the same plain rows the tracker stores."""
    from v2.institutional.client import fetch_recent_13f

    rows = []
    for filing, positions in fetch_recent_13f(cik, name, n_filings=n_filings):
        rows.append((filing.model_dump(), [position.model_dump() for position in positions]))
    return rows


def default_reader(cik: str, name: str, n_filings: int = 2) -> tuple[list[tuple[dict, list[dict]]], str]:
    """Tracked filings when the scheduler has them, else EDGAR; returns (filings, provenance)."""
    tracked = tracked_filings(cik, n_filings)
    if tracked:
        return tracked, "tracked"
    return live_filings(cik, name, n_filings), "edgar"


def _edgar_url(cik: str) -> str:
    return f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=13F-HR&dateb=&owner=include&count=10"


def _label(position: dict) -> str:
    ticker = position.get("ticker")
    issuer = position.get("issuer_name") or ""
    return f"{ticker}（{issuer}）" if ticker else issuer or position.get("cusip", "?")


_CHANGE_WORDS = {"new": "新建仓", "exit": "清仓", "increase": "增持", "decrease": "减持"}


def portfolio_envelope(cik: str, name: str, filings: list[tuple[dict, list[dict]]], *, top: int = DEFAULT_TOP, provenance: str = "tracked", run_id: str = "") -> ToolEnvelope:
    if not filings:
        return ToolEnvelope(CAPABILITY, ResultStatus.PARTIAL_DATA, subject=name, limitations=[f"{name}（CIK {cik}）没有可读取的 13F-HR 申报。"])
    filing, positions = filings[0]
    quarter, period, filed, accession = filing["quarter"], filing["period_of_report"], filing["filing_date"], filing["accession"]
    total = float(filing.get("portfolio_value") or sum(float(p["market_value"]) for p in positions))
    url = _edgar_url(cik)
    base = dict(source_id=SOURCE_ID, source_title=f"SEC Form 13F-HR {accession}", source_url=url, period=quarter, as_of=period, producer_run_id=run_id)
    evidence = [EvidenceItem(
        f"13f-{cik}-{quarter}-portfolio", name,
        f"{name} 在 {quarter}（报告期截至 {period}，申报日 {filed}）的 13F 组合总市值为 {total:,.0f} 美元，共 {len(positions)} 个持仓。",
        metric="portfolio_value", value=round(total, 2), unit="USD",
        metadata={"cik": cik, "accession": accession, "filing_date": filed, "position_count": len(positions), "date_basis": "filing", "provenance": provenance}, **base,
    )]
    ranked = sorted(positions, key=lambda p: -float(p["market_value"]))
    for rank, position in enumerate(ranked[:top], start=1):
        value = float(position["market_value"])
        weight = round(value / total * 100, 2) if total > 0 else 0.0
        evidence.append(EvidenceItem(
            f"13f-{cik}-{quarter}-pos-{position.get('ticker') or position['cusip']}", name,
            f"{name} 第 {rank} 大持仓 {_label(position)}：{int(position['shares']):,} 股，市值 {value:,.0f} 美元，占组合 {weight}%。",
            metric="position_value", value=round(value, 2), unit="USD",
            metadata={"rank": rank, "ticker": position.get("ticker"), "issuer": position.get("issuer_name"), "cusip": position["cusip"], "shares": int(position["shares"]), "weight_pct": weight, "date_basis": "filing"}, **base,
        ))
    limitations = [
        "13F 在季度结束后 45 天内申报，反映的是报告期末的持仓，之后可能已经变化；只含美股多头持仓，不含空头、期权以外的衍生品和非美资产。",
        f"仅列出市值最大的 {min(top, len(ranked))} 个持仓，不是完整组合。",
    ]
    if provenance == "tracked":
        limitations.append("数据来自本地 13F 跟踪库；若 EDGAR 已有更新申报而跟踪任务尚未运行，此处可能滞后。")
    status = ResultStatus.COMPLETED
    changes_out: list[dict] = []
    if len(filings) >= 2:
        from v2.institutional.detector import detect_changes

        previous, prev_positions = filings[1]
        prev_total = float(previous.get("portfolio_value") or sum(float(p["market_value"]) for p in prev_positions))
        changes = detect_changes(cik=cik, manager_name=name, quarter=quarter, current_positions=positions, prev_positions=prev_positions, current_total=total, prev_total=prev_total)
        for change in changes[:MAX_CHANGES]:
            label = f"{change.ticker}（{change.issuer_name}）" if change.ticker else change.issuer_name
            delta = change.current_value - change.prev_value
            evidence.append(EvidenceItem(
                f"13f-{cik}-{quarter}-chg-{change.ticker or change.cusip}", name,
                f"{name} 在 {quarter} 相比 {previous['quarter']} {_CHANGE_WORDS[change.change_type]} {label}：{change.prev_shares:,} 股 → {change.current_shares:,} 股，市值 {change.prev_value:,.0f} → {change.current_value:,.0f} 美元。",
                metric="position_change_value", value=round(delta, 2), unit="USD",
                metadata={"change_type": change.change_type, "ticker": change.ticker, "issuer": change.issuer_name, "cusip": change.cusip, "previous_quarter": previous["quarter"], "current_shares": change.current_shares, "prev_shares": change.prev_shares, "current_pct": round(change.current_pct * 100, 2), "prev_pct": round(change.prev_pct * 100, 2), "date_basis": "filing"}, **base,
            ))
            changes_out.append({"ticker": change.ticker, "issuer": change.issuer_name, "change_type": change.change_type, "delta_value": delta})
        if not changes:
            limitations.append(f"与 {previous['quarter']} 相比没有达到显著阈值的增减仓。")
    else:
        status = ResultStatus.PARTIAL_DATA
        limitations.append("只有一期申报可用，无法给出与上季度的增减仓对比。")
    metrics = {"portfolio_value_usd": round(total, 2), "position_count": len(positions), "top_listed": min(top, len(ranked)), "significant_changes": len(changes_out)}
    return ToolEnvelope(CAPABILITY, status, subject=name, as_of=period, summary=f"{name} {quarter} 13F：总市值 {total:,.0f} 美元，{len(positions)} 个持仓，{len(changes_out)} 项显著变动。",
                        metrics=metrics, findings=changes_out, evidence=evidence, limitations=limitations,
                        metadata={"cik": cik, "quarter": quarter, "period_of_report": period, "filing_date": filed, "accession": accession, "provenance": provenance, "filings_compared": len(filings)})


def register_institutional(registry, *, reader=None) -> None:
    """Register ``institutional.manager_portfolio``; ``reader(cik, name, n)`` returns (filings, provenance)."""
    reader = reader or default_reader
    schema = {"type": "object", "properties": {"manager": {"type": "string", "minLength": 1, "maxLength": 80}, "top": {"type": "integer", "minimum": 1, "maximum": 25}}, "required": ["manager"], "additionalProperties": False}
    spec = CapabilitySpec(CAPABILITY, "research", "Latest 13F-HR portfolio for a named manager: filing summary, top positions and significant quarter-over-quarter changes.", schema,
                          answer_guidance="thirteen_f：先写报告期（季度末）和申报滞后（13F 在季度结束后 45 天内提交，持仓可能已经变化），再说主要持仓和本期增减仓；没有增减仓数据时明说。")
    registry.catalog = CapabilityCatalog([*(s for s in registry.catalog.specs() if s.name != CAPABILITY), spec])

    def handler(arguments: dict[str, Any], context) -> ToolEnvelope:
        asked = str(arguments["manager"])
        resolved = resolve_manager(asked)
        if resolved is None:
            return ToolEnvelope(CAPABILITY, ResultStatus.PARTIAL_DATA, subject=asked,
                                limitations=[f"未识别的基金经理“{asked}”。目前跟踪的机构：{'、'.join(supported_managers())}。"])
        cik, name = resolved
        run_id = getattr(getattr(context, "v3_run", None), "run_id", "") or ""
        try:
            filings, provenance = reader(cik, name, 2)
        except Exception as exc:  # noqa: BLE001 — a data-source failure is a partial result, not a crash
            return ToolEnvelope(CAPABILITY, ResultStatus.PARTIAL_ERROR, subject=name, errors=[f"13F 数据读取失败：{type(exc).__name__}: {str(exc)[:160]}"])
        return portfolio_envelope(cik, name, filings, top=int(arguments.get("top") or DEFAULT_TOP), provenance=provenance, run_id=run_id)

    registry.register(CAPABILITY, handler)
