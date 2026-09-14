"""Shape an Agent V2 result for a phone screen, without python-telegram-bot.

The web frontends show the evidence ids, the synthesis badge and the
verification badge as UI; a Telegram message has to carry the same
information in its text.  Everything here is plain string work so the bot
tests can run where the Telegram library cannot be imported.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlparse

from v2.agent_v2.models import AgentResult, EvidenceItem, ResultStatus

_CITATION = re.compile(r"\[([A-Za-z0-9_.:~-]+)\]")

#: How the answer text came to be, in the words the web badge uses.
SYNTHESIS_LABELS = {
    "clean": "模型回答",
    "repaired": "模型回答（修正一轮）",
    "knowledge": "模型回答",
    "fallback": "兜底摘要",
    "deterministic": "规则摘要",
}


@dataclass(frozen=True)
class NumberedAnswer:
    text: str
    #: Evidence ids in order of first appearance; ``[n]`` in the text is ``ids[n - 1]``.
    ids: tuple[str, ...]


def number_citations(answer: str, evidence: list[EvidenceItem], *, order: list[str] | None = None) -> NumberedAnswer:
    """Replace ``[evidence-id]`` citations with ``[1]``, ``[2]``… by first appearance.

    Only ids that exist in the evidence list are renumbered, so a bracketed
    ticker or figure the model wrote stays as it is.  A run of adjacent
    citations such as ``[a][b]`` becomes ``[1][2]``.  Pass the ``order``
    of an earlier call to number further text (the sub-agent notes) in the
    same sequence: ids already seen keep their numbers, new ones continue.
    """

    known = {item.id for item in evidence}
    sequence = order if order is not None else []

    def replace(match: re.Match[str]) -> str:
        identifier = match.group(1)
        if identifier not in known:
            return match.group(0)
        if identifier not in sequence:
            sequence.append(identifier)
        return f"[{sequence.index(identifier) + 1}]"

    return NumberedAnswer(_CITATION.sub(replace, answer or ""), tuple(sequence))


def compact_attributions(answer: str, result: AgentResult) -> str:
    """Swap each worst-day attribution block for its one-paragraph form.

    The deterministic fallback pastes the attributor's narrative verbatim, so
    the substitution is exact; a model-written answer never contains the
    narrative and is left alone.
    """

    text = answer or ""
    for envelope in result.results:
        if envelope.capability != "market.attribute_move":
            continue
        full = str(envelope.metadata.get("narrative") or "").strip()
        compact = str(envelope.metadata.get("narrative_compact") or "").strip()
        if full and compact and full in text:
            text = text.replace(full, compact)
    return text


#: Where an evidence item came from, in the reader's language.
SOURCE_LABELS = {
    "market_data": "日线行情",
    "anomaly_memory": "盯盘记忆",
    "sec_edgar": "SEC 申报",
    "web_news": "网页新闻",
    "web_search": "网页搜索",
    "move_attribution": "归因判断",
    "research_engine": "研究引擎",
    "research_store": "研究库",
    "legacy_responder": "账户卡片",
    # The research engine's data sources, by their source ids.
    "fd_company": "公司资料（Financial Datasets）",
    "fd_metrics": "基本面指标（Financial Datasets）",
    "fd_earnings": "财报数据（Financial Datasets）",
    "fd_insiders": "内部人交易（Financial Datasets）",
    "yf_prices": "行情（Yahoo Finance）",
    "yf_calendar": "财报日历（Yahoo Finance）",
    "sec_filings": "SEC 申报（EDGAR）",
    "local_13f": "13F 持仓归档",
    "local_etf": "ARK 持仓归档",
    "macro_snapshot": "宏观快照（FRED、Yahoo Finance）",
    "supply_chain": "产业链关系（多源）",
    # Legacy cards carry their capability as source_id.
    "macro.overview": "宏观面板",
    "macro.release": "宏观数据",
    "account.portfolio": "账户卡片",
    "account.performance": "账户盈亏",
    "account.risk": "组合风险",
    "account.earnings_schedule": "财报日历",
    "state.read": "用户设置",
    "state.mutate": "账户操作",
    "institutional.manager_portfolio": "13F 持仓",
    "etf.ark_activity": "ARK 持仓",
}
_LEGACY_TITLE = "Existing deterministic responder"


def _origin(item: EvidenceItem) -> str:
    if item.source_id in SOURCE_LABELS:
        return SOURCE_LABELS[item.source_id]
    if item.source_title == _LEGACY_TITLE:
        return SOURCE_LABELS["legacy_responder"]
    if item.source_id.startswith("web:"):
        return f"网页（{item.source_id[4:]}）"
    module = str(item.metadata.get("module") or "")
    fallback = item.source_title or item.source_id or str(item.metadata.get("evidence_scope") or "")
    if not fallback and module:
        fallback = f"研究引擎·{module}"
    return _one_line(fallback) or "其他来源"


def _one_line(text: str, limit: int = 80) -> str:
    """One line of at most ``limit`` characters; a cut lands on a punctuation mark near the limit and never inside a bracket."""

    from v2.agent_v2.agents.move_attributor import clip

    flat = re.sub(r"\s+", " ", str(text or "")).strip()
    return flat if len(flat) <= limit else clip(flat, limit)


_NOTE_CITATION = re.compile(r"^(?P<head>.*?)(?P<tail>（引 \[[^\]]+\]）)\s*$", re.S)


def _note_line(note: str, limit: int) -> str:
    """One line of a sub-agent note: the text is cut, the trailing ``（引 [id]）`` never is."""

    match = _NOTE_CITATION.match(str(note or ""))
    if match is None:
        return _one_line(note, limit)
    tail = match.group("tail")
    return _one_line(match.group("head"), max(20, limit - len(tail))) + tail


def _ranges(numbers: list[int]) -> str:
    """``[2, 3, 4, 7, 9, 10]`` → ``2–4、7、9–10``."""

    parts: list[str] = []
    for number in sorted(set(numbers)):
        if parts and parts[-1][1] == number - 1:
            parts[-1][1] = number
        else:
            parts.append([number, number])
    return "、".join(str(low) if low == high else f"{low}–{high}" for low, high in parts)


@dataclass(frozen=True)
class SourceEntry:
    #: ``"3"`` for one item, ``"2–8、12"`` for a group of items from one origin.
    numbers: str
    label: str
    url: str = ""


def source_entries(ids: tuple[str, ...], evidence: list[EvidenceItem], *, max_linked: int = 8) -> list[SourceEntry]:
    """The 来源 list: linked items one per line, the rest grouped by origin.

    Every cited number appears exactly once.  Filings and news carry a
    title and a link, so each gets its own line; the many price and memory
    items behind a drawdown answer are one line per origin, because the
    answer already quotes their claims.
    """

    by_id = {item.id: item for item in evidence}
    # One line per page: several claims from the same article share a line.
    linked: dict[str, tuple[str, list[int]]] = {}
    grouped: dict[str, list[int]] = {}
    overflow: list[int] = []
    for index, identifier in enumerate(ids, start=1):
        item = by_id.get(identifier)
        if item is None:
            continue
        url = link_for(item)
        if url and url in linked:
            linked[url][1].append(index)
        elif url and len(linked) < max_linked:
            linked[url] = (_linked_label(item), [index])
        elif url:
            overflow.append(index)
        else:
            grouped.setdefault(_origin(item), []).append(index)
    entries = [SourceEntry(_ranges(numbers), label, url) for url, (label, numbers) in linked.items()]
    entries.extend(SourceEntry(_ranges(numbers), origin) for origin, numbers in grouped.items())
    if overflow:
        entries.append(SourceEntry(_ranges(overflow), "其余带链接的来源（略）"))
    return sorted(entries, key=lambda entry: int(re.match(r"\d+", entry.numbers).group(0)))


def _linked_label(item: EvidenceItem) -> str:
    """``标题（2026-09-04）``: the answer already quotes the claim, so the line names the page and its date."""

    title = _one_line(item.source_title) or _origin(item)
    day = str(item.as_of or "")[:10]
    if item.source_id == "sec_edgar" or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day) or day in title:
        return title
    return f"{title}（{day}）"


def link_for(item: EvidenceItem) -> str:
    url = item.source_url or ""
    return url if urlparse(url).scheme in {"http", "https"} else ""


def synthesis_label(result: AgentResult) -> str:
    synthesis = result.synthesis or {}
    outcome = str(synthesis.get("outcome") or "")
    label = SYNTHESIS_LABELS.get(outcome, "")
    revision = synthesis.get("debate_revision") or {}
    if label and revision:
        label += "，按反方修订" if revision.get("applied") else f"，反方意见未采纳（{revision.get('reason') or '修订未通过'}）"
    return label


#: Chinese names for the capabilities a progress line can mention.
CAPABILITY_LABELS = {
    "market.explain_move": "解释今日涨跌", "market.attribute_move": "归因下跌日", "market.performance": "取行情", "market.drawdown": "算回撤", "market.runup": "算涨幅", "market.anomaly_history": "查盯盘记录",
    "filings.recent": "列申报", "filings.read_events": "读申报", "web.research": "搜新闻", "research.stock": "跑研究引擎", "research.compare": "对比研究", "research.changes": "比对研究快照",
    "account.portfolio": "读持仓", "account.performance": "算盈亏", "account.risk": "算组合风险", "account.earnings_schedule": "查财报日历", "state.read": "读设置", "state.mutate": "执行操作",
    "macro.overview": "读宏观面板", "macro.release": "查宏观数据", "institutional.manager_portfolio": "查 13F", "etf.ark_activity": "查 ARK", "agent.investigate": "现场调查", "debate.challenge": "反方审阅",
}
_STAGE_LABELS = {"received": "收到", "routed": "已识别问题", "planned": "已规划", "executing": "执行中", "synthesizing": "整理回答", "verifying": "校验引用", "completed": "完成", "partial": "部分完成", "failed": "失败", "cancelled": "已取消"}


def progress_line(event, *, elapsed: float | None = None) -> str:
    """One line for the placeholder message: the stage, the capability being run, the elapsed time."""

    status = str(getattr(event.status, "value", event.status) or "")
    stage = _STAGE_LABELS.get(status, status)
    capability = str(getattr(event, "capability", "") or "")
    message = str(getattr(event, "message", "") or "")
    if capability:
        detail = CAPABILITY_LABELS.get(capability, capability)
        if message.startswith("follow-up"):
            detail = "追加：" + detail
    elif status == "planned":
        digits = "".join(ch for ch in message if ch.isdigit())
        detail = f"{digits} 个任务" if digits else ""
    else:
        detail = ""
    line = stage + (f"：{detail}" if detail else "")
    if elapsed is not None and elapsed >= 1:
        line += f" · {elapsed:.0f}s"
    return line


def verification_label(result: AgentResult) -> str:
    report = result.verification
    if report.ok and not report.warnings:
        return "通过"
    problems = len(report.unknown_citations) + len(report.ungrounded_numbers) + len(report.warnings)
    return f"有警告（{problems}）" if problems else "通过"


def warning_line(result: AgentResult) -> str:
    """The verifier's first complaint about the delivered answer, or empty."""

    report = result.verification
    problems = [*report.warnings]
    if report.unknown_citations:
        problems.append("未知引用：" + "、".join(report.unknown_citations[:3]))
    if report.ungrounded_numbers:
        problems.append("未落地数字：" + "、".join(report.ungrounded_numbers[:3]))
    return _one_line(problems[0], 120) if problems else ""


def _attempt_clauses(synthesis: dict) -> list[str]:
    """One clause per rejected draft: stage, then the verifier's first complaint."""

    stages = {"draft": "初稿", "repair": "修正稿", "error": "模型调用"}
    clauses: list[str] = []
    for attempt in synthesis.get("attempts") or []:
        if not isinstance(attempt, dict) or attempt.get("ok"):
            continue
        problems = [*(attempt.get("warnings") or [])]
        if attempt.get("unknown_citations"):
            problems.append("未知引用 " + "、".join(str(value) for value in attempt["unknown_citations"][:2]))
        if attempt.get("ungrounded_numbers"):
            problems.append("未落地数字 " + "、".join(str(value) for value in attempt["ungrounded_numbers"][:2]))
        stage = stages.get(str(attempt.get("stage")), str(attempt.get("stage") or "草稿"))
        clauses.append(f"{stage}：{_one_line(problems[0], 70) if problems else '校验未通过'}")
    return clauses


def budget_line(result: AgentResult) -> str:
    """``预算用尽：research.compare 未在 60 秒内完成`` when the run stopped on its deadline, else empty."""

    if result.stop_reason != "deadline":
        return ""
    from v2.agent_v2.execution import time_limit

    seconds = time_limit(result.plan.budget)
    timed_out = [r.capability for r in result.results if r.status == ResultStatus.FAILED and any("timed out" in error for error in r.errors)]
    skipped = [r.capability for r in result.results if r.status == ResultStatus.SKIPPED]
    parts = []
    if timed_out:
        parts.append("、".join(dict.fromkeys(timed_out)) + f" 未在 {seconds:.0f} 秒内完成")
    if skipped:
        parts.append("、".join(dict.fromkeys(skipped)) + " 未开始")
    return "预算用尽：" + ("；".join(parts) if parts else f"{seconds:.0f} 秒预算已耗尽")


def fallback_reason(result: AgentResult) -> str:
    """Why the model's drafts were rejected, one clause per attempt, or empty when not a fallback."""

    synthesis = result.synthesis or {}
    if synthesis.get("outcome") != "fallback":
        return ""
    clauses = _attempt_clauses(synthesis)
    return "；".join(clauses[:2]) if clauses else "模型草稿未通过校验"


def repair_reason(result: AgentResult) -> str:
    """What the first draft failed on when the repaired draft was delivered, or empty."""

    synthesis = result.synthesis or {}
    if synthesis.get("outcome") != "repaired":
        return ""
    clauses = _attempt_clauses(synthesis)
    return clauses[0] if clauses else ""


def completion_note(result: AgentResult) -> str:
    """How many citations the code completed on the model's drafts, or empty."""

    completions = (result.synthesis or {}).get("citation_completions") or []
    return f"引用补全 {len(completions)} 处" if completions else ""


_STOP_LABELS = {"cancelled": "用户取消", "finished": "完成", "rounds": "轮次用尽", "time": "超时", "no_model": "无模型", "no_budget": "无预算", "no_filings": "无申报"}
_CALL_LABELS = {"news": "新闻", "filing_events": "读申报", "memory": "记忆", "search": "搜索", "read": "读正文", "filings": "申报", "sections_read": "读节", "events": "事件", "objections": "反对"}


def agent_lines(result: AgentResult, *, limit: int = 8) -> list[str]:
    """One plain-text line per sub-agent run (and its nested reader), for the message footer."""

    from v2.agent_v2.models import sub_agent_summaries

    lines: list[str] = []
    for entry in sub_agent_summaries(result.results):
        calls = "、".join(f"{_CALL_LABELS.get(key, key)} {value}" for key, value in entry["calls"].items() if value)
        stop = _STOP_LABELS.get(entry["stop_reason"], entry["stop_reason"])
        seconds = entry["elapsed_ms"] / 1000
        lines.append(f"{entry['label']} {entry['subject']}：{entry['rounds']} 轮 · {seconds:.1f}s" + (f" · {calls}" if calls else "") + (f" · {stop}" if stop else "") + ("（盘中）" if entry["intraday"] else ""))
        for note in (entry.get("notes") or [])[:3]:
            lines.append(f"  · {_note_line(note, 110)}")
        challenge = entry.get("challenge") or {}
        if challenge.get("objection"):
            verdict = "反方降级" if challenge.get("downgraded") else "反方未采纳"
            lines.append(f"  · {verdict}：{_one_line(str(challenge['objection']), 90)}")
        memory = entry.get("memory") or {}
        if memory.get("conflict") or (memory.get("note") and not memory.get("written")):
            lines.append(f"  · 记忆{'冲突' if memory.get('conflict') else '未覆盖'}：{_one_line(str(memory.get('note') or ''), 90)}")
        for nested in entry["nested"]:
            nested_calls = "、".join(f"{_CALL_LABELS.get(key, key)} {value}" for key, value in nested["calls"].items() if value)
            lines.append(f"  ↳ {nested['label']}：{nested['rounds']} 轮 · {nested['elapsed_ms'] / 1000:.1f}s" + (f" · {nested_calls}" if nested_calls else "") + (f" · {_STOP_LABELS.get(nested['stop_reason'], nested['stop_reason'])}" if nested["stop_reason"] else ""))
        if len(lines) >= limit:
            break
    return lines[:limit]


def web_label(*, requested: bool, enabled: bool) -> str:
    """The 网页 field of the header, and the hint that tells the user how to change it."""

    if not enabled:
        return "未启用（服务端 AGENT_V2_WEB_ENABLED 未开）"
    if not requested:
        return "已关闭（去掉 --noweb 可用新闻归因）"
    return "已启用"
