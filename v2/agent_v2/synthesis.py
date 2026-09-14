"""Default evidence-preserving synthesis; replaceable with an LLM synthesizer.

The synthesizer knows nothing about any one domain.  An adapter that can
render its own result better than a claim list puts the prose in
``ToolEnvelope.metadata["narrative"]`` and the synthesizer uses it verbatim.
"""

from __future__ import annotations

import re
from datetime import date, timedelta
from dataclasses import dataclass, field
from typing import Any

from v2.agent_v2.models import (
    EvidenceItem,
    ExecutionPlan,
    NormalizedRequest,
    ToolEnvelope,
)
from v2.agent_v2.text import plain_text

_SUPERLATIVE = re.compile(r"最|哪只|哪个|哪几|哪些|排名|排序|前几|谁")


@dataclass
class RankingLead:
    text: str = ""
    entities: tuple[str, ...] = ()
    result: ToolEnvelope | None = field(default=None, repr=False)

    def __bool__(self) -> bool:
        return bool(self.text)


def choose_rankable(text: str, rules: Any) -> tuple[dict[str, Any], bool, bool] | None:
    """The rankable rule the wording selects, with its direction (low, high)."""

    if not isinstance(rules, list):
        return None
    usable = [rule for rule in rules if isinstance(rule, dict)]
    candidates = [rule for rule in usable if rule.get("topic") and re.search(str(rule["topic"]), text)]
    candidates += [rule for rule in usable if not rule.get("topic")]
    for rule in candidates:
        low = bool(re.search(str(rule.get("low") or "$^"), text))
        high = bool(re.search(str(rule.get("high") or "$^"), text))
        if low != high:
            return rule, low, high
    return None


def position_row(results: list[ToolEnvelope], ticker: str) -> tuple[ToolEnvelope, dict[str, Any]] | None:
    """The result and row describing ``ticker`` in a position table, if any result carries one."""

    for result in results:
        rows = result.metadata.get("positions")
        if not isinstance(rows, list):
            continue
        for row in rows:
            if isinstance(row, dict) and str(row.get("ticker") or "").upper() == ticker.upper():
                return result, row
    return None


def frame_lead(frame: dict[str, Any], results: list[ToolEnvelope]) -> str:
    """Open a framed follow-up by restating what it refers to, from this run's evidence.

    A question like "为什么跌这么多" after a P/L ranking is about the loss
    since purchase.  The restatement cites the freshly fetched position row,
    and when today's move points the other way it says so in one clause
    rather than letting the day's figure answer a question about months.
    """

    ticker = str(frame.get("ticker") or "")
    if not ticker:
        return ""
    found = position_row(results, ticker)
    lines: list[str] = []
    value: Any = None
    if frame.get("kind") in {"drawdown", "runup"}:
        wanted = "market.runup" if frame.get("kind") == "runup" else "market.drawdown"
        drawdown = next((result for result in results if result.capability == wanted and result.ok and result.subject.upper() == ticker.upper()), None)
        if drawdown is None:
            return ""
        peak = next((item for item in drawdown.evidence if item.metadata.get("evidence_scope") == "peak_trough"), None)
        window = next((item for item in drawdown.evidence if item.metadata.get("evidence_scope") == "window_return"), None)
        anchor = peak or window
        if anchor is None:
            return ""
        lines.append(f"你问的是 {ticker} {frame.get('label') or ('这段涨幅' if wanted == 'market.runup' else '这段跌幅')}：{anchor.claim.rstrip('。')}[{anchor.id}]。")
        number = drawdown.metrics.get("move", drawdown.metrics.get("drawdown")) if peak is not None else drawdown.metrics.get("window_return")
        value = float(number) * 100 if isinstance(number, (int, float)) else None
        if found is not None:
            source, row = found
            entry = row.get("avg_entry_price")
            entry_text = f"，成本价 ${float(entry):.2f}" if isinstance(entry, (int, float)) else ""
            citation = f"[{source.evidence[0].id}]" if source.evidence else ""
            lines.append(f"你持有该股，买入以来的浮动盈亏 {row.get('pl_pct_text') or row.get('pl_pct')}{entry_text}{citation}。")
    elif found is not None:
        source, row = found
        key, text_key = str(frame.get("field") or "pl_pct"), str(frame.get("text") or "pl_pct_text")
        value_text = row.get(text_key) or row.get(key)
        citation = f"[{source.evidence[0].id}]" if source.evidence else ""
        entry = row.get("avg_entry_price")
        entry_text = f"，成本价 ${float(entry):.2f}" if isinstance(entry, (int, float)) else ""
        lines.append(f"你问的是 {ticker} {frame.get('label') or '这一项'}：{value_text}{entry_text}{citation}。")
        value = row.get(key)
    else:
        return ""
    for result in results:
        if result.subject.upper() != ticker.upper() or not result.ok:
            continue
        returns = result.metrics.get("returns") if isinstance(result.metrics.get("returns"), dict) else {}
        today = returns.get("1d", result.metrics.get("price_change_pct"))
        if not isinstance(today, (int, float)) or not isinstance(value, (int, float)):
            continue
        candidates: list[str] = []
        if (today > 0) != (value > 0) and today != 0:
            price = next((item for item in result.evidence if item.metadata.get("evidence_scope") == "price"), None)
            direction = "上涨" if today > 0 else "下跌"
            # An intraday price carries a wording rule: the sentence citing
            # it must say it is intraday, or the verifier rejects it.
            intraday = price is not None and (price.metadata.get("is_intraday") or any(isinstance(rule, dict) and rule.get("require") for rule in price.metadata.get("constraints") or []))
            when = "今日盘中" if intraday else "今日"
            candidates.append(f"{when}为{direction}（{float(today):+.2%}），与{frame.get('label') or '上述区间'}是不同区间{f'[{price.id}]' if price else ''}。")
        candidates.extend(decline_timing(value, returns, result, held=found is not None))
        # Sentences the code writes go through the same verifier as the
        # model's; one that fails is dropped rather than shipped.
        lines.extend(sentence for sentence in candidates if _self_checks(sentence, result))
        break
    return "\n".join(lines)


def _self_checks(sentence: str, result: ToolEnvelope) -> bool:
    from v2.agent_v2.models import AnswerMode
    from v2.agent_v2.verification import verify_answer

    return verify_answer(sentence, list(result.evidence), answer_mode=AnswerMode.TOOL_GROUNDED, results=[result]).ok


_WINDOW_LABELS = (("5d", "近 5 日"), ("1m", "近 1 月"), ("3m", "近 3 月"), ("1y", "近 1 年"))


def decline_timing(total_pct: Any, returns: dict[str, Any], result: ToolEnvelope, *, held: bool = True) -> list[str]:
    """Where in time a loss since purchase sits, read off the return windows.

    Each window's return is compared with the loss: the first window that
    accounts for at least half of it is where the decline mostly happened;
    when none does, the decline predates the longest window.  A positive
    longest window while the position loses means the purchase came after
    the run-up, which is said as well.
    """

    if not isinstance(total_pct, (int, float)) or total_pct == 0:
        return []
    up = float(total_pct) > 0
    word = "涨幅" if up else "跌幅"
    total = float(total_pct) / 100.0
    windows = [(key, label, float(returns[key])) for key, label in _WINDOW_LABELS if isinstance(returns.get(key), (int, float))]
    if not windows:
        return []
    item = next((item for item in result.evidence if item.metadata.get("evidence_scope") == "returns"), None)
    citation = f"[{item.id}]" if item is not None else ""
    described = "、".join(f"{label} {value:+.2%}" for _, label, value in windows)
    sentences: list[str] = []
    for _, label, value in windows:
        if value * total > 0 and value / total >= 0.5:
            sentences.append(f"对照区间回报（{described}），这段{word}大部分落在{label}内{citation}。")
            break
    else:
        _, longest_label, _ = windows[-1]
        sentences.append(f"对照区间回报（{described}），这段{word}主要发生在{longest_label}以前{citation}。")
    _, longest_label, longest_value = windows[-1]
    if not up and longest_value > 0:
        if held:
            sentences.append(f"{longest_label} {longest_value:+.2%} 而该持仓仍在浮亏，说明买入点在这轮上涨之后的高位{citation}。")
        else:
            sentences.append(f"{longest_label}仍为 {longest_value:+.2%}，这段跌幅是一轮上涨之后的回撤{citation}。")
    elif up and longest_value < 0:
        if held:
            sentences.append(f"{longest_label} {longest_value:+.2%} 而该持仓仍在浮盈，说明买入点在这轮下跌之后的低位{citation}。")
        else:
            sentences.append(f"{longest_label}仍为 {longest_value:+.2%}，这段涨幅是一轮下跌之后的反弹{citation}。")
    return sentences


_DATED = re.compile(r"\d{4}-\d{2}-\d{2}|\d{4}\s*年\s*\d{1,2}\s*月|\d{1,2}\s*月\s*\d{1,2}\s*日|\b(?:Q[1-4]|FY)\s?\d{2,4}\b")


def benchmark_line(result: ToolEnvelope) -> str:
    """Only the benchmark-relative sentence of a performance narrative; the returns were used above."""

    narrative = plain_text(str(result.metadata.get("narrative") or ""))
    for line in narrative.split("\n"):
        if "相对" in line and "[" in line:
            return line.strip()
    return ""


def web_lines(result: ToolEnvelope, since: str = "") -> str:
    """Web search results in a framed answer: headline, published date and excerpt.

    A news snippet rarely carries its date in the text; the published date
    is metadata, so the window filter uses that and keeps undated items,
    marked as such.
    """

    lines: list[str] = []
    for item in result.evidence:
        if not item.metadata.get("citable", True) or item.metadata.get("citation_kind") in {"metrics", "limitations"}:
            continue
        day = str(item.as_of or item.metadata.get("published_at") or "")[:10]
        if since and day and day < since:
            continue
        excerpt = _WS_RUN.sub(" ", plain_text(item.claim)).strip()
        title = _WS_RUN.sub(" ", plain_text(item.source_title)).strip() or "（无标题）"
        lines.append(f"- {title}（{day or '日期未知'}）：{excerpt[:180]}{'…' if len(excerpt) > 180 else ''} [{item.id}]")
        if len(lines) >= 4:
            break
    if not lines:
        lines.append(f"{result.subject} 网页搜索未返回落在区间内的报道。")
    if not result.ok:
        detail = result.errors[0] if result.errors else "未知错误"
        lines.append(f"{result.capability} 未完成：{detail}")
    return "\n".join(lines)


_WS_RUN = re.compile(r"\s+")


def catalyst_lines(result: ToolEnvelope, since: str = "") -> str:
    """A research or history result in a framed answer: its dated, citable findings, not a thesis.

    ``since`` (ISO date) keeps only events inside the decline window when
    the item carries a date in its metadata or ``as_of``.
    """

    lines: list[str] = []
    for item in result.evidence:
        if not item.metadata.get("citable", True) or item.metadata.get("citation_kind") in {"metrics", "limitations"}:
            continue
        claim = plain_text(item.claim)
        # A catalyst is an event: it has a date.  Undated fundamentals and
        # valuation figures are not what "why did it fall" asks for.
        if not claim or not _DATED.search(claim):
            continue
        day = str(item.metadata.get("date") or item.as_of or "")[:10]
        if since and day and day < since:
            continue
        lines.append(f"- {claim} [{item.id}]")
        if len(lines) >= 6:
            break
    if not lines:
        lines.append(f"{result.subject} 期间未查到可核对的催化剂（财报、公告或新闻）。")
    limitation_item = next((item for item in result.evidence if item.metadata.get("citation_kind") == "limitations"), None)
    suffix = f" [{limitation_item.id}]" if limitation_item is not None else ""
    lines.extend(f"数据限制：{item}{suffix}" for item in user_facing_limitations(result.limitations)[:2])
    if not result.ok:
        detail = result.errors[0] if result.errors else "未知错误"
        lines.append(f"{result.capability} 未完成：{detail}")
    return "\n".join(lines)


_STRETCH = frozenset({"market.drawdown", "market.runup"})


def drawdown_lines(result: ToolEnvelope) -> str:
    """The stretch itself: peak to trough and the worst days; the window return was in the lead."""

    peak = next((item for item in result.evidence if item.metadata.get("evidence_scope") == "peak_trough"), None)
    span = next((item for item in result.evidence if item.metadata.get("evidence_scope") == "benchmark_span"), None)
    up = result.metadata.get("direction") == "up" or result.capability == "market.runup"
    worst = [item for item in result.evidence if item.metadata.get("evidence_scope") == ("best_day" if up else "worst_day")]
    lines: list[str] = []
    if peak is not None:
        lines.append(f"{plain_text(peak.claim).rstrip('。')}[{peak.id}]。")
    if span is not None:
        lines.append(f"{plain_text(span.claim).rstrip('。')}[{span.id}]。")
    days = []
    for item in worst:
        move = f"{float(item.value):+.2%}" if isinstance(item.value, (int, float)) else (_PERCENT.search(plain_text(item.claim)) or [""])[0]
        days.append(f"{item.metadata.get('date')} {move}[{item.id}]".replace("  ", " "))
    if days:
        lines.append(f"{result.subject} 期间{'涨幅' if up else '跌幅'}最大的交易日：" + "；".join(days) + "。")
    return "\n".join(lines) if lines else EvidenceSummarySynthesizer._render(result)


_FORM = re.compile(r"提交了\s*([0-9A-Z-]+[A-Z])")
_PERCENT = re.compile(r"[+-]\d+(?:\.\d+)?%")


def filing_line(result: ToolEnvelope, since: str = "", until: str = "", *, up: bool = False) -> str:
    """The filings inside the decline on one line; their contents were read per worst day."""

    dated = []
    for item in result.evidence:
        if item.metadata.get("evidence_scope") != "filing":
            continue
        day = str(item.metadata.get("date") or item.as_of or "")[:10]
        if (since and day and day < since) or (until and day and day > until):
            continue
        match = _FORM.search(plain_text(item.claim))
        form = str(item.metadata.get("form") or (match.group(1) if match else "申报"))
        dated.append(f"{day} {form}[{item.id}]")
    if not dated:
        return ""
    return f"{result.subject} 这段{'上涨' if up else '下跌'}期间 {len(dated)} 份申报：" + "；".join(dated) + "。"


def anomaly_lines(result: ToolEnvelope, since: str = "", until: str = "", covered: frozenset[str] = frozenset()) -> str:
    """Watch records inside the decline that no attribution block already explains.

    A retro-attribution record is the memory of an earlier attribution run;
    the fresh attribution of that day supersedes it.  Records after the
    trough are about the recovery, not the fall.
    """

    lines: list[str] = []
    for item in result.evidence:
        if item.metadata.get("evidence_scope") != "anomaly":
            continue
        day = str(item.metadata.get("date") or item.as_of or "")[:10]
        if (since and day and day < since) or (until and day and day > until) or day in covered:
            continue
        if "retro_attribution" in str(item.metadata.get("flags") or ""):
            continue
        lines.append(f"- {plain_text(item.claim)} [{item.id}]")
        if len(lines) >= 3:
            break
    return "\n".join(lines)


def _days_after(day: str, days: int) -> str:
    try:
        return (date.fromisoformat(day[:10]) + timedelta(days=days)).isoformat()
    except ValueError:
        return day


def ranking_lead(text: str, results: list[ToolEnvelope]) -> RankingLead:
    """Answer a superlative question directly from a result's ranked table.

    An adapter that returns a table publishes ``metadata["positions"]`` (rows)
    and ``metadata["rankable"]`` (which row field answers which wording).  The
    synthesizer knows nothing about the domain: it picks the rule whose words
    the question uses, sorts, and cites the result's evidence.
    """

    if not _SUPERLATIVE.search(text or ""):
        return RankingLead()
    for result in results:
        rows = result.metadata.get("positions")
        rules = result.metadata.get("rankable")
        if not isinstance(rows, list) or not isinstance(rules, list) or not result.evidence:
            continue
        choice = choose_rankable(text, rules)
        if choice is None:
            continue
        chosen, low, high = choice
        key, text_key = str(chosen.get("field") or ""), str(chosen.get("text") or "")
        ranked = [row for row in rows if isinstance(row, dict) and isinstance(row.get(key), (int, float))]
        if not ranked:
            continue
        ranked.sort(key=lambda row: float(row[key]), reverse=high)
        citation = f"[{result.evidence[0].id}]"

        def label(row: dict[str, Any]) -> str:
            value = row.get(text_key) if text_key else row.get(key)
            return f"{row.get('ticker', '')}（{value}）"

        head, rest = ranked[0], ranked[1:3]
        direction = "最低" if low else "最高"
        lead = f"按{chosen.get('label') or key}排序，{direction}的是 {label(head)}"
        if rest:
            lead += "，其次是 " + "、".join(label(row) for row in rest)
        entities = tuple(str(row.get("ticker", "")) for row in (head, *rest) if row.get("ticker"))
        return RankingLead(f"{lead}{citation}。", entities, result)
    return RankingLead()


def ranked_answer(lead: RankingLead, results: list[ToolEnvelope]) -> str:
    """A ranking answer: the conclusion, one line per named entity, and what was not covered.

    Everything else the run fetched stays in the evidence list; the reader
    asked which one, not for a tour of every holding.
    """

    lines = [lead.text]
    for entity in lead.entities:
        for result in results:
            if result is lead.result or result.subject.upper() != entity.upper():
                continue
            if not result.ok:
                detail = result.errors[0] if result.errors else "未知错误"
                lines.append(f"{entity} {result.capability} 未完成：{detail}")
                continue
            narrative = plain_text(str(result.metadata.get("narrative") or "").strip() or result.summary)
            first = narrative.split("\n")[0].strip()
            if first and "[" not in first and result.evidence:
                first += f" [{result.evidence[0].id}]"
            if first:
                lines.append(first)
    for result in results:
        coverage = result.metadata.get("fan_out_coverage")
        if not isinstance(coverage, dict):
            continue
        uncovered = [str(value) for value in coverage.get("uncovered") or []]
        citation = f" [{result.evidence[0].id}]" if result.evidence else ""
        if uncovered:
            lines.append(f"{result.capability} 未覆盖：{'、'.join(uncovered)}{citation}。")
    return "\n".join(lines)


#: Limitations that describe the adapter, not the data; a user reading "Legacy formatted output" learns nothing.
_INTERNAL_LIMITATIONS = ("Legacy formatted output", "structured field-level evidence")


def user_facing_limitations(limitations) -> list[str]:
    return [str(item) for item in limitations if not any(word in str(item) for word in _INTERNAL_LIMITATIONS)]


class EvidenceSummarySynthesizer:
    """Small deterministic fallback that keeps the V2 core runnable offline."""

    supports_general_knowledge = False

    def diagnostics(self) -> dict[str, Any]:
        return {"outcome": "deterministic", "draft": "", "attempts": []}

    def synthesize(
        self,
        request: NormalizedRequest,
        plan: ExecutionPlan,
        results: list[ToolEnvelope],
        evidence: list[EvidenceItem],
    ) -> str:
        if not results:
            if plan.direct_answer:
                return plan.direct_answer
            if plan.requires_confirmation:
                return "这是一个写操作。请先确认具体操作内容；当前没有执行任何修改。"
            if plan.answer_mode.value == "general_knowledge":
                return "该问题被识别为通用知识问题；尚未接入 Agent V2 的知识回答模型。"
            return "现有信息不足以确定需要调用的能力，请补充标的或希望查询的范围。"

        frame = plan.frame or request.metadata.get("context_frame")
        if isinstance(frame, dict) and frame.get("kind") in {"position", "drawdown", "runup"}:
            opening = frame_lead(frame, results)
            if opening:
                found = position_row(results, str(frame.get("ticker") or ""))
                # The position card was used by the opening (or is not about
                # this stock at all); either way it is not pasted below.
                source = found[0] if found else next((result for result in results if result.capability == "account.portfolio"), None)
                drawdown = next((result for result in results if result.capability in _STRETCH and result.ok), None)
                since = str(drawdown.metrics.get("window_start") or "") if drawdown is not None else ""
                # The stretch itself: peak to trough (or trough to peak).
                # Records after it belong to what came next; filings get
                # three more days, the same margin the attributor reads with.
                peak_date = str((drawdown.metrics.get("peak") or {}).get("date") or "") if drawdown is not None else ""
                trough_date = str((drawdown.metrics.get("trough") or {}).get("date") or "") if drawdown is not None else ""
                stretch = sorted(day for day in (peak_date, trough_date) if day)
                decline_start = stretch[0] if stretch else since
                trough_date = stretch[-1] if stretch else ""
                filings_until = _days_after(trough_date, 3) if trough_date else ""
                has_span = drawdown is not None and any(item.metadata.get("evidence_scope") == "benchmark_span" for item in drawdown.evidence)
                rising = drawdown is not None and (drawdown.capability == "market.runup" or drawdown.metadata.get("direction") == "up")
                attributed = frozenset(str(result.metadata.get("date") or "")[:10] for result in results if result.capability == "market.attribute_move" and result.ok)
                blocks = [opening]
                # A fixed reading order, not the order the tasks finished in:
                # the stretch, its filings and watch records, then each day.
                order = {"market.drawdown": 0, "market.runup": 0, "market.performance": 1, "filings.recent": 2, "filings.read_events": 3, "market.anomaly_history": 4, "web.research": 5, "market.attribute_move": 6}
                ordered = sorted((result for result in results if result is not source), key=lambda result: (order.get(result.capability, 7), str(result.metadata.get("date") or result.as_of or "")))
                for result in ordered:
                    if isinstance(result.metadata.get("fan_out_coverage"), dict):
                        blocks.append(self._render(result))  # just the coverage line
                    elif result.capability == "web.research":
                        blocks.append(web_lines(result, since))
                    elif result.capability in _STRETCH and result.ok:
                        blocks.append(drawdown_lines(result))
                    elif result.capability == "filings.recent" and result.ok and attributed:
                        # The attribution blocks below read the filings; here they are only listed.
                        blocks.append(filing_line(result, decline_start, filings_until, up=rising))
                    elif result.capability == "market.anomaly_history" and result.ok and attributed:
                        blocks.append(anomaly_lines(result, decline_start, trough_date, attributed))
                    elif result.capability in {"research.stock", "research.compare", "filings.recent", "filings.read_events", "market.anomaly_history"}:
                        blocks.append(catalyst_lines(result, since))
                    elif result.capability == "market.performance" and result.ok:
                        # The sector over the same stretch (in the drawdown block)
                        # answers "sector or stock"; last month's relatives do not.
                        if not has_span:
                            blocks.append(benchmark_line(result))
                    else:
                        blocks.append(self._render(result))
                return "\n\n".join(block for block in blocks if block)
        lead = ranking_lead(request.text, results)
        if lead:
            # The ranking answer leads; results about the ranked objects
            # themselves stay in the evidence list, anything else the question
            # also asked for (P&L, risk, macro) is still rendered below.
            source = lead.result
            ranked_subjects = {str(value).upper() for value in ((source.metadata.get("tickers") if source is not None else None) or [])}
            blocks = [ranked_answer(lead, results)]
            others = [
                result
                for result in results
                if result is not source and result.subject.upper() not in ranked_subjects and not isinstance(result.metadata.get("fan_out_coverage"), dict)
            ]
            blocks.extend(self._render(result) for result in others)
            return "\n\n".join(block for block in blocks if block)
        return "\n\n".join(block for block in (self._render(result) for result in results) if block)

    @staticmethod
    def _render(result: ToolEnvelope) -> str:
        if result.metadata.get("citation_kind") == "display":
            return ""  # a display-only envelope (the debate) is shown beside the answer, not in it
        narrative = str(result.metadata.get("narrative") or "").strip()
        if result.ok and narrative:
            return narrative
        lines: list[str] = []
        only_limitations = bool(result.evidence) and all(item.metadata.get("citation_kind") == "limitations" for item in result.evidence)
        if result.summary:
            lines.append(plain_text(result.summary))
        elif not result.ok:
            detail = result.errors[0] if result.errors else "未知错误"
            lines.append(f"{result.capability} 未完成：{detail}")
        elif not only_limitations:
            lines.append(f"{result.capability} 已完成。")
        # A limitations-only result (a fan-out coverage note) is rendered
        # by its limitation line below, which already cites the item.
        for item in [] if only_limitations else result.evidence[:4]:
            if not item.metadata.get("citable", True):
                continue
            if item.claim and item.claim != result.summary:
                lines.append(f"- {plain_text(item.claim)} [{item.id}]")
            elif item.claim and lines:
                lines[-1] += f" [{item.id}]"
        # Limitations often carry figures; cite the adapter's limitation
        # evidence when it exists so the line stays verifiable.
        limitation_item = next((item for item in result.evidence if item.metadata.get("citation_kind") == "limitations"), None)
        suffix = f" [{limitation_item.id}]" if limitation_item is not None else ""
        lines.extend(f"数据限制：{item}{suffix}" for item in user_facing_limitations(result.limitations)[:3])
        return "\n".join(lines)
