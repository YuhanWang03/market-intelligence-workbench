"""The shared toolbox and the investigator a planner assembles from it.

The filing reader, the news checker and the attributor each grew their own
"read a page", "read a section" and "search" handlers.  The toolbox is the
one place those tools live: declared once (name, description, schema,
handler), backed by the same ports the runtime already wires (the web
search, the EDGAR source, the anomaly memory), with the state a finish
needs to be verified (what was searched, what was read).

``Investigator`` is a sub-agent with no fixed role: the planner gives it a
task and the tools it may use, it works within the usual bounds, and every
finding it reports must quote text it actually read; unquoted findings are
dropped, as everywhere else.  It is registered as ``agent.investigate``.
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Callable
from urllib.parse import urldefrag, urlparse

from v2.agent_v2.agents.base import LoopLimits, Tool, ToolLoop, _schema, limits_for
from v2.agent_v2.agents.filing_reader import EdgarFilingSource, FilingRef, FilingSource, locate_quote
from v2.agent_v2.execution import CapabilityRegistry, ExecutionContext
from v2.agent_v2.models import EvidenceItem, ResultStatus, ToolEnvelope

SearchFn = Callable[..., list[dict[str, Any]]]
_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_WS = re.compile(r"\s+")

#: The tools a planner may hand to the investigator.
TOOL_NAMES = ("search_news", "read_page", "list_filings", "read_filing", "recall_memory")
#: The ones that read the web; withheld unless the run has the user's web consent.
WEB_TOOLS = ("search_news", "read_page")


def quote_rule(evidence_id: str, quote: str, *, head: int = 30) -> dict[str, str]:
    """The answer-level rule that an answer citing a finding carries its quote somewhere: the quote's opening words, whitespace-tolerant."""

    opening = " ".join(str(quote or "").split())[:head].rstrip(" ,.;:，。；：")
    pattern = r"\s*".join(re.escape(part) for part in opening.split(" ")) if opening else ""
    return {"quote_of": evidence_id, "require": pattern, "warning": f"引用了调查发现 [{evidence_id}] 却没有照抄它的引文；把原文“{str(quote or '')[:80]}”原样放进回答里引用它的那一行"}


def passages_around(text: str, find: str, *, limit: int, radius: int = 700) -> str:
    """The passages of ``text`` around each keyword in ``find`` (comma-separated, case-insensitive), merged and bounded to ``limit`` chars."""

    terms = [term.strip() for term in re.split(r"[,，、;；|]+", find) if term.strip()]
    low = text.lower()
    spans: list[tuple[int, int]] = []
    for term in terms:
        start = 0
        while True:
            at = low.find(term.lower(), start)
            if at < 0:
                break
            spans.append((max(0, at - radius), min(len(text), at + len(term) + radius)))
            start = at + len(term)
    if not spans:
        return ""
    spans.sort()
    merged: list[tuple[int, int]] = []
    for begin, end in spans:
        if merged and begin <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((begin, end))
    pieces: list[str] = []
    total = 0
    for begin, end in merged:
        piece = text[begin:end]
        if total + len(piece) > limit:
            piece = piece[: limit - total]
        if piece:
            pieces.append(piece)
            total += len(piece)
        if total >= limit:
            break
    return "\n……\n".join(pieces)


def _flat(value: Any, limit: int) -> str:
    return _WS.sub(" ", str(value or "")).strip()[:limit]


@dataclass
class ToolboxState:
    """What the tools fetched, keyed the way a finish refers to it."""

    pages: dict[str, dict[str, Any]] = field(default_factory=dict)
    read_pages: dict[str, str] = field(default_factory=dict)
    filings: list[FilingRef] = field(default_factory=list)
    read_sections: dict[tuple[int, str], str] = field(default_factory=dict)
    memory: list[dict[str, Any]] = field(default_factory=list)
    calls: dict[str, int] = field(default_factory=dict)

    def text_for(self, source: str) -> tuple[str, str, str]:
        """``(text, url, label)`` for a source key: ``p3`` (a page) or ``f2:s1`` (a filing section)."""

        if source in self.read_pages:
            row = self.pages[source]
            return self.read_pages[source], row["url"], row["title"]
        if source in self.pages:
            row = self.pages[source]
            return row["content"], row["url"], row["title"]
        if ":" in source:
            index, section = source.split(":", 1)
            try:
                key = (int(index.lstrip("f")), section)
            except ValueError:
                return "", "", ""
            if key in self.read_sections and 1 <= key[0] <= len(self.filings):
                ref = self.filings[key[0] - 1]
                return self.read_sections[key], ref.url, f"{ref.form} {ref.filing_date}"
        return "", "", ""


class Toolbox:
    def __init__(self, *, search: SearchFn | None = None, filing_source: FilingSource | None = None, recall: Callable[[str, str, int], list[Any]] | None = None, max_chars: int = 5000, max_searches: int = 3, max_reads: int = 4, days: int = 30, today: date | None = None) -> None:
        self.search = search
        self.filing_source = filing_source
        self.recall = recall
        self.max_chars = max_chars
        self.max_searches = max_searches
        self.max_reads = max_reads
        self.days = days
        self.today = today or date.today()
        self.state = ToolboxState()

    def available(self) -> tuple[str, ...]:
        names = []
        if self.search is not None:
            names += ["search_news", "read_page"]
        if self.filing_source is not None:
            names += ["list_filings", "read_filing"]
        if self.recall is not None:
            names.append("recall_memory")
        return tuple(names)

    def tools(self, names: tuple[str, ...] | list[str] | None = None) -> list[Tool]:
        wanted = [name for name in (names or TOOL_NAMES) if name in self.available()]
        table = {
            "search_news": Tool("search_news", "搜索新闻；query 用英文检索词，含公司名、事件关键词和月份。结果带 id（p1、p2…）。", _schema({"query": {"type": "string"}}, ["query"]), self._search_news),
            "read_page": Tool("read_page", "读一条搜索结果的正文；id 为结果 id。", _schema({"id": {"type": "string"}}, ["id"]), self._read_page),
            "list_filings": Tool("list_filings", "列出某只股票一段时间内的 SEC 申报，结果带序号 f1、f2…和章节列表。forms 指定表格（如 [\"10-K\"]、[\"10-Q\",\"8-K\"]、[\"4\"]），默认只列 8-K/6-K；年报条款要传 [\"10-K\"]。", _schema({"ticker": {"type": "string"}, "since": {"type": "string", "description": "YYYY-MM-DD，默认 90 天前"}, "until": {"type": "string", "description": "YYYY-MM-DD，默认今天"}, "forms": {"type": "array", "items": {"type": "string"}, "maxItems": 4}}, ["ticker"]), self._list_filings),
            "read_filing": Tool("read_filing", "读申报的一个章节；filing 为序号（1 起），section 为章节 id。长章节只返回开头；给 find（关键词，逗号分隔，如 \"FSD, Full Self-Driving, Autopilot\"）则返回关键词前后的段落。", _schema({"filing": {"type": "integer"}, "section": {"type": "string"}, "find": {"type": "string"}}, ["filing", "section"]), self._read_filing),
            "recall_memory": Tool("recall_memory", "查盯盘记忆里某只股票的异动记录；query 为关键词。", _schema({"ticker": {"type": "string"}, "query": {"type": "string"}}, ["ticker"]), self._recall_memory),
        }
        return [table[name] for name in wanted]

    # -- handlers ------------------------------------------------------------------

    def _count(self, name: str) -> int:
        self.state.calls[name] = self.state.calls.get(name, 0) + 1
        return self.state.calls[name]

    def _search_news(self, arguments: dict[str, Any]) -> str:
        if self.state.calls.get("search_news", 0) >= self.max_searches:
            raise ValueError(f"搜索次数已达上限 {self.max_searches}；请读已有结果或 finish。")
        query = " ".join(str(arguments.get("query") or "").split())[:300]
        if not query:
            raise ValueError("search_news 需要 query。")
        self._count("search_news")
        rows = list(self.search(query, days=self.days, max_results=6) or [])
        lines = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            url, _ = urldefrag(str(row.get("url") or "").strip())
            if urlparse(url).scheme not in {"http", "https"}:
                continue
            key = next((k for k, v in self.state.pages.items() if v["url"] == url), None)
            if key is None:
                key = f"p{len(self.state.pages) + 1}"
                self.state.pages[key] = {"url": url, "title": _flat(row.get("title") or urlparse(url).netloc, 200), "content": _flat(row.get("content") or row.get("snippet"), 600), "raw": _flat(row.get("raw_content"), 60_000), "published": str(row.get("published_date") or row.get("published_at") or "")[:10]}
            value = self.state.pages[key]
            lines.append(f"- id={key} | {value['published'] or '日期未知'} | {value['title']} | {value['content'][:240]}")
        return f"搜索“{query}”的结果：\n" + ("\n".join(lines) or "（无）")

    def _read_page(self, arguments: dict[str, Any]) -> str:
        key = str(arguments.get("id") or "")
        if key not in self.state.pages:
            raise ValueError("read_page 需要已有结果的 id。")
        if self.state.calls.get("read_page", 0) >= self.max_reads:
            raise ValueError(f"阅读次数已达上限 {self.max_reads}。")
        row = self.state.pages[key]
        text = row["raw"] or row["content"]
        if not text:
            raise ValueError(f"[{key}] 没有可读正文。")
        self._count("read_page")
        self.state.read_pages[key] = text[: self.max_chars]
        return f"[{key}] {row['title']}（{row['published'] or '日期未知'}）\n{self.state.read_pages[key]}"

    def _list_filings(self, arguments: dict[str, Any]) -> str:
        ticker = str(arguments.get("ticker") or "").upper()
        if not ticker:
            raise ValueError("list_filings 需要 ticker。")
        since = str(arguments.get("since") or (self.today - timedelta(days=90)).isoformat())[:10]
        until = str(arguments.get("until") or self.today.isoformat())[:10]
        self._count("list_filings")
        forms = tuple(str(form).upper() for form in (arguments.get("forms") or []) if str(form).strip())[:4]
        try:
            refs = list(self.filing_source.list_filings(ticker, since, until, forms or None))[:6]
        except TypeError:  # a source without the forms parameter: the current reports
            refs = list(self.filing_source.list_filings(ticker, since, until))[:6]
        self.state.filings = refs
        lines = []
        for index, ref in enumerate(refs, 1):
            try:
                sections = self.filing_source.outline(ref)
            except Exception:  # noqa: BLE001 — a filing whose outline fails to load can still be listed
                sections = []
            names = "、".join(f"{getattr(section, 'id', '')}（{_flat(getattr(section, 'title', ''), 40)}）" for section in sections[:8])
            lines.append(f"- f{index} | {ref.form} | {ref.filing_date} | 章节：{names or '（无）'}")
        return f"{ticker} {since} 至 {until} 的申报：\n" + ("\n".join(lines) or "（无）")

    def _read_filing(self, arguments: dict[str, Any]) -> str:
        try:
            index = int(arguments.get("filing"))
        except (TypeError, ValueError):
            raise ValueError("read_filing 需要 filing 序号。") from None
        section = str(arguments.get("section") or "")
        if not 1 <= index <= len(self.state.filings):
            raise ValueError("filing 序号不在 list_filings 的结果里。")
        if self.state.calls.get("read_filing", 0) >= self.max_reads:
            raise ValueError(f"阅读次数已达上限 {self.max_reads}。")
        self._count("read_filing")
        body = self.filing_source.read(self.state.filings[index - 1], section)
        find = str(arguments.get("find") or "").strip()
        text, heading = body[: self.max_chars], f"申报 f{index} 章节 {section} 的文本"
        if find:
            windows = passages_around(body, find, limit=self.max_chars)
            if windows:
                text, heading = windows, f"申报 f{index} 章节 {section} 里含“{find}”的段落"
            else:
                heading = f"申报 f{index} 章节 {section} 里没有“{find}”；章节开头"
        if len(body) > len(text):
            heading += f"（章节共 {len(body)} 字，只返回 {len(text)} 字）"
        seen = self.state.read_sections.get((index, section), "")
        # A second read of the same section (other keywords) adds to what a quote may be checked against.
        self.state.read_sections[(index, section)] = f"{seen}\n……\n{text}" if seen and text not in seen else (seen or text)
        return f"{heading}：\n{text or '（空）'}"

    def _recall_memory(self, arguments: dict[str, Any]) -> str:
        ticker = str(arguments.get("ticker") or "").upper()
        if not ticker:
            raise ValueError("recall_memory 需要 ticker。")
        self._count("recall_memory")
        rows = list(self.recall(ticker, str(arguments.get("query") or f"{ticker} 异动"), self.days) or [])
        lines = []
        for row in rows[:6]:
            day = str(getattr(row, "date", "") or "")[:10]
            entry = {"date": day, "flags": str(getattr(row, "flags", "") or ""), "doc": _flat(getattr(row, "doc", ""), 300)}
            self.state.memory.append(entry)
            lines.append(f"- {day} | {entry['flags'] or '无标志'} | {entry['doc']}")
        return "盯盘记忆：\n" + ("\n".join(lines) or "（无）")


_SYSTEM = """你是投研调查员，不回答用户问题，每一轮调用一个工具。
任务由规划器给出；你只能用提供的工具，按需搜索、读正文、列申报、读章节、查记忆，读到足够内容就 finish。
finish 里每条发现（finding）必须指向你本轮真正读到的来源（source 为结果 id 如 p2，或申报章节如 f1:s3），并附该来源里原样照抄的引文（quote，30 字以上）；日期用来源里写明的事件日期。没有可核实的发现就返回空 findings 并在 note 里说明。
不要编造来源，不要用常识充当发现。"""

_FINISH_SCHEMA = _schema({"findings": {"type": "array", "items": _schema({"date": {"type": "string", "description": "YYYY-MM-DD"}, "text": {"type": "string", "description": "一句中文概括"}, "source": {"type": "string", "description": "p2 或 f1:s3"}, "quote": {"type": "string"}}, ["date", "text", "source", "quote"])}, "note": {"type": "string"}}, ["findings"])


class _InvestigateLoop(ToolLoop):
    usage_source_name = "agent_v2.investigator"
    finish_description = "报告有出处、带引文的发现并结束；没有就返回空 findings 并说明。"

    def __init__(self, llm: Any, limits: LoopLimits, toolbox: Toolbox, names: tuple[str, ...]) -> None:
        self.toolbox = toolbox
        super().__init__(llm, limits, tools=toolbox.tools(names), finish_parameters=_FINISH_SCHEMA)


class Investigator:
    def __init__(self, llm: Any, toolbox_factory: Callable[[], Toolbox], *, max_rounds: int = 8, max_seconds: float = 90.0, max_findings: int = 6) -> None:
        self.llm = llm
        self.toolbox_factory = toolbox_factory
        self.max_rounds = max_rounds
        self.max_seconds = max_seconds
        self.max_findings = max_findings

    def run(self, task: str, context: ExecutionContext, *, ticker: str = "", tools: tuple[str, ...] | list[str] | None = None, recency_days: int = 30) -> ToolEnvelope:
        started = time.monotonic()
        toolbox = self.toolbox_factory()
        toolbox.days = min(3650, max(1, int(recency_days or 30)))
        names = tuple(name for name in (tools or TOOL_NAMES) if name in toolbox.available())
        # The web is read only with the user's consent, as for every other capability.
        withheld = tuple(name for name in names if name in WEB_TOOLS) if not getattr(context, "allow_web", False) else ()
        names = tuple(name for name in names if name not in withheld)
        limits = limits_for(context, max_rounds=self.max_rounds, max_seconds=self.max_seconds)
        loop = _InvestigateLoop(self.llm, limits, toolbox, names)
        brief = f"任务：{task}\n股票：{ticker or '（未指定）'}\n今天：{toolbox.today.isoformat()}\n可用工具：{'、'.join(names) or '（无）'}"
        outcome = loop.run(_SYSTEM, brief, finish_prompt="轮次已用完。现在只允许 finish：只报你已经读到并能引用原文的发现；没有就返回空 findings 并说明。")
        raw = [row for row in (outcome.final.get("findings") or []) if isinstance(row, dict)] if outcome.finished else []
        findings, dropped = _verify_findings(raw[: self.max_findings * 2], toolbox.state)
        note = str(outcome.final.get("note") or "") if outcome.finished else outcome.note
        if withheld:
            note = (note + "；" if note else "") + "网页未授权，未搜索新闻"
        if dropped:
            note = (note + "；" if note else "") + f"{dropped} 条发现没有可核对的引文，已丢弃"
        subject = ticker or task[:24]
        evidence: list[EvidenceItem] = []
        for finding in findings[: self.max_findings]:
            text, url, label = toolbox.state.text_for(finding["source"])
            digest = hashlib.sha1(f"{task}|{finding['source']}|{finding['quote']}".encode("utf-8")).hexdigest()[:12]
            evidence.append(EvidenceItem(
                id=f"evidence-investigate-{digest}",
                entity=ticker.upper() if ticker else subject,
                claim=f"{ticker.upper() + ' ' if ticker else ''}{finding['date']}：{finding['text']}（{label}：“{finding['quote'][:160]}”）。",
                as_of=finding["date"],
                source_id=f"web:{urlparse(url).netloc}" if url.startswith("http") and "sec.gov" not in url else ("sec_edgar" if url else "investigator"),
                source_title=label,
                source_url=url,
                producer_run_id=context.run_id,
                metadata={"evidence_type": "investigation", "date": finding["date"], "quote": finding["quote"], "source": finding["source"], "verified": True},
            ))
        if not evidence:
            evidence.append(EvidenceItem(id=f"evidence-investigate-none-{hashlib.sha1(task.encode('utf-8')).hexdigest()[:10]}", entity=subject, claim=f"针对“{task[:60]}”的调查没有找到可核实、带引文的发现。", source_id="investigator", source_title="调查", producer_run_id=context.run_id, metadata={"citation_kind": "limitations", "verified": True}))
        elapsed_ms = int((time.monotonic() - started) * 1000)
        status = ResultStatus.COMPLETED if findings else ResultStatus.PARTIAL_DATA
        if outcome.stop_reason in {"no_model", "no_budget"}:
            status = ResultStatus.PARTIAL_DATA
        narrative = ("；".join(f"{item.metadata['date']} {item.claim.split('：', 1)[1].split('（', 1)[0]}[{item.id}]" for item in evidence if item.metadata.get("evidence_type") == "investigation") + "。") if findings else f"{evidence[0].claim.rstrip('。')}[{evidence[0].id}]。"
        return ToolEnvelope(
            "agent.investigate",
            status,
            subject=subject,
            summary=f"调查：{len(findings)} 条有引文的发现（{'、'.join(f'{name} {count}' for name, count in toolbox.state.calls.items()) or '未调用工具'}）。",
            findings=[{"date": finding["date"], "text": finding["text"], "source": finding["source"]} for finding in findings],
            evidence=evidence,
            limitations=[note] if note else [],
            metrics={"findings": len(findings), "dropped": dropped, "rounds": outcome.rounds, "llm_calls": outcome.calls, "elapsed_ms": elapsed_ms, "stop_reason": outcome.stop_reason, "seconds_allowed": round(outcome.seconds_allowed, 1), **toolbox.state.calls},
            metadata={
                "narrative": narrative,
                "task": task,
                "answer_constraints": [quote_rule(item.id, item.metadata["quote"]) for item in evidence if item.metadata.get("evidence_type") == "investigation"],
                "agent": {"name": "investigator", "label": "现场调查", "subject": f"{subject} {task[:20]}", "rounds": outcome.rounds, "llm_calls": outcome.calls, "elapsed_ms": elapsed_ms, "seconds_allowed": round(outcome.seconds_allowed, 1), "stop_reason": outcome.stop_reason, "calls": dict(toolbox.state.calls), "yield": {"kept": len(findings), "dropped": dropped}, "tools": list(names), **({"withheld": list(withheld)} if withheld else {})},
                "trace": outcome.trace,
            },
        )


def _verify_findings(findings: list[dict[str, Any]], state: ToolboxState) -> tuple[list[dict[str, Any]], int]:
    kept: list[dict[str, Any]] = []
    dropped = 0
    for finding in findings:
        day = str(finding.get("date") or "")[:10]
        text = " ".join(str(finding.get("text") or "").split())[:200]
        source = str(finding.get("source") or "").strip()
        quote = str(finding.get("quote") or "").strip()
        haystack, _url, _label = state.text_for(source)
        if not _ISO.match(day) or not text or not quote or not haystack:
            dropped += 1
            continue
        located = locate_quote(quote, haystack, minimum=30, share=0.6)
        if located is None:
            dropped += 1
            continue
        kept.append({"date": day, "text": text, "source": source, "quote": located})
    return kept, dropped


def register_investigator(registry: CapabilityRegistry, llm: Any, *, search: SearchFn | None = None, filing_source: FilingSource | None = None, recall: Callable[[str, str, int], list[Any]] | None = None, today_factory: Callable[[], date] = date.today, **limits: Any) -> None:
    def toolbox() -> Toolbox:
        return Toolbox(search=search, filing_source=filing_source or EdgarFilingSource(), recall=recall, today=today_factory())

    investigator = Investigator(llm, toolbox, **limits)

    def handler(arguments: dict[str, Any], context: ExecutionContext) -> ToolEnvelope:
        return investigator.run(
            str(arguments.get("task") or ""),
            context,
            ticker=str(arguments.get("ticker") or "").upper(),
            tools=[str(name) for name in (arguments.get("tools") or [])] or None,
            recency_days=int(arguments.get("recency_days") or 30),
        )

    registry.register("agent.investigate", handler)
