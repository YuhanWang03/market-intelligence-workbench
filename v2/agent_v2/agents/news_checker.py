"""News checker: a bounded sub-agent behind ``web.research``.

The one-shot web adapter returned search snippets as evidence.  This
sub-agent searches, reads the pages that matter, and reports dated events
with a quote located in the text it read; a quote that is not in the page
is dropped, an event without a date is dropped.  It runs only with the
user's web consent, under the coordinator's remaining time, and cannot
speak to the user: its output is an evidence envelope like any capability.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Callable
from urllib.parse import urldefrag, urlparse

from v2.agent_v2.agents.base import LoopLimits, Tool, ToolLoop, _schema, limits_for
from v2.agent_v2.agents.filing_reader import locate_quote
from v2.agent_v2.execution import CapabilityRegistry, ExecutionContext
from v2.agent_v2.models import EvidenceItem, ResultStatus, ToolEnvelope

#: ``search(query, days=, max_results=) -> rows`` with title, url, content,
#: published_date and, when the provider supplies it, raw_content.
SearchFn = Callable[..., list[dict[str, Any]]]

_SYSTEM = """你是新闻核查者，不回答用户问题，每一轮调用一个工具。
任务：围绕给定股票和问题，找出可核实、带日期的事件。
工具：search 搜索（英文检索词，含公司名、事件关键词和月份）；read 读正文（每轮最多 2 篇，给结果 id，只读标题和摘要与问题相关的）；finish 报告事件并结束。
规则：finish 里每条事件必须指向你本轮真正拿到的结果 id，并附该结果正文或摘要里的原文引文（30 字以上，原样照抄）；日期用报道里写明的事件日期，没有就用发布日期；
标题党、分析师观点、长期展望不算事件；同一件事只报一次。先搜索，读到足够的正文后尽快结束，最多报 5 条。
如果无法调用工具，就只输出 JSON：{"action":"search","query":"..."}、{"action":"read","ids":["r1"]} 或 {"action":"finish","events":[...],"note":"..."}。"""

_EVENT_SCHEMA = _schema({"date": {"type": "string", "description": "YYYY-MM-DD"}, "text": {"type": "string", "description": "一句中文事件概括"}, "source": {"type": "string", "description": "结果 id，如 r3"}, "quote": {"type": "string", "description": "原文片段，30 字以上，原样照抄"}}, ["date", "text", "source", "quote"])
_FINISH_SCHEMA = _schema({"events": {"type": "array", "items": _EVENT_SCHEMA}, "note": {"type": "string"}, "filing_to_read": {"type": "string", "description": "如果某条报道提到公司在某一天提交了 SEC 申报（8-K、6-K、Form 4 等）而你没有读到申报原文，给出那一天 YYYY-MM-DD，申报阅读者会去读；否则留空"}}, ["events"])

_FINISH_NOW = "轮次已用完。现在只允许 finish：只报你已经拿到结果并能引用原文的事件；没有就返回空 events 并说明。"

_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_WS = re.compile(r"\s+")


@dataclass
class Found:
    """Search results the loop has seen, keyed the way the finish action refers to them."""

    rows: dict[str, dict[str, Any]] = field(default_factory=dict)
    read: dict[str, str] = field(default_factory=dict)
    searches: int = 0
    reads: int = 0


class _CheckLoop(ToolLoop):
    usage_source_name = "agent_v2.news_checker"
    finish_description = "报告有出处、带日期的事件并结束：只报已经拿到结果并能引用原文的事件，没有就返回空 events 并说明。"

    def __init__(self, llm: Any, limits: LoopLimits, *, search: SearchFn, days: int, max_searches: int, max_reads: int, max_chars: int, min_searches: int = 1) -> None:
        self.search = search
        self.days = days
        self.max_searches = max_searches
        self.min_searches = min(min_searches, max_searches)
        self.max_reads = max_reads
        self.max_chars = max_chars
        self.found = Found()
        self._refused_finish = False
        tools = [
            Tool("search", "搜索新闻；query 用英文检索词，含公司名、事件关键词和月份。", _schema({"query": {"type": "string"}}, ["query"]), self._search),
            Tool("read", "读已搜到结果的正文，每轮最多 2 篇；ids 为结果 id。", _schema({"ids": {"type": "array", "items": {"type": "string"}, "maxItems": 2}}, ["ids"]), self._read),
        ]
        super().__init__(llm, limits, tools=tools, finish_parameters=_FINISH_SCHEMA)

    def refuse_finish(self, action: dict[str, Any]) -> str | None:
        # A broad question ("what's the news") deserves a second angle before
        # the loop declares itself done on one search; once is enough to ask.
        if self.found.searches < self.min_searches and not self._refused_finish:
            self._refused_finish = True
            return f"目前只搜索了 {self.found.searches} 次；请换一个角度（另一个事件关键词或时间段）再搜索一次，然后再 finish。"
        return None

    def _search(self, arguments: dict[str, Any]) -> str:
        if self.found.searches >= self.max_searches:
            raise ValueError(f"搜索次数已达上限 {self.max_searches}；请读已有结果或 finish。")
        query = " ".join(str(arguments.get("query") or "").split())[:300]
        if not query:
            raise ValueError("search 需要 query。")
        self.found.searches += 1
        try:
            rows = list(self.search(query, days=self.days, max_results=6) or [])
        except Exception as exc:  # noqa: BLE001 — the provider failing is data for the envelope
            raise ValueError(f"搜索失败：{type(exc).__name__}。") from exc
        lines = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            url, _ = urldefrag(str(row.get("url") or "").strip())
            if urlparse(url).scheme not in {"http", "https"}:
                continue
            existing = next((key for key, value in self.found.rows.items() if value["url"] == url), None)
            if existing is None:
                key = f"r{len(self.found.rows) + 1}"
                self.found.rows[key] = {
                    "url": url,
                    "title": _flat(row.get("title") or urlparse(url).netloc, 200),
                    "content": _flat(row.get("content") or row.get("snippet"), 600),
                    "raw": _flat(row.get("raw_content"), 60_000),
                    "published": str(row.get("published_date") or row.get("published_at") or "")[:10],
                }
                existing = key
            value = self.found.rows[existing]
            lines.append(f"- id={existing} | {value['published'] or '日期未知'} | {value['title']} | {value['content'][:240]}")
        return f"搜索“{query}”的结果：\n" + ("\n".join(lines) or "（无）")

    def _read(self, arguments: dict[str, Any]) -> str:
        ids = [str(value) for value in (arguments.get("ids") or []) if str(value) in self.found.rows][:2]
        if not ids:
            raise ValueError("read 需要已有结果的 id。")
        parts = []
        for key in ids:
            if self.found.reads >= self.max_reads:
                parts.append(f"[{key}] 阅读次数已达上限 {self.max_reads}。")
                continue
            row = self.found.rows[key]
            text = row["raw"] or row["content"]
            if not text:
                parts.append(f"[{key}] 没有可读正文。")
                continue
            self.found.reads += 1
            self.found.read[key] = text[: self.max_chars]
            parts.append(f"[{key}] {row['title']}（{row['published'] or '日期未知'}）\n{self.found.read[key]}")
        return "\n\n".join(parts)


class NewsChecker:
    def __init__(self, llm: Any, search: SearchFn, *, max_rounds: int = 6, max_seconds: float = 75.0, max_searches: int = 3, max_reads: int = 3, max_chars: int = 5000, max_events: int = 5) -> None:
        """Bounds for one run; ``min_searches`` is per call because it depends on the question."""

        self.llm = llm
        self.search = search
        self.max_rounds = max(1, max_rounds)
        self.max_seconds = max(5.0, max_seconds)
        self.max_searches = max(1, max_searches)
        self.max_reads = max(0, max_reads)
        self.max_chars = max(500, max_chars)
        self.max_events = max(1, max_events)

    def run(self, ticker: str, context: ExecutionContext, *, query: str, topic: str = "", recency_days: int = 30, today: date | None = None, min_searches: int = 1) -> ToolEnvelope:
        current = today or date.today()
        days = min(3650, max(1, int(recency_days or 30)))
        since = (current - timedelta(days=days)).isoformat()
        limits = limits_for(context, max_rounds=self.max_rounds, max_seconds=self.max_seconds)
        loop = _CheckLoop(self.llm, limits, search=self.search, days=days, max_searches=self.max_searches, max_reads=self.max_reads, max_chars=self.max_chars, min_searches=max(1, int(min_searches or 1)))
        # ``ticker`` may name several stocks ("MU,SNDK") when the question compares them.
        tickers = [part for part in re.split(r"[,、/\s]+", (ticker or "").upper()) if part]
        ticker = "、".join(tickers)
        task = f"股票：{ticker or '（未指定）'}\n问题：{query}\n关注区间：{since} 至 {current.isoformat()}\n可用动作：search、read、finish"
        outcome = loop.run(_SYSTEM, task, finish_prompt=_FINISH_NOW)
        raw = [row for row in (outcome.final.get("events") or []) if isinstance(row, dict)] if outcome.finished else []
        events, dropped = _verify_events(raw[: self.max_events * 2], loop.found)
        note = str(outcome.final.get("note") or "") if outcome.finished else outcome.note
        if dropped:
            note = (note + "；" if note else "") + f"{dropped} 条事件没有日期或引文与正文不符，已丢弃"
        envelope = self._envelope(ticker, query, topic, since, current.isoformat(), events[: self.max_events], loop.found, note=note, outcome=outcome, run_id=context.run_id)
        envelope.metadata["agent"]["yield"] = {"kept": min(len(events), self.max_events), "dropped": dropped}
        # A story that names a filing the checker did not read: ask the run
        # board for the filing reader on that day, within the run's cap.
        follow_up = str(outcome.final.get("filing_to_read") or "")[:10] if outcome.finished else ""
        board = getattr(context, "board", None)
        if follow_up and _ISO.match(follow_up) and tickers and board is not None:
            accepted = board.request("filings.read_events", {"ticker": tickers[0], "around": follow_up}, purpose=f"read the {tickers[0]} filing of {follow_up} a news story named", requested_by="news_checker")
            envelope.metadata["agent"]["follow_up"] = {"filings_around": follow_up, "accepted": accepted}
            if accepted:
                envelope.metadata["agent"].setdefault("notes", []).append(f"报道提到 {follow_up} 的申报，已请申报阅读者读取")
        return envelope

    def _envelope(self, ticker: str, query: str, topic: str, since: str, until: str, events: list[dict[str, Any]], found: Found, *, note: str, outcome, run_id: str) -> ToolEnvelope:
        prefix = hashlib.sha1(f"{ticker}|{query}".encode("utf-8")).hexdigest()[:8]
        evidence: list[EvidenceItem] = []
        for index, event in enumerate(events, start=1):
            row = found.rows[event["source"]]
            netloc = urlparse(row["url"]).netloc.lower()
            evidence.append(
                EvidenceItem(
                    id=f"web-{prefix}-{index}",
                    entity=ticker.upper(),
                    claim=f"{ticker.upper() + ' ' if ticker and '、' not in ticker else ''}{event['date']}：{event['text']}（新闻：“{event['quote'][:160]}”）。",
                    as_of=event["date"],
                    source_id=f"web:{netloc}",
                    source_title=row["title"],
                    source_url=row["url"],
                    confidence=0.7,
                    producer_run_id=run_id,
                    metadata={"evidence_type": "news_event", "topic": topic, "date": event["date"], "quote": event["quote"], "published_at": row["published"], "read": event["source"] in found.read},
                )
            )
        limitations: list[str] = []
        if note:
            limitations.append(note)
        if not evidence:
            reason = f"{ticker or '该问题'} {since} 至 {until} 的网页搜索未找到可核实、带日期的事件。"
            evidence.append(EvidenceItem(id=f"web-{prefix}-none", entity=ticker.upper(), claim=reason, source_id="web_search", source_title="Web search", producer_run_id=run_id, metadata={"citation_kind": "limitations", "verified": True}))
            limitations.append(reason)
        narrative = ("；".join(f"{item.metadata['date']} {item.claim.split('：', 1)[1].split('（', 1)[0]}[{item.id}]" for item in evidence if item.metadata.get("evidence_type") == "news_event") + "。") if events else f"{evidence[0].claim.rstrip('。')}[{evidence[0].id}]。"
        status = ResultStatus.COMPLETED if events else ResultStatus.PARTIAL_DATA
        if outcome.stop_reason in {"no_model", "no_budget"}:
            status = ResultStatus.PARTIAL_DATA
        return ToolEnvelope(
            "web.research",
            status,
            subject=ticker.upper(),
            as_of=until,
            summary=f"网页核查：搜索 {found.searches} 次，读了 {found.reads} 篇，{len(events)} 条有出处的事件。",
            findings=[{"date": event["date"], "text": event["text"], "url": found.rows[event["source"]]["url"]} for event in events],
            evidence=evidence,
            limitations=limitations,
            metrics={"searches": found.searches, "reads": found.reads, "events": len(events), "rounds": outcome.rounds, "llm_calls": outcome.calls, "elapsed_ms": outcome.elapsed_ms, "stop_reason": outcome.stop_reason, "seconds_allowed": round(outcome.seconds_allowed, 1)},
            metadata={
                "narrative": narrative,
                "query": query,
                "topic": topic,
                "since": since,
                "until": until,
                "dates": [event["date"] for event in events],
                "trace": list(outcome.trace),
                "agent": {"name": "news_checker", "label": "新闻核查", "subject": f"{ticker or query[:20]} {since}…{until}", "rounds": outcome.rounds, "llm_calls": outcome.calls, "elapsed_ms": outcome.elapsed_ms, "seconds_allowed": round(outcome.seconds_allowed, 1), "stop_reason": outcome.stop_reason, "calls": {"search": found.searches, "read": found.reads}},
            },
        )


def _flat(value: Any, limit: int) -> str:
    text = _WS.sub(" ", str(value or "")).strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _verify_events(events: list[dict[str, Any]], found: Found) -> tuple[list[dict[str, Any]], int]:
    """Keep events with an ISO date, a known source and a quote located in what was read (or the snippet)."""

    kept: list[dict[str, Any]] = []
    dropped = 0
    seen: set[tuple[str, str]] = set()
    for event in events:
        source = str(event.get("source") or "")
        day = str(event.get("date") or "")[:10]
        text = _flat(event.get("text"), 200)
        quote = str(event.get("quote") or "").strip()
        row = found.rows.get(source)
        if row is None or not _ISO.match(day) or not text or not quote:
            dropped += 1
            continue
        haystack = found.read.get(source) or ""
        located = locate_quote(quote, haystack, minimum=30, share=0.6) if haystack else None
        if located is None:
            located = locate_quote(quote, row["content"], minimum=30, share=0.6)
        if located is None:
            dropped += 1
            continue
        if (day, text) in seen:
            continue
        seen.add((day, text))
        kept.append({"date": day, "text": text, "source": source, "quote": located})
    return kept, dropped


def _default_search(query: str, *, days: int, max_results: int) -> list[dict[str, Any]]:
    import os

    from v2.data.metered import TavilyClient

    client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    recent = days <= 30
    response = client.search(query=query, max_results=max_results, topic="news" if recent else "general", days=days, search_depth="basic", include_raw_content=True)
    return list(response.get("results", []))


def register_news_checker(registry: CapabilityRegistry, llm: Any, *, search: SearchFn | None = None, today_factory: Callable[[], date] = date.today, **limits: Any) -> None:
    checker = NewsChecker(llm, search or _default_search, **limits)

    def handler(arguments: dict[str, Any], context: ExecutionContext) -> ToolEnvelope:
        tickers = [str(value).upper() for value in (arguments.get("tickers") or []) if value]
        return checker.run(
            ",".join(tickers) if tickers else str(arguments.get("ticker") or "").upper(),
            context,
            query=str(arguments.get("query") or ""),
            topic=str(arguments.get("topic") or "general"),
            recency_days=int(arguments.get("recency_days") or 30),
            today=today_factory(),
            min_searches=int(arguments.get("min_searches") or 1),
        )

    registry.register("web.research", handler)
