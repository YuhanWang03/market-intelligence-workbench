"""The move attributor: a bounded sub-agent that explains one day's move for one stock.

Given a ticker and a date it assembles the day's facts from daily closes,
then lets the model decide what to look at: news (only when the user
allowed web search), the filings around the date (through the filing
reader, itself bounded), and the monitor's anomaly memory.  Every reason
it reports must point at something it actually fetched and quote it; a
reason supported only by memory is at most medium confidence.  What it
concludes is written back to the anomaly memory under its own id, so a
later question about the same stretch finds it.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable

from v2.agent_v2.agents.base import LoopLimits, Tool, ToolLoop, _schema, limits_for, structured_call
from v2.agent_v2.agents.filing_reader import EdgarFilingSource, FilingReader, FilingSource, locate_quote
from v2.agent_v2.execution import CapabilityRegistry, ExecutionContext
from v2.agent_v2.models import EvidenceItem, ResultStatus, ToolEnvelope

_WS = re.compile(r"\s+")
from v2.agent_v2.adapters.market import _CANDIDATE_RULE, _COUNT_LEAK_RULE  # noqa: E402 — the wording rules live next to the market data


@dataclass
class DayFacts:
    ticker: str
    date: str
    close: float
    change: float | None
    volume: int
    average_volume_30d: float | None
    volume_ratio: float | None
    high_52w: float
    low_52w: float
    sector_etf: str = ""
    sector_return_1d: float | None = None
    relative_1d: float | None = None
    is_latest: bool = False
    #: The day is today and the session is still open: the bar is not final.
    is_intraday: bool = False
    observed_at_label: str = ""


@dataclass
class Gathered:
    """What the loop fetched, keyed the way the finish action must refer to it."""

    news: dict[str, dict[str, Any]] = field(default_factory=dict)
    reader_runs: list[dict[str, Any]] = field(default_factory=list)
    filing_events: dict[str, EvidenceItem] = field(default_factory=dict)
    filing_notes: list[EvidenceItem] = field(default_factory=list)
    memory: dict[str, dict[str, Any]] = field(default_factory=dict)
    news_calls: int = 0
    reader_calls: int = 0
    memory_calls: int = 0


def day_facts(ticker: str, day: str, prices: list[Any], sector_etf: str = "", sector_prices: list[Any] | None = None, *, now: datetime | None = None) -> DayFacts | None:
    """The day's price, move, volume ratio and 52-week range from daily bars.

    With ``now`` given, a bar dated today during the regular session is
    marked intraday: its close is the last trade and its volume is partial.
    """

    rows = [row for row in prices if str(row.time)[:10] <= day]
    if len(rows) < 2:
        return None
    latest, previous = rows[-1], rows[-2]
    prior = rows[-31:-1]
    average = sum(float(row.volume) for row in prior) / len(prior) if prior else None
    window = rows[-252:]
    change = float(latest.close) / float(previous.close) - 1 if float(previous.close) > 0 else None
    sector_return = None
    if sector_prices:
        sector_rows = [row for row in sector_prices if str(row.time)[:10] <= str(latest.time)[:10]]
        if len(sector_rows) >= 2 and str(sector_rows[-1].time)[:10] == str(latest.time)[:10] and float(sector_rows[-2].close) > 0:
            sector_return = float(sector_rows[-1].close) / float(sector_rows[-2].close) - 1
    return DayFacts(
        ticker=ticker,
        date=str(latest.time)[:10],
        close=float(latest.close),
        change=change,
        volume=int(latest.volume),
        average_volume_30d=average,
        volume_ratio=(float(latest.volume) / average) if average else None,
        high_52w=max(float(row.close) for row in window),
        low_52w=min(float(row.close) for row in window),
        sector_etf=sector_etf,
        sector_return_1d=sector_return,
        relative_1d=(change - sector_return) if change is not None and sector_return is not None else None,
        is_latest=rows[-1] is prices[-1],
        **_session_state(str(latest.time)[:10], now),
    )


def _session_state(day: str, now: datetime | None) -> dict[str, Any]:
    if now is None:
        return {}
    from v2.agent_v2.adapters.market import _observation_state

    observation = _observation_state(day, now)
    return {"is_intraday": bool(observation["is_intraday"]), "observed_at_label": str(observation["observed_at_label"]) if observation["is_intraday"] else ""}


_SYSTEM = """你是异动归因者，不回答用户问题，每一轮调用一个工具。
任务：解释给定股票在给定日期的涨跌原因。
工具：news 搜当日新闻（仅在提供时可用，英文检索词含公司名和日期）；filing_events 读当日附近的申报；memory 查盯盘记忆；finish 报告原因并结束。
规则：finish 里每条原因必须指向你本轮真正拿到的来源并附原文引文：source 为 {"kind":"news","url":"..."} 或 {"kind":"filing","id":"证据 id"} 或 {"kind":"memory","date":"YYYY-MM-DD"}；
只有新闻或申报原文直接、同日、幅度相称地支持时才能标"高"；仅有盯盘记忆支持的最多标"中"；市场整体波动、传闻、幅度不相称的标"低"；找不到原因就返回空 reasons 并在 note 里说明。不要编造来源。
如果无法调用工具，就只输出 JSON：{"action":"news","query":"..."}、{"action":"filing_events"}、{"action":"memory","query":"..."} 或 {"action":"finish","reasons":[...],"next_steps":[...],"note":"..."}。"""

_SOURCE_SCHEMA = {"type": "object", "properties": {"kind": {"type": "string", "enum": ["news", "filing", "memory"]}, "url": {"type": "string"}, "id": {"type": "string"}, "date": {"type": "string"}}, "required": ["kind"]}
_REASON_SCHEMA = _schema({"text": {"type": "string", "description": "一句中文原因"}, "confidence": {"type": "string", "enum": ["高", "中", "低"]}, "source": _SOURCE_SCHEMA, "quote": {"type": "string", "description": "从该来源原样复制的一段原文"}}, ["text", "confidence", "source", "quote"])
_FINISH_SCHEMA = _schema({"reasons": {"type": "array", "items": _REASON_SCHEMA}, "next_steps": {"type": "array", "items": {"type": "string"}}, "note": {"type": "string", "description": "一句话说明结论强弱或还缺什么"}}, ["reasons"])

_FINISH_NOW = "轮次已用完。现在只允许 finish：只报你已经拿到来源并能引用原文的原因；没有就返回空 reasons 并说明。"

CHALLENGE_TOOL = {"type": "function", "function": {"name": "verdict", "description": "给出对高置信度驱动的反对意见。", "parameters": {"type": "object", "properties": {"objection": {"type": "string", "description": "一句中文，指出具体不足；没有就留空"}, "downgrade": {"type": "boolean"}}, "required": ["objection", "downgrade"]}}}

_CHALLENGE = """你是异动归因的反方，通过 verdict 工具给出结果。不要输出任何文字或分析过程，直接调用工具给出结果。给你一天的行情事实和归因者报出的"高置信度驱动"及其原文引文。
你的任务是找出这个驱动不足以解释当天涨跌的具体理由，只能用给你的事实和引文，不能编造：
幅度是否相称（引文里的事件能否解释这么大的涨跌）、时间是否对得上（事件是否发生在当日或前一晚）、板块是否同向同幅（那就是板块行情而非公司原因）、引文是否只是分析师观点或长期展望。
objection 一句中文指出具体不足（没有就留空），downgrade 只在理由具体且成立时为 true。如果无法调用工具，就只输出 {"objection":"...","downgrade":true|false}。"""

#: Judged, not matched: the attributor's reader did read the filings, so the answer may not say otherwise however it words it.
_UNREAD_RULE = {"forbid_claim": "申报的正文或内容没有被读取、只知道申报的类型和日期", "warning": ""}

#: Below this share of the stock's move, the sector's same-direction move is "the sector did it".
SECTOR_EXPLAINS_SHARE = 0.7


class _AttributionLoop(ToolLoop):
    usage_source_name = "agent_v2.move_attributor"
    finish_description = "报告涨跌原因并结束：只报已经拿到来源并能引用原文的原因，没有就返回空 reasons 并说明。"

    def __init__(self, llm: Any, limits: LoopLimits, *, facts: DayFacts, news: Callable[[str], list[dict[str, Any]]] | None, filing_events: Callable[[], ToolEnvelope] | None, recall: Callable[[str], list[Any]] | None) -> None:
        self.facts = facts
        self.news = news
        self.filing_events = filing_events
        self.recall = recall
        self.gathered = Gathered()
        self._refused_finish = False
        tools: list[Tool] = []
        unavailable: dict[str, str] = {}
        if news is not None:
            tools.append(Tool("news", "搜当日新闻；query 用英文检索词，含公司名和日期。", _schema({"query": {"type": "string"}}, ["query"]), self._news))
        else:
            unavailable["news"] = "用户未授权网页搜索，news 不可用；请用 filing_events 或 memory，或直接 finish。"
        if filing_events is not None:
            tools.append(Tool("filing_events", "读当日或前几天的 SEC 申报，返回有日期和原文引文的事件。", _schema({}), self._filing_events))
        else:
            unavailable["filing_events"] = "申报阅读不可用。"
        if recall is not None:
            tools.append(Tool("memory", "查盯盘记忆里这只股票的异动记录；query 为关键词。", _schema({"query": {"type": "string"}}), self._memory))
        else:
            unavailable["memory"] = "盯盘记忆不可用。"
        super().__init__(llm, limits, tools=tools, finish_parameters=_FINISH_SCHEMA, unavailable=unavailable)

    def refuse_finish(self, action: dict[str, Any]) -> str | None:
        # With the user's web consent the news is looked at once before the
        # loop settles for memory alone; asked once, never twice.
        if self.news is not None and self.gathered.news_calls == 0 and not self._refused_finish:
            self._refused_finish = True
            return "网页已授权但还没有搜过新闻；请先 news 搜索一次当日报道，再 finish。"
        return None

    def _news(self, arguments: dict[str, Any]) -> str:
        query = str(arguments.get("query") or f"{self.facts.ticker} stock {self.facts.date}")
        self.gathered.news_calls += 1
        try:
            rows = list(self.news(query) or [])
        except Exception as exc:  # noqa: BLE001 — a failed search is an observation, not a crash
            return f"新闻搜索失败：{type(exc).__name__}。"
        kept = [row for row in rows if _mentions(row, self.facts.ticker)]
        for row in kept:
            url = str(row.get("url") or "").strip()
            if url:
                self.gathered.news[url] = row
        listing = "\n".join(f"- {row.get('published_date') or row.get('published_at') or '日期未知'} | {row.get('title') or ''} | {row.get('url')}\n  {_WS.sub(' ', str(row.get('content') or ''))[:600]}" for row in kept[:6])
        return f"新闻搜索结果（已按是否提及 {self.facts.ticker} 过滤，{len(kept)}/{len(rows)} 条）：\n{listing or '（无）'}"

    def _filing_events(self, arguments: dict[str, Any]) -> str:
        self.gathered.reader_calls += 1
        envelope = self.filing_events()
        self.gathered.reader_runs.append({"filings": envelope.metrics.get("filings"), "sections_read": envelope.metrics.get("sections_read"), "events": envelope.metrics.get("events"), "rounds": envelope.metrics.get("rounds"), "stop_reason": envelope.metrics.get("stop_reason"), "elapsed_ms": envelope.metrics.get("elapsed_ms"), "trace": list(envelope.metadata.get("trace") or [])})
        lines = []
        for item in envelope.evidence:
            if item.metadata.get("evidence_scope") == "filing_event":
                self.gathered.filing_events[item.id] = item
                lines.append(f"- id={item.id} | {item.metadata.get('date')} | {item.claim}")
            elif item.metadata.get("citation_kind") == "limitations":
                self.gathered.filing_notes.append(item)
                lines.append(f"- （说明）{item.claim}")
        return "申报阅读者的结果：\n" + ("\n".join(lines) or "（无）")

    def _memory(self, arguments: dict[str, Any]) -> str:
        self.gathered.memory_calls += 1
        try:
            rows = list(self.recall(str(arguments.get("query") or f"{self.facts.ticker} 异动")) or [])
        except Exception as exc:  # noqa: BLE001
            return f"盯盘记忆查询失败：{type(exc).__name__}。"
        lines = []
        for row in rows[:6]:
            day = str(getattr(row, "date", "") or "")[:10]
            meta = dict(getattr(row, "metadata", None) or {})
            entry = {"date": day, "flags": str(getattr(row, "flags", "") or ""), "doc": _WS.sub(" ", str(getattr(row, "doc", "") or "")), "confidence": str(meta.get("confidence") or ""), "written_at": str(meta.get("written_at") or "")}
            self.gathered.memory[day] = entry
            stamp = f"（归因记录，{entry['confidence'] or '未知'}置信度，{entry['written_at'] or '日期未知'}写入）" if "retro_attribution" in entry["flags"] else ""
            lines.append(f"- {day} | {entry['flags'] or '无标志'}{stamp} | {entry['doc'][:300]}")
        return "盯盘记忆：\n" + ("\n".join(lines) or "（无）")


def _mentions(row: dict[str, Any], ticker: str) -> bool:
    text = f"{row.get('title') or ''} {row.get('content') or ''}".lower()
    return ticker.lower() in text


def _verify_reasons(reasons: list[dict[str, Any]], gathered: Gathered) -> tuple[list[dict[str, Any]], int]:
    kept: list[dict[str, Any]] = []
    dropped = 0
    for row in reasons:
        if not isinstance(row, dict) or not str(row.get("text") or "").strip():
            dropped += 1
            continue
        source = row.get("source") if isinstance(row.get("source"), dict) else {}
        kind = str(source.get("kind") or "")
        quote = str(row.get("quote") or "")
        level = str(row.get("confidence") or "中")
        located: str | None = None
        url = ""
        if kind == "news":
            item = gathered.news.get(str(source.get("url") or "").strip())
            if item is not None:
                url = str(item.get("url") or "")
                located = locate_quote(quote, f"{item.get('title') or ''}. {item.get('content') or ''}")
        elif kind == "filing":
            item = gathered.filing_events.get(str(source.get("id") or ""))
            if item is not None:
                url = item.source_url
                located = locate_quote(quote, f"{item.claim} {item.metadata.get('quote') or ''}")
        elif kind == "memory":
            entry = gathered.memory.get(str(source.get("date") or "")[:10])
            if entry is not None:
                located = locate_quote(quote, entry["doc"]) or entry["doc"][:200]
                if level == "高":
                    level = "中"
        if located is None or level not in {"高", "中", "低"}:
            dropped += 1
            continue
        kept.append({"text": str(row["text"]).strip(), "confidence": level, "kind": kind, "url": url, "quote": located, "source_id": str(source.get("id") or source.get("date") or url)})
    return kept, dropped


class MoveAttributor:
    """Explain one day's move with bounded, verifiable evidence; remember the result."""

    def __init__(
        self,
        llm: Any,
        *,
        price_source_factory: Callable[[], Any],
        news: Callable[[str, str], list[dict[str, Any]]] | None = None,
        filing_reader: FilingReader | None = None,
        memory_recall: Callable[[str, str, int], list[Any]] | None = None,
        memory_remember: Callable[[DayFacts, list[dict[str, Any]]], str] | None = None,
        sector_for: Callable[[str], str] | None = None,
        max_rounds: int = 5,
        max_seconds: float = 120.0,
    ) -> None:
        self.llm = llm
        self.price_source_factory = price_source_factory
        self.news = news
        self.filing_reader = filing_reader
        self.memory_recall = memory_recall
        self.memory_remember = memory_remember
        self.sector_for = sector_for
        self.max_rounds = max(1, max_rounds)
        self.max_seconds = max(5.0, max_seconds)

    def run(self, ticker: str, context: ExecutionContext, *, day: str = "", today: date | None = None, now: datetime | None = None, capability: str = "market.attribute_move") -> ToolEnvelope:
        """Explain one day's move.  ``day`` empty means the latest bar (today's, when the market is open).

        ``now`` marks a bar dated today as intraday while the session runs;
        ``capability`` names the envelope (``market.explain_move`` when the
        attributor answers a question about today).
        """

        current = today or (now.date() if now is not None else date.today())
        source = self.price_source_factory()
        start = (current - timedelta(days=430)).isoformat()
        from v2.agent_v2.adapters.market import _usable as usable_bars

        prices = usable_bars(source.get_prices(ticker, start, current.isoformat()))
        target = (day or current.isoformat())[:10]
        sector = self.sector_for(ticker) if self.sector_for else ""
        sector_prices = usable_bars(source.get_prices(sector, start, current.isoformat())) if sector and sector != ticker else []
        facts = day_facts(ticker, target, prices, sector, sector_prices, now=now)
        if facts is None:
            return ToolEnvelope(capability, ResultStatus.FAILED, subject=ticker, errors=["no price history for that date"])
        limits = limits_for(context, max_rounds=self.max_rounds, max_seconds=self.max_seconds)
        allow_news = bool(getattr(context, "allow_web", False)) and self.news is not None
        loop = _AttributionLoop(
            self.llm,
            limits,
            facts=facts,
            news=(lambda query: self.news(query, facts.date)) if allow_news else None,
            filing_events=(lambda: self.filing_reader.run(ticker, context, around=facts.date, today=current)) if self.filing_reader is not None else None,
            recall=(lambda query: self.memory_recall(ticker, query, max(30, (current - date.fromisoformat(facts.date)).days + 30))) if self.memory_recall is not None else None,
        )
        task = (
            f"股票：{ticker}\n日期：{facts.date}{'（今日，盘中，价格和成交量都不是最终值）' if facts.is_intraday else ''}\n当日涨跌：{_pct(facts.change)}，{'盘中价' if facts.is_intraday else '收盘'} {facts.close:.2f} 美元\n"
            f"成交量：{facts.volume} 股，为 30 日均量的 {facts.volume_ratio:.2f} 倍\n" if facts.volume_ratio is not None else f"股票：{ticker}\n日期：{facts.date}\n当日涨跌：{_pct(facts.change)}，收盘 {facts.close:.2f} 美元\n"
        )
        if facts.sector_return_1d is not None:
            task += f"行业基准 {facts.sector_etf} 当日 {_pct(facts.sector_return_1d)}，相对回报 {_pct(facts.relative_1d)}\n"
        task += f"可用动作：{'news、' if allow_news else ''}{'filing_events、' if self.filing_reader is not None else ''}{'memory、' if self.memory_recall is not None else ''}finish"
        # A filing in the three days up to the move is read before the model
        # chooses anything: an earnings 8-K the evening before is the
        # catalyst more often than not, and it must not depend on the model
        # deciding to look.
        preamble: list[dict[str, str]] = []
        if self.filing_reader is not None and self._filing_just_before(ticker, facts.date):
            loop.handle({"action": "filing_events"}, preamble)
            if preamble:
                preamble[-1]["content"] = "当日或前 3 天内有申报，已先读取。" + preamble[-1]["content"]
        outcome = loop.run(_SYSTEM, task, finish_prompt=_FINISH_NOW, preamble=preamble)
        raw = [row for row in (outcome.final.get("reasons") or []) if isinstance(row, dict)] if outcome.finished else []
        reasons, dropped = _verify_reasons(raw, loop.gathered)
        note = str(outcome.final.get("note") or "") if outcome.finished else outcome.note
        if dropped:
            note = (note + "；" if note else "") + f"{dropped} 条原因没有可核对的来源，已丢弃"
        challenge = self._challenge(facts, reasons, context)
        if challenge.get("objection"):
            note = (note + "；" if note else "") + f"反方意见：{challenge['objection']}" + ("（已降为中置信度）" if challenge.get("downgraded") else "（未采纳）")
        if challenge.get("called"):
            outcome.trace.append({"round": outcome.rounds + 1, "action": "challenge", "detail": (challenge.get("objection") or "无异议")[:90], "ms": int(challenge.get("ms") or 0)})
        next_steps = [str(step) for step in (outcome.final.get("next_steps") or []) if step] if outcome.finished else []
        remembered = ""
        memory_note = ""
        memory_conflict = False
        # An open session's facts are not final: nothing is written to memory yet.
        if self.memory_remember is not None and outcome.finished and not facts.is_intraday:
            try:
                decision = self.memory_remember(facts, reasons)
            except Exception:  # noqa: BLE001 — memory is optional infrastructure
                decision = ""
            if isinstance(decision, dict):
                remembered = str(decision.get("id") or "") if decision.get("written", True) else ""
                memory_note = str(decision.get("note") or "")
                memory_conflict = bool(decision.get("conflict"))
            else:
                remembered = str(decision or "")
        if memory_note:
            note = (note + "；" if note else "") + memory_note
        envelope = self._envelope(facts, reasons, loop.gathered, note=note, next_steps=next_steps, metrics={"rounds": outcome.rounds, "llm_calls": outcome.calls, "elapsed_ms": outcome.elapsed_ms, "stop_reason": outcome.stop_reason, "seconds_allowed": round(outcome.seconds_allowed, 1), "news_calls": loop.gathered.news_calls, "reader_calls": loop.gathered.reader_calls, "memory_calls": loop.gathered.memory_calls, "remembered_as": remembered}, allow_news=allow_news)
        envelope.capability = capability
        # What the sub-agent did, for the surfaces to show: one line per
        # round, plus the reader's own rounds when it was called.
        trace = list(outcome.trace)
        if preamble:
            trace.insert(0, {"round": 0, "action": "filing_events", "detail": "当日或前 3 天内有申报，先读", "ms": 0})
        envelope.metadata["trace"] = trace
        envelope.metadata["agent"] = {
            "name": "move_attributor",
            "label": "异动归因",
            "subject": f"{ticker} {facts.date}",
            "rounds": outcome.rounds,
            "llm_calls": outcome.calls,
            "elapsed_ms": outcome.elapsed_ms,
            "seconds_allowed": round(outcome.seconds_allowed, 1),
            "stop_reason": outcome.stop_reason,
            "calls": {"news": loop.gathered.news_calls, "filing_events": loop.gathered.reader_calls, "memory": loop.gathered.memory_calls},
            "reader_runs": loop.gathered.reader_runs,
            "intraday": facts.is_intraday,
            "yield": {"kept": len(reasons), "dropped": dropped, "confirmed": sum(1 for reason in reasons if reason["confidence"] == "高")},
            "memory": {"written": bool(remembered), "conflict": memory_conflict, "note": memory_note},
            "challenge": {key: value for key, value in challenge.items() if key != "ms"},
        }
        # A filing on the day whose news the attributor never looked at: ask the
        # run board for the news checker on that day, within the run's cap, so
        # the filing's account and the press's can be set side by side.
        board = getattr(context, "board", None)
        if allow_news and loop.gathered.filing_events and loop.gathered.news_calls == 0 and board is not None:
            window = max(7, (current - date.fromisoformat(facts.date)).days + 7)
            accepted = board.request("web.research", {"query": f"{ticker} stock news {facts.date}", "topic": "company_event", "ticker": ticker, "recency_days": min(3650, window), "min_searches": 1}, purpose=f"the news of {facts.date} beside the {ticker} filing the attributor read", requested_by="move_attributor")
            envelope.metadata["agent"]["follow_up"] = {"news_around": facts.date, "accepted": accepted}
            if accepted:
                envelope.metadata["agent"].setdefault("notes", []).append(f"读到 {facts.date} 前后的申报但没搜当日新闻，已请新闻核查者补查")
        return envelope

    def _challenge(self, facts: DayFacts, reasons: list[dict[str, Any]], context: ExecutionContext) -> dict[str, Any]:
        """Test a confirmed driver before it is reported as such.

        First the arithmetic nobody argues with: when the sector moved the
        same way and accounts for most of the move, the company-specific
        driver is at best a contributor.  Then, with time to spare, one
        adversarial model call looks for a concrete objection in the same
        sources; only a specific objection downgrades, and it is reported
        either way.
        """

        confirmed = [reason for reason in reasons if reason["confidence"] == "高"]
        if not confirmed:
            return {"called": False}
        change, sector = facts.change, facts.sector_return_1d
        if change and sector is not None and change * sector > 0 and abs(sector) >= SECTOR_EXPLAINS_SHARE * abs(change):
            objection = f"当日行业基准 {facts.sector_etf} 同向 {_pct(sector)}，占 {facts.ticker} {_pct(change)} 的大部分，公司特定原因未必是主因"
            for reason in confirmed:
                reason["confidence"] = "中"
            return {"called": False, "source": "sector", "objection": objection, "downgraded": True}
        if self.llm is None or (context.remaining_seconds() < 15 and getattr(context, "deadline", None) is not None):
            return {"called": False}
        from v2.usage_context import usage_source

        started = time.monotonic()
        brief = "\n".join(f"- {reason['text']}（{reason['kind']}：“{reason['quote'][:200]}”）" for reason in confirmed[:2])
        facts_text = f"{facts.ticker} {facts.date} {_pct(facts.change)}，收盘 {facts.close:.2f}" + (f"，行业基准 {facts.sector_etf} {_pct(facts.sector_return_1d)}" if facts.sector_return_1d is not None else "") + (f"，量比 {facts.volume_ratio:.2f}" if facts.volume_ratio is not None else "")
        try:
            with usage_source("agent_v2.challenger"):
                verdict = structured_call(self.llm, _CHALLENGE, f"行情事实：{facts_text}\n高置信度驱动：\n{brief}", CHALLENGE_TOOL)
            objection = str(verdict.get("objection") or "").strip()[:200]
            downgrade = bool(verdict.get("downgrade")) and bool(objection)
        except Exception as exc:  # noqa: BLE001 — a failed challenge changes nothing
            return {"called": True, "source": "model", "error": type(exc).__name__, "ms": int((time.monotonic() - started) * 1000)}
        if downgrade:
            for reason in confirmed:
                reason["confidence"] = "中"
        return {"called": True, "source": "model", "objection": objection, "downgraded": downgrade, "ms": int((time.monotonic() - started) * 1000)}

    def _filing_just_before(self, ticker: str, day: str) -> bool:
        """Whether EDGAR lists a filing dated within the three days up to ``day``."""

        source = getattr(self.filing_reader, "source", None)
        if source is None:
            return False
        anchor = date.fromisoformat(day[:10])
        try:
            refs = list(source.list_filings(ticker, (anchor - timedelta(days=3)).isoformat(), anchor.isoformat()) or [])
        except (OSError, ValueError, RuntimeError):  # EDGAR down or a bad row: the loop can still ask later
            return False
        return bool(refs)

    def _envelope(self, facts: DayFacts, reasons: list[dict[str, Any]], gathered: Gathered, *, note: str, next_steps: list[str], metrics: dict[str, Any], allow_news: bool) -> ToolEnvelope:
        ticker, day = facts.ticker, facts.date
        evidence: list[EvidenceItem] = []

        def item(kind: str, claim: str, **extra: Any) -> EvidenceItem:
            digest = hashlib.sha1(f"attribute|{kind}|{ticker}|{day}|{claim}".encode("utf-8")).hexdigest()[:16]
            metadata = {"evidence_scope": kind, **extra.pop("metadata", {})}
            return EvidenceItem(id=f"evidence-attribute-{kind}-{digest}", entity=ticker, claim=claim, as_of=day, source_id=extra.pop("source_id", "market_data"), source_title=extra.pop("source_title", "Daily OHLCV market data"), metadata=metadata, **extra)

        from v2.agent_v2.adapters.market import _INTRADAY_PRICE_RULE, _INTRADAY_VOLUME_RULE

        session = {"is_intraday": facts.is_intraday, "market_session": "REGULAR" if facts.is_intraday else "CLOSED", "volume_is_final": not facts.is_intraday}
        if facts.is_intraday:
            price_claim = f"{ticker} 截至 {facts.observed_at_label} 盘中报 {facts.close:.2f} 美元，相对前一交易日收盘价 {_pct(facts.change)}。"
            evidence.append(item("price", price_claim, metric="price_change_pct", value=facts.change, metadata={**session, "constraints": [_INTRADAY_PRICE_RULE]}))
        else:
            price_claim = f"{ticker} 在 {day} 收于 {facts.close:.2f} 美元，较前一交易日 {_pct(facts.change)}。"
            evidence.append(item("price", price_claim, metric="price_change_pct", value=facts.change, metadata=session))
        volume = None
        if facts.volume_ratio is not None:
            if facts.is_intraday:
                volume = item("volume", f"{ticker} 截至查询时的盘中累计成交量为 {int(facts.volume):,} 股，相当于 30 日完整交易日均量 {facts.average_volume_30d:,.0f} 股的 {facts.volume_ratio:.2f} 倍；当日未收盘，不能据此判定是否放量或缩量。", metric="volume_ratio", value=facts.volume_ratio, metadata={**session, "constraints": [_INTRADAY_VOLUME_RULE]})
            else:
                volume = item("volume", f"{ticker} {day} 成交量为 {int(facts.volume):,} 股，30 日均量为 {facts.average_volume_30d:,.0f} 股，量比 {facts.volume_ratio:.2f} 倍。", metric="volume_ratio", value=facts.volume_ratio, metadata=session)
            evidence.append(volume)
        benchmark = None
        if facts.sector_return_1d is not None:
            prefix = f"{ticker} 截至同一查询时点的盘中" if facts.is_intraday else f"{ticker} {day} "
            benchmark = item("benchmark", f"{prefix}行业基准 {facts.sector_etf} 单日回报为 {_pct(facts.sector_return_1d)}，{ticker} 相对回报为 {_pct(facts.relative_1d)}。", metadata={"benchmark": facts.sector_etf, **session})
            evidence.append(benchmark)
        high = 0
        reason_items: list[EvidenceItem] = []
        for reason in reasons:
            confirmed = reason["confidence"] == "高"
            high += int(confirmed)
            role = "driver" if confirmed else "candidate"
            qualifier = "高置信度归因" if confirmed else f"{reason['confidence']}置信度候选解释"
            source_label = {"news": "新闻", "filing": "申报", "memory": "盯盘记忆"}.get(reason["kind"], "来源")
            claim = f"{ticker} {day} {qualifier}：{reason['text']}（{source_label}：“{reason['quote'][:160]}”）。"
            metadata = {"claim_role": "confirmed_driver" if confirmed else "candidate_driver", "causal_confidence": reason["confidence"], "driver_text": lead_text(reason["text"]), "note": "", "source_kind": reason["kind"], "quote": reason["quote"], "supporting_sources": [{"title": "", "url": reason["url"]}] if reason["url"] else []}
            if not confirmed:
                metadata["constraints"] = [_CANDIDATE_RULE]
            reason_items.append(item(role, claim, source_id={"news": "web_news", "filing": "sec_edgar", "memory": "anomaly_memory"}[reason["kind"]], source_title=source_label, source_url=reason["url"], confidence={"高": 0.9, "中": 0.6, "低": 0.3}[reason["confidence"]], metadata=metadata))
        evidence.extend(reason_items)
        for filing_item in gathered.filing_events.values():
            if filing_item.id not in {existing.id for existing in evidence}:
                evidence.append(filing_item)
        assessment = item("attribution", f"{ticker} {day} 异动归因中有 {high} 个高置信度直接驱动，{len(reasons) - high} 个候选解释。" + ("" if high else " 现有证据不足以确认具体触发原因。"), metric="confirmed_driver_count", value=high, source_id="move_attribution", source_title="Move attribution", metadata={"claim_role": "attribution_assessment", "verified": True})
        evidence.append(assessment)
        limitations = [note] if note else []
        if high == 0:
            limitations.append("没有高置信度的同日催化剂证据，具体触发原因尚未确认。")
        if not allow_news:
            limitations.append("用户未授权网页搜索，归因未使用新闻。")
        answer_constraints: list[dict[str, Any]] = [dict(_COUNT_LEAK_RULE)]
        if gathered.filing_events:
            # The reader located events in the filings: an answer that says
            # they were not read contradicts its own evidence.
            answer_constraints.append({**_UNREAD_RULE, "warning": f"{day} 附近的申报已由申报阅读者读取并摘出事件，回答却称申报内容未读取；请引用那些申报事件"})
        if high == 0:
            answer_constraints.append({"max_cited": {"metadata": {"claim_role": "candidate_driver"}, "max": 1, "warning": "未确认直接驱动时展示了过多弱候选线索"}})
        read_events = [item for item in gathered.filing_events.values()]
        narrative = self._narrative(facts, evidence[0], volume, benchmark, reason_items, assessment, read_events)
        compact = self._compact_narrative(facts, evidence[0], benchmark, reason_items, assessment, read_events)
        return ToolEnvelope(
            "market.attribute_move",
            ResultStatus.COMPLETED if reasons else ResultStatus.PARTIAL_DATA,
            subject=ticker,
            as_of=day,
            summary=price_claim,
            metrics={"is_intraday": facts.is_intraday, "price": facts.close, "price_change_pct": facts.change, "volume_ratio": facts.volume_ratio, "sector_etf": facts.sector_etf, "sector_return_1d": facts.sector_return_1d, "relative_1d": facts.relative_1d, "confirmed_driver_count": high, "candidate_driver_count": len(reasons) - high, **metrics},
            findings=[{"claim": reason["text"], "causal_confidence": reason["confidence"], "confirmed": reason["confidence"] == "高", "evidence_ids": [reason_item.id]} for reason, reason_item in zip(reasons, reason_items)],
            evidence=evidence,
            limitations=limitations,
            metadata={"next_steps": next_steps, "require_cited_numbers": True, "answer_constraints": answer_constraints, "narrative": narrative, "narrative_compact": compact, "date": day, "is_intraday": facts.is_intraday},
        )

    @staticmethod
    def _narrative(facts: DayFacts, price: EvidenceItem, volume: EvidenceItem | None, benchmark: EvidenceItem | None, reasons: list[EvidenceItem], assessment: EvidenceItem, read_events: list[EvidenceItem] | None = None) -> str:
        direction = "上涨" if (facts.change or 0) > 0 else "下跌" if (facts.change or 0) < 0 else "基本持平"
        first = f"{price.claim.rstrip('。')}[{price.id}]。"
        if volume is not None:
            first += f"{volume.claim.rstrip('。')}[{volume.id}]。"
        if benchmark is not None:
            first += f"{benchmark.claim.rstrip('。')}[{benchmark.id}]。"
        confirmed = [item for item in reasons if item.metadata.get("claim_role") == "confirmed_driver"]
        candidates = [item for item in reasons if item.metadata.get("claim_role") == "candidate_driver"]
        if confirmed:
            second = "能直接支持的高置信度驱动：" + "；".join(f"{item.metadata['driver_text']}[{item.id}]" for item in confirmed[:2]) + "。"
        else:
            second = f"“为什么{direction}”目前还不能下定论：暂未找到可核实的同日催化剂，具体触发原因尚未确认[{assessment.id}]。"
        if candidates:
            best = max(candidates, key=lambda item: float(item.confidence or 0))
            second += f"最相关的一条候选线索是“{best.metadata['driver_text']}”，只能作为排查方向[{best.id}]。"
        # What the filing reader found is stated even when no reason was
        # built on it; otherwise the answer says the filings were unread.
        second += _read_events_sentence(read_events or [])
        if benchmark is not None and facts.relative_1d is not None:
            relation = "跑赢" if facts.relative_1d > 0 else "跑输"
            when = "截至查询时盘中" if facts.is_intraday else "当天"
            third = f"从盘面看，{when}{relation}行业基准 {facts.sector_etf} 约 {abs(facts.relative_1d):.2%}[{benchmark.id}]。"
            if facts.is_intraday:
                third += "当日未收盘，价格、成交量和归因都以收盘后为准。"
        else:
            third = "当日未收盘，价格、成交量和归因都以收盘后为准。" if facts.is_intraday else ""
        return "\n\n".join(part for part in (first, second, third) if part)

    @staticmethod
    def _compact_narrative(facts: DayFacts, price: EvidenceItem, benchmark: EvidenceItem | None, reasons: list[EvidenceItem], assessment: EvidenceItem, read_events: list[EvidenceItem] | None = None) -> str:
        """The same day in one short paragraph, for a phone screen.

        Date and move, how it compared with the sector, and the single best
        lead; the full narrative keeps the volume figure and the second lead.
        """

        sentence = f"{facts.date} {facts.ticker} {'盘中 ' if facts.is_intraday else ''}{_pct(facts.change)}[{price.id}]"
        if benchmark is not None and facts.relative_1d is not None:
            relation = "跑赢" if facts.relative_1d > 0 else "跑输"
            sentence += f"，{relation} {facts.sector_etf} 约 {abs(facts.relative_1d):.2%}[{benchmark.id}]"
        sentence += "。"
        confirmed = [item for item in reasons if item.metadata.get("claim_role") == "confirmed_driver"]
        candidates = [item for item in reasons if item.metadata.get("claim_role") == "candidate_driver"]
        if confirmed:
            best = confirmed[0]
            sentence += f"驱动：{best.metadata['driver_text']}[{best.id}]。"
        elif candidates:
            best = max(candidates, key=lambda item: float(item.confidence or 0))
            sentence += f"催化剂未确认；最相关线索：{best.metadata['driver_text']}[{best.id}]。"
        else:
            sentence += f"暂未找到可核实的同日催化剂[{assessment.id}]。"
        if not confirmed:
            sentence += _read_events_sentence((read_events or [])[:1])
        return sentence


def mark_read_filings(results: list[ToolEnvelope]) -> int:
    """Stamp ``filings.recent`` records whose filing an attributor's reader read, so the draft cannot call them unread.

    The stretch chain lists a day's filings (form and date only) and, separately,
    the attributor reads them and reports dated events.  Both reach the
    synthesizer; without this the draft picks the bare record and says the
    text was not read.  The record's claim gains the read events' ids.
    """

    read: dict[str, list[EvidenceItem]] = {}
    by_form_date: dict[tuple[str, str, str], list[EvidenceItem]] = {}
    for result in results:
        if result.capability not in {"market.attribute_move", "market.explain_move", "filings.read_events"}:
            continue
        for item in result.evidence:
            if item.metadata.get("evidence_scope") != "filing_event":
                continue
            url = (item.source_url or "").rstrip("/")
            if url:
                read.setdefault(url, []).append(item)
            key = (item.entity.upper(), str(item.metadata.get("form") or ""), str(item.metadata.get("filing_date") or ""))
            by_form_date.setdefault(key, []).append(item)
    if not read and not by_form_date:
        return 0
    marked = 0
    for result in results:
        if result.capability != "filings.recent":
            continue
        for item in result.evidence:
            if item.metadata.get("evidence_scope") != "filing" or item.metadata.get("read_by"):
                continue
            url = (item.source_url or "").rstrip("/")
            # Two same-day filings of one form are told apart by URL; form and date only stand in when the record has no link.
            events = read.get(url) if url else by_form_date.get((item.entity.upper(), str(item.metadata.get("form") or ""), str(item.metadata.get("date") or "")))
            if not events:
                continue
            ids = list(dict.fromkeys(event.id for event in events))[:3]
            # EvidenceItem is frozen; the ledger holds this same object, so the stamp reaches the synthesizer and the verifier alike.
            object.__setattr__(item, "claim", item.claim.rstrip("。") + "；申报阅读者已读取正文，读到的事件见 " + "、".join(f"[{event_id}]" for event_id in ids) + "。")
            item.metadata["read_by"] = ids
            marked += 1
    return marked


def lead_text(text: str) -> str:
    """A driver or lead as one sentence fragment: no inner full stops, no trailing punctuation.

    The verifier splits on 。 and asks every figure-bearing sentence for a
    citation; a quoted lead with its own full stops and figures would leave
    half of itself uncited.
    """

    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    cleaned = cleaned.rstrip("。；;，,.！!？? ")
    return re.sub(r"[。；;]\s*", "；", cleaned)


def _read_events_sentence(events: list[EvidenceItem]) -> str:
    """"申报读到：…" for the events the filing reader located, each cited; empty when it read none."""

    parts = []
    for item in events[:2]:
        text = item.claim.split("：", 1)[1] if "：" in item.claim else item.claim
        text = text.split("（", 1)[0].strip().rstrip("。")
        if text:
            parts.append(f"{item.metadata.get('date') or item.as_of} {text}[{item.id}]")
    return ("申报读到：" + "；".join(parts) + "。") if parts else ""


def _pct(value: float | None) -> str:
    return "数据不足" if value is None else f"{float(value):+.2%}"


def _default_news(query: str, day: str) -> list[dict[str, Any]]:
    import os

    from v2.data.metered import TavilyClient

    client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    recent = (date.today() - date.fromisoformat(day)).days <= 7
    response = client.search(query=query, max_results=6, topic="news" if recent else "general", days=7 if recent else 400, search_depth="basic")
    return list(response.get("results", []))


def _default_recall(ticker: str, query: str, lookback_days: int) -> list[Any]:
    from v2.memory import AnomalyMemory

    return AnomalyMemory().recall(ticker, query, lookback_days=lookback_days, n_results=6)


_RANK = {"高": 3, "中": 2, "低": 1}


_REASON_NOISE = re.compile(r"[\s，,。．.、；;：:（）()\[\]“”\"'’‘\-—–]|公司|市场|股价|当日|当天|引发|导致|因此|其|的|了|与|和|及|并|对|将|仍|在|被|为", re.I)


def clip(text: str, limit: int) -> str:
    """Cut at a punctuation mark near the limit and never leave a bracket open; ``…`` marks the cut."""

    flat = " ".join(str(text or "").split())
    if len(flat) <= limit:
        return flat
    cut = flat[: max(1, limit - 1)]
    tail = max(0, len(cut) - 25)
    marks = [cut.rfind(mark, tail) for mark in "，。；、）)"]
    best = max(marks)
    if best > 0:
        cut = cut[: best + 1] if cut[best] in "）)" else cut[:best]
    while cut.count("（") > cut.count("）") or cut.count("(") > cut.count(")"):
        opened = max(cut.rfind("（"), cut.rfind("("))
        if opened <= 0:
            break
        cut = cut[:opened].rstrip("，、；：,; ")
    return cut.rstrip("，、；：") + "…"


def same_reason(first: str, second: str, *, threshold: float = 0.42) -> bool:
    """Whether two top reasons describe the same event, allowing for rewording.

    Both are reduced to their content characters (particles, punctuation
    and boilerplate removed) and compared as sequences; a shared core
    such as 智能手机版税收入下滑 survives any amount of surrounding prose.
    """

    from difflib import SequenceMatcher

    a, b = _REASON_NOISE.sub("", first or "").lower(), _REASON_NOISE.sub("", second or "").lower()
    if not a or not b:
        return False
    if a in b or b in a:
        return True
    matcher = SequenceMatcher(None, a, b, autojunk=False)
    longest = matcher.find_longest_match(0, len(a), 0, len(b)).size
    return matcher.ratio() >= threshold or longest >= max(6, int(0.35 * min(len(a), len(b))))


def govern_memory(existing: Any, reasons: list[dict[str, Any]], *, today: str) -> dict[str, Any]:
    """Decide whether a fresh attribution replaces the stored one for that day.

    Rules: a stored record whose best confidence is higher than the new
    result's is kept (a later run with only candidates does not erase a
    confirmed driver); otherwise the new result is written with a version
    number, and when its top reason differs from the stored one the change
    is recorded (``previous_reason``) so the drift is visible, not silent.
    """

    best = max((_RANK.get(str(reason.get("confidence")), 0) for reason in reasons), default=0)
    top = next((str(reason.get("text") or "") for reason in sorted(reasons, key=lambda reason: -_RANK.get(str(reason.get("confidence")), 0))), "")
    meta = dict(getattr(existing, "metadata", None) or {}) if existing is not None else {}
    stored_best = int(meta.get("confidence_rank") or 0)
    stored_top = str(meta.get("top_reason") or "")
    version = int(meta.get("version") or 0)
    differs = bool(top and stored_top and not same_reason(top, stored_top))
    if existing is not None and stored_best > best:
        return {"write": False, "note": f"记忆中已有更高置信度的归因（{meta.get('confidence') or '?'}，{meta.get('written_at') or '早先'}写入），本次结论未覆盖", "metadata": meta, "conflict": differs}
    label = next((key for key, value in _RANK.items() if value == best), "无")
    metadata = {"confidence": label, "confidence_rank": best, "top_reason": top[:200], "version": version + 1, "written_at": today}
    conflict = existing is not None and differs
    if conflict:
        metadata["previous_reason"] = stored_top[:200]
        metadata["previous_confidence"] = str(meta.get("confidence") or "")
    note = f"与上次归因不同（上次：{clip(stored_top, 60)}），已覆盖为本次结论" if conflict else ""
    return {"write": True, "note": note, "metadata": metadata, "conflict": conflict}


def _default_remember(facts: DayFacts, reasons: list[dict[str, Any]]) -> dict[str, Any]:
    from v2.memory import AnomalyMemory
    from v2.monitoring.models import Anomaly, NewsSource, ScoredReason

    memory = AnomalyMemory()
    doc_id = f"{facts.ticker}_{facts.date}_retro"
    decision = govern_memory(memory.get(doc_id), reasons, today=date.today().isoformat())
    if not decision["write"]:
        return {"id": doc_id, "written": False, "note": decision["note"], "conflict": decision["conflict"]}
    anomaly = Anomaly(
        ticker=facts.ticker,
        date=facts.date,
        price=facts.close,
        price_change_pct=facts.change or 0.0,
        volume_today=facts.volume,
        volume_avg_30d=facts.average_volume_30d or 0.0,
        volume_ratio=facts.volume_ratio or 1.0,
        high_52w=facts.high_52w,
        low_52w=facts.low_52w,
        flags=["retro_attribution"],
        reasons=[ScoredReason(text=reason["text"], confidence=reason["confidence"]) for reason in reasons],
        sources=[NewsSource(title=reason["kind"], url=reason["url"]) for reason in reasons if reason["url"]],
    )
    written = memory.remember(anomaly, doc_id=doc_id, metadata=decision["metadata"])
    return {"id": written, "written": True, "note": decision["note"], "conflict": decision["conflict"]}


def _default_price_source():
    from v2.data.price_source import default_price_source

    return default_price_source()


def _default_sector(ticker: str) -> str:
    from v2.universe import sector_etf_for

    return sector_etf_for(ticker)


def register_move_attributor(
    registry: CapabilityRegistry,
    llm: Any,
    *,
    price_source_factory: Callable[[], Any] = _default_price_source,
    news: Callable[[str, str], list[dict[str, Any]]] | None = _default_news,
    filing_source: FilingSource | None = None,
    memory_recall: Callable[[str, str, int], list[Any]] | None = _default_recall,
    memory_remember: Callable[[DayFacts, list[dict[str, Any]]], str] | None = _default_remember,
    sector_for: Callable[[str], str] | None = _default_sector,
    today_factory: Callable[[], date] = date.today,
    now_factory: Callable[[], datetime | None] = lambda: datetime.now(tz=timezone.utc),
    **limits: Any,
) -> None:
    reader = FilingReader(llm, filing_source or EdgarFilingSource()) if llm is not None else None
    attributor = MoveAttributor(llm, price_source_factory=price_source_factory, news=news, filing_reader=reader, memory_recall=memory_recall, memory_remember=memory_remember, sector_for=sector_for, **limits)

    def handler(arguments: dict[str, Any], context: ExecutionContext) -> ToolEnvelope:
        return attributor.run(str(arguments.get("ticker") or "").upper(), context, day=str(arguments.get("date") or ""), today=today_factory(), now=now_factory())

    registry.register("market.attribute_move", handler)

    def explain_today(arguments: dict[str, Any], context: ExecutionContext) -> ToolEnvelope:
        # "Why did it move today" is the same job on the latest bar: the
        # filing of the day is read first, the news searched with consent,
        # the memory consulted; the envelope keeps the capability's name so
        # routing, guidance and the eval see market.explain_move.
        return attributor.run(str(arguments.get("ticker") or "").upper(), context, day="", today=today_factory(), now=now_factory(), capability="market.explain_move")

    if llm is not None:
        registry.register("market.explain_move", explain_today)
