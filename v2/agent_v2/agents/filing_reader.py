"""The filing reader: a bounded sub-agent that reads SEC filings around a date.

It lives behind the capability interface like any pure function: parameters
in, an evidence envelope out.  Inside, a small loop lets the model decide
which filing section to read next, at most a few rounds, and every event it
reports must quote text it actually read.  It never talks to the user and
never decides whether it should run; the planner does that.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Callable, Protocol

from v2.agent_v2.agents.base import LoopLimits, Tool, ToolLoop, _schema, limits_for
from v2.agent_v2.execution import CapabilityRegistry, ExecutionContext
from v2.agent_v2.models import EvidenceItem, ResultStatus, ToolEnvelope


@dataclass
class FilingRef:
    ticker: str
    form: str
    filing_date: str
    accession: str
    url: str = ""
    raw: Any = field(default=None, repr=False)


@dataclass
class Section:
    id: str
    title: str
    chars: int


class FilingSource(Protocol):
    def list_filings(self, ticker: str, since: str, until: str, forms: tuple[str, ...] | None = None) -> list[FilingRef]: ...

    def outline(self, ref: FilingRef) -> list[Section]: ...

    def read(self, ref: FilingRef, section_id: str) -> str: ...


_ITEM_HEADER = re.compile(r"(?im)^\s*item\s+(\d{1,2}\.\d{2})\b[^\n]{0,90}")
_HEADING = re.compile(r"(?m)^\s*(?:[Ee][Xx][Hh][Ii][Bb][Ii][Tt]\s+\d+(?:\.\d+)?[^\n]{0,80}|[A-Z][A-Z0-9 &,'()/.-]{6,70})\s*$")
_WS = re.compile(r"\s+")


def sections_of(text: str, form: str, *, chunk: int = 3500, max_sections: int = 20) -> list[tuple[str, str, str]]:
    """Split a filing's text into (id, title, body) sections.

    8-K bodies are cut at their ``Item X.YY`` headers; other forms at
    upper-case heading lines; anything else into fixed-size parts so the
    reader can still page through it.  A long exhibit whose every table
    header looks like a heading is merged back into at most
    ``max_sections`` runs, so the reader chooses among twenty entries,
    not two hundred.
    """

    text = text or ""
    if not text.strip():
        return []
    matches = list(_ITEM_HEADER.finditer(text)) if form.upper().startswith("8-K") else []
    if len(matches) < 2:
        matches = [match for match in _HEADING.finditer(text) if len(match.group(0).strip()) >= 8]
    if len(matches) >= 2:
        parts: list[tuple[str, str, str]] = []
        preamble = text[: matches[0].start()].strip()
        if preamble:
            # The cover page before the first heading is a section too.
            parts.append(("s0", _WS.sub(" ", preamble.splitlines()[0]).strip()[:90] or "正文", preamble))
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            title = _WS.sub(" ", match.group(0)).strip()[:90]
            body = text[match.start():end].strip()
            if body:
                parts.append((f"s{index + 1}", title, body))
        if parts:
            return _merge_sections(parts, max_sections)
    # No usable headings: page through the text in parts, sized so that the
    # whole document still fits in max_sections parts.
    size = max(chunk, -(-len(text) // max_sections))
    return [(f"part-{index + 1}", f"第 {index + 1} 段", text[start:start + size]) for index, start in enumerate(range(0, len(text), size))]


def _merge_sections(parts: list[tuple[str, str, str]], max_sections: int) -> list[tuple[str, str, str]]:
    """Merge consecutive sections until there are at most ``max_sections``, each about the same size."""

    if len(parts) <= max_sections:
        return parts
    total = sum(len(body) for _, _, body in parts)
    target = max(1, total // max_sections)
    merged: list[tuple[str, str, str]] = []
    bucket: list[tuple[str, str, str]] = []
    size = 0
    for part in parts:
        bucket.append(part)
        size += len(part[2])
        if size >= target and len(merged) < max_sections - 1:
            merged.append(_join(bucket))
            bucket, size = [], 0
    if bucket:
        merged.append(_join(bucket))
    return merged


def _join(bucket: list[tuple[str, str, str]]) -> tuple[str, str, str]:
    first_id, first_title, _ = bucket[0]
    title = first_title if len(bucket) == 1 else f"{first_title} …（含后续 {len(bucket) - 1} 节）"
    return first_id, title[:90], "\n\n".join(body for _, _, body in bucket)


class EdgarFilingSource:
    """Filings and their text through the existing EDGAR client and edgartools objects."""

    def __init__(self, fetch: Callable[[str, str, str, str], list[Any]] | None = None) -> None:
        self._fetch = fetch
        self._texts: dict[str, str] = {}
        self._sections: dict[str, list[tuple[str, str, str]]] = {}

    def _rows(self, ticker: str, form: str, since: str, until: str) -> list[Any]:
        if self._fetch is not None:
            return list(self._fetch(ticker, form, since, until) or [])
        from v2.sec import client as sec_client

        return list(sec_client.get_recent_filings(ticker, form, since, until) or [])

    def list_filings(self, ticker: str, since: str, until: str, forms: tuple[str, ...] | None = None) -> list[FilingRef]:
        """The filings of the given forms (default: the current reports, 8-K or 6-K for a foreign issuer), newest first."""

        refs: list[FilingRef] = []
        wanted = tuple(str(form).upper() for form in (forms or ()) if str(form).strip()) or ("8-K", "6-K")
        for form in wanted:
            for raw in self._rows(ticker, form, since, until):
                accession = str(getattr(raw, "accession_number", None) or getattr(raw, "accession_no", None) or "").strip()
                filing_date = str(getattr(raw, "filing_date", "") or "")[:10]
                actual_form = str(getattr(raw, "form", form) or form)
                url = str(getattr(raw, "homepage_url", None) or getattr(raw, "url", None) or "")
                if not url:
                    cik = str(getattr(raw, "cik", "") or "").lstrip("0")
                    url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession.replace('-', '')}/" if cik and accession else ""
                if accession:
                    refs.append(FilingRef(ticker, actual_form, filing_date, accession, url, raw))
            if refs and forms is None:
                break  # a domestic filer has 8-Ks; only a foreign issuer needs the 6-K pass
        refs.sort(key=lambda ref: ref.filing_date, reverse=True)
        return refs

    def _text(self, ref: FilingRef) -> str:
        if ref.accession in self._texts:
            return self._texts[ref.accession]
        text = _document_text(ref.raw)
        # A 6-K (and many 8-Ks) is a cover page; the press release or
        # results live in the EX-99 exhibits.  Append them under their own
        # headings so the outline offers them as sections.
        if ref.form.upper().startswith("6-K") or len(text) < 1500:
            for title, body in exhibit_texts(ref.raw):
                text = f"{text}\n\n{title}\n{body}" if text.strip() else f"{title}\n{body}"
        self._texts[ref.accession] = text
        return text

    def outline(self, ref: FilingRef) -> list[Section]:
        if ref.accession not in self._sections:
            self._sections[ref.accession] = sections_of(self._text(ref), ref.form)
        return [Section(section_id, title, len(body)) for section_id, title, body in self._sections[ref.accession]]

    def read(self, ref: FilingRef, section_id: str) -> str:
        self.outline(ref)
        for candidate, _, body in self._sections.get(ref.accession, []):
            if candidate == section_id:
                return body
        return ""


def _document_text(raw: Any) -> str:
    """The primary document's text through whichever accessor the SDK object offers."""

    for attr in ("text", "markdown"):
        method = getattr(raw, attr, None)
        if callable(method):
            try:
                text = str(method() or "")
            except Exception:  # noqa: BLE001 — a document that fails to load reads as empty
                continue
            if text.strip():
                return text
    return ""


_EXHIBIT_TYPE = re.compile(r"^EX-99", re.IGNORECASE)


def exhibit_texts(raw: Any, *, limit: int = 3, max_chars: int = 40_000) -> list[tuple[str, str]]:
    """(heading, text) for the EX-99 exhibits of a filing, in document order.

    edgartools exposes ``filing.attachments``; each attachment names its
    ``document_type`` and can render its text.  Anything that fails to
    load is skipped, so a filing without readable exhibits reads as its
    cover page alone.
    """

    attachments = getattr(raw, "attachments", None)
    if attachments is None:
        return []
    try:
        rows = list(attachments)
    except TypeError:
        rows = list(getattr(attachments, "documents", None) or [])
    found: list[tuple[str, str]] = []
    for row in rows:
        kind = str(getattr(row, "document_type", None) or getattr(row, "type", None) or "")
        if not _EXHIBIT_TYPE.match(kind.strip()):
            continue
        body = ""
        for attr in ("text", "markdown", "download"):
            method = getattr(row, attr, None)
            if not callable(method):
                continue
            try:
                value = method()
            except Exception:  # noqa: BLE001 — one exhibit failing to load is not the filing failing
                continue
            if isinstance(value, bytes):
                value = value.decode("utf-8", "ignore")
            body = str(value or "")
            if "<" in body[:200] and ">" in body[:400]:
                body = re.sub(r"<[^>]+>", " ", body)
            if body.strip():
                break
        if not body.strip():
            continue
        description = str(getattr(row, "description", None) or "").strip()
        heading = f"EXHIBIT {kind.upper().replace('EX-', '')}" + (f" {description}" if description else "")
        found.append((heading, _WS.sub(" ", body).strip()[:max_chars]))
        if len(found) >= limit:
            break
    return found


_SYSTEM = """你是申报阅读者，不回答用户问题，每一轮调用一个工具。
任务：在给定的 SEC 申报里找出与指定日期附近股价下跌可能相关的、有明确日期的事件（财报数字、指引、高管变动、诉讼、发行、重大合同、监管事项等）。
工具：read 读章节（一次最多 3 节，给申报序号和章节 id）；finish 报告事件并结束。
优先读 EXHIBIT 99.1、Outlook / Guidance（指引）、Item 2.02（业绩）、Item 5.02（高管变动）、Item 8.01（其他事项）这类章节；封面页和 Item 9.01 通常没有内容。股价在业绩日下跌时，指引和展望往往比业绩本身更关键。
规则：finish 里每条事件的 quote 必须逐字来自你已经读过的章节文本，不能改写、不能翻译；没有相关事件就返回空的 events 并在 note 里说明；不要编造日期。轮次有限，读到足够内容就尽早结束。
如果无法调用工具，就只输出 JSON：{"action":"read","reads":[{"filing":1,"section":"..."}]} 或 {"action":"finish","events":[...],"note":"..."}。"""

_EVENT_SCHEMA = _schema({"date": {"type": "string", "description": "YYYY-MM-DD"}, "summary": {"type": "string", "description": "一句中文概括"}, "quote": {"type": "string", "description": "从已读章节里原样复制的一段原文，不超过 300 字符"}, "filing": {"type": "integer", "description": "申报序号"}, "section": {"type": "string", "description": "章节 id"}}, ["date", "summary", "quote", "filing", "section"])
_FINISH_SCHEMA = _schema({"events": {"type": "array", "items": _EVENT_SCHEMA}, "note": {"type": "string", "description": "一句话说明还缺什么或为什么结束"}}, ["events"])

_FINISH_NOW = "轮次已用完。现在只允许 finish：把已读章节里有明确日期、且能逐字引用的事件整理出来；没有就返回空的 events 并说明。"


class _ReadLoop(ToolLoop):
    """The filing reader's loop: one tool, ``read``, serves up to three sections a round."""

    usage_source_name = "agent_v2.filing_reader"
    finish_description = "把已读章节里有明确日期、且能逐字引用的事件整理出来并结束；没有就返回空的 events 并在 note 里说明。"

    def __init__(self, llm: Any, limits: LoopLimits, source: FilingSource, chosen: list[FilingRef], max_chars: int) -> None:
        self.source = source
        self.chosen = chosen
        self.max_chars = max_chars
        self.read: dict[tuple[int, str], str] = {}
        read = Tool(
            "read",
            "读申报章节，一次最多 3 节；每项给申报序号 filing 和章节 id section。",
            _schema({"reads": {"type": "array", "maxItems": 3, "items": _schema({"filing": {"type": "integer"}, "section": {"type": "string"}}, ["filing", "section"])}}, ["reads"]),
            self._read,
        )
        super().__init__(llm, limits, tools=[read], finish_parameters=_FINISH_SCHEMA)

    def _read(self, arguments: dict[str, Any]) -> str:
        requests = arguments.get("reads")
        if not isinstance(requests, list):
            requests = [arguments]
        parts: list[str] = []
        for request in requests[:3]:
            try:
                index = int(request.get("filing"))
                section_id = str(request.get("section") or "")
            except (AttributeError, TypeError, ValueError):
                continue
            if not 1 <= index <= len(self.chosen):
                continue
            text = self.source.read(self.chosen[index - 1], section_id)[: self.max_chars]
            self.read[(index, section_id)] = text
            parts.append(f"申报 {index} 章节 {section_id} 的文本：\n{text or '（空）'}")
        if not parts:
            raise ValueError("read 需要 reads 列表，每项含 filing 序号和 section id。")
        return "\n\n".join(parts)


class FilingReader:
    """Read the filings around a date and report dated, quoted events."""

    def __init__(self, llm: Any, source: FilingSource, *, max_rounds: int = 6, max_seconds: float = 90.0, max_filings: int = 2, max_chars: int = 6000) -> None:
        self.llm = llm
        self.source = source
        self.max_rounds = max(1, max_rounds)
        self.max_seconds = max(5.0, max_seconds)
        self.max_filings = max(1, max_filings)
        self.max_chars = max(1000, max_chars)

    def run(self, ticker: str, context: ExecutionContext, *, since: str = "", until: str = "", around: str = "", max_filings: int | None = None, today: date | None = None) -> ToolEnvelope:
        current = today or date.today()
        if around:
            anchor = date.fromisoformat(around[:10])
            since = since or (anchor - timedelta(days=14)).isoformat()
            until = until or min(current, anchor + timedelta(days=3)).isoformat()
        since = since or (current - timedelta(days=90)).isoformat()
        until = until or current.isoformat()
        window = f"{since} 至 {until}"
        limits = limits_for(context, max_rounds=self.max_rounds, max_seconds=self.max_seconds)
        refs = self.source.list_filings(ticker, since, until)
        limit = max(1, min(int(max_filings or self.max_filings), 3))
        if around:
            refs.sort(key=lambda ref: abs((date.fromisoformat(ref.filing_date) - date.fromisoformat(around[:10])).days) if ref.filing_date else 999)
        chosen = refs[:limit]
        if not chosen:
            return self._envelope(ticker, window, around, [], [], [], note=f"{window} 未查到申报", metrics={"rounds": 0, "llm_calls": 0, "stop_reason": "no_filings", "elapsed_ms": 0}, status=ResultStatus.COMPLETED)
        if self.llm is None:
            return self._envelope(ticker, window, around, chosen, [], [], note="未配置模型，只列出申报，未读取内容", metrics={"rounds": 0, "llm_calls": 0}, status=ResultStatus.PARTIAL_DATA)

        outlines = {index: self.source.outline(ref) for index, ref in enumerate(chosen, 1)}
        listing = "\n".join(
            f"申报 {index}：{ref.form} {ref.filing_date}（{ref.accession}）\n" + "\n".join(f"  - {section.id}：{section.title}（{section.chars} 字符）" for section in outlines[index])
            for index, ref in enumerate(chosen, 1)
        )
        task = f"股票：{ticker}\n关注日期：{around or '无'}\n申报窗口：{window}\n{listing}"
        loop = _ReadLoop(self.llm, limits, self.source, chosen, self.max_chars)
        outcome = loop.run(_SYSTEM, task, finish_prompt=_FINISH_NOW)
        events = [row for row in (outcome.final.get("events") or []) if isinstance(row, dict)] if outcome.finished else []
        note = str(outcome.final.get("note") or "") if outcome.finished else outcome.note
        verified, dropped = _verify_events(events, loop.read)
        if dropped:
            note = (note + "；" if note else "") + f"{dropped} 条事件的引文与已读文本不符，已丢弃"
        status = ResultStatus.COMPLETED if verified else ResultStatus.PARTIAL_DATA
        metrics = {"rounds": outcome.rounds, "llm_calls": outcome.calls, "elapsed_ms": outcome.elapsed_ms, "stop_reason": outcome.stop_reason, "seconds_allowed": round(outcome.seconds_allowed, 1)}
        envelope = self._envelope(ticker, window, around, chosen, verified, sorted(loop.read), note=note, metrics=metrics, status=status)
        envelope.metadata["trace"] = list(outcome.trace)
        envelope.metadata["agent"] = {"name": "filing_reader", "label": "申报阅读", "subject": f"{ticker} {around or window}", "rounds": outcome.rounds, "llm_calls": outcome.calls, "elapsed_ms": outcome.elapsed_ms, "seconds_allowed": round(outcome.seconds_allowed, 1), "stop_reason": outcome.stop_reason, "calls": {"read": len(loop.read)}, "yield": {"kept": len(verified), "dropped": dropped}}
        return envelope

    def _envelope(self, ticker: str, window: str, around: str, refs: list[FilingRef], events: list[dict[str, Any]], read: list[tuple[int, str]], *, note: str, metrics: dict[str, Any], status: ResultStatus) -> ToolEnvelope:
        evidence: list[EvidenceItem] = []
        for row in events:
            ref = refs[int(row["filing"]) - 1]
            quote = str(row["quote"])
            section = str(row.get("section") or "")
            digest = hashlib.sha1(f"{ticker}|{ref.accession}|{section}|{quote}".encode("utf-8")).hexdigest()[:16]
            evidence.append(
                EvidenceItem(
                    id=f"evidence-filing-event-{digest}",
                    entity=ticker,
                    claim=f"{ticker} {row['date']}：{row['summary']}（{ref.form} {ref.filing_date} {section}：“{quote[:200]}”）",
                    as_of=str(row["date"]),
                    source_id="sec_edgar",
                    source_title=f"{ticker} {ref.form} {ref.filing_date}",
                    source_url=ref.url,
                    metadata={"evidence_scope": "filing_event", "date": str(row["date"]), "form": ref.form, "filing_date": ref.filing_date, "section": section, "quote": quote, "verified": True},
                )
            )
        limitations: list[str] = []
        if note:
            limitations.append(note)
        if not evidence:
            reason = f"{ticker} {window} 的 {len(refs)} 份申报中未读到与{around + ' 附近' if around else '该区间'}下跌相关的事件" if refs else f"{ticker} {window} 未查到申报"
            evidence.append(EvidenceItem(id=f"evidence-filing-event-none-{hashlib.sha1(f'{ticker}|{window}|{around}'.encode('utf-8')).hexdigest()[:12]}", entity=ticker, claim=reason + "。", source_id="sec_edgar", source_title="SEC EDGAR", metadata={"citation_kind": "limitations", "verified": True}))
        found = [item for item in evidence if item.metadata.get("evidence_scope") == "filing_event"]
        narrative = (f"{ticker} 申报中读到的事件：" + "；".join(f"{item.metadata['date']} {item.claim.split('：', 1)[1].split('（', 1)[0]}[{item.id}]" for item in found) + "。") if found else f"{evidence[0].claim.rstrip('。')}[{evidence[0].id}]。"
        return ToolEnvelope(
            "filings.read_events",
            status,
            subject=ticker,
            as_of=window.split(" 至 ")[-1],
            summary=f"{ticker} {window}：读了 {len(read)} 节，{len(found)} 条有出处的事件。",
            metrics={"filings": len(refs), "sections_read": len(read), "events": len(found), **metrics},
            evidence=evidence,
            limitations=limitations,
            metadata={"narrative": narrative, "dates": [item.metadata["date"] for item in found], "filings": [{"form": ref.form, "filing_date": ref.filing_date, "accession": ref.accession, "url": ref.url} for ref in refs], "reads": [f"{index}:{section}" for index, section in read], "around": around},
        )


def _verify_events(events: list[dict[str, Any]], read: dict[tuple[int, str], str]) -> tuple[list[dict[str, Any]], int]:
    """Keep only events whose quote can be located in a section the loop actually read.

    The quote is replaced by the section's own text at that spot, so what
    the answer cites is the filing's wording even when the model changed
    a comma or a number format.
    """

    kept: list[dict[str, Any]] = []
    dropped = 0
    for row in events:
        try:
            key = (int(row.get("filing")), str(row.get("section") or ""))
            quote = _WS.sub(" ", str(row.get("quote") or "")).strip()
            day = str(row.get("date") or "")[:10]
            date.fromisoformat(day)
        except (TypeError, ValueError):
            dropped += 1
            continue
        located = locate_quote(quote, read.get(key, "")) if quote and row.get("summary") else None
        if located is None:
            dropped += 1
            continue
        kept.append({**row, "date": day, "quote": located})
    return kept, dropped


_PUNCT = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'", "\u00a0": " ", "–": "-", "—": "-"})


def locate_quote(quote: str, text: str, *, minimum: int = 40, share: float = 0.6) -> str | None:
    """The passage of ``text`` the quote points at, or None when it is not there.

    An exact match wins.  Otherwise the longest common run between the two
    must cover at least ``share`` of the quote (and ``minimum`` characters);
    the returned passage is the text's own characters around that run.
    """

    from difflib import SequenceMatcher

    haystack = _WS.sub(" ", (text or "").translate(_PUNCT)).strip()
    needle = _WS.sub(" ", (quote or "").translate(_PUNCT)).strip()
    if not needle or not haystack:
        return None
    position = haystack.lower().find(needle.lower())
    if position >= 0:
        return haystack[position : position + len(needle)]
    # Small edits (a hyphen, a number format) break an exact match; the
    # matching runs together must still cover most of the quote and sit
    # within one passage of the text.
    blocks = [block for block in SequenceMatcher(None, haystack.lower(), needle.lower(), autojunk=False).get_matching_blocks() if block.size >= 3]
    covered = sum(block.size for block in blocks)
    if not blocks or covered < max(minimum, int(len(needle) * share)):
        return None
    first, last = blocks[0].a, blocks[-1].a + blocks[-1].size
    if last - first > 2 * len(needle) + 40:
        return None
    # Widen the passage to the sentence-ish boundaries around it in the text.
    start, end = first, last
    while start > 0 and haystack[start - 1] not in ".;。；\n" and first - start < 80:
        start -= 1
    while end < len(haystack) and haystack[end - 1] not in ".;。；" and end - last < 80:
        end += 1
    return haystack[start:end].strip()


def register_filing_reader(registry: CapabilityRegistry, llm: Any, *, source: FilingSource | None = None, today_factory: Callable[[], date] = date.today, **limits: Any) -> None:
    reader = FilingReader(llm, source or EdgarFilingSource(), **limits)

    def handler(arguments: dict[str, Any], context: ExecutionContext) -> ToolEnvelope:
        return reader.run(
            str(arguments.get("ticker") or "").upper(),
            context,
            since=str(arguments.get("since") or ""),
            until=str(arguments.get("until") or ""),
            around=str(arguments.get("around") or ""),
            max_filings=int(arguments.get("max_filings") or 0) or None,
            today=today_factory(),
        )

    registry.register("filings.read_events", handler)
