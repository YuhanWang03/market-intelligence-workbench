"""Adapter over the proven short-lived session resolver from Agent V1."""

from __future__ import annotations

import re
import threading
import time

from v2.agent_common import session as legacy_session
from v2.agent_v2.entities import extract_entities
from v2.agent_v2.models import AgentResult, ExecutionPlan, SessionResolution, ToolEnvelope
from v2.agent_v2.synthesis import choose_rankable, position_row

#: A follow-up that asks something about *the* stock without naming it: it
#: opens with the question itself.  "什么原因跌这么多" after "哪只跌得最多"
#: is about the one the answer named.
_FOLLOW_UP = re.compile(
    r"^(?:那|所以|然后|但|不过|嗯)?[\s,，、]*"
    r"(?:为什么|为啥|为何|什么原因|原因|怎么回事|怎么会|跌|涨|表现|走势|财报|估值|内部人|新闻|催化|风险"
    r"|还值得|值得|能不能|要不要|后续|接下来|前景|基本面|资金流|机构|供应链|产业链|目标价)",
    re.IGNORECASE,
)
#: Wording that names its own scope; such a question is not about the focus stock.
_OWN_SCOPE = re.compile(r"持仓|仓库|仓位|组合|账户|关注|自选|watchlist|portfolio|我的|宏观|市场|大盘|板块|指数|美联储|利率", re.IGNORECASE)
_CITATION = re.compile(r"\[[A-Za-z0-9_.:~-]+\]")


def focus_entities(answer: str) -> tuple[str, ...]:
    """The stocks an answer's first sentence names; a ranking answer names the winner there."""

    first = re.split(r"[。！？\n]", _CITATION.sub("", answer or ""), maxsplit=1)[0]
    return extract_entities(first)


def position_frame(result: AgentResult, ticker: str) -> dict:
    """What a turn said about ``ticker`` from a position table: column, label, value, cost basis."""

    found = position_row(result.results, ticker) if ticker else None
    if found is None:
        return {}
    source, row = found
    choice = choose_rankable(result.request.text, source.metadata.get("rankable"))
    rules = source.metadata.get("rankable") or []
    rule = choice[0] if choice else (rules[0] if rules and isinstance(rules[0], dict) else {})
    key = str(rule.get("field") or "pl_pct")
    return {
        "kind": "position",
        "ticker": ticker,
        "field": key,
        "text": str(rule.get("text") or f"{key}_text"),
        "label": str(rule.get("label") or key),
        "value": row.get(key),
        "value_text": row.get(str(rule.get("text") or f"{key}_text")),
        "avg_entry_price": row.get("avg_entry_price"),
        "source": source.capability,
    }


#: Sessions whose previous turn is kept for follow-ups; the oldest are dropped beyond this.
MAX_REMEMBERED_SESSIONS = 16
#: Envelope metadata a follow-up still needs; everything else (traces, sub-agent summaries, raw texts) is dropped.
_KEPT_METADATA = ("citation_kind", "fan_out_table", "answer_guidance", "narrative", "require_cited_numbers", "answer_constraints", "dimensions", "failed_tickers")


def _slim(envelope: ToolEnvelope) -> ToolEnvelope:
    """The envelope without what only the run that produced it needed."""

    from dataclasses import replace

    metadata = {key: value for key, value in (envelope.metadata or {}).items() if key in _KEPT_METADATA}
    return replace(envelope, evidence=list(envelope.evidence)[:40], findings=list(envelope.findings)[:12], metadata=metadata)


class ShortTermSession:
    """Per-session bounded memory; no database writes and no hidden reasoning."""

    def __init__(
        self,
        *,
        ttl_seconds: float = legacy_session.DEFAULT_TTL_SECONDS,
        max_turns: int = legacy_session.DEFAULT_MAX_TURNS,
        durable=None,
    ) -> None:
        self.store = legacy_session.SessionStore(
            ttl_seconds=ttl_seconds,
            max_turns=max_turns,
        )
        #: A ``SqliteSessionState`` (or anything with its methods): pending writes,
        #: clarifications and turns survive a restart; the previous turn's evidence does not.
        self.durable = durable
        self._hydrated: set[str] = set()
        self.ttl_seconds = float(ttl_seconds)
        self._pending: dict[str, tuple[float, ExecutionPlan]] = {}
        self._frames: dict[str, tuple[float, dict]] = {}
        #: session -> (expires_at, the original question, the clarifying question we asked)
        self._clarifications: dict[str, tuple[float, str, str]] = {}
        #: session -> (expires_at, the previous turn: question, answer, evidence, results); a
        #: follow-up about the answer itself ("第二点展开讲") is answered from it, no new calls.
        self._previous: dict[str, tuple[float, dict]] = {}
        self._lock = threading.Lock()

    def _frame(self, session_id: str, ticker: str) -> dict:
        with self._lock:
            entry = self._frames.get(session_id)
        if entry is None:
            return {}
        expires_at, frame = entry
        if time.monotonic() >= expires_at or str(frame.get("ticker") or "").upper() != ticker.upper():
            return {}
        return dict(frame)

    def resolve(self, session_id: str, text: str) -> SessionResolution:
        self._hydrate(session_id)
        raw = (text or "").strip()
        # A subject-less follow-up about the stock in focus: prepend it.  The
        # legacy resolver only knows pronouns and a bare "为什么".
        if raw and _FOLLOW_UP.match(raw) and not extract_entities(raw) and not _OWN_SCOPE.search(raw) and not legacy_session._PRONOUN.search(raw):
            focus = self.store.last_ticker(session_id)
            if focus:
                rewritten = f"{focus} {raw}"
                return SessionResolution(text=rewritten, rewritten=True, antecedent=focus, note=f"「{raw}」按上文补全为「{rewritten}」", frame=self._frame(session_id, focus))
        result = self.store.resolve(session_id, text)
        return SessionResolution(
            text=result.text,
            rewritten=result.rewritten,
            antecedent=result.antecedent,
            note=result.note,
            frame=self._frame(session_id, result.antecedent) if result.rewritten and result.antecedent else {},
        )

    def _hydrate(self, session_id: str) -> None:
        """After a restart the in-memory ring buffer is empty; load the session's durable turns once."""

        if self.durable is None or not session_id or session_id in self._hydrated:
            return
        self._hydrated.add(session_id)
        try:
            rows = self.durable.turns(session_id)
        except Exception:  # noqa: BLE001 — a broken store never breaks an answer
            return
        if self.store.recent(session_id, n=1):
            return
        for row in rows:
            self.store.record(session_id, legacy_session.Turn(query=row["query"], tickers=tuple(row["tickers"]), tools_used=tuple(row["tools"]), answer_digest=row["answer_digest"], path=row["path"], ts=float(row["ts"])))

    def recent_turns(self, session_id: str, n: int = 3) -> list[dict]:
        """The last ``n`` exchanges as the classifier and the synthesizer see them: question, answer digest, tickers."""

        if not session_id:
            return []
        self._hydrate(session_id)
        return [{"question": turn.query, "answer_digest": turn.answer_digest[:240], "tickers": list(turn.tickers)} for turn in self.store.recent(session_id, n=n)]

    def previous_turn(self, session_id: str) -> dict | None:
        """The previous answer with the evidence and results it was written from, while the session lasts."""

        with self._lock:
            entry = self._previous.get(session_id)
        if entry is None:
            return None
        expires_at, turn = entry
        return dict(turn) if time.monotonic() < expires_at else None

    def record(self, result: AgentResult) -> None:
        if not result.request.session_id:
            return
        if result.results or result.evidence:
            # Bounded three ways: one turn per session, slim envelopes (no traces, no
            # sub-agent summaries, no debate), and only the most recent sessions.  A
            # quality run opens a session per case; the full envelopes of forty
            # research runs held in memory got the process killed.
            kept = [_slim(item) for item in result.results if item.capability != "debate.challenge" and not item.metadata.get("fan_out_table")][:24]
            kept += [_slim(item) for item in result.results if item.metadata.get("fan_out_table")][:2]
            with self._lock:
                self._previous[result.request.session_id] = (time.monotonic() + self.ttl_seconds, {"question": result.request.original_text or result.request.text, "answer": (result.answer or "")[:4000], "evidence": list(result.evidence)[:60], "results": kept, "run_id": result.run_id})
                while len(self._previous) > MAX_REMEMBERED_SESSIONS:
                    self._previous.pop(next(iter(self._previous)))
        # A question with no ticker ("哪只跌得最多") gets its focus from the
        # answer, so the next turn can refer back to the stock it named.
        tickers = result.request.entities or focus_entities(result.answer)
        frame = position_frame(result, tickers[0]) if tickers else {}
        with self._lock:
            if frame:
                self._frames[result.request.session_id] = (time.monotonic() + self.ttl_seconds, frame)
            elif not result.request.metadata.get("context_frame"):
                # A turn about something else ends the frame; a framed follow-up keeps it.
                self._frames.pop(result.request.session_id, None)
        self._hydrated.add(result.request.session_id)
        self.store.record(
            result.request.session_id,
            legacy_session.Turn(
                query=result.request.text,
                tickers=tickers,
                tools_used=tuple(item.capability for item in result.results),
                answer_digest=result.answer[:300],
                path=result.route.kind.value,
            ),
        )
        if self.durable is not None:
            try:
                self.durable.record_turn(result.request.session_id, query=result.request.text, tickers=tuple(tickers), tools=tuple(item.capability for item in result.results), answer_digest=result.answer[:300], path=result.route.kind.value)
            except Exception:  # noqa: BLE001
                pass

    def set_pending(self, session_id: str, plan: ExecutionPlan) -> None:
        if not session_id:
            return
        with self._lock:
            self._pending[session_id] = (time.monotonic() + self.ttl_seconds, plan)
        if self.durable is not None:
            try:
                self.durable.set_pending(session_id, plan)
            except Exception:  # noqa: BLE001
                pass

    def set_clarification(self, session_id: str, original_text: str, question: str) -> None:
        """Remember that ``question`` was asked about ``original_text``; the next message answers it."""

        if not session_id:
            return
        with self._lock:
            self._clarifications[session_id] = (time.monotonic() + self.ttl_seconds, original_text, question)
        if self.durable is not None:
            try:
                self.durable.set_clarification(session_id, original_text, question)
            except Exception:  # noqa: BLE001
                pass

    def pop_clarification(self, session_id: str) -> tuple[str, str] | None:
        """The (original question, clarifying question) pair waiting on this session, once."""

        with self._lock:
            entry = self._clarifications.pop(session_id, None)
        durable = None
        if self.durable is not None:
            try:
                durable = self.durable.pop_clarification(session_id)  # consumed either way
            except Exception:  # noqa: BLE001
                durable = None
        if entry is None:
            return durable  # asked before a restart
        expires_at, original_text, question = entry
        return (original_text, question) if time.monotonic() < expires_at else None

    def pop_pending(self, session_id: str) -> ExecutionPlan | None:
        with self._lock:
            entry = self._pending.pop(session_id, None)
        durable = None
        if self.durable is not None:
            try:
                durable = self.durable.pop_pending(session_id)  # consumed either way
            except Exception:  # noqa: BLE001
                durable = None
        if entry is None:
            return durable  # held before a restart
        expires_at, plan = entry
        return plan if time.monotonic() < expires_at else None

    def clear(self, session_id: str) -> None:
        self.store.clear(session_id)
        if self.durable is not None:
            try:
                self.durable.clear(session_id)
            except Exception:  # noqa: BLE001
                pass
        with self._lock:
            self._pending.pop(session_id, None)
            self._frames.pop(session_id, None)
