"""Durable session state: what a restart of the bot must not forget.

Four small tables in one sqlite file (``data/agent_v2_session.sqlite`` by
default, ``AGENT_V2_SESSION_DB`` to move it):

* ``pending``        — the write plan a session is waiting to confirm ("确认" / "取消")
* ``clarifications`` — the question we asked and the original wording it was about
* ``turns``          — the recent exchanges (question, tickers, answer digest) a
                       follow-up refers back to
* ``active_runs``    — chats with an answer in progress, so a restart can tell
                       them to ask again instead of leaving a placeholder forever

Everything has a TTL; expired rows are ignored and pruned on write.  The
previous turn's full evidence stays in process memory on purpose: it is large,
minutes-lived, and a follow-up after a restart simply becomes a fresh question.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from v2.agent_v2.models import AnswerMode, BudgetClass, ExecutionPlan, PlanTask, RouteKind

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_PATH = _PROJECT_ROOT / "data" / "agent_v2_session.sqlite"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS pending (session_id TEXT PRIMARY KEY, expires_at REAL NOT NULL, plan TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS clarifications (session_id TEXT PRIMARY KEY, expires_at REAL NOT NULL, original_text TEXT NOT NULL, question TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS turns (session_id TEXT NOT NULL, ts REAL NOT NULL, query TEXT NOT NULL, tickers TEXT NOT NULL, tools TEXT NOT NULL, answer_digest TEXT NOT NULL, path TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS turns_session ON turns (session_id, ts);
CREATE TABLE IF NOT EXISTS active_runs (chat_id TEXT PRIMARY KEY, started_at REAL NOT NULL, question TEXT NOT NULL);
"""


def session_db_path() -> Path:
    return Path(os.environ.get("AGENT_V2_SESSION_DB") or _DEFAULT_PATH)


def plan_to_dict(plan: ExecutionPlan) -> dict[str, Any]:
    data = asdict(plan)
    data["route"] = plan.route.value
    data["budget"] = plan.budget.value
    data["answer_mode"] = plan.answer_mode.value
    return data


def plan_from_dict(data: dict[str, Any]) -> ExecutionPlan:
    tasks = tuple(PlanTask(id=str(row["id"]), capability=str(row["capability"]), arguments=dict(row.get("arguments") or {}), depends_on=tuple(row.get("depends_on") or ()), required=bool(row.get("required", True)), purpose=str(row.get("purpose") or ""), fan_out=row.get("fan_out")) for row in data.get("tasks") or [])
    return ExecutionPlan(
        objective=str(data.get("objective") or ""),
        route=RouteKind(data.get("route") or RouteKind.FAST_LOOKUP.value),
        tasks=tasks,
        budget=BudgetClass(data.get("budget") or BudgetClass.DIRECT.value),
        answer_mode=AnswerMode(data.get("answer_mode") or AnswerMode.TOOL_GROUNDED.value),
        requires_confirmation=bool(data.get("requires_confirmation")),
        web_fallback_allowed=bool(data.get("web_fallback_allowed")),
        assumptions=tuple(str(value) for value in data.get("assumptions") or ()),
        direct_answer=str(data.get("direct_answer") or ""),
        frame=dict(data.get("frame") or {}),
    )


class SqliteSessionState:
    """The durable half of a session; every method is safe to call from any thread."""

    def __init__(self, path: Path | None = None, *, ttl_seconds: float = 1800.0, max_turns: int = 12) -> None:
        self.path = path or session_db_path()
        self.ttl_seconds = float(ttl_seconds)
        self.max_turns = max_turns
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), timeout=10.0, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    # -- pending writes -----------------------------------------------------------

    def set_pending(self, session_id: str, plan: ExecutionPlan) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("INSERT OR REPLACE INTO pending (session_id, expires_at, plan) VALUES (?, ?, ?)", (session_id, time.time() + self.ttl_seconds, json.dumps(plan_to_dict(plan), ensure_ascii=False)))

    def pop_pending(self, session_id: str) -> ExecutionPlan | None:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT expires_at, plan FROM pending WHERE session_id = ?", (session_id,)).fetchone()
            conn.execute("DELETE FROM pending WHERE session_id = ?", (session_id,))
        if row is None or row[0] < time.time():
            return None
        try:
            return plan_from_dict(json.loads(row[1]))
        except (ValueError, KeyError, TypeError):
            return None

    # -- clarifications -------------------------------------------------------------

    def set_clarification(self, session_id: str, original_text: str, question: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("INSERT OR REPLACE INTO clarifications (session_id, expires_at, original_text, question) VALUES (?, ?, ?, ?)", (session_id, time.time() + self.ttl_seconds, original_text, question))

    def pop_clarification(self, session_id: str) -> tuple[str, str] | None:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT expires_at, original_text, question FROM clarifications WHERE session_id = ?", (session_id,)).fetchone()
            conn.execute("DELETE FROM clarifications WHERE session_id = ?", (session_id,))
        if row is None or row[0] < time.time():
            return None
        return str(row[1]), str(row[2])

    # -- turns ----------------------------------------------------------------------

    def record_turn(self, session_id: str, *, query: str, tickers: tuple[str, ...], tools: tuple[str, ...], answer_digest: str, path: str, ts: float | None = None) -> None:
        moment = time.time() if ts is None else ts
        with self._lock, self._connect() as conn:
            conn.execute("INSERT INTO turns (session_id, ts, query, tickers, tools, answer_digest, path) VALUES (?, ?, ?, ?, ?, ?, ?)", (session_id, moment, query, json.dumps(list(tickers)), json.dumps(list(tools)), answer_digest[:400], path))
            conn.execute("DELETE FROM turns WHERE session_id = ? AND ts NOT IN (SELECT ts FROM turns WHERE session_id = ? ORDER BY ts DESC LIMIT ?)", (session_id, session_id, self.max_turns))
            conn.execute("DELETE FROM turns WHERE ts < ?", (moment - self.ttl_seconds,))

    def turns(self, session_id: str) -> list[dict[str, Any]]:
        """The session's unexpired turns, oldest first."""

        cutoff = time.time() - self.ttl_seconds
        with self._lock, self._connect() as conn:
            rows = conn.execute("SELECT ts, query, tickers, tools, answer_digest, path FROM turns WHERE session_id = ? AND ts >= ? ORDER BY ts", (session_id, cutoff)).fetchall()
        return [{"ts": row[0], "query": row[1], "tickers": tuple(json.loads(row[2])), "tools": tuple(json.loads(row[3])), "answer_digest": row[4], "path": row[5]} for row in rows]

    def clear(self, session_id: str) -> None:
        with self._lock, self._connect() as conn:
            for table in ("pending", "clarifications", "turns"):
                conn.execute(f"DELETE FROM {table} WHERE session_id = ?", (session_id,))

    # -- in-flight runs (the surface's bookkeeping) ----------------------------------

    def mark_active(self, chat_id: str, question: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("INSERT OR REPLACE INTO active_runs (chat_id, started_at, question) VALUES (?, ?, ?)", (chat_id, time.time(), question[:200]))

    def clear_active(self, chat_id: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM active_runs WHERE chat_id = ?", (chat_id,))

    def take_interrupted(self) -> list[dict[str, Any]]:
        """Runs that were in progress when the process died, removed as they are read."""

        with self._lock, self._connect() as conn:
            rows = conn.execute("SELECT chat_id, started_at, question FROM active_runs").fetchall()
            conn.execute("DELETE FROM active_runs")
        return [{"chat_id": row[0], "started_at": row[1], "question": row[2]} for row in rows]
