"""The turns the owner keeps on the two example pages, shown to guests.

A guest cannot ask the agents anything, so the chat panel would be empty.
The owner has an example page per agent that works like the agent's own
chat, except that every question and its answer is stored here as it
arrives, survives reloads, and is what guests see. The list is one public
snapshot under ``EXAMPLES_PATH`` so guests read it through the same
frozen-snapshot door as every other guest read. Each turn keeps the
moments it was asked and answered: a guest reading it days later must see
that the figures belong to that day.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from app.public_snapshots import publish_snapshot, read_snapshot

EXAMPLES_PATH = "/api/public/agent-examples"
MODES = ("agent_v2", "agent_v3")
MAX_TURNS_PER_MODE = 40
_MAX_TURN_BYTES = 512 * 1024
_KEEP = ("mode", "question", "asked_at", "answered_at", "answer", "meta", "agent", "evidence")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _migrate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rows stored as question groups (the second version) become one turn per agent; flat rows pass through."""
    out: list[dict[str, Any]] = []
    for row in rows:
        if "answers" not in row:
            out.append(row)
            continue
        for mode in MODES:
            answer = (row.get("answers") or {}).get(mode)
            if isinstance(answer, dict) and str(answer.get("answer") or "").strip():
                out.append({"id": uuid.uuid4().hex[:12], "mode": mode, "question": row.get("question"), "asked_at": row.get("asked_at"), "published_at": row.get("published_at") or _now(), **{key: answer.get(key) for key in ("answered_at", "answer", "meta", "agent", "evidence")}})
    return out


def list_turns() -> list[dict[str, Any]]:
    snapshot = read_snapshot(EXAMPLES_PATH)
    rows = [row for row in ((snapshot or {}).get("payload") or []) if isinstance(row, dict)]
    turns = _migrate(rows)
    if any("answers" in row for row in rows):
        publish_snapshot(EXAMPLES_PATH, turns)  # guests read the stored rows as they are, so the migration is written back once
    return turns


def add_turn(turn: dict[str, Any]) -> dict[str, Any]:
    """Append one question with its answer; returns the stored row (with id and publish time)."""
    row = {key: turn.get(key) for key in _KEEP}
    if row.get("mode") not in MODES:
        raise ValueError("mode must be agent_v2 or agent_v3")
    if not str(row.get("question") or "").strip() or not str(row.get("answer") or "").strip():
        raise ValueError("a turn needs a question and an answer")
    if len(json.dumps(row, ensure_ascii=False).encode("utf-8")) > _MAX_TURN_BYTES:
        raise ValueError("turn is too large")
    rows = list_turns()
    if sum(1 for existing in rows if existing.get("mode") == row["mode"]) >= MAX_TURNS_PER_MODE:
        raise ValueError(f"at most {MAX_TURNS_PER_MODE} turns are kept per agent; delete some first")
    row["id"] = uuid.uuid4().hex[:12]
    row["published_at"] = _now()
    publish_snapshot(EXAMPLES_PATH, [*rows, row])
    return row


def remove_turn(turn_id: str) -> bool:
    rows = list_turns()
    kept = [row for row in rows if row.get("id") != turn_id]
    if len(kept) == len(rows):
        return False
    publish_snapshot(EXAMPLES_PATH, kept)
    return True


def clear_turns(mode: str) -> int:
    """Remove every turn of one agent; returns how many went."""
    if mode not in MODES:
        raise ValueError("mode must be agent_v2 or agent_v3")
    rows = list_turns()
    kept = [row for row in rows if row.get("mode") != mode]
    publish_snapshot(EXAMPLES_PATH, kept)
    return len(rows) - len(kept)
