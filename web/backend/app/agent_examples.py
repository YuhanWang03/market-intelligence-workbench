"""Question groups the owner publishes for guests: one question, both agents' answers.

A guest cannot ask the agents anything, so the chat panel would be empty.
The owner asks a question on the examples page, both agents answer it,
and the pair is published as one group; guests read the groups through
the same frozen-snapshot door as every other guest read. Each answer
keeps the moment it was given: a guest reading it days later must see
that the figures belong to that day.
"""

from __future__ import annotations

import json
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any

from app.public_snapshots import publish_snapshot, read_snapshot

EXAMPLES_PATH = "/api/public/agent-examples"
MAX_GROUPS = 30
MODES = ("agent_v2", "agent_v3")
_MAX_GROUP_BYTES = 1024 * 1024
_ANSWER_KEYS = ("answered_at", "answer", "meta", "agent", "evidence")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _answer(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict) or not str(raw.get("answer") or "").strip():
        return None
    return {key: raw.get(key) for key in _ANSWER_KEYS}


def _migrate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rows published one answer at a time (the first version) become groups, paired by their question text."""
    groups: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
    for row in rows:
        if "answers" in row:  # already a group
            groups[row["id"]] = row
            continue
        key = "q:" + str(row.get("question") or "").strip()
        group = groups.get(key)
        if group is None:
            group = groups[key] = {"id": uuid.uuid4().hex[:12], "question": row.get("question"), "asked_at": row.get("asked_at"), "published_at": row.get("published_at") or _now(), "answers": {}}
        answer = _answer(row)
        if answer and row.get("mode") in MODES:
            group["answers"][row["mode"]] = answer
    return list(groups.values())


def list_groups() -> list[dict[str, Any]]:
    snapshot = read_snapshot(EXAMPLES_PATH)
    rows = [row for row in ((snapshot or {}).get("payload") or []) if isinstance(row, dict)]
    return _migrate(rows)


def add_group(group: dict[str, Any]) -> dict[str, Any]:
    """Append one question with its answers; returns the stored group (with id and publish time)."""
    question = str(group.get("question") or "").strip()
    if not question:
        raise ValueError("a group needs a question")
    answers = {mode: answer for mode in MODES if (answer := _answer((group.get("answers") or {}).get(mode)))}
    if not answers:
        raise ValueError("a group needs at least one answer")
    row = {"id": uuid.uuid4().hex[:12], "question": question, "asked_at": str(group.get("asked_at") or _now()), "published_at": _now(), "answers": answers}
    if len(json.dumps(row, ensure_ascii=False).encode("utf-8")) > _MAX_GROUP_BYTES:
        raise ValueError("group is too large")
    rows = [*list_groups(), row]
    if len(rows) > MAX_GROUPS:
        raise ValueError(f"at most {MAX_GROUPS} groups can be published; remove one first")
    publish_snapshot(EXAMPLES_PATH, rows)
    return row


def remove_group(group_id: str) -> bool:
    rows = list_groups()
    kept = [row for row in rows if row.get("id") != group_id]
    if len(kept) == len(rows):
        return False
    publish_snapshot(EXAMPLES_PATH, kept)
    return True
