"""Answered questions the owner publishes for guests to read.

A guest cannot ask the agents anything, so the chat panel would be empty.
The owner instead picks real answers from their own session and publishes
them; the list is stored as one public snapshot under ``EXAMPLES_PATH`` so
guests read it through the same frozen-snapshot door as everything else.
Each example keeps the moment it was asked and answered: a guest reading
it days later must see that the figures belong to that day.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from app.public_snapshots import publish_snapshot, read_snapshot

EXAMPLES_PATH = "/api/public/agent-examples"
MAX_EXAMPLES = 12
_MAX_EXAMPLE_BYTES = 512 * 1024
_KEEP = ("question", "mode", "asked_at", "answered_at", "answer", "meta", "agent", "evidence")


def list_examples() -> list[dict[str, Any]]:
    snapshot = read_snapshot(EXAMPLES_PATH)
    rows = (snapshot or {}).get("payload") or []
    return [row for row in rows if isinstance(row, dict)]


def add_example(example: dict[str, Any]) -> dict[str, Any]:
    """Append one answered question; returns the stored row (with its id and publish time)."""
    row = {key: example.get(key) for key in _KEEP}
    if not str(row.get("question") or "").strip() or not str(row.get("answer") or "").strip():
        raise ValueError("an example needs a question and an answer")
    if row.get("mode") not in {"agent_v2", "agent_v3"}:
        raise ValueError("mode must be agent_v2 or agent_v3")
    if len(json.dumps(row, ensure_ascii=False).encode("utf-8")) > _MAX_EXAMPLE_BYTES:
        raise ValueError("example is too large")
    row["id"] = uuid.uuid4().hex[:12]
    row["published_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = [*list_examples(), row]
    if len(rows) > MAX_EXAMPLES:
        raise ValueError(f"at most {MAX_EXAMPLES} examples can be published; remove one first")
    publish_snapshot(EXAMPLES_PATH, rows)
    return row


def remove_example(example_id: str) -> bool:
    rows = list_examples()
    kept = [row for row in rows if row.get("id") != example_id]
    if len(kept) == len(rows):
        return False
    publish_snapshot(EXAMPLES_PATH, kept)
    return True
