"""User memory: what the user told us about our answers and how they want them.

Two kinds of rows, appended to ``data/agent_v2_user_memory.jsonl``:

* ``feedback`` — a verdict on the last answer of a session (``good`` /
  ``bad``) with an optional note ("不对，7/29 那天是财报").  Negative feedback
  becomes a quality case (``eval.quality_cases.from_feedback``), so the next
  quality run checks the correction stuck.
* ``preference`` — a standing instruction ("以后用收盘价口径", "回答短一点").
  The synthesizer sees the session's preferences on every answer.

The triggers are a fixed set of commands, not wording detection: ``不对`` /
``👎`` / ``反馈：…`` / ``对`` / ``👍`` / ``记住：…`` (or ``记住，…``) / ``以后…``.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_PATH = _PROJECT_ROOT / "data" / "agent_v2_user_memory.jsonl"

_BAD = ("不对", "错了", "不准", "👎", "答错了")
_GOOD = ("对", "👍", "不错", "正确", "很好")
_FEEDBACK_PREFIX = ("反馈：", "反馈:", "纠正：", "纠正:")
_PREFERENCE_PREFIX = ("记住", "以后", "今后", "之后都")


def parse_feedback(text: str) -> dict[str, str] | None:
    """A feedback or preference command, or None for an ordinary question.

    Returns ``{"kind": "feedback", "verdict": "bad"|"good", "note": ...}`` or
    ``{"kind": "preference", "note": ...}``.
    """

    raw = " ".join((text or "").split())
    if not raw:
        return None
    for prefix in _FEEDBACK_PREFIX:
        if raw.startswith(prefix):
            return {"kind": "feedback", "verdict": "bad", "note": raw[len(prefix):].strip()}
    for prefix in _PREFERENCE_PREFIX:
        if raw.startswith(prefix):
            # "记住：…", "记住，…", "记住 …" all carry the instruction after the word.
            note = raw[len(prefix):].lstrip("：:，, ").strip() if prefix == "记住" else raw
            return {"kind": "preference", "note": note} if note else None
    stripped = raw.rstrip("。！!，, ")
    if stripped in _BAD or any(stripped.startswith(word + "，") or stripped.startswith(word + ",") for word in _BAD if len(stripped) <= 40):
        note = stripped
        for word in _BAD:
            if note.startswith(word):
                note = note[len(word):].lstrip("，, ")
                break
        return {"kind": "feedback", "verdict": "bad", "note": note}
    if stripped in _GOOD:
        return {"kind": "feedback", "verdict": "good", "note": ""}
    return None


class UserMemory:
    """Append-only user memory with an in-process view of the last answer per session."""

    def __init__(self, path: Path | None = None, *, max_preferences: int = 8) -> None:
        self.path = path or Path(os.environ.get("AGENT_V2_USER_MEMORY") or _DEFAULT_PATH)
        self.max_preferences = max_preferences
        self._last: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    # -- writing ---------------------------------------------------------------

    def remember_answer(self, session_id: str, *, question: str, answer: str, run_id: str) -> None:
        """Keep the last question/answer of a session so feedback can refer to it."""

        if not session_id:
            return
        with self._lock:
            self._last[session_id] = {"question": question, "answer_digest": (answer or "")[:400], "run_id": run_id}

    def record_feedback(self, session_id: str, verdict: str, note: str = "") -> dict[str, Any] | None:
        """Attach a verdict to the session's last answer; None when there is no answer to judge."""

        last = self._last.get(session_id)
        if last is None:
            return None
        row = {"at": _now(), "kind": "feedback", "session_id": session_id, "verdict": "good" if verdict == "good" else "bad", "note": (note or "")[:300], **last}
        self._append(row)
        return row

    def add_preference(self, session_id: str, note: str) -> dict[str, Any]:
        row = {"at": _now(), "kind": "preference", "session_id": session_id, "note": (note or "")[:200]}
        self._append(row)
        return row

    def _append(self, row: dict[str, Any]) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.warning("user memory not written (%s): %s", self.path, exc)

    # -- reading ---------------------------------------------------------------

    def rows(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict) and row.get("kind"):
                    rows.append(row)
        return rows

    def preferences(self, session_id: str = "") -> list[str]:
        """The standing instructions, newest last, duplicates dropped; all sessions when ``session_id`` is empty."""

        notes: list[str] = []
        for row in self.rows():
            if row.get("kind") != "preference" or (session_id and str(row.get("session_id") or "") != session_id):
                continue
            note = str(row.get("note") or "").strip()
            if note and note not in notes:
                notes.append(note)
        return notes[-self.max_preferences :]

    def feedback(self, session_id: str = "") -> list[dict[str, Any]]:
        return [row for row in self.rows() if row.get("kind") == "feedback" and (not session_id or str(row.get("session_id") or "") == session_id)]

    def last_answer(self, session_id: str) -> dict[str, Any] | None:
        return dict(self._last[session_id]) if session_id in self._last else None


def _now() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def render_feedback(rows: list[dict[str, Any]]) -> str:
    """The feedback report: counts, and every negative verdict with the question and the note."""

    feedback = [row for row in rows if row.get("kind") == "feedback"]
    preferences = [row for row in rows if row.get("kind") == "preference"]
    lines = ["# 用户反馈报告", ""]
    if not feedback and not preferences:
        lines.append("还没有反馈或偏好记录。")
        return "\n".join(lines)
    good = sum(1 for row in feedback if row.get("verdict") == "good")
    bad = len(feedback) - good
    notes = list(dict.fromkeys(str(row.get("note") or "") for row in preferences if row.get("note")))
    lines.append(f"反馈 {len(feedback)} 条：好 {good}、不对 {bad}；偏好 {len(notes)} 条。")
    if bad:
        lines.append("")
        lines.append("| 时间 | 问题 | 用户指出 |")
        lines.append("|---|---|---|")
        for row in feedback:
            if row.get("verdict") == "bad":
                lines.append(f"| {str(row.get('at') or '')[:16].replace('T', ' ')} | {row.get('question') or ''} | {row.get('note') or '（未说明）'} |")
    if notes:
        lines.append("")
        lines.append("偏好：" + "；".join(notes[-8:]))
    lines.append("")
    lines.append("不对的反馈会进入质量评测（python -m v2.agent_v2.eval.quality run --feedback），检查纠正是否生效。")
    return "\n".join(lines)
