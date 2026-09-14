"""Persistent run log for every Lab tool (``data/lab.db``).

The old in-process list vanished on every backend restart. Each run now
stores its parameters, a compact summary for lists, and the full result so
a past run can be reopened in the UI exactly as it was.
"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

from app.config import SETTINGS

SCHEMA = """
CREATE TABLE IF NOT EXISTS lab_runs (
  id TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  created_at TEXT NOT NULL,
  params_json TEXT NOT NULL,
  summary_json TEXT NOT NULL,
  result_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_lab_runs_created ON lab_runs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_lab_runs_kind ON lab_runs(kind, created_at DESC);
"""


def default_path() -> Path:
    return Path(os.environ.get("WEB_LAB_DB") or (SETTINGS.repo_root / "data" / "lab.db"))


class LabRunStore:
    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path else default_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            yield conn
            conn.commit()
        finally:
            conn.close()

    def save(self, kind: str, *, params: dict[str, Any] | None, summary: dict[str, Any], result: dict[str, Any]) -> str:
        run_id = uuid.uuid4().hex[:12]
        now = datetime.now(timezone.utc).isoformat(timespec="microseconds")
        dump = lambda v: json.dumps(v, ensure_ascii=False, default=str)  # noqa: E731
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO lab_runs (id, kind, created_at, params_json, summary_json, result_json) VALUES (?,?,?,?,?,?)",
                (run_id, kind, now, dump(params or {}), dump({**summary, "id": run_id, "kind": kind, "ran_at": now}), dump({**result, "lab_run_id": run_id})),
            )
        return run_id

    def list(self, limit: int = 50, kind: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT summary_json FROM lab_runs" + (" WHERE kind = ?" if kind else "") + " ORDER BY created_at DESC, rowid DESC LIMIT ?"
        args: tuple[Any, ...] = (kind, limit) if kind else (limit,)
        with self._conn() as conn:
            rows = conn.execute(sql, args).fetchall()
        return [json.loads(r["summary_json"]) for r in rows]

    def get(self, run_id: str) -> dict[str, Any] | None:
        with self._conn() as conn:
            row = conn.execute("SELECT kind, params_json, result_json FROM lab_runs WHERE id = ?", (run_id,)).fetchone()
        if not row:
            return None
        return {"kind": row["kind"], "params": json.loads(row["params_json"]), "result": json.loads(row["result_json"])}

    def latest(self, kind: str) -> dict[str, Any] | None:
        rows = self.list(limit=1, kind=kind)
        return rows[0] if rows else None

    def delete(self, run_id: str) -> bool:
        with self._conn() as conn:
            return conn.execute("DELETE FROM lab_runs WHERE id = ?", (run_id,)).rowcount > 0

    def delete_older_than(self, days: int) -> int:
        """Drop runs created more than ``days`` ago (their full results included); returns how many."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(timespec="microseconds")
        with self._conn() as conn:
            return conn.execute("DELETE FROM lab_runs WHERE created_at < ?", (cutoff,)).rowcount

    def counts(self) -> dict[str, int]:
        with self._conn() as conn:
            rows = conn.execute("SELECT kind, COUNT(*) AS n FROM lab_runs GROUP BY kind").fetchall()
        return {r["kind"]: r["n"] for r in rows}
