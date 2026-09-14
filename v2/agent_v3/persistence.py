"""V3-only session metadata and conservative write replay journal."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

from v2.agent_v3.contracts import envelope_from, plain


class SessionStore:
    def __init__(self, path=":memory:", *, ttl=1800):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.lock = threading.RLock()
        self.ttl = ttl
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, payload TEXT NOT NULL, updated REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS mutations (id TEXT PRIMARY KEY, task TEXT NOT NULL, status TEXT NOT NULL, result TEXT);
            CREATE TABLE IF NOT EXISTS completed_reads (id TEXT PRIMARY KEY, task TEXT NOT NULL, result TEXT NOT NULL);
        """)
        self.conn.commit()

    def get(self, session_id):
        if not session_id:
            return {}
        with self.lock:
            row = self.conn.execute("SELECT payload, updated FROM sessions WHERE id=?", (session_id,)).fetchone()
        return json.loads(row[0]) if row and time.time() - row[1] < self.ttl else {}

    def put(self, session_id, payload):
        if not session_id:
            return
        encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False)
        if len(encoded) > 500000:
            # Keep the answer digest and pending action; omit oversized reusable evidence.
            payload = {**payload, "previous": None}
            encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False)
        with self.lock, self.conn:
            self.conn.execute("INSERT OR REPLACE INTO sessions VALUES (?, ?, ?)", (session_id, encoded, time.time()))
            self.conn.execute("DELETE FROM sessions WHERE updated < ?", (time.time() - self.ttl,))

    def mutate(self, key, task, invoke):
        serialized = json.dumps(plain(task), sort_keys=True, ensure_ascii=False)
        with self.lock, self.conn:
            row = self.conn.execute("SELECT task,status,result FROM mutations WHERE id=?", (key,)).fetchone()
            if row:
                if row[0] != serialized:
                    raise PermissionError("Confirmed operation changed")
                if row[1] == "completed":
                    return envelope_from(json.loads(row[2]))
                raise RuntimeError("Mutation outcome uncertain; inspect state before a new operation")
            self.conn.execute("INSERT INTO mutations VALUES (?, ?, 'started', NULL)", (key, serialized))
        # Commit 'started' before the side effect. A crash never silently retries it.
        result = invoke()
        with self.lock, self.conn:
            self.conn.execute("UPDATE mutations SET status='completed', result=? WHERE id=?", (json.dumps(plain(result), ensure_ascii=False), key))
        return result

    def read_completed(self, key, task, invoke):
        serialized = json.dumps(plain(task), sort_keys=True, ensure_ascii=False)
        with self.lock:
            row = self.conn.execute("SELECT task,result FROM completed_reads WHERE id=?", (key,)).fetchone()
        if row:
            if row[0] != serialized:
                raise ValueError("Recovered task differs from saved task")
            return envelope_from(json.loads(row[1]))
        result = invoke()
        if result.ok:
            with self.lock, self.conn:
                self.conn.execute("INSERT OR REPLACE INTO completed_reads VALUES (?,?,?)", (key,serialized,json.dumps(plain(result),ensure_ascii=False)))
        return result

    def close(self):
        self.conn.close()
