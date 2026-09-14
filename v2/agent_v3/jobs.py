"""Durable read-only Web jobs. Interrupted work is explicitly retryable."""
import json
import sqlite3
from pathlib import Path


class JobJournal:
    def __init__(self, path):
        self.path = Path(path)

    def save(self, job_id, payload):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with sqlite3.connect(self.path,timeout=10) as db:
            db.execute("CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
            db.execute("INSERT OR REPLACE INTO jobs VALUES (?,?)",(job_id,json.dumps(payload,ensure_ascii=False)))

    def get(self, job_id):
        if not self.path.exists():
            return {}
        with sqlite3.connect(self.path,timeout=10) as db:
            row=db.execute("SELECT payload FROM jobs WHERE id=?",(job_id,)).fetchone()
        return json.loads(row[0]) if row else {}
