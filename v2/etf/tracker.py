"""SQLite persistence for ETF daily holdings snapshots."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

from v2.etf.models import ETFHolding
from v2.observability import emit

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DB_PATH = _PROJECT_ROOT / "data" / "etf.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS snapshots (
    etf          TEXT NOT NULL,
    date         TEXT NOT NULL,
    ticker       TEXT NOT NULL,
    cusip        TEXT,
    company      TEXT,
    shares       REAL NOT NULL,
    market_value REAL NOT NULL,
    weight_pct   REAL NOT NULL,
    PRIMARY KEY (etf, date, ticker)
);

CREATE INDEX IF NOT EXISTS idx_snapshots_etf_date
    ON snapshots(etf, date DESC);
"""


@contextmanager
def _conn():
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH), timeout=10.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def save_snapshot(etf: str, snap_date: str, holdings: list[ETFHolding]) -> int:
    """Upsert all holdings for (etf, snap_date). Returns row count."""
    if not holdings:
        return 0
    with _conn() as c:
        # Idempotent: replace whatever was there for this (etf, date)
        c.execute("DELETE FROM snapshots WHERE etf=? AND date=?",
                  (etf, snap_date))
        c.executemany(
            """INSERT INTO snapshots
               (etf, date, ticker, cusip, company, shares, market_value, weight_pct)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (etf, snap_date, h.ticker or "?", h.cusip, h.company,
                 h.shares, h.market_value, h.weight_pct)
                for h in holdings
            ],
        )
        n = len(holdings)
        emit(
            "db_write", db="etf.db", fn="save_snapshot",
            etf=etf, snap_date=snap_date, rows=n,
        )
        return n


def get_snapshot(etf: str, snap_date: str) -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM snapshots WHERE etf=? AND date=?",
            (etf, snap_date),
        ).fetchall()
        return [dict(r) for r in rows]


def get_latest_snapshot_before(
    etf: str, before_date: str,
) -> list[dict] | None:
    """Most recent snapshot strictly before *before_date*. None if absent."""
    with _conn() as c:
        row = c.execute(
            """SELECT MAX(date) AS d FROM snapshots
               WHERE etf=? AND date<?""",
            (etf, before_date),
        ).fetchone()
        if not row or not row["d"]:
            return None
        rows = c.execute(
            "SELECT * FROM snapshots WHERE etf=? AND date=?",
            (etf, row["d"]),
        ).fetchall()
        return [dict(r) for r in rows]
