"""Persistent estimated-cost ledger for billable Financial Datasets requests."""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DB_PATH = _PROJECT_ROOT / "data" / "query_costs.db"
DEFAULT_PRICES_USD: dict[str, float] = {
    "financial_metrics": 0.02, "line_items": 0.02, "prices": 0.02,
    "earnings": 0.02, "insider_trades": 0.02, "news": 0.02,
    "company_facts": 0.02, "filings": 0.02,
}
_PATH_TO_ENDPOINT = {
    "/financial-metrics/": "financial_metrics", "/financials/": "line_items",
    "/prices/": "prices", "/earnings/": "earnings",
    "/insider-trades/": "insider_trades", "/news/": "news",
    "/company/facts/": "company_facts", "/filings/": "filings",
}
_SCHEMA = """
CREATE TABLE IF NOT EXISTS query_costs (
    id TEXT PRIMARY KEY, occurred_at TEXT NOT NULL, provider TEXT NOT NULL,
    endpoint TEXT NOT NULL, ticker TEXT, cost_usd REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_query_costs_time ON query_costs(occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_query_costs_endpoint ON query_costs(endpoint, occurred_at DESC);
"""


def _prices() -> dict[str, float]:
    table = dict(DEFAULT_PRICES_USD)
    raw = os.environ.get("FD_PRICES", "").strip()
    if raw:
        try:
            for key, value in json.loads(raw).items():
                if key in table:
                    table[key] = float(value)
        except (ValueError, TypeError, AttributeError):
            pass
    return table


@contextmanager
def _conn() -> Iterator[sqlite3.Connection]:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH), timeout=10.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def record_fd_request(path: str, params: dict | None = None) -> None:
    """Record one successful provider request; cache hits never reach here."""
    endpoint = _PATH_TO_ENDPOINT.get(path, path.strip("/").replace("/", "_") or "unknown")
    ticker = str((params or {}).get("ticker") or "").upper() or None
    with _conn() as conn:
        conn.execute(
            "INSERT INTO query_costs VALUES (?,?,?,?,?,?)",
            (uuid.uuid4().hex[:16], datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "Financial Datasets", endpoint, ticker, _prices().get(endpoint, 0.02)),
        )


def cost_report(limit: int = 100) -> dict:
    """Return aggregate totals and recent billable requests."""
    today = datetime.now(timezone.utc).date().isoformat()
    month = today[:7]
    with _conn() as conn:
        totals = conn.execute(
            """SELECT COALESCE(SUM(cost_usd),0) total,
               COALESCE(SUM(CASE WHEN substr(occurred_at,1,10)=? THEN cost_usd ELSE 0 END),0) today,
               COALESCE(SUM(CASE WHEN substr(occurred_at,1,7)=? THEN cost_usd ELSE 0 END),0) month,
               COUNT(*) requests FROM query_costs""", (today, month),
        ).fetchone()
        endpoints = conn.execute(
            """SELECT endpoint, COUNT(*) requests, ROUND(SUM(cost_usd),6) cost_usd
               FROM query_costs GROUP BY endpoint ORDER BY cost_usd DESC, endpoint"""
        ).fetchall()
        recent = conn.execute(
            """SELECT id,occurred_at,provider,endpoint,ticker,cost_usd FROM query_costs
               ORDER BY occurred_at DESC,rowid DESC LIMIT ?""", (max(1, min(limit, 500)),),
        ).fetchall()
    return {
        "currency": "USD", "basis": "successful_uncached_requests",
        "today_cost_usd": round(float(totals["today"]), 6),
        "month_cost_usd": round(float(totals["month"]), 6),
        "total_cost_usd": round(float(totals["total"]), 6),
        "total_requests": int(totals["requests"]),
        "by_endpoint": [dict(row) for row in endpoints],
        "recent": [dict(row) for row in recent],
    }
