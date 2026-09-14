"""SQLite persistence for 13F filings and positions.

Single-file DB at data/edgar.db. No external service.
The contextmanager get_db() ensures the schema is up before use.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

from v2.institutional.models import Filing, Position

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DB_PATH = _PROJECT_ROOT / "data" / "edgar.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS filings (
    cik              TEXT NOT NULL,
    accession        TEXT NOT NULL,
    manager_name     TEXT NOT NULL,
    quarter          TEXT NOT NULL,
    filing_date      TEXT NOT NULL,
    period_of_report TEXT NOT NULL,
    portfolio_value  REAL NOT NULL,
    n_positions      INTEGER NOT NULL,
    PRIMARY KEY (cik, accession)
);

CREATE TABLE IF NOT EXISTS positions (
    accession   TEXT NOT NULL,
    cusip       TEXT NOT NULL,
    ticker      TEXT,
    issuer_name TEXT NOT NULL,
    shares      INTEGER NOT NULL,
    market_value REAL NOT NULL,
    PRIMARY KEY (accession, cusip)
);

CREATE INDEX IF NOT EXISTS idx_filings_cik_period
    ON filings(cik, period_of_report DESC);
CREATE INDEX IF NOT EXISTS idx_positions_accession
    ON positions(accession);
"""


@contextmanager
def get_db():
    """Yield an SQLite connection with the schema ensured. Commits on exit."""
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def has_filing(conn, cik: str, accession: str) -> bool:
    cur = conn.execute(
        "SELECT 1 FROM filings WHERE cik=? AND accession=? LIMIT 1",
        (cik, accession),
    )
    return cur.fetchone() is not None


def save_filing(conn, filing: Filing, positions: list[Position]) -> None:
    """Insert (or replace) a filing and all its positions."""
    conn.execute(
        """INSERT OR REPLACE INTO filings
           (cik, accession, manager_name, quarter, filing_date,
            period_of_report, portfolio_value, n_positions)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            filing.cik, filing.accession, filing.manager_name, filing.quarter,
            filing.filing_date, filing.period_of_report,
            filing.portfolio_value, filing.n_positions,
        ),
    )
    conn.executemany(
        """INSERT OR REPLACE INTO positions
           (accession, cusip, ticker, issuer_name, shares, market_value)
           VALUES (?, ?, ?, ?, ?, ?)""",
        [
            (p.accession, p.cusip, p.ticker, p.issuer_name,
             p.shares, p.market_value)
            for p in positions
        ],
    )


def get_previous_filing(
    conn,
    cik: str,
    current_accession: str,
) -> dict | None:
    """Latest filing for *cik* that is NOT *current_accession*."""
    cur = conn.execute(
        """SELECT * FROM filings
           WHERE cik=? AND accession != ?
           ORDER BY period_of_report DESC
           LIMIT 1""",
        (cik, current_accession),
    )
    row = cur.fetchone()
    return dict(row) if row else None


def get_positions_for(conn, accession: str) -> list[dict]:
    cur = conn.execute(
        "SELECT * FROM positions WHERE accession=?",
        (accession,),
    )
    return [dict(r) for r in cur.fetchall()]
