"""Query the local Telegram-push archive.

Usage:
    poetry run python scripts/query_archive.py                       # recent 20
    poetry run python scripts/query_archive.py recent 50             # recent N
    poetry run python scripts/query_archive.py ticker NVDA           # search by ticker
    poetry run python scripts/query_archive.py agent anomaly         # filter by agent
    poetry run python scripts/query_archive.py search "AI 芯片"      # full-text search
    poetry run python scripts/query_archive.py stats                 # counts by agent
"""

from __future__ import annotations

import re
import sqlite3
import sys
from pathlib import Path

_DB = Path(__file__).resolve().parent.parent / "data" / "archive.db"

_HTML_TAGS = re.compile(r"<[^>]+>")


def _strip_html(text: str) -> str:
    """Crude HTML stripping for terminal display."""
    return _HTML_TAGS.sub("", text or "").strip()


def _print_row(row) -> None:
    text = _strip_html(row["text_html"] or "")
    snippet = text[:160].replace("\n", " ").replace("━", "-")
    extra = f" 📷 {row['image_path']}" if row["image_path"] else ""
    print(
        f"[{row['ts'][:19]}] {row['agent']:<14} {row['msg_type']:<5} "
        f"{row['tickers'] or '-':<20} {snippet}{extra}"
    )


def _conn():
    if not _DB.exists():
        print(f"No archive DB found at {_DB} — run an agent once to create it.")
        sys.exit(1)
    conn = sqlite3.connect(str(_DB))
    conn.row_factory = sqlite3.Row
    return conn


def recent(n: int = 20) -> None:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM pushes ORDER BY id DESC LIMIT ?",
            (n,),
        ).fetchall()
    for r in reversed(rows):  # oldest first → newest at bottom of scroll
        _print_row(r)
    print(f"\n— showing last {len(rows)} pushes —")


def by_ticker(ticker: str) -> None:
    ticker = ticker.upper()
    with _conn() as conn:
        rows = conn.execute(
            """SELECT * FROM pushes
               WHERE tickers LIKE ?
               ORDER BY id DESC LIMIT 100""",
            (f"%{ticker}%",),
        ).fetchall()
    for r in reversed(rows):
        if ticker in (r["tickers"] or "").split(","):
            _print_row(r)
    print(f"\n— {len(rows)} pushes mentioning {ticker} —")


def by_agent(agent: str) -> None:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM pushes WHERE agent=? ORDER BY id DESC LIMIT 50",
            (agent,),
        ).fetchall()
    for r in reversed(rows):
        _print_row(r)
    print(f"\n— last {len(rows)} {agent} pushes —")


def search(query: str) -> None:
    with _conn() as conn:
        rows = conn.execute(
            """SELECT * FROM pushes
               WHERE text_html LIKE ?
               ORDER BY id DESC LIMIT 50""",
            (f"%{query}%",),
        ).fetchall()
    for r in reversed(rows):
        _print_row(r)
    print(f"\n— {len(rows)} pushes containing '{query}' —")


def stats() -> None:
    with _conn() as conn:
        total = conn.execute("SELECT COUNT(*) c FROM pushes").fetchone()["c"]
        by_a = conn.execute(
            """SELECT agent, msg_type, COUNT(*) c
               FROM pushes GROUP BY agent, msg_type ORDER BY agent, msg_type"""
        ).fetchall()
        date_range = conn.execute(
            "SELECT MIN(ts) a, MAX(ts) b FROM pushes"
        ).fetchone()
    print(f"Archive: {total} pushes total")
    print(f"Range:   {date_range['a']} → {date_range['b']}")
    print()
    print(f"{'agent':<16} {'type':<7} {'count':>6}")
    print("─" * 32)
    for row in by_a:
        print(f"{row['agent']:<16} {row['msg_type']:<7} {row['c']:>6}")


def main() -> None:
    args = sys.argv[1:]
    if not args:
        recent(20)
    elif args[0] == "recent":
        recent(int(args[1]) if len(args) > 1 else 20)
    elif args[0] == "ticker" and len(args) > 1:
        by_ticker(args[1])
    elif args[0] == "agent" and len(args) > 1:
        by_agent(args[1])
    elif args[0] == "search" and len(args) > 1:
        search(args[1])
    elif args[0] == "stats":
        stats()
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
