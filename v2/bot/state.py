"""SQLite-backed state for the Telegram bot.

Stores the user's watchlist and per-feature settings. Single-user MVP — we
filter by TELEGRAM_CHAT_ID at the bot's authorization layer, so there's no
user_id column in the schema.

The DB lives at data/bot_state.db, separate from agent DBs so bot crashes
can't corrupt monitoring data and vice versa.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DB_PATH = _PROJECT_ROOT / "data" / "bot_state.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS watchlist (
    ticker     TEXT PRIMARY KEY,
    added_at   TEXT NOT NULL,
    note       TEXT
);

CREATE TABLE IF NOT EXISTS settings (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS alerts (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker        TEXT NOT NULL,
    direction     TEXT NOT NULL,           -- 'above' or 'below'
    target_price  REAL NOT NULL,
    created_at    TEXT NOT NULL,
    fired_at      TEXT,                    -- NULL until triggered
    fired_price   REAL                     -- price at trigger time
);

CREATE INDEX IF NOT EXISTS idx_alerts_unfired
    ON alerts(ticker, direction) WHERE fired_at IS NULL;

CREATE TABLE IF NOT EXISTS intraday_cooldown (
    ticker        TEXT PRIMARY KEY,
    last_fired_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS intraday_volume_baseline (
    ticker         TEXT PRIMARY KEY,
    avg_volume_30d REAL NOT NULL,
    updated_at     TEXT NOT NULL          -- ISO date; refresh once per ET trading day
);

CREATE TABLE IF NOT EXISTS intraday_universe_overrides (
    ticker     TEXT PRIMARY KEY,
    enabled    INTEGER NOT NULL CHECK (enabled IN (0, 1)),
    updated_at TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def _conn():
    """SQLite connection with WAL mode for safe concurrent reads."""
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH), timeout=10.0)
    conn.row_factory = sqlite3.Row
    # WAL lets the scheduler read watchlist while the bot is writing
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        conn.executescript(_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Watchlist API
# ---------------------------------------------------------------------------


def watchlist_list() -> list[dict]:
    """Return all watchlist entries (oldest first)."""
    with _conn() as conn:
        rows = conn.execute(
            "SELECT ticker, added_at, note FROM watchlist ORDER BY added_at ASC"
        ).fetchall()
    return [dict(r) for r in rows]


def watchlist_add(ticker: str, note: str = "") -> bool:
    """Add a ticker. Returns True if new, False if already present."""
    ticker = ticker.strip().upper()
    if not ticker or not ticker.isalpha() or len(ticker) > 5:
        raise ValueError(f"Invalid ticker: {ticker!r}")
    with _conn() as conn:
        existing = conn.execute(
            "SELECT 1 FROM watchlist WHERE ticker=?", (ticker,)
        ).fetchone()
        if existing:
            return False
        conn.execute(
            "INSERT INTO watchlist (ticker, added_at, note) VALUES (?, ?, ?)",
            (ticker, _now(), note),
        )
    return True


def watchlist_remove(ticker: str) -> bool:
    """Remove a ticker. Returns True if removed, False if not present."""
    ticker = ticker.strip().upper()
    with _conn() as conn:
        cur = conn.execute("DELETE FROM watchlist WHERE ticker=?", (ticker,))
    return (cur.rowcount or 0) > 0


def watchlist_contains(ticker: str) -> bool:
    ticker = ticker.strip().upper()
    with _conn() as conn:
        row = conn.execute(
            "SELECT 1 FROM watchlist WHERE ticker=?", (ticker,)
        ).fetchone()
    return row is not None


# ---------------------------------------------------------------------------
# Settings API (read-only for now; Stage 2 will add setters)
# ---------------------------------------------------------------------------


def settings_get(key: str, default: str = "") -> str:
    with _conn() as conn:
        row = conn.execute(
            "SELECT value FROM settings WHERE key=?", (key,)
        ).fetchone()
    return row["value"] if row else default


def settings_set(key: str, value: str) -> None:
    with _conn() as conn:
        conn.execute(
            """INSERT INTO settings (key, value, updated_at)
               VALUES (?, ?, ?)
               ON CONFLICT(key) DO UPDATE SET
                 value=excluded.value, updated_at=excluded.updated_at""",
            (key, value, _now()),
        )


def settings_all() -> dict[str, str]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT key, value FROM settings ORDER BY key"
        ).fetchall()
    return {r["key"]: r["value"] for r in rows}


# ---------------------------------------------------------------------------
# Alerts API — price-threshold triggers consumed by the streamer service
# ---------------------------------------------------------------------------


def alert_add(ticker: str, direction: str, target_price: float) -> int:
    """Create a new price alert. Returns the row id."""
    ticker = ticker.strip().upper()
    direction = direction.strip().lower()
    if direction not in ("above", "below"):
        raise ValueError(f"direction must be 'above' or 'below', got {direction!r}")
    if not ticker or not ticker.isalpha() or len(ticker) > 5:
        raise ValueError(f"Invalid ticker: {ticker!r}")
    if target_price <= 0:
        raise ValueError(f"target_price must be positive, got {target_price}")

    with _conn() as conn:
        cur = conn.execute(
            """INSERT INTO alerts (ticker, direction, target_price, created_at)
               VALUES (?, ?, ?, ?)""",
            (ticker, direction, float(target_price), _now()),
        )
        return int(cur.lastrowid or 0)


def alert_list(include_fired: bool = False) -> list[dict]:
    """Return alerts. By default unfired only — recently fired ones via the flag."""
    where = "" if include_fired else "WHERE fired_at IS NULL"
    with _conn() as conn:
        rows = conn.execute(
            f"""SELECT id, ticker, direction, target_price, created_at,
                       fired_at, fired_price
                FROM alerts {where}
                ORDER BY created_at DESC"""
        ).fetchall()
    return [dict(r) for r in rows]


def alert_remove(alert_id: int) -> bool:
    """Hard-delete an alert by id. Returns True if a row was removed."""
    with _conn() as conn:
        cur = conn.execute("DELETE FROM alerts WHERE id=?", (alert_id,))
    return (cur.rowcount or 0) > 0


def alert_unfired_tickers() -> list[str]:
    """Distinct tickers with at least one unfired alert — what the streamer polls."""
    with _conn() as conn:
        rows = conn.execute(
            """SELECT DISTINCT ticker FROM alerts
               WHERE fired_at IS NULL
               ORDER BY ticker"""
        ).fetchall()
    return [r["ticker"] for r in rows]


# ---------------------------------------------------------------------------
# Intraday anomaly scanner state — cooldown table + volume baselines
# ---------------------------------------------------------------------------


def _normalize_intraday_ticker(ticker: str) -> str:
    ticker = ticker.strip().upper()
    if not ticker or not ticker.isalpha() or len(ticker) > 5:
        raise ValueError(f"Invalid ticker: {ticker!r}")
    return ticker


def intraday_universe_list() -> list[str]:
    """Return the editable production universe used by the minute scanner.

    TECH_30 remains the auditable default.  This table only stores operator
    overrides, so an existing installation inherits future default-universe
    maintenance without requiring a data migration.
    """
    from v2.screening.universe import TECH_30

    with _conn() as conn:
        rows = conn.execute(
            "SELECT ticker, enabled FROM intraday_universe_overrides ORDER BY updated_at, ticker"
        ).fetchall()

    disabled = {row["ticker"] for row in rows if not row["enabled"]}
    result = [ticker for ticker in TECH_30 if ticker not in disabled]
    present = set(result)
    for row in rows:
        ticker = row["ticker"]
        if row["enabled"] and ticker not in present:
            result.append(ticker)
            present.add(ticker)
    return result


def intraday_universe_add(ticker: str) -> bool:
    """Enable *ticker* for minute-level anomaly scans."""
    from v2.screening.universe import TECH_30

    ticker = _normalize_intraday_ticker(ticker)
    with _conn() as conn:
        row = conn.execute(
            "SELECT enabled FROM intraday_universe_overrides WHERE ticker=?", (ticker,)
        ).fetchone()
        already_enabled = bool(row["enabled"]) if row is not None else ticker in TECH_30
        if already_enabled:
            return False
        conn.execute(
            """INSERT INTO intraday_universe_overrides (ticker, enabled, updated_at)
               VALUES (?, 1, ?)
               ON CONFLICT(ticker) DO UPDATE SET enabled=1, updated_at=excluded.updated_at""",
            (ticker, _now()),
        )
    return True


def intraday_universe_remove(ticker: str) -> bool:
    """Disable *ticker* for minute-level anomaly scans."""
    from v2.screening.universe import TECH_30

    ticker = _normalize_intraday_ticker(ticker)
    current = set(intraday_universe_list())
    if ticker not in current:
        return False
    with _conn() as conn:
        if ticker in TECH_30:
            conn.execute(
                """INSERT INTO intraday_universe_overrides (ticker, enabled, updated_at)
                   VALUES (?, 0, ?)
                   ON CONFLICT(ticker) DO UPDATE SET enabled=0, updated_at=excluded.updated_at""",
                (ticker, _now()),
            )
        else:
            conn.execute("DELETE FROM intraday_universe_overrides WHERE ticker=?", (ticker,))
    return True


def intraday_in_cooldown(ticker: str, minutes: int = 30) -> bool:
    """True if *ticker* fired an intraday anomaly within the last *minutes*."""
    from datetime import datetime, timedelta, timezone
    ticker = ticker.strip().upper()
    cutoff = (datetime.now(timezone.utc) - timedelta(minutes=minutes)).isoformat(
        timespec="seconds"
    )
    with _conn() as conn:
        row = conn.execute(
            "SELECT last_fired_at FROM intraday_cooldown WHERE ticker=?",
            (ticker,),
        ).fetchone()
    if row is None:
        return False
    return row["last_fired_at"] > cutoff


def intraday_record_fire(ticker: str) -> None:
    """Mark *ticker* as having fired an intraday anomaly just now."""
    ticker = ticker.strip().upper()
    with _conn() as conn:
        conn.execute(
            """INSERT INTO intraday_cooldown (ticker, last_fired_at)
               VALUES (?, ?)
               ON CONFLICT(ticker) DO UPDATE SET last_fired_at=excluded.last_fired_at""",
            (ticker, _now()),
        )


def baseline_get(ticker: str) -> tuple[float | None, str | None]:
    """Return (avg_volume_30d, updated_at_iso) or (None, None) if missing."""
    ticker = ticker.strip().upper()
    with _conn() as conn:
        row = conn.execute(
            "SELECT avg_volume_30d, updated_at FROM intraday_volume_baseline WHERE ticker=?",
            (ticker,),
        ).fetchone()
    if row is None:
        return None, None
    return float(row["avg_volume_30d"]), row["updated_at"]


def baseline_set(ticker: str, avg_volume_30d: float) -> None:
    """Upsert a 30-day average volume for *ticker*."""
    ticker = ticker.strip().upper()
    with _conn() as conn:
        conn.execute(
            """INSERT INTO intraday_volume_baseline (ticker, avg_volume_30d, updated_at)
               VALUES (?, ?, ?)
               ON CONFLICT(ticker) DO UPDATE SET
                 avg_volume_30d=excluded.avg_volume_30d,
                 updated_at=excluded.updated_at""",
            (ticker, float(avg_volume_30d), _now()),
        )


def alert_fire_check(ticker: str, current_price: float) -> list[dict]:
    """Find unfired alerts on *ticker* that *current_price* would trigger.

    Atomically marks them as fired (UPDATE … WHERE fired_at IS NULL guard)
    so a slow streamer + a fast one couldn't both fire the same alert.
    Returns the alerts that were just fired.
    """
    ticker = ticker.strip().upper()
    now_ts = _now()
    fired: list[dict] = []
    with _conn() as conn:
        # Atomically claim alerts that have crossed the threshold
        rows = conn.execute(
            """SELECT id, direction, target_price FROM alerts
               WHERE ticker=? AND fired_at IS NULL""",
            (ticker,),
        ).fetchall()
        for r in rows:
            triggered = (
                (r["direction"] == "above" and current_price >= r["target_price"]) or
                (r["direction"] == "below" and current_price <= r["target_price"])
            )
            if not triggered:
                continue
            cur = conn.execute(
                """UPDATE alerts
                   SET fired_at=?, fired_price=?
                   WHERE id=? AND fired_at IS NULL""",
                (now_ts, float(current_price), r["id"]),
            )
            if (cur.rowcount or 0) > 0:
                fired.append({
                    "id": r["id"],
                    "ticker": ticker,
                    "direction": r["direction"],
                    "target_price": r["target_price"],
                    "fired_price": float(current_price),
                    "fired_at": now_ts,
                })
    return fired
