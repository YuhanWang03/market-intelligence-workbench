"""SQLite-backed archive of Telegram pushes.

Most messages are stored before delivery so failed pushes leave a diagnostic
record. Real-time anomaly and price-alert callers can instead opt into
Telegram-first persistence: only successfully delivered alerts enter the feed.

Image files live alongside the DB under data/images/YYYY-MM-DD/.
"""

from __future__ import annotations

import logging
import re
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import date, datetime, time as dtime, timedelta, timezone
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DB_PATH = _PROJECT_ROOT / "data" / "archive.db"
_IMG_ROOT = _PROJECT_ROOT / "data" / "images"
_ET = ZoneInfo("US/Eastern")
_REALTIME_AGENTS = ("intraday_anomaly", "alert", "anomaly")


def _observed(day: date) -> date:
    if day.weekday() == 5:
        return day - timedelta(days=1)
    if day.weekday() == 6:
        return day + timedelta(days=1)
    return day


def _nth_weekday(year: int, month: int, weekday: int, occurrence: int) -> date:
    day = date(year, month, 1)
    return day + timedelta(days=(weekday - day.weekday()) % 7 + 7 * (occurrence - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    if month == 12:
        day = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        day = date(year, month + 1, 1) - timedelta(days=1)
    return day - timedelta(days=(day.weekday() - weekday) % 7)


def _easter_sunday(year: int) -> date:
    """Gregorian Easter, used for NYSE Good Friday closures."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = (h + l - 7 * m + 114) % 31 + 1
    return date(year, month, day)


def _us_market_holidays(year: int) -> set[date]:
    holidays = {
        _observed(date(year, 1, 1)),
        _nth_weekday(year, 1, 0, 3),       # Martin Luther King Jr. Day
        _nth_weekday(year, 2, 0, 3),       # Washington's Birthday
        _easter_sunday(year) - timedelta(days=2),
        _last_weekday(year, 5, 0),         # Memorial Day
        _observed(date(year, 7, 4)),
        _nth_weekday(year, 9, 0, 1),       # Labor Day
        _nth_weekday(year, 11, 3, 4),      # Thanksgiving
        _observed(date(year, 12, 25)),
    }
    if year >= 2022:
        holidays.add(_observed(date(year, 6, 19)))
    # When next New Year's Day is Saturday, its observed closure is Dec 31.
    next_new_year = _observed(date(year + 1, 1, 1))
    if next_new_year.year == year:
        holidays.add(next_new_year)
    return holidays


def is_us_trading_day(day: date) -> bool:
    return day.weekday() < 5 and day not in _us_market_holidays(day.year)


def recent_trading_day_cutoff_iso(
    trading_days: int = 2,
    *,
    now: datetime | None = None,
) -> str:
    """UTC timestamp at the start of the earliest retained US trading day."""
    if trading_days < 1:
        raise ValueError("trading_days must be at least 1")
    cursor = (now or datetime.now(timezone.utc)).astimezone(_ET).date()
    while not is_us_trading_day(cursor):
        cursor -= timedelta(days=1)
    for _ in range(trading_days - 1):
        cursor -= timedelta(days=1)
        while not is_us_trading_day(cursor):
            cursor -= timedelta(days=1)
    return datetime.combine(cursor, dtime.min, tzinfo=_ET).astimezone(timezone.utc).isoformat()


def trading_day_expiry_iso(
    trading_days: int = 2,
    *,
    now: datetime | None = None,
) -> str:
    """Expire after a message has belonged to ``trading_days`` sessions."""
    if trading_days < 1:
        raise ValueError("trading_days must be at least 1")
    cursor = (now or datetime.now(timezone.utc)).astimezone(_ET).date()
    for _ in range(trading_days):
        cursor += timedelta(days=1)
        while not is_us_trading_day(cursor):
            cursor += timedelta(days=1)
    return datetime.combine(cursor, dtime.min, tzinfo=_ET).astimezone(timezone.utc).isoformat()

# Tickers appear inside <b>…</b> blocks. The old anchor `<b>TICKER</b>` only
# matched bare-ticker headers (e.g. screening cards). Anomaly cards have
# composite bolded headers like `<b>🚨 异动 · NVDA · 2026-05-27</b>` where
# the ticker is just one word among many — so we now scan every bold block
# for ticker-shaped words and filter out common false positives.
_BOLD_BLOCK_PATTERN = re.compile(r"<b>([^<]+)</b>")
_TICKER_WORD = re.compile(r"\b([A-Z]{2,5})\b")
_TICKER_BLOCKLIST = frozenset({
    "AI", "API", "CEO", "CFO", "COO", "CIO", "CTO", "EPS", "ETF", "EU", "EUR",
    "FY", "GDP", "GPU", "HBM", "ID", "IDE", "IT", "LLC", "NA", "NLP", "NYC",
    "PR", "Q1", "Q2", "Q3", "Q4", "RAG", "ROE", "ROI", "TTM", "UI", "UK",
    "US", "USA", "USD", "VS", "WSB", "YOY",
})


def _extract_tickers(text: str) -> set[str]:
    """Find ticker-shaped tokens in every <b>…</b> block, with a blocklist."""
    tickers: set[str] = set()
    for block in _BOLD_BLOCK_PATTERN.findall(text):
        for match in _TICKER_WORD.findall(block):
            if match in _TICKER_BLOCKLIST:
                continue
            tickers.add(match)
    return tickers

_SCHEMA = """
CREATE TABLE IF NOT EXISTS pushes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              TEXT NOT NULL,
    agent           TEXT NOT NULL,
    msg_type        TEXT NOT NULL,
    text_html       TEXT,
    image_path      TEXT,
    tickers         TEXT
);
CREATE INDEX IF NOT EXISTS idx_pushes_ts        ON pushes(ts DESC);
CREATE INDEX IF NOT EXISTS idx_pushes_agent     ON pushes(agent);
CREATE INDEX IF NOT EXISTS idx_pushes_tickers   ON pushes(tickers);
"""

# Columns added by Phase 2 (dashboard auto-push feed). Each is added
# idempotently at startup via PRAGMA table_info — SQLite's
# ADD COLUMN IF NOT EXISTS only landed in 3.35 and we want to support older.
_PHASE2_COLUMNS = (
    ("trace_json",  "TEXT"),  # JSON dump of v2.observability trace events
    ("title",       "TEXT"),  # short human title for the feed card
    ("expires_at",  "TEXT"),  # ISO ts after which the row may be cleaned up
)

# Phase 0 priority system (P0/P1/P2/P3). Same idempotent ALTER pattern.
_PRIORITY_COLUMNS = (
    ("importance_score", "INTEGER"),  # 0-100
    ("priority_tier",    "TEXT"),     # "P0" / "P1" / "P2" / "P3"
    ("priority_reasons", "TEXT"),     # comma-joined adjustment trail (debug)
)

# Indexes that depend on Phase 2 + priority columns. Created AFTER the
# columns exist.
_PHASE2_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_pushes_expires  ON pushes(expires_at);
CREATE INDEX IF NOT EXISTS idx_pushes_priority ON pushes(priority_tier, ts DESC);

-- P2 digest queue: lightweight row pointing at pushes(id). The cron
-- p2_digest_to_telegram.py drains this table once per day.
CREATE TABLE IF NOT EXISTS p2_digest_pending (
    push_id    INTEGER PRIMARY KEY,
    queued_at  TEXT NOT NULL,
    title      TEXT,
    tier       TEXT,
    FOREIGN KEY (push_id) REFERENCES pushes(id)
);

-- Earnings Agent dedup table (Phase 1). Tracks which (ticker, quarter)
-- pairs have already been shipped so the 21:00 ET cron can be retried
-- without spamming duplicate cards. outcome ∈ {"summarized","pending",
-- "skipped"} — only "summarized" rows count for dedup.
CREATE TABLE IF NOT EXISTS earnings_summarized (
    ticker        TEXT NOT NULL,
    report_period TEXT NOT NULL,
    summarized_at TEXT NOT NULL,
    outcome       TEXT,
    PRIMARY KEY (ticker, report_period)
);

-- Phase 2.5 full: per-day positions snapshot, written by ⑨b at 16:25 ET
-- Mon-Fri. Consumed by ⑩ Fri 19:00 ET to compute per-position weekly
-- attribution (avg_weight × weekly_return). PRIMARY KEY on
-- (snapshot_date, ticker) makes the ⑨b INSERT OR REPLACE idempotent
-- — a same-day cron rerun overwrites instead of duplicating rows.
CREATE TABLE IF NOT EXISTS positions_snapshot (
    snapshot_date TEXT NOT NULL,
    ticker        TEXT NOT NULL,
    market_value  REAL NOT NULL,
    weight        REAL NOT NULL,
    sector_etf    TEXT,
    PRIMARY KEY (snapshot_date, ticker)
);
-- "today's positions" lookup (⑨b idempotent write checks)
CREATE INDEX IF NOT EXISTS idx_positions_snapshot_date
    ON positions_snapshot(snapshot_date);
-- "this ticker's 5-day window" lookup (⑩ attribution loop)
CREATE INDEX IF NOT EXISTS idx_positions_snapshot_ticker
    ON positions_snapshot(ticker, snapshot_date);
"""


def _ensure_phase2_columns(conn: sqlite3.Connection) -> None:
    """Add the trace/title/expires_at + priority columns if missing.

    Robustness notes:

    - The cols snapshot is re-read AFTER every successful or failed
      ALTER. A naive "compute once, then loop" approach is vulnerable
      to TOCTOU: when two processes init Archive at the same moment,
      one wins the race and ALTERs trace_json, the other's snapshot
      still says trace_json is missing and its ALTER raises
      "duplicate column name", which would abort the loop and leave
      later columns (e.g. priority_tier) unmigrated. Re-reading + a
      per-column try/except converges to the right end state from
      any starting point.

    - "duplicate column name" specifically is swallowed: it means
      the column already exists (we raced), so the migration goal
      for this name is already satisfied. Any other OperationalError
      propagates so the operator sees it.
    """
    def _live_cols() -> set[str]:
        return {row[1] for row in conn.execute("PRAGMA table_info(pushes)")}

    cols = _live_cols()
    for name, ddl in (*_PHASE2_COLUMNS, *_PRIORITY_COLUMNS):
        if name in cols:
            continue
        try:
            conn.execute(f"ALTER TABLE pushes ADD COLUMN {name} {ddl}")
            cols.add(name)
        except sqlite3.OperationalError as exc:
            if "duplicate column" in str(exc).lower():
                # Lost the race — column already exists. Refresh and
                # keep going so subsequent iterations see reality.
                cols = _live_cols()
                continue
            raise

    # Final sanity check — if any required column is STILL missing
    # after the loop (e.g. a partial migration somewhere upstream
    # left the table in an inconsistent state), surface it loudly
    # rather than letting the next save fail with a cryptic
    # "no such column" error.
    final_cols = _live_cols()
    required = {n for n, _ in (*_PHASE2_COLUMNS, *_PRIORITY_COLUMNS)}
    missing = required - final_cols
    if missing:
        raise sqlite3.OperationalError(
            f"archive.db migration incomplete — still missing: {sorted(missing)}"
        )


class Archive:
    """Per-agent archive client. Pass to TelegramNotifier(archive=...).

    A single Archive instance is tied to one agent label (e.g. "screen") so
    each script knows what to tag its rows with. The underlying DB is shared.
    """

    def __init__(self, agent: str) -> None:
        self._agent = agent
        _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _IMG_ROOT.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(_SCHEMA)
            _ensure_phase2_columns(conn)
            conn.executescript(_PHASE2_INDEXES)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(str(_DB_PATH), timeout=10.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def prune_realtime(self, trading_days: int = 2) -> int:
        """Delete real-time feed rows older than the retained trading window."""
        cutoff = recent_trading_day_cutoff_iso(trading_days)
        placeholders = ",".join("?" for _ in _REALTIME_AGENTS)
        predicate = (
            f"(agent IN ({placeholders}) OR msg_type = 'intraday_anomaly') "
            "AND ts < ?"
        )
        try:
            with self._conn() as conn:
                params = (*_REALTIME_AGENTS, cutoff)
                conn.execute(
                    f"DELETE FROM p2_digest_pending WHERE push_id IN "
                    f"(SELECT id FROM pushes WHERE {predicate})",
                    params,
                )
                cur = conn.execute(f"DELETE FROM pushes WHERE {predicate}", params)
                return max(cur.rowcount, 0)
        except sqlite3.Error as exc:
            logger.warning("Real-time archive pruning failed: %s", exc)
            return 0

    # ------------------------------------------------------------------
    # Write side
    # ------------------------------------------------------------------

    def save_text(
        self,
        text: str,
        *,
        tickers: Iterable[str] = (),
        trace_json: str | None = None,
        title: str | None = None,
        expires_at: str | None = None,
        importance_score: int | None = None,
        priority_tier: str | None = None,
        priority_reasons: str | None = None,
    ) -> int:
        """Archive a text push. Returns the row id (also useful as a handle).

        Phase 2 + priority fields are all optional — callers that don't
        pass them get the same behavior as before.
        """
        return self._insert(
            msg_type="text",
            text_html=text,
            image_path=None,
            tickers=self._resolve_tickers(text, tickers),
            trace_json=trace_json,
            title=title,
            expires_at=expires_at,
            importance_score=importance_score,
            priority_tier=priority_tier,
            priority_reasons=priority_reasons,
        )

    def save_photo(
        self,
        image: bytes,
        caption: str = "",
        *,
        tickers: Iterable[str] = (),
        trace_json: str | None = None,
        title: str | None = None,
        expires_at: str | None = None,
        importance_score: int | None = None,
        priority_tier: str | None = None,
        priority_reasons: str | None = None,
    ) -> int:
        """Archive a photo push (image written to disk + caption to DB)."""
        ts_iso = _now_iso()
        date_dir = ts_iso[:10]  # YYYY-MM-DD
        target_dir = _IMG_ROOT / date_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        # Deterministic-ish filename, never collides
        primary_ticker = (
            _first_ticker(tickers)
            or _first_ticker(_extract_tickers(caption or ""))
            or "img"
        )
        suffix = uuid.uuid4().hex[:8]
        img_path = target_dir / f"{self._agent}_{primary_ticker}_{suffix}.png"
        try:
            img_path.write_bytes(image)
        except OSError as exc:
            logger.warning("Failed to write image %s: %s", img_path, exc)
            img_path = None

        rel_path = (
            str(img_path.relative_to(_PROJECT_ROOT))
            if img_path is not None else None
        )
        return self._insert(
            msg_type="photo",
            text_html=caption or None,
            image_path=rel_path,
            tickers=self._resolve_tickers(caption or "", tickers),
            ts_override=ts_iso,
            trace_json=trace_json,
            title=title,
            expires_at=expires_at,
            importance_score=importance_score,
            priority_tier=priority_tier,
            priority_reasons=priority_reasons,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _insert(
        self,
        *,
        msg_type: str,
        text_html: str | None,
        image_path: str | None,
        tickers: str | None,
        ts_override: str | None = None,
        trace_json: str | None = None,
        title: str | None = None,
        expires_at: str | None = None,
        importance_score: int | None = None,
        priority_tier: str | None = None,
        priority_reasons: str | None = None,
    ) -> int:
        ts = ts_override or _now_iso()
        try:
            with self._conn() as conn:
                cur = conn.execute(
                    """INSERT INTO pushes
                       (ts, agent, msg_type, text_html, image_path,
                        tickers, trace_json, title, expires_at,
                        importance_score, priority_tier, priority_reasons)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (ts, self._agent, msg_type, text_html, image_path,
                     tickers, trace_json, title, expires_at,
                     importance_score, priority_tier, priority_reasons),
                )
                row_id = cur.lastrowid or 0
                # Enqueue P2 rows for the daily digest cron.
                if priority_tier == "P2" and row_id:
                    conn.execute(
                        """INSERT OR REPLACE INTO p2_digest_pending
                           (push_id, queued_at, title, tier)
                           VALUES (?, ?, ?, ?)""",
                        (row_id, ts, title, priority_tier),
                    )
                return row_id
        except sqlite3.Error as exc:
            logger.warning("Archive insert failed (%s): %s", self._agent, exc)
            return 0

    # ---- P2 digest helpers ---------------------------------------------

    def get_pending_p2_digest(self) -> list[dict]:
        """Return P2 pushes still waiting to be summarized + sent.

        Joined against pushes so the digest cron has the title + agent +
        timestamp without two round-trips.
        """
        try:
            with self._conn() as conn:
                rows = conn.execute(
                    """SELECT p.id, p.ts, p.agent, p.title, p.priority_tier,
                              p.tickers, p.importance_score
                       FROM p2_digest_pending q
                       JOIN pushes p ON p.id = q.push_id
                       ORDER BY p.ts ASC"""
                ).fetchall()
            return [dict(r) for r in rows]
        except sqlite3.Error as exc:
            logger.warning("get_pending_p2_digest failed: %s", exc)
            return []

    # ---- Earnings dedup helpers -----------------------------------------

    def mark_earnings_summarized(
        self,
        ticker: str,
        report_period: str,
        outcome: str = "summarized",
    ) -> None:
        """Record that ``ticker``'s ``report_period`` filing has been
        processed by the 21:00 ET cron.

        ``outcome`` is one of ``"summarized"`` / ``"pending"`` / ``"skipped"``.
        Only ``"summarized"`` rows count toward the dedup set — pending
        markers stay in the table as historical evidence but don't block
        a later real summary from being pushed.
        """
        try:
            with self._conn() as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO earnings_summarized
                       (ticker, report_period, summarized_at, outcome)
                       VALUES (?, ?, ?, ?)""",
                    (ticker.upper(), report_period, _now_iso(), outcome),
                )
        except sqlite3.Error as exc:
            logger.warning("mark_earnings_summarized(%s, %s) failed: %s",
                           ticker, report_period, exc)

    def is_earnings_summarized(self, ticker: str, report_period: str) -> bool:
        """True iff ``(ticker, report_period)`` already shipped a real summary."""
        try:
            with self._conn() as conn:
                row = conn.execute(
                    """SELECT 1 FROM earnings_summarized
                       WHERE ticker = ? AND report_period = ?
                         AND outcome = 'summarized'""",
                    (ticker.upper(), report_period),
                ).fetchone()
            return row is not None
        except sqlite3.Error as exc:
            logger.warning("is_earnings_summarized(%s, %s) failed: %s",
                           ticker, report_period, exc)
            return False

    def get_summarized_set(self) -> set[tuple[str, str]]:
        """Return all ``(ticker, report_period)`` pairs with outcome=summarized.

        Cron passes this as ``already_summarized`` into
        :func:`v2.earnings.pipeline.run_summaries` so retried runs don't
        re-push the same card.
        """
        try:
            with self._conn() as conn:
                rows = conn.execute(
                    """SELECT ticker, report_period FROM earnings_summarized
                       WHERE outcome = 'summarized'"""
                ).fetchall()
            return {(r[0], r[1]) for r in rows}
        except sqlite3.Error as exc:
            logger.warning("get_summarized_set failed: %s", exc)
            return set()

    def clear_p2_digest(self, push_ids: Iterable[int]) -> int:
        """Drop the listed push ids from the digest queue.

        The pushes(id) row itself stays — only its digest-pending marker
        is removed. Returns number of rows deleted.
        """
        ids = [int(i) for i in push_ids if i]
        if not ids:
            return 0
        placeholders = ",".join("?" for _ in ids)
        try:
            with self._conn() as conn:
                cur = conn.execute(
                    f"DELETE FROM p2_digest_pending WHERE push_id IN ({placeholders})",
                    ids,
                )
                return cur.rowcount
        except sqlite3.Error as exc:
            logger.warning("clear_p2_digest failed: %s", exc)
            return 0

    # ---- Phase 2.5 full — positions_snapshot table ---------------------

    def write_position_snapshots(self, rows: Iterable[dict]) -> int:
        """Bulk INSERT OR REPLACE into ``positions_snapshot``.

        Each row dict must carry: ``snapshot_date`` (ISO),
        ``ticker``, ``market_value``, ``weight``, ``sector_etf``
        (``None`` allowed). Returns the number of rows written.

        Idempotent on (snapshot_date, ticker) — a same-day rerun
        overwrites the prior entry instead of duplicating. Returns 0
        on any SQLite error (matches the rest of this class's
        silent-skip posture so the cron stays alive on a transient
        lock conflict).
        """
        payload = [
            (
                r["snapshot_date"], r["ticker"],
                float(r["market_value"]), float(r["weight"]),
                r.get("sector_etf"),
            )
            for r in rows
        ]
        if not payload:
            return 0
        try:
            with self._conn() as conn:
                conn.executemany(
                    """INSERT OR REPLACE INTO positions_snapshot
                       (snapshot_date, ticker, market_value, weight, sector_etf)
                       VALUES (?, ?, ?, ?, ?)""",
                    payload,
                )
                return len(payload)
        except sqlite3.Error as exc:
            logger.warning("write_position_snapshots failed: %s", exc)
            return 0

    def get_position_snapshots(
        self,
        *,
        ticker: str | None = None,
        since: str | None = None,
        until: str | None = None,
    ) -> list[dict]:
        """Query ``positions_snapshot``. All filters optional.

        Returns rows as dicts ordered by (snapshot_date ASC, ticker ASC)
        so the ⑩ attribution loop can iterate the time series in order
        without sorting downstream.
        """
        clauses: list[str] = []
        params: list = []
        if ticker is not None:
            clauses.append("ticker = ?")
            params.append(ticker)
        if since is not None:
            clauses.append("snapshot_date >= ?")
            params.append(since)
        if until is not None:
            clauses.append("snapshot_date <= ?")
            params.append(until)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = (
            f"SELECT snapshot_date, ticker, market_value, weight, sector_etf "
            f"FROM positions_snapshot {where} "
            f"ORDER BY snapshot_date ASC, ticker ASC"
        )
        try:
            with self._conn() as conn:
                cur = conn.execute(sql, tuple(params))
                return [dict(row) for row in cur.fetchall()]
        except sqlite3.Error as exc:
            logger.warning("get_position_snapshots failed: %s", exc)
            return []

    def _resolve_tickers(
        self,
        text: str,
        explicit: Iterable[str],
    ) -> str | None:
        """Union explicit-passed tickers with any auto-detected from HTML."""
        found: set[str] = set(t.upper() for t in explicit if t)
        if text:
            found.update(_extract_tickers(text))
        return ",".join(sorted(found)) if found else None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _first_ticker(it) -> str | None:
    for x in it:
        if x:
            return str(x).upper()
    return None
