"""Unusual options activity tracking (Phase B ③b).

Pipeline:
1. take_snapshot(ticker) — fetch current options chain via yfinance, sum the
   call/put open interest + volume on the nearest expiration, write a row to
   data/options.db.
2. detect_burst(ticker, current_snapshot) — compare today's call_oi / put_oi
   against the trailing 14-day mean from the DB. If ≥ 3.0×, return a burst signal.

Cold start: first ~5 trading days have no baseline → no burst detection.
After that the signal becomes available.

Used as augmentation in detectors.detect() — only invoked when a price-level
anomaly has already fired, to keep yfinance call count low.
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from datetime import date, timedelta
from pathlib import Path

import yfinance as yf

from v2.monitoring.models import OptionsBurst, OptionsSnapshot

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DB_PATH = _PROJECT_ROOT / "data" / "options.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS options_snapshots (
    ticker      TEXT NOT NULL,
    date        TEXT NOT NULL,
    call_oi     INTEGER NOT NULL,
    call_volume INTEGER NOT NULL,
    put_oi      INTEGER NOT NULL,
    put_volume  INTEGER NOT NULL,
    expiration  TEXT,
    PRIMARY KEY (ticker, date)
);
CREATE INDEX IF NOT EXISTS idx_options_ticker_date
    ON options_snapshots(ticker, date DESC);
"""

# Detection knobs
_BURST_MULTIPLE = 3.0          # ≥ this multiple of baseline = burst
_LOOKBACK_DAYS = 14
_MIN_BASELINE_SAMPLES = 5      # need at least N days before we'll detect


class OptionsTracker:
    """Daily snapshot store + burst detector for the options market."""

    def __init__(self) -> None:
        _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(str(_DB_PATH), timeout=10.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def take_snapshot(self, ticker: str) -> OptionsSnapshot | None:
        """Fetch current options chain via yfinance, persist + return."""
        try:
            ytk = yf.Ticker(ticker)
            expirations = ytk.options
            if not expirations:
                return None
            # Use the nearest expiration (most liquid)
            exp = expirations[0]
            chain = ytk.option_chain(exp)
            calls, puts = chain.calls, chain.puts
        except Exception as exc:
            logger.warning("yfinance option_chain(%s) failed: %s", ticker, exc)
            return None

        try:
            call_oi = int(calls["openInterest"].fillna(0).sum())
            call_vol = int(calls["volume"].fillna(0).sum())
            put_oi = int(puts["openInterest"].fillna(0).sum())
            put_vol = int(puts["volume"].fillna(0).sum())
        except (KeyError, AttributeError) as exc:
            logger.warning("options OI parse failed for %s: %s", ticker, exc)
            return None

        snapshot = OptionsSnapshot(
            ticker=ticker,
            date=date.today().isoformat(),
            call_oi=call_oi,
            call_volume=call_vol,
            put_oi=put_oi,
            put_volume=put_vol,
            expiration=exp,
        )

        try:
            with self._conn() as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO options_snapshots
                       (ticker, date, call_oi, call_volume, put_oi, put_volume, expiration)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        snapshot.ticker, snapshot.date,
                        snapshot.call_oi, snapshot.call_volume,
                        snapshot.put_oi, snapshot.put_volume,
                        snapshot.expiration,
                    ),
                )
        except sqlite3.Error as exc:
            logger.warning("options snapshot save failed for %s: %s", ticker, exc)

        return snapshot

    # ------------------------------------------------------------------
    # Burst detection
    # ------------------------------------------------------------------

    def detect_burst(
        self,
        ticker: str,
        current: OptionsSnapshot,
    ) -> OptionsBurst | None:
        """Compare current snapshot to the trailing 14-day baseline.

        Returns OptionsBurst if call_oi or put_oi is ≥ 3× the recent mean
        (and the recent mean is significant), else None.
        """
        cutoff = (date.today() - timedelta(days=_LOOKBACK_DAYS)).isoformat()
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT call_oi, put_oi FROM options_snapshots
                   WHERE ticker=? AND date >= ? AND date < ?""",
                (ticker, cutoff, current.date),
            ).fetchall()

        if len(rows) < _MIN_BASELINE_SAMPLES:
            return None  # cold start

        avg_call = sum(r["call_oi"] for r in rows) / len(rows)
        avg_put = sum(r["put_oi"] for r in rows) / len(rows)

        call_ratio = (current.call_oi / avg_call) if avg_call > 0 else 1.0
        put_ratio = (current.put_oi / avg_put) if avg_put > 0 else 1.0

        # Pick whichever side burst — prefer the larger ratio
        if call_ratio >= _BURST_MULTIPLE and call_ratio >= put_ratio:
            return OptionsBurst(
                side="call",
                ratio=float(call_ratio),
                current_oi=current.call_oi,
                baseline_avg_oi=int(avg_call),
                baseline_days=len(rows),
            )
        if put_ratio >= _BURST_MULTIPLE:
            return OptionsBurst(
                side="put",
                ratio=float(put_ratio),
                current_oi=current.put_oi,
                baseline_avg_oi=int(avg_put),
                baseline_days=len(rows),
            )
        return None
