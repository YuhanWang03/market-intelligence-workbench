"""SQLite-backed response cache for FDClient.

Why: FD bills per API call. Multiple agents (① screening / ② monitoring /
③ lateral) often query the same ticker on the same day. Without a cache,
each agent re-fetches the same data and burns credits. With a cache, the
first call hits the API, subsequent calls within TTL hit SQLite.

Per-endpoint TTL reflects how fast each dataset changes:
    company_facts     7 days  (registration data — rarely changes)
    financial_metrics 24 hr   (TTM ratios — daily refresh max)
    earnings          24 hr   (quarterly — only changes once a quarter)
    earnings_history  24 hr
    insider_trades    6 hr    (Form 4 can land multiple times per day)
    prices            6 hr    (daily bars)

Endpoints not in TTL_HOURS are not cached (e.g. /news/ always fresh).

CachedFDClient wraps FDClient — same method signatures, zero downstream
changes required. Pure decorator pattern.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

from v2.data.client import FDClient
from v2.data.models import (
    CompanyFacts,
    CompanyNews,
    Earnings,
    EarningsRecord,
    FinancialMetrics,
    InsiderTrade,
    Price,
)

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DB_PATH = _PROJECT_ROOT / "data" / "fd_cache.db"

_TTL_HOURS: dict[str, float] = {
    "get_company_facts":    24 * 7,
    "get_financial_metrics": 24,
    "get_earnings":          24,
    "get_earnings_history":  24,
    "get_insider_trades":     6,
    "get_prices":             6,
    # "get_news" deliberately omitted → never cached (always fresh)
}

# Pydantic model class per endpoint (for deserialization)
_RESPONSE_TYPES: dict[str, type] = {
    "get_company_facts":    CompanyFacts,
    "get_financial_metrics": FinancialMetrics,
    "get_earnings":          Earnings,
    "get_earnings_history":  EarningsRecord,
    "get_insider_trades":    InsiderTrade,
    "get_prices":            Price,
    "get_news":              CompanyNews,
}

# Whether endpoint returns a list vs single object
_RETURNS_LIST: dict[str, bool] = {
    "get_company_facts":    False,
    "get_financial_metrics": True,
    "get_earnings":          False,
    "get_earnings_history":  True,
    "get_insider_trades":    True,
    "get_prices":            True,
    "get_news":              True,
}


_SCHEMA = """
CREATE TABLE IF NOT EXISTS fd_cache (
    cache_key     TEXT PRIMARY KEY,
    endpoint      TEXT NOT NULL,
    ticker        TEXT,
    response_json TEXT NOT NULL,
    expires_at    REAL NOT NULL,
    created_at    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_fd_cache_expires ON fd_cache(expires_at);
CREATE INDEX IF NOT EXISTS idx_fd_cache_endpoint ON fd_cache(endpoint);
"""


class CachedFDClient:
    """Wraps FDClient with SQLite-backed response caching."""

    def __init__(self, base: FDClient | None = None) -> None:
        self._base = base or FDClient()
        self.hits = 0
        self.misses = 0
        _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    # ------------------------------------------------------------------
    # Context manager (mirror FDClient's interface)
    # ------------------------------------------------------------------

    def __enter__(self) -> CachedFDClient:
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def close(self) -> None:
        self._base.close()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(str(_DB_PATH), timeout=10.0)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Cache primitives
    # ------------------------------------------------------------------

    def _lookup(self, endpoint: str, args: tuple, kwargs: dict) -> Any:
        """Return cached response if valid, else None."""
        if endpoint not in _TTL_HOURS:
            return None
        key = _make_key(endpoint, args, kwargs)
        with self._conn() as conn:
            row = conn.execute(
                "SELECT response_json, expires_at FROM fd_cache WHERE cache_key=?",
                (key,),
            ).fetchone()
        if row is None or row[1] < time.time():
            return None
        try:
            data = json.loads(row[0])
            return _deserialize(endpoint, data)
        except (json.JSONDecodeError, Exception) as exc:
            logger.warning("cache deserialize failed for %s: %s", endpoint, exc)
            return None

    def _store(self, endpoint: str, args: tuple, kwargs: dict, response: Any) -> None:
        ttl_hours = _TTL_HOURS.get(endpoint)
        if ttl_hours is None or response is None:
            return
        try:
            serialized = json.dumps(_serialize(endpoint, response), default=str)
        except Exception as exc:
            logger.warning("cache serialize failed for %s: %s", endpoint, exc)
            return
        key = _make_key(endpoint, args, kwargs)
        ticker = _extract_ticker(args, kwargs)
        now = time.time()
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO fd_cache
                   (cache_key, endpoint, ticker, response_json, expires_at, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (key, endpoint, ticker, serialized, now + ttl_hours * 3600, now),
            )

    def _call(self, endpoint: str, *args, **kwargs) -> Any:
        cached = self._lookup(endpoint, args, kwargs)
        if cached is not None:
            self.hits += 1
            return cached
        self.misses += 1
        result = getattr(self._base, endpoint)(*args, **kwargs)
        self._store(endpoint, args, kwargs, result)
        return result

    # ------------------------------------------------------------------
    # Wrapped methods — mirror FDClient signature 1:1
    # ------------------------------------------------------------------

    def get_prices(self, ticker, start_date, end_date, **kwargs):
        return self._call("get_prices", ticker, start_date, end_date, **kwargs)

    def get_financial_metrics(self, ticker, end_date, period="ttm", limit=10):
        return self._call(
            "get_financial_metrics", ticker, end_date, period=period, limit=limit,
        )

    def get_news(self, ticker, end_date, start_date=None, limit=1000):
        return self._call(
            "get_news", ticker, end_date, start_date=start_date, limit=limit,
        )

    def get_insider_trades(self, ticker, end_date, start_date=None, limit=1000):
        return self._call(
            "get_insider_trades", ticker, end_date,
            start_date=start_date, limit=limit,
        )

    def get_company_facts(self, ticker):
        return self._call("get_company_facts", ticker)

    def get_earnings(self, ticker):
        return self._call("get_earnings", ticker)

    def get_earnings_history(self, ticker, limit=12):
        return self._call("get_earnings_history", ticker, limit=limit)

    def get_market_cap(self, ticker, end_date):
        # Composite call — uses other cached endpoints internally, so cache by composition
        return self._base.get_market_cap(ticker, end_date)

    # ------------------------------------------------------------------
    # Stats / maintenance
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Return cache hit rate + row counts. Useful for telemetry."""
        with self._conn() as conn:
            total = conn.execute("SELECT COUNT(*) FROM fd_cache").fetchone()[0]
            live = conn.execute(
                "SELECT COUNT(*) FROM fd_cache WHERE expires_at >= ?",
                (time.time(),),
            ).fetchone()[0]
            by_endpoint = dict(conn.execute(
                """SELECT endpoint, COUNT(*) FROM fd_cache
                   WHERE expires_at >= ? GROUP BY endpoint""",
                (time.time(),),
            ).fetchall())
        attempts = self.hits + self.misses
        hit_rate = (self.hits / attempts) if attempts else 0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "rows_total": total,
            "rows_live": live,
            "by_endpoint": by_endpoint,
        }

    def purge_expired(self) -> int:
        """Drop rows whose TTL has passed. Returns rows deleted."""
        with self._conn() as conn:
            cur = conn.execute(
                "DELETE FROM fd_cache WHERE expires_at < ?",
                (time.time(),),
            )
            return cur.rowcount or 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_key(endpoint: str, args: tuple, kwargs: dict) -> str:
    """SHA1 hash of normalized (endpoint, args, kwargs) — deterministic across runs."""
    payload = json.dumps(
        [endpoint, list(args), {k: v for k, v in sorted(kwargs.items())}],
        sort_keys=True,
        default=str,
    )
    return hashlib.sha1(payload.encode()).hexdigest()


def _extract_ticker(args: tuple, kwargs: dict) -> str:
    """Pull the ticker out of (args, kwargs) — almost always the first positional arg."""
    if args:
        return str(args[0])
    return str(kwargs.get("ticker", ""))


def _serialize(endpoint: str, response: Any) -> Any:
    """Convert Pydantic models to JSON-friendly dicts."""
    if response is None:
        return None
    if _RETURNS_LIST.get(endpoint, False):
        return [item.model_dump(mode="json") for item in response]
    return response.model_dump(mode="json")


def _deserialize(endpoint: str, data: Any) -> Any:
    """Reconstruct Pydantic models from cached dicts."""
    if data is None:
        return None
    cls = _RESPONSE_TYPES.get(endpoint)
    if cls is None:
        return data
    if _RETURNS_LIST.get(endpoint, False):
        if not isinstance(data, list):
            return []
        return [cls.model_validate(item) for item in data]
    return cls.model_validate(data)
