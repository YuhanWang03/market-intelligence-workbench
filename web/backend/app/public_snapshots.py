"""Frozen, read-only API responses used by the public guest experience."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from urllib.parse import parse_qsl, urlencode, urlsplit

from app.config import SETTINGS


_LOCK = RLock()
_MAX_PAYLOAD_BYTES = 16 * 1024 * 1024

_EXACT_PATHS = {
    "/api/portfolio",
    "/api/risk",
    "/api/tickertape",
    "/api/activity",
    "/api/macro",
    "/api/history",
    "/api/flow_status",
    "/api/recommendations",
    "/api/monitoring/universe",
    "/api/watchlist",
    "/api/price-alerts",
    "/api/costs",
    "/api/lab/screening/criteria",
    "/api/lab/universes",
    "/api/lab/signals",
    "/api/lab/runs",
    "/api/lab/committee/personas",
    "/api/lab/committee/runs",
    "/api/lab/committee/scoreboard",
    "/api/lab/committee/pricing",
    "/api/research/results/NVDA",
    "/api/research/history/NVDA",
    "/api/research/history/NVDA/compare",
    "/api/research/peers/NVDA",
}


def _db_path() -> Path:
    return Path(getattr(SETTINGS, "public_snapshot_db_path", SETTINGS.repo_root / "data" / "public_snapshots.db"))


def _connect() -> sqlite3.Connection:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS public_snapshots (
            request_key TEXT PRIMARY KEY,
            request_path TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            published_at TEXT NOT NULL,
            schema_version INTEGER NOT NULL DEFAULT 1
        )
        """
    )
    return conn


def normalize_snapshot_path(raw: str) -> str:
    if not raw or len(raw) > 2048 or not raw.startswith("/api/"):
        raise ValueError("invalid snapshot path")
    parsed = urlsplit(raw)
    if parsed.scheme or parsed.netloc or parsed.fragment:
        raise ValueError("snapshot path must be local")
    query = urlencode(sorted(parse_qsl(parsed.query, keep_blank_values=True)))
    return f"{parsed.path}?{query}" if query else parsed.path


def snapshot_path_allowed(raw: str) -> bool:
    try:
        path = urlsplit(normalize_snapshot_path(raw)).path
    except ValueError:
        return False
    if path in _EXACT_PATHS:
        return True
    if path.startswith("/api/price-history/"):
        return len(path.split("/")) == 4
    if path.startswith("/api/lab/runs/"):
        return len(path.split("/")) == 5
    if path.startswith("/api/lab/committee/runs/"):
        return len(path.split("/")) == 6
    return False


def publish_snapshot(raw_path: str, payload: object) -> dict[str, str | int]:
    key = normalize_snapshot_path(raw_path)
    if not snapshot_path_allowed(key):
        raise ValueError("this endpoint cannot be published to guests")
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    size = len(encoded.encode("utf-8"))
    if size > _MAX_PAYLOAD_BYTES:
        raise ValueError("snapshot payload is too large")
    published_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with _LOCK:
        conn = _connect()
        try:
            conn.execute(
                """
                INSERT INTO public_snapshots(request_key, request_path, payload_json, published_at, schema_version)
                VALUES (?, ?, ?, ?, 1)
                ON CONFLICT(request_key) DO UPDATE SET
                    payload_json=excluded.payload_json,
                    published_at=excluded.published_at,
                    schema_version=excluded.schema_version
                """,
                (key, urlsplit(key).path, encoded, published_at),
            )
            conn.commit()
        finally:
            conn.close()
    return {"path": key, "published_at": published_at, "bytes": size}


def read_snapshot(raw_path: str) -> dict | None:
    key = normalize_snapshot_path(raw_path)
    if not snapshot_path_allowed(key):
        raise ValueError("this endpoint is not available to guests")
    with _LOCK:
        conn = _connect()
        try:
            row = conn.execute(
                "SELECT payload_json, published_at, schema_version FROM public_snapshots WHERE request_key=?",
                (key,),
            ).fetchone()
        finally:
            conn.close()
    if row is None:
        return None
    return {
        "path": key,
        "payload": json.loads(row["payload_json"]),
        "published_at": row["published_at"],
        "schema_version": row["schema_version"],
    }


def latest_snapshot_time() -> str | None:
    with _LOCK:
        conn = _connect()
        try:
            row = conn.execute("SELECT MAX(published_at) AS published_at FROM public_snapshots").fetchone()
        finally:
            conn.close()
    return str(row["published_at"]) if row and row["published_at"] else None
