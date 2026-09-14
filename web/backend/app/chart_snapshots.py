"""Background publication of frozen holding chart data for guest mode."""

from __future__ import annotations

from datetime import datetime, timezone
from threading import RLock
from typing import Iterable
from urllib.parse import quote, urlencode

from app.public_snapshots import publish_snapshot


DEFAULT_PRICE_RANGES = ("1W", "1M", "3M", "6M", "1Y", "3Y")
_STATE_LOCK = RLock()
_STATE: dict[str, object] = {
    "status": "idle",
    "started_at": None,
    "completed_at": None,
    "symbols": [],
    "ranges": [],
    "total": 0,
    "published": 0,
    "failed": 0,
    "errors": [],
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def chart_snapshot_status() -> dict[str, object]:
    with _STATE_LOCK:
        return dict(_STATE)


def reserve_chart_snapshot_refresh(symbols: Iterable[str], ranges: Iterable[str]) -> tuple[bool, dict[str, object]]:
    normalized_symbols = list(dict.fromkeys(str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()))
    normalized_ranges = list(dict.fromkeys(str(range_key).strip().upper() for range_key in ranges if str(range_key).strip()))
    with _STATE_LOCK:
        if _STATE["status"] == "running":
            return False, dict(_STATE)
        _STATE.update({
            "status": "running",
            "started_at": _utc_now(),
            "completed_at": None,
            "symbols": normalized_symbols,
            "ranges": normalized_ranges,
            "total": len(normalized_symbols) * len(normalized_ranges),
            "published": 0,
            "failed": 0,
            "errors": [],
        })
        return True, dict(_STATE)


def publish_holding_chart_snapshots(symbols: Iterable[str], ranges: Iterable[str]) -> None:
    """Fetch sequentially to avoid a burst of provider requests on refresh."""
    from app.routers.portfolio import _fetch_price_history

    normalized_symbols = list(dict.fromkeys(str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()))
    normalized_ranges = list(dict.fromkeys(str(range_key).strip().upper() for range_key in ranges if str(range_key).strip()))
    errors: list[dict[str, str]] = []
    published = 0
    failed = 0
    try:
        for symbol in normalized_symbols:
            for range_key in normalized_ranges:
                try:
                    payload = _fetch_price_history(symbol, range_key, False)
                    query = urlencode({"range": range_key, "extended": "false"})
                    publish_snapshot(f"/api/price-history/{quote(symbol, safe='.-')}?{query}", payload)
                    published += 1
                except Exception as exc:  # Keep the remaining symbols/ranges publishable.
                    failed += 1
                    if len(errors) < 25:
                        errors.append({"symbol": symbol, "range": range_key, "error": str(exc)[:300]})
                finally:
                    with _STATE_LOCK:
                        _STATE.update({"published": published, "failed": failed, "errors": list(errors)})
    finally:
        with _STATE_LOCK:
            _STATE.update({
                "status": "completed" if failed == 0 else "completed_with_errors",
                "completed_at": _utc_now(),
                "published": published,
                "failed": failed,
                "errors": list(errors),
            })
