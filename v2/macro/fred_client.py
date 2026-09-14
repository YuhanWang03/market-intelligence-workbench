"""FRED client wrapper — Phase 4 Stage 1.

Mixed-transport wrapper:

- :func:`get_series` + :func:`get_latest_value` use the ``fredapi.Fred``
  Python wrapper (these methods exist in the package).
- :func:`get_release_dates` calls the REST endpoint directly via
  ``httpx`` because ``fredapi`` 0.5.2 doesn't expose
  ``/fred/release/dates`` — caught during prod-box ``_seed_calendar``
  run on 2026-06; see the function docstring for the FRED API contract.

Both transports share:

- Lazy init / env-var check (raises :class:`FredUnavailable` when
  ``FRED_API_KEY`` is unset).
- Bounded retry on transient errors (3 attempts, 1s linear backoff).
  FRED occasionally 502s mid-day; the cron path treats a failed
  fetch as a warning, not a fatal.
- Test seam (``client_factory`` for the fredapi path,
  ``http_get`` for the REST path).

This module does NOT cache results. The cron pulls fresh data on each
run; the dashboard backend has its own cache layer for repeated bot
queries.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Callable

import httpx

try:
    from fredapi import Fred
except ImportError:                                  # sandbox path
    Fred = None                                       # type: ignore[assignment]


logger = logging.getLogger(__name__)


_RETRY_ATTEMPTS = 3
_RETRY_BACKOFF_SEC = 1.0
_FRED_REST_BASE = "https://api.stlouisfed.org/fred"


class FredUnavailable(RuntimeError):
    """Raised when FRED_API_KEY is missing, fredapi failed to install,
    or the FRED REST API returned after exhausting retries."""


def _api_key() -> str:
    """Read FRED_API_KEY or raise FredUnavailable. Shared by both
    transports so the error message is identical."""
    key = os.environ.get("FRED_API_KEY", "").strip()
    if not key:
        raise FredUnavailable("FRED_API_KEY not set in env")
    return key


def _build_default_client() -> "Fred":
    if Fred is None:
        raise FredUnavailable("fredapi package not installed")
    return Fred(api_key=_api_key())


def _with_retry(fn: Callable, *args, **kwargs):
    """Run ``fn(*args, **kwargs)`` with linear backoff. Raises the last
    exception if all attempts fail."""
    last_exc: Exception | None = None
    for attempt in range(_RETRY_ATTEMPTS):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:                      # noqa: BLE001
            last_exc = exc
            if attempt + 1 < _RETRY_ATTEMPTS:
                time.sleep(_RETRY_BACKOFF_SEC * (attempt + 1))
                logger.info(
                    "FRED retry %d/%d after %s: %s",
                    attempt + 1, _RETRY_ATTEMPTS,
                    type(exc).__name__, exc,
                )
    assert last_exc is not None
    raise last_exc


# ---------------------------------------------------------------------------
# Public API — series (fredapi wrapper)
# ---------------------------------------------------------------------------

def get_series(
    series_id: str,
    *,
    start: str | None = None,
    end: str | None = None,
    client_factory: Callable[[], "Fred"] | None = None,
):
    """Pull a FRED series. Returns a pandas.Series (the raw fredapi
    return shape — index is date, values are floats).

    ``client_factory`` is a test seam. Default behavior uses the real
    Fred client built from the env var.
    """
    factory = client_factory or _build_default_client
    fred = factory()
    return _with_retry(
        fred.get_series, series_id,
        observation_start=start, observation_end=end,
    )


def get_latest_value(
    series_id: str,
    *,
    client_factory: Callable[[], "Fred"] | None = None,
) -> float | None:
    """Convenience: get the most recent non-NaN value of ``series_id``.
    Returns None when the series is empty (covers brand-new series IDs
    or temporarily unavailable data)."""
    series = get_series(series_id, client_factory=client_factory)
    try:
        clean = series.dropna()
    except AttributeError:
        return None
    if clean.empty:
        return None
    return float(clean.iloc[-1])


# ---------------------------------------------------------------------------
# Public API — release dates (REST direct, fredapi doesn't expose this)
# ---------------------------------------------------------------------------

def get_release_dates(
    release_id: int,
    *,
    start: str,
    end: str,
    include_no_data: bool = True,
    http_get: Callable | None = None,
) -> list[str]:
    """Return scheduled release dates (ISO strings) for a FRED release
    in the window ``[start, end]``.

    Uses the REST endpoint ``GET /fred/release/dates`` directly because
    fredapi 0.5.2 doesn't expose this method (the bug was found during
    the prod-box ``_seed_calendar`` run on 2026-06 — the wrapper raised
    ``'Fred' object has no attribute 'get_release_dates'``).

    Args:
        release_id: FRED release ID. Verified IDs (catalog 2026-06):
            10 = CPI, 50 = NFP, 54 = PCE, 53 = GDP, 46 = PPI,
            21 = Initial Claims, 34 = ECI.
        start, end: ISO date strings. The REST API uses these as
            ``realtime_start`` / ``realtime_end`` filters.
        include_no_data: if True (default), future scheduled dates
            without data yet are still returned — exactly what
            ``_seed_calendar`` needs to populate the forward calendar.
        http_get: test seam. Callable with the same signature as
            ``httpx.get`` (url, *, params, timeout). Default uses
            ``httpx.get`` with a 30-second timeout.

    Returns:
        Sorted list of ISO date strings, e.g. ``["2026-06-10",
        "2026-07-15", ...]``.

    Raises:
        FredUnavailable: when ``FRED_API_KEY`` is unset OR after 3
            retries the HTTP call still fails.
    """
    key = _api_key()
    url = f"{_FRED_REST_BASE}/release/dates"
    params = {
        "release_id": release_id,
        "api_key": key,
        "file_type": "json",
        "realtime_start": start,
        "realtime_end": end,
        "include_release_dates_with_no_data": (
            "true" if include_no_data else "false"
        ),
    }
    fetch = http_get if http_get is not None else httpx.get

    last_exc: Exception | None = None
    for attempt in range(_RETRY_ATTEMPTS):
        try:
            r = fetch(url, params=params, timeout=30.0)
            r.raise_for_status()
            data = r.json()
            return [entry["date"] for entry in data.get("release_dates", [])]
        except (httpx.HTTPError, httpx.TimeoutException) as exc:
            last_exc = exc
            if attempt + 1 < _RETRY_ATTEMPTS:
                time.sleep(_RETRY_BACKOFF_SEC * (attempt + 1))
                logger.info(
                    "FRED /release/dates retry %d/%d after %s: %s",
                    attempt + 1, _RETRY_ATTEMPTS,
                    type(exc).__name__, exc,
                )
    raise FredUnavailable(
        f"FRED /release/dates failed after {_RETRY_ATTEMPTS} retries: {last_exc}"
    )


__all__ = [
    "FredUnavailable",
    "get_series",
    "get_latest_value",
    "get_release_dates",
]
