"""edgartools wrapper for 8-K and Form 4 filings.

Reuses the ``set_identity()`` pattern from v2/institutional/client.py
(SEC requires a real User-Agent string; both modules call into the
same process-wide edgartools identity, set once on first import).

Two correctness pins from Stage 0 verification:

1. **edgartools is uniformly 403-blocked from sandbox IPs**. Production
   prod box with a real residential / cloud IP works fine. Sandbox unit
   tests inject a fake client via ``edgar_client=`` parameter.

2. **SEC EDGAR rate limit is 10 req/s shared across all clients**. We
   throttle at 200 ms between requests (5 req/s, half the limit, leaves
   headroom for the dashboard's /api/explain calls that also hit EDGAR).

Soft-failure contract:
- Unknown CIK (new IPO ticker not yet in SEC's database) → ``[]``, no raise
- EDGAR 503 / 429 / 404 → ``[]`` + warning collected by caller
- Network timeout → ``[]`` + warning
- Malformed filing rows → skipped one-by-one, batch continues
"""

from __future__ import annotations

import logging
import os
import threading
import time

from edgar import Company, set_identity

logger = logging.getLogger(__name__)

_IDENTITY_SET = False

# Stage 0 spec: 200ms between requests = 5 req/s, half SEC's 10 req/s
# limit. Headroom for concurrent dashboard /api/explain calls.
_THROTTLE_SEC = 0.2
_THROTTLE_LOCK = threading.Lock()
_LAST_REQUEST_TS = 0.0


def _ensure_identity() -> None:
    """Set SEC identity once per process — required by SEC's fair-use policy."""
    global _IDENTITY_SET
    if _IDENTITY_SET:
        return
    identity = os.environ.get(
        "EDGAR_IDENTITY",
        "Market Intelligence Workbench contact@example.com",
    )
    set_identity(identity)
    _IDENTITY_SET = True


def _throttle() -> None:
    """Sleep so the next request lands ≥ _THROTTLE_SEC after the last one.

    Thread-safe — multiple cron scripts could share this process if a
    future deploy consolidates them, and the dashboard backend already
    runs independently against EDGAR for explain-event lookups.
    """
    global _LAST_REQUEST_TS
    with _THROTTLE_LOCK:
        elapsed = time.time() - _LAST_REQUEST_TS
        if elapsed < _THROTTLE_SEC:
            time.sleep(_THROTTLE_SEC - elapsed)
        _LAST_REQUEST_TS = time.time()


def get_recent_filings(
    ticker: str,
    form: str,
    since_iso: str,
    until_iso: str,
) -> list:
    """Fetch filings for ``ticker`` of type ``form`` filed in [since, until].

    Args:
        ticker: US equity ticker (e.g. ``"AAPL"``). edgartools resolves
            CIK via ``Company(ticker)``; unknown ticker returns ``[]``.
        form: SEC form code, e.g. ``"8-K"`` or ``"4"``. Amendments
            (8-K/A, 4/A) are included automatically by edgartools when
            ``amendments=True`` (its default).
        since_iso: inclusive lower bound, ISO date.
        until_iso: inclusive upper bound, ISO date.

    Returns:
        List of edgartools Filing objects. Empty list on:
        - unknown ticker / CIK lookup failure
        - EDGAR HTTP error (5xx / 4xx)
        - any other exception during fetch

    Never raises — cron path needs guaranteed return.
    """
    _ensure_identity()
    _throttle()

    try:
        co = Company(ticker)
    except Exception as exc:
        # Unknown ticker / IPO not yet indexed / network blip
        logger.warning("SEC company lookup failed for %s: %s", ticker, exc)
        return []

    try:
        filings = co.get_filings(
            form=form,
            filing_date=(since_iso, until_iso),
        )
    except Exception as exc:
        logger.warning(
            "SEC get_filings(%s, %s) failed for %s: %s",
            form, (since_iso, until_iso), ticker, exc,
        )
        return []

    if filings is None:
        return []

    try:
        # edgartools EntityFilings is iterable. Coerce defensively in
        # case the SDK changes return type.
        return list(filings)
    except Exception as exc:
        logger.warning(
            "SEC filings iteration failed for %s: %s", ticker, exc,
        )
        return []
