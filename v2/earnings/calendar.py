"""Forward earnings calendar via yfinance.

FD has no forward calendar — verified in Stage 0. yfinance ``Ticker.calendar``
and ``Ticker.get_earnings_dates`` are the only source we use for "when is
$TICKER next reporting".

Design constraints (from caveat handling):

1. **Tolerant** — a single ticker's failure must not stop the batch. yfinance
   is scraped under the hood and routinely returns ``{}`` for a transient
   yahoo error. Treat that as "skip this ticker today, retry tomorrow" — not
   a crash, not a ``notify_on_error`` page.

2. **No persistence** — companies amend their release date. Each cron run
   re-fetches the calendar from scratch; we never cache a date overnight.

3. **Ticker filter** — yfinance coverage is patchy for non-US tickers.
   Accept ``^[A-Z]{1,5}$`` plus a small US-class-share whitelist
   (BRK.A/B, BF.A/B). Everything else is silently skipped.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any

import yfinance as yf

from v2.earnings.models import EarningsEvent, When
from v2.observability.trace import emit

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ticker filter
# ---------------------------------------------------------------------------

_PLAIN_TICKER_RE = re.compile(r"^[A-Z]{1,5}$")

# US dual-class structures that contain dots — explicitly allowed.
_CLASS_SHARE_WHITELIST: frozenset[str] = frozenset({
    "BRK.A", "BRK.B",
    "BF.A", "BF.B",
    "GOOGL", "GOOG",          # already plain, included for completeness
    "LEN.B", "MOG.A", "MOG.B",
    "PBR.A",
})


def is_supported_ticker(ticker: str) -> bool:
    """True iff yfinance is expected to cover this ticker's calendar.

    Anything with ``.`` or ``-`` is rejected unless explicitly whitelisted —
    they're typically foreign ADRs, dual-class non-US, or preferred shares
    that yfinance returns stale/empty calendars for.
    """
    if not ticker:
        return False
    t = ticker.upper()
    if t in _CLASS_SHARE_WHITELIST:
        return True
    if t in KNOWN_ETFS or _remembered_empty(t):
        # An ETF has no earnings; yfinance answers its calendar request with
        # a 404 (logged as an error) on every holding, every run.
        return False
    return bool(_PLAIN_TICKER_RE.match(t))


#: Funds that show up in portfolios and watchlists; none of them report earnings.
KNOWN_ETFS = frozenset({
    "SPY", "IVV", "VOO", "VTI", "QQQ", "DIA", "IWM", "SMH", "SOXX", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB", "XLRE", "XLC",
    "ARKK", "ARKW", "ARKG", "ARKQ", "ARKF", "ARKX", "TQQQ", "SQQQ", "SOXL", "SOXS", "NUGT", "DUST", "GLD", "SLV", "TLT", "IEF", "SHY", "HYG", "LQD",
    "VXX", "UVXY", "VIXY", "VIXM", "EEM", "EFA", "VEA", "VWO", "BND", "AGG", "USO", "UNG", "XBI", "IBB", "KWEB", "FXI",
})
_EMPTY_CALENDAR: dict[str, "datetime"] = {}
_EMPTY_CALENDAR_TTL_HOURS = 24


def _remembered_empty(ticker: str) -> bool:
    seen = _EMPTY_CALENDAR.get(ticker)
    if seen is None:
        return False
    if datetime.now(timezone.utc) - seen > timedelta(hours=_EMPTY_CALENDAR_TTL_HOURS):
        _EMPTY_CALENDAR.pop(ticker, None)
        return False
    return True


def remember_empty_calendar(ticker: str) -> None:
    """A ticker whose calendar came back empty or 404 is not asked again for a day."""

    _EMPTY_CALENDAR[ticker.upper()] = datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Batch result bookkeeping
# ---------------------------------------------------------------------------

@dataclass
class CalendarBatchResult:
    """Outcome of a batch fetch — used for the operator's trace summary."""

    events: dict[str, EarningsEvent]    # ticker → event (only successes)
    skipped_unsupported: list[str]      # rejected by ticker filter
    skipped_empty: list[str]            # yfinance returned no calendar
    errors: list[str]                   # exception during fetch


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_upcoming(ticker: str) -> EarningsEvent | None:
    """Fetch the next earnings release for one ticker. None on any failure.

    Soft failure semantics — never raises. Caller can treat None as "skip".
    """
    if not is_supported_ticker(ticker):
        return None
    return _fetch_one(ticker)


def get_upcoming_batch(tickers: list[str]) -> CalendarBatchResult:
    """Fetch upcoming earnings for a batch of tickers.

    One trace ``api_call`` event is emitted per ticker attempted, plus a
    rollup ``transform`` event with the success/skip/error counts. Suitable
    for the cron path where we want a single human-readable summary.
    """
    events: dict[str, EarningsEvent] = {}
    skipped_unsupported: list[str] = []
    skipped_empty: list[str] = []
    errors: list[str] = []

    for ticker in tickers:
        if not is_supported_ticker(ticker):
            skipped_unsupported.append(ticker)
            continue
        try:
            ev = _fetch_one(ticker)
        except Exception as exc:
            # Defence in depth — _fetch_one already swallows known failures.
            logger.warning("calendar fetch crashed for %s: %s", ticker, exc)
            errors.append(f"{ticker}:{exc}")
            continue
        if ev is None:
            skipped_empty.append(ticker)
            remember_empty_calendar(ticker)
        else:
            events[ticker] = ev

    emit(
        "transform",
        op="earnings_calendar_batch",
        ok=len(events),
        empty=len(skipped_empty),
        unsupported=len(skipped_unsupported),
        errors=len(errors),
        total=len(tickers),
    )
    return CalendarBatchResult(
        events=events,
        skipped_unsupported=skipped_unsupported,
        skipped_empty=skipped_empty,
        errors=errors,
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _fetch_one(ticker: str) -> EarningsEvent | None:
    """Internal — assumes ticker has already passed is_supported_ticker."""
    emit("api_call", provider="yfinance", endpoint="calendar", ticker=ticker)
    try:
        tk = yf.Ticker(ticker)
        cal = tk.calendar
    except Exception as exc:
        logger.warning("yfinance Ticker(%s).calendar raised: %s", ticker, exc)
        return None

    if not cal:
        # yfinance returned {} — either no upcoming release on file or a
        # transient scrape failure. Both look the same; both = skip.
        return None

    raw_date = _extract_calendar_date(cal)
    if raw_date is None:
        return None
    release_iso = _to_iso(raw_date)
    if release_iso is None:
        return None

    # Don't report a past date — yfinance occasionally returns the prior
    # release that wasn't cleared yet.
    if release_iso < date.today().isoformat():
        return None

    return EarningsEvent(
        ticker=ticker,
        release_date=release_iso,
        when=_extract_when(cal),
        eps_estimate=_extract_float(cal, "Earnings Average", "EPS Estimate", "epsEstimate"),
        revenue_estimate=_extract_float(cal, "Revenue Average", "Revenue Estimate", "revenueEstimate"),
        n_analysts=None,
        source="yfinance",
    )


def _extract_calendar_date(cal: Any) -> Any:
    """Pull the earnings date out of whatever shape yfinance handed us.

    yfinance has been refactored several times — calendar comes back as
    either ``dict`` (current 0.2.x) or ``pandas.DataFrame`` (older). Be
    defensive: any path that doesn't look like a date returns None.
    """
    if isinstance(cal, dict):
        val = cal.get("Earnings Date") or cal.get("earningsDate") or cal.get("earnings_date")
        if isinstance(val, list) and val:
            return val[0]
        return val
    # Pandas DataFrame fallback (older yfinance)
    try:
        return cal.loc["Earnings Date"].iloc[0]
    except Exception:
        return None


def _to_iso(raw: Any) -> str | None:
    """Coerce yfinance's date-ish value to an ISO ``YYYY-MM-DD`` string."""
    if raw is None:
        return None
    if isinstance(raw, str):
        # Sometimes already an ISO string
        try:
            return date.fromisoformat(raw[:10]).isoformat()
        except ValueError:
            return None
    if isinstance(raw, datetime):
        return raw.date().isoformat()
    if isinstance(raw, date):
        return raw.isoformat()
    # pandas.Timestamp duck-types as having .date()
    to_date = getattr(raw, "date", None)
    if callable(to_date):
        try:
            d = to_date()
            if isinstance(d, date):
                return d.isoformat()
        except Exception:
            return None
    return None


def _extract_when(cal: Any) -> When:
    """Heuristic for BMO/AMC. yfinance rarely populates this — usually unknown."""
    if not isinstance(cal, dict):
        return "unknown"
    hint = cal.get("Earnings Call Time") or cal.get("earningsCallTime")
    if isinstance(hint, str):
        h = hint.lower()
        if "before" in h or "bmo" in h or "premarket" in h:
            return "bmo"
        if "after" in h or "amc" in h or "postmarket" in h:
            return "amc"
    return "unknown"


def _extract_float(cal: Any, *keys: str) -> float | None:
    if not isinstance(cal, dict):
        return None
    for k in keys:
        v = cal.get(k)
        if v is None:
            continue
        if isinstance(v, (int, float)):
            return float(v)
        try:
            return float(v)
        except (TypeError, ValueError):
            continue
    return None
