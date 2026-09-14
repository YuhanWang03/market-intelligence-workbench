"""Macro release calendar — hardcoded 2026 schedule.

⚠️ This file is regenerated once per year (typically in January) by
running :mod:`v2.macro._seed_calendar` against the FRED API. The
auto-generated block below is wrapped in clearly-marked sentinels so
the next refresh can splice in updated dates without touching the
surrounding plumbing.

Sources used by the seeder:
- BLS schedule: https://www.bls.gov/schedule/news_release/
  (CPI / NFP / PPI / ECI / Claims via FRED release IDs)
- BEA schedule: https://www.bea.gov/news/schedule
  (PCE / GDP / Personal Income via FRED release IDs)
- FOMC calendar: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
  (statement + SEP dates — FOMC does NOT live in FRED)

The cron path is read-only against this dict — it never hits the
internet, never re-fetches dates. That makes the macro crons
deterministic in production and trivially mockable in tests.
"""

from __future__ import annotations

import logging
from datetime import date

logger = logging.getLogger(__name__)


# ISO date the calendar was last regenerated. The staleness check warns
# at 6 months — at that point the schedule is stale enough that 1+
# release dates may have shifted by ± 1 business day.
_LAST_UPDATED = "2026-06-05"


_SOURCE_URLS = {
    "BLS": "https://www.bls.gov/schedule/news_release/",
    "BEA": "https://www.bea.gov/news/schedule",
    "Fed": "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",
}


# ===========================================================================
# AUTOGEN BLOCK START — replace via `poetry run python -m v2.macro._seed_calendar`
# Each entry: {ISO_date: [(release_type, label, source_authority)]}
# `source_authority` is one of: "BLS" / "BEA" / "Fed" / "FRED"
#
# Generated 2026-06-05 from FRED API (release IDs 10/50/54/53/46) + Fed
# Aug 9 announcement (FOMC). Window: 2026-06-05 → 2026-12-23.
# 40 entries across 33 dates: 7×{NFP, CPI, PPI, PCE, GDP}, 5×FOMC.
#
# Claims (ICSA) is deliberately NOT in this dict — Initial Jobless
# Claims releases every Thursday at 08:30 ET on a deterministic
# schedule, so the ⑯ cron uses a CronTrigger(day_of_week="thu") gate
# and calls build_claims_event() unconditionally. Putting it in the
# calendar would either require pulling the correct weekly FRED
# release_id (85, not 21 which is the Unemployment Insurance Weekly
# Claims monthly summary release we mis-mapped originally) AND a
# 52-entry calendar that buys us nothing over a weekday trigger.
# ===========================================================================

_2026_RELEASES: dict[str, list[tuple[str, str, str]]] = {
    "2026-06-05": [("NFP", "NFP (FRED release_id=50)", "BLS")],
    "2026-06-10": [("CPI", "CPI (FRED release_id=10)", "BLS")],
    "2026-06-11": [("PPI", "PPI (FRED release_id=46)", "BLS")],
    "2026-06-17": [("FOMC", "Jun FOMC + SEP", "Fed")],
    "2026-06-25": [("PCE", "PCE (FRED release_id=54)", "BEA"), ("GDP", "GDP (FRED release_id=53)", "BEA")],
    "2026-07-02": [("NFP", "NFP (FRED release_id=50)", "BLS")],
    "2026-07-14": [("CPI", "CPI (FRED release_id=10)", "BLS")],
    "2026-07-15": [("PPI", "PPI (FRED release_id=46)", "BLS")],
    "2026-07-29": [("FOMC", "Jul FOMC", "Fed")],
    "2026-07-30": [("PCE", "PCE (FRED release_id=54)", "BEA"), ("GDP", "GDP (FRED release_id=53)", "BEA")],
    "2026-08-07": [("NFP", "NFP (FRED release_id=50)", "BLS")],
    "2026-08-12": [("CPI", "CPI (FRED release_id=10)", "BLS")],
    "2026-08-13": [("PPI", "PPI (FRED release_id=46)", "BLS")],
    "2026-08-26": [("PCE", "PCE (FRED release_id=54)", "BEA"), ("GDP", "GDP (FRED release_id=53)", "BEA")],
    "2026-09-04": [("NFP", "NFP (FRED release_id=50)", "BLS")],
    "2026-09-10": [("PPI", "PPI (FRED release_id=46)", "BLS")],
    "2026-09-11": [("CPI", "CPI (FRED release_id=10)", "BLS")],
    "2026-09-16": [("FOMC", "Sep FOMC + SEP", "Fed")],
    "2026-09-30": [("PCE", "PCE (FRED release_id=54)", "BEA"), ("GDP", "GDP (FRED release_id=53)", "BEA")],
    "2026-10-02": [("NFP", "NFP (FRED release_id=50)", "BLS")],
    "2026-10-14": [("CPI", "CPI (FRED release_id=10)", "BLS")],
    "2026-10-15": [("PPI", "PPI (FRED release_id=46)", "BLS")],
    "2026-10-28": [("FOMC", "Oct FOMC", "Fed")],
    "2026-10-29": [("PCE", "PCE (FRED release_id=54)", "BEA"), ("GDP", "GDP (FRED release_id=53)", "BEA")],
    "2026-11-06": [("NFP", "NFP (FRED release_id=50)", "BLS")],
    "2026-11-10": [("CPI", "CPI (FRED release_id=10)", "BLS")],
    "2026-11-13": [("PPI", "PPI (FRED release_id=46)", "BLS")],
    "2026-11-25": [("PCE", "PCE (FRED release_id=54)", "BEA"), ("GDP", "GDP (FRED release_id=53)", "BEA")],
    "2026-12-04": [("NFP", "NFP (FRED release_id=50)", "BLS")],
    "2026-12-09": [("FOMC", "Dec FOMC + SEP", "Fed")],
    "2026-12-10": [("CPI", "CPI (FRED release_id=10)", "BLS")],
    "2026-12-15": [("PPI", "PPI (FRED release_id=46)", "BLS")],
    "2026-12-23": [("PCE", "PCE (FRED release_id=54)", "BEA"), ("GDP", "GDP (FRED release_id=53)", "BEA")],
}

# ===========================================================================
# AUTOGEN BLOCK END
# ===========================================================================


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_release_today(today_iso: str) -> list[tuple[str, str, str]]:
    """Return the list of releases scheduled for ``today_iso``.

    Empty list means "no scheduled release today" — the ⑮ cron uses
    this as the gate to decide whether to push anything.
    """
    _staleness_check()
    return list(_2026_RELEASES.get(today_iso, []))


def get_releases_in_window(
    start_iso: str, end_iso: str,
) -> dict[str, list[tuple[str, str, str]]]:
    """Return all releases between ``start_iso`` and ``end_iso``
    inclusive, keyed by date. Used by ⑰ weekly recap.

    Dates without scheduled releases are NOT included in the result;
    callers iterate the dict.
    """
    _staleness_check()
    out: dict[str, list[tuple[str, str, str]]] = {}
    for iso, entries in _2026_RELEASES.items():
        if start_iso <= iso <= end_iso:
            out[iso] = list(entries)
    return out


def is_fomc_day(today_iso: str) -> bool:
    """True iff today's release list contains a FOMC entry. ⑮ cron uses
    this to route through ``fomc_parser`` instead of ``summarizer``."""
    for rel_type, _label, _src in get_release_today(today_iso):
        if rel_type == "FOMC":
            return True
    return False


def _staleness_check() -> None:
    """Warn (not raise) when the hardcoded calendar is more than 6
    months old. The cron path continues — running with a slightly
    stale calendar is better than crashing, and the warning shows up
    in the trace for the operator to fix on the next quarterly review.
    """
    try:
        last = date.fromisoformat(_LAST_UPDATED)
    except ValueError:
        logger.warning("release_calendar._LAST_UPDATED malformed: %r", _LAST_UPDATED)
        return
    age_days = (date.today() - last).days
    if age_days > 180:
        logger.warning(
            "release_calendar.py %d days old — please rerun "
            "`python -m v2.macro._seed_calendar` and commit. Sources: %s",
            age_days, _SOURCE_URLS,
        )


def get_last_updated() -> str:
    """Inspect helper for the final-check script."""
    return _LAST_UPDATED


__all__ = [
    "get_release_today",
    "get_releases_in_window",
    "is_fomc_day",
    "get_last_updated",
]
