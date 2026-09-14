"""One-shot script — regenerate :mod:`v2.macro.release_calendar`'s
``_2026_RELEASES`` dict from the FRED API + the Fed FOMC calendar.

Usage:
    poetry run python -m v2.macro._seed_calendar

Output:
    Prints a paste-ready ``_2026_RELEASES = {...}`` block to stdout
    along with a header indicating ``_LAST_UPDATED`` and source URLs.

After running, paste the printed block into
``v2/macro/release_calendar.py`` between the
``AUTOGEN BLOCK START`` / ``AUTOGEN BLOCK END`` sentinels, update
``_LAST_UPDATED`` to today's date, and commit. The cron path does NOT
call this script.

FRED release IDs verified 2026-06 (FRED catalog):
- 10 = CPI            (BLS)
- 50 = NFP            (BLS, Employment Situation)
- 54 = PCE            (BEA, Personal Income & Outlays)
- 53 = GDP            (BEA, GDP)
- 46 = PPI            (BLS)
- 21 = Initial Claims (BLS — weekly)

FOMC dates are NOT in FRED; we hardcode them here from the official
Fed announcement (federalreserve.gov/monetarypolicy/fomccalendars.htm,
published each August for the following year).
"""

from __future__ import annotations

import os
import sys
from datetime import date, timedelta

from v2.macro.fred_client import (
    FredUnavailable,
    get_release_dates,
)


# ---------------------------------------------------------------------------
# Static FOMC schedule — copied from the Fed announcement.
# Refresh this list when the Fed publishes the next year's calendar.
# ---------------------------------------------------------------------------

FOMC_2026: list[tuple[str, str]] = [
    ("2026-06-17", "Jun FOMC + SEP"),
    ("2026-07-29", "Jul FOMC"),
    ("2026-09-16", "Sep FOMC + SEP"),
    ("2026-10-28", "Oct FOMC"),
    ("2026-12-09", "Dec FOMC + SEP"),
]


FRED_RELEASES: dict[str, tuple[int, str]] = {
    # release_type → (release_id, source_authority)
    "CPI":    (10, "BLS"),
    "NFP":    (50, "BLS"),
    "PCE":    (54, "BEA"),
    "GDP":    (53, "BEA"),
    "PPI":    (46, "BLS"),
    # Initial Jobless Claims is NOT seeded into the calendar. ICSA
    # releases every Thursday 08:30 ET — the ⑯ cron uses a
    # CronTrigger(day_of_week="thu") gate instead of a calendar lookup.
    # Adding 52 weekly entries buys no value over a weekday trigger,
    # and the FRED release_id we originally tried (21) was the wrong
    # one — it's the monthly Unemployment Insurance Weekly Claims
    # summary, not the weekly ICSA itself (release_id 85 if we ever
    # decide we do want it in the calendar).
}


def _today_iso() -> str:
    return date.today().isoformat()


def _seed(start_iso: str, end_iso: str) -> dict[str, list[tuple[str, str, str]]]:
    calendar: dict[str, list[tuple[str, str, str]]] = {}

    # FRED-sourced monthly + weekly releases (REST direct — fredapi
    # wrapper does NOT expose this endpoint; see fred_client.py for
    # the rationale).
    for name, (rid, source) in FRED_RELEASES.items():
        try:
            dates = get_release_dates(
                rid, start=start_iso, end=end_iso,
                include_no_data=True,
            )
        except FredUnavailable as exc:
            print(f"# FATAL: {exc}", file=sys.stderr)
            sys.exit(1)
        except Exception as exc:                       # noqa: BLE001
            print(f"# WARNING {name} (release_id={rid}): {exc}", file=sys.stderr)
            continue

        for iso in dates:
            # REST returns ISO strings already; defensive str() in case
            # a future wrapper change yields a date object.
            iso_str = iso if isinstance(iso, str) else str(iso)
            label = f"{name} (FRED release_id={rid})"
            calendar.setdefault(iso_str, []).append((name, label, source))

    # FOMC — hardcoded (not in FRED).
    for iso, label in FOMC_2026:
        calendar.setdefault(iso, []).append(("FOMC", label, "Fed"))

    return calendar


def _print_paste_block(calendar: dict[str, list[tuple[str, str, str]]]) -> None:
    print(f"# Generated {_today_iso()} from FRED + Fed announcement.")
    print(f"# Update _LAST_UPDATED = \"{_today_iso()}\" after pasting.")
    print()
    print("_2026_RELEASES: dict[str, list[tuple[str, str, str]]] = {")
    for iso in sorted(calendar.keys()):
        entries = calendar[iso]
        if len(entries) == 1:
            rel_type, desc, source = entries[0]
            print(
                f'    "{iso}": [("{rel_type}", "{desc}", "{source}")],'
            )
        else:
            inner = ", ".join(
                f'("{rt}", "{desc}", "{src}")' for rt, desc, src in entries
            )
            print(f'    "{iso}": [{inner}],')
    print("}")


def main() -> int:
    today = date.today()
    end = today + timedelta(days=240)                  # ~8 months forward
    print(
        f"# Pulling FRED + FOMC schedule from {today.isoformat()} to "
        f"{end.isoformat()}...",
        file=sys.stderr,
    )
    if not os.environ.get("FRED_API_KEY"):
        print("# FATAL: FRED_API_KEY not set in env.", file=sys.stderr)
        return 1
    calendar = _seed(today.isoformat(), end.isoformat())
    print(f"# Got {sum(len(v) for v in calendar.values())} release entries "
          f"across {len(calendar)} dates.", file=sys.stderr)
    _print_paste_block(calendar)
    return 0


if __name__ == "__main__":
    sys.exit(main())
