"""Launch the scheduler — keep this terminal open for it to keep running.

Usage:
    poetry run python scripts/run_scheduler.py            # normal mode
    poetry run python scripts/run_scheduler.py --test     # run all jobs once and exit

The scheduler runs in US Eastern time and triggers:
  • ① Daily Screen        — Mon-Fri 17:30 ET
  • ② Anomaly Monitor     — Mon-Fri 17:35 ET
  • ③ Lateral Expansion   — Mondays 18:00 ET
  • ④ Institutional 13F   — Tue/Fri 18:00 ET
  • ④b 13F Backfill       — Sundays 18:30 ET
  • ⑤ ETF Daily Snapshot  — Mon-Fri 17:00 ET

Closing this terminal will stop the scheduler. To run "always-on", wrap
with `nssm` (Windows service) or use Windows Task Scheduler for startup.
"""

from __future__ import annotations

import logging
import sys

from dotenv import load_dotenv

from v2.scheduler import run_scheduler

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)

load_dotenv()


if __name__ == "__main__":
    test_now = "--test" in sys.argv
    run_scheduler(test_now=test_now)
