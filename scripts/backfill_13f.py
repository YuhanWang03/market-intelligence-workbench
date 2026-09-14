"""One-shot backfill: re-fetch 13F filings for all tracked managers and
re-save with CUSIP-aggregation enabled.

Run once after deploying the CUSIP aggregation fix so existing edgar.db rows
(which suffered from "last subsidiary wins" data loss on AAPL etc.) get
overwritten with full aggregated values.

Idempotent — re-running is safe.
Does NOT push to Telegram.
"""

from __future__ import annotations

import logging
import sys

from dotenv import load_dotenv

from v2.institutional import MANAGERS
from v2.institutional.client import fetch_recent_13f
from v2.institutional.tracker import get_db, save_filing
from v2.reporting import notify_on_error

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


@notify_on_error("13f-backfill")
def main() -> int:
    load_dotenv()
    total_filings = 0
    total_positions = 0

    for cik, manager_name in MANAGERS:
        logger.info("Fetching %s (CIK %s)...", manager_name, cik)
        try:
            recent = fetch_recent_13f(cik, manager_name, n_filings=2)
        except Exception as exc:
            logger.warning("  EDGAR fetch failed: %s", exc)
            continue

        if not recent:
            logger.warning("  No filings returned")
            continue

        with get_db() as conn:
            for filing, positions in recent:
                # Wipe stale per-accession position rows so the aggregated
                # set is the sole source of truth.
                conn.execute(
                    "DELETE FROM positions WHERE accession=?",
                    (filing.accession,),
                )
                save_filing(conn, filing, positions)
                logger.info(
                    "  Saved %s %s · %d positions · portfolio=$%.1fB",
                    filing.quarter, filing.accession[-12:],
                    len(positions), filing.portfolio_value / 1e9,
                )
                total_filings += 1
                total_positions += len(positions)

    logger.info(
        "Done. %d filings · %d positions saved.",
        total_filings, total_positions,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
