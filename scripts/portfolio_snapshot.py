"""⑨b Positions Snapshot — Mon-Fri 16:25 ET.

Silent backend job. Pulls Alpaca's current positions, flattens them
into :class:`v2.portfolio.models.PositionFlat`, and writes a row per
holding into the ``positions_snapshot`` table. Consumed by ⑩
Portfolio Weekly (Fri 19:00 ET) to compute per-position attribution
across the rolling 5-day window.

Design notes:

- **Silent**: no Telegram push, no priority score. Archive still
  captures the trace (responder_name='_r_positions_snapshot') so the
  dashboard's auto-push feed can show "ran successfully" without
  spamming the Telegram channel.
- **Idempotent**: ``write_daily_snapshot`` uses INSERT OR REPLACE on
  (snapshot_date, ticker). A same-day rerun overwrites the prior row.
- **Time slot**: 16:25 ET. 5 minutes before ⑭ Macro Daily Snapshot
  (16:30 ET) to keep the log timeline unambiguous; serialised under
  the scheduler's thread pool. After Alpaca's 16:00 close, captures
  the EOD positions snapshot.
- **Failure mode**: Alpaca outage → ``get_flat_positions`` returns
  empty + warnings; cron logs and exits 0. Holiday days where
  positions exist but Alpaca's after-hours feed is briefly degraded
  still write whatever we got.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

from v2.archive import Archive
from v2.observability import capture_trace_with_framing, install_all
from v2.portfolio.positions import get_flat_positions
from v2.portfolio.snapshot import write_daily_snapshot
from v2.reporting import notify_on_error


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


_TZ_ET = ZoneInfo("US/Eastern")


@notify_on_error("Positions Snapshot")
def main() -> int:
    load_dotenv()
    install_all()

    archive = Archive("portfolio")
    today_iso = datetime.now(_TZ_ET).date().isoformat()

    with capture_trace_with_framing(
        agent="portfolio", intent="positions_snapshot",
        text=f"(后台) 持仓快照 · {today_iso}",
        responder_name="_r_positions_snapshot",
    ) as trace:
        try:
            positions, portfolio_value, cash, warnings = get_flat_positions()
        except Exception as exc:
            logger.warning("positions_snapshot: Alpaca fetch failed: %s", exc)
            trace.emit(
                "chat_message", role="bot",
                text=f"持仓快照失败 · {today_iso} · {type(exc).__name__}",
            )
            return 0

        if warnings:
            for w in warnings:
                logger.info("positions_snapshot warning: %s", w)

        rows_written = write_daily_snapshot(
            positions=positions,
            snapshot_date=today_iso,
            archive=archive,
        )
        trace.emit(
            "chat_message", role="bot",
            text=(
                f"持仓快照 · {today_iso} · {rows_written} 行写入 · "
                f"{len(positions)} 持仓 · 组合 ${portfolio_value:,.0f}"
            ),
        )

    logger.info(
        "positions_snapshot: %s rows=%d positions=%d portfolio_value=%.2f",
        today_iso, rows_written, len(positions), portfolio_value,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
