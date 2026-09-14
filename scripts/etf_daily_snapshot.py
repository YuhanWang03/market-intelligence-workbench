"""Daily ARK ETF holdings snapshot — runs after US market close.

For each ARK fund, fetches the latest CSV, diffs against yesterday,
saves to data/etf.db, and renders the dashboard-feed card. One archive
push per fund so each card shows a focused trace.

Telegram-quiet by design: no notifier.send_* call. The dashboard feed
picks it up via direct Archive.save_text.

Triggered by the scheduler Mon-Fri 17:00 ET — ARK posts daily files
around 16:00 ET, giving us an hour of buffer.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv

from v2.archive import Archive
from v2.etf import (
    SUPPORTED_FUNDS,
    compute_daily_changes,
    fetch_holdings,
    get_latest_snapshot_before,
    save_snapshot,
)
from v2.observability import capture_trace_with_framing, install_all
from v2.reporting import format_etf_snapshot, notify_on_error
from v2.reporting.priority import compute_importance

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


@notify_on_error("etf-daily")
def main() -> int:
    load_dotenv()
    install_all()

    archive = Archive(agent="etf")
    total_funds = 0
    total_rows = 0

    for symbol in SUPPORTED_FUNDS:
        logger.info("Processing %s ...", symbol)
        # One push per fund. capture_trace_with_framing emits
        # session_start / intent_classified / module_enter on entry and
        # module_exit / session_end on exit, so each card's saved trace
        # renders identically to a dashboard query.
        with capture_trace_with_framing(
            agent="etf", intent="etf_view",
            text=f"(自动推送) {symbol} 每日持仓",
            responder_name="_r_etf_snapshot",
        ) as trace:
            try:
                holdings, snap_date = fetch_holdings(symbol)
            except Exception as exc:
                logger.warning("  %s fetch failed: %s", symbol, exc)
                continue

            if not holdings:
                logger.warning("  %s returned no holdings", symbol)
                continue

            # Diff against yesterday's snapshot (if any). The
            # compute_daily_changes() emit on the v2 side fires
            # transform/etf_diff inside the trace context. When there's
            # NO baseline (first run, or yesterday's row got cleaned up)
            # we emit an explicit skipped event so the "对比" pipeline
            # pill still has something to highlight.
            from v2.observability import emit as _emit
            daily_changes = None
            try:
                prev = get_latest_snapshot_before(symbol, snap_date)
                if prev:
                    daily_changes = compute_daily_changes(prev, holdings)
                else:
                    _emit(
                        "transform", op="etf_diff",
                        etf=symbol, skipped=True,
                        reason="no_baseline_snapshot",
                        note=f"{symbol} 首次入库或昨日数据缺失，跳过 diff",
                    )
            except Exception as exc:
                logger.warning("  %s diff failed: %s", symbol, exc)

            # Persist today's snapshot. save_snapshot's emit fires
            # db_write inside the trace context.
            try:
                n = save_snapshot(symbol, snap_date, holdings)
                logger.info("  %s · %s · %d rows saved", symbol, snap_date, n)
                total_funds += 1
                total_rows += n
            except Exception as exc:
                logger.warning("  %s save failed: %s", symbol, exc)
                continue

            # Render the card. format_etf_snapshot's emit fires
            # render/etf_snapshot inside the trace context.
            text = format_etf_snapshot(
                symbol, holdings, snap_date,
                top_n=15, daily_changes=daily_changes,
            )

            # Explicit chat_message so the saved trace has a complete
            # reply event. capture_trace_with_framing will then emit
            # module_exit + session_end on context exit.
            trace.emit("chat_message", role="bot", text=text[:500])

        # `with` block has exited — module_exit + session_end are now
        # in trace.events. Persist a dashboard-only card (no Telegram).
        # Priority P2 by default; the daily digest cron rolls these up.
        priority = compute_importance("etf_daily", {})
        archive.save_text(
            text,
            tickers=[symbol],
            trace_json=json.dumps(trace.events, ensure_ascii=False) if trace.events else None,
            title=f"ARK {symbol} 每日持仓",
            expires_at=(datetime.now(timezone.utc) + timedelta(days=2)).isoformat(),
            importance_score=priority.score,
            priority_tier=priority.tier,
            priority_reasons=",".join(priority.reasons),
        )

    logger.info(
        "Done. %d funds · %d position-rows saved.",
        total_funds, total_rows,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
