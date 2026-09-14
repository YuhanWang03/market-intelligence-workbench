"""SEC weekly insider digest — Fri 19:15 ET (⑫b, Phase 3.5).

Aggregates ⑫ SEC Form 4 daily pushes from the past Mon-Fri week into a
single weekly summary card. Sits between ⑩ Portfolio Weekly (19:00 ET)
and ⑰ Macro Weekly Recap (19:30 ET).

Data source: title-only fallback (Phase 3.5 Stage 0 Decision 4). The
⑫ daily cron's per-A/M/F/G/C breakdown only exists in runtime memory,
NOT persisted to archive — so this digest aggregates at the
(ticker, direction) granularity reachable from ``archive.pushes.title``.
A full structured breakdown is deferred to Phase 3.5.5 (which would
add a ``form4_transactions`` table — out of scope here per Stage 0).

Priority:
- ``sec_insider_digest`` base = P2 (operator-visibility floor, same
  posture as ⑩⑰).
- Bumped to P1 when ≥3 tickers crossed the unusual threshold this week.
- Always pushes — empty weeks get a short "本周 ⑫ 推送平静" card so the
  operator sees the agent ran.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

from v2.archive import Archive
from v2.observability import capture_trace_with_framing, install_all
from v2.reporting import (
    TelegramNotifier,
    format_sec_insider_digest,
    notify_on_error,
)
from v2.reporting.priority import compute_importance
from v2.sec.insider_digest import build_weekly_digest, default_week_window

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


_TZ_ET = ZoneInfo("US/Eastern")


@notify_on_error("SEC Insider Digest")
def main() -> int:
    load_dotenv()
    install_all()

    today_iso = datetime.now(_TZ_ET).date().isoformat()
    week_start, week_end = default_week_window(today_iso)

    archive = Archive("sec")

    with capture_trace_with_framing(
        agent="sec", intent="insider_digest_view",
        text=f"(自动推送) SEC 内部人活动周报 · {week_start} → {week_end}",
        responder_name="_r_sec_insider_digest",
    ) as trace:
        digest = build_weekly_digest(archive, week_start, week_end)
        trace.emit(
            "chat_message", role="bot",
            text=(
                f"SEC 内部人周报 · {week_start}→{week_end} · "
                f"total={digest.total_push_count} · "
                f"tickers={digest.total_tickers_active} · "
                f"unusual={len(digest.unusual_tickers)}"
            ),
        )

        priority = compute_importance(
            "sec_insider_digest",
            {"unusual_ticker_count": len(digest.unusual_tickers)},
        )
        text = format_sec_insider_digest(digest)

        notifier = TelegramNotifier(archive=archive)
        try:
            notifier.send_text(
                text,
                trace=trace,
                title=f"SEC 内部人周报 · {week_start}→{week_end} · {priority.tier}",
                tickers=list(digest.unusual_tickers),
                priority=priority,
            )
        except Exception as exc:
            logger.warning("Insider digest push failed: %s", exc)
            return 1

    logger.info(
        "SEC insider digest complete: window=%s→%s tier=%s "
        "total=%d tickers=%d unusual=%d",
        week_start, week_end, priority.tier,
        digest.total_push_count, digest.total_tickers_active,
        len(digest.unusual_tickers),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
