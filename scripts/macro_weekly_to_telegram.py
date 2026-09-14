"""Macro weekly recap — Fri 19:30 ET.

The seventeenth scheduled agent. Pinned at 19:30 ET by Stage 0 design
to clear ⑨ Portfolio Risk (18:30) and ⑩ Portfolio Weekly (19:00).

Recap content:
- This-week releases that already fired (from release_calendar window)
- Next-week schedule preview
- 1W deltas on VIX / DGS10 / DGS2 / T10Y2Y (Python-computed; no LLM)

Priority is always ``macro_weekly`` (P1 floor). Same posture as
⑩ Portfolio Weekly — operator visibility is the point, not
event-driven escalation. Mid-week shock signals come from ⑭/⑮ instead.

Card formatter is inline (Stage 5 lifts to v2.reporting.format_macro_weekly).
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

from v2.archive import Archive
from v2.macro import build_weekly_recap
from v2.observability import capture_trace_with_framing, install_all
from v2.reporting import (
    TelegramNotifier,
    format_macro_weekly_recap,
    notify_on_error,
)
from v2.reporting.priority import compute_importance

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


_TZ_ET = ZoneInfo("US/Eastern")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@notify_on_error("Macro Weekly Recap")
def main() -> int:
    load_dotenv()
    install_all()

    today_iso = datetime.now(_TZ_ET).date().isoformat()
    archive = Archive("macro")

    with capture_trace_with_framing(
        agent="macro", intent="macro_weekly_view",
        text=f"(自动推送) 宏观周报 · {today_iso}",
        responder_name="_r_macro_weekly",
    ) as trace:
        recap = build_weekly_recap(today_iso)
        trace.emit(
            "chat_message", role="bot",
            text=(
                f"宏观周报 · {today_iso} · "
                f"this_week={len(recap.get('this_week_releases') or {})} · "
                f"next_week={len(recap.get('next_week_releases') or {})}"
            ),
        )

        priority = compute_importance("macro_weekly", {})
        text = format_macro_weekly_recap(recap)

        notifier = TelegramNotifier(archive=archive)
        try:
            notifier.send_text(
                text,
                trace=trace,
                title=f"宏观周报 · {today_iso} · {priority.tier}",
                tickers=[],
                priority=priority,
            )
        except Exception as exc:
            logger.warning("Weekly recap push failed: %s", exc)
            return 1

    logger.info(
        "Macro weekly complete: tier=%s this_week=%d next_week=%d",
        priority.tier,
        len(recap.get('this_week_releases') or {}),
        len(recap.get('next_week_releases') or {}),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
