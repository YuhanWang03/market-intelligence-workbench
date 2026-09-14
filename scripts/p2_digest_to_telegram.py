"""Daily roll-up of P2-tier pushes from the last 24 hours.

P2 pushes (importance 40-59) don't fire Telegram messages on their own
— they go to archive.db so the dashboard auto-push feed can see them,
and this cron sweeps the digest queue once a day and sends a single
combined Telegram message. Keeps the chat from drowning in routine
data-ingestion noise while still surfacing the headline counts.

Triggered by the scheduler Mon-Fri 16:45 ET — sits just before the
17:00 ET ETF / 17:30 ET screen / 17:35 ET anomaly cron block, so it
covers everything that landed before market close.
"""

from __future__ import annotations

import logging
import sys

from dotenv import load_dotenv

from v2.archive import Archive
from v2.observability import capture_trace_with_framing, install_all
from v2.reporting import TelegramNotifier, notify_on_error
from v2.reporting.priority import compute_importance


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


_AGENT_ICON = {
    "anomaly":          "⚡",
    "institutional":    "🏛",
    "lateral":          "🕸",
    "etf":              "📈",
    "screen":           "📋",
    "alert":            "🔔",
    "intraday_anomaly": "⚡",
    "streamer":         "📡",
}


def _format_digest(pending: list[dict]) -> str:
    """Render the queued P2 rows into one Telegram message."""
    lines: list[str] = [
        f"<b>📋 今日 P2 汇总 · {len(pending)} 条</b>",
        "━━━━━━━━━━━━━━━━━━━━",
    ]
    for row in pending:
        icon = _AGENT_ICON.get(row["agent"], "•")
        ts = (row.get("ts") or "")[11:16]  # HH:MM portion of ISO
        title = row.get("title") or row.get("agent") or "(untitled)"
        lines.append(f"• {icon} [{ts}] {title}")
    lines.append("")
    lines.append(
        "<i>明细在 dashboard 自动推送页查看；P2 不再单独推送。</i>"
    )
    return "\n".join(lines)


@notify_on_error("p2-digest")
def main() -> int:
    load_dotenv()
    install_all()

    archive = Archive("p2_digest")
    pending = archive.get_pending_p2_digest()
    if not pending:
        logger.info("No P2 entries pending — staying silent.")
        return 0

    logger.info("Aggregating %d P2 entries", len(pending))

    with capture_trace_with_framing(
        agent="p2_digest", intent="summary",
        text=f"(自动汇总) 今日 P2 · {len(pending)} 条",
        responder_name="_r_p2_digest",
    ) as trace:
        text = _format_digest(pending)
        trace.emit("chat_message", role="bot", text=text[:500])

    # The digest itself is P1 — operator must see it.
    priority = compute_importance("p2_digest", {})
    notifier = TelegramNotifier(archive=archive)
    notifier.send_text(
        text,
        trace=trace,
        title=f"P2 汇总 · {len(pending)} 条",
        priority=priority,
    )

    cleared = archive.clear_p2_digest([row["id"] for row in pending])
    logger.info("Cleared %d entries from p2_digest_pending", cleared)
    return 0


if __name__ == "__main__":
    sys.exit(main())
