"""Earnings reminders cron — daily 08:00 ET.

Pulls watchlist + Alpaca-held tickers, asks yfinance for the next release
on each, and pushes a Telegram card for any ticker that lands in
D-3 / D-1 / D-0. Priorities follow the Phase 0 rubric:

    D-3 → earnings_reminder_d3 (P2, base 45)
    D-1 → earnings_reminder_d1 (P1, base 60)
    D-0 → earnings_reminder_d0 (P1, base 60)
    held position +15, watchlist +10  (handled by compute_importance)

Caveat handling (per Stage 2 plan):
- yfinance single-ticker failures stay silent — calendar.get_upcoming_batch
  already swallows them.
- No date persistence — we re-fetch every morning, so amended release
  dates self-heal next run.

Triggered by the scheduler Mon-Fri 08:00 ET.
"""

from __future__ import annotations

import logging
import sys
from datetime import date

from dotenv import load_dotenv

from v2.archive import Archive
from v2.bot import state as bot_state
from v2.earnings import Reminder, run_reminders
from v2.observability import capture_trace_with_framing, install_all
from v2.reporting import TelegramNotifier, format_earnings_reminder, notify_on_error
from v2.reporting.priority import compute_importance


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


_EVENT_KIND_BY_TAG = {
    "D-3": "earnings_reminder_d3",
    "D-1": "earnings_reminder_d1",
    "D-0": "earnings_reminder_d0",
}


def _resolve_universe() -> tuple[list[str], set[str], set[str]]:
    """Return (sorted_universe, held_tickers, watchlist_tickers).

    Held + watchlist are returned separately so the priority scorer can
    distinguish (held = +15, watchlist = +10). Alpaca being unconfigured
    or down → empty held set, not a crash.
    """
    watchlist = {row["ticker"].upper() for row in bot_state.watchlist_list()}

    held: set[str] = set()
    try:
        from v2.broker import AlpacaUnavailable, get_portfolio
        portfolio = get_portfolio()
        held = {p["symbol"].upper() for p in portfolio.get("positions", [])}
    except AlpacaUnavailable as exc:
        logger.info("Alpaca unavailable, proceeding watchlist-only: %s", exc)
    except Exception as exc:
        logger.warning("Alpaca portfolio fetch crashed: %s", exc)

    universe = sorted(watchlist | held)
    return universe, held, watchlist


def _emit_one(
    notifier: TelegramNotifier,
    trace,
    reminder: Reminder,
    *,
    is_held: bool,
    is_watchlist: bool,
) -> None:
    event_kind = _EVENT_KIND_BY_TAG[reminder.tag]
    priority = compute_importance(
        event_kind,
        {"is_held_position": is_held, "is_watchlist": is_watchlist},
    )
    text = format_earnings_reminder(
        reminder.event, tag=reminder.tag,
        is_held=is_held, is_watchlist=is_watchlist,
    )
    notifier.send_text(
        text,
        trace=trace,
        title=f"财报提醒 · {reminder.event.ticker} · {reminder.tag}",
        tickers=[reminder.event.ticker],
        priority=priority,
    )


@notify_on_error("Earnings Reminders")
def main() -> int:
    load_dotenv()
    install_all()

    universe, held, watchlist = _resolve_universe()
    if not universe:
        logger.info("Empty watchlist + holdings — nothing to remind.")
        return 0

    today_iso = date.today().isoformat()
    archive = Archive("earnings")

    with capture_trace_with_framing(
        agent="earnings", intent="earnings_view",
        text=f"(自动推送) 财报提醒扫描 · {len(universe)} 只",
        responder_name="_r_earnings_reminders",
    ) as trace:
        run = run_reminders(universe, today=today_iso)

        if not run.reminders:
            logger.info(
                "No D-3/D-1/D-0 hits across %d tickers — staying silent.",
                len(universe),
            )
            return 0

        notifier = TelegramNotifier(archive=archive)
        logger.info(
            "Pushing %d reminders (universe=%d, held=%d, watchlist=%d)",
            len(run.reminders), len(universe), len(held), len(watchlist),
        )

        for reminder in run.reminders:
            ticker = reminder.event.ticker
            is_held = ticker in held
            # Held trumps watchlist for the priority bonus.
            is_watchlist = (ticker in watchlist) and not is_held
            try:
                _emit_one(
                    notifier, trace, reminder,
                    is_held=is_held, is_watchlist=is_watchlist,
                )
            except Exception as exc:
                # Per-ticker failure must not stop the batch.
                logger.warning(
                    "reminder push failed for %s (%s): %s",
                    ticker, reminder.tag, exc,
                )

    return 0


if __name__ == "__main__":
    sys.exit(main())
