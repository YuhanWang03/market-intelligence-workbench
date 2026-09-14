"""Earnings post-release summaries cron — daily 21:00 ET.

For each ticker in (watchlist ∪ Alpaca holdings) whose yfinance calendar
says "release today", asks FD whether actuals have landed. If yes, builds
an :class:`EarningsSummary` (LLM-narrated bull/bear/narrative) and pushes
a Telegram card; priority is P0 when |EPS surprise| ≥ 10% with held/watchlist
adjustments. If FD hasn't ingested yet, pushes a short "数据待落地" P2
reminder and marks the row pending — the next morning's reminder cron will
already have moved on, but the next 21:00 ET cron picks it up because the
dedup table only counts ``outcome="summarized"``.

Triggered by the scheduler Mon-Fri 21:00 ET.
"""

from __future__ import annotations

import logging
import sys
from datetime import date

from dotenv import load_dotenv

from v2.archive import Archive
from v2.bot import state as bot_state
from v2.data import CachedFDClient
from v2.earnings import (
    EarningsSummary,
    SummaryOutcome,
    get_upcoming_batch,
    run_summaries,
)
from v2.observability import capture_trace_with_framing, install_all
from v2.reporting import (
    TelegramNotifier,
    format_earnings_pending,
    format_earnings_summary,
    notify_on_error,
)
from v2.reporting.priority import compute_importance


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _resolve_universe() -> tuple[list[str], set[str], set[str]]:
    """Mirror of earnings_reminders._resolve_universe — kept inline to avoid
    a shared helper module for two callers."""
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

    return sorted(watchlist | held), held, watchlist


def _eps_surprise_pct(s: EarningsSummary) -> float | None:
    """Kept as a script-local helper purely for the priority metadata.

    The card itself computes the same number internally — but we need it
    here before constructing the card to feed into compute_importance.
    """
    if s.eps_actual is None or s.eps_estimate is None or s.eps_estimate == 0:
        return None
    return (s.eps_actual - s.eps_estimate) / abs(s.eps_estimate)


def _push_summarized(
    notifier: TelegramNotifier,
    trace,
    archive: Archive,
    outcome: SummaryOutcome,
    *,
    is_held: bool,
    is_watchlist: bool,
) -> None:
    summary = outcome.summary
    assert summary is not None
    eps_pct = _eps_surprise_pct(summary) or 0.0
    md = {
        "surprise_pct": eps_pct,
        "is_held_position": is_held,
        "is_watchlist": is_watchlist,
        # Hook left wired — when Stage 5's transcript parser detects
        # "guidance lowered" we'll set this. Stays False for now.
        "guidance_lowered": False,
    }
    # Phase 3.5 — 10-Q auditor flags surface into priority metadata so
    # going_concern / material_weakness pushes the priority +20 / +15
    # (the rule lives in v2.reporting.priority but only fires when the
    # cron forwards the flags here). Duck-typed access so we don't pull
    # in v2.sec at import time.
    tq = getattr(summary, "ten_q_delta", None)
    if tq is not None:
        if getattr(tq, "has_going_concern", False):
            md["has_going_concern"] = True
        if getattr(tq, "has_material_weakness", False):
            md["has_material_weakness"] = True
    priority = compute_importance("earnings_summary", md)

    text = format_earnings_summary(
        summary, is_held=is_held, is_watchlist=is_watchlist,
    )
    notifier.send_text(
        text,
        trace=trace,
        title=f"财报 · {summary.ticker} · {summary.eps_surprise}",
        tickers=[summary.ticker],
        priority=priority,
    )
    archive.mark_earnings_summarized(
        summary.ticker, summary.report_period, outcome="summarized",
    )


def _push_pending(
    notifier: TelegramNotifier,
    trace,
    archive: Archive,
    outcome: SummaryOutcome,
    *,
    today_iso: str,
    is_held: bool,
    is_watchlist: bool,
) -> None:
    priority = compute_importance(
        "earnings_pending",
        {"is_held_position": is_held, "is_watchlist": is_watchlist},
    )
    text = format_earnings_pending(
        outcome.ticker, today_iso=today_iso,
        is_held=is_held, is_watchlist=is_watchlist,
    )
    notifier.send_text(
        text,
        trace=trace,
        title=f"财报待落地 · {outcome.ticker}",
        tickers=[outcome.ticker],
        priority=priority,
    )
    # Pending key uses today's date as a placeholder so it doesn't collide
    # with the eventual real (ticker, report_period) row.
    archive.mark_earnings_summarized(
        outcome.ticker, f"pending_{today_iso}", outcome="pending",
    )


@notify_on_error("Earnings Summaries")
def main() -> int:
    load_dotenv()
    install_all()

    universe, held, watchlist = _resolve_universe()
    if not universe:
        logger.info("Empty watchlist + holdings — nothing to summarize.")
        return 0

    today_iso = date.today().isoformat()
    archive = Archive("earnings")

    # Filter universe to tickers whose yfinance calendar says today.
    cal_batch = get_upcoming_batch(universe)
    releasing_today = [
        t for t, ev in cal_batch.events.items()
        if ev.d_minus(today_iso) == 0
    ]

    if not releasing_today:
        logger.info(
            "No releases today across %d tickers — staying silent.",
            len(universe),
        )
        return 0

    logger.info(
        "Tickers releasing today: %s",
        ", ".join(releasing_today),
    )

    already = archive.get_summarized_set()

    with CachedFDClient() as fd:
        with capture_trace_with_framing(
            agent="earnings", intent="earnings_view",
            text=f"(自动推送) 财报总结 · {len(releasing_today)} 只",
            responder_name="_r_earnings_summaries",
        ) as trace:
            run = run_summaries(
                releasing_today, fd,
                today=today_iso,
                already_summarized=already,
            )

            notifier = TelegramNotifier(archive=archive)
            for outcome in run.outcomes:
                ticker = outcome.ticker
                is_held = ticker in held
                is_watchlist = (ticker in watchlist) and not is_held
                try:
                    if outcome.status == "summarized":
                        _push_summarized(
                            notifier, trace, archive, outcome,
                            is_held=is_held, is_watchlist=is_watchlist,
                        )
                    elif outcome.status == "pending":
                        _push_pending(
                            notifier, trace, archive, outcome,
                            today_iso=today_iso,
                            is_held=is_held, is_watchlist=is_watchlist,
                        )
                except Exception as exc:
                    logger.warning(
                        "summary push failed for %s (%s): %s",
                        ticker, outcome.status, exc,
                    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
