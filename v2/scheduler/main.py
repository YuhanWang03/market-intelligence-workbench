"""APScheduler entry — configure jobs and start the blocking scheduler.

When started, pushes a Telegram message listing each job's next run time
so you know it's alive.

Scope (post-simplification): only the daily-push CORE runs on a schedule —
② Anomaly, ⑦/⑧ Earnings, ⑨ Portfolio Risk, plus 📋 P2 digest and ⑥ archive
cleanup infra. Everything else was demoted to on-demand bot queries; see the
DEMOTED note at the end of build_scheduler().
"""

from __future__ import annotations

import logging
from datetime import datetime
from zoneinfo import ZoneInfo

from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from v2.scheduler.jobs import (
    anomaly_monitor_job,
    archive_cleanup_job,
    earnings_reminders_job,
    earnings_summaries_job,
    p2_digest_job,
    persona_forward_backfill_job,
    portfolio_risk_job,
)

logger = logging.getLogger(__name__)

# All cron triggers run in US Eastern (where NYSE/NASDAQ live).
_TZ = ZoneInfo("US/Eastern")


def build_scheduler() -> BlockingScheduler:
    """Configure jobs without starting. Returns the scheduler ready to .start().

    Only the daily-push core is scheduled now (6 jobs). Everything else moved
    to on-demand bot queries — see the DEMOTED note at the bottom.
    """
    scheduler = BlockingScheduler(
        timezone=_TZ,
        executors={"default": ThreadPoolExecutor(max_workers=8)},
    )

    # ② Anomaly Monitor — 17:35 ET Mon-Fri. Post-close anomaly detect +
    # Tavily multi-source attribution. The highest-signal daily push.
    scheduler.add_job(
        anomaly_monitor_job,
        CronTrigger(hour=17, minute=35, day_of_week="mon-fri", timezone=_TZ),
        id="anomaly_monitor",
        name="② Anomaly Monitor",
        misfire_grace_time=3600,
        coalesce=True,
    )

    # ⑦ Earnings reminders — 08:00 ET Mon-Fri. watchlist + holdings that land
    # in D-3 / D-1 / D-0.
    scheduler.add_job(
        earnings_reminders_job,
        CronTrigger(hour=8, minute=0, day_of_week="mon-fri", timezone=_TZ),
        id="earnings_reminders",
        name="⑦ Earnings Reminders (Mon-Fri 08:00 ET)",
        misfire_grace_time=3600,
        coalesce=True,
    )

    # ⑧ Earnings summaries — 21:00 ET Mon-Fri. Post-release card if FD has the
    # actuals; a short pending placeholder otherwise (retried next run).
    scheduler.add_job(
        earnings_summaries_job,
        CronTrigger(hour=21, minute=0, day_of_week="mon-fri", timezone=_TZ),
        id="earnings_summaries",
        name="⑧ Earnings Summaries (Mon-Fri 21:00 ET)",
        misfire_grace_time=3600,
        coalesce=True,
    )

    # ⑨ Portfolio risk — 18:30 ET Mon-Fri. Concentration / sector exposure /
    # P&L / drawdown / 7-day earnings risk. Daily loss ≥ 5% bumps to P0.
    scheduler.add_job(
        portfolio_risk_job,
        CronTrigger(hour=18, minute=30, day_of_week="mon-fri", timezone=_TZ),
        id="portfolio_risk",
        name="⑨ Portfolio Risk (Mon-Fri 18:30 ET)",
        misfire_grace_time=3600,
        coalesce=True,
    )

    # 📋 P2 digest — 16:45 ET Mon-Fri. Rolls the day's P2 archive rows from the
    # core agents into one card so low-priority items don't spam.
    scheduler.add_job(
        p2_digest_job,
        CronTrigger(hour=16, minute=45, day_of_week="mon-fri", timezone=_TZ),
        id="p2_digest",
        name="📋 P2 Digest (Mon-Fri 16:45 ET)",
        misfire_grace_time=1800,
        coalesce=True,
    )

    # ⑥ Archive cleanup — 02:00 ET daily. Sweeps dashboard-feed rows past their
    # 2-day TTL so the archive doesn't grow unbounded.
    scheduler.add_job(
        archive_cleanup_job,
        CronTrigger(hour=2, minute=0, timezone=_TZ),
        id="archive_cleanup",
        name="⑥ Archive Cleanup",
        misfire_grace_time=3600,
        coalesce=True,
    )

    # ⑯ Persona forward-return backfill — 02:30 ET daily, infra, no push.
    # Scores past 投资人委员会 votes against realised 1m / 3m returns so the
    # Lab scoreboard has look-ahead-free hit rates. No-op until votes exist.
    scheduler.add_job(
        persona_forward_backfill_job,
        CronTrigger(hour=2, minute=30, timezone=_TZ),
        id="persona_forward_backfill",
        name="⑯ Persona Forward Backfill",
        misfire_grace_time=3600,
        coalesce=True,
    )

    # ------------------------------------------------------------------
    # DEMOTED to on-demand (bot pull) — no scheduled push anymore.
    # Each fetches live when queried, so nothing here feeds a DB the bot
    # depends on (/13f hits EDGAR live, /etf pulls the ARK CSV live, etc.).
    # To re-enable a push, re-add an add_job (git history has the old
    # schedules); the launcher scripts under scripts/ also still run manually.
    #   ① Daily Screen           → scripts/daily_screen_to_telegram.py (manual; no bot cmd)
    #   ③ Lateral Expansion      → /chain TICKER
    #   ④ Institutional 13F      → /13f MANAGER          (live EDGAR)
    #   ⑤ ETF Snapshot / ⑬ ARK   → /etf SYMBOL           (live ARK CSV)
    #   ⑩ Portfolio Weekly / ⑨b  → /portfolio · /pnl · /risk
    #   ⑪ SEC 8-K                → /8k TICKER            (live SEC)
    #   ⑫ SEC Form 4 / ⑫b Digest → /insiders TICKER      (live SEC)
    #   ⑭⑮⑯⑰ Macro              → /macro · /cpi · /fomc · /yields
    #   ⑱ Money-Flow Divergence  → /flow TICKER
    # ------------------------------------------------------------------
    return scheduler


def run_scheduler(test_now: bool = False) -> None:
    """Start the scheduler. Blocks until Ctrl+C.

    If *test_now* is True, run each job once immediately then exit —
    useful for verifying everything's wired up correctly.
    """
    scheduler = build_scheduler()

    if test_now:
        logger.info("Test mode: running all jobs once for verification...")
        for job in scheduler.get_jobs():
            logger.info("--- Running %s ---", job.name)
            job.func()
        logger.info("All jobs done. Exiting test mode.")
        return

    # BlockingScheduler doesn't populate Job.next_run_time until start() is
    # called — but start() blocks. Compute upcoming fires directly from each
    # trigger so we can announce them before the loop begins.
    now = datetime.now(_TZ)
    job_info: list[tuple[str, datetime | None]] = []
    for job in scheduler.get_jobs():
        try:
            next_fire = job.trigger.get_next_fire_time(None, now)
        except Exception:
            next_fire = None
        job_info.append((job.name, next_fire))

    _push_startup_message(job_info)

    logger.info("Scheduler started. Jobs:")
    for name, next_fire in job_info:
        next_str = (
            next_fire.strftime("%Y-%m-%d %H:%M 美东") if next_fire else "—"
        )
        logger.info("  • %s — next: %s", name, next_str)

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler shutting down")


def _push_startup_message(job_info: list[tuple[str, datetime | None]]) -> None:
    """Send a Telegram message with the list of next run times — best effort."""
    try:
        # Local import — we don't want scheduler module to require Telegram env
        from v2.reporting import TelegramNotifier

        notifier = TelegramNotifier()
        lines: list[str] = ["<b>🤖 Scheduler 已启动</b>", ""]
        for name, next_fire in job_info:
            next_str = (
                next_fire.strftime("%Y-%m-%d %H:%M 美东") if next_fire else "—"
            )
            lines.append(f"• {name}")
            lines.append(f"  下次: <code>{next_str}</code>")
        notifier.send_text("\n".join(lines))
    except Exception as exc:
        logger.warning("Failed to push startup notification: %s", exc)
