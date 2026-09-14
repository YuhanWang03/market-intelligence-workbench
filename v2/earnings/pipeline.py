"""Earnings Agent orchestrator.

Two pipelines, both driven from the cron entry points in
``scripts/earnings_reminders.py`` / ``scripts/earnings_summaries.py``:

- :func:`run_reminders` — daily 08:00 ET. Pulls the watchlist, fetches the
  yfinance calendar for each ticker, and returns the set of D-3 / D-1 / D-0
  reminders to push.
- :func:`run_summaries` — daily 21:00 ET. For tickers whose calendar said
  "release today", queries FD for the actual filing. If FD has it, emits a
  full ``EarningsSummary`` (LLM-narrated). If FD hasn't ingested yet,
  emits a *pending* marker so the next morning's run can retry.

Both functions are deliberately sink-pure: they return data, they don't
push to Telegram. Stage 2's cron scripts do the push so unit tests can
exercise the pipeline without a notifier.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable, Literal

from v2.earnings import calendar as calendar_mod
from v2.earnings import historical as historical_mod
from v2.earnings import summarizer as summarizer_mod
from v2.earnings import transcript as transcript_mod
from v2.earnings.models import EarningsEvent, EarningsHistorical, EarningsSummary
from v2.observability.trace import emit

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reminders
# ---------------------------------------------------------------------------

ReminderTag = Literal["D-3", "D-1", "D-0"]


@dataclass(frozen=True)
class Reminder:
    """One reminder card to push.

    ``event`` carries the full calendar context; ``tag`` says which window
    we're in (D-3 / D-1 / D-0). The Stage-3 priority labeller maps tag →
    P0/P1/P2.
    """

    event: EarningsEvent
    tag: ReminderTag


@dataclass
class ReminderRun:
    """Result envelope for :func:`run_reminders`."""

    today: str
    reminders: list[Reminder] = field(default_factory=list)
    calendar: calendar_mod.CalendarBatchResult | None = None


def run_reminders(
    watchlist: list[str],
    *,
    today: str | None = None,
    windows: tuple[int, ...] = (3, 1, 0),
) -> ReminderRun:
    """Build the reminder set for ``today``'s 08:00 ET run.

    ``windows`` is the set of D-N values to emit. Default is the standard
    D-3 / D-1 / D-0 cadence; the test seam can pass other windows.
    """
    today_iso = today or date.today().isoformat()
    if not watchlist:
        return ReminderRun(today=today_iso)

    batch = calendar_mod.get_upcoming_batch(watchlist)

    reminders: list[Reminder] = []
    for ticker, event in batch.events.items():
        d_minus = event.d_minus(today_iso)
        if d_minus not in windows:
            continue
        tag = _tag_for_d_minus(d_minus)
        if tag is None:
            continue
        reminders.append(Reminder(event=event, tag=tag))

    emit(
        "transform",
        op="earnings_reminders",
        watchlist=len(watchlist),
        upcoming=len(batch.events),
        emitted=len(reminders),
    )
    return ReminderRun(today=today_iso, reminders=reminders, calendar=batch)


def _tag_for_d_minus(d: int) -> ReminderTag | None:
    if d == 3:
        return "D-3"
    if d == 1:
        return "D-1"
    if d == 0:
        return "D-0"
    return None


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

SummaryStatus = Literal["summarized", "pending"]


@dataclass
class SummaryOutcome:
    """One ticker's outcome from the 21:00 ET cron.

    - ``status == "summarized"`` → ``summary`` is populated; push the card.
    - ``status == "pending"`` → FD hasn't ingested yet; push a short
      "data 待落地" reminder and retry tomorrow.
    """

    ticker: str
    status: SummaryStatus
    summary: EarningsSummary | None = None
    report_period: str = ""           # for pending de-dup keying


@dataclass
class SummaryRun:
    today: str
    outcomes: list[SummaryOutcome] = field(default_factory=list)


def _fetch_recent_ten_q(
    ticker: str,
    today_iso: str,
    *,
    lookback_days: int = 200,
    fetcher: Callable | None = None,
    parser_fn: Callable | None = None,
    diff_fn: Callable | None = None,
):
    """Phase 3.5: pull most-recent 10-Q for ``ticker`` + diff vs prior.

    Returns a populated ``TenQDelta`` (with diff fields filled if a
    prior 10-Q is within lookback) or ``None``. Any sub-failure
    silently returns ``None`` — the ⑧ card section is purely additive
    and must not break the existing earnings flow when SEC is flaky.

    ``lookback_days=200`` covers ~2 quarterly cycles (90 + 90 + buffer);
    prevents stale 10-Qs from older quarters from polluting the section.

    Latency: edgartools ``filing.obj()`` is 3-8s per call. Worst case
    for ⑧'s 3-5 daily tickers × 2 lookups (current + prior) → ~30-80s
    extra. Acceptable for ⑧'s 21:00 ET slot.

    Test seams (``fetcher`` / ``parser_fn`` / ``diff_fn``) keep the
    smoke tests offline without dragging in v2.sec.client.
    """
    from datetime import date, timedelta

    if fetcher is None:
        from v2.sec import client as _sec_client
        fetcher = _sec_client.get_recent_filings
    if parser_fn is None:
        from v2.sec.ten_q_parser import parse_ten_q as _parse
        parser_fn = _parse
    if diff_fn is None:
        from v2.sec.ten_q_parser import diff_ten_q as _diff
        diff_fn = _diff

    try:
        today_d = date.fromisoformat(today_iso)
    except ValueError:
        return None
    since = (today_d - timedelta(days=lookback_days)).isoformat()
    until = today_iso

    try:
        filings = fetcher(ticker, "10-Q", since, until)
    except Exception as exc:                          # noqa: BLE001
        logger.info("_fetch_recent_ten_q: SEC fetch failed for %s: %s",
                    ticker, exc)
        return None

    if not filings:
        return None

    # filings is typically sorted descending by filing_date already
    # (edgar default). Take [0] = most recent, [1] = prior if present.
    current_delta = parser_fn(filings[0])
    if current_delta is None:
        return None

    prior_delta = None
    if len(filings) >= 2:
        prior_delta = parser_fn(filings[1])

    return diff_fn(current_delta, prior_delta)


def run_summaries(
    tickers_releasing_today: list[str],
    fd: Any,
    *,
    today: str | None = None,
    already_summarized: set[tuple[str, str]] | None = None,
    summarize_fn: Callable[..., tuple[dict[str, str], int]] | None = None,
    transcript_fn: Callable[..., transcript_mod.TranscriptHit | None] | None = None,
    ten_q_fn: Callable[..., object] | None = None,
) -> SummaryRun:
    """Emit post-release summaries for tickers that had a release today.

    Args:
        tickers_releasing_today: tickers whose calendar said today.
        fd: FDClient-shaped object (real or Mock).
        today: ISO date — defaults to today.
        already_summarized: set of ``(ticker, report_period)`` keys already
            pushed. Lets the cron skip duplicates after a retry / restart.
        summarize_fn: test seam for the LLM call. Defaults to
            ``summarizer.summarize``.
        transcript_fn: test seam for Tavily lookup. Defaults to
            ``transcript.find_transcript``.

    The cron in Stage 2 is expected to record successes into the
    ``earnings_summarized`` archive table so subsequent runs can populate
    ``already_summarized``.
    """
    today_iso = today or date.today().isoformat()
    seen = already_summarized or set()
    summarize_fn = summarize_fn or summarizer_mod.summarize
    transcript_fn = transcript_fn or transcript_mod.find_transcript

    outcomes: list[SummaryOutcome] = []
    summarized_count = 0
    pending_count = 0

    for ticker in tickers_releasing_today:
        latest = historical_mod.get_latest_actual(fd, ticker)
        if latest is None or not latest.has_quarterly_data:
            outcomes.append(
                SummaryOutcome(ticker=ticker, status="pending", report_period="")
            )
            pending_count += 1
            continue

        key = (ticker, latest.report_period)
        if key in seen:
            # Already shipped earlier today (retry path). Skip silently.
            continue

        recent = historical_mod.get_recent(fd, ticker, limit=4)
        hit = None
        try:
            hit = transcript_fn(ticker, latest.report_period)
        except Exception as exc:
            logger.warning("transcript lookup raised for %s: %s", ticker, exc)

        snippet = hit.snippet if hit else None
        blurb, _tokens = summarize_fn(ticker, latest, recent=recent, transcript_snippet=snippet)

        # Phase 3.5 — optional 10-Q delta. Silent skip on SEC failure
        # so a transient EDGAR outage doesn't block the earnings card.
        try:
            ten_q_delta = (
                ten_q_fn(ticker, today_iso) if ten_q_fn is not None
                else _fetch_recent_ten_q(ticker, today_iso)
            )
        except Exception as exc:                      # noqa: BLE001
            logger.warning(
                "10-Q lookup raised for %s — skipping section: %s",
                ticker, exc,
            )
            ten_q_delta = None

        summary = EarningsSummary(
            ticker=ticker,
            report_period=latest.report_period,
            filing_date=latest.filing_date,
            eps_surprise=latest.eps_surprise,
            eps_actual=latest.eps_actual,
            eps_estimate=latest.eps_estimate,
            revenue_actual=latest.revenue_actual,
            revenue_estimate=latest.revenue_estimate,
            last_4q_surprises=[
                r.eps_surprise for r in recent if r.eps_surprise != "UNKNOWN"
            ][:4],
            transcript_url=hit.url if hit else None,
            bull=blurb.get("bull", ""),
            bear=blurb.get("bear", ""),
            narrative=blurb.get("narrative", ""),
            ten_q_delta=ten_q_delta,
        )
        outcomes.append(SummaryOutcome(
            ticker=ticker,
            status="summarized",
            summary=summary,
            report_period=latest.report_period,
        ))
        summarized_count += 1

    emit(
        "transform",
        op="earnings_summaries",
        candidates=len(tickers_releasing_today),
        summarized=summarized_count,
        pending=pending_count,
    )
    return SummaryRun(today=today_iso, outcomes=outcomes)
