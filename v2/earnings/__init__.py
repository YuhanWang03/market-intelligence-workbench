"""Earnings Agent — watchlist/holdings calendar reminders + post-release summaries."""

from v2.earnings.calendar import (
    CalendarBatchResult,
    get_upcoming,
    get_upcoming_batch,
    is_supported_ticker,
)
from v2.earnings.historical import (
    get_latest_actual,
    get_recent,
    surprise_history,
)
from v2.earnings.models import (
    EarningsEvent,
    EarningsHistorical,
    EarningsSummary,
    EpsSurprise,
    When,
)
from v2.earnings.pipeline import (
    Reminder,
    ReminderRun,
    ReminderTag,
    SummaryOutcome,
    SummaryRun,
    SummaryStatus,
    run_reminders,
    run_summaries,
)
from v2.earnings.summarizer import summarize
from v2.earnings.transcript import TranscriptHit, find_transcript

__all__ = [
    "CalendarBatchResult",
    "EarningsEvent",
    "EarningsHistorical",
    "EarningsSummary",
    "EpsSurprise",
    "Reminder",
    "ReminderRun",
    "ReminderTag",
    "SummaryOutcome",
    "SummaryRun",
    "SummaryStatus",
    "TranscriptHit",
    "When",
    "find_transcript",
    "get_latest_actual",
    "get_recent",
    "get_upcoming",
    "get_upcoming_batch",
    "is_supported_ticker",
    "run_reminders",
    "run_summaries",
    "summarize",
    "surprise_history",
]
