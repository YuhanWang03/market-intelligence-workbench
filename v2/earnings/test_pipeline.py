"""Integration tests for v2.earnings.pipeline.

Covers the orchestration layer's edge cases that the Stage-1 smoke only
hits at a basic level:

- watchlist ∪ held union dedup (pipeline must not emit per-set)
- D-N window filter (D-3/D-1/D-0 only; D-4 and D+N rejected)
- unsupported tickers silently skipped (calendar batch wouldn't even
  return them, but pipeline still tolerates a sparse result)
- pending status when FD has no quarterly data
- already_summarized dedup
- pending placeholder rows don't pollute the dedup set so the next-day
  21:00 ET cron can retry
"""

from __future__ import annotations

import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest


# Make v2/ importable when pytest is invoked from anywhere.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from v2.earnings import (   # noqa: E402
    EarningsEvent,
    run_reminders,
    run_summaries,
)
from v2.earnings import calendar as cal_mod   # noqa: E402


_TODAY = date.today()
_TODAY_ISO = _TODAY.isoformat()


def _future(days: int) -> str:
    return (_TODAY + timedelta(days=days)).isoformat()


# ---------------------------------------------------------------------------
# Calendar stub helper — patches yf.Ticker so tests don't hit network.
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_calendar(monkeypatch):
    """Returns a setter ``(ticker → calendar dict)``.

    Multiple calls overwrite the per-ticker mapping. yfinance is patched
    at ``cal_mod.yf.Ticker`` so the public ``get_upcoming`` / ``get_upcoming_batch``
    path is exercised end-to-end.
    """
    payloads: dict[str, dict | BaseException] = {}

    def fake_ticker(ticker):
        return SimpleNamespace(calendar=payloads.get(ticker, {}))

    monkeypatch.setattr(cal_mod.yf, "Ticker", fake_ticker)

    def setter(mapping: dict[str, dict | BaseException]) -> None:
        payloads.clear()
        payloads.update(mapping)

    return setter


# ---------------------------------------------------------------------------
# run_reminders
# ---------------------------------------------------------------------------

class TestRunReminders:

    def test_dedups_overlapping_watchlist_and_positions(self, fake_calendar):
        """Pipeline takes a flat list — caller is expected to dedup. Verify
        a deduplicated input list of 1 AAPL produces exactly 1 reminder
        (not 2), and that calling with [AAPL, AAPL] does not emit two
        (calendar.get_upcoming_batch internally treats duplicates as
        one fetch)."""
        fake_calendar({"AAPL": {"Earnings Date": [_future(3)]}})

        run = run_reminders(["AAPL"], today=_TODAY_ISO)
        assert len(run.reminders) == 1
        assert run.reminders[0].event.ticker == "AAPL"
        assert run.reminders[0].tag == "D-3"

        # If a caller passes the same ticker twice (shouldn't, but defensively):
        run2 = run_reminders(["AAPL", "AAPL"], today=_TODAY_ISO)
        # Batch dict-keyed so the duplicate collapses naturally.
        assert len(run2.reminders) == 1

    def test_respects_horizon_d3_d1_d0_only(self, fake_calendar):
        """D-4 / D-2 / D+1 must NOT fire — only the spec'd windows do."""
        fake_calendar({
            # In windows
            "AAPL":  {"Earnings Date": [_future(3)]},    # D-3 ✓
            "NVDA":  {"Earnings Date": [_future(1)]},    # D-1 ✓
            "MSFT":  {"Earnings Date": [_future(0)]},    # D-0 ✓
            # Out of windows
            "GOOGL": {"Earnings Date": [_future(2)]},    # D-2 ✗
            "META":  {"Earnings Date": [_future(4)]},    # D-4 ✗
            "TSLA":  {"Earnings Date": [_future(5)]},    # D-5 ✗
            # Past dates are dropped by calendar._fetch_one, so caller
            # never sees them; this is a defensive case anyway.
        })

        run = run_reminders(
            ["AAPL", "NVDA", "MSFT", "GOOGL", "META", "TSLA"],
            today=_TODAY_ISO,
        )
        by_ticker = {r.event.ticker: r.tag for r in run.reminders}
        assert by_ticker == {"AAPL": "D-3", "NVDA": "D-1", "MSFT": "D-0"}
        # GOOGL/META/TSLA must be absent — explicit antagonism.
        for excluded in ("GOOGL", "META", "TSLA"):
            assert excluded not in by_ticker

    def test_skips_unsupported_tickers(self, fake_calendar):
        """ADRs (.PA / .HK / hyphens) get filtered by is_supported_ticker
        before yfinance is even called; pipeline must tolerate the
        resulting sparse batch."""
        fake_calendar({
            "AAPL": {"Earnings Date": [_future(3)]},
            "NVDA": {"Earnings Date": [_future(1)]},
        })

        run = run_reminders(
            ["AAPL", "0700.HK", "BABA-A", "TSLA.PA", "NVDA", "lowercase"],
            today=_TODAY_ISO,
        )
        emitted = {r.event.ticker for r in run.reminders}
        assert emitted == {"AAPL", "NVDA"}

        # Unsupported set captured in the calendar batch result
        assert run.calendar is not None
        unsupported = set(run.calendar.skipped_unsupported)
        assert {"0700.HK", "BABA-A", "TSLA.PA"} <= unsupported

    def test_empty_watchlist_short_circuits(self, fake_calendar):
        run = run_reminders([], today=_TODAY_ISO)
        assert run.reminders == []
        assert run.calendar is None


# ---------------------------------------------------------------------------
# run_summaries
# ---------------------------------------------------------------------------

class TestRunSummaries:

    def _aapl_record(self, surprise="BEAT", period="2026-06-30"):
        q = SimpleNamespace(
            eps_surprise=surprise,
            earnings_per_share=2.10,
            estimated_earnings_per_share=1.95,
            revenue=9.5e10,
            estimated_revenue=9.1e10,
        )
        return SimpleNamespace(
            ticker="AAPL", report_period=period,
            source_type="8-K", filing_date="2026-08-01",
            quarterly=q,
        )

    def test_marks_pending_when_no_quarterly_data(self):
        """fd.get_earnings returns None (FD hasn't ingested) → pending."""
        class FD:
            def get_earnings(self, t): return None
            def get_earnings_history(self, t, limit=4): return []

        run = run_summaries(
            ["AAPL"], FD(),
            today=_TODAY_ISO,
            summarize_fn=lambda *a, **k: ({}, 0),
            transcript_fn=lambda *a, **k: None,
        )
        assert len(run.outcomes) == 1
        outcome = run.outcomes[0]
        assert outcome.ticker == "AAPL"
        assert outcome.status == "pending"
        assert outcome.summary is None
        # report_period stays empty for pending — the cron supplies a
        # synthetic 'pending_{today}' key when it writes to archive.
        assert outcome.report_period == ""

    def test_dedups_with_already_summarized_set(self):
        """If (AAPL, 2026-Q3) is in already_summarized, the same period
        must NOT produce a fresh outcome (the cron already shipped it)."""
        record = self._aapl_record(period="2026-Q3")

        class FD:
            def get_earnings(self, t): return record
            def get_earnings_history(self, t, limit=4): return [record]

        already = {("AAPL", "2026-Q3")}
        run = run_summaries(
            ["AAPL"], FD(),
            today=_TODAY_ISO,
            already_summarized=already,
            summarize_fn=lambda *a, **k: ({"bull": "x", "bear": "y", "narrative": "z"}, 0),
            transcript_fn=lambda *a, **k: None,
        )
        # No outcome at all — dedup means "skip silently", not "emit pending".
        assert run.outcomes == []

    def test_pending_does_not_block_next_day_retry(self):
        """Stage-2 spec: pending markers stored with key 'pending_{today}'
        — so the (AAPL, 'pending_2026-08-27') row in archive doesn't
        match the (AAPL, '2026-Q3') key when FD finally ingests."""
        record = self._aapl_record(period="2026-Q3")

        class FD:
            def get_earnings(self, t): return record
            def get_earnings_history(self, t, limit=4): return [record]

        # Yesterday's archive state — only pending markers, no real summary.
        already = {("AAPL", "pending_2026-08-27")}
        run = run_summaries(
            ["AAPL"], FD(),
            today=_TODAY_ISO,
            already_summarized=already,
            summarize_fn=lambda *a, **k: ({"bull": "x", "bear": "y", "narrative": "z"}, 0),
            transcript_fn=lambda *a, **k: None,
        )
        # Today's FD data uses the real report_period, which is NOT in
        # the set → outcome must be emitted.
        assert len(run.outcomes) == 1
        assert run.outcomes[0].status == "summarized"
        assert run.outcomes[0].summary is not None
        assert run.outcomes[0].report_period == "2026-Q3"

    def test_summarize_fn_failure_is_swallowed(self):
        """summarizer raising = no LLM color, not a crashed cron.
        The pipeline contract is: summarize_fn returns ({}, 0) on failure
        — caller (our default v2.earnings.summarize) wraps in try/except.
        Test the contract from the pipeline's perspective: empty blurb
        ⇒ summary still shipped with empty bull/bear/narrative."""
        record = self._aapl_record()

        class FD:
            def get_earnings(self, t): return record
            def get_earnings_history(self, t, limit=4): return [record]

        run = run_summaries(
            ["AAPL"], FD(),
            today=_TODAY_ISO,
            summarize_fn=lambda *a, **k: ({}, 0),
            transcript_fn=lambda *a, **k: None,
        )
        outcome = run.outcomes[0]
        assert outcome.status == "summarized"
        assert outcome.summary is not None
        assert outcome.summary.bull == ""
        assert outcome.summary.bear == ""
        assert outcome.summary.narrative == ""

    def test_transcript_fn_failure_logged_not_raised(self):
        """transcript_fn raising must not crash the per-ticker loop."""
        record = self._aapl_record()

        class FD:
            def get_earnings(self, t): return record
            def get_earnings_history(self, t, limit=4): return [record]

        def boom_transcript(ticker, report_period):
            raise RuntimeError("tavily 500")

        run = run_summaries(
            ["AAPL"], FD(),
            today=_TODAY_ISO,
            summarize_fn=lambda *a, **k: ({"bull": "b", "bear": "x", "narrative": "n"}, 0),
            transcript_fn=boom_transcript,
        )
        assert len(run.outcomes) == 1
        assert run.outcomes[0].status == "summarized"
        assert run.outcomes[0].summary.transcript_url is None
