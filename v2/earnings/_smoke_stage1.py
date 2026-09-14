"""Stage-1 smoke test for v2/earnings/{calendar,historical}.

Network is blocked in the dev sandbox, so this is intentionally offline:
- Ticker filter: pure regex, runs anywhere.
- Calendar tolerance: monkey-patches ``yf.Ticker`` to inject the failure
  shapes yfinance has been seen to return ({}, dict with a future date,
  dict with a past date, raising .calendar).
- Historical normalisation: feeds a fake FD with shapes lifted from
  v2/backtesting/test_backtest.py.

Run: ``poetry run python -m v2.earnings._smoke_stage1``
"""

from __future__ import annotations

import sys
import traceback
from datetime import date, timedelta
from types import SimpleNamespace


def _section(title: str) -> None:
    print()
    print("=" * 64)
    print(title)
    print("=" * 64)


# ---------------------------------------------------------------------------
# 1. Ticker filter
# ---------------------------------------------------------------------------

def smoke_ticker_filter() -> None:
    from v2.earnings import is_supported_ticker
    cases = [
        ("AAPL", True),
        ("NVDA", True),
        ("BRK.A", True),
        ("BRK.B", True),
        ("BF.A", True),
        ("0700.HK", False),     # foreign
        ("BABA-A", False),       # dash
        ("", False),
        ("toolongticker", False),
        ("goog", True),          # case-insensitive
    ]
    bad = []
    for tk, expected in cases:
        got = is_supported_ticker(tk)
        status = "ok" if got == expected else "FAIL"
        print(f"  {status:4s} is_supported_ticker({tk!r:<14}) = {got} (want {expected})")
        if got != expected:
            bad.append(tk)
    assert not bad, f"ticker filter regressions: {bad}"


# ---------------------------------------------------------------------------
# 2. Calendar tolerance
# ---------------------------------------------------------------------------

def smoke_calendar() -> None:
    """Use the real calendar.get_upcoming with a swapped-in yf.Ticker."""
    from v2.earnings import calendar as cal_mod

    future = (date.today() + timedelta(days=21)).isoformat()
    past = (date.today() - timedelta(days=2)).isoformat()

    class FakeTicker:
        def __init__(self, calendar_payload):
            self._cal = calendar_payload
            self._raise = isinstance(calendar_payload, BaseException)

        @property
        def calendar(self):
            if self._raise:
                raise self._cal
            return self._cal

    scenarios = {
        # canonical happy path — dict shape from yfinance 0.2.x
        "happy_amc": (
            {
                "Earnings Date": [future],
                "EPS Estimate": 1.42,
                "Revenue Estimate": 9.3e10,
                "Earnings Call Time": "after market close",
            },
            "ok",
        ),
        # empty dict — soft skip
        "empty_dict": ({}, "none"),
        # past date — silently drop, calendar is stale
        "stale_past": ({"Earnings Date": [past]}, "none"),
        # exception at attribute access — soft skip
        "raises": (RuntimeError("yahoo 500"), "none"),
        # date already an ISO string (not list)
        "iso_string": ({"Earnings Date": future}, "ok"),
    }

    bad = []
    for name, (payload, expect) in scenarios.items():
        ft = FakeTicker(payload)
        orig = cal_mod.yf.Ticker
        cal_mod.yf.Ticker = lambda _t, _ft=ft: _ft
        try:
            ev = cal_mod.get_upcoming("AAPL")
        finally:
            cal_mod.yf.Ticker = orig

        got = "ok" if ev is not None else "none"
        ok = got == expect
        line = f"  {'ok' if ok else 'FAIL':4s} scenario={name:<14} → {got}"
        if ev is not None:
            line += f"  (release={ev.release_date}, when={ev.when}, eps_est={ev.eps_estimate})"
        print(line)
        if not ok:
            bad.append(name)
    assert not bad, f"calendar scenarios failing: {bad}"


# ---------------------------------------------------------------------------
# 3. Historical normalisation
# ---------------------------------------------------------------------------

def smoke_historical() -> None:
    from v2.earnings import get_latest_actual, get_recent, surprise_history

    # Mirror the shapes from v2/backtesting/test_backtest.py — SimpleNamespace
    # duck-types EarningsRecord / EarningsData without depending on v2.data.
    def make(rp, fd_, surprise, eps_a=None, eps_e=None, rev_a=None, rev_e=None):
        q = SimpleNamespace(
            eps_surprise=surprise,
            earnings_per_share=eps_a,
            estimated_earnings_per_share=eps_e,
            revenue=rev_a,
            estimated_revenue=rev_e,
        )
        return SimpleNamespace(
            ticker="AAPL", report_period=rp, source_type="8-K",
            filing_date=fd_, quarterly=q,
        )

    canned = [
        make("2025-06-28", "2025-08-01", "BEAT", 2.10, 1.95, 9.5e10, 9.1e10),
        make("2025-03-29", "2025-05-02", "MISS", 1.30, 1.50, 8.6e10, 8.9e10),
        make("2024-12-28", "2025-02-01", "MEET", 2.40, 2.40, 1.2e11, 1.2e11),
        make("2024-09-28", "2024-11-01", None,   None, None, None,   None),  # no surprise
    ]

    class FakeFD:
        def get_earnings(self, ticker):
            return canned[0]                  # most recent

        def get_earnings_history(self, ticker, limit=4):
            return canned[:limit]

    fd = FakeFD()

    latest = get_latest_actual(fd, "AAPL")
    assert latest is not None, "latest should be non-None"
    print(f"  ok   latest: surprise={latest.eps_surprise} eps={latest.eps_actual}/"
          f"{latest.eps_estimate} pct={latest.eps_surprise_pct():.4f}")

    recent = get_recent(fd, "AAPL", limit=4)
    # The "no surprise" record has UNKNOWN + no actuals → filtered by has_quarterly_data
    assert len(recent) == 3, f"expected 3 with quarterly data, got {len(recent)}: {recent}"
    streak = surprise_history(recent)
    print(f"  ok   recent surprises (newest first): {streak}")
    assert streak == ["BEAT", "MISS", "MEET"], streak

    # FD failure → empty list, never raises
    class BoomFD:
        def get_earnings(self, t): raise RuntimeError("FD 503")
        def get_earnings_history(self, t, limit=4): raise RuntimeError("FD 503")
    boom = BoomFD()
    assert get_latest_actual(boom, "AAPL") is None
    assert get_recent(boom, "AAPL") == []
    print("  ok   FD raising → soft None / [] (no exception bubbled)")


# ---------------------------------------------------------------------------
# 4. Batch result rollup
# ---------------------------------------------------------------------------

def smoke_batch() -> None:
    from v2.earnings import calendar as cal_mod

    future = (date.today() + timedelta(days=10)).isoformat()
    fake_cal_by_ticker = {
        "AAPL": {"Earnings Date": [future], "EPS Estimate": 1.4},
        "NVDA": {},                                # empty
        "MSFT": {"Earnings Date": [future]},
    }

    def fake_ticker_factory(t):
        payload = fake_cal_by_ticker.get(t, {})
        return SimpleNamespace(calendar=payload)

    orig = cal_mod.yf.Ticker
    cal_mod.yf.Ticker = fake_ticker_factory
    try:
        result = cal_mod.get_upcoming_batch(
            ["AAPL", "NVDA", "MSFT", "0700.HK", "BAD-TICKER"]
        )
    finally:
        cal_mod.yf.Ticker = orig

    print(f"  ok   events={list(result.events)}")
    print(f"  ok   skipped_empty={result.skipped_empty}")
    print(f"  ok   skipped_unsupported={result.skipped_unsupported}")
    assert set(result.events) == {"AAPL", "MSFT"}
    assert result.skipped_empty == ["NVDA"]
    assert set(result.skipped_unsupported) == {"0700.HK", "BAD-TICKER"}
    assert result.errors == []


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def smoke_pipeline_reminders() -> None:
    from v2.earnings import calendar as cal_mod
    from v2.earnings import run_reminders

    today = date.today()
    d3 = (today + timedelta(days=3)).isoformat()
    d1 = (today + timedelta(days=1)).isoformat()
    d0 = today.isoformat()
    d5 = (today + timedelta(days=5)).isoformat()  # out of window — should drop

    payloads = {
        "AAPL": {"Earnings Date": [d3]},
        "NVDA": {"Earnings Date": [d1]},
        "MSFT": {"Earnings Date": [d0]},
        "TSLA": {"Earnings Date": [d5]},
        "META": {},                                # empty calendar → skip
    }

    def fake_ticker(t):
        return SimpleNamespace(calendar=payloads.get(t, {}))

    orig = cal_mod.yf.Ticker
    cal_mod.yf.Ticker = fake_ticker
    try:
        run = run_reminders(
            ["AAPL", "NVDA", "MSFT", "TSLA", "META", "0700.HK"],
            today=today.isoformat(),
        )
    finally:
        cal_mod.yf.Ticker = orig

    by_ticker = {r.event.ticker: r.tag for r in run.reminders}
    print(f"  ok   reminders: {by_ticker}")
    assert by_ticker == {"AAPL": "D-3", "NVDA": "D-1", "MSFT": "D-0"}, by_ticker
    # Out-of-window TSLA must NOT appear; META (empty) and 0700.HK
    # (unsupported) must also be absent.
    assert "TSLA" not in by_ticker
    assert "META" not in by_ticker
    assert "0700.HK" not in by_ticker


def smoke_pipeline_summaries() -> None:
    from v2.earnings import run_summaries

    # Two tickers: AAPL has FD data ready, NVDA is still PENDING.
    aapl_q = SimpleNamespace(
        eps_surprise="BEAT", earnings_per_share=2.10,
        estimated_earnings_per_share=1.95, revenue=9.5e10,
        estimated_revenue=9.1e10,
    )
    aapl_record = SimpleNamespace(
        ticker="AAPL", report_period="2026-06-30", source_type="8-K",
        filing_date="2026-08-01", quarterly=aapl_q,
    )

    class FakeFD:
        def get_earnings(self, t):
            if t == "AAPL":
                return aapl_record
            return None                       # NVDA: not yet filed
        def get_earnings_history(self, t, limit=4):
            return [aapl_record] if t == "AAPL" else []

    fd = FakeFD()
    captured_prompts = []

    def fake_summarize(ticker, latest, recent=None, transcript_snippet=None):
        captured_prompts.append((ticker, latest.eps_surprise, transcript_snippet))
        return {"bull": "BEAT 显著且连续", "bear": "指引偏保守",
                "narrative": "本季强劲"}, 123

    def no_transcript(ticker, period):
        return None

    run = run_summaries(
        ["AAPL", "NVDA"], fd,
        today=date.today().isoformat(),
        summarize_fn=fake_summarize,
        transcript_fn=no_transcript,
    )

    by_status = {o.ticker: o.status for o in run.outcomes}
    print(f"  ok   outcome statuses: {by_status}")
    assert by_status == {"AAPL": "summarized", "NVDA": "pending"}, by_status

    aapl = next(o for o in run.outcomes if o.ticker == "AAPL")
    assert aapl.summary is not None
    assert aapl.summary.bull == "BEAT 显著且连续"
    assert aapl.summary.eps_surprise == "BEAT"
    assert aapl.summary.eps_actual == 2.10
    assert aapl.summary.transcript_url is None
    print(f"  ok   AAPL summary: surprise={aapl.summary.eps_surprise} "
          f"bull={aapl.summary.bull!r}")

    # Re-run with already_summarized set → AAPL must be skipped.
    run2 = run_summaries(
        ["AAPL", "NVDA"], fd,
        today=date.today().isoformat(),
        already_summarized={("AAPL", "2026-06-30")},
        summarize_fn=fake_summarize,
        transcript_fn=no_transcript,
    )
    out2 = {o.ticker: o.status for o in run2.outcomes}
    print(f"  ok   dedup re-run: {out2}")
    # AAPL was de-duped (not present); NVDA still pending.
    assert out2 == {"NVDA": "pending"}, out2


def main() -> int:
    failed = []
    for name, fn in [
        ("ticker_filter", smoke_ticker_filter),
        ("calendar", smoke_calendar),
        ("historical", smoke_historical),
        ("batch", smoke_batch),
        ("pipeline_reminders", smoke_pipeline_reminders),
        ("pipeline_summaries", smoke_pipeline_summaries),
    ]:
        _section(name)
        try:
            fn()
        except Exception:
            traceback.print_exc()
            failed.append(name)

    print()
    if failed:
        print(f"FAILED: {failed}")
        return 1
    print("ALL SMOKE TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
