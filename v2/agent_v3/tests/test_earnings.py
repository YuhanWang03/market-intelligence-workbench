"""account.earnings_schedule under V3: universe, horizon, and one evidence item per scheduled release."""
from datetime import date
from types import SimpleNamespace

import pytest

from v2.agent_v3.earnings import CAPABILITY, register_earnings, schedule_envelope
from v2.agent_v3.tools import Registry

TODAY = date(2026, 9, 16)


def _event(ticker, release, when="unknown", eps=None, revenue=None, analysts=None):
    return SimpleNamespace(ticker=ticker, release_date=release, when=when, eps_estimate=eps, revenue_estimate=revenue, n_analysts=analysts, source="yfinance")


def _batch(events, unsupported=(), empty=(), errors=()):
    return SimpleNamespace(events={e.ticker: e for e in events}, skipped_unsupported=list(unsupported), skipped_empty=list(empty), errors=list(errors))


def test_releases_inside_the_horizon_become_evidence_and_others_are_named():
    batch = _batch([_event("MU", "2026-09-24", "amc", eps=2.51, revenue=8_900_000_000, analysts=25), _event("NVDA", "2026-11-19", "amc"), _event("AAPL", "2026-10-29", "amc")])
    envelope = schedule_envelope({"MU", "NVDA"}, {"NVDA", "AAPL"}, batch, days=14, today=TODAY, run_id="run-z")
    assert envelope.status.value == "completed"
    by_id = {item.id: item for item in envelope.evidence}
    window = by_id["earnings-window-2026-09-16-14d"]
    assert window.value == 1 and window.metadata["universe"] == ["AAPL", "MU", "NVDA"]
    mu = by_id["earnings-MU-2026-09-24"]
    assert mu.value == 8 and mu.unit == "days" and mu.period == "2026-09-24"
    assert "持仓" in mu.claim and "盘后" in mu.claim and "2.51" in mu.claim and "25 位分析师" in mu.claim
    assert mu.metadata["held"] is True and mu.metadata["watchlist"] is False and mu.producer_run_id == "run-z"
    assert "earnings-AAPL-2026-10-29" not in by_id
    assert any("2 个标的" in note and "AAPL 2026-10-29" in note for note in envelope.limitations)
    assert envelope.findings == [{"ticker": "MU", "release_date": "2026-09-24", "when": "amc"}]


def test_gaps_are_reported_not_invented():
    batch = _batch([], unsupported=["BRK.B"], empty=["HPE"], errors=["INTC:timeout"])
    envelope = schedule_envelope({"BRK.B", "HPE", "INTC"}, set(), batch, days=30, today=TODAY)
    assert envelope.status.value == "partial_data"
    assert len(envelope.evidence) == 1 and envelope.evidence[0].value == 0
    assert any("BRK.B" in note for note in envelope.limitations) and any("HPE" in note for note in envelope.limitations)
    assert envelope.errors == ["日历查询出错：INTC:timeout"]


def test_empty_universe_is_partial_data_with_no_calendar_call():
    called = []
    registry = Registry()
    register_earnings(registry, universe=lambda: (set(), set()), calendar=lambda tickers: called.append(tickers), today=lambda: TODAY)
    result = registry.handlers[CAPABILITY]({}, None)
    assert result.status.value == "partial_data" and called == [] and "都为空" in result.limitations[0]


def test_handler_passes_the_sorted_universe_and_horizon():
    seen = []
    def calendar(tickers):
        seen.append(tickers)
        return _batch([_event("AAPL", "2026-09-20")])
    registry = Registry()
    register_earnings(registry, universe=lambda: ({"NVDA"}, {"AAPL"}), calendar=calendar, today=lambda: TODAY)
    result = registry.handlers[CAPABILITY]({"days": 7}, None)
    assert seen == [["AAPL", "NVDA"]] and result.metrics == {"universe": 2, "scheduled_within_horizon": 1, "beyond_horizon": 0, "horizon_days": 7}
    from v2.agent_v2.models import PlanTask
    registry.validate(PlanTask("e", CAPABILITY, {"days": 14}))
    registry.validate(PlanTask("e", CAPABILITY, {}))
    with pytest.raises(Exception):
        registry.validate(PlanTask("e", CAPABILITY, {"days": 0}))


def test_universe_failure_is_a_partial_error():
    registry = Registry()
    def universe():
        raise ConnectionError("broker down")
    register_earnings(registry, universe=universe, calendar=lambda t: None, today=lambda: TODAY)
    result = registry.handlers[CAPABILITY]({}, None)
    assert result.status.value == "partial_error" and "broker down" in result.errors[0]
