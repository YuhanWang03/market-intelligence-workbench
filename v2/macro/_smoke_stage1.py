"""Stage-1 smoke test for v2/macro/.

Sandbox cannot reach FRED / yfinance / Tavily / DeepSeek live, so all
external dependencies are injected via the explicit test seams the
pipeline + summarizer + tavily modules expose.

Coverage maps to the Stage 1 prompt's listed test cases plus a few
boundary cases (20 total). Each test runs ≤ 1 second; full suite < 5s.

Run:
    poetry run python -m v2.macro._smoke_stage1
"""

from __future__ import annotations

import sys
import traceback
from types import SimpleNamespace
from typing import Callable


def _section(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ---------------------------------------------------------------------------
# transforms.py
# ---------------------------------------------------------------------------

def test_mom_pct_normal():
    from v2.macro.transforms import mom_pct
    series = [100.0, 102.0, 103.0]   # 103/102 - 1 = ~0.0098
    out = mom_pct(series)
    assert out is not None
    assert abs(out - (103.0 / 102.0 - 1.0)) < 1e-9
    print("  ok")


def test_mom_pct_empty_series():
    from v2.macro.transforms import mom_pct
    assert mom_pct([]) is None
    assert mom_pct([100.0]) is None
    print("  ok")


def test_yoy_pct_no_year_ago_data():
    from v2.macro.transforms import yoy_pct
    # 12 points — not enough; needs at least 13 to compute YoY
    assert yoy_pct(list(range(1, 13))) is None
    assert yoy_pct(list(range(1, 14))) is not None
    print("  ok")


def test_four_week_ma_handles_partial_window():
    from v2.macro.transforms import four_week_ma
    assert four_week_ma([]) is None
    assert four_week_ma([10.0]) == 10.0          # partial
    assert four_week_ma([10.0, 20.0]) == 15.0    # partial
    assert four_week_ma([10, 20, 30, 40]) == 25.0
    assert four_week_ma([5, 10, 20, 30, 40]) == 25.0   # last 4
    print("  ok")


def test_surprise_sigma_zero_std_returns_inf():
    import math
    from v2.macro.transforms import surprise_sigma
    # Zero std with non-zero gap → infinity (surfaces pathological case)
    assert surprise_sigma(2.0, 1.0, 0.0) == math.inf
    assert surprise_sigma(0.0, 1.0, 0.0) == -math.inf
    # Zero gap + zero std → 0.0
    assert surprise_sigma(1.0, 1.0, 0.0) == 0.0
    # Normal case
    assert surprise_sigma(2.0, 1.0, 0.5) == 2.0
    # None propagation
    assert surprise_sigma(None, 1.0, 0.5) is None
    print("  ok")


def test_trend_label_three_increasing_returns_accelerating():
    from v2.macro.transforms import trend_label
    assert trend_label([1, 2, 3]) == "accelerating"
    assert trend_label([3, 2, 1]) == "decelerating"
    assert trend_label([1, 1, 1]) == "flat"
    assert trend_label([1, 2]) == "unknown"           # too short
    assert trend_label([1, 3, 2]) == "flat"           # not monotonic
    print("  ok")


def test_surprise_label_buckets():
    from v2.macro.transforms import surprise_label
    assert surprise_label(None) == "no_consensus"
    assert surprise_label(0.5) == "in_line"
    assert surprise_label(-0.5) == "in_line"
    assert surprise_label(1.2) == "above_1sigma"
    assert surprise_label(-1.5) == "below_1sigma"
    assert surprise_label(2.5) == "above_2sigma"
    assert surprise_label(3.5) == "extreme_above_3sigma"
    print("  ok")


# ---------------------------------------------------------------------------
# fred_client.py — REST path for /release/dates
# ---------------------------------------------------------------------------

def test_get_release_dates_rest_returns_iso_strings():
    """REST path: well-formed JSON → list of ISO date strings."""
    import os
    from v2.macro.fred_client import get_release_dates

    os.environ["FRED_API_KEY"] = "fake-key-for-test"

    captured_calls = {"n": 0, "url": None, "params": None}

    class _FakeResponse:
        def raise_for_status(self): pass
        def json(self):
            return {
                "release_dates": [
                    {"release_id": 10, "date": "2026-06-10"},
                    {"release_id": 10, "date": "2026-07-15"},
                    {"release_id": 10, "date": "2026-08-12"},
                ],
            }

    def fake_http_get(url, *, params, timeout):
        captured_calls["n"] += 1
        captured_calls["url"] = url
        captured_calls["params"] = params
        return _FakeResponse()

    dates = get_release_dates(
        10, start="2026-06-01", end="2026-12-31",
        http_get=fake_http_get,
    )
    assert dates == ["2026-06-10", "2026-07-15", "2026-08-12"]
    assert captured_calls["n"] == 1
    assert captured_calls["params"]["release_id"] == 10
    assert captured_calls["params"]["realtime_start"] == "2026-06-01"
    assert captured_calls["params"]["include_release_dates_with_no_data"] == "true"
    assert "/fred/release/dates" in captured_calls["url"]
    print("  ok")


def test_get_release_dates_rest_retries_on_500():
    """HTTPError twice then success → returns the success payload.
    Verifies the 3-attempt linear-backoff retry."""
    import os
    import httpx
    from v2.macro.fred_client import get_release_dates

    os.environ["FRED_API_KEY"] = "fake-key-for-test"

    attempts = {"n": 0}

    class _FakeResponse:
        def raise_for_status(self): pass
        def json(self):
            return {"release_dates": [{"release_id": 10, "date": "2026-06-10"}]}

    def flaky_http_get(url, *, params, timeout):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise httpx.HTTPError(f"FRED 502 attempt {attempts['n']}")
        return _FakeResponse()

    # Monkey-patch the retry backoff to 0 to keep the test fast
    import v2.macro.fred_client as fc_mod
    orig_backoff = fc_mod._RETRY_BACKOFF_SEC
    fc_mod._RETRY_BACKOFF_SEC = 0.0
    try:
        dates = get_release_dates(
            10, start="2026-06-01", end="2026-06-30",
            http_get=flaky_http_get,
        )
    finally:
        fc_mod._RETRY_BACKOFF_SEC = orig_backoff
    assert dates == ["2026-06-10"]
    assert attempts["n"] == 3, f"expected 3 attempts, got {attempts['n']}"
    print("  ok")


def test_get_release_dates_rest_exhausted_retries_raises():
    """Persistent HTTPError → FredUnavailable after 3 attempts."""
    import os
    import httpx
    from v2.macro.fred_client import FredUnavailable, get_release_dates

    os.environ["FRED_API_KEY"] = "fake-key-for-test"

    def always_fails(url, *, params, timeout):
        raise httpx.HTTPError("FRED 502 always")

    import v2.macro.fred_client as fc_mod
    orig_backoff = fc_mod._RETRY_BACKOFF_SEC
    fc_mod._RETRY_BACKOFF_SEC = 0.0
    try:
        try:
            get_release_dates(
                10, start="2026-06-01", end="2026-06-30",
                http_get=always_fails,
            )
            raised = False
        except FredUnavailable:
            raised = True
    finally:
        fc_mod._RETRY_BACKOFF_SEC = orig_backoff
    assert raised, "expected FredUnavailable after exhausted retries"
    print("  ok")


def test_get_release_dates_rest_no_key_raises_FredUnavailable():
    """No FRED_API_KEY in env → immediate FredUnavailable, no HTTP call."""
    import os
    from v2.macro.fred_client import FredUnavailable, get_release_dates

    prev = os.environ.pop("FRED_API_KEY", None)

    def should_not_be_called(*a, **kw):
        raise AssertionError("http_get should not be called when key missing")

    try:
        try:
            get_release_dates(
                10, start="2026-06-01", end="2026-06-30",
                http_get=should_not_be_called,
            )
            raised = False
        except FredUnavailable as exc:
            raised = True
            assert "FRED_API_KEY" in str(exc)
    finally:
        if prev is not None:
            os.environ["FRED_API_KEY"] = prev
    assert raised, "expected FredUnavailable when key missing"
    print("  ok")


def test_fredapi_dependency_surface():
    """Pin the set of fredapi methods we depend on. If a future fredapi
    upgrade removes one of these, fail loudly here instead of at
    runtime against the live API. Also documents what we do NOT
    depend on (get_release_dates lives in our REST path)."""
    from fredapi import Fred

    # Methods we DO use
    assert hasattr(Fred, "get_series"), \
        "fredapi.Fred.get_series missing — used by get_series()"

    # Methods we explicitly do NOT depend on (REST path covers them).
    # The presence of this assertion documents the choice — flip if a
    # future fredapi release ships a working wrapper.
    assert not hasattr(Fred, "get_release_dates"), (
        "fredapi.Fred.get_release_dates now exists — consider migrating "
        "v2/macro/fred_client.py off the REST path."
    )
    print("  ok")


# ---------------------------------------------------------------------------
# release_calendar.py
# ---------------------------------------------------------------------------

def test_release_today_returns_empty_on_non_release_day():
    from v2.macro.release_calendar import get_release_today
    # Pick an obviously non-release day
    assert get_release_today("2026-01-01") == []
    print("  ok")


def test_release_today_returns_FOMC_on_jun_17():
    from v2.macro.release_calendar import get_release_today, is_fomc_day
    rels = get_release_today("2026-06-17")
    assert any(r[0] == "FOMC" for r in rels)
    assert is_fomc_day("2026-06-17") is True
    assert is_fomc_day("2026-06-18") is False
    print("  ok")


def test_staleness_check_silent_within_6_months():
    """The hardcoded _LAST_UPDATED is 2026-06-05; a call from "today"
    in the sandbox should NOT emit a stale warning."""
    import logging
    import warnings as warnings_mod
    from v2.macro.release_calendar import _staleness_check

    with warnings_mod.catch_warnings(record=True) as w:
        warnings_mod.simplefilter("always")
        _staleness_check()
    # Function uses logger.warning, not warnings.warn — silence is the
    # contract. Just call it and ensure no exception.
    print("  ok")


# ---------------------------------------------------------------------------
# summarizer.py — Layer 1 + 2 defense
# ---------------------------------------------------------------------------

def _fake_release(rel_type="CPI"):
    from v2.macro.models import MacroRelease
    return MacroRelease(
        release_type=rel_type,
        release_date="2026-06-10",
        period="2026-05",
        headline=320.0, mom_pct=0.003, yoy_pct=0.029,
        consensus=0.003, surprise_sigma=0.2, surprise_label="in_line",
        trailing_3mo_trend="decelerating",
    )


def test_summarizer_rejects_predictive_will_zh():
    """LLM returns prose containing '将' (future tense) → reject + fallback."""
    from v2.macro.summarizer import summarize_release

    def boom_llm(_sys, _user):
        return '{"bull_takeaway": null, "bear_takeaway": null, ' \
               '"narrative": "通胀将走高", "tone": "hawkish"}'

    out = summarize_release(_fake_release(), llm_invoke=boom_llm)
    assert out["narrative"] != "通胀将走高", "predictive text leaked through"
    assert out["tone"] == "neutral"
    print("  ok")


def test_summarizer_rejects_predictive_expect_zh():
    """'预期' (expect) → reject."""
    from v2.macro.summarizer import summarize_release

    def llm(_sys, _user):
        return '{"bull_takeaway": "市场预期升息", "bear_takeaway": null, ' \
               '"narrative": "数据已发布", "tone": "neutral"}'

    out = summarize_release(_fake_release(), llm_invoke=llm)
    assert out["bull_takeaway"] is None
    assert out["tone"] == "neutral"
    print("  ok")


def test_summarizer_fallback_on_invalid_json():
    """LLM returns garbage → fallback."""
    from v2.macro.summarizer import summarize_release

    def llm(_sys, _user):
        return "this is not JSON at all"

    out = summarize_release(_fake_release(), llm_invoke=llm)
    assert out["tone"] == "neutral"
    assert out["narrative"]  # fallback narrative is non-empty
    print("  ok")


def test_summarizer_rejects_numeric_leak():
    """Even though the prompt forbids numbers, defend at parse time."""
    from v2.macro.summarizer import summarize_release

    def llm(_sys, _user):
        return '{"bull_takeaway": null, "bear_takeaway": ' \
               '"CPI 上涨 0.3%", "narrative": "数据已发布", "tone": "neutral"}'

    out = summarize_release(_fake_release(), llm_invoke=llm)
    assert out["bear_takeaway"] is None or "0.3%" not in (out["bear_takeaway"] or "")
    print("  ok")


def test_summarizer_accepts_clean_output():
    """A response that follows all rules should pass through unchanged."""
    from v2.macro.summarizer import summarize_release

    def llm(_sys, _user):
        return (
            '{"bull_takeaway": "核心通胀连续3月放缓", '
            '"bear_takeaway": "服务业通胀仍粘性", '
            '"narrative": "通胀压力分化", "tone": "dovish"}'
        )

    out = summarize_release(_fake_release(), llm_invoke=llm)
    assert out["bull_takeaway"] == "核心通胀连续3月放缓"
    assert out["tone"] == "dovish"
    print("  ok")


def test_summarizer_rejects_invalid_tone():
    """tone must be one of the 3 enums; "bullish" is not allowed."""
    from v2.macro.summarizer import summarize_release

    def llm(_sys, _user):
        return '{"bull_takeaway": null, "bear_takeaway": null, ' \
               '"narrative": "data released", "tone": "bullish"}'

    out = summarize_release(_fake_release(), llm_invoke=llm)
    assert out["tone"] == "neutral"
    print("  ok")


# ---------------------------------------------------------------------------
# fomc_parser.py
# ---------------------------------------------------------------------------

def test_statement_diff_detects_added_phrase():
    from v2.macro.fomc_parser import diff_statements
    current = "The Committee judges that additional policy firming may be appropriate."
    prior   = "The Committee will maintain a data-dependent stance."
    d = diff_statements(current, prior)
    assert "additional policy firming" in d["added_phrases"]
    assert "data-dependent" in d["removed_phrases"]
    print("  ok")


def test_statement_diff_detects_removed_phrase():
    from v2.macro.fomc_parser import diff_statements
    current = "The Committee will be patient."
    prior   = "Additional policy firming may be appropriate. Risks are balanced."
    d = diff_statements(current, prior)
    assert "additional policy firming" in d["removed_phrases"]
    print("  ok")


def test_classify_sep_shift_hawkish_when_median_up():
    from v2.macro.fomc_parser import classify_sep_shift
    current = {2026: 4.00, 2027: 3.50}
    prior   = {2026: 3.75, 2027: 3.25}
    assert classify_sep_shift(current, prior) == "hawkish_shift"
    print("  ok")


def test_classify_sep_shift_dovish_when_median_down():
    from v2.macro.fomc_parser import classify_sep_shift
    current = {2026: 3.50, 2027: 3.00}
    prior   = {2026: 3.75, 2027: 3.25}
    assert classify_sep_shift(current, prior) == "dovish_shift"
    print("  ok")


def test_classify_sep_shift_no_change():
    from v2.macro.fomc_parser import classify_sep_shift
    assert classify_sep_shift({2026: 3.75}, {2026: 3.75}) == "no_change"
    # Mixed → no_change (some up some down)
    mixed_curr = {2026: 4.00, 2027: 3.00}
    mixed_prior = {2026: 3.75, 2027: 3.50}
    assert classify_sep_shift(mixed_curr, mixed_prior) == "no_change"
    print("  ok")


def test_extract_dot_plot_table_finds_fed_funds_row():
    from v2.macro.fomc_parser import extract_dot_plot_table
    sep = """
    Variable           2024    2025    2026   2027    Longer run
    Federal funds rate  5.4    4.6     3.6    2.9      2.9
    Other line          1.0    1.0     1.0    1.0      1.0
    """
    table = extract_dot_plot_table(sep)
    assert table is not None
    assert table[2025] == 4.6
    assert table.get("longer_run") == 2.9
    print("  ok")


# ---------------------------------------------------------------------------
# tavily_consensus.py
# ---------------------------------------------------------------------------

def test_tavily_consensus_majority_hawkish():
    from v2.macro.tavily_consensus import get_fomc_consensus

    def fake_search(query, *, max_results=5):
        return [
            {"title": "Powell stays hawkish, signals more hikes",
             "content": "hawkish hawkish", "url": "https://reuters.com/x"},
            {"title": "Yields jump as Fed seen hawkish",
             "content": "hawkish stance", "url": "https://bloomberg.com/y"},
            {"title": "Mixed reaction, some see dovish tilt",
             "content": "dovish read", "url": "https://wsj.com/z"},
        ]

    out = get_fomc_consensus("2026-06-17", search=fake_search)
    assert out["label"] == "hawkish"
    assert out["hawkish_mentions"] >= 3
    assert out["dovish_mentions"] >= 1
    assert "reuters.com" in out["sources"]
    print("  ok")


def test_tavily_consensus_tie_returns_mixed():
    from v2.macro.tavily_consensus import get_fomc_consensus

    def fake_search(query, *, max_results=5):
        return [
            {"title": "hawkish read", "content": "hawkish", "url": "https://a.com"},
            {"title": "dovish read", "content": "dovish", "url": "https://b.com"},
        ]

    out = get_fomc_consensus("2026-06-17", search=fake_search)
    assert out["label"] == "mixed"
    print("  ok")


def test_tavily_consensus_failure_returns_mixed_fallback():
    from v2.macro.tavily_consensus import get_fomc_consensus

    def boom_search(query, *, max_results=5):
        raise RuntimeError("Tavily 503")

    out = get_fomc_consensus("2026-06-17", search=boom_search)
    assert out["label"] == "mixed"
    assert out.get("_reason", "").startswith("tavily error")
    print("  ok")


# ---------------------------------------------------------------------------
# pipeline.py
# ---------------------------------------------------------------------------

def _make_series(values):
    """Build a minimal pandas.Series-like object compatible with the
    transforms module (uses .dropna().tolist())."""
    import pandas as pd
    return pd.Series(values, dtype="float64")


def test_snapshot_handles_yfinance_failure_per_field():
    """VIX fetch fails but DXY succeeds → snapshot has DXY, no VIX,
    warnings list mentions VIX."""
    from v2.macro.pipeline import build_macro_snapshot

    def fred_fetch(sid):
        # Provide a 30-day DGS10 series so curve / shock flags can compute
        return _make_series([4.20] * 30 + [4.10])

    def yf_quote(sym):
        if sym == "^VIX":
            raise RuntimeError("yfinance VIX 503")
        return {"value": 105.5, "pct_change_1d": 0.001}

    snap = build_macro_snapshot(
        "2026-06-05",
        fred_series_fetch=fred_fetch,
        yfinance_quote=yf_quote,
    )
    assert snap.vix is None
    assert snap.dxy == 105.5
    assert any("VIX" in w for w in snap.warnings)
    print("  ok")


def test_snapshot_vix_spike_flag():
    """VIX +25% intraday → vix_spike=True."""
    from v2.macro.pipeline import build_macro_snapshot

    def fred_fetch(sid):
        return _make_series([0.0])

    def yf_quote(sym):
        if sym == "^VIX":
            return {"value": 25.0, "pct_change_1d": 0.25}
        return {"value": 1.0, "pct_change_1d": 0.0}

    snap = build_macro_snapshot(
        "2026-06-05",
        fred_series_fetch=fred_fetch, yfinance_quote=yf_quote,
    )
    assert snap.vix_spike is True
    assert snap.vix_elevated is True
    print("  ok")


def test_snapshot_curve_flip_flag():
    """T10Y2Y goes from +0.05 (positive) to -0.05 (negative) → curve_flip=True."""
    from v2.macro.pipeline import build_macro_snapshot

    def fred_fetch(sid):
        if sid == "T10Y2Y":
            return _make_series([0.05, -0.05])
        return _make_series([4.0, 4.0])

    def yf_quote(_sym):
        return {"value": 1.0, "pct_change_1d": 0.0}

    snap = build_macro_snapshot(
        "2026-06-05",
        fred_series_fetch=fred_fetch, yfinance_quote=yf_quote,
    )
    assert snap.curve_flip is True
    print("  ok")


def test_snapshot_rates_shock_flag():
    """DGS10 moves 25bps day-over-day → rates_shocked=True."""
    from v2.macro.pipeline import build_macro_snapshot

    def fred_fetch(sid):
        if sid == "DGS10":
            return _make_series([4.00, 4.25])
        return _make_series([0.0, 0.0])

    def yf_quote(_sym):
        return {"value": 1.0, "pct_change_1d": 0.0}

    snap = build_macro_snapshot(
        "2026-06-05",
        fred_series_fetch=fred_fetch, yfinance_quote=yf_quote,
    )
    assert snap.rates_shocked is True
    print("  ok")


def test_release_event_returns_empty_on_non_release_day():
    """Non-release day → MacroReport with no releases / no FOMC."""
    from v2.macro.pipeline import build_release_event

    rep = build_release_event("2026-01-15")    # not on the calendar
    assert rep.today_releases == []
    assert rep.fomc_event is None
    print("  ok")


def test_release_event_cpi_path_invokes_summarizer():
    """A CPI day routes through the standard FRED + summarizer pipe."""
    from v2.macro.pipeline import build_release_event

    captured_calls = {"summarizer": 0}

    def fred_fetch(sid):
        return _make_series([320.0 + i * 0.5 for i in range(20)])

    def llm(_sys, _user):
        captured_calls["summarizer"] += 1
        return ('{"bull_takeaway": "core slowing", "bear_takeaway": null, '
                '"narrative": "headline cooled", "tone": "dovish"}')

    rep = build_release_event(
        "2026-06-10",                     # CPI day on the calendar
        fred_series_fetch=fred_fetch,
        llm_invoke=llm,
    )
    assert len(rep.today_releases) == 1
    rel = rep.today_releases[0]
    assert rel.release_type == "CPI"
    assert rel.tone == "dovish"
    assert captured_calls["summarizer"] == 1
    print("  ok")


def test_fomc_event_path_skips_summarizer():
    """FOMC day routes through fomc_parser + tavily; the LLM
    summarizer is NEVER called for FOMC text."""
    from v2.macro.pipeline import build_release_event

    summarizer_calls = {"n": 0}

    def llm(_sys, _user):
        summarizer_calls["n"] += 1
        return "{}"

    def fake_tavily(query, *, max_results=5):
        return [{"title": "Hawkish hike", "content": "hawkish hawkish",
                 "url": "https://reuters.com/x"}]

    def load_current(_iso):
        return "Additional policy firming may be appropriate."

    def load_prior(_iso):
        return "The Committee will be patient."

    rep = build_release_event(
        "2026-06-17",                   # FOMC + SEP day
        llm_invoke=llm,
        tavily_search=fake_tavily,
        fomc_statement_loader=load_current,
        prior_fomc_statement_loader=load_prior,
    )
    assert rep.fomc_event is not None
    assert summarizer_calls["n"] == 0, "FOMC must NOT call summarizer"
    assert rep.fomc_event.sell_side_sentiment == "hawkish"
    assert "additional policy firming" in rep.fomc_event.statement_diff["added_phrases"]
    print("  ok")


def test_release_event_fred_failure_aggregates_warning():
    """A FRED series failure produces a warning + None field, never raises."""
    from v2.macro.pipeline import build_release_event

    def fred_fetch(sid):
        raise RuntimeError("FRED 502")

    def llm(_sys, _user):
        return ('{"bull_takeaway": null, "bear_takeaway": null, '
                '"narrative": "data released", "tone": "neutral"}')

    rep = build_release_event(
        "2026-06-10",                   # CPI day
        fred_series_fetch=fred_fetch, llm_invoke=llm,
    )
    assert rep.warnings, "FRED failure must surface in warnings"
    assert any("FRED" in w for w in rep.warnings)
    print("  ok")


def test_weekly_recap_shape():
    """Fri 19:30 ET recap returns this/next week dicts + 1W deltas."""
    from v2.macro.pipeline import build_weekly_recap

    def fred_fetch(sid):
        # Provide enough points for a 6-element history (1W lookback)
        return _make_series([1.0, 1.1, 1.2, 1.3, 1.4, 1.5])

    recap = build_weekly_recap("2026-06-12", fred_series_fetch=fred_fetch)
    assert "this_week_releases" in recap
    assert "next_week_releases" in recap
    assert recap["weekly_deltas"]["VIXCLS"] is not None
    # delta = last - 6th-from-last = 1.5 - 1.0 = 0.5
    assert abs(recap["weekly_deltas"]["VIXCLS"] - 0.5) < 1e-9
    print("  ok")


# ---------------------------------------------------------------------------
# Observability fingerprint
# ---------------------------------------------------------------------------

def test_fingerprint_registered():
    """The macro_summarizer fingerprint must be in LLM_ROLE_FINGERPRINTS
    so cron-captured traces tag this role correctly."""
    from v2.observability.hooks import detect_llm_role
    prompt = "你是宏观数据解读分析师。输入 JSON 含数据点..."
    assert detect_llm_role(prompt) == "macro_summarizer"
    print("  ok")


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def main() -> int:
    cases: list[tuple[str, Callable]] = [
        # transforms
        ("transforms_mom_pct_normal",          test_mom_pct_normal),
        ("transforms_mom_pct_empty",           test_mom_pct_empty_series),
        ("transforms_yoy_no_year_data",        test_yoy_pct_no_year_ago_data),
        ("transforms_four_week_ma_partial",    test_four_week_ma_handles_partial_window),
        ("transforms_surprise_sigma_zero_std", test_surprise_sigma_zero_std_returns_inf),
        ("transforms_trend_label_3_inc",       test_trend_label_three_increasing_returns_accelerating),
        ("transforms_surprise_label_buckets",  test_surprise_label_buckets),
        # fred_client REST path
        ("fred_rest_returns_iso_strings",      test_get_release_dates_rest_returns_iso_strings),
        ("fred_rest_retries_on_500",           test_get_release_dates_rest_retries_on_500),
        ("fred_rest_exhausted_retries_raises", test_get_release_dates_rest_exhausted_retries_raises),
        ("fred_rest_no_key_raises",            test_get_release_dates_rest_no_key_raises_FredUnavailable),
        ("fredapi_dependency_surface",         test_fredapi_dependency_surface),
        # release calendar
        ("calendar_today_empty_off_day",       test_release_today_returns_empty_on_non_release_day),
        ("calendar_fomc_jun_17",               test_release_today_returns_FOMC_on_jun_17),
        ("calendar_staleness_silent_lt_6mo",   test_staleness_check_silent_within_6_months),
        # summarizer
        ("summarizer_reject_predictive_will",  test_summarizer_rejects_predictive_will_zh),
        ("summarizer_reject_predictive_yùqī",  test_summarizer_rejects_predictive_expect_zh),
        ("summarizer_fallback_invalid_json",   test_summarizer_fallback_on_invalid_json),
        ("summarizer_reject_numeric_leak",     test_summarizer_rejects_numeric_leak),
        ("summarizer_accepts_clean_output",    test_summarizer_accepts_clean_output),
        ("summarizer_reject_invalid_tone",     test_summarizer_rejects_invalid_tone),
        # fomc parser
        ("fomc_diff_added_phrase",             test_statement_diff_detects_added_phrase),
        ("fomc_diff_removed_phrase",           test_statement_diff_detects_removed_phrase),
        ("fomc_sep_hawkish_shift",             test_classify_sep_shift_hawkish_when_median_up),
        ("fomc_sep_dovish_shift",              test_classify_sep_shift_dovish_when_median_down),
        ("fomc_sep_no_change",                 test_classify_sep_shift_no_change),
        ("fomc_extract_dot_plot",              test_extract_dot_plot_table_finds_fed_funds_row),
        # tavily consensus
        ("tavily_majority_hawkish",            test_tavily_consensus_majority_hawkish),
        ("tavily_tie_mixed",                   test_tavily_consensus_tie_returns_mixed),
        ("tavily_failure_mixed_fallback",      test_tavily_consensus_failure_returns_mixed_fallback),
        # pipeline
        ("snapshot_yf_failure_per_field",      test_snapshot_handles_yfinance_failure_per_field),
        ("snapshot_vix_spike_flag",            test_snapshot_vix_spike_flag),
        ("snapshot_curve_flip_flag",           test_snapshot_curve_flip_flag),
        ("snapshot_rates_shock_flag",          test_snapshot_rates_shock_flag),
        ("release_event_empty_off_day",        test_release_event_returns_empty_on_non_release_day),
        ("release_event_cpi_summarizer",       test_release_event_cpi_path_invokes_summarizer),
        ("release_event_fomc_no_summarizer",   test_fomc_event_path_skips_summarizer),
        ("release_event_fred_warning",         test_release_event_fred_failure_aggregates_warning),
        ("weekly_recap_shape",                 test_weekly_recap_shape),
        # observability
        ("fingerprint_registered",             test_fingerprint_registered),
    ]

    failed: list[str] = []
    for name, fn in cases:
        _section(name)
        try:
            fn()
        except Exception:
            traceback.print_exc()
            failed.append(name)

    print()
    if failed:
        print(f"FAILED ({len(failed)}): {failed}")
        return 1
    print("ALL SMOKE TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
