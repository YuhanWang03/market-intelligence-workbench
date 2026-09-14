"""Macro Agent pipeline orchestrator — Phase 4 Stage 1.

Three public builders, one per cron context:

- :func:`build_macro_snapshot` — ⑭ 16:30 ET daily snapshot.
- :func:`build_release_event` — ⑮ 09:00 ET fires only on release days.
- :func:`build_weekly_recap` — ⑰ Fri 19:30 ET weekly summary.

(⑯ Thursday Claims uses :func:`build_release_event` too; it's the same
shape as a CPI release, just with a Claims-specific series tuple.)

Every builder follows the Phase 1/2/3 (metrics, warnings) contract:
sub-failures aggregate to ``MacroReport.warnings``, never raise.
That way the cron always returns *something* and the operator sees a
degraded card instead of a silent dropped run.

Test seams: every external dependency (fred_client, market_client,
calendar, summarizer, tavily) takes an injectable callable so the
Stage 1 smoke can run offline.
"""

from __future__ import annotations

import logging
from typing import Callable

from v2.macro import fred_client as fred_mod
from v2.macro import market_client as market_mod
from v2.macro import release_calendar as cal
from v2.macro import summarizer as summarizer_mod
from v2.macro import tavily_consensus as tavily_mod
from v2.macro import transforms as tx
from v2.macro.models import (
    FOMCEvent, MacroRelease, MacroReport, MacroSnapshot,
)
from v2.macro.series import RELEASE_TO_SERIES, SNAPSHOT_FRED_SERIES

logger = logging.getLogger(__name__)


# Thresholds — pinned, not LLM-tunable
_VIX_SPIKE_PCT = 0.20
_VIX_ELEVATED_PCT = 0.10
_RATES_SHOCK_BPS = 0.20            # DGS10 daily change of ≥ 20 bps (in pct points)


# ---------------------------------------------------------------------------
# ⑭ Snapshot — 16:30 ET daily close
# ---------------------------------------------------------------------------

def build_macro_snapshot(
    today_iso: str,
    *,
    fred_series_fetch: Callable | None = None,
    yfinance_quote: Callable | None = None,
) -> MacroSnapshot:
    """Build the post-close macro snapshot.

    Test seams:
        ``fred_series_fetch(series_id) -> pandas.Series`` — defaults to
        :func:`v2.macro.fred_client.get_series`.
        ``yfinance_quote(symbol) -> dict | None`` — defaults to
        :func:`v2.macro.market_client._safe_quote`.

    Field-level sub-failures land in ``warnings`` and the field stays
    None. The ⑭ cron always pushes a card so the operator knows the
    agent ran.
    """
    fetch_fred = fred_series_fetch or _default_fred_fetch
    fetch_yf = yfinance_quote or _default_yf_quote

    warnings: list[str] = []
    snap = MacroSnapshot(snapshot_date=today_iso)

    # ---- Markets (yfinance) ----
    vix = _safe_yf(fetch_yf, market_mod.VIX_SYMBOL, warnings, "VIX")
    if vix:
        snap.vix = vix.get("value")
        snap.vix_pct_change_1d = vix.get("pct_change_1d")

    dxy = _safe_yf(fetch_yf, market_mod.DXY_SYMBOL, warnings, "DXY")
    if dxy:
        snap.dxy = dxy.get("value")

    wti = _safe_yf(fetch_yf, market_mod.WTI_SYMBOL, warnings, "WTI")
    if wti:
        snap.wti_crude = wti.get("value")

    gold = _safe_yf(fetch_yf, market_mod.GOLD_SYMBOL, warnings, "Gold")
    if gold:
        snap.gold = gold.get("value")

    # ---- Rates (FRED canonical) ----
    series_values: dict[str, list[float]] = {}
    for sid in SNAPSHOT_FRED_SERIES:
        try:
            s = fetch_fred(sid)
            series_values[sid] = tx._to_list(s)
        except Exception as exc:                       # noqa: BLE001
            warnings.append(f"FRED {sid}: {type(exc).__name__}")
            series_values[sid] = []

    snap.fed_funds_upper = _latest(series_values.get("DFEDTARU"))
    snap.fed_funds_lower = _latest(series_values.get("DFEDTARL"))
    snap.dgs2 = _latest(series_values.get("DGS2"))
    snap.dgs10 = _latest(series_values.get("DGS10"))
    snap.t10y2y = _latest(series_values.get("T10Y2Y"))
    snap.t10y2y_prior = _prior(series_values.get("T10Y2Y"))

    # ---- Anomaly flags ----
    if snap.vix_pct_change_1d is not None:
        snap.vix_spike = snap.vix_pct_change_1d >= _VIX_SPIKE_PCT
        snap.vix_elevated = snap.vix_pct_change_1d >= _VIX_ELEVATED_PCT

    if snap.t10y2y is not None and snap.t10y2y_prior is not None:
        # Flip: sign changed between today and prior
        snap.curve_flip = (snap.t10y2y * snap.t10y2y_prior) < 0

    dgs10_vals = series_values.get("DGS10") or []
    if len(dgs10_vals) >= 2:
        daily_change_bps = abs(dgs10_vals[-1] - dgs10_vals[-2])
        snap.rates_shocked = daily_change_bps >= _RATES_SHOCK_BPS

    snap.warnings = warnings
    return snap


# ---------------------------------------------------------------------------
# ⑮ Release event — CPI / PCE / NFP / GDP / PPI
# ---------------------------------------------------------------------------

def build_release_event(
    today_iso: str,
    *,
    fred_series_fetch: Callable | None = None,
    llm_invoke: Callable | None = None,
    tavily_search: Callable | None = None,
    fomc_statement_loader: Callable | None = None,
    prior_fomc_statement_loader: Callable | None = None,
) -> MacroReport:
    """Build the release-day report.

    Looks up :mod:`release_calendar` for what fires today. Non-FOMC
    releases get the standard path (FRED → transforms → summarizer).
    FOMC days route through :mod:`fomc_parser` + :mod:`tavily_consensus`
    — the summarizer is NOT invoked for FOMC text per Stage 0 design.
    """
    fetch_fred = fred_series_fetch or _default_fred_fetch

    warnings: list[str] = []
    report = MacroReport(report_date=today_iso)

    todays = cal.get_release_today(today_iso)
    if not todays:
        return report

    for rel_type, label, _source in todays:
        if rel_type == "FOMC":
            fomc = _build_fomc(
                today_iso, label,
                statement_loader=fomc_statement_loader,
                prior_loader=prior_fomc_statement_loader,
                tavily_search=tavily_search,
                warnings=warnings,
            )
            report.fomc_event = fomc
            continue

        release = _build_one_release(
            rel_type, today_iso, label,
            fetch_fred=fetch_fred,
            llm_invoke=llm_invoke,
            warnings=warnings,
        )
        if release is not None:
            report.today_releases.append(release)

    report.warnings = warnings
    return report


def _build_one_release(
    rel_type: str,
    today_iso: str,
    label: str,
    *,
    fetch_fred: Callable,
    llm_invoke: Callable | None,
    warnings: list[str],
) -> MacroRelease | None:
    """Build a single non-FOMC MacroRelease."""
    series_ids = RELEASE_TO_SERIES.get(rel_type, [])
    if not series_ids:
        warnings.append(f"no FRED series mapped for release_type={rel_type}")
        return None

    series_values: dict[str, list[float]] = {}
    for sid in series_ids:
        try:
            s = fetch_fred(sid)
            series_values[sid] = tx._to_list(s)
        except Exception as exc:                       # noqa: BLE001
            warnings.append(f"FRED {sid}: {type(exc).__name__}")
            series_values[sid] = []

    release = MacroRelease(
        release_type=rel_type,
        release_date=today_iso,
        period=label,
    )

    primary_id = series_ids[0]
    core_id = series_ids[1] if len(series_ids) > 1 else None

    primary = series_values.get(primary_id) or []
    release.headline = primary[-1] if primary else None
    release.prior_value = primary[-2] if len(primary) >= 2 else None

    if core_id:
        core = series_values.get(core_id) or []
        release.core = core[-1] if core else None

    # Trend on the primary series
    release.trailing_3mo_trend = tx.trend_label(primary or [])

    # mom / yoy via the catalog-specified transform
    release.mom_pct = tx.mom_pct(primary)
    release.yoy_pct = tx.yoy_pct(primary)

    # ---- LLM template-fill (Layer 1+2) ----
    try:
        labels = summarizer_mod.summarize_release(
            release, llm_invoke=llm_invoke,
        )
        release.bull_takeaway = labels.get("bull_takeaway")
        release.bear_takeaway = labels.get("bear_takeaway")
        release.narrative = labels.get("narrative")
        release.tone = labels.get("tone") or "neutral"
    except Exception as exc:                           # noqa: BLE001
        warnings.append(f"summarizer({rel_type}): {type(exc).__name__}")
        release.tone = "neutral"

    return release


def _build_fomc(
    fomc_date: str,
    label: str,
    *,
    statement_loader: Callable | None,
    prior_loader: Callable | None,
    tavily_search: Callable | None,
    warnings: list[str],
) -> FOMCEvent:
    """Build a FOMCEvent: statement diff + dot plot + sell-side."""
    from v2.macro import fomc_parser

    statement_text = ""
    statement_diff: dict = {}
    has_sep = "SEP" in label
    sep_medians = None
    sep_shift = "no_change"
    sentiment = None
    sources: list[str] = []

    if statement_loader is not None:
        try:
            statement_text = statement_loader(fomc_date) or ""
        except Exception as exc:                       # noqa: BLE001
            warnings.append(f"fomc_statement_loader: {type(exc).__name__}")

        prior_text = ""
        if prior_loader is not None:
            try:
                prior_text = prior_loader(fomc_date) or ""
            except Exception as exc:                   # noqa: BLE001
                warnings.append(f"fomc_prior_loader: {type(exc).__name__}")

        statement_diff = fomc_parser.diff_statements(statement_text, prior_text)

        if has_sep and statement_text:
            sep_medians = fomc_parser.extract_dot_plot_table(statement_text)
            if sep_medians and prior_text:
                prior_medians = fomc_parser.extract_dot_plot_table(prior_text)
                if prior_medians:
                    sep_shift = fomc_parser.classify_sep_shift(
                        sep_medians, prior_medians,
                    )

    # Tavily sell-side aggregate
    try:
        consensus = tavily_mod.get_fomc_consensus(
            fomc_date, search=tavily_search,
        )
        sentiment = consensus.get("label")
        sources = consensus.get("sources") or []
    except Exception as exc:                           # noqa: BLE001
        warnings.append(f"tavily_consensus: {type(exc).__name__}")

    return FOMCEvent(
        meeting_date=fomc_date,
        statement_text=statement_text,
        statement_diff=statement_diff,
        has_sep=has_sep,
        sep_median_dots=sep_medians,
        sep_dot_plot_change=sep_shift,
        sell_side_sentiment=sentiment,
        sell_side_sources=sources,
    )


# ---------------------------------------------------------------------------
# ⑯ Claims — Thursday 09:30 ET
# ---------------------------------------------------------------------------

def build_claims_event(
    today_iso: str,
    *,
    fred_series_fetch: Callable | None = None,
    llm_invoke: Callable | None = None,
) -> MacroRelease | None:
    """Build the Thursday Initial Jobless Claims release report.

    Unlike ⑮, this builder does NOT gate on
    :func:`release_calendar.get_release_today`. ICSA releases every
    Thursday 08:30 ET on a deterministic weekly cadence, so the ⑯
    cron uses ``CronTrigger(day_of_week="thu")`` and calls this
    function unconditionally; the calendar would carry no extra
    information (and the FRED release_id we'd seed for ICSA — 85 —
    is not in the calendar dict by design).

    Sets ``release.core`` to the 4-week MA smoothed level, which is
    the figure operators actually care about for the trend signal
    (the raw weekly number is too noisy).
    """
    warnings: list[str] = []
    fetch_fred = fred_series_fetch or _default_fred_fetch
    release = _build_one_release(
        "Claims", today_iso, "Initial Claims",
        fetch_fred=fetch_fred,
        llm_invoke=llm_invoke,
        warnings=warnings,
    )
    if release is not None and release.headline is not None:
        # ICSA: smoothed level (4-week MA) is the headline figure ops cares about
        series_id = RELEASE_TO_SERIES["Claims"][0]
        try:
            s = fetch_fred(series_id)
            smoothed = tx.four_week_ma(s)
            if smoothed is not None:
                release.core = smoothed
        except Exception:                              # noqa: BLE001
            pass
    return release


# ---------------------------------------------------------------------------
# ⑰ Weekly recap — Friday 19:30 ET
# ---------------------------------------------------------------------------

def build_weekly_recap(
    today_iso: str,
    *,
    fred_series_fetch: Callable | None = None,
) -> dict:
    """Aggregate the week's releases and preview next week.

    The recap is a dict (not a full MacroReport) because it's a
    summary surface and the cron's formatter will render it from
    this shape. Stages 5 will lift the formatter; Stage 1 just
    pins the data contract.
    """
    from datetime import date, timedelta

    today_d = date.fromisoformat(today_iso)
    week_start = (today_d - timedelta(days=today_d.weekday())).isoformat()
    next_week_end = (today_d + timedelta(days=7)).isoformat()

    this_week = cal.get_releases_in_window(week_start, today_iso)
    next_week = cal.get_releases_in_window(today_iso, next_week_end)
    # Remove today from next_week to avoid double-counting
    next_week.pop(today_iso, None)

    fetch_fred = fred_series_fetch or _default_fred_fetch

    # 1W deltas for headline rates / VIX
    deltas: dict[str, float | None] = {}
    for sid in ("VIXCLS", "DGS10", "DGS2", "T10Y2Y"):
        try:
            s = fetch_fred(sid)
            vals = tx._to_list(s)
        except Exception:                              # noqa: BLE001
            deltas[sid] = None
            continue
        if len(vals) >= 6:
            deltas[sid] = vals[-1] - vals[-6]
        else:
            deltas[sid] = None

    return {
        "week_start": week_start,
        "week_end": today_iso,
        "this_week_releases": this_week,
        "next_week_releases": next_week,
        "weekly_deltas": deltas,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_fred_fetch(series_id: str):
    return fred_mod.get_series(series_id)


def _default_yf_quote(symbol: str):
    return market_mod._safe_quote(symbol)


def _safe_yf(
    fetch_yf: Callable,
    symbol: str,
    warnings: list[str],
    label: str,
) -> dict | None:
    try:
        return fetch_yf(symbol)
    except Exception as exc:                           # noqa: BLE001
        warnings.append(f"yfinance {label}: {type(exc).__name__}")
        return None


def _latest(vals: list[float] | None) -> float | None:
    if not vals:
        return None
    return vals[-1]


def _prior(vals: list[float] | None) -> float | None:
    if not vals or len(vals) < 2:
        return None
    return vals[-2]


__all__ = [
    "build_macro_snapshot",
    "build_release_event",
    "build_claims_event",
    "build_weekly_recap",
]
