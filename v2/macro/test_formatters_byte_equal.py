"""Byte-equal pin tests for the 6 public macro card formatters.

Mirrors Phase 1's earnings byte-equal pattern, Phase 2's portfolio
byte-equal pattern, and Phase 3's SEC byte-equal pattern. Pin the
exact card body so a Stage 7 README edit or any unrelated refactor
cannot silently shift the cron / bot output.

Imports go through the ``v2.macro._bot_cards`` source-of-truth module
to keep tests sandbox-runnable. ``v2.reporting.format_macro_*`` is
the same function object — verified via identity assertion at the bottom.

Coverage map:

⑭ Snapshot card  (3 tests):
- normal P3 day
- VIX +25% spike (vix_spike + vix_elevated flags)
- curve flip with rates_shocked

⑮ Release card  (3 tests):
- CPI in-line (P2 base)
- CPI extreme surprise (σ=3.5, used with bot next_release_date suffix)
- summarizer LLM-fail fallback (null tone / no bull / no bear)

⑮ FOMC card  (2 tests):
- May FOMC, no SEP
- Jun FOMC + SEP hawkish_shift

⑯ Claims card  (1 test):
- Thursday Claims (normal, no consensus)

⑰ Weekly recap card  (2 tests):
- normal week with releases
- quiet week (no releases)

/macro dashboard  (3 tests):
- full data + populated calendar
- partial data + warnings + empty calendar
- spike + curve flip visible

Surface  (2 tests):
- ``_bot_cards.__all__`` contract
- v2.reporting shim identity (importlib bypass)
"""

from __future__ import annotations

import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


from v2.macro._bot_cards import (  # noqa: E402
    format_macro_claims_card,
    format_macro_daily_snapshot,
    format_macro_dashboard,
    format_macro_fomc_card,
    format_macro_release_card,
    format_macro_weekly_recap,
)
from v2.macro.models import (  # noqa: E402
    FOMCEvent, MacroRelease, MacroSnapshot,
)


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

def _normal_snapshot() -> MacroSnapshot:
    """⑭ default day — VIX flat, curve normal, no anomaly flags."""
    return MacroSnapshot(
        snapshot_date="2026-06-12",
        vix=14.20, vix_pct_change_1d=0.008,
        dxy=99.50, wti_crude=78.40, gold=2650.5,
        fed_funds_upper=5.50, fed_funds_lower=5.25,
        dgs2=4.21, dgs10=4.42, t10y2y=0.21, t10y2y_prior=0.18,
    )


def _spike_snapshot() -> MacroSnapshot:
    """⑭ VIX +25% spike + curve flip same day."""
    return MacroSnapshot(
        snapshot_date="2026-06-13",
        vix=28.50, vix_pct_change_1d=0.25,
        vix_spike=True, vix_elevated=True,
        dxy=99.5, wti_crude=78.4, gold=2650.5,
        fed_funds_upper=5.50, fed_funds_lower=5.25,
        dgs2=4.21, dgs10=4.42, t10y2y=-0.10, t10y2y_prior=0.05,
        curve_flip=True,
    )


def _rates_shock_snapshot() -> MacroSnapshot:
    """⑭ DGS10 +20bps day-over-day shock."""
    return MacroSnapshot(
        snapshot_date="2026-06-14",
        vix=18.0, vix_pct_change_1d=0.05,
        dxy=99.5, wti_crude=78.4, gold=2650.5,
        fed_funds_upper=5.50, fed_funds_lower=5.25,
        dgs2=4.20, dgs10=4.65, t10y2y=0.45, t10y2y_prior=0.45,
        rates_shocked=True,
    )


def _cpi_in_line_release() -> MacroRelease:
    return MacroRelease(
        release_type="CPI", release_date="2026-06-10",
        period="CPI May 2026",
        headline=320.5, core=315.2,
        mom_pct=0.003, yoy_pct=0.029,
        consensus=0.003, surprise_sigma=0.2, surprise_label="in_line",
        trailing_3mo_trend="decelerating",
        bull_takeaway="核心 YoY 连续 3 月放缓",
        bear_takeaway="MoM 仍高于 Fed 目标",
        narrative="Headline 与预期持平",
        tone="neutral",
    )


def _cpi_extreme_surprise_release() -> MacroRelease:
    return MacroRelease(
        release_type="CPI", release_date="2026-06-10",
        period="CPI May 2026",
        headline=320.5, core=315.2,
        mom_pct=0.005, yoy_pct=0.035,
        consensus=0.003, surprise_sigma=3.5, surprise_label="extreme_above_3sigma",
        trailing_3mo_trend="accelerating",
        bull_takeaway=None,
        bear_takeaway="3σ 以上向上偏离",
        narrative="通胀大幅高于预期",
        tone="hawkish",
    )


def _cpi_llm_fallback_release() -> MacroRelease:
    """summarizer fallback path — Layer 1+2 rejected the LLM output
    and returned neutral defaults. bull / bear are None and narrative
    is the canonical fallback string."""
    return MacroRelease(
        release_type="CPI", release_date="2026-06-10",
        period="CPI May 2026",
        headline=320.5, core=315.2,
        mom_pct=0.003, yoy_pct=0.029,
        consensus=0.003, surprise_sigma=0.5, surprise_label="in_line",
        trailing_3mo_trend="flat",
        bull_takeaway=None,
        bear_takeaway=None,
        narrative="数据已发布，详见上方数值。",
        tone="neutral",
    )


def _fomc_may_no_sep() -> FOMCEvent:
    return FOMCEvent(
        meeting_date="2026-05-07",
        statement_diff={
            "added_phrases": ["elevated"],
            "removed_phrases": ["modest"],
            "unchanged_phrases": [],
        },
        has_sep=False,
        sep_median_dots=None,
        sep_dot_plot_change="no_change",
        sell_side_sentiment="hawkish",
        sell_side_sources=["reuters.com", "bloomberg.com", "wsj.com"],
    )


def _fomc_jun_sep_hawkish() -> FOMCEvent:
    return FOMCEvent(
        meeting_date="2026-06-17",
        statement_diff={
            "added_phrases": ["additional policy firming"],
            "removed_phrases": ["data-dependent"],
            "unchanged_phrases": [],
        },
        has_sep=True,
        sep_median_dots={2026: 4.00, 2027: 3.50, "longer_run": 2.875},
        sep_dot_plot_change="hawkish_shift",
        sell_side_sentiment="hawkish",
        sell_side_sources=["reuters.com", "bloomberg.com", "wsj.com"],
    )


def _claims_release() -> MacroRelease:
    return MacroRelease(
        release_type="Claims", release_date="2026-06-18",
        period="Initial Claims",
        headline=242000, core=236500, prior_value=228000,
        trailing_3mo_trend="accelerating",
        narrative="周度初请连续 3 周抬升",
        tone="hawkish",
        bear_takeaway="劳动力市场冷却信号",
    )


def _weekly_recap_normal() -> dict:
    return {
        "week_start": "2026-06-08",
        "week_end": "2026-06-12",
        "weekly_deltas": {
            "VIXCLS": 2.10, "DGS10": -0.05, "DGS2": -0.02, "T10Y2Y": -0.03,
        },
        "this_week_releases": {
            "2026-06-10": [("CPI", "CPI May 2026", "BLS")],
            "2026-06-11": [("PPI", "PPI May 2026", "BLS")],
        },
        "next_week_releases": {
            "2026-06-17": [("FOMC", "Jun FOMC + SEP", "Fed")],
        },
    }


def _weekly_recap_quiet() -> dict:
    return {
        "week_start": "2026-07-06",
        "week_end": "2026-07-10",
        "weekly_deltas": {
            "VIXCLS": 0.20, "DGS10": 0.01, "DGS2": 0.00, "T10Y2Y": 0.01,
        },
        "this_week_releases": {},
        "next_week_releases": {},
    }


# ---------------------------------------------------------------------------
# ⑭ format_macro_daily_snapshot byte-equal
# ---------------------------------------------------------------------------

def test_snapshot_normal_day_byte_equal():
    actual = format_macro_daily_snapshot(_normal_snapshot())
    expected = (
        "<b>📊 宏观日终 · 2026-06-12</b>\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "<b>市场</b>\n"
        "  VIX: <code>14.20</code> (+0.80%)\n"
        "  DXY: <code>99.50</code> · WTI: <code>78.40</code> · Gold: <code>2650.5</code>\n"
        "\n"
        "<b>利率 (FRED EOD)</b>\n"
        "  Fed Funds: <code>5.25% – 5.50%</code>\n"
        "  2Y: <code>4.21%</code> · 10Y: <code>4.42%</code>\n"
        "  10Y-2Y: <code>+21bp</code>"
    )
    assert actual == expected, (
        f"\n--- actual ---\n{actual}\n\n--- expected ---\n{expected}"
    )


def test_snapshot_vix_spike_byte_equal():
    actual = format_macro_daily_snapshot(_spike_snapshot())
    expected = (
        "<b>🚨 宏观警报 · 2026-06-13</b>\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "<b>市场</b>\n"
        "  VIX: <code>28.50</code> (+25.00%) <b>🚨 +20%</b>\n"
        "  DXY: <code>99.50</code> · WTI: <code>78.40</code> · Gold: <code>2650.5</code>\n"
        "\n"
        "<b>利率 (FRED EOD)</b>\n"
        "  Fed Funds: <code>5.25% – 5.50%</code>\n"
        "  2Y: <code>4.21%</code> · 10Y: <code>4.42%</code>\n"
        "  10Y-2Y: <code>-10bp</code> <b>📉 今日翻转</b>"
    )
    assert actual == expected


def test_snapshot_rates_shock_byte_equal():
    actual = format_macro_daily_snapshot(_rates_shock_snapshot())
    expected = (
        "<b>📉 宏观警报 · 2026-06-14</b>\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "<b>市场</b>\n"
        "  VIX: <code>18.00</code> (+5.00%)\n"
        "  DXY: <code>99.50</code> · WTI: <code>78.40</code> · Gold: <code>2650.5</code>\n"
        "\n"
        "<b>利率 (FRED EOD)</b>\n"
        "  Fed Funds: <code>5.25% – 5.50%</code>\n"
        "  2Y: <code>4.20%</code> · 10Y: <code>4.65%</code>\n"
        "  10Y-2Y: <code>+45bp</code>\n"
        "  <b>⚠️ 10Y 单日 ≥ 20bps 异动</b>"
    )
    assert actual == expected


# ---------------------------------------------------------------------------
# ⑮ format_macro_release_card byte-equal
# ---------------------------------------------------------------------------

def test_release_cpi_in_line_p2_byte_equal():
    actual = format_macro_release_card(_cpi_in_line_release(), tier="P2")
    expected = (
        "<b>📅 CPI · CPI May 2026</b>\n"
        "发布日：<code>2026-06-10</code> · 评级：<b>P2</b>\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "  MoM: <code>+0.30%</code>\n"
        "  YoY: <code>+2.90%</code>\n"
        "  Headline: <code>320.50</code>\n"
        "  Core: <code>315.20</code>\n"
        "  Consensus: <code>+0.30%</code> (+0.2σ, in_line)\n"
        "  3M 趋势: <i>decelerating</i>\n"
        "\n"
        "<b>⚪ 解读</b> <i>(neutral)</i>\n"
        "  Headline 与预期持平\n"
        "  🟢 核心 YoY 连续 3 月放缓\n"
        "  🔴 MoM 仍高于 Fed 目标"
    )
    assert actual == expected, (
        f"\n--- actual ---\n{actual}\n\n--- expected ---\n{expected}"
    )


def test_release_cpi_extreme_surprise_p0_byte_equal():
    """Extreme surprise + asymmetric LLM output (no bull, tone=hawkish)."""
    actual = format_macro_release_card(
        _cpi_extreme_surprise_release(), tier="P0",
    )
    expected = (
        "<b>📅 CPI · CPI May 2026</b>\n"
        "发布日：<code>2026-06-10</code> · 评级：<b>P0</b>\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "  MoM: <code>+0.50%</code>\n"
        "  YoY: <code>+3.50%</code>\n"
        "  Headline: <code>320.50</code>\n"
        "  Core: <code>315.20</code>\n"
        "  Consensus: <code>+0.30%</code> (+3.5σ, extreme_above_3sigma)\n"
        "  3M 趋势: <i>accelerating</i>\n"
        "\n"
        "<b>🟥 解读</b> <i>(hawkish)</i>\n"
        "  通胀大幅高于预期\n"
        "  🔴 3σ 以上向上偏离"
    )
    assert actual == expected


def test_release_summarizer_fallback_byte_equal():
    """LLM-fail path: null bull / null bear / canonical fallback narrative."""
    actual = format_macro_release_card(
        _cpi_llm_fallback_release(), tier="P2",
    )
    expected = (
        "<b>📅 CPI · CPI May 2026</b>\n"
        "发布日：<code>2026-06-10</code> · 评级：<b>P2</b>\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "  MoM: <code>+0.30%</code>\n"
        "  YoY: <code>+2.90%</code>\n"
        "  Headline: <code>320.50</code>\n"
        "  Core: <code>315.20</code>\n"
        "  Consensus: <code>+0.30%</code> (+0.5σ, in_line)\n"
        "  3M 趋势: <i>flat</i>\n"
        "\n"
        "<b>⚪ 解读</b> <i>(neutral)</i>\n"
        "  数据已发布，详见上方数值。"
    )
    assert actual == expected


# ---------------------------------------------------------------------------
# ⑮ format_macro_fomc_card byte-equal
# ---------------------------------------------------------------------------

def test_fomc_may_no_sep_byte_equal():
    actual = format_macro_fomc_card(_fomc_may_no_sep(), tier="P1")
    expected = (
        "<b>🏛️ FOMC · 2026-05-07</b>\n"
        "会议日：<code>2026-05-07</code> · 评级：<b>P1</b>\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "<b>📌 Statement 新增措辞</b>\n"
        "  ➕ <i>elevated</i>\n"
        "<b>📌 Statement 移除措辞</b>\n"
        "  ➖ <i>modest</i>\n"
        "\n"
        "<b>📰 卖方读数</b>: <i>hawkish</i> <i>(Tavily majority vote)</i>\n"
        "  来源: reuters.com, bloomberg.com, wsj.com"
    )
    assert actual == expected, (
        f"\n--- actual ---\n{actual}\n\n--- expected ---\n{expected}"
    )


def test_fomc_jun_sep_hawkish_byte_equal():
    actual = format_macro_fomc_card(_fomc_jun_sep_hawkish(), tier="P0")
    expected = (
        "<b>🏛️ FOMC · 2026-06-17</b>\n"
        "会议日：<code>2026-06-17</code> · 评级：<b>P0</b>\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "<b>📌 Statement 新增措辞</b>\n"
        "  ➕ <i>additional policy firming</i>\n"
        "<b>📌 Statement 移除措辞</b>\n"
        "  ➖ <i>data-dependent</i>\n"
        "\n"
        "<b>📊 SEP Dot Plot</b> <i>(hawkish_shift)</i>\n"
        "  2026: <code>4.00%</code>\n"
        "  2027: <code>3.50%</code>\n"
        "  longer_run: <code>2.88%</code>\n"
        "\n"
        "<b>📰 卖方读数</b>: <i>hawkish</i> <i>(Tavily majority vote)</i>\n"
        "  来源: reuters.com, bloomberg.com, wsj.com"
    )
    assert actual == expected


# ---------------------------------------------------------------------------
# ⑯ format_macro_claims_card byte-equal
# ---------------------------------------------------------------------------

def test_claims_normal_byte_equal():
    actual = format_macro_claims_card(_claims_release(), tier="P2")
    expected = (
        "<b>📅 Initial Claims · Initial Claims</b>\n"
        "发布日：<code>2026-06-18</code> · 评级：<b>P2</b>\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "  本周: <code>242,000</code>\n"
        "  4W MA: <code>236,500</code> <i>(smoothed)</i>\n"
        "  上周值: <code>228,000</code>\n"
        "  3M 趋势: <i>accelerating</i>\n"
        "\n"
        "<b>🟥 解读</b> <i>(hawkish)</i>\n"
        "  周度初请连续 3 周抬升\n"
        "  🔴 劳动力市场冷却信号"
    )
    assert actual == expected, (
        f"\n--- actual ---\n{actual}\n\n--- expected ---\n{expected}"
    )


# ---------------------------------------------------------------------------
# ⑰ format_macro_weekly_recap byte-equal
# ---------------------------------------------------------------------------

def test_weekly_recap_normal_byte_equal():
    actual = format_macro_weekly_recap(_weekly_recap_normal())
    expected = (
        "<b>📊 宏观周报 · 2026-06-08 → 2026-06-12</b>\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "<b>本周变化</b>\n"
        "  VIX: <code>+2.10 pts</code>\n"
        "  10Y: <code>-5bp</code>\n"
        "  2Y: <code>-2bp</code>\n"
        "  10Y-2Y: <code>-3bp</code>\n"
        "\n"
        "<b>本周已发布</b>\n"
        "  <code>2026-06-10</code>: CPI\n"
        "  <code>2026-06-11</code>: PPI\n"
        "\n"
        "<b>下周预告</b>\n"
        "  <code>2026-06-17</code>: FOMC"
    )
    assert actual == expected, (
        f"\n--- actual ---\n{actual}\n\n--- expected ---\n{expected}"
    )


def test_weekly_recap_quiet_week_byte_equal():
    actual = format_macro_weekly_recap(_weekly_recap_quiet())
    expected = (
        "<b>📊 宏观周报 · 2026-07-06 → 2026-07-10</b>\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "<b>本周变化</b>\n"
        "  VIX: <code>+0.20 pts</code>\n"
        "  10Y: <code>+1bp</code>\n"
        "  2Y: <code>+0bp</code>\n"
        "  10Y-2Y: <code>+1bp</code>\n"
        "\n"
        "<i>本周无 release 触发</i>\n"
        "\n"
        "<i>下周无重大 release</i>"
    )
    assert actual == expected


# ---------------------------------------------------------------------------
# /macro format_macro_dashboard byte-equal
# ---------------------------------------------------------------------------

def test_dashboard_full_data_byte_equal():
    snap = _normal_snapshot()
    window = {
        "2026-06-10": [("CPI", "CPI May 2026", "BLS")],
        "2026-06-11": [("PPI", "PPI May 2026", "BLS")],
        "2026-06-17": [("FOMC", "Jun FOMC + SEP", "Fed")],
        "2026-06-25": [("PCE", "PCE May 2026", "BEA"),
                       ("GDP", "GDP Q1 2026", "BEA")],
    }
    actual = format_macro_dashboard(snap, window, "2026-06-12")
    expected = (
        "<b>🌐 宏观 dashboard · 2026-06-12</b>\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "<b>📊 市场状态</b>\n"
        "  VIX: <code>14.20</code> (+0.80%)\n"
        "  DXY: <code>99.50</code> · WTI: <code>78.40</code> · Gold: <code>2,650.5</code>\n"
        "\n"
        "<b>🏛 收益率</b>\n"
        "  Fed Funds: <code>5.25% – 5.50%</code>\n"
        "  2Y: <code>4.21%</code> · 10Y: <code>4.42%</code>\n"
        "  10-2 spread: <code>+21bp</code>\n"
        "\n"
        "<b>📅 最近 release</b>\n"
        "  <code>2026-06-10</code> · CPI\n"
        "  <code>2026-06-11</code> · PPI\n"
        "\n"
        "<b>📅 下次 release</b>\n"
        "  <code>2026-06-17</code> (5 天后) · FOMC\n"
        "  <code>2026-06-25</code> (13 天后) · PCE / GDP"
    )
    assert actual == expected, (
        f"\n--- actual ---\n{actual}\n\n--- expected ---\n{expected}"
    )


def test_dashboard_partial_data_with_warnings_byte_equal():
    snap = MacroSnapshot(
        snapshot_date="2026-06-12",
        vix=None, vix_pct_change_1d=None,
        dxy=99.50, wti_crude=None, gold=2650.5,
        fed_funds_upper=5.50, fed_funds_lower=5.25,
        dgs2=4.21, dgs10=4.42, t10y2y=0.21, t10y2y_prior=0.18,
        warnings=["yfinance VIX: HTTPError",
                  "yfinance WTI: TimeoutException"],
    )
    actual = format_macro_dashboard(snap, {}, "2026-06-12")
    expected = (
        "<b>🌐 宏观 dashboard · 2026-06-12</b>\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "<b>📊 市场状态</b>\n"
        "  VIX: <code>—</code> (—)\n"
        "  DXY: <code>99.50</code> · WTI: <code>—</code> · Gold: <code>2,650.5</code>\n"
        "\n"
        "<b>🏛 收益率</b>\n"
        "  Fed Funds: <code>5.25% – 5.50%</code>\n"
        "  2Y: <code>4.21%</code> · 10Y: <code>4.42%</code>\n"
        "  10-2 spread: <code>+21bp</code>\n"
        "\n"
        "<i>⚠️ 数据不全:</i>\n"
        "  • <i>yfinance VIX: HTTPError</i>\n"
        "  • <i>yfinance WTI: TimeoutException</i>"
    )
    assert actual == expected


def test_dashboard_spike_curve_flip_visible_byte_equal():
    """VIX spike + curve flip — both tags rendered on the dashboard."""
    actual = format_macro_dashboard(_spike_snapshot(), {}, "2026-06-13")
    expected = (
        "<b>🌐 宏观 dashboard · 2026-06-13</b>\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "<b>📊 市场状态</b>\n"
        "  VIX: <code>28.50</code> (+25.00%) <b>🚨 +20%</b>\n"
        "  DXY: <code>99.50</code> · WTI: <code>78.40</code> · Gold: <code>2,650.5</code>\n"
        "\n"
        "<b>🏛 收益率</b>\n"
        "  Fed Funds: <code>5.25% – 5.50%</code>\n"
        "  2Y: <code>4.21%</code> · 10Y: <code>4.42%</code>\n"
        "  10-2 spread: <code>-10bp</code> <b>📉 今日翻转</b>"
    )
    assert actual == expected


# ---------------------------------------------------------------------------
# Surface contract + v2.reporting identity check
# ---------------------------------------------------------------------------

def test_bot_cards_module_exposes_all_six_formatters():
    """``v2.macro._bot_cards.__all__`` is the contract surface."""
    from v2.macro import _bot_cards
    for name in (
        "format_macro_daily_snapshot",
        "format_macro_release_card",
        "format_macro_fomc_card",
        "format_macro_claims_card",
        "format_macro_weekly_recap",
        "format_macro_dashboard",
    ):
        assert hasattr(_bot_cards, name), f"_bot_cards missing {name}"
        assert name in _bot_cards.__all__, f"{name} not in __all__"


def test_reporting_shim_re_exports_same_identities():
    """v2.reporting.format_macro_* IS v2.macro._bot_cards.format_macro_*.

    Bypasses v2.reporting's heavy package init via importlib so this
    test stays sandbox-runnable (same pattern as Phase 2/3 byte-equal
    tests)."""
    import importlib.util
    import sys as _sys
    import types as _types

    pkg = _types.ModuleType("v2.reporting")
    pkg.__path__ = [str(_REPO_ROOT / "v2" / "reporting")]
    _sys.modules.setdefault("v2.reporting", pkg)

    spec = importlib.util.spec_from_file_location(
        "v2.reporting._macro_formatters",
        _REPO_ROOT / "v2" / "reporting" / "_macro_formatters.py",
    )
    shim = importlib.util.module_from_spec(spec)
    _sys.modules["v2.reporting._macro_formatters"] = shim
    spec.loader.exec_module(shim)

    from v2.macro import _bot_cards as src
    assert shim.format_macro_daily_snapshot is src.format_macro_daily_snapshot
    assert shim.format_macro_release_card is src.format_macro_release_card
    assert shim.format_macro_fomc_card is src.format_macro_fomc_card
    assert shim.format_macro_claims_card is src.format_macro_claims_card
    assert shim.format_macro_weekly_recap is src.format_macro_weekly_recap
    assert shim.format_macro_dashboard is src.format_macro_dashboard
