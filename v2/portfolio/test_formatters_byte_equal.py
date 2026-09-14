"""Byte-equal pin tests for the 4 public portfolio card formatters.

Mirrors Phase 1's earnings byte-equal pattern. The fixtures here are
exactly the Stage 2.5 dry-run synthetic data, with the Stage 5 sign
fix applied to drawdown (now non-negative magnitudes).

Imports go through the v2.portfolio source-of-truth module to keep the
tests sandbox-runnable. v2.reporting.format_portfolio_* is the same
function (verified via identity assertion at the bottom).
"""

from __future__ import annotations

import re
import sys
from datetime import date
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


from v2.portfolio._bot_cards import (   # noqa: E402
    format_pnl_period,
    format_risk_card,
    format_risk_view,
    format_weekly_card,
)
from v2.portfolio.models import (   # noqa: E402
    ConcentrationMetrics, DrawdownMetrics, EarningsRiskItem,
    ExposureMetrics, PnLMetrics, PositionFlat, RiskReport,
)


# ---------------------------------------------------------------------------
# Fixtures — Stage 2.5 dry-run synthetic data
# ---------------------------------------------------------------------------

def _risk_report_full() -> RiskReport:
    """The ⑨ multi-factor → P0 case from Stage 2.5."""
    r = RiskReport(
        snapshot_date="2026-06-04",
        portfolio_value=128_600.0,
        cash=25_400.0,
        cash_pct=25_400 / 128_600,
        concentration=ConcentrationMetrics(
            top_1_pct=0.35, top_3_pct=0.52, top_5_pct=0.65,
            hhi=0.18, n_positions=8,
        ),
        exposure=ExposureMetrics(
            by_sector={"SMH": 0.38, "XLK": 0.30, "XLF": 0.15, "OTHER": 0.17},
            largest_sector="SMH", largest_sector_pct=0.38,
        ),
        pnl=PnLMetrics(
            daily_pnl=-1856.0, daily_pnl_pct=-0.0178,
            weekly_pnl_pct=-0.023, monthly_pnl_pct=-0.034,
        ),
        # Stage 5: non-negative magnitudes.
        drawdown=DrawdownMetrics(
            current_drawdown_pct=0.034,
            max_drawdown_pct=0.12,
            peak_value=107_000.0,
            peak_date=date(2026, 6, 1),
        ),
        earnings_risk_next_7d=[
            EarningsRiskItem(ticker="AAPL", release_date="2026-06-07",
                             days_until=3, estimated_eps=1.52,
                             estimated_revenue=None),
            EarningsRiskItem(ticker="NVDA", release_date="2026-06-08",
                             days_until=4, estimated_eps=0.65,
                             estimated_revenue=None),
        ],
        warnings=[],
    )
    r.positions = [
        PositionFlat("NVDA", 36120, 0.35, "SMH"),
        PositionFlat("AAPL", 20640, 0.20, "XLK"),
        PositionFlat("JPM",  15480, 0.15, "XLF"),
    ]
    return r


def _risk_report_broad_etf() -> RiskReport:
    """Phase 2.5-mini fixture: real-account shape from 2026-06-04.

    IVV-heavy book (86%) with some SMH-class exposure. Exercises:
    - sector_bucket_for(IVV) → "BROAD" (not "OTHER" anymore)
    - _SECTOR_NAME["BROAD"] → "大盘ETF" label rendering
    - _is_broad_concentration() → True (both Top 1 and largest sector
      trip their BROAD branch)
    - Clarifier note "注：BROAD ETF 内部已分散" appended after alerts.
    """
    r = RiskReport(
        snapshot_date="2026-06-04",
        portfolio_value=100_239.0,
        cash=12_123.0,
        cash_pct=12_123 / 100_239,
        concentration=ConcentrationMetrics(
            top_1_pct=0.863, top_3_pct=0.977, top_5_pct=0.977,
            hhi=0.75, n_positions=2,
        ),
        exposure=ExposureMetrics(
            by_sector={"BROAD": 0.866, "SMH": 0.134},
            largest_sector="BROAD", largest_sector_pct=0.866,
        ),
        pnl=PnLMetrics(
            daily_pnl=239.0, daily_pnl_pct=0.0024,
            weekly_pnl_pct=None, monthly_pnl_pct=None,
        ),
        drawdown=DrawdownMetrics(
            current_drawdown_pct=0.0,
            max_drawdown_pct=0.0,
            peak_value=100_000.0,
            peak_date=date(2026, 5, 29),
        ),
        earnings_risk_next_7d=[],
        warnings=[],
    )
    r.positions = [
        PositionFlat("IVV",  76_117.0, 0.863, "BROAD"),
        PositionFlat("NVDA", 11_815.0, 0.134, "SMH"),
    ]
    return r


def _risk_report_weekly_clean() -> RiskReport:
    """The ⑩ clean-week case from Stage 2.5."""
    r = RiskReport(
        snapshot_date="2026-06-06",
        portfolio_value=130_000.0,
        cash=25_400.0,
        cash_pct=25_400 / 130_000,
        concentration=ConcentrationMetrics(
            top_1_pct=0.22, top_3_pct=0.48, top_5_pct=0.62,
            hhi=0.14, n_positions=8,
        ),
        exposure=ExposureMetrics(
            by_sector={"SMH": 0.30, "XLK": 0.35, "XLF": 0.18, "OTHER": 0.17},
            largest_sector="XLK", largest_sector_pct=0.35,
        ),
        pnl=PnLMetrics(
            daily_pnl=320.0, daily_pnl_pct=0.0031,
            weekly_pnl_pct=0.015, monthly_pnl_pct=0.028,
        ),
        drawdown=DrawdownMetrics(
            current_drawdown_pct=0.005,
            max_drawdown_pct=0.018,
            peak_value=105_100.0,
            peak_date=date(2026, 6, 3),
        ),
        earnings_risk_next_7d=[],
        warnings=[],
    )
    r.positions = [
        PositionFlat("NVDA", 23000, 0.22, "SMH"),
        PositionFlat("MSFT", 18000, 0.17, "XLK"),
    ]
    return r


# ---------------------------------------------------------------------------
# format_risk_card — ⑨ multi-factor → P0 byte-equal
# ---------------------------------------------------------------------------

def test_risk_card_byte_equal_snapshot():
    """Pins the exact card body for the ⑨ multi-factor → P0 case.

    Drawdown is rendered as ``-3.40%`` / ``-12.00%`` (Stage 5 sign fix:
    drawdown is always a loss, renderer always prepends ``-``).
    """
    actual = format_risk_card(_risk_report_full())

    expected = (
        "<b>💼 组合风险 · 2026-06-04</b>\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "组合价值 <code>$128.6K</code> "
        "(持仓 <code>$103.2K</code> · 现金 <code>$25.4K</code>, 19.8%)\n"
        "今日 P/L 🔴 -1.78% (-<code>$1.9K</code>)\n"
        "本周 -2.30% · 本月 -3.40%\n"
        "<b>📊 集中度</b>\n"
        "  Top 1: <b>NVDA</b> 35.0% ⚠️\n"
        "  Top 5: 65.0%\n"
        "  HHI: 0.18 (中等集中)\n"
        "<b>🏭 行业暴露</b>\n"
        "  SMH (半导体): 38.0% ⚠️\n"
        "  XLK (科技): 30.0% ⚠️\n"
        "  OTHER (其他): 17.0%\n"
        "  XLF (金融): 15.0%\n"
        "<b>📉 回撤 (1M)</b> 当前 -3.40% · "
        "最大 -12.00% (峰值 <code>$107.0K</code> @ 2026-06-01)\n"
        "<b>📅 未来 7 天财报风险</b> (2 只)\n"
        "  <code>2026-06-07</code> <b>AAPL</b> (D-3)\n"
        "  <code>2026-06-08</code> <b>NVDA</b> (D-4)\n"
        "⚠️ <i>单票 NVDA > 30% / SMH 行业 > 30% / 1M 回撤 > 10%</i>"
    )

    assert actual == expected, (
        f"\n--- actual ---\n{actual}\n\n--- expected ---\n{expected}"
    )


def test_risk_card_broad_etf_byte_equal_snapshot():
    """Phase 2.5-mini: BROAD ETF concentration renders the clarifier note.

    Account is 86% IVV — concentration alerts fire as before, but the
    card now (a) shows ``BROAD (大盘ETF)`` instead of ``OTHER (其他)``
    in the exposure block, and (b) appends a single italic note line
    after the alert footer.
    """
    actual = format_risk_card(_risk_report_broad_etf())

    expected = (
        "<b>💼 组合风险 · 2026-06-04</b>\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "组合价值 <code>$100.2K</code> "
        "(持仓 <code>$88.1K</code> · 现金 <code>$12.1K</code>, 12.1%)\n"
        "今日 P/L 🟢 +0.24% (+<code>$239</code>)\n"
        "<b>📊 集中度</b>\n"
        "  Top 1: <b>IVV</b> 86.3% ⚠️\n"
        "  Top 5: 97.7%\n"
        "  HHI: 0.75 (高度集中)\n"
        "<b>🏭 行业暴露</b>\n"
        "  BROAD (大盘ETF): 86.6% ⚠️\n"
        "  SMH (半导体): 13.4%\n"
        "<b>📉 回撤 (1M)</b> 当前 0.00% · 最大 0.00% "
        "(峰值 <code>$100.0K</code> @ 2026-05-29)\n"
        "⚠️ <i>单票 IVV > 30% / BROAD 行业 > 30%</i>\n"
        "<i>注：BROAD ETF 内部已分散，集中度风险不等同于单股</i>"
    )

    assert actual == expected, (
        f"\n--- actual ---\n{actual}\n\n--- expected ---\n{expected}"
    )


def test_risk_view_is_byte_equal_to_risk_card():
    """The bot ``/risk`` card and the cron ⑨ card MUST render the same
    body — priority chips are added by the notifier layer, not the
    formatter. Stage 5 makes risk_view a thin alias."""
    r = _risk_report_full()
    assert format_risk_view(r) == format_risk_card(r)


# ---------------------------------------------------------------------------
# format_weekly_card — ⑩ clean week byte-equal
# ---------------------------------------------------------------------------

def test_weekly_card_byte_equal_snapshot():
    """Clean week, attribution arg defaulted to None → no attribution
    block rendered (Phase 2.5 full silent-when-fresh contract)."""
    actual = format_weekly_card(_risk_report_weekly_clean())

    expected = (
        "<b>📊 周 P&amp;L 复盘 · 2026-06-06</b>\n"
        "<i>(截至昨日收盘的口径)</i>\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "组合价值 <code>$130.0K</code> "
        "(持仓 <code>$104.6K</code> · 现金 <code>$25.4K</code>, 19.5%)\n"
        "<b>本周回报</b> 🟢 +1.50%\n"
        "<b>本月回报</b> 🟢 +2.80%\n"
        "<b>📉 1M 最大回撤</b> -1.80%\n"
        "  峰值 <code>$105.1K</code> @ <code>2026-06-03</code>\n"
        "  当前距峰 -0.50%\n"
        "<b>🏭 主要行业暴露</b>\n"
        "  XLK: 35.0%\n"
        "  SMH: 30.0%\n"
        "  XLF: 18.0%"
    )

    assert actual == expected, (
        f"\n--- actual ---\n{actual}\n\n--- expected ---\n{expected}"
    )


def test_weekly_card_with_full_attribution_byte_equal():
    """5+ days snapshot → best / worst / net rows rendered."""
    from v2.portfolio.snapshot import AttributionItem

    attribution = [
        AttributionItem(ticker="NVDA", avg_weight=0.30,
                        weekly_return=0.05, contribution=0.015),
        AttributionItem(ticker="MSFT", avg_weight=0.20,
                        weekly_return=0.01, contribution=0.002),
        AttributionItem(ticker="JPM",  avg_weight=0.15,
                        weekly_return=-0.02, contribution=-0.003),
    ]
    actual = format_weekly_card(
        _risk_report_weekly_clean(),
        attribution=attribution,
        snapshot_days_available=5,
    )
    expected = (
        "<b>📊 周 P&amp;L 复盘 · 2026-06-06</b>\n"
        "<i>(截至昨日收盘的口径)</i>\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "组合价值 <code>$130.0K</code> "
        "(持仓 <code>$104.6K</code> · 现金 <code>$25.4K</code>, 19.5%)\n"
        "<b>本周回报</b> 🟢 +1.50%\n"
        "<b>本月回报</b> 🟢 +2.80%\n"
        "<b>📉 1M 最大回撤</b> -1.80%\n"
        "  峰值 <code>$105.1K</code> @ <code>2026-06-03</code>\n"
        "  当前距峰 -0.50%\n"
        "<b>🏭 主要行业暴露</b>\n"
        "  XLK: 35.0%\n"
        "  SMH: 30.0%\n"
        "  XLF: 18.0%\n"
        "<b>📊 本周 per-position 表现归因</b>\n"
        "  最佳: <b>NVDA</b> 🟢 +5.00% (贡献 🟢 +1.50%)\n"
        "  最差: <b>JPM</b> 🔴 -2.00% (贡献 🔴 -0.30%)\n"
        "  净贡献: <code>🟢 +1.40%</code>"
    )
    assert actual == expected, (
        f"\n--- actual ---\n{actual}\n\n--- expected ---\n{expected}"
    )


def test_weekly_card_insufficient_snapshots_byte_equal():
    """3-day window (< WEEKLY_MIN_DAYS=5) → italic '累积中' fallback."""
    from v2.portfolio.snapshot import AttributionItem

    attribution = [
        AttributionItem(ticker="NVDA", avg_weight=0.30,
                        weekly_return=0.03, contribution=0.009),
    ]
    actual = format_weekly_card(
        _risk_report_weekly_clean(),
        attribution=attribution,
        snapshot_days_available=3,
    )
    expected = (
        "<b>📊 周 P&amp;L 复盘 · 2026-06-06</b>\n"
        "<i>(截至昨日收盘的口径)</i>\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "组合价值 <code>$130.0K</code> "
        "(持仓 <code>$104.6K</code> · 现金 <code>$25.4K</code>, 19.5%)\n"
        "<b>本周回报</b> 🟢 +1.50%\n"
        "<b>本月回报</b> 🟢 +2.80%\n"
        "<b>📉 1M 最大回撤</b> -1.80%\n"
        "  峰值 <code>$105.1K</code> @ <code>2026-06-03</code>\n"
        "  当前距峰 -0.50%\n"
        "<b>🏭 主要行业暴露</b>\n"
        "  XLK: 35.0%\n"
        "  SMH: 30.0%\n"
        "  XLF: 18.0%\n"
        "<i>归因数据累积中 (3/5 天)</i>"
    )
    assert actual == expected, (
        f"\n--- actual ---\n{actual}\n\n--- expected ---\n{expected}"
    )


def test_weekly_card_no_snapshots_silent():
    """0 snapshots (fresh account / cron never ran) → no block, no italic.

    Phase 2.5 full contract: don't apologise for missing data on a
    fresh account — silently omit the section so the card reads cleanly
    on the first Friday after deploy."""
    actual = format_weekly_card(
        _risk_report_weekly_clean(),
        attribution=[],
        snapshot_days_available=0,
    )
    # Same as the default test_weekly_card_byte_equal_snapshot output
    assert "归因" not in actual
    assert "per-position" not in actual


# ---------------------------------------------------------------------------
# format_pnl_period — bot /pnl week / /pnl month
# ---------------------------------------------------------------------------

def test_pnl_period_week_byte_equal():
    metrics = PnLMetrics(
        daily_pnl=320.0, daily_pnl_pct=0.0031,
        weekly_pnl_pct=0.015, monthly_pnl_pct=0.028,
    )
    actual = format_pnl_period("week", metrics)
    expected = (
        "<b>📊 本周 P&amp;L 摘要</b>\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "本周回报：🟢 <code>+1.50%</code>\n"
        "<i>(参考 · 今日 🟢 +0.31%)</i>"
    )
    assert actual == expected, f"\n--- actual ---\n{actual}\n\n--- expected ---\n{expected}"


def test_pnl_period_month_byte_equal():
    metrics = PnLMetrics(
        daily_pnl=-1856.0, daily_pnl_pct=-0.0178,
        weekly_pnl_pct=-0.023, monthly_pnl_pct=-0.034,
    )
    actual = format_pnl_period("month", metrics)
    expected = (
        "<b>📊 本月 P&amp;L 摘要</b>\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "本月回报：🔴 <code>-3.40%</code>\n"
        "<i>(参考 · 今日 🔴 -1.78%)</i>"
    )
    assert actual == expected, f"\n--- actual ---\n{actual}\n\n--- expected ---\n{expected}"


def test_pnl_period_week_data_insufficient():
    """Account < 5 trading days → '数据不足' line, daily context still shown."""
    metrics = PnLMetrics(
        daily_pnl=50.0, daily_pnl_pct=0.005,
        weekly_pnl_pct=None, monthly_pnl_pct=None,
    )
    actual = format_pnl_period("week", metrics)
    expected = (
        "<b>📊 本周 P&amp;L 摘要</b>\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "<i>数据不足（账户历史少于 5 个交易日）</i>\n"
        "<i>(参考 · 今日 🟢 +0.50%)</i>"
    )
    assert actual == expected


def test_pnl_period_day_raises_value_error():
    """'day' should NOT route here — caller uses format_pnl from
    v2.reporting instead. Misuse must surface immediately."""
    metrics = PnLMetrics(daily_pnl=0.0, daily_pnl_pct=0.0,
                         weekly_pnl_pct=None, monthly_pnl_pct=None)
    with pytest.raises(ValueError, match="format_pnl"):
        format_pnl_period("day", metrics)


# ---------------------------------------------------------------------------
# HTML safety lint — all formatter outputs must be Telegram-parseable
# ---------------------------------------------------------------------------

_UNESCAPED_LT = re.compile(r"<[^a-zA-Z/!]")


def _pnl_metrics_week_full() -> PnLMetrics:
    return PnLMetrics(daily_pnl=320.0, daily_pnl_pct=0.0031,
                      weekly_pnl_pct=0.015, monthly_pnl_pct=0.028)


def _pnl_metrics_month_full() -> PnLMetrics:
    return PnLMetrics(daily_pnl=-1856.0, daily_pnl_pct=-0.0178,
                      weekly_pnl_pct=-0.023, monthly_pnl_pct=-0.034)


def _pnl_metrics_insufficient() -> PnLMetrics:
    return PnLMetrics(daily_pnl=50.0, daily_pnl_pct=0.005,
                      weekly_pnl_pct=None, monthly_pnl_pct=None)


@pytest.mark.parametrize("case_name, render", [
    ("risk_card_full",
     lambda: format_risk_card(_risk_report_full())),
    ("risk_card_broad_etf",
     lambda: format_risk_card(_risk_report_broad_etf())),
    ("risk_view_full",
     lambda: format_risk_view(_risk_report_full())),
    ("weekly_card_clean",
     lambda: format_weekly_card(_risk_report_weekly_clean())),
    ("pnl_period_week_full",
     lambda: format_pnl_period("week", _pnl_metrics_week_full())),
    ("pnl_period_month_full",
     lambda: format_pnl_period("month", _pnl_metrics_month_full())),
    ("pnl_period_week_insufficient",
     lambda: format_pnl_period("week", _pnl_metrics_insufficient())),
    ("pnl_period_month_insufficient",
     lambda: format_pnl_period("month", _pnl_metrics_insufficient())),
])
def test_formatter_output_html_safe(case_name, render):
    """No formatter output may contain ``<`` followed by a non-tag char.

    Telegram's HTML parser rejects such input with ``BadRequest: Can't parse
    entities``. With no error handler registered the placeholder message hangs
    forever and the bot appears frozen.

    Regression for the 2026-06-04 hot patch ``5f61795`` (``账户历史 < N 个交易日``).
    Any new formatter that ships a literal ``<`` next to whitespace, digits,
    or punctuation will fail this lint — fix by using ``&lt;`` or a Chinese
    word like ``少于``.
    """
    rendered = render()
    bad = _UNESCAPED_LT.findall(rendered)
    assert not bad, (
        f"{case_name}: unescaped '<' found: {bad}\n"
        f"Fix: replace literal '<' with '&lt;' or a Chinese alternative.\n"
        f"--- rendered ---\n{rendered}"
    )


# ---------------------------------------------------------------------------
# Surface: all 4 public names exposed under both module paths
# ---------------------------------------------------------------------------

def test_bot_cards_module_exposes_all_four_formatters():
    """``v2.portfolio._bot_cards.__all__`` is the contract surface."""
    from v2.portfolio import _bot_cards
    for name in (
        "format_risk_card",
        "format_risk_view",
        "format_weekly_card",
        "format_pnl_period",
    ):
        assert hasattr(_bot_cards, name), f"_bot_cards missing {name}"
        assert name in _bot_cards.__all__, f"{name} not in __all__"


def test_reporting_shim_re_exports_same_identities():
    """Bypass v2.reporting's heavy package init via importlib so this
    test stays sandbox-runnable. Asserts the public re-exports point
    at the same function objects as the source-of-truth module."""
    import importlib.util
    import sys as _sys
    import types as _types

    pkg = _types.ModuleType("v2.reporting")
    pkg.__path__ = [str(_REPO_ROOT / "v2" / "reporting")]
    _sys.modules.setdefault("v2.reporting", pkg)

    spec = importlib.util.spec_from_file_location(
        "v2.reporting._portfolio_formatters",
        _REPO_ROOT / "v2" / "reporting" / "_portfolio_formatters.py",
    )
    shim = importlib.util.module_from_spec(spec)
    _sys.modules["v2.reporting._portfolio_formatters"] = shim
    spec.loader.exec_module(shim)

    from v2.portfolio import _bot_cards as src
    assert shim.format_portfolio_risk_card is src.format_risk_card
    assert shim.format_portfolio_risk_view is src.format_risk_view
    assert shim.format_portfolio_weekly_card is src.format_weekly_card
    assert shim.format_portfolio_pnl_period is src.format_pnl_period
