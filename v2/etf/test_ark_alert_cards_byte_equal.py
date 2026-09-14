"""Phase 5a Stage 3 byte-equal pins for ARK alert formatters.

Same posture as Phase 3 SEC byte-equal / Phase 3.5 insider digest
byte-equal — pin the exact rendered string so a Stage 5 README edit
or unrelated refactor cannot silently shift cron / bot output.

Covers all 4 single-alert action templates + multi-fund banner +
user-universe badge + 3 summary states (normal / partial-fail /
quiet) for a total of 9 pinned cases.

Imports go through ``v2.etf._ark_alert_cards`` (the source-of-truth
module) so the tests stay sandbox-runnable. ``v2.reporting.format_ark_*``
identity is verified via the shim-import assertion at the bottom.
"""

from __future__ import annotations

import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


from v2.etf._ark_alert_cards import (   # noqa: E402
    format_ark_alert,
    format_ark_summary,
)
from v2.etf.alerts import ArkAlert, ArkScanResult   # noqa: E402


# ---------------------------------------------------------------------------
# Fixture builders — shared shape with Stage 2 dry-run for traceability
# ---------------------------------------------------------------------------

def _nvda_new_held() -> ArkAlert:
    return ArkAlert(
        fund="ARKK", ticker="NVDA", company="NVIDIA Corp",
        action="new_position",
        yesterday_weight=None, today_weight=1.85,
        weight_change_relative=1.0,
        shares_change=250_000,
        market_value_usd=31_500_000.0,
        is_in_user_universe=True,
        is_multi_fund=False,
    )


def _irbt_liquidated() -> ArkAlert:
    return ArkAlert(
        fund="ARKQ", ticker="IRBT", company="iRobot Corp",
        action="liquidated",
        yesterday_weight=0.8, today_weight=None,
        weight_change_relative=-1.0,
        shares_change=-60_000,
        market_value_usd=12_100_000.0,
        is_in_user_universe=False,
        is_multi_fund=False,
    )


def _tsla_increase_held() -> ArkAlert:
    return ArkAlert(
        fund="ARKK", ticker="TSLA", company="Tesla Inc",
        action="increase",
        yesterday_weight=8.1, today_weight=10.2,
        weight_change_relative=0.255,
        shares_change=500_000,
        market_value_usd=125_000_000.0,
        is_in_user_universe=True,
        is_multi_fund=False,
    )


def _coin_decrease() -> ArkAlert:
    return ArkAlert(
        fund="ARKF", ticker="COIN", company="Coinbase",
        action="decrease",
        yesterday_weight=5.0, today_weight=3.5,
        weight_change_relative=-0.30,
        shares_change=-120_000,
        market_value_usd=18_000_000.0,
        is_in_user_universe=False,
        is_multi_fund=False,
    )


def _tsmc_multi_fund_held_arkk() -> ArkAlert:
    return ArkAlert(
        fund="ARKK", ticker="TSMC", company="Taiwan Semiconductor",
        action="increase",
        yesterday_weight=2.3, today_weight=3.5,
        weight_change_relative=0.30,
        shares_change=100_000,
        market_value_usd=15_000_000.0,
        is_in_user_universe=True,
        is_multi_fund=True,
    )


def _tsmc_multi_fund_held_arkq() -> ArkAlert:
    return ArkAlert(
        fund="ARKQ", ticker="TSMC", company="Taiwan Semiconductor",
        action="increase",
        yesterday_weight=1.1, today_weight=2.0,
        weight_change_relative=0.40,
        shares_change=80_000,
        market_value_usd=12_000_000.0,
        is_in_user_universe=True,
        is_multi_fund=True,
    )


# ---------------------------------------------------------------------------
# format_ark_alert — 4 action templates
# ---------------------------------------------------------------------------

def test_alert_new_position_held_byte_equal():
    """new_position with user-universe badge, no multi-fund."""
    actual = format_ark_alert(_nvda_new_held())
    expected = (
        "<b>🟢 ARK 新建仓 · NVDA</b>\n"
        "Fund: <code>ARKK</code> · 今日权重: <code>1.85%</code>\n"
        "🟢 持仓股 / 关注列表\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "买入: <code>250,000</code> shares · ≈ <code>$31.5M</code>\n"
        "昨日: 未持有"
    )
    assert actual == expected, (
        f"\n--- actual ---\n{actual}\n--- expected ---\n{expected}"
    )


def test_alert_liquidated_byte_equal():
    """liquidated card — no today_weight on header line, '完全清仓' body."""
    actual = format_ark_alert(_irbt_liquidated())
    expected = (
        "<b>🔴 ARK 清仓 · IRBT</b>\n"
        "Fund: <code>ARKQ</code>\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "昨日权重: <code>0.80%</code> · <code>$12.1M</code>\n"
        "今日: 完全清仓"
    )
    assert actual == expected, (
        f"\n--- actual ---\n{actual}\n--- expected ---\n{expected}"
    )


def test_alert_increase_held_byte_equal():
    """increase card with signed (+25.5%) relative + user-universe badge."""
    actual = format_ark_alert(_tsla_increase_held())
    expected = (
        "<b>📈 ARK 增持 · TSLA</b>\n"
        "Fund: <code>ARKK</code> · 今日权重: <code>10.20%</code> (+25.5%)\n"
        "🟢 持仓股 / 关注列表\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "增持: <code>500,000</code> shares · ≈ <code>$125.0M</code>\n"
        "昨日权重: <code>8.10%</code>"
    )
    assert actual == expected


def test_alert_decrease_byte_equal():
    """decrease card — signed (-30.0%) relative, shares displayed as
    abs magnitude, no user-universe badge."""
    actual = format_ark_alert(_coin_decrease())
    expected = (
        "<b>📉 ARK 减持 · COIN</b>\n"
        "Fund: <code>ARKF</code> · 今日权重: <code>3.50%</code> (-30.0%)\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "减持: <code>120,000</code> shares · ≈ <code>$18.0M</code>\n"
        "昨日权重: <code>5.00%</code>"
    )
    assert actual == expected


def test_alert_multi_fund_banner_before_badge_byte_equal():
    """Polish 1 pin — 🚨 multi-fund banner rendered ABOVE 🟢 user-universe
    badge, both rendered ABOVE the ━━━ divider."""
    actual = format_ark_alert(_tsmc_multi_fund_held_arkk())
    expected = (
        "<b>📈 ARK 增持 · TSMC</b>\n"
        "Fund: <code>ARKK</code> · 今日权重: <code>3.50%</code> (+30.0%)\n"
        "🚨 <i>多 Fund 协同（详见总览）</i>\n"
        "🟢 持仓股 / 关注列表\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "增持: <code>100,000</code> shares · ≈ <code>$15.0M</code>\n"
        "昨日权重: <code>2.30%</code>"
    )
    assert actual == expected, (
        f"\n--- actual ---\n{actual}\n--- expected ---\n{expected}"
    )


# ---------------------------------------------------------------------------
# format_ark_summary — 3 states (normal / partial-failure / quiet)
# ---------------------------------------------------------------------------

def test_summary_normal_day_byte_equal():
    """Polish 2 + Polish 3 pin — 6 alerts (2 multi-fund TSMC + NVDA +
    TSLA + IRBT + COIN), buys-then-sells action ordering, (4/4 ARK
    funds) coverage fraction."""
    result = ArkScanResult(
        scan_date="2026-06-09",
        funds_scanned=["ARKK", "ARKW", "ARKG", "ARKF"],
        funds_attempted=["ARKK", "ARKW", "ARKG", "ARKF"],
        alerts=[
            _nvda_new_held(),
            _irbt_liquidated(),
            _tsla_increase_held(),
            _tsmc_multi_fund_held_arkk(),
            _tsmc_multi_fund_held_arkq(),
            _coin_decrease(),
        ],
        warnings=[],
    )
    actual = format_ark_summary(result)
    expected = (
        "<b>🔔 ARK 调仓总览 · 2026-06-09</b>\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "<b>本日触发 alerts: 6</b>\n"
        "  🚨 多 Fund 协同: <code>2</code>\n"
        "  🟢 涉及持仓 / 关注: <code>4</code>\n"
        "\n"
        "<b>行动分布</b>\n"
        "  🟢 新建仓: <code>1</code>\n"
        "  📈 增持: <code>3</code>\n"
        "  🔴 清仓: <code>1</code>\n"
        "  📉 减持: <code>1</code>\n"
        "\n"
        "<b>涉及 user universe:</b> NVDA · TSLA · TSMC <i>(3 个)</i>\n"
        "\n"
        "<b>多 Fund 协同</b>\n"
        "  <b>TSMC</b>: ARKK + ARKQ\n"
        "\n"
        "<b>本日扫描 funds:</b> ARKK / ARKW / ARKG / ARKF <i>(4/4 ARK funds)</i>"
    )
    assert actual == expected, (
        f"\n--- actual ---\n{actual}\n--- expected ---\n{expected}"
    )


def test_summary_partial_failure_byte_equal():
    """ARKG failed → funds_scanned=3, funds_attempted=4 → (3/4) + warnings line."""
    result = ArkScanResult(
        scan_date="2026-06-09",
        funds_scanned=["ARKK", "ARKW", "ARKF"],
        funds_attempted=["ARKK", "ARKW", "ARKG", "ARKF"],
        alerts=[
            _nvda_new_held(),
            _irbt_liquidated(),
            _tsla_increase_held(),
        ],
        warnings=["ARKG fetch failed: HTTP 503"],
    )
    actual = format_ark_summary(result)
    expected = (
        "<b>🔔 ARK 调仓总览 · 2026-06-09</b>\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "<b>本日触发 alerts: 3</b>\n"
        "  🟢 涉及持仓 / 关注: <code>2</code>\n"
        "\n"
        "<b>行动分布</b>\n"
        "  🟢 新建仓: <code>1</code>\n"
        "  📈 增持: <code>1</code>\n"
        "  🔴 清仓: <code>1</code>\n"
        "\n"
        "<b>涉及 user universe:</b> NVDA · TSLA <i>(2 个)</i>\n"
        "\n"
        "<b>本日扫描 funds:</b> ARKK / ARKW / ARKF <i>(3/4 ARK funds)</i>\n"
        "\n"
        "<i>warnings: 1</i>"
    )
    assert actual == expected, (
        f"\n--- actual ---\n{actual}\n--- expected ---\n{expected}"
    )


def test_summary_quiet_day_byte_equal():
    """Empty alerts (all 4 funds scanned successfully, just no
    significant rebalances) — operator-visibility floor placeholder."""
    result = ArkScanResult(
        scan_date="2026-06-09",
        funds_scanned=["ARKK", "ARKW", "ARKG", "ARKF"],
        funds_attempted=["ARKK", "ARKW", "ARKG", "ARKF"],
        alerts=[],
        warnings=[],
    )
    actual = format_ark_summary(result)
    expected = (
        "<b>🔔 ARK 调仓总览 · 2026-06-09</b>\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "<i>本日 ARK 调仓平静（0 触发）</i>\n"
        "已扫描: <code>ARKK / ARKW / ARKG / ARKF</code> "
        "<i>(4/4 ARK funds)</i>"
    )
    assert actual == expected, (
        f"\n--- actual ---\n{actual}\n--- expected ---\n{expected}"
    )


# ---------------------------------------------------------------------------
# HTML safety — ticker / company / fund must be html.escape'd
# ---------------------------------------------------------------------------

def test_alert_html_escapes_malicious_ticker():
    """A pathological ticker with HTML control chars should be escaped
    everywhere it appears in the card (header, fund line is fund-only)."""
    alert = ArkAlert(
        fund="ARKK", ticker="A<script>", company="Evil Co",
        action="new_position",
        yesterday_weight=None, today_weight=1.5,
        weight_change_relative=1.0,
        shares_change=10_000, market_value_usd=1_000_000.0,
        is_in_user_universe=False, is_multi_fund=False,
    )
    text = format_ark_alert(alert)
    # raw "<script>" must NOT appear; escaped form must
    assert "<script>" not in text
    assert "&lt;script&gt;" in text


def test_summary_html_escapes_malicious_ticker_in_unusual_block():
    """Summary's "涉及 user universe" block joins tickers — must escape."""
    alert = ArkAlert(
        fund="ARKK", ticker="A&B", company="Amp",
        action="new_position",
        yesterday_weight=None, today_weight=2.0,
        weight_change_relative=1.0,
        shares_change=10_000, market_value_usd=1_000_000.0,
        is_in_user_universe=True, is_multi_fund=False,
    )
    result = ArkScanResult(
        scan_date="2026-06-09",
        funds_scanned=["ARKK"],
        funds_attempted=["ARKK"],
        alerts=[alert],
        warnings=[],
    )
    text = format_ark_summary(result)
    # The literal "A&B" must not appear unescaped
    assert "A&amp;B" in text
    # Pin that the unescaped form isn't there as a standalone segment.
    # ("A&B" appearing inside "A&amp;B" trivially matches; check for the
    # naked '&' between A and B as a substring instead.)
    assert "A&B" not in text or text.count("A&amp;B") == text.count("A&B")


# ---------------------------------------------------------------------------
# Public-API shim identity (4-layer pattern consistency)
# ---------------------------------------------------------------------------

def test_reporting_shim_re_exports_same_identities():
    """v2.reporting.format_ark_* must be the SAME function objects as
    v2.etf._ark_alert_cards.* — Phase 1-4 pattern. Sandbox-safe load
    via importlib.spec to bypass v2.reporting's heavy __init__."""
    import importlib.util
    import sys as _sys
    import types as _types

    pkg = _types.ModuleType("v2.reporting")
    pkg.__path__ = [str(_REPO_ROOT / "v2" / "reporting")]
    _sys.modules.setdefault("v2.reporting", pkg)

    spec = importlib.util.spec_from_file_location(
        "v2.reporting._ark_alert_formatters",
        _REPO_ROOT / "v2" / "reporting" / "_ark_alert_formatters.py",
    )
    shim = importlib.util.module_from_spec(spec)
    _sys.modules["v2.reporting._ark_alert_formatters"] = shim
    spec.loader.exec_module(shim)

    from v2.etf import _ark_alert_cards as src
    assert shim.format_ark_alert is src.format_ark_alert
    assert shim.format_ark_summary is src.format_ark_summary
