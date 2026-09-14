"""Byte-equal pin tests for the 5 public SEC card formatters.

Mirrors Phase 1's earnings byte-equal pattern and Phase 2's portfolio
byte-equal pattern. Pin the exact card body so a Stage 7 README edit
or any unrelated refactor cannot silently shift the cron / bot output.

Imports go through the v2.sec source-of-truth module to keep the tests
sandbox-runnable. v2.reporting.format_sec_* is the same function
(verified via identity assertion at the bottom).
"""

from __future__ import annotations

import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


from v2.sec._bot_cards import (   # noqa: E402
    format_sec_8k_card,
    format_sec_8k_view,
    format_sec_form4_cluster_card,
    format_sec_form4_individual_card,
    format_sec_form4_view,
    format_sec_insider_digest,
)
from v2.sec.insider_digest import WeeklyInsiderSummary   # noqa: E402
from v2.sec.models import (   # noqa: E402
    EightKEvent, EightKItem, Form4Cluster, Form4Transaction, SecFiling,
)


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

def _hpe_multi_item_event() -> EightKEvent:
    """HPE Stage 2 dry-run: 1.01 + 2.02 + 5.02 + 7.01 + 9.01."""
    return EightKEvent(
        filing=SecFiling(
            ticker="HPE", cik="0001645590", form="8-K",
            filing_date="2026-06-04",
            accession_number="0000123-26-000001",
        ),
        items=[
            EightKItem("1.01", "P1", "重大商业合约 (新签)", {}),
            EightKItem("2.02", "P2", "财报数据", {}),
            EightKItem("5.02", "P0", "高管 / 董事会变动", {
                "departures": [{
                    "name": "John Smith",
                    "title": "Chief Executive Officer",
                }],
                "appointments": [{
                    "name": "Jane Doe",
                    "title": "Interim Chief Executive Officer",
                }],
                "has_senior_exec": True,
            }),
            EightKItem("7.01", "P2", "Reg FD 自愿披露", {}),
            EightKItem("9.01", "P3", "财务报表 / 附件", {}),
        ],
    )


def _only_101_event() -> EightKEvent:
    """Single 1.01 (material agreement) event."""
    return EightKEvent(
        filing=SecFiling(
            ticker="AAPL", cik="0000320193", form="8-K",
            filing_date="2026-06-04",
            accession_number="0000123-26-000100",
        ),
        items=[
            EightKItem("1.01", "P1", "重大商业合约 (新签)", {}),
        ],
    )


def _llm_fail_5_02_event() -> EightKEvent:
    """5.02 item with empty extracted_meta — LLM failure path."""
    return EightKEvent(
        filing=SecFiling(
            ticker="XYZ", cik="0000099", form="8-K",
            filing_date="2026-06-04",
            accession_number="ACC-99",
        ),
        items=[
            EightKItem("5.02", "P1", "高管 / 董事会变动", {}),
        ],
    )


def _2_02_only_event() -> EightKEvent:
    """Pure earnings 8-K — 2.02 + 9.01 only."""
    return EightKEvent(
        filing=SecFiling(
            ticker="MSFT", cik="0000789019", form="8-K",
            filing_date="2026-06-04",
            accession_number="ACC-2-02",
        ),
        items=[
            EightKItem("2.02", "P2", "财报数据", {}),
            EightKItem("9.01", "P3", "财务报表 / 附件", {}),
        ],
    )


# ---------------------------------------------------------------------------
# format_sec_8k_card — cron ⑪ byte-equal
# ---------------------------------------------------------------------------

def test_8k_card_p0_5_02_byte_equal():
    """HPE multi-item filing → single P0 card with 5.02 抽取 sub-block."""
    actual = format_sec_8k_card(
        _hpe_multi_item_event(), is_held=True, is_watchlist=False,
    )
    expected = (
        "<b>🚨 SEC 8-K · HPE · P0</b>\n"
        "申报日：<code>2026-06-04</code>\n"
        "🟢 持仓股\n"
        "\n"
        "<b>项目</b>\n"
        "  📋 <code>1.01</code> [P1] 重大商业合约 (新签)\n"
        "  📎 <code>2.02</code> [P2] 财报数据  <i>(数据由 ⑧ 处理)</i>\n"
        "  🚨 <code>5.02</code> [P0] 高管 / 董事会变动\n"
        "  📎 <code>7.01</code> [P2] Reg FD 自愿披露\n"
        "  📌 <code>9.01</code> [P3] 财务报表 / 附件\n"
        "\n"
        "<b>5.02 抽取</b>\n"
        "  📤 离职：John Smith (Chief Executive Officer)\n"
        "  📥 任命：Jane Doe (Interim Chief Executive Officer)"
    )
    assert actual == expected, (
        f"\n--- actual ---\n{actual}\n\n--- expected ---\n{expected}"
    )


def test_8k_card_p1_only_byte_equal():
    """Single 1.01 filing — no 5.02 抽取 block."""
    actual = format_sec_8k_card(
        _only_101_event(), is_held=True, is_watchlist=False,
    )
    expected = (
        "<b>📋 SEC 8-K · AAPL · P1</b>\n"
        "申报日：<code>2026-06-04</code>\n"
        "🟢 持仓股\n"
        "\n"
        "<b>项目</b>\n"
        "  📋 <code>1.01</code> [P1] 重大商业合约 (新签)"
    )
    assert actual == expected


def test_8k_card_2_02_only_renders():
    """2.02+9.01 filing still renders — the cron skips 2.02-only via
    EightKEvent.is_2_02_only BEFORE invoking the formatter, so the
    formatter itself never needs to gate. This test pins that the
    formatter is content-blind: it renders whatever it gets."""
    actual = format_sec_8k_card(
        _2_02_only_event(), is_held=False, is_watchlist=False,
    )
    expected = (
        "<b>📎 SEC 8-K · MSFT · P2</b>\n"
        "申报日：<code>2026-06-04</code>\n"
        "\n"
        "<b>项目</b>\n"
        "  📎 <code>2.02</code> [P2] 财报数据  <i>(数据由 ⑧ 处理)</i>\n"
        "  📌 <code>9.01</code> [P3] 财务报表 / 附件"
    )
    assert actual == expected


def test_8k_card_5_02_llm_failure_no_extract_block():
    """LLM-fail 5.02 (empty extracted_meta) → cron card omits the
    抽取 sub-block silently. Bot card shows '(姓名待解析)' instead."""
    actual = format_sec_8k_card(
        _llm_fail_5_02_event(), is_held=False, is_watchlist=False,
    )
    expected = (
        "<b>📋 SEC 8-K · XYZ · P1</b>\n"
        "申报日：<code>2026-06-04</code>\n"
        "\n"
        "<b>项目</b>\n"
        "  📋 <code>5.02</code> [P1] 高管 / 董事会变动"
    )
    assert actual == expected


# ---------------------------------------------------------------------------
# format_sec_8k_view — /8k bot byte-equal
# ---------------------------------------------------------------------------

def test_8k_view_multiple_filings_byte_equal():
    """/8k HPE with the multi-item HPE filing + a smaller 1.01 filing."""
    actual = format_sec_8k_view(
        [_hpe_multi_item_event(), _only_101_event()],
        ticker="HPE", days=30,
    )
    expected = (
        "<b>📋 SEC 8-K · HPE · 过去 30 天</b>\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "共 <b>2</b> 个 filings · <b>6</b> 个 items\n"
        "\n"
        "<b>2026-06-04</b> · <code>0000123-26-000001</code>\n"
        "  📋 <code>1.01</code> [P1] 重大商业合约 (新签)\n"
        "  📎 <code>2.02</code> [P2] 财报数据 <i>(⑧ 处理)</i>\n"
        "  🚨 <code>5.02</code> [P0] 高管 / 董事会变动\n"
        "       离职: <b>John Smith</b> (Chief Executive Officer)\n"
        "       任命: <b>Jane Doe</b> (Interim Chief Executive Officer)\n"
        "  📎 <code>7.01</code> [P2] Reg FD 自愿披露\n"
        "  📌 <code>9.01</code> [P3] 财务报表 / 附件\n"
        "\n"
        "<b>2026-06-04</b> · <code>0000123-26-000100</code>\n"
        "  📋 <code>1.01</code> [P1] 重大商业合约 (新签)"
    )
    assert actual == expected, (
        f"\n--- actual ---\n{actual}\n\n--- expected ---\n{expected}"
    )


def test_8k_view_empty_byte_equal():
    """Empty events → '无 8-K 申报' placeholder card."""
    actual = format_sec_8k_view([], ticker="AAPL", days=30)
    expected = (
        "<b>📋 SEC 8-K · AAPL · 过去 30 天</b>\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "<i>无 8-K 申报</i>"
    )
    assert actual == expected


def test_8k_view_5_02_llm_fail_placeholder_byte_equal():
    """/8k with 5.02 LLM failure shows '(姓名待解析)' placeholder."""
    actual = format_sec_8k_view(
        [_llm_fail_5_02_event()], ticker="XYZ", days=30,
    )
    expected = (
        "<b>📋 SEC 8-K · XYZ · 过去 30 天</b>\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "共 <b>1</b> 个 filings · <b>1</b> 个 items\n"
        "\n"
        "<b>2026-06-04</b> · <code>ACC-99</code>\n"
        "  📋 <code>5.02</code> [P1] 高管 / 董事会变动\n"
        "       <i>(姓名待解析)</i>"
    )
    assert actual == expected


# ---------------------------------------------------------------------------
# format_sec_form4_individual_card — cron ⑫ byte-equal
# ---------------------------------------------------------------------------

def _nvda_filing(acc: str = "0000123-26-000999") -> SecFiling:
    return SecFiling(
        ticker="NVDA", cik="0001045810", form="4",
        filing_date="2026-06-04", accession_number=acc,
    )


def test_form4_individual_ceo_2_5m_byte_equal():
    """Stage 2 calibration: Jensen Huang $2.5M discretionary purchase."""
    tx = Form4Transaction(
        filing=_nvda_filing(), insider_name="Jen-Hsun Huang",
        insider_role="CEO", transaction_code="P",
        transaction_date="2026-06-04",
        shares=20000.0, price=125.0, transaction_usd=2_500_000.0,
        is_10b5_1=False, direct_indirect="D",
    )
    actual = format_sec_form4_individual_card(
        tx, is_held=True, is_watchlist=False,
    )
    expected = (
        "<b>📥 内部人买入 · NVDA</b>\n"
        "申报：<code>2026-06-04</code> · 交易：<code>2026-06-04</code>\n"
        "🟢 持仓股\n"
        "\n"
        "申报人：<b>Jen-Hsun Huang</b> · 🔴 CEO\n"
        "交易：<code>20,000</code> 股 × $125/股 = <b>$2.50M</b> <i>(discretionary)</i>"
    )
    assert actual == expected


def test_form4_individual_10b5_1_demoted_byte_equal():
    """$1M plan sale by a Director — 10b5-1 tag explicit."""
    tx = Form4Transaction(
        filing=_nvda_filing(acc="0000123-26-000888"),
        insider_name="Jane Smith", insider_role="Director",
        transaction_code="S", transaction_date="2026-06-04",
        shares=8000.0, price=125.0, transaction_usd=1_000_000.0,
        is_10b5_1=True, direct_indirect="D",
    )
    actual = format_sec_form4_individual_card(
        tx, is_held=False, is_watchlist=True,
    )
    expected = (
        "<b>📤 内部人卖出 · NVDA</b>\n"
        "申报：<code>2026-06-04</code> · 交易：<code>2026-06-04</code>\n"
        "👁 关注列表\n"
        "\n"
        "申报人：<b>Jane Smith</b> · 🟢 董事\n"
        "交易：<code>8,000</code> 股 × $125/股 = <b>$1.00M</b> <i>(10b5-1 plan)</i>"
    )
    assert actual == expected


# ---------------------------------------------------------------------------
# format_sec_form4_cluster_card — cron ⑫ byte-equal
# ---------------------------------------------------------------------------

def test_form4_cluster_4_purchase_byte_equal():
    """Stage 2 calibration: ARM 4-director same-day purchase cluster."""
    arm_f = SecFiling(
        ticker="ARM", cik="0001973239", form="4",
        filing_date="2026-06-04",
        accession_number="0000123-26-000777",
    )
    txs = [
        Form4Transaction(
            filing=arm_f, insider_name=n, insider_role="Director",
            transaction_code="P", transaction_date="2026-06-04",
            shares=1000.0, price=100.0, transaction_usd=100_000.0,
            is_10b5_1=False, direct_indirect="D",
        )
        for n in ["Alice Wong", "Bob Chen", "Carol Davis", "David Lee"]
    ]
    cluster = Form4Cluster(
        ticker="ARM", cluster_date="2026-06-04", direction="purchase",
        transaction_count=4, total_usd=400_000.0,
        insider_names=["Alice Wong", "Bob Chen", "Carol Davis", "David Lee"],
        transactions=txs,
    )
    actual = format_sec_form4_cluster_card(
        cluster, is_held=True, is_watchlist=False,
    )
    expected = (
        "<b>📥 内部人集群买入 · ARM</b>\n"
        "日期：<code>2026-06-04</code> · 4 笔 / 4 人\n"
        "🟢 持仓股\n"
        "\n"
        "总金额：<b>$400.0K</b>\n"
        "\n"
        "<b>申报人</b>\n"
        "  • Alice Wong\n"
        "  • Bob Chen\n"
        "  • Carol Davis\n"
        "  • David Lee"
    )
    assert actual == expected


# ---------------------------------------------------------------------------
# format_sec_form4_view — /insiders bot byte-equal
# ---------------------------------------------------------------------------

def test_form4_view_with_clusters_byte_equal():
    """/insiders NVDA — 1 P + 1 S + cluster + noise counts."""
    p1 = Form4Transaction(
        filing=_nvda_filing(acc="ACC-P1"), insider_name="Alice",
        insider_role="CEO", transaction_code="P",
        transaction_date="2026-06-04",
        shares=10000.0, price=125.0, transaction_usd=1_250_000.0,
        is_10b5_1=False, direct_indirect="D",
    )
    s1 = Form4Transaction(
        filing=_nvda_filing(acc="ACC-S1"), insider_name="Bob",
        insider_role="CFO", transaction_code="S",
        transaction_date="2026-06-04",
        shares=5000.0, price=125.0, transaction_usd=625_000.0,
        is_10b5_1=True, direct_indirect="D",
    )
    cluster = Form4Cluster(
        ticker="NVDA", cluster_date="2026-06-04", direction="purchase",
        transaction_count=3, total_usd=300_000.0,
        insider_names=["Alice", "Carol", "Dave"],
        transactions=[],
    )
    actual = format_sec_form4_view(
        "NVDA", [p1, s1], [cluster], {"A": 8, "M": 3}, 90,
    )
    expected = (
        "<b>📥 内部人交易摘要 · NVDA · 过去 90 天</b>\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "<b>P (Purchase): 1 笔</b>, 总 $1.25M\n"
        "  最大: Alice (CEO) · $1.25M · 2026-06-04\n"
        "<b>S (Sale):</b> 1 笔, 总 $625.0K\n"
        "  最大: Bob (CFO) · $625.0K · 2026-06-04 · 10b5-1\n"
        "\n"
        "<b>A</b>: 8 笔  <i>(Award (薪酬授予))</i>\n"
        "<b>M</b>: 3 笔  <i>(Exercise (option vest))</i>\n"
        "\n"
        "<b>集群:</b> 1 个\n"
        "  2026-06-04 · 3 笔 / 3 人 买入 · $300.0K\n"
        "    Alice, Carol, Dave"
    )
    assert actual == expected, (
        f"\n--- actual ---\n{actual}\n\n--- expected ---\n{expected}"
    )


def test_form4_view_only_noise_byte_equal():
    """Only A/F noise codes → '0 笔' for both P/S, noise counts shown."""
    actual = format_sec_form4_view(
        "TSLA", [], [], {"A": 5, "F": 2}, 90,
    )
    expected = (
        "<b>📥 内部人交易摘要 · TSLA · 过去 90 天</b>\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "<b>P (Purchase): 0 笔</b>\n"
        "<b>S (Sale):</b> 0 笔\n"
        "\n"
        "<b>A</b>: 5 笔  <i>(Award (薪酬授予))</i>\n"
        "<b>F</b>: 2 笔  <i>(Tax (RSU vest 税款))</i>\n"
        "\n"
        "<b>集群:</b> 无同日 ≥3 distinct insiders 集群 (90 天内)"
    )
    assert actual == expected


def test_form4_view_empty_byte_equal():
    """No transactions + no noise → '无 Form 4 申报' placeholder."""
    actual = format_sec_form4_view("AAPL", [], [], {}, 90)
    expected = (
        "<b>📥 内部人交易摘要 · AAPL · 过去 90 天</b>\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "<i>无 Form 4 申报</i>"
    )
    assert actual == expected


# ---------------------------------------------------------------------------
# Surface: all 5 public names exposed under both module paths
# ---------------------------------------------------------------------------

def test_bot_cards_module_exposes_all_five_formatters():
    """``v2.sec._bot_cards.__all__`` is the contract surface."""
    from v2.sec import _bot_cards
    for name in (
        "format_sec_8k_card",
        "format_sec_8k_view",
        "format_sec_form4_individual_card",
        "format_sec_form4_cluster_card",
        "format_sec_form4_view",
        "format_sec_insider_digest",
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
        "v2.reporting._sec_formatters",
        _REPO_ROOT / "v2" / "reporting" / "_sec_formatters.py",
    )
    shim = importlib.util.module_from_spec(spec)
    _sys.modules["v2.reporting._sec_formatters"] = shim
    spec.loader.exec_module(shim)

    from v2.sec import _bot_cards as src
    assert shim.format_sec_8k_card is src.format_sec_8k_card
    assert shim.format_sec_8k_view is src.format_sec_8k_view
    assert shim.format_sec_form4_individual_card is src.format_sec_form4_individual_card
    assert shim.format_sec_form4_cluster_card is src.format_sec_form4_cluster_card
    assert shim.format_sec_form4_view is src.format_sec_form4_view
    assert shim.format_sec_insider_digest is src.format_sec_insider_digest


# ---------------------------------------------------------------------------
# format_sec_insider_digest — ⑫b weekly digest byte-equal (Phase 3.5)
# ---------------------------------------------------------------------------

def _digest_normal() -> WeeklyInsiderSummary:
    """7 pushes, 5 tickers, 1 buy cluster + 1 sell cluster, no unusual."""
    return WeeklyInsiderSummary(
        week_start="2026-06-01", week_end="2026-06-05",
        purchase_push_count=2, sale_push_count=3,
        cluster_purchase_count=1, cluster_sale_count=1,
        by_ticker={"NVDA": 2, "AAPL": 2, "MSFT": 1, "ARM": 1, "TSLA": 1},
        unusual_tickers=[],
        total_tickers_active=5, total_push_count=7,
    )


def _digest_quiet() -> WeeklyInsiderSummary:
    """Empty week — is_quiet_week=True, no footer caption."""
    return WeeklyInsiderSummary(
        week_start="2026-06-01", week_end="2026-06-05",
    )


def _digest_unusual_three() -> WeeklyInsiderSummary:
    """3 unusual tickers — no overflow (≤5)."""
    return WeeklyInsiderSummary(
        week_start="2026-06-01", week_end="2026-06-05",
        purchase_push_count=4, sale_push_count=8,
        cluster_purchase_count=2, cluster_sale_count=1,
        by_ticker={"NVDA": 5, "ARM": 4, "TSLA": 3, "AAPL": 2, "MSFT": 1},
        unusual_tickers=["NVDA", "ARM", "TSLA"],
        total_tickers_active=5, total_push_count=15,
    )


def _digest_unusual_overflow() -> WeeklyInsiderSummary:
    """8 unusual tickers — top 5 + '... 另 3 只' tail."""
    return WeeklyInsiderSummary(
        week_start="2026-06-01", week_end="2026-06-05",
        purchase_push_count=12, sale_push_count=18,
        cluster_purchase_count=3, cluster_sale_count=2,
        by_ticker={
            "NVDA": 7, "ARM": 6, "TSLA": 5, "AAPL": 4, "MSFT": 4,
            "META": 3, "AMZN": 3, "GOOGL": 3, "AVGO": 2,
        },
        unusual_tickers=[
            "NVDA", "ARM", "TSLA", "AAPL", "MSFT",
            "META", "AMZN", "GOOGL",
        ],
        total_tickers_active=9, total_push_count=35,
    )


def test_insider_digest_normal_week_byte_equal():
    """7 pushes, 5 tickers, both cluster lines — full footer caption."""
    actual = format_sec_insider_digest(_digest_normal())
    expected = (
        "<b>📥 内部人活动周报 · 2026-06-01 → 2026-06-05</b>\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "<b>本周总览</b>\n"
        "  总 push 数: <code>7</code> · 涉及 ticker: <code>5</code> 只\n"
        "\n"
        "<b>方向分布</b>\n"
        "  📥 买入 (purchase): <code>2</code> 笔\n"
        "  📤 卖出 (sale): <code>3</code> 笔\n"
        "  🔗 集群 (cluster): <code>2</code> 笔 (买入 1 / 卖出 1)\n"
        "\n"
        "<i>注：基于 ⑫ push title 统计（Phase 3.5 简化口径，"
        "per-code A/M/F/G/C breakdown 见 Phase 3.5.5）</i>"
    )
    assert actual == expected, (
        f"\n--- actual ---\n{actual}\n\n--- expected ---\n{expected}"
    )


def test_insider_digest_quiet_week_no_footer_byte_equal():
    """Empty week — placeholder line, no footer caption per Polish 2."""
    actual = format_sec_insider_digest(_digest_quiet())
    expected = (
        "<b>📥 内部人活动周报 · 2026-06-01 → 2026-06-05</b>\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "<i>本周 ⑫ Form 4 推送平静（0 笔）</i>"
    )
    assert actual == expected, (
        f"\n--- actual ---\n{actual}\n\n--- expected ---\n{expected}"
    )


def test_insider_digest_unusual_three_tickers_byte_equal():
    """3 unusual tickers — ≤5 so no overflow tail."""
    actual = format_sec_insider_digest(_digest_unusual_three())
    expected = (
        "<b>📥 内部人活动周报 · 2026-06-01 → 2026-06-05</b>\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "<b>本周总览</b>\n"
        "  总 push 数: <code>15</code> · 涉及 ticker: <code>5</code> 只\n"
        "\n"
        "<b>方向分布</b>\n"
        "  📥 买入 (purchase): <code>4</code> 笔\n"
        "  📤 卖出 (sale): <code>8</code> 笔\n"
        "  🔗 集群 (cluster): <code>3</code> 笔 (买入 2 / 卖出 1)\n"
        "\n"
        "<b>⚠️ 异常活跃 ticker</b> (≥3 pushes)\n"
        "  <b>NVDA</b>: 5 pushes\n"
        "  <b>ARM</b>: 4 pushes\n"
        "  <b>TSLA</b>: 3 pushes\n"
        "\n"
        "<i>注：基于 ⑫ push title 统计（Phase 3.5 简化口径，"
        "per-code A/M/F/G/C breakdown 见 Phase 3.5.5）</i>"
    )
    assert actual == expected, (
        f"\n--- actual ---\n{actual}\n\n--- expected ---\n{expected}"
    )


def test_insider_digest_unusual_overflow_top5_byte_equal():
    """8 unusual tickers — top 5 listed + '... 另 3 只' tail per Polish 3."""
    actual = format_sec_insider_digest(_digest_unusual_overflow())
    expected = (
        "<b>📥 内部人活动周报 · 2026-06-01 → 2026-06-05</b>\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "<b>本周总览</b>\n"
        "  总 push 数: <code>35</code> · 涉及 ticker: <code>9</code> 只\n"
        "\n"
        "<b>方向分布</b>\n"
        "  📥 买入 (purchase): <code>12</code> 笔\n"
        "  📤 卖出 (sale): <code>18</code> 笔\n"
        "  🔗 集群 (cluster): <code>5</code> 笔 (买入 3 / 卖出 2)\n"
        "\n"
        "<b>⚠️ 异常活跃 ticker</b> (≥3 pushes)\n"
        "  <b>NVDA</b>: 7 pushes\n"
        "  <b>ARM</b>: 6 pushes\n"
        "  <b>TSLA</b>: 5 pushes\n"
        "  <b>AAPL</b>: 4 pushes\n"
        "  <b>MSFT</b>: 4 pushes\n"
        "  <i>... 另 3 只</i>\n"
        "\n"
        "<i>注：基于 ⑫ push title 统计（Phase 3.5 简化口径，"
        "per-code A/M/F/G/C breakdown 见 Phase 3.5.5）</i>"
    )
    assert actual == expected, (
        f"\n--- actual ---\n{actual}\n\n--- expected ---\n{expected}"
    )
