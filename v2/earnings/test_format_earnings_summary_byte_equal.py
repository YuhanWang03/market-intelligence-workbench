"""Byte-equal pin tests for ⑧ Earnings Summary card with Phase 3.5
10-Q delta section.

Mirrors the SEC byte-equal pattern in
``v2/sec/test_formatters_byte_equal.py``. Pinning the rendered card body
keeps the Phase 3.5 ``📋 10-Q MD&A 关键变化`` block from silently
drifting under a later refactor.

Pre-truncated mda_added_paragraphs (with the "…" suffix already
applied) simulate what ``diff_ten_q`` produces post-Polish 1 — the
formatter is content-blind, it just renders what the parser hands it.
"""

from __future__ import annotations

import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


from v2.earnings._bot_cards import format_earnings_summary   # noqa: E402
from v2.earnings.models import EarningsSummary               # noqa: E402
from v2.sec.ten_q_parser import TenQDelta                    # noqa: E402


def _nvda_summary_with_10q() -> EarningsSummary:
    """NVDA Q1 2026 BEAT with 3 MD&A added paragraphs + 1 new RF, no
    auditor flags. Paragraphs are pre-truncated to 80 chars + '…'."""
    ten_q = TenQDelta(
        ticker="NVDA",
        filing_date="2026-05-20",
        period="Q1 2026",
        mda_added_paragraphs=[
            "Revenue growth driven by strong demand in AI infrastructure across data ce…",
            "Operating margin compressed by inventory write-downs related to H100 obsol…",
            "Cash flow from operations remained robust at $2.3B for the quarter despite…",
        ],
        new_risk_factor_count=1,
        has_going_concern=False,
        has_material_weakness=False,
    )
    return EarningsSummary(
        ticker="NVDA",
        report_period="2026-04-30",
        filing_date="2026-05-20",
        eps_surprise="BEAT",
        eps_actual=0.72,
        eps_estimate=0.65,
        revenue_actual=31_000_000_000.0,
        revenue_estimate=29_500_000_000.0,
        last_4q_surprises=["BEAT", "BEAT", "MEET", "BEAT"],
        transcript_url=(
            "https://www.fool.com/earnings/call-transcripts/2026/05/20/nvda-q1-2026/"
        ),
        bull="数据中心业务同比 +85%，Blackwell 提前量产兑现订单可见性",
        bear="毛利率环比 -120bps 反映 H100 库存清理压力",
        narrative="本季是 Hopper → Blackwell 平滑过渡的关键节点，订单簿看 H2 持续走强",
        ten_q_delta=ten_q,
    )


def test_earnings_summary_with_ten_q_delta_byte_equal():
    """⑧ card with Phase 3.5 10-Q delta section — full byte-equal pin
    including 3 MD&A paragraphs (each ending in '…' from the 80-char
    truncation) and the '1 个新 risk factor 段落' tail."""
    actual = format_earnings_summary(
        _nvda_summary_with_10q(), is_held=True, is_watchlist=False,
    )
    expected = (
        "<b>🟢 财报发布 · NVDA · BEAT</b>\n"
        "报告期：<code>2026-04-30</code> · 申报：<code>2026-05-20</code>\n"
        "🟢 持仓股\n"
        "\n"
        "EPS：<code>0.72</code> vs 预期 <code>0.65</code> (+10.8%)\n"
        "营收：<code>$31.00B</code> vs 预期 <code>$29.50B</code> (+5.1%)\n"
        "最近 4 季：<code>BEAT → BEAT → MEET → BEAT</code>\n"
        "\n"
        "👍 数据中心业务同比 +85%，Blackwell 提前量产兑现订单可见性\n"
        "👎 毛利率环比 -120bps 反映 H100 库存清理压力\n"
        "\n"
        "<i>本季是 Hopper → Blackwell 平滑过渡的关键节点，订单簿看 H2 持续走强</i>\n"
        "\n"
        '📜 <a href="https://www.fool.com/earnings/call-transcripts/2026/05/20/nvda-q1-2026/">电话会记录</a>\n'
        "\n"
        "<b>📋 10-Q MD&amp;A 关键变化</b>\n"
        "  ➕ <i>Revenue growth driven by strong demand in AI infrastructure across data ce…</i>\n"
        "  ➕ <i>Operating margin compressed by inventory write-downs related to H100 obsol…</i>\n"
        "  ➕ <i>Cash flow from operations remained robust at $2.3B for the quarter despite…</i>\n"
        "  📌 1 个新 risk factor 段落"
    )
    assert actual == expected, (
        f"\n--- actual ---\n{actual}\n\n--- expected ---\n{expected}"
    )
