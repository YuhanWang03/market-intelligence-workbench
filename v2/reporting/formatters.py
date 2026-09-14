"""Convert v2 result objects into Telegram-ready text and images.

Pure functions — no IO, no notifier knowledge. They take a result object
and return either an HTML string or a PNG byte buffer.
"""

from __future__ import annotations

import html
from io import BytesIO

import matplotlib

matplotlib.use("Agg")  # headless — no display window on Windows
import matplotlib.pyplot as plt
from matplotlib import font_manager as _fm
from pathlib import Path as _Path

# On Linux, matplotlib's font discovery sometimes misses system-installed
# CJK fonts. Explicitly register Noto Sans CJK from its known apt paths so
# we don't rely on the discovery cache being fresh.
_LINUX_CJK_FONT_FILES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK.ttc",
    "/usr/share/fonts/truetype/noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",  # last resort
]
for _path in _LINUX_CJK_FONT_FILES:
    if _Path(_path).exists():
        try:
            _fm.fontManager.addfont(_path)
        except Exception:
            pass
        break

# CJK-capable font fallback chain — covers Windows / macOS / Linux
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",       # Windows
    "PingFang SC",           # macOS
    "Heiti SC",              # macOS legacy
    "Noto Sans CJK SC",      # Linux (apt install fonts-noto-cjk)
    "Noto Sans SC",          # Linux alternative name
    "Noto Sans CJK JP",      # Linux fallback (JP variant covers most CJK)
    "WenQuanYi Micro Hei",   # Linux fallback
    "SimHei",
    "Arial Unicode MS",
    "sans-serif",
]
plt.rcParams["axes.unicode_minus"] = False  # render minus signs correctly with CJK fonts

from v2.backtesting.models import BacktestResult
from v2.institutional.models import InstitutionalReport, PositionChange
from v2.lateral.models import CATEGORIES, LateralResult
from v2.monitoring.models import Anomaly
from v2.observability import emit
from v2.screening.models import ScreenResult


def format_backtest_summary(
    result: BacktestResult,
    *,
    strategy_name: str = "Backtest",
    universe_size: int | None = None,
) -> str:
    """One-card HTML summary suitable for Telegram."""
    m = result.metrics
    if m is None or not result.trades:
        return f"<b>{html.escape(strategy_name)}</b>\nNo trades generated."

    first = result.trades[0].entry_date
    last = result.trades[-1].exit_date

    universe_line = (
        f"Universe: <code>{universe_size}</code> tickers · "
        if universe_size is not None
        else ""
    )

    return (
        f"<b>📊 {html.escape(strategy_name)} Backtest</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"{universe_line}{m.avg_holding_days:.0f}-day hold\n"
        f"Period: <code>{first}</code> → <code>{last}</code>\n\n"
        f"💰 Total Return:   <b>{m.total_return_pct:+.2%}</b>\n"
        f"📈 Annualized:     <b>{m.annualized_return_pct:+.2%}</b>\n"
        f"⚡ Sharpe:          <b>{m.sharpe_ratio:.2f}</b>\n"
        f"📉 Max Drawdown:   <b>-{m.max_drawdown_pct:.2%}</b>\n"
        f"🎯 Win Rate:       <b>{m.win_rate:.1%}</b>\n\n"
        f"Trades: <b>{m.n_trades}</b> "
        f"({m.n_long} long, {m.n_short} short)\n"
        f"Avg / Trade: <b>{m.avg_return_pct:+.2%}</b>"
    )


def render_equity_curve(
    result: BacktestResult,
    *,
    title: str = "Equity Curve",
) -> bytes:
    """Render the equity curve as a PNG byte string."""
    curve = result.equity_curve
    if not curve:
        # Fallback: empty plot
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    else:
        fig, ax = plt.subplots(figsize=(10, 4))
        x = list(range(len(curve)))
        starting = curve[0]
        is_positive = curve[-1] >= starting
        color = "#2ca02c" if is_positive else "#d62728"

        ax.plot(x, curve, linewidth=2, color=color)
        ax.fill_between(x, curve, starting, alpha=0.15, color=color)
        ax.axhline(starting, color="#666", linewidth=0.8, linestyle="--", alpha=0.7)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Trade #")
        ax.set_ylabel("Equity ($)")
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def format_screening_result(result: ScreenResult) -> str:
    """One-card HTML summary of a fundamental screen for Telegram.

    Layout per candidate (改进 ②/③/④ applied):
        🟢 TICKER  $price  ±%   [tags]
           市值 $X · 毛利 X.X% · 营收 +X.X% (预期 +Y% 🟢)
           💡 bull logic (no numbers — Template Fill mode)
           ⚠️ bear logic
    """
    if not result.candidates:
        return (
            f"<b>📋 科技股筛选 · {result.date}</b>\n"
            f"扫描 {result.universe_size} · <i>无候选通过筛选</i>"
        )

    lines: list[str] = [
        f"<b>📋 科技股筛选 · {result.date}</b>",
        "━━━━━━━━━━━━━━━━━━━━",
        f"扫描 <b>{result.universe_size}</b> · 通过 <b>{len(result.candidates)}</b>",
        "",
    ]

    for c in result.candidates:
        change = f"  {c.price_change:+.2%}" if c.price_change is not None else ""
        tag_str = _format_tags(c.tags)
        lines.append(
            f"🟢 <b>{c.ticker}</b>  <code>${c.price:,.2f}</code>{change}{tag_str}"
        )
        lines.append(_format_metrics_line(c))
        sec_line = _format_sector_line(c)
        if sec_line:
            lines.append(sec_line)
        if c.bull:
            lines.append(f"   💡 {html.escape(c.bull)}")
        if c.bear:
            lines.append(f"   ⚠️ {html.escape(c.bear)}")
        lines.append("")

    # Step 5: data-provenance line — what inputs the narrator actually saw
    lines.append(
        "<i>📊 数据范围：FD finmetrics (TTM) · "
        "prices (252d) · earnings (latest Q) · Tavily news (7d)</i>"
    )

    # Always show FD count (including 0) so cache effectiveness is visible.
    # When fd_calls == 0 + a non-trivial cache exists, that's the headline metric.
    api_parts = [f"FD ×{result.fd_calls}"]
    if result.tavily_calls:
        api_parts.append(f"Tavily ×{result.tavily_calls}")
    lines.append(
        f"<i>🔧 DeepSeek {result.llm_tokens:,} tokens · {' · '.join(api_parts)}</i>"
    )
    emit(
        "render", card="screening_result",
        universe_size=result.universe_size,
        candidates=len(result.candidates),
        llm_tokens=result.llm_tokens,
        fd_calls=result.fd_calls,
        tavily_calls=result.tavily_calls,
    )
    return "\n".join(lines)


def _format_tags(tags: list[str]) -> str:
    """Format tag chips that appear after the ticker line: [tag1 | tag2 | tag3]."""
    if not tags:
        return ""
    return "  <i>[" + " | ".join(html.escape(t) for t in tags) + "]</i>"


def _format_sector_line(c) -> str:
    """One-line sector-relative comparison. Empty string if no data."""
    if not c.sector_etf or c.return_1w is None or c.sector_return_1w is None:
        return ""
    diff = c.sector_diff_1w_pp
    if diff is None:
        return ""
    arrow = "↑" if diff > 0 else ("↓" if diff < 0 else "→")
    label = "领涨" if diff >= 0.015 else ("掉队" if diff <= -0.015 else "同步")
    return (
        f"   <i>对比 <b>{html.escape(c.sector_etf)}</b> "
        f"7d <code>{c.sector_return_1w:+.1%}</code>  "
        f"差 {arrow}<code>{diff * 100:+.1f}pp</code>  "
        f"<b>{label}</b></i>"
    )


def _format_metrics_line(c) -> str:
    """One-line metrics summary; uses 1-decimal precision so it matches the
    LLM's internal precision (改进 ③ data consistency fix).

    Includes Wall Street expectation comparison if available (改进 ④).
    """
    parts: list[str] = [
        f"市值 <code>${_short_money(c.market_cap)}</code>",
        f"毛利 <code>{_pct1_or_dash(c.gross_margin)}</code>",
        f"波动 <code>{_pct1_or_dash(c.volatility)}</code>",
    ]
    # Revenue growth with optional Wall Street expectation
    rev_part = f"营收 <code>{_pct1_or_dash(c.revenue_growth, signed=True)}</code>"
    if c.revenue_surprise_pct is not None:
        beat = c.revenue_surprise_pct
        emoji = "🟢" if beat >= 0.02 else "🔴" if beat <= -0.02 else "🟡"
        rev_part += (
            f" <i>(预期 {_pct1_or_dash(_back_solve_estimate_pct(c), signed=True)}"
            f" {emoji} {beat:+.1%})</i>"
        )
    parts.insert(2, rev_part)
    return "   " + " · ".join(parts)


def _back_solve_estimate_pct(c) -> float | None:
    """Reverse-engineer the Wall Street estimated growth from actual/estimate.

    If FD reports revenue_growth (TTM-over-TTM) and a surprise_pct, the
    consensus growth = (1 + actual_growth) / (1 + surprise) - 1. But that
    mixes timeframes. Simpler & more honest: just compute (estimate -
    prior_revenue) / prior_revenue if we have all three. For now, infer
    estimate growth from the surprise + reported growth:

        actual_revenue = est_revenue × (1 + surprise)
        ⇒ est_growth = actual_growth - surprise (1st-order approximation)
    """
    if c.revenue_growth is None or c.revenue_surprise_pct is None:
        return None
    return c.revenue_growth - c.revenue_surprise_pct


def _pct1_or_dash(value: float | None, *, signed: bool = False) -> str:
    """Like _pct_or_dash but with 1 decimal place — matches LLM precision."""
    if value is None:
        return "—"
    return f"{value:+.1%}" if signed else f"{value:.1%}"


def _short_money(value: float | None) -> str:
    """Format a dollar amount as $X.YT / $X.YB / $X.YM."""
    if value is None:
        return "—"
    if value >= 1e12:
        return f"{value / 1e12:.1f}T"
    if value >= 1e9:
        return f"{value / 1e9:.1f}B"
    if value >= 1e6:
        return f"{value / 1e6:.1f}M"
    return f"{value:,.0f}"


def _pct_or_dash(value: float | None, *, signed: bool = False) -> str:
    """Format a fractional value as a percent, or dash if None."""
    if value is None:
        return "—"
    return f"{value:+.0%}" if signed else f"{value:.0%}"


def _short_count(v: float | None) -> str:
    """Format a count (e.g. share volume) as 234.5M / 12.3K / 567."""
    if v is None:
        return "—"
    if v >= 1e9:
        return f"{v / 1e9:.1f}B"
    if v >= 1e6:
        return f"{v / 1e6:.1f}M"
    if v >= 1e3:
        return f"{v / 1e3:.1f}K"
    return f"{v:,.0f}"


# ---------------------------------------------------------------------------
# Monitoring (玩法 ②)
# ---------------------------------------------------------------------------

_CONFIDENCE_EMOJI = {
    "高": "🟢",
    "中": "🟡",
    "低": "🔴",
}


def _confidence_chip(level: str) -> str:
    """Inline chip for confidence display: [高 🟢] / [中 🟡] / [低 🔴]."""
    emoji = _CONFIDENCE_EMOJI.get(level, "⚪")
    return f"[<b>{level}</b> {emoji}]"


_FLAG_CHIP = {
    "volume_spike":     lambda a: f"📊 成交量 {a.volume_ratio:.1f}x",
    "52w_high":         lambda a: "🎯 52 周新高",
    "52w_low":          lambda a: "⚠️ 52 周新低",
    "insider_buying":   lambda a: (
        f"💰 内部人净买 ${_short_money(a.insider.net_value)}"
        if a.insider else "💰 内部人买入"
    ),
    "insider_selling":  lambda a: (
        f"💸 内部人净卖 ${_short_money(abs(a.insider.net_value))}"
        if a.insider else "💸 内部人卖出"
    ),
    "options_call_burst": lambda a: (
        f"📞 看涨期权 {a.options_burst.ratio:.1f}×"
        if a.options_burst else "📞 看涨期权异动"
    ),
    "options_put_burst": lambda a: (
        f"📉 看跌期权 {a.options_burst.ratio:.1f}×"
        if a.options_burst else "📉 看跌期权异动"
    ),
    "contrarian_move": lambda a: (
        "★ 逆势上涨"
        if (a.price_change_pct or 0) > 0 else
        "★ 逆势下跌"
    ),
}


def format_anomaly_alert(anomaly: Anomaly) -> str:
    """HTML caption for the anomaly photo. Fits within Telegram's 1024-char cap."""
    chips = [_FLAG_CHIP[f](anomaly) for f in anomaly.flags if f in _FLAG_CHIP]

    # Path-3 tag: 52-week extreme without volume confirmation.
    # Textbook "soft breakout" — keep pushing but flag for observation. If
    # these reliably underperform real (volume-confirmed) breakouts over time,
    # we'll promote this from a label to a hard filter.
    if anomaly.volume_ratio < 1.0:
        if "52w_high" in anomaly.flags:
            chips.append(f"⚠️ 软新高 量{anomaly.volume_ratio:.2f}x")
        elif "52w_low" in anomaly.flags:
            chips.append(f"⚠️ 软新低 量{anomaly.volume_ratio:.2f}x")

    flag_line = " · ".join(chips)

    lines: list[str] = [
        f"<b>🚨 异动 · {html.escape(anomaly.ticker)} · {anomaly.date}</b>",
        "━━━━━━━━━━━━━━━━━━━━",
        flag_line,
        "",
        f"价格 <code>${anomaly.price:,.2f}</code>  <b>{anomaly.price_change_pct:+.2%}</b>",
        f"成交量 <code>{_short_count(anomaly.volume_today)}</code> "
        f"(30 日均 <code>{_short_count(anomaly.volume_avg_30d)}</code>)",
    ]

    # Sector relative-strength line — only show when we successfully fetched
    # the benchmark ETF prices for this run.
    if anomaly.sector_etf and anomaly.sector_return_1d is not None:
        rel_pp = anomaly.relative_1d_pp
        arrow = "↑" if (rel_pp or 0) > 0 else ("↓" if (rel_pp or 0) < 0 else "→")
        contra_chip = " <b>★ 逆势</b>" if anomaly.contrarian else ""
        lines.append(
            f"<i>对比 <b>{anomaly.sector_etf}</b> "
            f"{anomaly.sector_return_1d:+.2%}  ·  "
            f"差 {arrow}<code>{(rel_pp or 0) * 100:+.2f}pp</code></i>"
            f"{contra_chip}"
        )

    # Phase B 3b: options burst detail (if burst detected)
    if anomaly.options_burst is not None:
        ob = anomaly.options_burst
        side_cn = "看涨" if ob.side == "call" else "看跌"
        lines.append("")
        lines.append(
            f"<b>📞 期权异动：</b>{side_cn}持仓 <b>{ob.ratio:.1f}×</b> "
            f"基线（OI <code>{_short_count(ob.current_oi)}</code> vs "
            f"{ob.baseline_days} 天均 <code>{_short_count(ob.baseline_avg_oi)}</code>）"
        )

    # Phase B 3a: insider activity detail block (if any)
    if anomaly.insider is not None:
        ins = anomaly.insider
        net = ins.net_value
        verb = "净买入" if net > 0 else "净卖出"
        lines.append("")
        lines.append(
            f"<b>👥 内部人活动（近 30 日）：</b>"
        )
        lines.append(
            f"   {verb} <code>${_short_money(abs(net))}</code> · "
            f"{ins.trade_count} 笔交易"
        )
        for exec_ in ins.executives[:3]:
            arrow = "买入" if exec_.direction == "buy" else "卖出"
            lines.append(
                f"   {html.escape(exec_.title.split(',')[0][:25])} "
                f"{html.escape(exec_.name.split(',')[0][:20])} "
                f"{arrow} <code>${_short_money(exec_.value)}</code>"
            )

    if anomaly.reasons:
        lines.append("")
        lines.append("<b>🔍 归因分析（置信度评估）：</b>")
        for i, scored in enumerate(anomaly.reasons, 1):
            chip = _confidence_chip(scored.confidence)
            line = f"{i}. {chip} {html.escape(scored.text)}"
            if scored.note:
                line += f"  <i>（{html.escape(scored.note)}）</i>"
            lines.append(line)
        # Phase A: show how many noise items the entity filter caught
        if anomaly.filtered_count > 0:
            lines.append(
                f"<i>⛔ 已过滤 {anomaly.filtered_count} 条噪音"
                f"（实体校验：与 {html.escape(anomaly.ticker)} 无关）</i>"
            )
    elif anomaly.tavily_calls > 0:
        lines.append("")
        if anomaly.filtered_count > 0:
            lines.append(
                f"<i>🔍 Tavily 返回的新闻全部被实体校验过滤"
                f"（{anomaly.filtered_count} 条均与 {html.escape(anomaly.ticker)} 无关）</i>"
            )
        else:
            lines.append("<i>🔍 未找到与异动相关的最近新闻</i>")

    # Phase C: historical context from ChromaDB RAG (if any past similar anomalies)
    if anomaly.historical_context:
        lines.append("")
        count_with_today = len(anomaly.historical_context) + 1
        lines.append(
            f"<b>🧠 历史关联：</b>过去 30 天内 "
            f"<b>{html.escape(anomaly.ticker)}</b> 第 "
            f"<b>{count_with_today}</b> 次类似异动"
        )
        for h in anomaly.historical_context[:3]:
            flag_disp = h.flags.replace(",", " · ") if h.flags else "—"
            lines.append(
                f"   • <code>{html.escape(h.date)}</code> "
                f"({html.escape(flag_disp)})"
            )

    # Phase A: actionable follow-up observations
    if anomaly.next_steps:
        lines.append("")
        lines.append("<b>💡 下一步观察：</b>")
        for step in anomaly.next_steps:
            lines.append(f"• {html.escape(step)}")

    sources_block: list[str] = []
    if anomaly.sources:
        src_lines: list[str] = []
        for s in anomaly.sources:
            if " | " in (s.title or ""):      # skip aggregator page titles
                continue
            title = html.escape(s.title)[:55] or "link"
            url = html.escape(s.url)
            src_lines.append(f'• <a href="{url}">{title}</a>')
        if src_lines:
            sources_block.append("")
            sources_block.append("<b>来源：</b>")
            sources_block.extend(src_lines)

    footer = (
        f"<i>🔧 Tavily ×{anomaly.tavily_calls} · "
        f"DeepSeek {anomaly.llm_tokens} tokens</i>"
    )

    body = "\n".join(lines + sources_block) + "\n\n" + footer
    # Telegram photo caption hard limit is 1024 chars — drop sources if needed
    if len(body) > 1024 and sources_block:
        body = "\n".join(lines) + "\n\n" + footer
    emit(
        "render",
        card="anomaly_card",
        ticker=anomaly.ticker,
        date=anomaly.date,
        num_reasons=len(anomaly.reasons or []),
        num_sources=len(anomaly.sources or []),
        filtered_news=anomaly.filtered_count or 0,
    )
    return body


# ---------------------------------------------------------------------------
# Lateral expansion (玩法 ③)
# ---------------------------------------------------------------------------

_CATEGORY_DISPLAY = {
    "supplier":     "🔹 供应商",
    "customer":     "🔹 客户",
    "smaller_peer": "🔹 同业小市值",
    "beneficiary":  "🔹 间接受益方",
}


def format_lateral_result(result: LateralResult) -> str:
    """HTML summary of one lateral-expansion pass, grouped by seed × category."""
    if not result.neighbors:
        return (
            f"<b>🕸️ Watchlist 邻居发现 · {result.date}</b>\n"
            f"<i>LLM 未生成任何候选</i>"
        )

    hallucinations = [n for n in result.neighbors if not n.exists]
    in_universe = [n for n in result.neighbors if n.exists and n.already_in_universe]
    new_real = [n for n in result.neighbors if n.exists and not n.already_in_universe]
    passers = [n for n in new_real if n.passed_filter]

    lines: list[str] = [
        f"<b>🕸️ Watchlist 邻居发现 · {result.date}</b>",
        "━━━━━━━━━━━━━━━━━━━━",
        f"基于 <b>{len(result.seeds)}</b> 只 seed · "
        f"LLM 生成 <b>{len(result.neighbors)}</b> 个候选",
        f"存在 <b>{len(result.neighbors) - len(hallucinations)}</b> · "
        f"已在 universe <b>{len(in_universe)}</b> · "
        f"硬筛通过 <b>{len(passers)}</b>",
        "",
    ]

    # Group new+real neighbors by seed -> category, preserving multi-labels.
    # Same ticker can appear under multiple (seed, category) slots.
    grouped: dict[str, dict[str, list]] = {}
    for n in new_real:
        for label in n.labels:
            grouped.setdefault(label.seed, {}).setdefault(label.category, []).append(n)

    for seed in result.seeds:
        seed_data = grouped.get(seed)
        if not seed_data:
            continue
        lines.append(f"<b>【{html.escape(seed)}】</b>")
        for category in CATEGORIES:
            cat_neighbors = seed_data.get(category)
            if not cat_neighbors:
                continue
            lines.append(_CATEGORY_DISPLAY[category])
            for n in cat_neighbors:
                lines.append(_format_neighbor_line(n, seed, category))
        lines.append("")

    if in_universe:
        tickers = sorted({n.ticker for n in in_universe})
        lines.append(
            f"<b>★ 已在 universe ({len(tickers)}):</b> "
            f"{html.escape(', '.join(tickers))}"
        )

    if hallucinations:
        tickers = sorted({n.ticker for n in hallucinations})
        display = (
            ", ".join(tickers[:8]) + f", … +{len(tickers) - 8}"
            if len(tickers) > 8 else ", ".join(tickers)
        )
        lines.append(
            f"<b>❌ LLM 幻觉 ({len(tickers)}):</b> {html.escape(display)}"
        )

    if in_universe or hallucinations:
        lines.append("")

    footer_parts = [
        f"DeepSeek {result.llm_tokens:,} tokens",
        f"FD ×{result.api_calls}",
    ]
    if result.tavily_calls:
        footer_parts.append(f"Tavily ×{result.tavily_calls}")
        verified_count = sum(1 for n in result.neighbors if n.relation_verified)
        checked_count = sum(1 for n in result.neighbors if n.relation_checked)
        if checked_count:
            footer_parts.append(f"关系验证 {verified_count}/{checked_count}")

    lines.append(f"<i>🔧 {' · '.join(footer_parts)}</i>")

    body = "\n".join(lines)
    # Defensive truncation if we somehow blow past 4096
    if len(body) > 4000:
        body = body[:3950] + "\n<i>(truncated)</i>"
    emit(
        "render",
        card="lateral_result",
        seeds=len(result.seeds or []),
        verified_neighbors=len(result.neighbors or []),
    )
    return body


def _format_neighbor_line(neighbor, seed: str, category: str) -> str:
    """One line per (neighbor, seed, category) cell — passer or rejected."""
    # Find the label specific to this seed+category for its reason text
    matching = next(
        (l for l in neighbor.labels if l.seed == seed and l.category == category),
        None,
    )
    reason_chip = (
        f" <i>「{html.escape(matching.reason)}」</i>"
        if matching and matching.reason else ""
    )

    # Step 3: Tavily relation-verification chip
    if neighbor.relation_verified:
        rel_chip = " <b>✓</b>"
    elif neighbor.relation_checked:
        rel_chip = " <i>⚠ 关系无据</i>"
    else:
        rel_chip = ""

    if neighbor.passed_filter and neighbor.candidate is not None:
        first_line = (
            f"  ✅ <b>{html.escape(neighbor.ticker)}</b>{rel_chip} "
            f"<code>${neighbor.candidate.price:,.2f}</code>"
            f"{reason_chip}"
        )
        # Bull/bear on a second indented line — keeps the ticker line scannable.
        bull = html.escape(neighbor.bull) if neighbor.bull else ""
        bear = html.escape(neighbor.bear) if neighbor.bear else ""
        if bull and bear:
            return f"{first_line}\n     💡 {bull} / ⚠️ {bear}"
        if bull:
            return f"{first_line}\n     💡 {bull}"
        if bear:
            return f"{first_line}\n     ⚠️ {bear}"
        return first_line
    else:
        return (
            f"  🟡 {html.escape(neighbor.ticker)}{rel_chip}{reason_chip} "
            f"<i>{html.escape(neighbor.failed_reason or '未通过')}</i>"
        )


# ---------------------------------------------------------------------------
# Institutional 13F (玩法 ④b)
# ---------------------------------------------------------------------------

_CHANGE_CHIP = {
    "new":      "🟢 新进",
    "exit":     "🔴 清仓",
    "increase": "🟢 加仓",
    "decrease": "🟡 减仓",
}


def format_institutional_messages(report: InstitutionalReport) -> list[str]:
    """Split the institutional report into multiple Telegram-sized messages.

    Returns a list: one summary header, then one message per manager.
    Each individual message stays comfortably under the 4096-char limit.
    """
    if not report.new_filings:
        return [
            f"<b>🏛️ 机构持仓变动 · {report.date}</b>\n"
            f"<i>无新 13F 文件</i>"
        ]

    # Group changes per manager
    by_manager: dict[str, list[PositionChange]] = {}
    for c in report.changes:
        by_manager.setdefault(c.manager_name, []).append(c)

    messages: list[str] = []

    # 1) Summary header
    messages.append(_format_institutional_summary(report))
    emit(
        "render",
        card="institutional_summary",
        num_managers=len(report.new_filings),
        num_changes=len(report.changes),
    )

    # 2) One message per manager (preserves new_filings order)
    total = len(report.new_filings)
    for idx, filing in enumerate(report.new_filings, 1):
        m_changes = by_manager.get(filing.manager_name, [])
        messages.append(_format_one_manager(filing, m_changes, idx, total))
        emit(
            "render",
            card="manager_detail",
            manager=filing.manager_name,
            changes_shown=min(10, len(m_changes)),
            changes_total=len(m_changes),
        )

    return messages


def _format_institutional_summary(report: InstitutionalReport) -> str:
    """The leading header card that orients the user before the per-manager dump."""
    lines = [
        f"<b>🏛️ 机构持仓变动 · {report.date}</b>",
        "━━━━━━━━━━━━━━━━━━━━",
        f"新 13F: <b>{len(report.new_filings)}</b> 个 manager · "
        f"显著变动 <b>{len(report.changes)}</b> 笔",
        "",
        "<i>↓ 详情见后续消息（每个 manager 一条）</i>",
        "",
        f"<i>🔧 EDGAR ×{report.api_calls} · "
        f"DeepSeek {report.llm_tokens:,} tokens</i>",
    ]
    return "\n".join(lines)


def _format_one_manager(
    filing,
    changes: list[PositionChange],
    idx: int,
    total: int,
) -> str:
    """One Telegram message for a single manager's changes."""
    warning = _quarter_age_warning(filing.quarter)
    lines = [
        f"<b>🏛️ {html.escape(filing.manager_name)} · "
        f"{html.escape(filing.quarter)}</b>  <i>({idx}/{total})</i>{warning}",
        f"📅 Filed {filing.filing_date} · "
        f"Portfolio <code>${_short_money(filing.portfolio_value)}</code>",
        "",
    ]

    if not changes:
        lines.append("<i>无显著变动</i>")
    else:
        # Up to 10 per manager — fits comfortably in one Telegram message
        for c in changes[:10]:
            lines.append(_format_change_line(c))
            if c.interpretation:
                lines.append(f"   💡 {html.escape(c.interpretation)}")
        if len(changes) > 10:
            lines.append("")
            lines.append(
                f"<i>… 其余 {len(changes) - 10} 笔小幅变动省略</i>"
            )

    return _safe_truncate("\n".join(lines))


def _safe_truncate(body: str, max_len: int = 4000) -> str:
    """Truncate while keeping HTML well-formed (close any opened tags).

    Telegram's HTML parser is strict — an unclosed <code>/<b>/<i> raises
    BadRequest. We:
      1. Cut at max_len
      2. Back up if we sliced in the middle of a tag
      3. Back up to the last newline for clean visuals
      4. Close any tag types we left open
    """
    import re

    if len(body) <= max_len:
        return body

    truncated = body[:max_len]

    # Step 1: avoid cutting inside a tag
    last_gt = truncated.rfind(">")
    last_lt = truncated.rfind("<")
    if last_lt > last_gt:
        truncated = truncated[:last_lt]

    # Step 2: prefer a newline boundary for cleanliness
    last_nl = truncated.rfind("\n")
    if last_nl > max_len - 500:
        truncated = truncated[:last_nl]

    # Step 3: balance any unclosed tags
    for tag in ("b", "code", "i", "pre", "a"):
        opens = len(re.findall(rf"<{tag}\b[^>]*>", truncated))
        closes = len(re.findall(rf"</{tag}>", truncated))
        if opens > closes:
            truncated += f"</{tag}>" * (opens - closes)

    return truncated.rstrip() + "\n\n<i>(消息过长，已截断)</i>"


def format_portfolio_snapshot(filing, positions, top_n: int = 10) -> str:
    """Top-N current holdings ranked by market value — for /13f full view."""
    if not positions:
        return (
            f"<b>🏛️ {html.escape(filing.manager_name)} · "
            f"{html.escape(filing.quarter)}</b>\n<i>无持仓数据</i>"
        )

    def _v(p):
        return p["market_value"] if isinstance(p, dict) else p.market_value

    def _t(p):
        return (p["ticker"] if isinstance(p, dict) else p.ticker) or ""

    def _c(p):
        return p["cusip"] if isinstance(p, dict) else p.cusip

    def _i(p):
        return (p["issuer_name"] if isinstance(p, dict) else p.issuer_name) or ""

    sorted_pos = sorted(positions, key=_v, reverse=True)
    total = filing.portfolio_value or 1
    warning = _quarter_age_warning(filing.quarter)

    lines = [
        f"<b>🏛️ {html.escape(filing.manager_name)} · "
        f"{html.escape(filing.quarter)} · 完整组合{warning}</b>",
        f"<i>📅 Filed {filing.filing_date} · Total "
        f"<code>${_short_money(filing.portfolio_value)}</code></i>",
        "━━━━━━━━━━━━━━━━━━━━",
    ]
    for i, p in enumerate(sorted_pos[:top_n], 1):
        ticker = _t(p) or (_c(p) or "?")[:8]
        value = _v(p)
        pct = value / total
        issuer = _i(p)
        lines.append(
            f" {i:>2}. <b>{html.escape(ticker)}</b>  "
            f"<code>${_short_money(value)}</code>  "
            f"<code>{pct:.1%}</code>"
            f"  <i>{html.escape(issuer[:30])}</i>"
        )

    if len(sorted_pos) > top_n:
        lines.append(
            f"<i>其余 {len(sorted_pos) - top_n} 个持仓省略</i>"
        )
    emit(
        "render",
        card="portfolio_snapshot",
        manager=filing.manager_name,
        quarter=filing.quarter,
        total_value_usd=filing.portfolio_value,
        positions_shown=min(top_n, len(sorted_pos)),
        positions_total=len(sorted_pos),
    )
    return "\n".join(lines)


def format_portfolio(snapshot: dict) -> str:
    """Format Alpaca portfolio snapshot for /portfolio command."""
    acc = snapshot.get("account", {}) or {}
    positions = snapshot.get("positions", []) or []
    paper_chip = "📝 PAPER" if acc.get("paper") else "💵 LIVE"

    lines = [
        f"<b>💼 Alpaca 账户 · {paper_chip}</b>",
        "━━━━━━━━━━━━━━━━━━━━",
        f"组合价值 <code>${_short_money(acc.get('portfolio_value'))}</code>  ·  "
        f"现金 <code>${_short_money(acc.get('cash'))}</code>",
        f"购买力 <code>${_short_money(acc.get('buying_power'))}</code>  ·  "
        f"状态 <i>{html.escape(acc.get('status', '—'))}</i>",
        "",
    ]

    if not positions:
        lines.append("<i>当前无持仓。用 Alpaca 网页或 API 下单后这里会更新。</i>")
        return "\n".join(lines)

    lines.append(f"<b>持仓（{len(positions)}）</b>")
    for p in positions[:15]:
        upl = p.get("unrealized_pl", 0) or 0
        upl_pct = p.get("unrealized_pl_pct", 0) or 0
        side_chip = "🟢" if upl >= 0 else "🔴"
        lines.append(
            f"  {side_chip} <b>{html.escape(p['symbol'])}</b>  "
            f"<code>{p.get('qty', 0):,.0f} sh</code>  @ "
            f"<code>${p.get('avg_entry_price', 0):,.2f}</code>"
        )
        lines.append(
            f"     市值 <code>${_short_money(p.get('market_value'))}</code>  ·  "
            f"P/L <code>${_short_money(abs(upl))}</code> "
            f"<b>{upl_pct:+.2%}</b>"
        )

    if len(positions) > 15:
        lines.append(f"<i>其余 {len(positions) - 15} 个持仓省略</i>")
    emit("render", card="portfolio_card", positions=len(positions))
    return "\n".join(lines)


def format_pnl(snapshot: dict) -> str:
    """Format Alpaca P&L snapshot for /pnl command."""
    paper_chip = "📝 PAPER" if snapshot.get("paper") else "💵 LIVE"
    intraday = snapshot.get("intraday_pl", 0) or 0
    intraday_pct = snapshot.get("intraday_pl_pct", 0) or 0
    arrow = "📈" if intraday > 0 else ("📉" if intraday < 0 else "📊")

    lines = [
        f"<b>{arrow} 当日盈亏 · {paper_chip}</b>",
        f"<i>{snapshot.get('date', '—')}</i>",
        "━━━━━━━━━━━━━━━━━━━━",
        f"当前权益 <code>${_short_money(snapshot.get('equity'))}</code>",
        f"昨日收盘 <code>${_short_money(snapshot.get('last_equity'))}</code>",
        f"日内 P/L <code>${_short_money(abs(intraday))}</code>  "
        f"<b>{intraday_pct:+.2%}</b>",
        "",
        f"组合价值 <code>${_short_money(snapshot.get('portfolio_value'))}</code>  ·  "
        f"现金 <code>${_short_money(snapshot.get('cash'))}</code>",
        f"多头敞口 <code>${_short_money(snapshot.get('long_value'))}</code>  ·  "
        f"空头敞口 <code>${_short_money(abs(snapshot.get('short_value', 0)))}</code>",
        f"持仓数 <code>{snapshot.get('position_count', 0)}</code>",
    ]
    emit("render", card="pnl_card")
    return "\n".join(lines)


def format_alert_list(alerts: list[dict]) -> str:
    """Format the user's open price alerts."""
    if not alerts:
        return (
            "<b>🔔 价格提醒</b>\n"
            "<i>当前没有未触发的提醒。</i>\n"
            "用 <code>/alert TICKER PRICE [above|below]</code> 创建。"
        )
    lines = [
        f"<b>🔔 价格提醒（{len(alerts)} 条未触发）</b>",
        "━━━━━━━━━━━━━━━━━━━━",
    ]
    for a in alerts:
        sign = "≥" if a["direction"] == "above" else "≤"
        lines.append(
            f"  <code>#{a['id']}</code>  <b>{html.escape(a['ticker'])}</b>  "
            f"{sign} <code>${a['target_price']:,.2f}</code>  "
            f"<i>· created {a['created_at'][:10]}</i>"
        )
    lines.append("")
    lines.append("<i>用 <code>/alert_remove ID</code> 删除任一条。</i>")
    emit("render", card="alerts_list", num_alerts=len(alerts))
    return "\n".join(lines)


def format_intraday_anomaly(signal: dict) -> str:
    """Lightweight intraday-anomaly card. No LLM attribution by design —
    intraday fires only on hard metrics, deeper analysis happens at 17:35 ET
    when the post-market Anomaly Monitor runs.
    """
    ticker = signal["ticker"]
    pct = signal["price_change_pct"]
    arrow = "📈" if pct >= 0 else "📉"
    contra = " <b>★ 逆势</b>" if signal.get("contrarian") else ""
    time_chip = signal.get("time_et", "")

    lines = [
        f"<b>⚡ 盘中异动 · {html.escape(ticker)} · {html.escape(time_chip)} ET</b>",
        "━━━━━━━━━━━━━━━━━━━━",
        f"{arrow} 现价 <code>${signal['price']:,.2f}</code>  "
        f"<b>{pct:+.2%}</b>  "
        f"<i>vs 开盘 <code>${signal['open_price']:,.2f}</code></i>",
        f"当日成交 <code>{_short_count(signal['volume_today'])}</code>  ·  "
        f"节奏 <b>{signal['volume_pace']:.1f}×</b>  "
        f"<i>(已交易 {signal['market_progress'] * 100:.0f}% 时段)</i>",
    ]

    if signal.get("sector_etf") and signal.get("sector_return") is not None:
        rel = signal.get("relative_pp") or 0
        sec_arrow = "↑" if rel > 0 else ("↓" if rel < 0 else "→")
        lines.append(
            f"<i>对比 <b>{html.escape(signal['sector_etf'])}</b> "
            f"{signal['sector_return']:+.2%}  ·  "
            f"差 {sec_arrow}<code>{rel * 100:+.2f}pp</code></i>"
            f"{contra}"
        )

    lines.append("")
    lines.append(
        f"<i>用 <code>/why {html.escape(ticker)}</code> 看盘后完整归因（17:35 ET 起效）</i>"
    )
    return "\n".join(lines)


def format_alert_fired(fired: dict, ticker_change_pct: float | None = None) -> str:
    """Format a triggered alert for push notification."""
    sign = "突破" if fired["direction"] == "above" else "跌破"
    arrow = "📈" if fired["direction"] == "above" else "📉"
    chg = ""
    if ticker_change_pct is not None:
        chg = f"  <b>{ticker_change_pct:+.2%}</b>"
    return (
        f"<b>{arrow} 价格提醒触发 · {html.escape(fired['ticker'])}</b>\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        f"现价 <code>${fired['fired_price']:,.2f}</code>{chg}\n"
        f"{sign}目标 <code>${fired['target_price']:,.2f}</code>\n"
        "\n"
        f"<i>用 <code>/why {html.escape(fired['ticker'])}</code> 看异动归因，"
        f"或 <code>/summary {html.escape(fired['ticker'])}</code> 看综合概览。</i>"
    )


def format_holders(
    ticker: str,
    held: list[dict],
    not_held: list[str],
    unknown: list[str] | None = None,
) -> str:
    """Cross-manager holder distribution with 3-state classification.

    held     — confirmed holdings (DB has filing AND ticker present)
    not_held — confirmed absent  (DB has filing, ticker NOT present)
    unknown  — no filing in DB yet (cannot determine, scheduler hasn't caught up)
    """
    unknown = unknown or []
    total = len(held) + len(not_held) + len(unknown)
    lines = [
        f"<b>🏛️ {html.escape(ticker)} 机构持有人分布</b>",
        f"<i>追踪 {total} manager · 持有 {len(held)} · "
        f"未持有 {len(not_held)} · 待更新 {len(unknown)}</i>",
        "━━━━━━━━━━━━━━━━━━━━",
    ]

    if held:
        lines.append("<b>持有：</b>")
        for i, h in enumerate(held, 1):
            stale = _quarter_age_warning(h["quarter"], terse=True)
            lines.append(
                f" {i:>2}. <b>{html.escape(h['manager'])}</b>  "
                f"<code>${_short_money(h['value'])}</code>  "
                f"<code>{h['pct']:.2%}</code> of portfolio  "
                f"<i>{html.escape(h['quarter'])}{stale}</i>"
            )
    else:
        lines.append("<i>已有数据的 manager 均未持有该 ticker</i>")

    if not_held:
        lines.append("")
        lines.append(
            f"<b>已确认未持有：</b> <i>{html.escape(', '.join(not_held))}</i>"
        )

    if unknown:
        lines.append("")
        lines.append(
            f"<b>待更新（DB 暂无 13F）：</b> <i>{html.escape(', '.join(unknown))}</i>"
        )
        lines.append(
            "<i>下次 scheduler 跑完（周二 / 周五 18:00 ET）会补齐。</i>"
        )
    emit("render", card="holders_card", ticker=ticker,
         num_held=len(held or []), num_not_held=len(not_held or []),
         num_unknown=len(unknown or []))
    return "\n".join(lines)


def format_etf_snapshot(
    etf: str,
    holdings: list[dict],
    snapshot_date: str,
    top_n: int = 15,
    daily_changes: list[dict] | None = None,
) -> str:
    """ETF current holdings + optional 24h changes for /etf SYMBOL."""
    if not holdings:
        return (
            f"<b>📈 {html.escape(etf)} · {html.escape(snapshot_date)}</b>\n"
            "<i>无 CSV 数据返回</i>"
        )

    # Holdings may arrive as ETFHolding dataclass OR plain dict
    def _g(h, k, default=None):
        if isinstance(h, dict):
            return h.get(k, default)
        return getattr(h, k, default)

    sorted_h = sorted(holdings, key=lambda h: _g(h, "weight_pct", 0), reverse=True)
    total_value = sum(_g(h, "market_value", 0) for h in holdings)

    lines = [
        f"<b>📈 {html.escape(etf)} · {html.escape(snapshot_date)}</b>",
        f"<i>{len(holdings)} 个持仓 · Total "
        f"<code>${_short_money(total_value)}</code></i>",
        "━━━━━━━━━━━━━━━━━━━━",
    ]

    for i, h in enumerate(sorted_h[:top_n], 1):
        ticker = html.escape(_g(h, "ticker") or "?")
        wt = _g(h, "weight_pct", 0) or 0
        mv = _g(h, "market_value", 0) or 0
        company = _g(h, "company", "") or ""
        lines.append(
            f" {i:>2}. <b>{ticker}</b>  "
            f"<code>{wt:.2f}%</code>  "
            f"<code>${_short_money(mv)}</code>"
            f"  <i>{html.escape(company[:25])}</i>"
        )

    if len(sorted_h) > top_n:
        lines.append(f"<i>其余 {len(sorted_h) - top_n} 个持仓省略</i>")

    # 24h changes block (if we have a yesterday snapshot)
    if daily_changes:
        ups = [c for c in daily_changes if c.get("shares_diff", 0) > 0][:5]
        downs = [c for c in daily_changes if c.get("shares_diff", 0) < 0][:5]
        new = [c for c in daily_changes if c.get("is_new")][:5]
        exits = [c for c in daily_changes if c.get("is_exit")][:5]

        if new or exits or ups or downs:
            lines.append("")
            lines.append("<b>📊 24h 调仓：</b>")
            for c in new:
                lines.append(
                    f"  🟢 新进 <b>{html.escape(c['ticker'])}</b>  "
                    f"<code>{c.get('weight_pct', 0):.2f}%</code>"
                )
            for c in exits:
                lines.append(
                    f"  🔴 清仓 <b>{html.escape(c['ticker'])}</b>"
                )
            for c in ups:
                lines.append(
                    f"  🟢 加仓 <b>{html.escape(c['ticker'])}</b>  "
                    f"<code>{c.get('shares_diff_pct', 0):+.1%}</code>"
                )
            for c in downs:
                lines.append(
                    f"  🟡 减仓 <b>{html.escape(c['ticker'])}</b>  "
                    f"<code>{c.get('shares_diff_pct', 0):+.1%}</code>"
                )

    emit("render", card="etf_snapshot",
         etf=etf, snapshot_date=snapshot_date,
         positions=len(holdings or []),
         daily_changes=len(daily_changes or []))
    return "\n".join(lines)


def _quarter_age_warning(quarter: str, *, terse: bool = False) -> str:
    """Stale-filing chip — '⚠️' / '⚠️ 已 N 月未更新' for quarters older than 6 months.

    13F-HR is filed within 45 days of quarter end. Anything older than ~6 months
    means the manager skipped a quarter; >1 year usually means they fell below
    the $100M reporting threshold (Greenlight, post-2023) or deregistered.
    """
    from datetime import date as _date

    try:
        year_str, q_str = quarter.split("-Q")
        q = int(q_str)
        month_end = {1: 3, 2: 6, 3: 9, 4: 12}[q]
        day_end = 31 if month_end in (3, 12) else 30
        qd = _date(int(year_str), month_end, day_end)
    except (ValueError, KeyError, AttributeError):
        return ""

    days = (_date.today() - qd).days
    if days <= 180:
        return ""
    if terse:
        return " ⚠️"
    months = days // 30
    return f" ⚠️ <i>已 {months} 月未更新</i>"


def _format_change_line(c: PositionChange) -> str:
    """One line summarizing a single position change."""
    chip = _CHANGE_CHIP.get(c.change_type, "•")
    ticker_label = (
        f"<b>{html.escape(c.ticker)}</b>"
        if c.ticker
        else f"<code>{html.escape(c.cusip[:8])}</code>"
    )
    # For exits show the previous value (what was sold); else current.
    value_to_show = c.prev_value if c.change_type == "exit" else c.current_value
    value_str = _short_money(value_to_show)
    pct_str = (
        f" ({c.current_pct:.1%})"
        if c.change_type != "exit" and c.current_pct > 0.005
        else ""
    )
    universe_tag = " ★ 已在 universe" if c.in_universe else ""
    issuer = html.escape(c.issuer_name[:28])

    return (
        f"   {chip} {ticker_label} {issuer} "
        f"<code>${value_str}</code>{pct_str}{universe_tag}"
    )


def render_price_sparkline(
    prices: list[float],
    *,
    title: str = "",
) -> bytes:
    """Compact price line for the recent N days — used in anomaly alerts."""
    fig, ax = plt.subplots(figsize=(8, 2.8))
    if not prices:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
    else:
        x = list(range(len(prices)))
        is_up = prices[-1] >= prices[0]
        color = "#2ca02c" if is_up else "#d62728"
        ax.plot(x, prices, linewidth=2.5, color=color, marker="o", markersize=4)
        ax.fill_between(x, prices, prices[0], alpha=0.15, color=color)
        ax.axhline(prices[0], color="#666", linewidth=0.8, linestyle="--", alpha=0.5)
        if title:
            ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel("Close ($)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.3)
        ax.set_xticks([])

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Earnings cards (Phase 1)
# ---------------------------------------------------------------------------
# Implementation lives in v2/reporting/_earnings_formatters.py — kept
# separate so unit tests can import it without dragging in matplotlib /
# v2.backtesting / v2.monitoring (this module's transitive deps include
# v2.data through v2.lateral, which production has and sandbox does not).
# Re-exported here so v2.reporting.formatters and v2.reporting both expose
# the same surface as every other format_* function.
from v2.reporting._earnings_formatters import (  # noqa: E402
    format_earnings_calendar,
    format_earnings_pending,
    format_earnings_reminder,
    format_earnings_summary,
    format_earnings_view,
)


# ---------------------------------------------------------------------------
# Portfolio cards (Phase 2)
# ---------------------------------------------------------------------------
# Same pattern as the earnings re-export above. Implementation lives in
# v2/portfolio/_bot_cards.py to keep byte-equal tests sandbox-runnable.
from v2.reporting._portfolio_formatters import (  # noqa: E402
    format_portfolio_pnl_period,
    format_portfolio_risk_card,
    format_portfolio_risk_view,
    format_portfolio_weekly_card,
)


# ---------------------------------------------------------------------------
# SEC cards (Phase 3)
# ---------------------------------------------------------------------------
# Same pattern. Implementation lives in v2/sec/_bot_cards.py.
from v2.reporting._sec_formatters import (  # noqa: E402
    format_sec_8k_card,
    format_sec_8k_view,
    format_sec_form4_cluster_card,
    format_sec_form4_individual_card,
    format_sec_form4_view,
    format_sec_insider_digest,
)


# ---------------------------------------------------------------------------
# Macro cards (Phase 4)
# ---------------------------------------------------------------------------
# Same pattern. Implementation lives in v2/macro/_bot_cards.py.
from v2.reporting._macro_formatters import (  # noqa: E402
    format_macro_claims_card,
    format_macro_daily_snapshot,
    format_macro_dashboard,
    format_macro_fomc_card,
    format_macro_release_card,
    format_macro_weekly_recap,
)


# ---------------------------------------------------------------------------
# ARK alert cards (Phase 5a)
# ---------------------------------------------------------------------------
# Same pattern. Implementation lives in v2/etf/_ark_alert_cards.py
# (sandbox-runnable; v2/etf/ has no v2.data deps).
from v2.reporting._ark_alert_formatters import (  # noqa: E402
    format_ark_alert,
    format_ark_summary,
)
