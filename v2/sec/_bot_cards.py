"""SEC card formatters — pure functions, no v2.data deps.

Single source of truth since Stage 5. Re-exported through
``v2.reporting.format_sec_*`` (and ``v2.reporting.formatters``) so
production cron + bot code consume the public namespace. The
implementation lives in ``v2/sec/`` (not ``v2/reporting/``) for the
same reason as Phase 1's earnings cards and Phase 2's portfolio cards:
``v2/reporting/__init__.py`` transitively pulls matplotlib + v2.lateral,
which require v2.data. Keeping the implementation here lets the
byte-equal tests stay sandbox-runnable.

Five public formatters:

- :func:`format_sec_8k_card` — ⑪ daily 17:05 ET cron card (single
  8-K filing with all its items aggregated).
- :func:`format_sec_8k_view` — ``/8k`` bot card. Multi-filing window
  with explicit empty state.
- :func:`format_sec_form4_individual_card` — ⑫ daily 17:45 ET cron
  card for one P or S transaction.
- :func:`format_sec_form4_cluster_card` — ⑫ cron card for ≥3
  distinct-insider same-day same-direction cluster.
- :func:`format_sec_form4_view` — ``/insiders`` bot card. Summary
  view: P/S top-line, A/M/F/G/C aggregated counts, cluster section.
"""

from __future__ import annotations

import html

from v2.sec.models import EightKEvent, Form4Cluster, Form4Transaction


__all__ = [
    "format_sec_8k_card",
    "format_sec_8k_view",
    "format_sec_form4_individual_card",
    "format_sec_form4_cluster_card",
    "format_sec_form4_view",
    "format_sec_insider_digest",
]


_TIER_EMOJI = {"P0": "🚨", "P1": "📋", "P2": "📎", "P3": "📌"}


def _tier_emoji(tier: str) -> str:
    return _TIER_EMOJI.get(tier, "📌")


# ---------------------------------------------------------------------------
# 8-K — single filing card (⑪ cron)
# ---------------------------------------------------------------------------

def format_sec_8k_card(
    event: EightKEvent,
    *,
    is_held: bool,
    is_watchlist: bool,
) -> str:
    """Render one 8-K filing as a single card.

    Items listed in document order with per-item tier badge. The 2.02
    item, if present alongside other material items, gets a "(⑧ 处理)"
    annotation so the reader knows the earnings data is covered by
    the 21:00 ET earnings cron — not missed.

    The 5.02 extraction sub-block is rendered when the item's
    ``extracted_meta`` carries non-empty departures / appointments;
    LLM failure leaves the meta empty and the sub-block silently drops.
    """
    filing = event.filing
    tier_top = event.max_priority_tier
    emoji = _tier_emoji(tier_top)

    badge = "🟢 持仓股" if is_held else "👁 关注列表" if is_watchlist else ""

    lines: list[str] = [
        f"<b>{emoji} SEC 8-K · {filing.ticker} · {tier_top}</b>",
        f"申报日：<code>{filing.filing_date}</code>"
        + (" <i>(amendment)</i>" if filing.is_amendment else ""),
    ]
    if badge:
        lines.append(badge)

    lines.append("")
    lines.append("<b>项目</b>")
    for item in event.items:
        emoji_i = _tier_emoji(item.priority_tier)
        annotation = "  <i>(数据由 ⑧ 处理)</i>" if item.code == "2.02" else ""
        lines.append(
            f"  {emoji_i} <code>{item.code}</code> "
            f"[{item.priority_tier}] {item.description}{annotation}"
        )

    # 5.02 extraction summary if present
    item_5_02 = next(
        (it for it in event.items if it.code == "5.02"), None,
    )
    if item_5_02 and item_5_02.extracted_meta:
        meta = item_5_02.extracted_meta
        departures = meta.get("departures") or []
        appointments = meta.get("appointments") or []
        if departures or appointments:
            lines.append("")
            lines.append("<b>5.02 抽取</b>")
            for d in departures[:3]:
                name = d.get("name", "")
                title = d.get("title", "")
                if name:
                    lines.append(f"  📤 离职：{name} ({title})")
            for a in appointments[:3]:
                name = a.get("name", "")
                title = a.get("title", "")
                if name:
                    lines.append(f"  📥 任命：{name} ({title})")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 8-K — multi-filing view (/8k bot)
# ---------------------------------------------------------------------------

def format_sec_8k_view(
    events: list[EightKEvent],
    ticker: str,
    days: int,
    *,
    membership_note: str = "",
) -> str:
    """Render the ``/8k TICKER`` window view.

    Empty events list → ``"无 8-K 申报"`` placeholder. Multi-filing
    list renders newest first (caller is responsible for ordering;
    SEC client already returns newest first).

    All user-supplied strings (ticker, filing_date, accession_number,
    item.code, item.description) are HTML-escaped here — the bot's
    surface needs the escape even though the cron card's source data
    comes from the SEC and is trusted.
    """
    if not events:
        lines = [
            f"<b>📋 SEC 8-K · {html.escape(ticker)} · 过去 {days} 天</b>",
            "━━━━━━━━━━━━━━━━━━━━",
            "<i>无 8-K 申报</i>",
        ]
        if membership_note:
            lines.append("")
            lines.append(membership_note)
        return "\n".join(lines)

    total_items = sum(len(ev.items) for ev in events)

    lines: list[str] = [
        f"<b>📋 SEC 8-K · {html.escape(ticker)} · 过去 {days} 天</b>",
        "━━━━━━━━━━━━━━━━━━━━",
        f"共 <b>{len(events)}</b> 个 filings · "
        f"<b>{total_items}</b> 个 items",
    ]
    if membership_note:
        lines.append(membership_note)

    for ev in events:
        f = ev.filing
        lines.append("")
        amendment_tag = " <i>(amendment)</i>" if f.is_amendment else ""
        lines.append(
            f"<b>{html.escape(f.filing_date)}</b> · "
            f"<code>{html.escape(f.accession_number)}</code>{amendment_tag}"
        )

        for it in ev.items:
            emoji = _tier_emoji(it.priority_tier)
            annotation = ""
            if it.code == "2.02":
                annotation = " <i>(⑧ 处理)</i>"
            lines.append(
                f"  {emoji} <code>{html.escape(it.code)}</code> "
                f"[{it.priority_tier}] {html.escape(it.description)}{annotation}"
            )
            if it.code == "5.02":
                lines.extend(_format_5_02_extract_lines(it.extracted_meta))

    return "\n".join(lines)


def _format_5_02_extract_lines(extracted: dict) -> list[str]:
    """5.02 LLM extraction sub-block (departures + appointments).

    Empty extracted dict → ``"(姓名待解析)"`` placeholder per Stage 4
    spec (covers both LLM-failure and silent-no-people cases).
    """
    if not extracted:
        return ["       <i>(姓名待解析)</i>"]

    departures = extracted.get("departures") or []
    appointments = extracted.get("appointments") or []
    out: list[str] = []
    for p in departures[:3]:
        name = html.escape(str(p.get("name", "") or ""))
        title = html.escape(str(p.get("title", "") or ""))
        if name:
            out.append(f"       离职: <b>{name}</b> ({title})")
    for p in appointments[:3]:
        name = html.escape(str(p.get("name", "") or ""))
        title = html.escape(str(p.get("title", "") or ""))
        if name:
            out.append(f"       任命: <b>{name}</b> ({title})")
    if not out:
        out.append("       <i>(姓名待解析)</i>")
    return out


# ---------------------------------------------------------------------------
# Form 4 — single P/S transaction card (⑫ cron)
# ---------------------------------------------------------------------------

_ROLE_PRETTY = {
    "CEO": "🔴 CEO",
    "CFO": "🔴 CFO",
    "COO": "🟠 COO",
    "Chairman": "🟠 Chairman",
    "President": "🟠 President",
    "GC": "🟡 General Counsel",
    "Director": "🟢 董事",
    "Officer": "🔵 高管",
    "10% holder": "🔵 10% 大股东",
}


def _role_label(role: str | None) -> str:
    if not role:
        return ""
    return _ROLE_PRETTY.get(role, role)


def _fmt_usd_cron(v: float | None) -> str:
    """USD formatter for cron Form 4 cards (M / K / bare units)."""
    if v is None:
        return "未披露"
    if v >= 1_000_000:
        return f"${v / 1_000_000:.2f}M"
    if v >= 1_000:
        return f"${v / 1_000:.1f}K"
    return f"${v:,.0f}"


def format_sec_form4_individual_card(
    tx: Form4Transaction,
    *,
    is_held: bool,
    is_watchlist: bool,
) -> str:
    """Single P or S transaction → one card.

    10b5-1 plan vs discretionary trade is annotated explicitly so the
    reader can tell pre-arranged sales (low signal) from spontaneous
    insider purchases (high signal).
    """
    f = tx.filing
    badge = "🟢 持仓股" if is_held else "👁 关注列表" if is_watchlist else ""

    if tx.transaction_code == "P":
        direction_label = "📥 内部人买入"
    else:
        direction_label = "📤 内部人卖出"

    plan_tag = "<i>(10b5-1 plan)</i>" if tx.is_10b5_1 else "<i>(discretionary)</i>"

    lines: list[str] = [
        f"<b>{direction_label} · {f.ticker}</b>",
        f"申报：<code>{f.filing_date}</code> · 交易：<code>{tx.transaction_date}</code>",
    ]
    if badge:
        lines.append(badge)
    lines.append("")

    role = _role_label(tx.insider_role)
    role_suffix = f" · {role}" if role else ""
    lines.append(f"申报人：<b>{tx.insider_name or '?'}</b>{role_suffix}")

    lines.append(
        f"交易：<code>{tx.shares:,.0f}</code> 股 × "
        f"{_fmt_usd_cron(tx.price)}/股 = <b>{_fmt_usd_cron(tx.transaction_usd)}</b> "
        f"{plan_tag}"
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Form 4 — cluster card (⑫ cron)
# ---------------------------------------------------------------------------

def format_sec_form4_cluster_card(
    cluster: Form4Cluster,
    *,
    is_held: bool,
    is_watchlist: bool,
) -> str:
    """Cluster card — N insiders same day same direction."""
    badge = "🟢 持仓股" if is_held else "👁 关注列表" if is_watchlist else ""

    if cluster.direction == "purchase":
        emoji, label = "📥", "内部人集群买入"
    else:
        emoji, label = "📤", "内部人集群卖出"

    lines: list[str] = [
        f"<b>{emoji} {label} · {cluster.ticker}</b>",
        f"日期：<code>{cluster.cluster_date}</code> · "
        f"{cluster.transaction_count} 笔 / {len(cluster.insider_names)} 人",
    ]
    if badge:
        lines.append(badge)
    lines.append("")

    lines.append(f"总金额：<b>{_fmt_usd_cron(cluster.total_usd)}</b>")
    lines.append("")

    lines.append("<b>申报人</b>")
    for name in cluster.insider_names[:6]:
        lines.append(f"  • {name}")
    if len(cluster.insider_names) > 6:
        lines.append(f"  ... 另 {len(cluster.insider_names) - 6} 人")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Form 4 — multi-Form 4 view (/insiders bot)
# ---------------------------------------------------------------------------

_NOISE_LABELS = {
    "A": "Award (薪酬授予)",
    "M": "Exercise (option vest)",
    "F": "Tax (RSU vest 税款)",
    "G": "Gift (赠予)",
    "C": "Conversion (转换)",
}


def _fmt_money_kb(v: float | None) -> str:
    """USD formatter for bot insider view — same K/M but signed-safe."""
    if v is None:
        return "未披露"
    abs_v = abs(v)
    if abs_v >= 1_000_000:
        return f"${v / 1_000_000:.2f}M"
    if abs_v >= 1_000:
        return f"${v / 1_000:.1f}K"
    return f"${v:,.0f}"


def _fmt_tx_one_liner(tx: Form4Transaction) -> str:
    """One-line description of a Form 4 transaction for the view card."""
    name = html.escape(tx.insider_name or "?")
    role_tag = f" ({html.escape(tx.insider_role)})" if tx.insider_role else ""
    plan = " · 10b5-1" if tx.is_10b5_1 else ""
    usd = _fmt_money_kb(tx.transaction_usd)
    date_s = html.escape(tx.transaction_date or tx.filing.filing_date)
    return f"{name}{role_tag} · {usd} · {date_s}{plan}"


def format_sec_form4_view(
    ticker: str,
    transactions: list[Form4Transaction],
    clusters: list[Form4Cluster],
    noise_summary: dict[str, int],
    days_back: int,
    *,
    membership_note: str = "",
) -> str:
    """Render the ``/insiders TICKER`` window summary card.

    Splits transactions by code into:
    - P (Purchase) — count + total USD + biggest one-liner
    - S (Sale) — count + total USD + biggest one-liner
    - A / M / F / G / C — counts only (noise codes per Stage 0)

    Empty transactions AND empty noise → ``"无 Form 4 申报"`` placeholder.
    """
    if not transactions and not noise_summary:
        lines = [
            f"<b>📥 内部人交易摘要 · {html.escape(ticker)} · 过去 {days_back} 天</b>",
            "━━━━━━━━━━━━━━━━━━━━",
            "<i>无 Form 4 申报</i>",
        ]
        if membership_note:
            lines.append("")
            lines.append(membership_note)
        return "\n".join(lines)

    purchases = [t for t in transactions if t.transaction_code == "P"]
    sales = [t for t in transactions if t.transaction_code == "S"]

    lines: list[str] = [
        f"<b>📥 内部人交易摘要 · {html.escape(ticker)} · 过去 {days_back} 天</b>",
        "━━━━━━━━━━━━━━━━━━━━",
    ]
    if membership_note:
        lines.append(membership_note)
        lines.append("")

    # Purchase block
    if purchases:
        total_p = sum((t.transaction_usd or 0.0) for t in purchases)
        lines.append(
            f"<b>P (Purchase): {len(purchases)} 笔</b>, 总 {_fmt_money_kb(total_p)}"
        )
        biggest_p = max(purchases, key=lambda t: (t.transaction_usd or 0.0))
        lines.append(f"  最大: {_fmt_tx_one_liner(biggest_p)}")
    else:
        lines.append("<b>P (Purchase): 0 笔</b>")

    # Sale block
    if sales:
        total_s = sum((t.transaction_usd or 0.0) for t in sales)
        lines.append(
            f"<b>S (Sale):</b> {len(sales)} 笔, 总 {_fmt_money_kb(total_s)}"
        )
        biggest_s = max(sales, key=lambda t: (t.transaction_usd or 0.0))
        lines.append(f"  最大: {_fmt_tx_one_liner(biggest_s)}")
    else:
        lines.append("<b>S (Sale):</b> 0 笔")

    # Noise counts (A/M/F/G/C)
    if noise_summary:
        lines.append("")
        for code in ("A", "M", "F", "G", "C"):
            n = noise_summary.get(code, 0)
            if n > 0:
                label = _NOISE_LABELS[code]
                lines.append(f"<b>{code}</b>: {n} 笔  <i>({label})</i>")

    # Clusters
    lines.append("")
    if not clusters:
        lines.append(
            f"<b>集群:</b> 无同日 ≥3 distinct insiders 集群 ({days_back} 天内)"
        )
    else:
        lines.append(f"<b>集群:</b> {len(clusters)} 个")
        for c in clusters[:3]:
            direction_label = "买入" if c.direction == "purchase" else "卖出"
            names_preview = ", ".join(html.escape(n) for n in c.insider_names[:4])
            more = f", +{len(c.insider_names) - 4}" if len(c.insider_names) > 4 else ""
            lines.append(
                f"  {html.escape(c.cluster_date)} · {c.transaction_count} 笔 / "
                f"{len(c.insider_names)} 人 {direction_label} · "
                f"{_fmt_money_kb(c.total_usd)}"
            )
            lines.append(f"    {names_preview}{more}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# ⑫b Weekly insider digest (Phase 3.5)
# ---------------------------------------------------------------------------

def format_sec_insider_digest(summary) -> str:
    """⑫b Fri 19:15 ET card — title-only aggregation per Phase 3.5
    Decision 4.

    ``summary`` is a :class:`v2.sec.insider_digest.WeeklyInsiderSummary`
    (duck-typed so this module doesn't need a runtime import — keeps
    the cross-module surface minimal). Render strategy:

    - Empty week → operator-visibility floor card, no footer caption
      (the 平静 placeholder is self-explanatory).
    - Non-empty → 总览 + 方向分布 + (optional) 异常活跃 ticker block +
      footer caption explaining the title-only granularity.
    """
    week_start = getattr(summary, "week_start", "—")
    week_end = getattr(summary, "week_end", "—")

    lines: list[str] = [
        f"<b>📥 内部人活动周报 · {html.escape(str(week_start))} → "
        f"{html.escape(str(week_end))}</b>",
        "━━━━━━━━━━━━━━━━━━━━",
    ]

    if getattr(summary, "is_quiet_week", False):
        # Quiet week — skip the Phase 3.5 口径 footer; the placeholder
        # line is self-explanatory and the caption only matters when
        # actual counts are on display.
        lines.append("<i>本周 ⑫ Form 4 推送平静（0 笔）</i>")
        return "\n".join(lines)

    # 总览
    lines.append("<b>本周总览</b>")
    lines.append(
        f"  总 push 数: <code>{summary.total_push_count}</code> · "
        f"涉及 ticker: <code>{summary.total_tickers_active}</code> 只"
    )

    # 方向分布
    lines.append("")
    lines.append("<b>方向分布</b>")
    lines.append(
        f"  📥 买入 (purchase): <code>{summary.purchase_push_count}</code> 笔"
    )
    lines.append(
        f"  📤 卖出 (sale): <code>{summary.sale_push_count}</code> 笔"
    )
    cluster_total = summary.cluster_purchase_count + summary.cluster_sale_count
    if cluster_total:
        lines.append(
            f"  🔗 集群 (cluster): <code>{cluster_total}</code> 笔"
            f" (买入 {summary.cluster_purchase_count} / "
            f"卖出 {summary.cluster_sale_count})"
        )

    # 异常活跃 ticker — top 5 (the digest's unusual_tickers list is
    # already ordered by push count desc via Counter.most_common).
    # Overflow gets a "... 另 N 只" tail so the reader knows the list
    # was truncated for display, not exhaustive.
    unusual = list(getattr(summary, "unusual_tickers", []) or [])
    if unusual:
        lines.append("")
        lines.append("<b>⚠️ 异常活跃 ticker</b> (≥3 pushes)")
        by_ticker = getattr(summary, "by_ticker", {}) or {}
        for t in unusual[:5]:
            n = by_ticker.get(t, 0)
            lines.append(f"  <b>{html.escape(t)}</b>: {n} pushes")
        if len(unusual) > 5:
            lines.append(f"  <i>... 另 {len(unusual) - 5} 只</i>")

    # Footer caption
    lines.append("")
    lines.append(
        "<i>注：基于 ⑫ push title 统计（Phase 3.5 简化口径，"
        "per-code A/M/F/G/C breakdown 见 Phase 3.5.5）</i>"
    )

    return "\n".join(lines)
