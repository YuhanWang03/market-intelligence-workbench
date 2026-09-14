"""Earnings card formatters — pure functions, no v2.data deps.

Single source of truth since Stage 5. Re-exported through
``v2.reporting.format_earnings_*`` and ``v2.earnings._bot_cards`` so
production cron + bot code can keep using the public reporting namespace
without touching this module's import path.

The implementation lives under ``v2/earnings/`` (not ``v2/reporting/``)
because the latter package's ``__init__`` pulls in matplotlib +
v2.backtesting + v2.monitoring transitively, which would force tests to
either install the production-only ``v2.data`` package or jump through
``importlib.util`` to bypass the package init. Keeping the
implementation here lets the cards stay unit-tested end-to-end.

Two badge styles intentionally coexist:

- **Bot query path** (``/earnings TICKER``, calendar view): ⭐ chips.
  Compact, consistent with the rest of the bot's dashboard-style cards.
- **Cron push path** (reminders, post-release summary, pending fallback):
  emoji-prefixed labels (``🟢 持仓股`` / ``👁 关注列表``). More verbose
  because the user sees them once on lockscreen, not in a chat scroll.
"""

from __future__ import annotations

import html
from datetime import date as _date
from typing import Iterable

from v2.earnings.models import (
    EarningsEvent,
    EarningsHistorical,
    EarningsSummary,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_WHEN_LABEL = {
    "bmo": "盘前",
    "amc": "盘后",
    "unknown": "时间未公布",
}

_SURPRISE_EMOJI = {
    "BEAT": "🟢",
    "MISS": "🔴",
    "MEET": "🟡",
    "UNKNOWN": "❔",
}

_TAG_EMOJI = {
    "D-3": "📅",
    "D-1": "⏰",
    "D-0": "🎯",
}


def _star_badge(is_held: bool, is_watchlist: bool) -> str:
    """Bot-path badge — ⭐ chips, both possible if user is in both sets."""
    chips: list[str] = []
    if is_held:
        chips.append("⭐持仓")
    if is_watchlist:
        chips.append("⭐watchlist")
    return " ".join(chips)


def _cron_badge(is_held: bool, is_watchlist: bool) -> str:
    """Cron-path badge — verbose, held trumps watchlist."""
    if is_held:
        return "🟢 持仓股"
    if is_watchlist:
        return "👁 关注列表"
    return ""


def _fmt_money(v: float | None) -> str:
    if v is None:
        return "—"
    if abs(v) >= 1e9:
        return f"${v / 1e9:.2f}B"
    if abs(v) >= 1e6:
        return f"${v / 1e6:.1f}M"
    return f"${v:,.0f}"


def _fmt_eps(v: float | None) -> str:
    if v is None:
        return "—"
    return f"${v:.2f}"


def _fmt_pct_signed(v: float | None) -> str:
    """Empty string if None. Otherwise ' 🟢 +%' / ' 🔴 -%'."""
    if v is None:
        return ""
    return f" 🟢 +{v:.1%}" if v >= 0 else f" 🔴 {v:.1%}"


# ---------------------------------------------------------------------------
# /earnings TICKER (bot query path)
# ---------------------------------------------------------------------------

def format_earnings_view(
    ticker: str,
    *,
    next_event: EarningsEvent | None,
    last_event: EarningsHistorical | None,
    is_held: bool,
    is_watchlist: bool,
    today_iso: str | None = None,
) -> str:
    """Single-ticker earnings card for ``/earnings TICKER`` + ``earnings_view`` NL."""
    ticker = ticker.upper()
    today_iso = today_iso or _date.today().isoformat()

    lines: list[str] = [
        f"<b>📞 {html.escape(ticker)} 财报</b>",
        "━━━━━━━━━━━━━━━━━━━━",
    ]

    if next_event is not None:
        when = _WHEN_LABEL.get(next_event.when, next_event.when)
        d_minus = next_event.d_minus(today_iso)
        d_label = (
            f"D-{d_minus}" if d_minus > 0
            else "今日" if d_minus == 0
            else f"D+{-d_minus}"
        )
        lines.append(
            f"下次：<code>{next_event.release_date}</code> {when} (<b>{d_label}</b>)"
        )
        extras: list[str] = []
        if next_event.eps_estimate is not None:
            extras.append(f"预期 EPS <code>{_fmt_eps(next_event.eps_estimate)}</code>")
        if next_event.revenue_estimate is not None:
            extras.append(
                f"预期营收 <code>{_fmt_money(next_event.revenue_estimate)}</code>"
            )
        if extras:
            lines.append("  " + " · ".join(extras))
    else:
        lines.append("<i>暂未发现即将的财报安排</i>")

    badge = _star_badge(is_held, is_watchlist)
    if badge:
        lines.append(badge)

    lines.append("")

    if last_event is not None and last_event.has_quarterly_data:
        surprise_emoji = _SURPRISE_EMOJI.get(last_event.eps_surprise, "•")
        lines.append(
            f"<b>上次 ({last_event.report_period})</b> "
            f"{surprise_emoji} {last_event.eps_surprise}"
        )
        rev_pct = last_event.revenue_surprise_pct()
        eps_pct = last_event.eps_surprise_pct()
        if last_event.revenue_actual is not None and last_event.revenue_estimate is not None:
            lines.append(
                f"  💰 营收 <code>{_fmt_money(last_event.revenue_actual)}</code> "
                f"(预期 <code>{_fmt_money(last_event.revenue_estimate)}</code>)"
                f"{_fmt_pct_signed(rev_pct)}"
            )
        if last_event.eps_actual is not None and last_event.eps_estimate is not None:
            lines.append(
                f"  📊 EPS <code>{_fmt_eps(last_event.eps_actual)}</code> "
                f"(预期 <code>{_fmt_eps(last_event.eps_estimate)}</code>)"
                f"{_fmt_pct_signed(eps_pct)}"
            )
    else:
        lines.append("<i>暂无最近财报数据</i>")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# /earnings (bot calendar path)
# ---------------------------------------------------------------------------

def format_earnings_calendar(
    events: Iterable[EarningsEvent],
    *,
    horizon_days: int,
    held: set[str],
    watchlist: set[str],
    today_iso: str | None = None,
) -> str:
    """N-day forward calendar list with ⭐ chips, sorted by release date."""
    today_iso = today_iso or _date.today().isoformat()

    rows: list[tuple[str, EarningsEvent]] = []
    for ev in events:
        d_minus = ev.d_minus(today_iso)
        if 0 <= d_minus <= horizon_days:
            rows.append((ev.release_date, ev))
    rows.sort(key=lambda r: r[0])

    if not rows:
        return (
            f"<b>📅 未来 {horizon_days} 天财报日历</b>\n"
            f"<i>无 watchlist 或持仓 标的发财报</i>"
        )

    lines: list[str] = [
        f"<b>📅 未来 {horizon_days} 天财报日历 · {len(rows)} 只标的</b>",
        "━━━━━━━━━━━━━━━━━━━━",
    ]
    for _, ev in rows:
        ticker = ev.ticker.upper()
        when = _WHEN_LABEL.get(ev.when, ev.when)
        is_held = ticker in held
        is_wl = ticker in watchlist and not is_held
        badge = _star_badge(is_held, is_wl)
        line = f"<code>{ev.release_date}</code>  <b>{html.escape(ticker)}</b> {when}"
        if badge:
            line = f"{line}  {badge}"
        lines.append(line)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Reminders cron card (D-3 / D-1 / D-0)
# ---------------------------------------------------------------------------

def format_earnings_reminder(
    event: EarningsEvent,
    *,
    tag: str,
    is_held: bool,
    is_watchlist: bool,
) -> str:
    """D-3 / D-1 / D-0 reminder card pushed by the 08:00 ET cron."""
    emoji = _TAG_EMOJI[tag]
    when = _WHEN_LABEL.get(event.when, event.when)
    badge = _cron_badge(is_held, is_watchlist)

    lines: list[str] = [
        f"<b>{emoji} 财报提醒 · {event.ticker} · {tag}</b>",
        f"发布日：<code>{event.release_date}</code>（{when}）",
    ]
    if badge:
        lines.append(badge)

    extras: list[str] = []
    if event.eps_estimate is not None:
        extras.append(f"EPS 预期：<code>{event.eps_estimate:.2f}</code>")
    if event.revenue_estimate is not None:
        extras.append(
            f"营收预期：<code>${event.revenue_estimate / 1e9:.2f}B</code>"
        )
    if extras:
        lines.append("")
        lines.extend(extras)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Post-release summary cron cards
# ---------------------------------------------------------------------------

def _summary_eps_pct(s: EarningsSummary) -> float | None:
    if s.eps_actual is None or s.eps_estimate is None or s.eps_estimate == 0:
        return None
    return (s.eps_actual - s.eps_estimate) / abs(s.eps_estimate)


def _summary_revenue_pct(s: EarningsSummary) -> float | None:
    if s.revenue_actual is None or s.revenue_estimate is None or s.revenue_estimate <= 0:
        return None
    return (s.revenue_actual - s.revenue_estimate) / s.revenue_estimate


def format_earnings_summary(
    summary: EarningsSummary,
    *,
    is_held: bool,
    is_watchlist: bool,
) -> str:
    """Post-release summary card pushed by the 21:00 ET cron."""
    emoji = _SURPRISE_EMOJI.get(summary.eps_surprise, "•")
    badge = _cron_badge(is_held, is_watchlist)

    lines: list[str] = [
        f"<b>{emoji} 财报发布 · {summary.ticker} · {summary.eps_surprise}</b>",
        f"报告期：<code>{summary.report_period}</code> · "
        f"申报：<code>{summary.filing_date}</code>",
    ]
    if badge:
        lines.append(badge)
    lines.append("")

    eps_pct = _summary_eps_pct(summary)
    if summary.eps_actual is not None and summary.eps_estimate is not None:
        delta = f" ({eps_pct:+.1%})" if eps_pct is not None else ""
        lines.append(
            f"EPS：<code>{summary.eps_actual:.2f}</code> vs 预期 "
            f"<code>{summary.eps_estimate:.2f}</code>{delta}"
        )
    rev_pct = _summary_revenue_pct(summary)
    if summary.revenue_actual is not None and summary.revenue_estimate is not None:
        delta = f" ({rev_pct:+.1%})" if rev_pct is not None else ""
        lines.append(
            f"营收：<code>${summary.revenue_actual / 1e9:.2f}B</code> vs 预期 "
            f"<code>${summary.revenue_estimate / 1e9:.2f}B</code>{delta}"
        )

    if summary.last_4q_surprises:
        streak = " → ".join(summary.last_4q_surprises)
        lines.append(f"最近 4 季：<code>{streak}</code>")

    if summary.bull or summary.bear:
        lines.append("")
        if summary.bull:
            lines.append(f"👍 {summary.bull}")
        if summary.bear:
            lines.append(f"👎 {summary.bear}")
    if summary.narrative:
        lines.append("")
        lines.append(f"<i>{summary.narrative}</i>")
    if summary.transcript_url:
        lines.append("")
        lines.append(f'📜 <a href="{summary.transcript_url}">电话会记录</a>')

    # Phase 3.5 — optional 10-Q delta section. Rendered ONLY when
    # ten_q_delta is present AND has something meaningful to surface
    # (at least one added MD&A paragraph, an auditor flag, or new
    # risk factors). A bare TenQDelta with all empty fields means the
    # 10-Q was successfully fetched but the diff was uninformative
    # (first quarter after deploy / nothing changed) — silent skip
    # avoids cluttering the card with an empty header.
    tq = summary.ten_q_delta
    if tq is not None and _has_meaningful_ten_q_signal(tq):
        lines.append("")
        lines.append("<b>📋 10-Q MD&amp;A 关键变化</b>")
        for para in (getattr(tq, "mda_added_paragraphs", None) or [])[:3]:
            lines.append(f"  ➕ <i>{html.escape(str(para))}</i>")
        if getattr(tq, "has_going_concern", False):
            lines.append("  ⚠️ <b>Going concern</b> 关键词出现")
        if getattr(tq, "has_material_weakness", False):
            lines.append("  ⚠️ <b>Material weakness</b> auditor finding")
        new_rf_n = int(getattr(tq, "new_risk_factor_count", 0) or 0)
        if new_rf_n:
            lines.append(f"  📌 {new_rf_n} 个新 risk factor 段落")

    return "\n".join(lines)


def _has_meaningful_ten_q_signal(tq) -> bool:
    """True iff the TenQDelta carries at least one signal worth
    rendering. Avoids empty '📋 10-Q MD&A 关键变化' header on a
    no-diff filing."""
    if getattr(tq, "mda_added_paragraphs", None):
        return True
    if getattr(tq, "has_going_concern", False):
        return True
    if getattr(tq, "has_material_weakness", False):
        return True
    if int(getattr(tq, "new_risk_factor_count", 0) or 0) > 0:
        return True
    return False


def format_earnings_pending(
    ticker: str,
    *,
    today_iso: str,
    is_held: bool,
    is_watchlist: bool,
) -> str:
    """Placeholder card — calendar said today, FD hasn't ingested actuals yet."""
    badge = _cron_badge(is_held, is_watchlist)
    lines = [
        f"<b>⏳ 财报数据待落地 · {ticker}</b>",
        f"发布日：<code>{today_iso}</code>",
    ]
    if badge:
        lines.append(badge)
    lines.append("")
    lines.append("<i>FD 实际数据尚未入库，明天 21:00 ET 自动重试。</i>")
    return "\n".join(lines)


__all__ = [
    "format_earnings_calendar",
    "format_earnings_pending",
    "format_earnings_reminder",
    "format_earnings_summary",
    "format_earnings_view",
]
