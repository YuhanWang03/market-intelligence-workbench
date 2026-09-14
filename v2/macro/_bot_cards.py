"""Macro card formatters — pure functions, no v2.data deps.

Single source of truth since Stage 5. Re-exported through
``v2.reporting.format_macro_*`` (and ``v2.reporting.formatters``) so
production cron + bot code consume the public namespace. The
implementation lives in ``v2/macro/`` (not ``v2/reporting/``) for the
same reason as Phase 1's earnings cards, Phase 2's portfolio cards,
and Phase 3's SEC cards: ``v2/reporting/__init__.py`` transitively
pulls matplotlib + v2.lateral, which require v2.data. Keeping the
implementation here lets the byte-equal tests stay sandbox-runnable.

Six public formatters:

- :func:`format_macro_daily_snapshot` — ⑭ daily 16:30 ET cron card.
  Auto-prefixes 🚨 / 📉 / 📊 by inspecting the snapshot's anomaly
  flags (no external ``kind`` parameter — the snapshot self-classifies).
- :func:`format_macro_release_card` — ⑮ daily 09:00 ET cron card for
  CPI / PCE / NFP / GDP / PPI. Also used by the bot ``/cpi`` responder
  via the optional ``next_release_date`` kwarg.
- :func:`format_macro_fomc_card` — ⑮ FOMC-day card (statement diff +
  SEP + Tavily sell-side, no LLM verdict). Bot ``/fomc`` shares this
  via ``next_fomc_date``.
- :func:`format_macro_claims_card` — ⑯ Thursday Claims card. Bot
  ``release_check(release_type='claims')`` reuses.
- :func:`format_macro_weekly_recap` — ⑰ Friday 19:30 ET recap.
- :func:`format_macro_dashboard` — ``/macro`` bot dashboard. Same
  market + yields panels as the snapshot card plus the 14-day past /
  30-day future release calendar.

All user-supplied / LLM-supplied / external-source strings are
HTML-escaped before insertion. The HTML-safety lint
(``test_formatters_html_safe.py``) pins this on hostile fixtures.
"""

from __future__ import annotations

import html
from datetime import date

from v2.macro.models import (
    FOMCEvent, MacroRelease, MacroSnapshot,
)


__all__ = [
    "format_macro_daily_snapshot",
    "format_macro_release_card",
    "format_macro_fomc_card",
    "format_macro_claims_card",
    "format_macro_weekly_recap",
    "format_macro_dashboard",
]


# ---------------------------------------------------------------------------
# Shared formatting helpers (deterministic — no external deps)
# ---------------------------------------------------------------------------

def _fmt_pct(p: float | None, *, signed: bool = True) -> str:
    if p is None:
        return "—"
    sign = "+" if signed and p >= 0 else ""
    return f"{sign}{p:.2%}"


def _fmt_level(v: float | None, *, places: int = 2) -> str:
    if v is None:
        return "—"
    return f"{v:.{places}f}"


def _fmt_num(v: float | None, *, places: int = 2) -> str:
    if v is None:
        return "—"
    return f"{v:,.{places}f}"


def _fmt_delta(v: float | None, *, places: int = 2) -> str:
    if v is None:
        return "—"
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.{places}f}"


def _fmt_bps(v: float | None) -> str:
    """Render a yield differential (e.g. T10Y2Y = 0.21) as basis points."""
    if v is None:
        return "—"
    return f"{int(round(v * 100)):+d}bp"


def _days_between(iso_a: str, iso_b: str) -> int:
    try:
        return (date.fromisoformat(iso_b) - date.fromisoformat(iso_a)).days
    except ValueError:
        return 0


_TONE_EMOJI = {
    "hawkish": "🟥",
    "dovish":  "🟩",
    "neutral": "⚪",
}


_DELTA_LABELS = (
    ("VIXCLS", "VIX"),
    ("DGS10",  "10Y"),
    ("DGS2",   "2Y"),
    ("T10Y2Y", "10Y-2Y"),
)


# ---------------------------------------------------------------------------
# ⑭ Snapshot card (cron)
# ---------------------------------------------------------------------------

def _snapshot_icon(snap: MacroSnapshot) -> str:
    """Pick the header icon from the snapshot's anomaly flags. Highest-
    priority anomaly wins; default is the daily ambient marker."""
    if snap.vix_spike:
        return "🚨 宏观警报"
    if snap.curve_flip or snap.rates_shocked or snap.vix_elevated:
        return "📉 宏观警报"
    return "📊 宏观日终"


def format_macro_daily_snapshot(snap: MacroSnapshot) -> str:
    """Render the ⑭ daily 16:30 ET snapshot card.

    The icon prefix is derived from the snapshot's own anomaly flags
    (vix_spike → 🚨, curve_flip / rates_shocked / vix_elevated → 📉,
    else 📊). Warnings list rendered at the bottom; truncated to 6.
    """
    icon = _snapshot_icon(snap)
    lines: list[str] = [
        f"<b>{icon} · {html.escape(snap.snapshot_date)}</b>",
        "━━━━━━━━━━━━━━━━━━━━",
    ]

    # Markets
    lines.append("<b>市场</b>")
    spike_tag = ""
    if snap.vix_spike:
        spike_tag = " <b>🚨 +20%</b>"
    elif snap.vix_elevated:
        spike_tag = " ⚠️ 偏高"
    lines.append(
        f"  VIX: <code>{_fmt_level(snap.vix)}</code> "
        f"({_fmt_pct(snap.vix_pct_change_1d)}){spike_tag}"
    )
    lines.append(
        f"  DXY: <code>{_fmt_level(snap.dxy)}</code> · "
        f"WTI: <code>{_fmt_level(snap.wti_crude)}</code> · "
        f"Gold: <code>{_fmt_level(snap.gold, places=1)}</code>"
    )

    # Rates
    lines.append("")
    lines.append("<b>利率 (FRED EOD)</b>")
    ff_band = "—"
    if snap.fed_funds_upper is not None and snap.fed_funds_lower is not None:
        ff_band = f"{snap.fed_funds_lower:.2f}% – {snap.fed_funds_upper:.2f}%"
    lines.append(f"  Fed Funds: <code>{ff_band}</code>")
    lines.append(
        f"  2Y: <code>{_fmt_level(snap.dgs2)}%</code> · "
        f"10Y: <code>{_fmt_level(snap.dgs10)}%</code>"
    )

    curve_tag = ""
    if snap.curve_flip:
        curve_tag = " <b>📉 今日翻转</b>"
    elif snap.t10y2y is not None and snap.t10y2y < 0:
        curve_tag = " <i>(倒挂)</i>"
    lines.append(
        f"  10Y-2Y: <code>{_fmt_bps(snap.t10y2y)}</code>{curve_tag}"
    )

    if snap.rates_shocked:
        lines.append("  <b>⚠️ 10Y 单日 ≥ 20bps 异动</b>")

    if snap.warnings:
        lines.append("")
        lines.append("<i>⚠️ 数据不全:</i>")
        for w in snap.warnings[:6]:
            lines.append(f"  • <i>{html.escape(w)}</i>")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# ⑮ Release card (cron + /cpi etc. bot)
# ---------------------------------------------------------------------------

def format_macro_release_card(
    release: MacroRelease,
    *,
    tier: str | None = None,
    next_release_date: str | None = None,
) -> str:
    """Render a non-FOMC release card (CPI / PCE / NFP / GDP / PPI).

    Cron path passes ``tier`` (rendered in the header subtitle); bot
    path passes ``next_release_date`` (rendered as a "下次发布" suffix).
    The bot's days-until calculation uses the system clock at format
    time — deterministic for the test fixtures since they pin both
    today and ``next_release_date``.
    """
    tone_emoji = _TONE_EMOJI.get(release.tone or "neutral", "⚪")

    header_subtitle = f"发布日：<code>{html.escape(release.release_date)}</code>"
    if tier:
        header_subtitle += f" · 评级：<b>{html.escape(tier)}</b>"

    lines: list[str] = [
        f"<b>📅 {html.escape(release.release_type)} · "
        f"{html.escape(release.period)}</b>",
        header_subtitle,
        "━━━━━━━━━━━━━━━━━━━━",
    ]

    # Numeric panel
    if release.mom_pct is not None:
        lines.append(f"  MoM: <code>{_fmt_pct(release.mom_pct)}</code>")
    if release.yoy_pct is not None:
        lines.append(f"  YoY: <code>{_fmt_pct(release.yoy_pct)}</code>")
    if release.headline is not None:
        lines.append(f"  Headline: <code>{_fmt_num(release.headline)}</code>")
    if release.core is not None:
        lines.append(f"  Core: <code>{_fmt_num(release.core)}</code>")

    if release.consensus is not None:
        # Match the consensus format to whatever change metric the
        # release reports. CPI / PCE / PPI / GDP carry mom_pct so
        # consensus is also a fraction (render as pct). NFP uses
        # mom_change_k so consensus is an absolute count (render
        # as a number). This keeps the consensus line readable on
        # the same scale as the MoM/YoY rows above it.
        if release.mom_pct is not None:
            consensus_str = _fmt_pct(release.consensus)
        else:
            consensus_str = _fmt_num(release.consensus)
        sigma_str = ""
        if release.surprise_sigma is not None:
            sigma_str = (
                f" ({release.surprise_sigma:+.1f}σ, "
                f"{html.escape(release.surprise_label)})"
            )
        lines.append(
            f"  Consensus: <code>{consensus_str}</code>{sigma_str}"
        )

    lines.append(f"  3M 趋势: <i>{html.escape(release.trailing_3mo_trend)}</i>")

    # LLM qualitative panel (Layer 1+2 sanitized upstream; we still escape)
    if release.narrative:
        lines.append("")
        lines.append(
            f"<b>{tone_emoji} 解读</b> "
            f"<i>({html.escape(release.tone or 'neutral')})</i>"
        )
        lines.append(f"  {html.escape(release.narrative)}")
    if release.bull_takeaway:
        lines.append(f"  🟢 {html.escape(release.bull_takeaway)}")
    if release.bear_takeaway:
        lines.append(f"  🔴 {html.escape(release.bear_takeaway)}")

    if next_release_date:
        days_out = _days_between(date.today().isoformat(), next_release_date)
        suffix = "今天" if days_out == 0 else f"{days_out} 天后"
        lines.append("")
        lines.append(
            f"📅 下次发布: <code>{html.escape(next_release_date)}</code> "
            f"({suffix})"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# ⑮ FOMC card (cron + /fomc bot) — Layer 3 defense
# ---------------------------------------------------------------------------

def format_macro_fomc_card(
    event: FOMCEvent,
    *,
    tier: str | None = None,
    next_fomc_date: str | None = None,
) -> str:
    """Render an FOMC card.

    Layer 3 hallucination defense: the body is built from Python
    statement diff + SEP dot-plot + Tavily majority vote. No LLM
    verdict appears in the output; even ``sell_side_sentiment`` is a
    counted majority vote, not a model judgment.

    Cron path passes ``tier`` (rendered in the header); bot path
    passes ``next_fomc_date``.
    """
    header_subtitle = (
        f"会议日：<code>{html.escape(event.meeting_date)}</code>"
    )
    if tier:
        header_subtitle += f" · 评级：<b>{html.escape(tier)}</b>"

    lines: list[str] = [
        f"<b>🏛️ FOMC · {html.escape(event.meeting_date)}</b>",
        header_subtitle,
        "━━━━━━━━━━━━━━━━━━━━",
    ]

    diff = event.statement_diff or {}
    added = diff.get("added_phrases") or []
    removed = diff.get("removed_phrases") or []

    if added:
        lines.append("<b>📌 Statement 新增措辞</b>")
        for p in added[:6]:
            lines.append(f"  ➕ <i>{html.escape(str(p))}</i>")
    if removed:
        lines.append("<b>📌 Statement 移除措辞</b>")
        for p in removed[:6]:
            lines.append(f"  ➖ <i>{html.escape(str(p))}</i>")
    if not added and not removed:
        lines.append("<b>📌 Statement</b> <i>(关键措辞无变动)</i>")

    if event.has_sep:
        lines.append("")
        lines.append(
            f"<b>📊 SEP Dot Plot</b> "
            f"<i>({html.escape(event.sep_dot_plot_change)})</i>"
        )
        if event.sep_median_dots:
            for k, v in event.sep_median_dots.items():
                lines.append(
                    f"  {html.escape(str(k))}: <code>{v:.2f}%</code>"
                )
        else:
            lines.append("  <i>(数据待解析)</i>")

    if event.sell_side_sentiment:
        lines.append("")
        lines.append(
            f"<b>📰 卖方读数</b>: "
            f"<i>{html.escape(event.sell_side_sentiment)}</i> "
            f"<i>(Tavily majority vote)</i>"
        )
        if event.sell_side_sources:
            sources_short = ", ".join(
                html.escape(str(s)) for s in event.sell_side_sources[:4]
            )
            lines.append(f"  来源: {sources_short}")

    if next_fomc_date:
        days_out = _days_between(date.today().isoformat(), next_fomc_date)
        suffix = "今天" if days_out == 0 else f"{days_out} 天后"
        lines.append("")
        lines.append(
            f"📅 下次 FOMC: <code>{html.escape(next_fomc_date)}</code> "
            f"({suffix})"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# ⑯ Claims card (cron)
# ---------------------------------------------------------------------------

def format_macro_claims_card(
    release: MacroRelease,
    *,
    tier: str | None = None,
    next_release_date: str | None = None,
) -> str:
    """Render the Thursday Initial Jobless Claims card.

    Differs from the generic release card in that ``release.core`` is
    the 4-week MA smoothed level (operators' preferred figure) and
    ``release.prior_value`` is shown explicitly. No mom/yoy panel —
    ICSA is reported as a weekly count, not a rate.
    """
    tone_emoji = _TONE_EMOJI.get(release.tone or "neutral", "⚪")

    header_subtitle = f"发布日：<code>{html.escape(release.release_date)}</code>"
    if tier:
        header_subtitle += f" · 评级：<b>{html.escape(tier)}</b>"

    lines: list[str] = [
        f"<b>📅 Initial Claims · {html.escape(release.period)}</b>",
        header_subtitle,
        "━━━━━━━━━━━━━━━━━━━━",
        f"  本周: <code>{_fmt_num(release.headline, places=0)}</code>",
        f"  4W MA: <code>{_fmt_num(release.core, places=0)}</code> "
        "<i>(smoothed)</i>",
        f"  上周值: <code>{_fmt_num(release.prior_value, places=0)}</code>",
        f"  3M 趋势: <i>{html.escape(release.trailing_3mo_trend)}</i>",
    ]

    if release.consensus is not None and release.surprise_sigma is not None:
        lines.append(
            f"  Consensus: <code>{_fmt_num(release.consensus, places=0)}</code>"
            f" ({release.surprise_sigma:+.1f}σ, "
            f"{html.escape(release.surprise_label)})"
        )

    if release.narrative:
        lines.append("")
        lines.append(
            f"<b>{tone_emoji} 解读</b> "
            f"<i>({html.escape(release.tone or 'neutral')})</i>"
        )
        lines.append(f"  {html.escape(release.narrative)}")
    if release.bull_takeaway:
        lines.append(f"  🟢 {html.escape(release.bull_takeaway)}")
    if release.bear_takeaway:
        lines.append(f"  🔴 {html.escape(release.bear_takeaway)}")

    if next_release_date:
        days_out = _days_between(date.today().isoformat(), next_release_date)
        suffix = "今天" if days_out == 0 else f"{days_out} 天后"
        lines.append("")
        lines.append(
            f"📅 下周 Claims: <code>{html.escape(next_release_date)}</code> "
            f"({suffix})"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# ⑰ Weekly recap card (cron)
# ---------------------------------------------------------------------------

def format_macro_weekly_recap(recap: dict) -> str:
    """Render the Fri 19:30 ET weekly recap.

    The ``recap`` dict shape is the one produced by
    :func:`v2.macro.pipeline.build_weekly_recap`:
    ``{week_start, week_end, weekly_deltas, this_week_releases,
       next_week_releases}``.
    """
    week_start = recap.get("week_start", "—")
    week_end = recap.get("week_end", "—")

    lines: list[str] = [
        f"<b>📊 宏观周报 · {html.escape(str(week_start))} → "
        f"{html.escape(str(week_end))}</b>",
        "━━━━━━━━━━━━━━━━━━━━",
    ]

    deltas = recap.get("weekly_deltas") or {}
    lines.append("<b>本周变化</b>")
    for sid, label in _DELTA_LABELS:
        d = deltas.get(sid)
        # VIX is an absolute index level — week-over-week change is
        # in "pts". Treasury yields (DGS10 / DGS2 / T10Y2Y) come as
        # fractions in pct units, so weekly change renders as bps —
        # the industry-standard unit and consistent with the ⑭
        # snapshot + /macro dashboard spreads.
        if sid == "VIXCLS":
            rendered = f"{_fmt_delta(d)} pts" if d is not None else "—"
        else:
            rendered = _fmt_bps(d)
        lines.append(f"  {label}: <code>{rendered}</code>")

    this_week = recap.get("this_week_releases") or {}
    lines.append("")
    if this_week:
        lines.append("<b>本周已发布</b>")
        for iso in sorted(this_week.keys()):
            entries = this_week[iso]
            labels = " / ".join(
                html.escape(str(rt)) for rt, _, _ in entries
            )
            lines.append(f"  <code>{html.escape(iso)}</code>: {labels}")
    else:
        lines.append("<i>本周无 release 触发</i>")

    next_week = recap.get("next_week_releases") or {}
    lines.append("")
    if next_week:
        lines.append("<b>下周预告</b>")
        for iso in sorted(next_week.keys()):
            entries = next_week[iso]
            labels = " / ".join(
                html.escape(str(rt)) for rt, _, _ in entries
            )
            lines.append(f"  <code>{html.escape(iso)}</code>: {labels}")
    else:
        lines.append("<i>下周无重大 release</i>")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# /macro dashboard (bot)
# ---------------------------------------------------------------------------

def format_macro_dashboard(
    snap: MacroSnapshot,
    calendar_window: dict[str, list[tuple[str, str, str]]],
    today_iso: str,
) -> str:
    """Render the ``/macro`` bot dashboard.

    Layout: header → 市场状态 → 收益率 → 最近 release (past) →
    下次 release (upcoming) → warnings list.

    ``calendar_window`` is the dict returned by
    :func:`release_calendar.get_releases_in_window`: ``{ISO: [(rel_type,
    label, source), ...]}``. The dashboard splits dates < today_iso vs
    >= today_iso and shows the 3 most recent past + 5 nearest future.
    """
    lines: list[str] = [
        f"<b>🌐 宏观 dashboard · {html.escape(today_iso)}</b>",
        "━━━━━━━━━━━━━━━━━━━━",
    ]

    # Market block
    lines.append("<b>📊 市场状态</b>")
    spike_tag = ""
    if snap.vix_spike:
        spike_tag = " <b>🚨 +20%</b>"
    elif snap.vix_elevated:
        spike_tag = " ⚠️"
    lines.append(
        f"  VIX: <code>{_fmt_num(snap.vix)}</code> "
        f"({_fmt_pct(snap.vix_pct_change_1d)}){spike_tag}"
    )
    lines.append(
        f"  DXY: <code>{_fmt_num(snap.dxy)}</code> · "
        f"WTI: <code>{_fmt_num(snap.wti_crude)}</code> · "
        f"Gold: <code>{_fmt_num(snap.gold, places=1)}</code>"
    )

    # Yields block
    lines.append("")
    lines.append("<b>🏛 收益率</b>")
    ff_band = "—"
    if snap.fed_funds_upper is not None and snap.fed_funds_lower is not None:
        ff_band = f"{snap.fed_funds_lower:.2f}% – {snap.fed_funds_upper:.2f}%"
    lines.append(f"  Fed Funds: <code>{ff_band}</code>")
    lines.append(
        f"  2Y: <code>{_fmt_num(snap.dgs2)}%</code> · "
        f"10Y: <code>{_fmt_num(snap.dgs10)}%</code>"
    )
    curve_tag = ""
    if snap.curve_flip:
        curve_tag = " <b>📉 今日翻转</b>"
    elif snap.t10y2y is not None and snap.t10y2y < 0:
        curve_tag = " <i>(倒挂)</i>"
    lines.append(
        f"  10-2 spread: <code>{_fmt_bps(snap.t10y2y)}</code>{curve_tag}"
    )

    sorted_dates = sorted(calendar_window.keys())
    past = [d for d in sorted_dates if d < today_iso]
    upcoming = [d for d in sorted_dates if d >= today_iso]

    if past:
        lines.append("")
        lines.append("<b>📅 最近 release</b>")
        for d in past[-3:]:
            for rel_type, _label, _src in calendar_window[d][:1]:
                lines.append(
                    f"  <code>{html.escape(d)}</code> · "
                    f"{html.escape(rel_type)}"
                )

    if upcoming:
        lines.append("")
        lines.append("<b>📅 下次 release</b>")
        for d in upcoming[:5]:
            entries = calendar_window[d]
            type_str = " / ".join(
                html.escape(rt) for rt, _, _ in entries
            )
            days_out = _days_between(today_iso, d)
            days_str = "今天" if days_out == 0 else f"{days_out} 天后"
            lines.append(
                f"  <code>{html.escape(d)}</code> ({days_str}) · {type_str}"
            )

    if snap.warnings:
        lines.append("")
        lines.append("<i>⚠️ 数据不全:</i>")
        for w in snap.warnings[:4]:
            lines.append(f"  • <i>{html.escape(w)}</i>")

    return "\n".join(lines)
