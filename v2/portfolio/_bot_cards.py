"""Portfolio card formatters — pure functions, no v2.data deps.

Single source of truth since Stage 5. Re-exported through
``v2.reporting.format_portfolio_*`` (and ``v2.reporting.formatters``) so
production cron + bot code consume the public namespace. The
implementation lives in ``v2/portfolio/`` (not ``v2/reporting/``) for
the same reason as Phase 1's earnings cards: ``v2/reporting/__init__.py``
transitively pulls matplotlib + v2.lateral, which require v2.data.
Keeping the implementation here lets the byte-equal tests stay
sandbox-runnable.

Four public formatters:

- :func:`format_risk_card` — ⑨ daily 18:30 ET cron card (full risk panel).
- :func:`format_risk_view` — ``/risk`` bot card. Byte-equal alias of
  ``format_risk_card`` — same visual surface, no priority chip
  (priority is a notifier-layer concern, not a renderer concern).
- :func:`format_weekly_card` — ⑩ Fri 19:00 ET cron card (weekly recap,
  no earnings-risk section, includes drawdown breakdown).
- :func:`format_pnl_period` — ``/pnl week|month`` bot card. Single-period
  summary; ``/pnl day`` falls through to ``format_pnl`` (the pre-existing
  daily formatter in v2.reporting).
"""

from __future__ import annotations

import html

from v2.portfolio.models import PnLMetrics, RiskReport
from v2.universe import BROAD_BUCKET, BROAD_MARKET_ETFS


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_SECTOR_NAME = {
    "SMH":   "半导体",
    "XLK":   "科技",
    "XLF":   "金融",
    "XLV":   "医药",
    "XLP":   "消费 staples",
    "XLE":   "能源",
    "XLC":   "通信",
    "XLI":   "工业",
    "KWEB":  "中概",
    "SPY":   "大盘",
    "BROAD": "大盘ETF",
    "OTHER": "其他",
}


def _fmt_money(v: float | None) -> str:
    """Compact USD string. Returns bare value (no <code>) — callers
    wrap with <code>...</code> as appropriate."""
    if v is None:
        return "—"
    if abs(v) >= 1_000_000:
        return f"${v / 1_000_000:.1f}M"
    if abs(v) >= 1_000:
        return f"${v / 1_000:.1f}K"
    return f"${v:,.0f}"


def _fmt_signed_pct(v: float | None) -> str:
    """Daily P&L percent with emoji prefix. None → '数据不足'."""
    if v is None:
        return "数据不足"
    if v > 0:
        return f"🟢 +{v:.2%}"
    if v < 0:
        return f"🔴 {v:.2%}"
    return f"🟡 {v:.2%}"


def _fmt_drawdown(magnitude: float | None) -> str:
    """Drawdown is always a loss → always prepend '-' (or '0.00%' for
    zero / new ATH). Input is the non-negative magnitude per the Stage-5
    convention in :mod:`v2.portfolio.drawdown`."""
    if magnitude is None:
        return "数据不足"
    if magnitude <= 0.0:
        return "0.00%"
    return f"-{magnitude:.2%}"


def _hhi_label(hhi: float) -> str:
    if hhi >= 0.25:
        return "高度集中"
    if hhi >= 0.15:
        return "中等集中"
    if hhi >= 0.08:
        return "适度分散"
    return "高度分散"


# ---------------------------------------------------------------------------
# ⑨ Daily risk card  +  /risk bot card (byte-equal)
# ---------------------------------------------------------------------------

def format_risk_card(report: RiskReport) -> str:
    """⑨ cron + ``/risk`` bot card. Single render path; priority is added
    by the notifier layer downstream, not embedded in this string."""
    lines: list[str] = [
        f"<b>💼 组合风险 · {report.snapshot_date}</b>",
        "━━━━━━━━━━━━━━━━━━━━",
    ]

    # ---- Account header (Stage 2.5 layout A) ----
    if report.portfolio_value > 0:
        lines.append(
            f"组合价值 <code>{_fmt_money(report.portfolio_value)}</code> "
            f"(持仓 <code>{_fmt_money(report.invested_value)}</code> · "
            f"现金 <code>{_fmt_money(report.cash)}</code>, "
            f"{report.cash_pct:.1%})"
        )
    else:
        lines.append("<i>账户暂无数据</i>")

    # ---- P&L block ----
    pnl = report.pnl
    if pnl.daily_pnl_pct is not None and pnl.daily_pnl is not None:
        sign = "+" if pnl.daily_pnl >= 0 else "-"
        lines.append(
            f"今日 P/L {_fmt_signed_pct(pnl.daily_pnl_pct)} "
            f"({sign}<code>{_fmt_money(abs(pnl.daily_pnl))}</code>)"
        )
    if pnl.weekly_pnl_pct is not None or pnl.monthly_pnl_pct is not None:
        wk = (f"本周 {pnl.weekly_pnl_pct:+.2%}"
              if pnl.weekly_pnl_pct is not None else "本周 数据不足")
        mo = (f"本月 {pnl.monthly_pnl_pct:+.2%}"
              if pnl.monthly_pnl_pct is not None else "本月 数据不足")
        lines.append(f"{wk} · {mo}")

    # ---- Concentration ----
    if report.positions:
        c = report.concentration
        top_ticker = report.positions[0].ticker
        warn_top1 = " ⚠️" if c.top_1_pct >= 0.20 else ""
        lines.append("<b>📊 集中度</b>")
        lines.append(
            f"  Top 1: <b>{html.escape(top_ticker)}</b> "
            f"{c.top_1_pct:.1%}{warn_top1}"
        )
        lines.append(f"  Top 5: {c.top_5_pct:.1%}")
        lines.append(f"  HHI: {c.hhi:.2f} ({_hhi_label(c.hhi)})")

        # ---- Exposure ----
        e = report.exposure
        if e.by_sector:
            lines.append("<b>🏭 行业暴露</b>")
            sorted_buckets = sorted(
                e.by_sector.items(), key=lambda kv: kv[1], reverse=True,
            )
            for etf, w in sorted_buckets[:4]:
                name = _SECTOR_NAME.get(etf, etf)
                warn = " ⚠️" if w >= 0.30 else ""
                lines.append(f"  {etf} ({name}): {w:.1%}{warn}")
            if len(sorted_buckets) > 4:
                rest = sum(w for _, w in sorted_buckets[4:])
                lines.append(f"  其余: {rest:.1%}")

    # ---- Drawdown ----
    dd = report.drawdown
    if dd.max_drawdown_pct is not None:
        if dd.peak_value is not None and dd.peak_date is not None:
            peak = (
                f" (峰值 <code>{_fmt_money(dd.peak_value)}</code> @ "
                f"{dd.peak_date})"
            )
        else:
            peak = ""
        lines.append(
            f"<b>📉 回撤 (1M)</b> 当前 {_fmt_drawdown(dd.current_drawdown_pct)} · "
            f"最大 {_fmt_drawdown(dd.max_drawdown_pct)}{peak}"
        )

    # ---- Earnings risk ----
    if report.earnings_risk_next_7d:
        lines.append(
            f"<b>📅 未来 7 天财报风险</b> ({len(report.earnings_risk_next_7d)} 只)"
        )
        for item in report.earnings_risk_next_7d[:5]:
            tag = f"D-{item.days_until}" if item.days_until > 0 else "今日"
            lines.append(
                f"  <code>{item.release_date}</code> "
                f"<b>{html.escape(item.ticker)}</b> ({tag})"
            )

    # ---- Alert footer ----
    alerts = _build_alerts(report)
    if alerts:
        lines.append("⚠️ <i>" + " / ".join(alerts) + "</i>")
    if _is_broad_concentration(report):
        # Phase 2.5-mini clarifier: broad-market ETFs aren't single-name risk.
        lines.append(
            "<i>注：BROAD ETF 内部已分散，集中度风险不等同于单股</i>"
        )

    # ---- Data-quality warnings ----
    if report.warnings:
        lines.append("<i>⚠ 数据不全：</i>")
        for w in report.warnings[:3]:
            lines.append(f"<i>  • {w[:80]}</i>")

    return "\n".join(lines)


def format_risk_view(report: RiskReport) -> str:
    """``/risk`` bot card — byte-equal alias of :func:`format_risk_card`.

    Both surfaces render the same body. The priority chip (🚨🚨🚨 etc.)
    is added by ``TelegramNotifier`` based on the ``priority=`` kwarg,
    NOT embedded in the formatted string. The bot path passes no
    priority kwarg, so its message has no chip — same body, different
    notifier behavior.
    """
    return format_risk_card(report)


def _is_broad_concentration(report: RiskReport) -> bool:
    """True iff a concentration alert is driven by broad-market ETF exposure.

    Used to print a Phase-2.5-mini clarifier ("IVV 内部已分散，集中度风险不
    等同于单股") below the alert footer. The user still sees the raw alert
    — we only qualify it, never suppress.
    """
    if report.positions:
        top_ticker = report.positions[0].ticker.upper()
        if (report.concentration.top_1_pct >= 0.20
                and top_ticker in BROAD_MARKET_ETFS):
            return True
    if (report.exposure.largest_sector == BROAD_BUCKET
            and report.exposure.largest_sector_pct >= 0.30):
        return True
    return False


def _build_alerts(report: RiskReport) -> list[str]:
    """Bottom-of-card alert strings."""
    out: list[str] = []
    if report.positions:
        c = report.concentration
        if c.top_1_pct >= 0.30:
            out.append(f"单票 {report.positions[0].ticker} > 30%")
        elif c.top_1_pct >= 0.20:
            out.append(f"单票 {report.positions[0].ticker} > 20%")
    if report.exposure.largest_sector_pct >= 0.30:
        out.append(f"{report.exposure.largest_sector} 行业 > 30%")
    if (report.drawdown.max_drawdown_pct is not None
            and report.drawdown.max_drawdown_pct >= 0.10):
        # Stage 5: max_drawdown_pct is non-negative magnitude, so the
        # threshold check compares directly without abs().
        out.append("1M 回撤 > 10%")
    if (report.pnl.daily_pnl_pct is not None
            and report.pnl.daily_pnl_pct <= -0.05):
        out.append("单日亏损 ≥ 5%")
    if len(report.earnings_risk_next_7d) >= 3:
        out.append(f"未来 7 天 {len(report.earnings_risk_next_7d)} 只财报")
    return out


# ---------------------------------------------------------------------------
# ⑩ Weekly recap card
# ---------------------------------------------------------------------------

def format_weekly_card(
    report: RiskReport,
    attribution: list | None = None,
    snapshot_days_available: int = 0,
) -> str:
    """⑩ Fri 19:00 ET cron card.

    Focuses on portfolio-level weekly numbers (no earnings-risk section,
    no alert footer — the daily ⑨ already surfaces those).

    Phase 2.5 full per-position attribution:

    - ``attribution`` is the list of :class:`AttributionItem` from
      :func:`v2.portfolio.snapshot.compute_weekly_attribution` —
      already sorted by contribution descending. The card surfaces
      best / worst / net.
    - ``snapshot_days_available`` is how many days of positions
      snapshots fed the attribution (cron only writes Mon-Fri so a
      full week = 5). Below
      :data:`v2.portfolio.snapshot.WEEKLY_MIN_DAYS` (= 5) we render
      "归因数据累积中 (N/5 天)" instead of the best/worst rows.
    - When ``snapshot_days_available == 0`` the attribution block is
      silently omitted — fresh-account state, nothing to apologise for.
    """
    today = report.snapshot_date
    pnl = report.pnl
    dd = report.drawdown

    lines: list[str] = [
        f"<b>📊 周 P&amp;L 复盘 · {today}</b>",
        "<i>(截至昨日收盘的口径)</i>",
        "━━━━━━━━━━━━━━━━━━━━",
    ]

    if report.portfolio_value > 0:
        lines.append(
            f"组合价值 <code>{_fmt_money(report.portfolio_value)}</code> "
            f"(持仓 <code>{_fmt_money(report.invested_value)}</code> · "
            f"现金 <code>{_fmt_money(report.cash)}</code>, "
            f"{report.cash_pct:.1%})"
        )

    lines.append(f"<b>本周回报</b> {_fmt_signed_pct(pnl.weekly_pnl_pct)}")
    lines.append(f"<b>本月回报</b> {_fmt_signed_pct(pnl.monthly_pnl_pct)}")

    if dd.max_drawdown_pct is not None:
        lines.append(f"<b>📉 1M 最大回撤</b> {_fmt_drawdown(dd.max_drawdown_pct)}")
        if dd.peak_value is not None and dd.peak_date is not None:
            lines.append(
                f"  峰值 <code>{_fmt_money(dd.peak_value)}</code> @ "
                f"<code>{dd.peak_date}</code>"
            )
        if dd.current_drawdown_pct is not None:
            lines.append(
                f"  当前距峰 {_fmt_drawdown(dd.current_drawdown_pct)}"
            )

    if report.exposure.by_sector:
        lines.append("<b>🏭 主要行业暴露</b>")
        sorted_buckets = sorted(
            report.exposure.by_sector.items(),
            key=lambda kv: kv[1], reverse=True,
        )
        for etf, w in sorted_buckets[:3]:
            lines.append(f"  {etf}: {w:.1%}")

    if report.earnings_risk_next_7d:
        lines.append(
            f"<b>📅 下周财报：</b>{len(report.earnings_risk_next_7d)} 只"
        )
        for item in report.earnings_risk_next_7d[:5]:
            lines.append(
                f"  <code>{item.release_date}</code> "
                f"<b>{html.escape(item.ticker)}</b>"
            )

    # Phase 2.5 full — per-position weekly attribution. Sourced from
    # ⑨b's positions_snapshot table via compute_weekly_attribution.
    if attribution and snapshot_days_available >= 5:
        lines.append("<b>📊 本周 per-position 表现归因</b>")
        best = attribution[0]
        worst = attribution[-1]
        lines.append(
            f"  最佳: <b>{html.escape(best.ticker)}</b> "
            f"{_fmt_signed_pct(best.weekly_return)} "
            f"(贡献 {_fmt_signed_pct(best.contribution)})"
        )
        # Only render "最差" when it's a distinct entry — single-position
        # weeks would otherwise repeat the same row.
        if worst.ticker != best.ticker:
            lines.append(
                f"  最差: <b>{html.escape(worst.ticker)}</b> "
                f"{_fmt_signed_pct(worst.weekly_return)} "
                f"(贡献 {_fmt_signed_pct(worst.contribution)})"
            )
        net = sum(a.contribution for a in attribution)
        lines.append(f"  净贡献: <code>{_fmt_signed_pct(net)}</code>")
    elif snapshot_days_available > 0:
        lines.append(
            f"<i>归因数据累积中 ({snapshot_days_available}/5 天)</i>"
        )
    # else: snapshot_days_available == 0 → silent (fresh account)

    if report.warnings:
        lines.append("<i>⚠ 数据不全：</i>")
        for w in report.warnings[:3]:
            lines.append(f"<i>  • {w[:80]}</i>")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# /pnl [week | month]  (day path uses format_pnl from v2.reporting)
# ---------------------------------------------------------------------------

_PNL_PERIOD_LABEL = {
    "week":  "本周",
    "month": "本月",
}

_PNL_PERIOD_MIN_DAYS = {
    "week":  "5 个交易日",
    "month": "21 个交易日",
}


def format_pnl_period(period: str, metrics: PnLMetrics) -> str:
    """``/pnl week`` or ``/pnl month`` summary card.

    The ``day`` case is intentionally NOT handled here — it routes to
    the pre-existing ``v2.reporting.format_pnl`` so the original byte-
    equal /pnl behavior is preserved. Passing ``period="day"`` here
    raises ValueError to surface the routing mistake.
    """
    if period not in _PNL_PERIOD_LABEL:
        raise ValueError(
            f"format_pnl_period expects 'week' or 'month', got {period!r}; "
            "'day' uses v2.reporting.format_pnl"
        )

    label = _PNL_PERIOD_LABEL[period]
    value = metrics.weekly_pnl_pct if period == "week" else metrics.monthly_pnl_pct

    lines: list[str] = [
        f"<b>📊 {label} P&amp;L 摘要</b>",
        "━━━━━━━━━━━━━━━━━━━━",
    ]

    if value is None:
        lines.append(
            f"<i>数据不足（账户历史少于 {_PNL_PERIOD_MIN_DAYS[period]}）</i>"
        )
    else:
        sign = "🟢" if value >= 0 else "🔴"
        lines.append(f"{label}回报：{sign} <code>{value:+.2%}</code>")

    if metrics.daily_pnl_pct is not None:
        sign_d = "🟢" if metrics.daily_pnl_pct >= 0 else "🔴"
        lines.append(
            f"<i>(参考 · 今日 {sign_d} {metrics.daily_pnl_pct:+.2%})</i>"
        )

    return "\n".join(lines)


__all__ = [
    "format_pnl_period",
    "format_risk_card",
    "format_risk_view",
    "format_weekly_card",
]
