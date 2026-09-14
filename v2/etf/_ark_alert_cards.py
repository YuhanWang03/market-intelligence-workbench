"""ARK alert card formatters — Phase 5a.

Two public formatters, both pure functions with no v2.data deps so the
byte-equal tests stay sandbox-runnable:

- :func:`format_ark_alert` — single :class:`v2.etf.alerts.ArkAlert`
  rendered as one card. Four templates per action (new_position /
  liquidated / increase / decrease) with optional ``is_in_user_universe``
  badge + ``is_multi_fund`` coordination banner.
- :func:`format_ark_summary` — :class:`v2.etf.alerts.ArkScanResult`
  rendered as the overview card that closes the ⑬ run.

Implementation lives in ``v2/etf/`` (not ``v2/reporting/``) for the
same Phase 1/2/3 reason: the full v2.reporting package init pulls
matplotlib + v2.lateral → v2.data. Keeping the source-of-truth here
lets pytest collect the byte-equal pin without the production deps.

The card body is wrapped through ``v2/reporting/_ark_alert_formatters.py``
into the public ``v2.reporting`` namespace so ⑬'s cron + future bot
queries consume the same function objects.

HTML safety: every user-controllable string (ticker, company,
fund symbol) is ``html.escape``'d before render — same lint posture as
Phase 3's SEC formatters.
"""

from __future__ import annotations

import html
from collections import Counter

from v2.etf.alerts import ArkAlert, ArkScanResult


__all__ = ["format_ark_alert", "format_ark_summary"]


# ---------------------------------------------------------------------------
# Compact formatters — local copies to keep this module dependency-light.
# (Lifting v2.portfolio._bot_cards' _fmt_money would drag v2.broker via
# RiskReport schema imports — not worth the runtime coupling for two
# helpers.)
# ---------------------------------------------------------------------------

def _fmt_usd(v: float | None) -> str:
    """Compact USD: $X.XM / $X.XK / $X bare. Returns '—' for None."""
    if v is None:
        return "—"
    abs_v = abs(v)
    if abs_v >= 1_000_000:
        return f"${v / 1_000_000:.1f}M"
    if abs_v >= 1_000:
        return f"${v / 1_000:.1f}K"
    return f"${v:,.0f}"


def _fmt_shares(n: int | None) -> str:
    """Signed integer with thousand-separators. Returns '—' for None."""
    if n is None:
        return "—"
    return f"{n:,}"


def _fmt_csv_weight_pct(v: float | None) -> str:
    """Convert CSV-native pct unit (1.85 == 1.85%) to display string."""
    if v is None:
        return "—"
    return f"{v:.2f}%"


def _fmt_signed_relative_pct(v: float | None) -> str:
    """Decimal fraction (0.255 → '+25.5%'). Returns '—' for None."""
    if v is None:
        return "—"
    sign = "+" if v >= 0 else ""
    return f"{sign}{v * 100:.1f}%"


# ---------------------------------------------------------------------------
# Banners (rendered conditionally on the top of an alert card)
# ---------------------------------------------------------------------------

def _user_universe_badge(alert: ArkAlert) -> str | None:
    if alert.is_in_user_universe:
        return "🟢 持仓股 / 关注列表"
    return None


def _multi_fund_banner(alert: ArkAlert) -> str | None:
    """Banner line shown when ≥2 funds hit the same (ticker, direction)
    on the same day. The aggregation detail (which funds total $) lives
    in the summary card — individual cards just tag the coordination
    so the user knows this isn't a one-fund move."""
    if alert.is_multi_fund:
        return "🚨 <i>多 Fund 协同（详见总览）</i>"
    return None


# ---------------------------------------------------------------------------
# format_ark_alert — 4 templates by action
# ---------------------------------------------------------------------------

# Insertion order matters — format_ark_summary iterates this dict to render
# the action-distribution block. Polish 2: buys cluster (new_position +
# increase) BEFORE sells cluster (liquidated + decrease) so the visual
# grouping matches reader mental model. Individual-card lookups use the
# key directly so order doesn't affect them.
_ACTION_HEADER = {
    "new_position": ("🟢", "新建仓"),
    "increase":     ("📈", "增持"),
    "liquidated":   ("🔴", "清仓"),
    "decrease":     ("📉", "减持"),
}


def format_ark_alert(alert: ArkAlert) -> str:
    """Render one ArkAlert as a card.

    Card shape:

      <b>{emoji} ARK {action_zh} · {ticker}</b>
      Fund: <code>{fund}</code> [· today_weight metadata if applicable]
      [optional badge line: 🟢 持仓股 / 关注列表]
      [optional banner: 🚨 多 Fund 协同 (详见总览)]
      ━━━━━━━━━━━━━━━━━━━━
      {action-specific body}
    """
    emoji, action_zh = _ACTION_HEADER.get(alert.action, ("📌", str(alert.action)))
    ticker_esc = html.escape(alert.ticker)
    fund_esc = html.escape(alert.fund)

    lines: list[str] = [
        f"<b>{emoji} ARK {action_zh} · {ticker_esc}</b>",
    ]

    # Header line: fund + today's weight (when applicable) + signed relative
    # change for rebalances.
    header_extras: list[str] = [f"Fund: <code>{fund_esc}</code>"]
    if alert.action in ("new_position", "increase", "decrease"):
        if alert.today_weight is not None:
            header_extras.append(
                f"今日权重: <code>{_fmt_csv_weight_pct(alert.today_weight)}</code>"
            )
    if alert.action in ("increase", "decrease"):
        header_extras.append(
            f"({_fmt_signed_relative_pct(alert.weight_change_relative)})"
        )
    if alert.action == "liquidated":
        pass  # fund only on header for liquidation

    # Join header extras with " · " between fund / weight / pct
    # but keep the "(±X%)" tight against the preceding weight chunk.
    if len(header_extras) == 1:
        lines.append(header_extras[0])
    else:
        # First two pieces joined by " · "; trailing "(±X%)" joined by space
        head = " · ".join(
            p for p in header_extras if not p.startswith("(")
        )
        tail = " ".join(p for p in header_extras if p.startswith("("))
        lines.append(f"{head} {tail}".rstrip())

    # Polish 1: multi-fund banner FIRST (visual priority for P0 escalation
    # factor), then the user-universe badge — emoji order on mobile reads
    # as 🚨 → 🟢 → ━━━ for a coordinated held-position alert.
    banner = _multi_fund_banner(alert)
    if banner:
        lines.append(banner)
    badge = _user_universe_badge(alert)
    if badge:
        lines.append(badge)

    lines.append("━━━━━━━━━━━━━━━━━━━━")

    # Action-specific body
    if alert.action == "new_position":
        lines.append(
            f"买入: <code>{_fmt_shares(alert.shares_change)}</code> shares "
            f"· ≈ <code>{_fmt_usd(alert.market_value_usd)}</code>"
        )
        lines.append("昨日: 未持有")

    elif alert.action == "liquidated":
        lines.append(
            f"昨日权重: <code>{_fmt_csv_weight_pct(alert.yesterday_weight)}</code> "
            f"· <code>{_fmt_usd(alert.market_value_usd)}</code>"
        )
        lines.append("今日: 完全清仓")

    elif alert.action == "increase":
        lines.append(
            f"增持: <code>{_fmt_shares(alert.shares_change)}</code> shares "
            f"· ≈ <code>{_fmt_usd(alert.market_value_usd)}</code>"
        )
        lines.append(
            f"昨日权重: <code>{_fmt_csv_weight_pct(alert.yesterday_weight)}</code>"
        )

    elif alert.action == "decrease":
        # shares_change is negative for a sell — display its magnitude
        # next to the explicit "减持" label.
        magnitude = abs(alert.shares_change) if alert.shares_change else 0
        lines.append(
            f"减持: <code>{_fmt_shares(magnitude)}</code> shares "
            f"· ≈ <code>{_fmt_usd(alert.market_value_usd)}</code>"
        )
        lines.append(
            f"昨日权重: <code>{_fmt_csv_weight_pct(alert.yesterday_weight)}</code>"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# format_ark_summary — overview card
# ---------------------------------------------------------------------------

# Display order is fixed so dashboard / Telegram render deterministically.
_FUND_ORDER = ("ARKK", "ARKW", "ARKG", "ARKF", "ARKQ", "ARKX")


def format_ark_summary(result: ArkScanResult) -> str:
    """Render the overview card that closes a ⑬ ARK Alerts run.

    Empty alerts → simple "本日 ARK 调仓平静" line so the dashboard
    feed still gets the operator-visibility row. ⑬'s cron pushes this
    only when ``alerts`` is non-empty (silent days don't add cards).
    """
    scan_date = html.escape(str(result.scan_date or "—"))
    funds_scanned = list(result.funds_scanned or [])
    funds_attempted = list(result.funds_attempted or [])

    lines: list[str] = [
        f"<b>🔔 ARK 调仓总览 · {scan_date}</b>",
        "━━━━━━━━━━━━━━━━━━━━",
    ]

    if not result.alerts:
        lines.append("<i>本日 ARK 调仓平静（0 触发）</i>")
        if funds_scanned:
            funds_str = " / ".join(
                html.escape(f) for f in _sorted_funds(funds_scanned)
            )
            coverage = _coverage_str(funds_scanned, funds_attempted)
            lines.append(
                f"已扫描: <code>{funds_str}</code> {coverage}"
            )
        if result.warnings:
            lines.append("")
            lines.append(f"<i>warnings: {len(result.warnings)}</i>")
        return "\n".join(lines)

    # Aggregate alerts by action and by tier suggestion (simulated here
    # via heuristics on flags — the production priority lives in
    # v2.reporting.priority but the summary card just needs a rough
    # P0/P1 split for at-a-glance scanning).
    n_total = len(result.alerts)
    n_multi = sum(1 for a in result.alerts if a.is_multi_fund)
    n_user = sum(1 for a in result.alerts if a.is_in_user_universe)
    by_action = Counter(a.action for a in result.alerts)

    lines.append(f"<b>本日触发 alerts: {n_total}</b>")
    if n_multi:
        lines.append(f"  🚨 多 Fund 协同: <code>{n_multi}</code>")
    if n_user:
        lines.append(f"  🟢 涉及持仓 / 关注: <code>{n_user}</code>")

    # Action breakdown
    lines.append("")
    lines.append("<b>行动分布</b>")
    for action_key, (emoji, action_zh) in _ACTION_HEADER.items():
        n = by_action.get(action_key, 0)
        if n:
            lines.append(f"  {emoji} {action_zh}: <code>{n}</code>")

    # User universe tickers (sorted for stable display)
    if n_user:
        user_tickers = sorted({
            a.ticker for a in result.alerts if a.is_in_user_universe
        })
        joined = " · ".join(html.escape(t) for t in user_tickers)
        lines.append("")
        lines.append(
            f"<b>涉及 user universe:</b> {joined} "
            f"<i>({len(user_tickers)} 个)</i>"
        )

    # Multi-fund coordinated tickers (groups: ticker → set of funds)
    if n_multi:
        coord: dict[str, set[str]] = {}
        for a in result.alerts:
            if a.is_multi_fund:
                coord.setdefault(a.ticker, set()).add(a.fund)
        lines.append("")
        lines.append("<b>多 Fund 协同</b>")
        for t in sorted(coord):
            funds_str = " + ".join(
                html.escape(f) for f in _sorted_funds(coord[t])
            )
            lines.append(
                f"  <b>{html.escape(t)}</b>: {funds_str}"
            )

    # Funds covered today (always show — operator wants to know which
    # ARK funds were scanned, even on zero-alert days the empty branch
    # above already covers it). Polish 3: when funds_attempted is set
    # we render "(succeeded / attempted)" so partial failures (e.g.
    # ARKG CSV 503) are transparent.
    if funds_scanned:
        funds_str = " / ".join(html.escape(f) for f in _sorted_funds(funds_scanned))
        coverage = _coverage_str(funds_scanned, funds_attempted)
        lines.append("")
        lines.append(
            f"<b>本日扫描 funds:</b> {funds_str} {coverage}"
        )

    if result.warnings:
        lines.append("")
        lines.append(f"<i>warnings: {len(result.warnings)}</i>")

    return "\n".join(lines)


def _sorted_funds(funds) -> list[str]:
    """Return funds in _FUND_ORDER first, then any unknown funds appended
    alphabetically. Keeps display deterministic across the codebase."""
    funds = set(funds)
    out = [f for f in _FUND_ORDER if f in funds]
    out += sorted(funds - set(_FUND_ORDER))
    return out


def _coverage_str(
    funds_scanned: list[str], funds_attempted: list[str],
) -> str:
    """Render the "(N/M ARK funds)" partial-coverage fraction.

    When ``funds_attempted`` is empty (caller didn't track), fall back
    to the legacy "(N 个)" display so old call sites stay rendering.
    """
    n_scanned = len(funds_scanned)
    if not funds_attempted:
        return f"<i>({n_scanned} 个)</i>"
    n_attempted = len(funds_attempted)
    return f"<i>({n_scanned}/{n_attempted} ARK funds)</i>"
