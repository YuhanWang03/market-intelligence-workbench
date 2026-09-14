"""Portfolio weekly review — Fri 19:00 ET.

The tenth scheduled agent. Renders a weekly recap focusing on
portfolio-level numbers (weekly return, drawdown trajectory, sector
exposure drift) plus a small matplotlib equity-curve chart showing
the 1-month peak and current position.

Per-position weekly attribution ("best stock / worst stock this week")
is intentionally skipped in v1: Alpaca doesn't expose per-position
equity history, and faking it from ``unrealized_pl_pct`` (since entry,
not since week start) would be misleading. Full attribution would need
a daily ``positions_snapshot`` table — deferred to Phase 2.5 if asked.

Always P1 — weekly recap is operator-visible regardless of whether any
single number trips a P0 alert. (For mid-week P0 risks, the daily
⑨ Portfolio Risk cron at 18:30 ET fires.)

Card formatter lives in :func:`v2.reporting.format_portfolio_weekly_card`
(Stage 5 lift). The equity-curve chart rendering stays here — it's a
photo-side concern with matplotlib, not a text formatter.
"""

from __future__ import annotations

import io
import logging
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

from v2.archive import Archive
from v2.broker import AlpacaUnavailable, get_portfolio_history
from v2.observability import capture_trace_with_framing, install_all
from v2.portfolio import build_risk_report
from v2.portfolio.snapshot import (
    compute_weekly_attribution,
    read_weekly_window,
)
from v2.reporting import (
    TelegramNotifier,
    format_portfolio_weekly_card,
    notify_on_error,
)
from v2.reporting.priority import compute_importance


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


_TZ_ET = ZoneInfo("US/Eastern")


# ---------------------------------------------------------------------------
# Equity curve chart (photo-side, stays in the cron script)
# ---------------------------------------------------------------------------

def _render_equity_chart(title: str) -> bytes | None:
    """Render a 1-month equity curve PNG. Returns None if history is
    unavailable — caller falls back to text-only push."""
    try:
        history = get_portfolio_history(period="1M", timeframe="1D")
    except AlpacaUnavailable as exc:
        logger.info("equity chart skipped (Alpaca unavailable: %s)", exc)
        return None
    except Exception as exc:
        logger.warning("equity chart fetch failed: %s", exc)
        return None

    equity = history.get("equity") or []
    if len(equity) < 2:
        return None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    timestamps = history.get("timestamp") or list(range(len(equity)))
    dates = [datetime.fromtimestamp(int(t)) for t in timestamps[:len(equity)]]

    peak_idx = max(range(len(equity)), key=lambda i: equity[i])

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(dates, equity, color="#1f77b4", linewidth=2)
    ax.fill_between(dates, equity, min(equity), alpha=0.15, color="#1f77b4")

    ax.scatter([dates[peak_idx]], [equity[peak_idx]],
               color="#d62728", s=60, zorder=5)
    ax.annotate(
        f"峰值 ${equity[peak_idx]:,.0f}",
        xy=(dates[peak_idx], equity[peak_idx]),
        xytext=(8, 8), textcoords="offset points",
        fontsize=9, color="#d62728",
    )

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel("Equity (USD)")
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@notify_on_error("Portfolio Weekly")
def main() -> int:
    load_dotenv()
    install_all()

    archive = Archive("portfolio")
    today_iso = datetime.now(_TZ_ET).date().isoformat()

    with capture_trace_with_framing(
        agent="portfolio", intent="portfolio_risk_view",
        text=f"(自动推送) 组合周报 · {today_iso}",
        responder_name="_r_portfolio_weekly",
    ) as trace:
        report = build_risk_report(today_iso=today_iso)
        trace.emit("chat_message", role="bot",
                   text=f"组合周报 · {len(report.positions)} 持仓")

    # Weekly recap is always P1 — operator visibility. The cron passes
    # EMPTY metadata to compute_importance by design: a clean week
    # (top_1 < 20%, daily_pnl > -2%, max_dd < 10%) would otherwise land
    # P2 and skip the operator's Telegram. The floor below lifts that
    # natural P2 to P1 with an explicit '+10_weekly_recap_floor' reason
    # in the trace so the elevation is auditable. If a future redesign
    # wants the floor to be inert on signal-rich weeks, pass the report
    # metadata here AND keep the floor branch — it'll only fire when
    # truly empty.
    priority = compute_importance("portfolio_risk", {})
    if priority.tier == "P2":
        from v2.reporting.priority import PriorityResult
        priority = PriorityResult(
            score=65, tier="P1",
            reasons=list(priority.reasons) + ["+10_weekly_recap_floor"],
        )

    # Phase 2.5 full — per-position weekly attribution. Reads ⑨b's
    # rolling 7-day window (Mon-Fri snapshots, typically 5 trading
    # days). compute_weekly_attribution returns [] if no snapshots
    # exist yet (fresh account) — the card silently omits the block;
    # 1-4 snapshots → "归因数据累积中"; 5+ → best/worst/net rows.
    snapshots = read_weekly_window(archive, today_iso, window_days=7)
    attribution = compute_weekly_attribution(snapshots, report.positions)
    # Per-ticker "days available" can vary (mid-week add/remove); the
    # card's gating threshold uses the longest series we observed —
    # if at least one ticker has 5+ days the window is "full" for
    # display purposes.
    snapshot_days_available = (
        max((len(snaps) for snaps in snapshots.values()), default=0)
    )

    text = format_portfolio_weekly_card(
        report,
        attribution=attribution,
        snapshot_days_available=snapshot_days_available,
    )
    chart = _render_equity_chart(f"组合权益曲线 · 1M · {today_iso}")

    notifier = TelegramNotifier(archive=archive)
    if chart is not None:
        notifier.send_photo(
            chart, caption=text,
            trace=trace,
            title=f"组合周报 · {today_iso}",
            tickers=[p.ticker for p in report.positions][:10],
            priority=priority,
        )
    else:
        notifier.send_text(
            text,
            trace=trace,
            title=f"组合周报 · {today_iso}",
            tickers=[p.ticker for p in report.positions][:10],
            priority=priority,
        )

    logger.info(
        "pushed weekly recap %s (score=%d tier=%s chart=%s)",
        today_iso, priority.score, priority.tier,
        "yes" if chart else "no",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
