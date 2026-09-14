"""⑬ ARK Alerts — Mon-Fri 08:30 ET pre-market.

Consumes the daily ARK CSV snapshots already populated by ⑤ ETF Daily
Snapshot (17:00 ET evening before) and pushes Telegram alerts whenever
a significant rebalance is detected:

- ``new_position`` with today_weight ≥ 0.5%
- ``liquidated`` with yesterday_weight ≥ 0.5%
- ``increase`` / ``decrease`` with |relative| ≥ 20%

Multi-fund coordinated moves (same ticker, same direction, ≥2 funds) +
user-universe membership (held ∪ watchlist) escalate priority via the
``v2.reporting.priority`` adjustments.

Quiet days (no alerts) → silent skip; archive logs the trace so the
dashboard feed still records that ⑬ ran.

First-deploy edge: ⑤ populates the ``etf.db.snapshots`` baseline; on a
fresh install with no yesterday snapshot, ``get_latest_snapshot_before``
returns None and the fund contributes zero alerts.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

from v2.archive import Archive
from v2.bot import state as bot_state
from v2.etf import (
    SUPPORTED_FUNDS,
    compute_daily_changes,
    fetch_holdings,
    get_latest_snapshot_before,
)
from v2.etf.alerts import ArkScanResult, classify_alerts
from v2.observability import capture_trace_with_framing, install_all
from v2.reporting import (
    TelegramNotifier,
    format_ark_alert,
    format_ark_summary,
    notify_on_error,
)
from v2.reporting.priority import compute_importance


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


_TZ_ET = ZoneInfo("US/Eastern")


def _resolve_user_universe() -> set[str]:
    """Held ∪ watchlist, uppercase. Alpaca down → silent fall back to
    watchlist-only (per Phase 1+3 cron convention)."""
    watchlist = {row["ticker"].upper() for row in bot_state.watchlist_list()}
    held: set[str] = set()
    try:
        from v2.broker import get_portfolio
        portfolio = get_portfolio()
        held = {p["symbol"].upper() for p in portfolio.get("positions", [])}
    except Exception as exc:
        logger.info("Alpaca unavailable, watchlist-only universe: %s", exc)
    return watchlist | held


def _gather_diff_inputs(
    today_iso: str,
    funds: list[str],
) -> tuple[dict, dict, dict, list[str], list[str]]:
    """Pull today's holdings + yesterday's snapshot for each fund.

    Returns:
        changes_by_fund, today_holdings_by_fund, yesterday_rows_by_fund,
        funds_scanned, warnings
    """
    changes_by_fund: dict = {}
    today_by_fund: dict = {}
    yest_by_fund: dict = {}
    funds_scanned: list[str] = []
    warnings: list[str] = []

    for fund in funds:
        try:
            holdings, snap_date = fetch_holdings(fund)
        except Exception as exc:
            warnings.append(f"{fund} fetch failed: {exc}")
            logger.warning("⑬ %s fetch failed: %s", fund, exc)
            continue
        if not holdings:
            warnings.append(f"{fund} returned no holdings")
            logger.warning("⑬ %s returned no holdings", fund)
            continue

        try:
            prev = get_latest_snapshot_before(fund, snap_date)
        except Exception as exc:
            warnings.append(f"{fund} baseline lookup failed: {exc}")
            logger.warning("⑬ %s baseline lookup failed: %s", fund, exc)
            prev = None

        funds_scanned.append(fund)
        today_by_fund[fund] = holdings
        yest_by_fund[fund] = prev or []

        if not prev:
            # First-deploy / missing baseline — leave changes_by_fund
            # entry empty; classify_alerts will produce no alerts for
            # this fund. ⑤ continues to populate the baseline at 17:00 ET
            # so subsequent days will have signal.
            changes_by_fund[fund] = []
            logger.info(
                "⑬ %s: no yesterday baseline (first deploy?), 0 alerts",
                fund,
            )
            continue

        try:
            changes_by_fund[fund] = compute_daily_changes(prev, holdings)
        except Exception as exc:
            warnings.append(f"{fund} diff failed: {exc}")
            logger.warning("⑬ %s diff failed: %s", fund, exc)
            changes_by_fund[fund] = []

    return changes_by_fund, today_by_fund, yest_by_fund, funds_scanned, warnings


def _alert_metadata(alert) -> dict:
    """Build the metadata dict compute_importance expects for an ARK
    alert. ArkAlert.today_weight / yesterday_weight are CSV pct units
    (1.85 == 1.85%) — priority's threshold check expects DECIMAL
    fraction units (0.02 == 2%), so we divide by 100 here. The
    formatter card still renders the CSV-native pct since it owns its
    own display."""
    today_w_frac = (alert.today_weight or 0.0) / 100.0
    yest_w_frac = (alert.yesterday_weight or 0.0) / 100.0
    return {
        "action": alert.action,
        "today_weight": today_w_frac,
        "yesterday_weight": yest_w_frac,
        "is_in_user_universe": alert.is_in_user_universe,
        "is_multi_fund": alert.is_multi_fund,
    }


def _push_alert(notifier, trace, alert) -> None:
    metadata = _alert_metadata(alert)
    priority = compute_importance("ark_alert_p1", metadata)
    text = format_ark_alert(alert)
    action_zh = {
        "new_position": "新建仓",
        "liquidated":   "清仓",
        "increase":     "增持",
        "decrease":     "减持",
    }.get(alert.action, alert.action)
    notifier.send_text(
        text,
        trace=trace,
        title=f"ARK {action_zh} · {alert.ticker} · {alert.fund}",
        tickers=[alert.ticker],
        priority=priority,
    )


def _push_summary(notifier, trace, result: ArkScanResult) -> None:
    """Overview card — P2 default. Operator-visibility floor; the
    individual alert cards already carry their own priority tier."""
    priority = compute_importance("ark_alert_p2", {})
    text = format_ark_summary(result)
    notifier.send_text(
        text,
        trace=trace,
        title=f"ARK 调仓总览 · {result.scan_date}",
        tickers=sorted({a.ticker for a in result.alerts}),
        priority=priority,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@notify_on_error("ARK Alerts")
def main() -> int:
    load_dotenv()
    install_all()

    today_iso = datetime.now(_TZ_ET).date().isoformat()
    user_universe = _resolve_user_universe()
    archive = Archive("ark")

    with capture_trace_with_framing(
        agent="ark", intent="ark_alert_view",
        text=f"(自动推送) ARK 调仓扫描 · {today_iso}",
        responder_name="_r_ark_alerts",
    ) as trace:
        (
            changes_by_fund, today_by_fund, yest_by_fund,
            funds_scanned, warnings,
        ) = _gather_diff_inputs(today_iso, SUPPORTED_FUNDS)

        alerts = classify_alerts(
            changes_by_fund=changes_by_fund,
            today_holdings_by_fund=today_by_fund,
            yesterday_rows_by_fund=yest_by_fund,
            user_universe=user_universe,
        )

        result = ArkScanResult(
            scan_date=today_iso,
            funds_scanned=funds_scanned,
            funds_attempted=list(SUPPORTED_FUNDS),
            alerts=alerts,
            warnings=warnings,
        )

        trace.emit(
            "chat_message", role="bot",
            text=(
                f"ARK 调仓扫描 · {today_iso} · "
                f"funds={len(funds_scanned)} · "
                f"alerts={len(alerts)} · "
                f"multi={sum(1 for a in alerts if a.is_multi_fund)} · "
                f"user_universe={sum(1 for a in alerts if a.is_in_user_universe)}"
            ),
        )

        if not alerts:
            logger.info(
                "⑬ ARK quiet day: %d funds scanned, 0 alerts — silent skip",
                len(funds_scanned),
            )
            return 0

        notifier = TelegramNotifier(archive=archive)

        # Multi-fund / user-universe alerts ranked higher visually go first
        # so they hit the top of the feed; ties broken by descending
        # absolute relative change.
        sorted_alerts = sorted(
            alerts,
            key=lambda a: (
                not a.is_multi_fund,
                not a.is_in_user_universe,
                -abs(a.weight_change_relative or 0.0),
            ),
        )

        for alert in sorted_alerts:
            try:
                _push_alert(notifier, trace, alert)
            except Exception as exc:
                logger.warning(
                    "⑬ alert push failed for %s/%s: %s",
                    alert.fund, alert.ticker, exc,
                )

        # Overview last so it sits below the individual cards
        try:
            _push_summary(notifier, trace, result)
        except Exception as exc:
            logger.warning("⑬ summary push failed: %s", exc)

    logger.info(
        "⑬ ARK Alerts done: %d funds / %d alerts / %d multi-fund / "
        "%d user-universe / %d warnings",
        len(funds_scanned), len(alerts),
        sum(1 for a in alerts if a.is_multi_fund),
        sum(1 for a in alerts if a.is_in_user_universe),
        len(warnings),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
