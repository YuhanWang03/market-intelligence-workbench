"""Macro daily snapshot — Mon-Fri 16:30 ET.

The fourteenth scheduled agent. Pulls post-close ambient market levels
(VIX / DXY / WTI / Gold) from yfinance and Treasury rates / Fed Funds
(canonical EOD) from FRED, then routes the push through a priority
kind that reflects the dominant anomaly flag:

- ``macro_vix_spike``   (P0 base, +up to 20 by magnitude)  — VIX +20%
- ``macro_curve_flip``  (P1 base, +10 if inverted)         — T10Y2Y sign flip
- ``macro_snapshot_p3`` (P3)                                — default daily

VIX-elevated (+10% < pct < +20%) and rates_shocked (|ΔDGS10| ≥ 20bps)
also escalate to ``macro_curve_flip`` since both are P1-class
volatility signals — the kind name is the routing label, not a literal
event-type assertion.

Card formatter is inline here for Stage 2 (Stage 5 lifts to
``v2.reporting.format_macro_*``).
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

from v2.archive import Archive
from v2.macro import build_macro_snapshot
from v2.observability import capture_trace_with_framing, install_all
from v2.reporting import (
    TelegramNotifier,
    format_macro_daily_snapshot,
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
# Priority routing
# ---------------------------------------------------------------------------

def _classify(snap) -> tuple[str, dict]:
    """Return (priority_kind, metadata) for a snapshot.

    Highest-priority anomaly wins: VIX spike > curve flip > VIX
    elevated / rates shocked > default ambient.
    """
    if snap.vix_spike:
        return "macro_vix_spike", {
            "vix_pct_change_1d": snap.vix_pct_change_1d,
            "vix": snap.vix,
        }
    if snap.curve_flip:
        return "macro_curve_flip", {
            "t10y2y": snap.t10y2y,
            "t10y2y_prior": snap.t10y2y_prior,
        }
    if snap.vix_elevated or snap.rates_shocked:
        return "macro_curve_flip", {
            "vix_pct_change_1d": snap.vix_pct_change_1d,
            "rates_shocked": snap.rates_shocked,
        }
    return "macro_snapshot_p3", {}


# ---------------------------------------------------------------------------
# Push helper
# ---------------------------------------------------------------------------

def _emit(notifier, trace, snap) -> None:
    kind, md = _classify(snap)
    priority = compute_importance(kind, md)

    text = format_macro_daily_snapshot(snap)
    notifier.send_text(
        text,
        trace=trace,
        title=f"宏观日终 · {snap.snapshot_date} · {priority.tier}",
        tickers=[],
        priority=priority,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@notify_on_error("Macro Daily Snapshot")
def main() -> int:
    load_dotenv()
    install_all()

    today_iso = datetime.now(_TZ_ET).date().isoformat()
    archive = Archive("macro")

    with capture_trace_with_framing(
        agent="macro", intent="macro_snapshot_view",
        text=f"(自动推送) 宏观日终快照 · {today_iso}",
        responder_name="_r_macro_snapshot",
    ) as trace:
        snap = build_macro_snapshot(today_iso)
        trace.emit(
            "chat_message", role="bot",
            text=(
                f"宏观日终 · VIX={snap.vix} ({snap.vix_pct_change_1d}) · "
                f"DGS10={snap.dgs10} · T10Y2Y={snap.t10y2y} · "
                f"warnings={len(snap.warnings)}"
            ),
        )

        notifier = TelegramNotifier(archive=archive)
        try:
            _emit(notifier, trace, snap)
        except Exception as exc:
            logger.warning("macro snapshot push failed: %s", exc)
            return 1

    logger.info(
        "Macro snapshot complete: VIX=%s ΔVIX=%s spike=%s flip=%s "
        "rates_shocked=%s warnings=%d",
        snap.vix, snap.vix_pct_change_1d, snap.vix_spike, snap.curve_flip,
        snap.rates_shocked, len(snap.warnings),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
