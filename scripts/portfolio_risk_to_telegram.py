"""Portfolio risk daily report — Mon-Fri 18:30 ET.

The ninth scheduled agent. Builds a :class:`RiskReport` covering
positions / concentration / sector exposure / P&L / drawdown / 7-day
earnings risk, and pushes one Telegram card. Priority is computed from
the report:

    base = 55 (P2)
    + daily_pnl_pct ≤ -5%    → +30  (→ P0)
    + daily_pnl_pct ≤ -2%    → +10  (→ P1)
    + top_1_pct ≥ 30%        → +20  (→ P1)
    + top_1_pct ≥ 20%        → +10
    + max_drawdown_pct ≥ 10% → +15
    + n_earnings_next_7d ≥ 3 → +10

Caveats (Stage-2 plan):
- Alpaca unavailable → cron still pushes a P2 "组合数据暂不可用" card
  using the warnings populated by ``build_risk_report``.
- Card formatter lives in :mod:`v2.reporting.format_portfolio_risk_card`
  (Stage 5 lift) — same source of truth as the bot's ``/risk`` path,
  so the byte-equal contract holds across both surfaces.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

from v2.archive import Archive
from v2.observability import capture_trace_with_framing, install_all
from v2.portfolio import build_risk_report
from v2.reporting import (
    TelegramNotifier,
    format_portfolio_risk_card,
    notify_on_error,
)
from v2.reporting.priority import compute_importance


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


_TZ_ET = ZoneInfo("US/Eastern")


@notify_on_error("Portfolio Risk")
def main() -> int:
    load_dotenv()
    install_all()

    archive = Archive("portfolio")
    today_iso = datetime.now(_TZ_ET).date().isoformat()

    with capture_trace_with_framing(
        agent="portfolio", intent="portfolio_risk_view",
        text=f"(自动推送) 组合风险 · {today_iso}",
        responder_name="_r_portfolio_risk",
    ) as trace:
        report = build_risk_report(today_iso=today_iso)
        trace.emit("chat_message", role="bot",
                   text=f"组合风险卡 · {len(report.positions)} 持仓 · "
                        f"{len(report.warnings)} 警告")

    priority = compute_importance(
        "portfolio_risk",
        {
            "daily_pnl_pct":      report.pnl.daily_pnl_pct or 0.0,
            "top_1_pct":          report.concentration.top_1_pct,
            "max_drawdown_pct":   report.drawdown.max_drawdown_pct or 0.0,
            "n_earnings_next_7d": len(report.earnings_risk_next_7d),
        },
    )
    text = format_portfolio_risk_card(report)

    notifier = TelegramNotifier(archive=archive)
    notifier.send_text(
        text,
        trace=trace,
        title=f"组合风险 · {today_iso}",
        tickers=[p.ticker for p in report.positions][:10],
        priority=priority,
    )

    logger.info(
        "pushed risk card %s (score=%d tier=%s positions=%d warnings=%d)",
        today_iso, priority.score, priority.tier,
        len(report.positions), len(report.warnings),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
