"""⑱ Detect money-flow divergence on TECH_30 and push cards to Telegram.

Three-axis (price / CMF / RSI) divergence → accumulation (疑似吸筹) or
distribution (疑似派发). "strong" verdicts push immediately (P1); "moderate"
ones archive into the daily P2 digest. Quiet days exit silently.

Usage:
    poetry run python scripts/moneyflow_to_telegram.py
"""

from __future__ import annotations

import logging

from dotenv import load_dotenv

logging.basicConfig(level=logging.WARNING, format="  [%(levelname)s] %(message)s")

from v2.archive import Archive
from v2.data import CachedFDClient
from v2.moneyflow import DEFAULT_CONFIG, format_signal_card, run_moneyflow
from v2.observability import capture_trace_with_framing, install_all
from v2.reporting import TelegramNotifier, notify_on_error
from v2.reporting.priority import compute_importance
from v2.screening import TECH_30

load_dotenv()


@notify_on_error("Money Flow Divergence")
def main() -> None:
    install_all()
    print(f"Scanning {len(TECH_30)} tickers for money-flow divergence...")

    with CachedFDClient() as fd:
        result = run_moneyflow(TECH_30, fd, DEFAULT_CONFIG)

    if not result.signals:
        print("No divergence on the latest trading day — staying silent.")
        return

    print(
        f"Detected {len(result.signals)} divergences "
        f"({result.fd_calls} FD calls, {result.llm_tokens} tokens): "
        + ", ".join(f"{s.ticker}/{s.kind}/{s.strength}" for s in result.signals)
    )

    notifier = TelegramNotifier(archive=Archive(agent="moneyflow"))
    for s in result.signals:
        with capture_trace_with_framing(
            agent="moneyflow", intent="explain_move",
            text=f"(自动推送) 资金流背离 {s.ticker}",
            responder_name="_r_moneyflow_divergence",
        ) as trace:
            caption = format_signal_card(s, price_window=DEFAULT_CONFIG.price_window)
            trace.emit("chat_message", role="bot", text=caption[:500])

        priority = compute_importance(
            "moneyflow_divergence",
            {
                "strength": s.strength,
                "rsi_divergence": s.rsi_divergence,
                # Held / watchlist booleans left off for MVP (would need bot
                # state + Alpaca at agent runtime) — mirrors the anomaly cron.
                # Base + strength already differentiate P1 vs P2.
            },
        )
        notifier.send_text(
            caption,
            trace=trace,
            title=f"资金流背离 · {s.ticker}",
            tickers=[s.ticker],
            priority=priority,
        )
        print(f"    {s.ticker}: {priority.tier} ({','.join(priority.reasons)})")

    print("Done.")


if __name__ == "__main__":
    main()
