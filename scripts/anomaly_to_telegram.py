"""Detect anomalies on TECH_30 and push attributed alerts to Telegram.

On quiet days this script silently exits — no message is sent.

Usage:
    poetry run python scripts/anomaly_to_telegram.py
"""

from __future__ import annotations

import logging

from dotenv import load_dotenv

# Surface WARN+ logs from v2.monitoring so we can see Tavily-empty cases.
logging.basicConfig(level=logging.WARNING, format="  [%(levelname)s] %(message)s")

from v2.archive import Archive
from v2.data import CachedFDClient
from v2.data.price_source import default_price_source
from v2.memory import AnomalyMemory
from v2.monitoring import DEFAULT_CONFIG, attribute, run_monitoring
from v2.observability import capture_trace_with_framing, install_all
from v2.reporting import (
    TelegramNotifier,
    format_anomaly_alert,
    notify_on_error,
    render_price_sparkline,
)
from v2.reporting.priority import compute_importance
from v2.screening import TECH_30

load_dotenv()


@notify_on_error("Anomaly Monitor")
def main() -> None:
    install_all()
    print(f"Monitoring {len(TECH_30)} tickers...")
    # Keep FD context open through attribution so we can fetch company names
    # for the entity-filter step.
    # Phase 4.5-mini: daily prices come from yfinance (no FD 3-day lag).
    # FD is still used for financials / earnings / insider inside the
    # detectors. Ops can flip back to FD via V2_PRICE_SOURCE=fd in env.
    price_source = default_price_source()
    with CachedFDClient() as fd:
        anomalies = run_monitoring(
            TECH_30, fd, DEFAULT_CONFIG, price_source=price_source,
        )

        if not anomalies:
            print("No anomalies on the latest trading day — staying silent.")
            return

        print(
            f"Detected {len(anomalies)} anomalies: "
            f"{', '.join(a.ticker + str(a.flags) for a in anomalies)}"
        )

        # Phase C: long-term anomaly memory (ChromaDB + OpenAI embeddings).
        # Best-effort — if it fails to initialize, we skip historical context.
        try:
            memory = AnomalyMemory()
        except Exception as exc:
            print(f"  [warn] AnomalyMemory init failed: {exc}; running without RAG")
            memory = None

        notifier = TelegramNotifier(archive=Archive(agent="anomaly"))
        for anomaly in anomalies:
            print(f"  Attributing {anomaly.ticker}...")
            with capture_trace_with_framing(
                agent="anomaly", intent="explain_move",
                text=f"(自动推送) 异动 {anomaly.ticker}",
                responder_name="_r_anomaly_monitor",
            ) as trace:
                attribute(anomaly, fd_client=fd, memory=memory)
                chart = render_price_sparkline(
                    anomaly.recent_prices,
                    title=f"{anomaly.ticker} · 最近 {len(anomaly.recent_prices)} 日",
                )
                caption = format_anomaly_alert(anomaly)
                trace.emit("chat_message", role="bot", text=caption[:500])
            priority = compute_importance(
                "anomaly_attribution",
                {
                    "reasons_count": len(anomaly.reasons or []),
                    "flags": list(anomaly.flags or []),
                    "price_change_pct": anomaly.price_change_pct,
                    # Held-position / watchlist booleans aren't trivial
                    # to compute here (would need to crack open the bot
                    # state DB and Alpaca creds at agent runtime). Leave
                    # them off for now — base score is still differentiated
                    # via reasons_count + flags. Future: wire bot.state +
                    # broker.get_portfolio look-ups so the scorer sees the
                    # full picture.
                },
            )
            notifier.send_photo(
                chart, caption=caption,
                trace=trace,
                title=f"异动 · {anomaly.ticker}",
                tickers=[anomaly.ticker],
                priority=priority,
            )
            print(
                f"    pushed ({len(anomaly.reasons)} reasons, "
                f"{len(anomaly.next_steps)} next-steps, "
                f"⛔ filtered {anomaly.filtered_count}, "
                f"🧠 {len(anomaly.historical_context)} historical, "
                f"{anomaly.llm_tokens} tokens)"
            )

    print("Done.")


if __name__ == "__main__":
    main()
