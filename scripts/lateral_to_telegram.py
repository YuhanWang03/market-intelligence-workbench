"""Run one pass of lateral expansion and push the result to Telegram.

Usage:
    poetry run python scripts/lateral_to_telegram.py
"""

from __future__ import annotations

import logging

from dotenv import load_dotenv

from v2.archive import Archive
from v2.data import CachedFDClient
from v2.data.price_source import default_price_source
from v2.lateral import DEFAULT_SEEDS, LATERAL_FILTERS, run_lateral_expansion
from v2.observability import capture_trace_with_framing, install_all
from v2.reporting import TelegramNotifier, format_lateral_result, notify_on_error
from v2.reporting.priority import compute_importance
from v2.screening import TECH_30

# Show INFO from the orchestrator so we can watch progress.
logging.basicConfig(level=logging.INFO, format="  [%(levelname)s] %(message)s")

load_dotenv()


@notify_on_error("Lateral Expansion")
def main() -> None:
    install_all()
    seeds = DEFAULT_SEEDS
    universe = set(TECH_30)

    print(f"Lateral expansion · {len(seeds)} seeds: {', '.join(seeds)}")

    with capture_trace_with_framing(
        agent="lateral", intent="chain",
        text="(自动推送) 产业链横向扩展",
        responder_name="_r_lateral_expansion",
    ) as trace:
        # Phase 4.5-mini: daily prices come from yfinance (no FD 3-day lag).
        # FD still serves financials + earnings inside build_candidate.
        # Ops can flip back via V2_PRICE_SOURCE=fd.
        price_source = default_price_source()
        with CachedFDClient() as fd:
            result = run_lateral_expansion(
                seeds=seeds,
                universe=universe,
                fd_client=fd,
                filter_config=LATERAL_FILTERS,
                price_source=price_source,
            )

        passers = sum(
            1 for n in result.neighbors
            if n.exists and not n.already_in_universe and n.passed_filter
        )
        hallucinated = sum(1 for n in result.neighbors if not n.exists)
        print(
            f"\nDone. {len(result.neighbors)} unique candidates · "
            f"{passers} passed full filter · {hallucinated} hallucinations"
        )
        text = format_lateral_result(result)
        trace.emit("chat_message", role="bot", text=text[:500])

    print("Pushing to Telegram...")
    priority = compute_importance("lateral_expansion", {})  # P2
    notifier = TelegramNotifier(archive=Archive(agent="lateral"))
    notifier.send_text(
        text,
        trace=trace,
        title=f"产业链 · {', '.join(seeds[:3])}{' …' if len(seeds) > 3 else ''}",
        tickers=seeds,
        priority=priority,
    )
    print("Pushed.")


if __name__ == "__main__":
    main()
