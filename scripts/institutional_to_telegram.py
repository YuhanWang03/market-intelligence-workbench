"""Run institutional 13F tracking and push to Telegram.

Usage:
    poetry run python scripts/institutional_to_telegram.py
"""

from __future__ import annotations

import logging
import time

from dotenv import load_dotenv

from v2.archive import Archive
from v2.institutional import run_institutional_pipeline
from v2.observability import capture_trace_with_framing, install_all
from v2.reporting import (
    TelegramNotifier,
    format_institutional_messages,
    notify_on_error,
)
from v2.reporting.priority import compute_importance
from v2.screening import TECH_30

logging.basicConfig(level=logging.INFO, format="  [%(levelname)s] %(message)s")

load_dotenv()


@notify_on_error("Institutional 13F")
def main() -> None:
    install_all()
    print("Running institutional 13F pipeline...")

    # Capture the entire pipeline trace under one Trace object. The
    # underlying run touches every manager in a tight loop, so events for
    # different managers are interleaved — we'd need a deeper refactor to
    # split them cleanly. For dashboard purposes, the summary card carries
    # the full trace; per-manager cards carry titles only.
    with capture_trace_with_framing(
        agent="institutional", intent="thirteen_f",
        text="(自动推送) 13F 总览",
        responder_name="_r_institutional",
    ) as trace:
        report = run_institutional_pipeline(universe=set(TECH_30))

        print(
            f"\nDone. {len(report.new_filings)} new filings · "
            f"{len(report.changes)} significant changes"
        )

        if not report.new_filings:
            print("No new 13F filings since last run — staying silent.")
            return

        messages = format_institutional_messages(report)
        # First message (the summary card) is what the saved trace is
        # attached to. Emit chat_message with the summary so the trace
        # has a closing reply event.
        if messages:
            trace.emit("chat_message", role="bot", text=messages[0][:500])

    print(f"Pushing {len(messages)} messages to Telegram...")

    # Manager name for each per-manager message (preserves new_filings
    # order — same order format_institutional_messages produced).
    manager_names = [f.manager_name for f in report.new_filings]

    notifier = TelegramNotifier(archive=Archive(agent="institutional"))
    for i, msg in enumerate(messages, 1):
        # First message is the summary header; the rest are per-manager.
        is_summary = i == 1
        if is_summary:
            title = f"13F 总览 · {len(report.new_filings)} managers"
            attached_trace = trace
        else:
            manager = manager_names[i - 2] if i - 2 < len(manager_names) else "?"
            title = f"13F · {manager}"
            attached_trace = None  # avoid duplicating ~30KB per manager

        priority = compute_importance("institutional_13f", {})  # P1
        notifier.send_text(msg, trace=attached_trace, title=title, priority=priority)
        print(f"  [{i}/{len(messages)}] sent")
        if i < len(messages):
            time.sleep(0.3)   # gentle pacing to avoid rate limit spikes

    print("Pushed.")


if __name__ == "__main__":
    main()
