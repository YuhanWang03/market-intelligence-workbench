"""Intraday price-alert streamer — keep this terminal open to keep running.

Polls every minute during US market hours (9:00 - 16:30 ET, Mon-Fri),
checks open alerts against current Alpaca prices, and fires triggered
ones to Telegram.

Usage:
    poetry run python scripts/run_streamer.py
    poetry run python scripts/run_streamer.py --test-now
        — force ONE poll right now (ignores market-hours gate), then exit.
"""

from __future__ import annotations

import logging
import sys

from v2.streamer import run_streamer

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)


if __name__ == "__main__":
    test_now = "--test-now" in sys.argv
    run_streamer(test_now=test_now)
