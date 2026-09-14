"""Launch the interactive Telegram bot — keep this process alive forever.

Usage:
    poetry run python scripts/run_telegram_bot.py

For production, run as a systemd service (see hedge-fund-bot.service).
"""

from __future__ import annotations

import logging

from dotenv import load_dotenv

from v2.bot import run_bot

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-7s %(name)s %(message)s",
    datefmt="%H:%M:%S",
)

load_dotenv()


if __name__ == "__main__":
    run_bot()
