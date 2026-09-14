"""Print chat_ids that have messaged your bot recently.

Run this AFTER you've sent /start (or any message) to the bot in Telegram.

Usage:
    poetry run python scripts/find_chat_id.py
"""

from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv
from telegram import Bot

load_dotenv()


async def main() -> None:
    bot = Bot(token=os.environ["TELEGRAM_BOT_TOKEN"])
    async with bot:
        me = await bot.get_me()
        print(f"Bot: @{me.username} (id={me.id})\n")

        updates = await bot.get_updates(timeout=5)
        if not updates:
            print("No updates. Send /start to your bot in Telegram, then re-run.")
            return

        seen: set[int] = set()
        for u in updates:
            msg = u.message or u.edited_message
            if not msg:
                continue
            chat = msg.chat
            if chat.id in seen:
                continue
            seen.add(chat.id)
            who = chat.username or chat.first_name or chat.title or "?"
            print(f"chat_id={chat.id:<15} type={chat.type:<8} who={who}")

        print(f"\nCurrent .env TELEGRAM_CHAT_ID={os.environ.get('TELEGRAM_CHAT_ID')}")


if __name__ == "__main__":
    asyncio.run(main())
