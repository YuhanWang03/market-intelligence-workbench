"""One-shot verification that the bot can push to your chat.

Usage:
    poetry run python scripts/verify_telegram.py
"""

from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv
from telegram import Bot

load_dotenv()


async def main() -> None:
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    chat_id = os.environ["TELEGRAM_CHAT_ID"]

    bot = Bot(token=token)
    async with bot:
        me = await bot.get_me()
        msg = await bot.send_message(
            chat_id=chat_id,
            text=(
                "<b>Hedge Fund Bot online</b>\n"
                f"Bot: <code>@{me.username}</code>\n"
                "Verification successful."
            ),
            parse_mode="HTML",
        )
        print(f"OK — sent message id={msg.message_id} to chat {chat_id}")


if __name__ == "__main__":
    asyncio.run(main())
