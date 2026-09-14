"""Interactive Telegram bot — slash commands + (Stage 3) NL intent routing.

The Telegram runtime is optional for the web application.  Keep its heavy
dependency graph lazy so callers can use ``v2.bot.state`` without having to
install python-telegram-bot.
"""

from __future__ import annotations

from typing import Any

__all__ = ["build_application", "run_bot"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from v2.bot.main import build_application, run_bot

        return {"build_application": build_application, "run_bot": run_bot}[name]
    raise AttributeError(name)
