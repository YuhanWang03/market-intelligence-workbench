"""Transport-neutral facades for future Web and Telegram integration."""

from v2.agent_v2.interfaces.telegram import (
    TelegramFacade,
    TelegramMessage,
    TelegramTransport,
)
from v2.agent_v2.interfaces.web import WebFacade, WebRequest

__all__ = ["TelegramFacade", "TelegramMessage", "TelegramTransport", "WebFacade", "WebRequest"]
