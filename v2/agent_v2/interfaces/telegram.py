"""Telegram integration contract without importing python-telegram-bot."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Protocol

from v2.agent_v2.models import AgentResult, ProgressEvent
from v2.agent_v2.ports import AgentPort


@dataclass(frozen=True)
class TelegramMessage:
    chat_id: int
    text: str
    message_id: int | None = None


class TelegramTransport(Protocol):
    async def typing(self, chat_id: int) -> None:
        ...

    async def progress(self, chat_id: int, event: ProgressEvent) -> None:
        ...

    async def deliver(self, chat_id: int, result: AgentResult) -> None:
        ...


class TelegramFacade:
    def __init__(self, agent: AgentPort) -> None:
        self.agent = agent

    async def handle(
        self,
        message: TelegramMessage,
        transport: TelegramTransport,
        *,
        allow_web: bool = False,
        cancel_event=None,
    ) -> AgentResult:
        await transport.typing(message.chat_id)
        loop = asyncio.get_running_loop()
        pending_progress = []

        def progress(event: ProgressEvent) -> None:
            pending_progress.append(
                asyncio.run_coroutine_threadsafe(
                    transport.progress(message.chat_id, event),
                    loop,
                )
            )

        run_kwargs = {"session_id": str(message.chat_id), "allow_web": allow_web, "on_progress": progress}
        if cancel_event is not None:
            run_kwargs["cancel_event"] = cancel_event
        result = await asyncio.to_thread(self.agent.run, message.text, **run_kwargs)
        for future in pending_progress:
            await asyncio.wrap_future(future)
        await transport.deliver(message.chat_id, result)
        return result
