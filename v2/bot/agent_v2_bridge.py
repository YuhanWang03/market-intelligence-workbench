"""Production Telegram transport for Agent V2: plain messages and ``/ask_v2``."""

from __future__ import annotations

import asyncio
import html
import logging
import os
import threading
import time
from typing import Any

from v2.agent_common import telegram_delivery as delivery
from v2.agent_common import presentation
from v2.agent_v2.interfaces import telegram_format
from v2.agent_v2.interfaces.telegram import TelegramFacade, TelegramMessage
from v2.agent_v2.orchestrator import AgentV2Config
from v2.agent_v2.runtime import build_workspace_agent

_AGENT = None
_AGENT_WEB_ENABLED = None
_AGENT_LOCK = threading.Lock()
#: chat id -> the cancel event of the run in progress for that chat.
_ACTIVE_RUNS: dict[int, threading.Event] = {}
#: chat id -> the lock that runs one question of a chat at a time (the
#: handlers are non-blocking so cancel words arrive while a run is in progress).
_CHAT_LOCKS: dict[int, asyncio.Lock] = {}
QUEUED_NOTICE = "上一条问题还在处理中，这条会在它结束后处理；回复「取消」可先停止上一条。"


def chat_lock(chat_id: int) -> asyncio.Lock:
    lock = _CHAT_LOCKS.get(chat_id)
    if lock is None:
        lock = _CHAT_LOCKS[chat_id] = asyncio.Lock()
    return lock
_CANCEL_WORDS = {"取消", "停", "停止", "别查了", "stop", "cancel"}


INTERRUPTED_NOTICE = "刚才处理「{question}」时服务重启了，那次没有完成；请把问题再发一次。"


def _durable_state():
    """The agent's durable session state, when the runtime has one."""

    session = getattr(_get_agent(), "session", None)
    return getattr(session, "durable", None)


async def notify_interrupted_runs(bot: Any) -> int:
    """At startup: tell every chat whose answer was in progress when the process died; returns the chats told."""

    try:
        durable = _durable_state()
    except Exception:  # noqa: BLE001 — the runtime failing to build is reported elsewhere
        return 0
    if durable is None:
        return 0
    told = 0
    for row in durable.take_interrupted():
        try:
            await bot.send_message(chat_id=int(row["chat_id"]), text=INTERRUPTED_NOTICE.format(question=str(row.get("question") or "")[:60]))
            told += 1
        except Exception as exc:  # noqa: BLE001 — a chat that cannot be reached is not worth a crash at startup
            logging.getLogger(__name__).warning("interrupted-run notice not sent to %s: %s", row.get("chat_id"), exc)
    return told


def cancel_active_run(chat_id: int) -> bool:
    """Set the cancel flag of the chat's run in progress; False when nothing is running."""

    event = _ACTIVE_RUNS.get(chat_id)
    if event is None:
        return False
    event.set()
    return True


def is_cancel_word(text: str) -> bool:
    return (text or "").strip().rstrip("。！!.").lower() in _CANCEL_WORDS


def feedback_reply(chat_id: int, text: str) -> str | None:
    """Handle a feedback or preference command against the chat's last answer; None when the text is a question."""

    from v2.agent_v2.memory import parse_feedback

    parsed = parse_feedback(text)
    if parsed is None:
        return None
    memory = getattr(_get_agent(), "memory", None)
    if memory is None:
        return "这个部署没有开启用户记忆。"
    session_id = str(chat_id)
    if parsed["kind"] == "preference":
        memory.add_preference(session_id, parsed["note"])
        return f"记下了：{parsed['note']}。之后的回答会照这个来。"
    row = memory.record_feedback(session_id, parsed["verdict"], parsed.get("note", ""))
    if row is None:
        return "这个会话里还没有可以评价的回答。"
    if parsed["verdict"] == "good":
        return "收到，记为一次好的回答。"
    return "收到，已记为有误" + (f"：{parsed['note']}" if parsed.get("note") else "") + "。这条会进入质量评测，下次改动时会检查是否纠正。"


def _web_enabled() -> bool:
    return os.environ.get("AGENT_V2_WEB_ENABLED", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def web_default() -> bool:
    """Whether a plain message may use the web unless it says ``--noweb``.

    The bot answers one authorised owner, so the consent the web page asks
    for with a checkbox is given once here, by configuration.  The server
    flag ``AGENT_V2_WEB_ENABLED`` stays the master switch.
    """

    return os.environ.get("TELEGRAM_WEB_DEFAULT", "1").strip().lower() not in {"0", "false", "no", "off"}


def split_web_consent(text: str) -> tuple[str, bool]:
    """Take ``--web`` / ``--noweb`` out of a message; what is left is the question.

    ``--web`` always allows the web for this message and ``--noweb`` always
    forbids it; without either, :func:`web_default` decides.
    """

    tokens = text.split()
    lowered = [token.lower() for token in tokens]
    if "--noweb" in lowered:
        allow_web = False
    elif "--web" in lowered:
        allow_web = True
    else:
        allow_web = web_default()
    question = " ".join(token for token in tokens if token.lower() not in {"--web", "--noweb"}).strip()
    return question, allow_web


def _get_agent():
    global _AGENT, _AGENT_WEB_ENABLED
    web_enabled = _web_enabled()
    if _AGENT is None or _AGENT_WEB_ENABLED != web_enabled:
        with _AGENT_LOCK:
            if _AGENT is None or _AGENT_WEB_ENABLED != web_enabled:
                _AGENT = build_workspace_agent(
                    config=AgentV2Config(),
                    enable_web=web_enabled,
                )
                _AGENT_WEB_ENABLED = web_enabled
    return _AGENT


class TelegramBotTransport:
    """Adapt python-telegram-bot objects to the framework-neutral facade."""

    def __init__(self, context: Any, placeholder: Any, *, web_requested: bool) -> None:
        self.context = context
        self.placeholder = placeholder
        self.web_requested = web_requested
        self._last_progress_at = 0.0
        self._last_status = ""
        self._started = time.monotonic()
        self._stages: list[str] = []

    async def typing(self, chat_id: int) -> None:
        try:
            await self.context.bot.send_chat_action(chat_id=chat_id, action="typing")
        except Exception:  # noqa: BLE001 — typing is cosmetic
            pass

    async def progress(self, chat_id: int, event) -> None:
        now = time.monotonic()
        line = telegram_format.progress_line(event, elapsed=now - self._started)
        stage = line.split(" · ")[0]
        if stage != (self._stages[-1] if self._stages else ""):
            self._stages.append(stage)
        # One edit every few seconds at most: Telegram rate-limits edits, and the last stage is what matters.
        if now - self._last_progress_at < 3 and stage == self._last_status:
            return
        self._last_status = stage
        self._last_progress_at = now
        recent = self._stages[-4:]
        body = "\n".join(("▸ " if index == len(recent) - 1 else "· ") + html.escape(item) for index, item in enumerate(recent))
        try:
            await self.placeholder.edit_text(
                f"<b>Agent V2 · 处理中 {now - self._started:.0f}s</b>\n<i>{body}</i>\n<i>回复「取消」可停止</i>",
                parse_mode="HTML",
                disable_web_page_preview=True,
            )
        except Exception:  # noqa: BLE001 — final delivery still matters
            pass

    async def deliver(self, chat_id: int, result) -> None:
        header = self._header(result)
        numbered = telegram_format.number_citations(
            telegram_format.compact_attributions(result.answer, result),
            result.evidence,
        )
        answer = presentation.to_telegram_html(numbered.text)
        # The sub-agent notes (the debater's objections) cite evidence too:
        # they continue the answer's numbering and their sources join the list.
        order = list(numbered.ids)
        agents = [telegram_format.number_citations(line, result.evidence, order=order).text for line in telegram_format.agent_lines(result)]
        sources = []
        for entry in telegram_format.source_entries(tuple(order), result.evidence):
            label = html.escape(entry.label)
            if entry.url:
                label = f'<a href="{html.escape(entry.url, quote=True)}">{label}</a>'
            sources.append(f"{html.escape(entry.numbers)}. {label}")
        suffix = "\n\n<b>来源</b>\n" + "\n".join(sources) if sources else ""
        if agents:
            suffix += "\n\n<b>子智能体</b>\n" + "\n".join(html.escape(line) for line in agents)
        await delivery._deliver(self.placeholder, header + answer + suffix)

    def _header(self, result) -> str:
        fields = [
            f"路径：{html.escape(result.route.kind.value)}",
            f"回答：{html.escape(result.answer_mode.value)}",
        ]
        synthesis = telegram_format.synthesis_label(result)
        if synthesis:
            note = telegram_format.completion_note(result)
            fields.append(f"合成：{html.escape(synthesis + (f'，{note}' if note else ''))}")
        fields.append(f"校验：{html.escape(telegram_format.verification_label(result))}")
        fields.append(
            "网页：" + html.escape(telegram_format.web_label(requested=self.web_requested, enabled=_web_enabled()))
        )
        lines = [f"<b>Agent V2 · {html.escape(result.status.value)}</b>", f"<i>{' · '.join(fields)}</i>"]
        warning = telegram_format.warning_line(result)
        if warning:
            lines.append(f"<i>⚠ 校验：{html.escape(warning)}</i>")
        budget = telegram_format.budget_line(result)
        if budget:
            lines.append(f"<i>⏱ {html.escape(budget)}</i>")
        if result.stop_reason == "cancelled":
            lines.append("<i>⏹ 已按要求停止，下面是停止前拿到的部分</i>")
        reason = telegram_format.fallback_reason(result)
        if reason:
            lines.append(f"<i>兜底原因：{html.escape(reason)}</i>")
        repaired = telegram_format.repair_reason(result)
        if repaired:
            lines.append(f"<i>修正原因：{html.escape(repaired)}</i>")
        return "\n".join(lines) + "\n\n"


async def handle_agent_v2(
    update: Any,
    context: Any,
    text: str,
    *,
    allow_web: bool = False,
) -> Any:
    """Run one explicit Agent V2 request without touching normal NL routing."""

    chat = update.effective_chat
    message = update.message
    if is_cancel_word(text) and cancel_active_run(chat.id):
        await message.reply_html("正在停止当前的处理…")
        return None
    reply = feedback_reply(chat.id, text)
    if reply is not None:
        await message.reply_html(html.escape(reply))
        return None
    lock = chat_lock(chat.id)
    if lock.locked():
        # One answer per chat at a time, in the order asked; the user is told
        # the earlier one is still running and can stop it.
        await message.reply_html(QUEUED_NOTICE)
    async with lock:
        placeholder = await message.reply_html("🧭 Agent V2 正在识别问题…\n<i>回复「取消」可停止</i>")
        transport = TelegramBotTransport(
            context,
            placeholder,
            web_requested=allow_web,
        )
        cancel_event = threading.Event()
        _ACTIVE_RUNS[chat.id] = cancel_event
        durable = _durable_state()
        if durable is not None:
            durable.mark_active(str(chat.id), text)
        try:
            return await TelegramFacade(_get_agent()).handle(
                TelegramMessage(
                    chat_id=chat.id,
                    text=text,
                    message_id=getattr(message, "message_id", None),
                ),
                transport,
                allow_web=allow_web,
                cancel_event=cancel_event,
            )
        finally:
            if _ACTIVE_RUNS.get(chat.id) is cancel_event:
                _ACTIVE_RUNS.pop(chat.id, None)
            if durable is not None:
                durable.clear_active(str(chat.id))
