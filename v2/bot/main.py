"""Bot application launcher — registers handlers and runs the polling loop."""

from __future__ import annotations

import logging
import os

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
)

from v2.bot import commands

logger = logging.getLogger(__name__)


async def _after_start(application) -> None:
    """Chats whose Agent V2 answer was cut off by the restart hear about it once."""

    from v2.bot.agent_v2_bridge import notify_interrupted_runs

    try:
        told = await notify_interrupted_runs(application.bot)
    except Exception as exc:  # noqa: BLE001 — startup must not fail on a notice
        logger.warning("interrupted-run notices failed: %s", exc)
        return
    if told:
        logger.info("told %d chat(s) their answer was interrupted by the restart", told)


async def _error_handler(update: object, context) -> None:
    """Surface unhandled bot errors to the user instead of silent fail.

    Without this, an exception inside a handler (e.g.
    ``telegram.error.BadRequest`` from malformed card HTML) leaves the
    placeholder message stuck forever and the bot appears frozen.
    With this, the user gets a short message naming the error class,
    and the full traceback lands in ``bot.err`` for diagnosis.

    Added 2026-06-04 after the ``/pnl week`` HTML escape silent-fail
    incident (hot patch 5f61795).
    """
    logger.exception("Unhandled bot error", exc_info=context.error)

    if not isinstance(update, Update):
        return
    chat = update.effective_chat
    if chat is None:
        return

    err_type = type(context.error).__name__ if context.error else "Unknown"
    try:
        await chat.send_message(
            f"⚠️ 命令执行失败：{err_type}\n"
            f"详情已记录到 bot.err 日志，请联系开发者。"
        )
    except Exception:
        logger.exception("Error handler itself failed — giving up")


def build_application() -> Application:
    """Wire up the bot — call build_application().run_polling() to launch."""
    token = os.environ["TELEGRAM_BOT_TOKEN"]

    app = Application.builder().token(token).post_init(_after_start).build()

    # Stage 1 commands (watchlist + meta)
    app.add_handler(CommandHandler("start", commands.cmd_start))
    app.add_handler(CommandHandler("help", commands.cmd_help))
    app.add_handler(CommandHandler("watchlist", commands.cmd_watchlist))
    app.add_handler(CommandHandler("add", commands.cmd_add))
    app.add_handler(CommandHandler("remove", commands.cmd_remove))

    # Stage 2 action commands (FD + LLM-heavy)
    app.add_handler(CommandHandler("why", commands.cmd_why))
    app.add_handler(CommandHandler("summary", commands.cmd_summary))
    app.add_handler(CommandHandler("chain", commands.cmd_chain))
    app.add_handler(CommandHandler("13f", commands.cmd_13f))
    app.add_handler(CommandHandler("holders", commands.cmd_holders))
    app.add_handler(CommandHandler("etf", commands.cmd_etf))
    app.add_handler(CommandHandler("alert", commands.cmd_alert))
    app.add_handler(CommandHandler("alerts", commands.cmd_alerts))
    app.add_handler(CommandHandler("alert_remove", commands.cmd_alert_remove))
    app.add_handler(CommandHandler("portfolio", commands.cmd_portfolio))
    app.add_handler(CommandHandler("pnl", commands.cmd_pnl))
    app.add_handler(CommandHandler("settings", commands.cmd_settings))
    app.add_handler(CommandHandler("earnings", commands.cmd_earnings))
    app.add_handler(CommandHandler("risk", commands.cmd_risk))
    app.add_handler(CommandHandler("8k", commands.cmd_8k))
    app.add_handler(CommandHandler("insiders", commands.cmd_insiders))
    app.add_handler(CommandHandler("flow", commands.cmd_flow))

    # Phase 4 macro commands
    app.add_handler(CommandHandler("macro", commands.cmd_macro))
    app.add_handler(CommandHandler("cpi", commands.cmd_cpi))
    app.add_handler(CommandHandler("fomc", commands.cmd_fomc))
    app.add_handler(CommandHandler("yields", commands.cmd_yields))

    # Stage 3 — NL intent classifier + dispatch
    #
    # /ask remains an alias of /ask_v2 after retiring V1.
    app.add_handler(CommandHandler("ask", commands.cmd_agent_v2, block=False))
    # Agent V2 handlers do not block the update queue: a "取消" sent while an
    # answer is in progress must reach the bridge before that answer ends.
    # The bridge serialises the questions of one chat itself.
    app.add_handler(CommandHandler("ask_v2", commands.cmd_agent_v2, block=False))
    app.add_handler(CommandHandler("ask_v3", commands.cmd_agent_v3, block=False))
    app.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, commands.cmd_nl, block=False)
    )

    app.add_error_handler(_error_handler)

    return app


def run_bot() -> None:
    """Start the long-polling loop. Blocks until killed."""
    app = build_application()
    logger.info("Bot starting (polling mode)...")
    app.run_polling(allowed_updates=["message"])
