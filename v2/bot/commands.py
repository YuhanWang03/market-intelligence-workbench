"""Stage 1 slash command handlers.

Stage 2 will add: /why /summary /chain /13f /settings (action commands).
Stage 3 will add: NL → intent routing.

All handlers are async (python-telegram-bot 21 is async-only). Long-running
agent work belongs in Stage 2; Stage 1 is fast SQLite-only lookups.
"""

from __future__ import annotations

import asyncio
import html
import logging
import os
from functools import wraps
from io import BytesIO

from telegram import Update
from telegram.ext import ContextTypes

from v2.bot import intent, responders, state

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Authorization — single-user MVP
# ---------------------------------------------------------------------------


def _allowed_chat_id() -> int | None:
    raw = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    try:
        return int(raw)
    except ValueError:
        return None


def authorized_only(handler):
    """Decorator: drop messages from chat_ids other than the configured owner."""

    @wraps(handler)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        allowed = _allowed_chat_id()
        chat = update.effective_chat
        if allowed is None or chat is None or chat.id != allowed:
            logger.info("Rejecting message from chat_id=%s",
                        chat.id if chat else "?")
            if chat is not None:
                await context.bot.send_message(
                    chat_id=chat.id,
                    text="❌ Not authorized — this bot is single-user.",
                )
            return
        from v2.usage_context import usage_channel
        with usage_channel('telegram'):
            return await handler(update, context)
    return wrapper


# ---------------------------------------------------------------------------
# Help / Start
# ---------------------------------------------------------------------------


_HELP_TEXT = (
    "<b>🤖 Hedge Fund Bot · 命令列表</b>\n"
    "━━━━━━━━━━━━━━━━━━━━\n"
    "<b>Watchlist 管理（Stage 1）</b>\n"
    "  /watchlist          — 查看当前关注列表\n"
    "  /add NVDA           — 添加 ticker\n"
    "  /remove TSLA        — 移除 ticker\n"
    "\n"
    "<b>分析命令</b>\n"
    "  /why NVDA           — 解释最近异动原因\n"
    "  /summary NVDA       — 7 天多维度总结\n"
    "  /chain NVDA         — 产业链邻居\n"
    "  /13f BRK            — Manager 完整组合 + 持仓变动\n"
    "  /holders NVDA       — 哪些机构持有该股票\n"
    "  /etf ARKK           — ARK 基金每日持仓 + 24h 调仓\n"
    "  /earnings AAPL      — 单股财报（下次日期 + 上次结果）\n"
    "  /earnings           — 未来 14 天财报日历（watchlist + 持仓）\n"
    "  /settings           — 查看推送阈值\n"
    "\n"
    "<b>盘中提醒</b>\n"
    "  /alert NVDA 130 above   — 突破/跌破提醒（默认 above）\n"
    "  /alerts                 — 查看未触发提醒\n"
    "  /alert_remove ID        — 删除一条提醒\n"
    "\n"
    "<b>Alpaca 账户（paper）</b>\n"
    "  /portfolio          — 当前持仓 + 现金\n"
    "  /pnl [day|week|month] — 盈亏（默认 day）\n"
    "  /risk               — 组合风险快照（集中度 / 暴露 / 回撤 / 7d 财报）\n"
    "\n"
    "<b>SEC 监控</b>\n"
    "  /8k TICKER          — 最近 30 天 8-K 申报（含 5.02 LLM 抽取）\n"
    "  /insiders TICKER [DAYS] — 内部人交易摘要（默认 90 天）\n"
    "\n"
    "<b>自然语言</b>\n"
    "  直接发问即走 Agent V2：规划、调工具、带证据引用回答，追问接上文\n"
    "  例：「我的持仓中哪只跌的最狠」「为什么跌这么狠」「ARM 从高点为什么跌了这么多」\n"
    "  网页证据默认允许（服务端须开 AGENT_V2_WEB_ENABLED）；问题里加 --noweb 则本条不用网页\n"
    "  /ask 问题         — Agent V2（/ask_v2 的别名）\n"
    "  /ask_v2 [--web|--noweb] 问题 — 与直接发问相同\n"
    "\n"
    "<i>本 bot 受单用户授权——只响应所有者的消息。</i>"
)


@authorized_only
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_html(_HELP_TEXT)


@authorized_only
async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_html(_HELP_TEXT)


# ---------------------------------------------------------------------------
# Watchlist commands
# ---------------------------------------------------------------------------


@authorized_only
async def cmd_watchlist(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    items = state.watchlist_list()
    if not items:
        await update.message.reply_html(
            "<b>📋 Watchlist 为空</b>\n"
            "用 <code>/add TICKER</code> 添加股票。"
        )
        return

    lines = [f"<b>📋 Watchlist ({len(items)})</b>", "━━━━━━━━━━━━━━━━━━━━"]
    for it in items:
        added = it["added_at"][:10]  # YYYY-MM-DD
        note = f" — <i>{html.escape(it['note'])}</i>" if it.get("note") else ""
        lines.append(f"• <b>{html.escape(it['ticker'])}</b>  "
                     f"<code>{added}</code>{note}")
    await update.message.reply_html("\n".join(lines))


@authorized_only
async def cmd_add(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_html(
            "用法：<code>/add TICKER</code>\n"
            "例：<code>/add NVDA</code>"
        )
        return

    ticker = context.args[0].upper()
    try:
        added = state.watchlist_add(ticker)
    except ValueError as exc:
        await update.message.reply_html(
            f"❌ Invalid ticker <code>{html.escape(ticker)}</code>: {exc}"
        )
        return

    if not added:
        await update.message.reply_html(
            f"ℹ️ <b>{html.escape(ticker)}</b> already in watchlist."
        )
        return

    count = len(state.watchlist_list())
    await update.message.reply_html(
        f"✅ Added <b>{html.escape(ticker)}</b> · watchlist size = {count}"
    )


@authorized_only
async def cmd_remove(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_html(
            "用法：<code>/remove TICKER</code>"
        )
        return

    ticker = context.args[0].upper()
    removed = state.watchlist_remove(ticker)
    if not removed:
        await update.message.reply_html(
            f"ℹ️ <b>{html.escape(ticker)}</b> not in watchlist."
        )
        return

    count = len(state.watchlist_list())
    await update.message.reply_html(
        f"🗑 Removed <b>{html.escape(ticker)}</b> · watchlist size = {count}"
    )


# ---------------------------------------------------------------------------
# Stage 2 — action commands (FD + LLM heavy)
# ---------------------------------------------------------------------------


async def _run_blocking(func, *args):
    """Run a synchronous responder in the default thread executor so we don't
    block the bot's event loop while FD / Tavily / DeepSeek calls are in flight."""
    from contextvars import copy_context
    return await asyncio.get_running_loop().run_in_executor(None, copy_context().run, func, *args)


@authorized_only
async def cmd_why(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_html("用法：<code>/why TICKER</code>")
        return
    ticker = context.args[0].upper()
    placeholder = await update.message.reply_html(
        f"🔍 正在分析 <b>{html.escape(ticker)}</b> 最近异动...\n"
        "<i>预计 15-25 秒</i>"
    )
    result = await _run_blocking(responders.explain_move, ticker)
    await placeholder.edit_text(result, parse_mode="HTML",
                                 disable_web_page_preview=True)


@authorized_only
async def cmd_summary(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_html("用法：<code>/summary TICKER</code>")
        return
    ticker = context.args[0].upper()
    placeholder = await update.message.reply_html(
        f"📊 汇总 <b>{html.escape(ticker)}</b> 多维度数据..."
    )
    result = await _run_blocking(responders.summary, ticker)
    await placeholder.edit_text(result, parse_mode="HTML",
                                 disable_web_page_preview=True)


@authorized_only
async def cmd_chain(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_html("用法：<code>/chain TICKER</code>")
        return
    ticker = context.args[0].upper()
    placeholder = await update.message.reply_html(
        f"🕸 正在挖掘 <b>{html.escape(ticker)}</b> 产业链...\n"
        "<i>预计 30-45 秒</i>"
    )
    result = await _run_blocking(responders.chain, ticker)
    await placeholder.edit_text(result, parse_mode="HTML",
                                 disable_web_page_preview=True)


@authorized_only
async def cmd_13f(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_html(
            "用法：<code>/13f MANAGER</code>\n"
            "示例：<code>/13f brk</code> · <code>/13f burry</code> · "
            "<code>/13f ark</code>"
        )
        return
    name = " ".join(context.args)
    placeholder = await update.message.reply_html(
        f"🏛 拉取 <b>{html.escape(name)}</b> 最新 13F..."
    )
    messages = await _run_blocking(responders.institutional_quick, name)

    if not messages:
        await placeholder.edit_text("⚠️ 无返回", parse_mode="HTML")
        return

    # First message replaces the placeholder; rest are sent as new messages
    await placeholder.edit_text(messages[0], parse_mode="HTML",
                                 disable_web_page_preview=True)
    for msg in messages[1:]:
        await update.message.reply_html(msg, disable_web_page_preview=True)


@authorized_only
async def cmd_holders(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_html(
            "用法：<code>/holders TICKER</code>\n"
            "示例：<code>/holders NVDA</code>"
        )
        return
    ticker = context.args[0].upper()
    placeholder = await update.message.reply_html(
        f"🏛 查询 <b>{html.escape(ticker)}</b> 的机构持有人分布..."
    )
    result = await _run_blocking(responders.holders, ticker)
    await placeholder.edit_text(result, parse_mode="HTML",
                                 disable_web_page_preview=True)


@authorized_only
async def cmd_etf(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_html(
            "用法：<code>/etf SYMBOL</code>\n"
            "示例：<code>/etf ARKK</code> · <code>/etf ARKG</code>"
        )
        return
    symbol = context.args[0].upper()
    placeholder = await update.message.reply_html(
        f"📈 拉取 <b>{html.escape(symbol)}</b> 最新每日持仓..."
    )
    result = await _run_blocking(responders.etf_view, symbol)
    await placeholder.edit_text(result, parse_mode="HTML",
                                 disable_web_page_preview=True)


@authorized_only
async def cmd_alert(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if len(context.args) < 2:
        await update.message.reply_html(
            "用法：<code>/alert TICKER PRICE [above|below]</code>\n"
            "示例：<code>/alert NVDA 130 above</code> · "
            "<code>/alert AAPL 200 below</code>"
        )
        return
    ticker = context.args[0].upper()
    try:
        target_price = float(context.args[1])
    except ValueError:
        await update.message.reply_html(
            f"<b>🚫 价格必须是数字：</b> <code>{html.escape(context.args[1])}</code>"
        )
        return
    direction = (context.args[2].lower() if len(context.args) >= 3 else "above")
    result = await _run_blocking(
        responders.alert_set, ticker, target_price, direction,
    )
    await update.message.reply_html(result)


@authorized_only
async def cmd_alerts(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    result = await _run_blocking(responders.alert_list_view)
    await update.message.reply_html(result)


@authorized_only
async def cmd_alert_remove(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_html("用法：<code>/alert_remove ID</code>")
        return
    try:
        alert_id = int(context.args[0])
    except ValueError:
        await update.message.reply_html(
            f"<b>🚫 ID 必须是数字：</b> <code>{html.escape(context.args[0])}</code>"
        )
        return
    result = await _run_blocking(responders.alert_remove_view, alert_id)
    await update.message.reply_html(result)


@authorized_only
async def cmd_portfolio(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    placeholder = await update.message.reply_html("💼 拉取 Alpaca 账户...")
    result = await _run_blocking(responders.portfolio_view)
    await placeholder.edit_text(result, parse_mode="HTML",
                                 disable_web_page_preview=True)


@authorized_only
async def cmd_pnl(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/pnl [day | week | month]`` — default ``day`` matches the
    pre-Phase-2 behavior exactly. ``week`` / ``month`` route to the
    period responder which reads portfolio_history.
    """
    args = context.args or []
    period = (args[0].strip().lower() if args else "day")

    if period == "day":
        placeholder = await update.message.reply_html("📊 计算盈亏...")
        result = await _run_blocking(responders.pnl_view)
    elif period in ("week", "month"):
        label = "本周" if period == "week" else "本月"
        placeholder = await update.message.reply_html(
            f"📊 计算{label}盈亏..."
        )
        result = await _run_blocking(
            responders.pnl_period, {"period": period},
        )
    else:
        await update.message.reply_html(
            f"<b>🚫 未知周期：</b> <code>{html.escape(period)}</code>\n"
            "用法：<code>/pnl</code> · <code>/pnl day</code> · "
            "<code>/pnl week</code> · <code>/pnl month</code>"
        )
        return

    await placeholder.edit_text(result, parse_mode="HTML",
                                 disable_web_page_preview=True)


@authorized_only
async def cmd_risk(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/risk`` — real-time portfolio risk card.

    Builds a fresh RiskReport via Alpaca (positions + portfolio_history)
    and yfinance (held-position earnings ≤ 7d). Read-only: no archive
    write, no priority computed."""
    placeholder = await update.message.reply_html(
        "💼 拉取组合风险快照...\n<i>预计 5-10 秒</i>"
    )
    result = await _run_blocking(responders.risk_view, {})
    await placeholder.edit_text(result, parse_mode="HTML",
                                 disable_web_page_preview=True)


@authorized_only
async def cmd_settings(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = await _run_blocking(responders.settings_view)
    await update.message.reply_html(text)


@authorized_only
async def cmd_earnings(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/earnings [TICKER]`` — single-ticker card, or 14-day calendar.

    With no args: returns the calendar across watchlist ∪ Alpaca holdings.
    With a ticker: returns the next-release + last-filing card.
    """
    args = context.args or []
    if not args:
        placeholder = await update.message.reply_html("📅 拉取财报日历...")
        result = await _run_blocking(
            responders.earnings_calendar, {"days_horizon": 14},
        )
    else:
        ticker = args[0].upper()
        placeholder = await update.message.reply_html(
            f"📞 查询 <b>{html.escape(ticker)}</b> 财报..."
        )
        result = await _run_blocking(
            responders.earnings_view, {"ticker": ticker},
        )
    await placeholder.edit_text(
        result, parse_mode="HTML", disable_web_page_preview=True,
    )


@authorized_only
async def cmd_8k(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/8k TICKER`` — last-30-days 8-K summary for one ticker.

    Reuses the cron's 5.02 LLM extractor. May take 5-10s when 5.02 is
    present (LLM call); empty/no-5.02 results return in <2s.
    """
    if not context.args:
        await update.message.reply_html(
            "用法：<code>/8k TICKER</code>\n例：<code>/8k AAPL</code>"
        )
        return
    ticker = context.args[0].upper()
    placeholder = await update.message.reply_html(
        f"📋 拉取 <b>{html.escape(ticker)}</b> 最近 30 天 8-K...\n"
        "<i>预计 5-10 秒（含 5.02 LLM 抽取）</i>"
    )
    result = await _run_blocking(
        responders.eight_k_view, {"ticker": ticker},
    )
    await placeholder.edit_text(
        result, parse_mode="HTML", disable_web_page_preview=True,
    )


async def _send_moneyflow(update, placeholder, caption: str, chart, prefix: str = "") -> None:
    """Deliver a money-flow view: photo (card as caption) when a chart was
    rendered, else fall back to editing the placeholder text. ``prefix`` is
    an optional HTML routing chip (NL path)."""
    body = prefix + caption
    if chart:
        await update.message.reply_photo(
            photo=BytesIO(chart), caption=body, parse_mode="HTML",
        )
        try:
            await placeholder.delete()
        except Exception:
            pass
    else:
        await placeholder.edit_text(
            body, parse_mode="HTML", disable_web_page_preview=True,
        )


@authorized_only
async def cmd_flow(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/flow TICKER`` — on-demand money-flow divergence (CMF/RSI vs price).

    Always shows the three-axis read; adds an accumulation/distribution
    verdict + LLM 多空 narration when a divergence fires. ~3-6s (one FD
    price call + optional DeepSeek narration).
    """
    if not context.args:
        await update.message.reply_html(
            "用法：<code>/flow TICKER</code>\n例：<code>/flow MSFT</code>"
        )
        return
    ticker = context.args[0].upper()
    placeholder = await update.message.reply_html(
        f"📊 分析 <b>{html.escape(ticker)}</b> 资金流背离...\n"
        "<i>价格 / CMF / RSI 三轴（预计 3-6 秒）</i>"
    )
    caption, chart = await _run_blocking(
        responders.moneyflow_view, {"ticker": ticker},
    )
    await _send_moneyflow(update, placeholder, caption, chart)


@authorized_only
async def cmd_macro(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/macro`` — real-time macro dashboard.

    No args. Pulls live VIX/yields/calendar via build_macro_snapshot
    + release_calendar lookups. Read-only: no archive write, no
    priority computed.
    """
    placeholder = await update.message.reply_html(
        "🌐 拉取宏观 dashboard...\n"
        "<i>VIX / 收益率 / 最近 release (预计 3-5 秒)</i>"
    )
    result = await _run_blocking(responders.macro_view, {})
    await placeholder.edit_text(
        result, parse_mode="HTML", disable_web_page_preview=True,
    )


_RELEASE_LABELS = {
    "CPI": "📈 CPI · 通胀",
    "PCE": "📈 PCE · 通胀",
    "NFP": "📈 NFP · 就业",
    "GDP": "📈 GDP · 经济增长",
    "PPI": "📈 PPI · 生产者物价",
    "Claims": "📈 Initial Claims · 失业金申请",
    "FOMC": "🏛 FOMC · Fed 利率决议",
}


async def _release_check_handler(
    update: Update, release_type: str, *,
    needs_llm: bool = True,
) -> None:
    """Backend shared by /cpi /pce /nfp /gdp /ppi /claims /fomc.

    ``release_type`` MUST be one of the closed enum values used by
    the responder ("CPI" / "PCE" / "NFP" / "GDP" / "PPI" / "Claims"
    / "FOMC"). FOMC takes longer because the responder runs Python
    statement diff + Tavily aggregate (Layer 3 path).
    """
    label = _RELEASE_LABELS.get(release_type, release_type)
    eta_blurb = (
        "<i>预计 8-15 秒 (含 FRED + LLM template-fill + Tavily)</i>"
        if needs_llm else "<i>预计 5-8 秒 (FRED + LLM)</i>"
    )
    placeholder = await update.message.reply_html(
        f"{label} 拉取中...\n{eta_blurb}"
    )
    result = await _run_blocking(
        responders.release_check, {"release_type": release_type.lower()},
    )
    await placeholder.edit_text(
        result, parse_mode="HTML", disable_web_page_preview=True,
    )


@authorized_only
async def cmd_cpi(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/cpi`` — latest CPI release with summarizer output."""
    await _release_check_handler(update, "CPI")


@authorized_only
async def cmd_fomc(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/fomc`` — most recent FOMC decision (statement diff + SEP +
    Tavily sell-side aggregate). Layer 3 path: no LLM hawkish/dovish
    verdict."""
    await _release_check_handler(update, "FOMC", needs_llm=False)


@authorized_only
async def cmd_yields(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/yields`` — current Treasury curve + 10Y-2Y / 10Y-3M spreads.

    Reuses the macro_view dashboard (the yields panel is the
    information operators want from /yields). Stage 5 may split this
    into a dedicated narrower card.
    """
    placeholder = await update.message.reply_html(
        "🏛 拉取收益率曲线...\n<i>FRED canonical EOD (预计 2-4 秒)</i>"
    )
    result = await _run_blocking(responders.macro_view, {})
    await placeholder.edit_text(
        result, parse_mode="HTML", disable_web_page_preview=True,
    )


@authorized_only
async def cmd_insiders(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/insiders TICKER [DAYS]`` — Form 4 summary for one ticker.

    Optional days arg: 7-365, default 90. Pure-Python aggregation
    (no LLM), typically returns in <2s.

    Examples:
        /insiders NVDA
        /insiders NVDA 30
    """
    if not context.args:
        await update.message.reply_html(
            "用法：<code>/insiders TICKER [DAYS]</code>\n"
            "例：<code>/insiders NVDA</code> · <code>/insiders NVDA 30</code>"
        )
        return
    ticker = context.args[0].upper()
    args_dict: dict = {"ticker": ticker}
    if len(context.args) >= 2:
        try:
            days = int(context.args[1])
            args_dict["days_back"] = days
        except ValueError:
            await update.message.reply_html(
                f"<b>🚫 DAYS 必须是数字：</b> "
                f"<code>{html.escape(context.args[1])}</code>"
            )
            return
    placeholder = await update.message.reply_html(
        f"📥 拉取 <b>{html.escape(ticker)}</b> 内部人交易..."
    )
    result = await _run_blocking(responders.insider_view, args_dict)
    await placeholder.edit_text(
        result, parse_mode="HTML", disable_web_page_preview=True,
    )


# ---------------------------------------------------------------------------
# Stage 3 — NL → Intent classifier and router
# ---------------------------------------------------------------------------


@authorized_only
async def cmd_agent_v2(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Explicit opt-in to Agent V2; ``--web`` is a second network consent."""

    from v2.bot.agent_v2_bridge import handle_agent_v2, split_web_consent

    text, allow_web = split_web_consent(" ".join(context.args or []))
    if not text:
        await update.message.reply_html(
            "用法：<code>/ask_v2 [--web|--noweb] 问题</code>\n"
            "例：<code>/ask_v2 比较 NVDA 和 AMD 的风险</code>"
        )
        return

    await handle_agent_v2(update, context, text, allow_web=allow_web)


@authorized_only
async def cmd_agent_v3(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    from v2.bot.agent_v3_bridge import handle_agent_v3
    await handle_agent_v3(update, context)


@authorized_only
async def cmd_nl(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Plain messages use Agent V2; /ask_v3 explicitly selects Agent V3."""
    from v2.bot.agent_v2_bridge import handle_agent_v2, split_web_consent

    text = (update.message.text or "").strip()
    question, allow_web = split_web_consent(text)
    if question:
        await handle_agent_v2(update, context, question, allow_web=allow_web)


def _recent_anomalies() -> str:
    """Pull the last 5 anomaly pushes from archive.db."""
    import sqlite3
    from pathlib import Path

    db = Path(__file__).resolve().parents[2] / "data" / "archive.db"
    if not db.exists():
        return "<i>archive.db 不存在</i>"

    try:
        conn = sqlite3.connect(str(db))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """SELECT ts, tickers, text_html FROM pushes
               WHERE agent='anomaly' ORDER BY id DESC LIMIT 5"""
        ).fetchall()
        conn.close()
    except sqlite3.Error as exc:
        return f"<i>archive 查询失败: {html.escape(str(exc))}</i>"

    if not rows:
        return "<i>近期 archive 中无异动记录</i>"

    lines = ["<b>🚨 近期异动（archive 最新 5 条）</b>", ""]
    for r in rows:
        ts = (r["ts"] or "")[:19].replace("T", " ")
        tickers = r["tickers"] or "?"
        lines.append(f"  • <code>{html.escape(ts)}</code> · "
                     f"<b>{html.escape(tickers)}</b>")
    lines.append("")
    lines.append("<i>使用 <code>/why TICKER</code> 重新解读任意一条。</i>")
    return "\n".join(lines)
