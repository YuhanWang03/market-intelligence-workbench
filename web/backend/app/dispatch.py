"""Map a classified NL intent to an HTML reply (+ optional chart).

Reuses the production v2 responders with the *exact* call conventions the
Telegram bot uses (v2/bot/commands.py:cmd_nl), so the web chat and the bot
return byte-identical cards. Blocking by design — the route runs it in a
threadpool.
"""

from __future__ import annotations

import base64
import html as _html


def _err(msg: str) -> dict:
    return {"html": f"<b>⚠️ {_html.escape(msg)}</b>"}


# Same shape v2.bot.intent.classify() returns, so dispatch() consumes either.
_PARSED_BASE = {
    "intent": "unknown", "ticker": "", "manager": "", "etf": "",
    "period": "", "release_type": "", "days_back": 0, "days_horizon": 0,
}


def parse_slash(text: str) -> dict | None:
    """Map a Telegram-style slash command to a parsed-intent dict.

    Returns None for non-slash text (caller falls back to the LLM
    classifier) or an unknown command.
    """
    text = text.strip()
    if not text.startswith("/"):
        return None
    parts = text[1:].split()
    if not parts:
        return None
    cmd = parts[0].lower()
    a1 = parts[1] if len(parts) > 1 else ""
    a2 = parts[2] if len(parts) > 2 else ""

    def p(**kw) -> dict:
        return {**_PARSED_BASE, **kw}

    table = {
        "flow":      lambda: p(intent="moneyflow_view", ticker=a1.upper()),
        "why":       lambda: p(intent="explain_move", ticker=a1.upper()),
        "summary":   lambda: p(intent="summary", ticker=a1.upper()),
        "chain":     lambda: p(intent="chain", ticker=a1.upper()),
        "13f":       lambda: p(intent="thirteen_f", manager=a1.lower()),
        "holders":   lambda: p(intent="holders_view", ticker=a1.upper()),
        "etf":       lambda: p(intent="etf_view", etf=a1.upper()),
        "portfolio": lambda: p(intent="portfolio_view"),
        "pnl":       lambda: p(intent="pnl_period", period=a1.lower()) if a1 else p(intent="pnl_view"),
        "risk":      lambda: p(intent="risk_view"),
        "earnings":  lambda: p(intent="earnings_view", ticker=a1.upper()) if a1 else p(intent="earnings_calendar"),
        "8k":        lambda: p(intent="eight_k_view", ticker=a1.upper()),
        "insiders":  lambda: p(intent="insider_view", ticker=a1.upper(),
                               days_back=int(a2) if a2.isdigit() else 0),
        "macro":     lambda: p(intent="macro_view"),
        "cpi":       lambda: p(intent="release_check", release_type="cpi"),
        "fomc":      lambda: p(intent="release_check", release_type="fomc"),
        "yields":    lambda: p(intent="macro_view"),
        "watchlist": lambda: p(intent="watchlist_view"),
        "technical": lambda: p(intent="technical_view", ticker=a1.upper(),
                                period=(a2.upper() or "3M")),
    }
    fn = table.get(cmd)
    return fn() if fn else None


def dispatch(parsed: dict) -> dict:
    """parsed = output of v2.bot.intent.classify(). Returns a dict with:
    html (str), optional extra_html (list[str]), optional chart_b64 (str)."""
    from v2.bot import responders, state
    from v2.research import compat as research_compat

    intent = parsed.get("intent", "unknown")
    ticker = parsed.get("ticker", "")
    manager = parsed.get("manager", "")
    etf = parsed.get("etf", "")
    period = parsed.get("period", "")
    release_type = parsed.get("release_type", "")
    days_back = parsed.get("days_back", 0)
    days_horizon = parsed.get("days_horizon", 0)

    try:
        if intent == "explain_move":
            return _err("无法识别 ticker") if not ticker else {"html": responders.explain_move(ticker)}
        if intent == "summary":
            return _err("无法识别 ticker") if not ticker else {"html": responders.summary(ticker)}
        if intent == "chain":
            if not ticker:
                return _err("无法识别 ticker")
            formatted, data = research_compat.chain(ticker)
            return {"html": formatted, "data": data}
        if intent == "thirteen_f":
            if not manager:
                return _err("无法识别 manager 名称")
            msgs = responders.institutional_quick(manager)
            return {"html": msgs[0] if msgs else "⚠️ 无返回", "extra_html": msgs[1:]}
        if intent == "holders_view":
            return _err("无法识别 ticker") if not ticker else {"html": research_compat.holders(ticker)}
        if intent == "etf_view":
            return {"html": responders.etf_view(etf or "ARKK")}
        if intent == "watchlist_view":
            items = state.watchlist_list()
            if not items:
                return {"html": "<b>📋 Watchlist 为空</b>"}
            tks = ", ".join(_html.escape(it["ticker"]) for it in items)
            return {"html": f"<b>📋 Watchlist ({len(items)})</b>\n{tks}"}
        if intent == "watchlist_add":
            if not ticker:
                return _err("没说要加哪只股票")
            try:
                added = state.watchlist_add(ticker)
            except ValueError:
                return _err(f"Invalid ticker: {ticker}")
            return {"html": f"✅ Added <b>{_html.escape(ticker)}</b>" if added
                    else f"ℹ️ <b>{_html.escape(ticker)}</b> 已在 watchlist"}
        if intent == "watchlist_remove":
            if not ticker:
                return _err("没说要移除哪只股票")
            removed = state.watchlist_remove(ticker)
            return {"html": f"🗑 Removed <b>{_html.escape(ticker)}</b>" if removed
                    else f"ℹ️ <b>{_html.escape(ticker)}</b> 不在 watchlist"}
        if intent == "settings":
            return {"html": responders.settings_view()}
        if intent == "portfolio_view":
            return {"html": responders.portfolio_view()}
        if intent == "pnl_view":
            return {"html": responders.pnl_view()}
        if intent == "pnl_period":
            return {"html": responders.pnl_period({"period": period} if period else {})}
        if intent == "risk_view":
            return {"html": responders.risk_view({})}
        if intent == "earnings_view":
            return _err("无法识别 ticker") if not ticker else {"html": research_compat.earnings(ticker)}
        if intent == "earnings_calendar":
            return {"html": responders.earnings_calendar({"days_horizon": days_horizon} if days_horizon else {})}
        if intent == "eight_k_view":
            return _err("没说要查哪只股票的 8-K") if not ticker else {"html": responders.eight_k_view({"ticker": ticker})}
        if intent == "insider_view":
            if not ticker:
                return _err("没说要查哪只股票的内部人交易")
            args = {"ticker": ticker}
            if days_back:
                args["days_back"] = days_back
            return {"html": responders.insider_view(args)}
        if intent == "macro_view":
            return {"html": research_compat.macro()}
        if intent == "release_check":
            return {"html": responders.release_check({"release_type": release_type} if release_type else {})}
        if intent == "moneyflow_view":
            if not ticker:
                return _err("没说要看哪只股票的资金流")
            caption, chart = responders.moneyflow_view({"ticker": ticker})
            out: dict = {"html": caption}
            if chart:
                out["chart_b64"] = base64.b64encode(chart).decode("ascii")
            return out
        if intent == "technical_view":
            if not ticker:
                return _err("无法识别 ticker")
            from app.routers.portfolio import _fetch_price_history

            range_key = period.upper() if period else "3M"
            payload = _fetch_price_history(ticker, range_key, False)
            analysis = payload["technicalAnalysis"]
            quote = payload["quote"]
            def fmt(value, digits=2):
                return "—" if value is None else f"{value:.{digits}f}"
            def fmt_pct(value):
                return "—" if value is None else f"{value * 100:+.2f}%"
            def zones(items):
                return " · ".join(
                    f"{item['id']} {fmt(item['lower'])}–{fmt(item['upper'])}"
                    f"（score {fmt(item['score'], 1)} / {item['touchCount']}次）"
                    for item in items[:3]
                ) or "暂无可靠区域"
            session_names = {
                "PRE_MARKET": "盘前", "REGULAR": "盘中",
                "AFTER_HOURS": "盘后", "CLOSED": "已收盘",
            }
            html = (
                f"<b>📈 {ticker} 结构化技术分析 · {analysis['timeframe']}</b><br>"
                f"最新价 ${fmt(analysis['currentPrice'])} · "
                f"{session_names.get(analysis['marketSession'], analysis['marketSession'])}"
                f"{' · 延迟行情' if analysis['isDelayed'] else ''}<br>"
                f"SMA20 {fmt(analysis['SMA20'])} · SMA50 {fmt(analysis['SMA50'])} · "
                f"SMA200 {fmt(analysis['SMA200'])}<br>"
                f"本K {fmt_pct(analysis.get('barReturn'))} · "
                f"当日 {fmt_pct(analysis.get('dayReturn'))} · "
                f"前收 ${fmt(analysis.get('previousRegularClose'))}<br>"
                f"ATR14 {fmt(analysis['ATR14'])} · 量比 {fmt(analysis['volumeRatio'])}x · "
                f"趋势 {analysis['trend']}<br>"
                f"<b>支撑：</b>{zones(analysis['supports'])}<br>"
                f"<b>阻力：</b>{zones(analysis['resistances'])}<br>"
                f"突破状态：{analysis['breakoutStatus']}<br>"
                f"<small>来源 {quote['source']} · {payload['adjustmentMode']} · "
                f"指标按 {analysis['timeframe']} K线计算，不从图片推断</small>"
            )
            return {"html": html, "technical_analysis": analysis}
        return {"html": "🤔 暂不支持这个查询，换个说法试试，或用左侧面板。"}
    except Exception as exc:  # never 500 the chat on a responder error
        return {"html": f"❌ Error: <code>{_html.escape(str(exc))}</code>"}
