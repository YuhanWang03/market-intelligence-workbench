"""Natural-language intent classifier (Stage 3).

User sends free-form text → DeepSeek (temperature=0) maps to a closed enum
of intents → bot routes to the same responders as the slash commands.

Critical design constraint: the LLM only DECIDES which tool to call. It
does not generate answers. Outputs that don't parse to a valid enum value
become "unknown" — guaranteed bounded behavior.

If the user wanted DeepSeek to write financial analysis from imagination,
they could ask DeepSeek directly. The point of this bot is grounded,
multi-source, verified output.
"""

from __future__ import annotations

import json
import logging
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from v2.data.metered import ChatDeepSeek

logger = logging.getLogger(__name__)

IntentName = Literal[
    "explain_move",
    "summary",
    "chain",
    "thirteen_f",
    "holders_view",
    "etf_view",
    "watchlist_view",
    "watchlist_add",
    "watchlist_remove",
    "settings",
    "find_anomalies",
    "alert_set",
    "alert_list",
    "portfolio_view",
    "pnl_view",
    "earnings_view",
    "earnings_calendar",
    "risk_view",
    "pnl_period",
    "eight_k_view",
    "insider_view",
    "macro_view",
    "release_check",
    "moneyflow_view",
    "unknown",
]

_VALID_INTENTS = {
    "explain_move",
    "summary",
    "chain",
    "thirteen_f",
    "holders_view",
    "etf_view",
    "watchlist_view",
    "watchlist_add",
    "watchlist_remove",
    "settings",
    "find_anomalies",
    "alert_set",
    "alert_list",
    "portfolio_view",
    "pnl_view",
    "earnings_view",
    "earnings_calendar",
    "risk_view",
    "pnl_period",
    "eight_k_view",
    "insider_view",
    "macro_view",
    "release_check",
    "moneyflow_view",
    "unknown",
}

# Closed enum for pnl_period.period. LLM outputs outside this set are
# silently coerced to "day" (the default behavior, matches /pnl no-arg).
_VALID_PNL_PERIODS = frozenset({"day", "week", "month"})

# Closed enum for release_check.release_type (Phase 4 Stage 4). Values
# outside this whitelist coerce to "cpi" — same posture as period.
_VALID_RELEASE_TYPES = frozenset({"cpi", "pce", "nfp", "gdp", "ppi", "claims", "fomc"})

# Bounds for insider_view.days_back. Default 90; user can override 7-365.
_INSIDER_DAYS_BACK_MIN = 7
_INSIDER_DAYS_BACK_MAX = 365
_INSIDER_DAYS_BACK_DEFAULT = 90


_SYSTEM_PROMPT = (
    "你是一个股票分析助手的【意图分类器】。\n"
    "把用户的话归类到下列**固定 23 个 intent 之一**，并提取参数。"
    "你不要回答问题，只做分类。\n"
    "\n"
    "【支持的 intents】\n"
    "- explain_move: 用户想知道某只股票最近**为什么**涨/跌/异动（强调原因/催化剂）。"
    "例：「NVDA 为什么涨」「苹果今天为什么跌」「特斯拉异动原因」「英伟达咋了」\n"
    "- summary: 用户想要某只股票的**综合概览 / 分析**（价格+财务+财报+新闻）。"
    "例：「分析一下 NVDA」「看看 AAPL」「NVDA 怎么样」「英伟达综合分析」"
    "「介绍下特斯拉」「AMD 基本面」\n"
    "- chain: 用户想找某只股票的产业链/上下游/同业相关股\n"
    "- thirteen_f: 用户问某个机构 manager 最新持仓（巴菲特/Burry/Ark/Pershing/Citadel 等）\n"
    "- holders_view: 用户问某只股票被哪些机构持有 / 谁持仓 / 哪些大佬买了\n"
    "- etf_view: 用户问 ARK 系列 ETF（ARKK/ARKQ/ARKG/ARKW/ARKF）的最新每日持仓\n"
    "- watchlist_view: 用户想看自己的关注列表\n"
    "- watchlist_add: 用户想添加股票到 watchlist\n"
    "- watchlist_remove: 用户想从 watchlist 移除股票\n"
    "- settings: 用户想看推送阈值设置\n"
    "- find_anomalies: 用户想了解最近市场异动情况\n"
    "- alert_set: 用户想设置价格提醒 / 突破提醒 / 跌破提醒（含 ticker + 价格 + 方向）\n"
    "- alert_list: 用户想查看已设的价格提醒\n"
    "- portfolio_view: 用户想看自己的 Alpaca 账户当前持仓\n"
    "- pnl_view: 用户想看自己的账户盈亏 / 资产总额 / 当日 P&L\n"
    "- earnings_view: 用户问某只股票的财报情况（下次日期 / 上次结果 / surprise 等）。"
    "例：「AAPL 什么时候发财报」「苹果上次财报怎么样」「NVDA Q3 财报多少」「TSLA 财报日期」\n"
    "- earnings_calendar: 用户问 watchlist / 持仓 / 未来 N 天的财报日历。"
    "例：「下周谁要发财报」「我的持仓哪些要发财报」「未来 14 天财报」「这周财报安排」\n"
    "- risk_view: 用户想看组合的风险全景：集中度 / 行业暴露 / 回撤 / 近期财报风险。"
    "例：「我的组合风险」「组合集中度怎么样」「我对哪个 sector 暴露最多」"
    "「组合 drawdown」「我持仓里有几只快出财报」\n"
    "- pnl_period: 用户问指定周期的 P&L。例：「这周亏了多少」「本月赚了多少」"
    "「上周 pnl」「月度收益」「过去一个月赚了多少」\n"
    "- eight_k_view: 用户想看某只股票的 SEC 8-K 申报历史（最近 30 天）。"
    "例：「AAPL 最近有什么 8-K」「NVDA 8-K」「苹果有发什么 SEC 公告」"
    "「特斯拉最近的 8-K 申报」「META 8K」\n"
    "- insider_view: 用户想看某只股票的内部人交易（Form 4）。"
    "例：「NVDA 内部人交易」「苹果高管买卖」「TSLA insider trading」"
    "「META 过去 30 天 Form 4」「AMZN 高管最近有没有买入」「ARM insider」\n"
    "- macro_view: 用户想看宏观 dashboard 综合状态（VIX / 收益率 / 最近 release）。"
    "例：「宏观怎么样」「macro」「市场环境」「今天 macro 状态」「what's the macro picture」\n"
    "- release_check: 用户想看某类宏观 release 的最新数据（CPI / PCE / NFP / GDP / PPI / Claims / FOMC）。"
    "例：「最近 CPI」「上次 FOMC」「NFP 数据」「PCE 通胀」「GDP 怎么样」"
    "「claims 失业」「最近的 Fed 决议」\n"
    "- moneyflow_view: 用户想看某只股票的资金流入/流出与股价是否背离"
    "（吸筹 / 派发 / 洗盘 / 出货 / 主力资金 / 量价背离 / CMF / 资金面）。"
    "例：「MSFT 资金流怎么样」「微软是不是有主力在吸筹」「美光是不是在出货」"
    "「NVDA 资金流入还是流出」「苹果量价背离吗」「特斯拉资金面」\n"
    "- unknown: 都不匹配 / 含糊不清 / 与股票无关\n"
    "\n"
    "【参数提取规则】\n"
    "- ticker: 美股 ticker（1-5 大写字母）。如果用户用公司名，映射到 ticker"
    "（如 苹果→AAPL、英伟达→NVDA、微软→MSFT、谷歌→GOOGL、特斯拉→TSLA、"
    "Meta→META、亚马逊→AMZN、AMD→AMD、Palantir→PLTR、Snowflake→SNOW 等）\n"
    "- manager: 13F manager 别名，用小写。"
    "支持: brk/berkshire/buffett, burry/scion, ackman/pershing, einhorn/greenlight, "
    "renaissance/rentech, twosigma, deshaw/shaw, citadel, coatue, ark/cathie/wood\n"
    "- etf: 当 intent=etf_view 时，提取 ETF 符号（仅大写）。"
    "支持 ARKK / ARKG / ARKW / ARKF（ARKQ 暂不可用）。"
    "用户说『ARK 创新基金』→ ARKK，『ARK 基因组』→ ARKG，"
    "『Cathie 今天买啥』默认 → ARKK。\n"
    "- target_price: 当 intent=alert_set 时，提取目标价（数字，不带美元符号）\n"
    "- direction: 当 intent=alert_set 时，提取方向（仅 'above' 或 'below'）。"
    "「突破」「涨到」「站上」→ above；「跌破」「跌到」「跌穿」→ below\n"
    "- days_horizon: 当 intent=earnings_calendar 时，提取用户问的天数窗口（整数）。"
    "「下周」→ 7，「未来两周」/「下两周」→ 14，「这周」→ 5，「下个月」→ 30。"
    "用户没指定时输出 0（responder 默认 14）。\n"
    "- period: 当 intent=pnl_period 时，**仅** 'day'、'week'、'month' 三选一。"
    "「今天」「日内」「当日」→ day；「本周」「这周」「上周」「周度」→ week；"
    "「本月」「这个月」「月度」「过去一个月」→ month。"
    "其他任何值（包括 quarter / ytd / all 等）一律输出空字符串。"
    "用户没指定时也输出空字符串（responder 默认 day）。\n"
    "- days_back: 当 intent=insider_view 时，提取用户问的回溯天数（整数）。"
    "「过去 30 天」/「最近 30 天」→ 30；「过去三个月」→ 90；「半年」→ 180；"
    "「过去一年」→ 365。范围 7-365；用户没指定时输出 0（responder 默认 90）。\n"
    "- release_type: 当 intent=release_check 时，**仅** 'cpi'、'pce'、'nfp'、"
    "'gdp'、'ppi'、'claims'、'fomc' 七选一（全小写）。"
    "「CPI」「通胀数据」「物价指数」→ cpi；「PCE」「核心 PCE」→ pce；"
    "「NFP」「就业」「非农」「payrolls」→ nfp；「GDP」「经济增长」→ gdp；"
    "「PPI」「生产者物价」→ ppi；「初请」「失业金」「claims」→ claims；"
    "「FOMC」「Fed 决议」「美联储会议」→ fomc。"
    "用户没指定或不在枚举内时输出空字符串（responder 默认 cpi）。\n"
    "\n"
    "【约束】\n"
    "1. **只输出 JSON，不要 markdown，不要解释**\n"
    "2. 不确定时输出 unknown\n"
    "3. ticker / manager / etf / direction / period / release_type 字段没有时输出空字符串\n"
    "4. target_price / days_horizon / days_back 没有时输出 0\n"
    "\n"
    "【JSON 格式】\n"
    '{"intent": "explain_move", "ticker": "NVDA", "manager": "", "etf": "", '
    '"target_price": 0, "direction": "", "days_horizon": 0, "period": "", '
    '"days_back": 0, "release_type": "", "raw": "..."}'
)


def classify(text: str) -> dict:
    """Map a free-form user message to a fixed intent + extracted params.

    Returns a dict with these keys:
        intent  — one of the 9 valid intent strings (defaults to "unknown")
        ticker  — uppercase US ticker, or "" if not present
        manager — lowercase manager alias, or "" if not present
        raw     — the LLM's own short echo of the user's intent (debug)
    """
    try:
        llm = ChatDeepSeek(model="deepseek-chat", temperature=0.0)
        response = llm.invoke([
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=text),
        ])
    except Exception as exc:
        logger.warning("Intent classifier LLM failed: %s", exc)
        return _unknown(text)

    content = (response.content or "").strip()
    # Strip ```json fences if present
    if content.startswith("```"):
        lines = content.split("\n")[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        content = "\n".join(lines).strip()

    try:
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            return _unknown(text)
    except json.JSONDecodeError as exc:
        logger.warning("Intent JSON parse failed (%r): %s", content[:120], exc)
        return _unknown(text)

    intent = str(parsed.get("intent", "")).strip()
    if intent not in _VALID_INTENTS:
        intent = "unknown"

    # Defensive numeric parsing — model sometimes returns strings
    try:
        target_price = float(parsed.get("target_price", 0) or 0)
    except (TypeError, ValueError):
        target_price = 0.0
    try:
        days_horizon = int(parsed.get("days_horizon", 0) or 0)
    except (TypeError, ValueError):
        days_horizon = 0

    # Strict enum for period — values outside the whitelist degrade to
    # empty string. The responder applies its own default ("day") on
    # empty. We DO NOT silently coerce e.g. "quarter" → "day" here so
    # the responder can surface the rejection to the user.
    period_raw = str(parsed.get("period", "")).strip().lower()
    period = period_raw if period_raw in _VALID_PNL_PERIODS else ""

    # days_back: bounded 7-365, default 0 means "use responder default (90)"
    try:
        days_back_raw = int(parsed.get("days_back", 0) or 0)
    except (TypeError, ValueError):
        days_back_raw = 0
    if days_back_raw == 0:
        days_back = 0     # responder applies _INSIDER_DAYS_BACK_DEFAULT
    elif days_back_raw < _INSIDER_DAYS_BACK_MIN:
        days_back = _INSIDER_DAYS_BACK_MIN
    elif days_back_raw > _INSIDER_DAYS_BACK_MAX:
        days_back = _INSIDER_DAYS_BACK_MAX
    else:
        days_back = days_back_raw

    # release_type: closed enum, empty → responder default "cpi"
    release_type_raw = str(parsed.get("release_type", "")).strip().lower()
    if release_type_raw in _VALID_RELEASE_TYPES:
        release_type = release_type_raw
    else:
        release_type = ""

    return {
        "intent": intent,
        "ticker": str(parsed.get("ticker", "")).strip().upper(),
        "manager": str(parsed.get("manager", "")).strip().lower(),
        "etf": str(parsed.get("etf", "")).strip().upper(),
        "target_price": target_price,
        "direction": str(parsed.get("direction", "")).strip().lower(),
        "days_horizon": days_horizon,
        "period": period,
        "days_back": days_back,
        "release_type": release_type,
        "raw": str(parsed.get("raw", text))[:80],
    }


def _unknown(text: str) -> dict:
    return {
        "intent": "unknown",
        "ticker": "",
        "manager": "",
        "etf": "",
        "target_price": 0.0,
        "direction": "",
        "days_horizon": 0,
        "period": "",
        "days_back": 0,
        "release_type": "",
        "raw": text[:80],
    }
