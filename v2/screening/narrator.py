"""LLM narrator — turns candidate metrics into one-line bull/bear notes.

The narrator is an EXPLAINER, not a recommender. The hard-rule screen
has already decided what passes; the narrator just adds color so a human
can scan the output faster. Constraint: never fabricate numbers.
"""

from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage
from v2.data.metered import ChatDeepSeek

from v2.screening.models import ScreenCandidate

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = (
    "你是一名资深科技股分析师。对每只股票给出 bull + bear 各一句**动态增量**判断。\n"
    "\n"
    "【关键约束 — Template Fill 模式】\n"
    "1. **严禁输出任何具体数字** — 百分比、美元金额、估值倍数、年份数字 全部禁止\n"
    "2. 数字会由代码自动从结构化数据填入卡片，你不需要重复\n"
    "3. 可以用定性词：「显著」「加速」「放缓」「领先」「承压」「主导」「侵蚀」\n"
    "\n"
    "【关键约束 — 动态增量（改进 ①）】\n"
    "1. 我会给你四类动态数据：\n"
    "   - recent_news：最近 7 天的新闻标题（带 snippet）\n"
    "   - peer_diff_pp：1 周回报相对同业中位数的百分点差异\n"
    "   - sector_diff_pp：1 周回报相对板块 ETF（XLK 或 SMH）的百分点差异\n"
    "   - earnings_surprise_pct：本季营收实际 vs 预期的偏差\n"
    "2. **优先**基于这些动态数据提炼边际变化\n"
    "3. **其次**才是常识级护城河（NVDA 主导 AI 芯片、META 是广告生意等）\n"
    "4. bull/bear 应像「最新边际变化」而非「百科常识」\n"
    "5. sector_diff_pp 正值大 → 「板块强势中的领涨」/「逆板块上行」；\n"
    "   负值大 → 「板块强势中掉队」/「板块拖累」\n"
    "\n"
    "【内容要求】\n"
    "- bull：抓增长引擎 / 最新利好（≤ 35 字）\n"
    "- bear：抓最关键的边际变化（竞争加剧、监管收紧、客户流失、估值透支）（≤ 35 字）\n"
    "- bear 不要写「波动率高」—— 已被前置过滤掉\n"
    "\n"
    "【输出格式】\n"
    "只输出 JSON，不要 markdown：\n"
    '{"TICKER": {"bull": "...", "bear": "..."}, ...}\n'
    "\n"
    "【反例 vs 正例】\n"
    "反例（静态百科 + 含数字）：\n"
    '  {"bull": "AI 训练芯片近乎垄断，撑起 71% 毛利"} ← 含数字 + 全市场都知道\n'
    "\n"
    "正例（动态增量 + 纯逻辑）：\n"
    '  {"bull": "本季营收显著超预期，Blackwell 出货验证 AI 需求未见顶",\n'
    '   "bear": "超大客户自研 ASIC 配比上调，订单边际承压"}'
)


def narrate(candidates: list[ScreenCandidate]) -> tuple[dict[str, dict], int]:
    """Generate bull/bear blurbs for each candidate in a single batch call.

    Returns:
        (narrations, total_tokens) where narrations maps ticker -> {"bull": ..., "bear": ...}
        Returns ({}, 0) on any failure — the caller should keep going without notes.
    """
    if not candidates:
        return {}, 0

    payload: dict = {}
    for c in candidates:
        item: dict = {
            "market_cap_billions": round(c.market_cap / 1e9, 1) if c.market_cap else None,
            "revenue_growth_pct": round(c.revenue_growth * 100, 1) if c.revenue_growth else None,
            "gross_margin_pct": round(c.gross_margin * 100, 1) if c.gross_margin else None,
            "annualized_volatility_pct": round(c.volatility * 100, 1) if c.volatility else None,
        }
        # 改进 ① — dynamic delta context
        if c.revenue_surprise_pct is not None:
            item["earnings_surprise_pct"] = round(c.revenue_surprise_pct * 100, 1)
        if c.peer_diff_1w is not None:
            item["peer_diff_pp"] = round(c.peer_diff_1w * 100, 1)
        if c.return_1w is not None:
            item["return_1w_pct"] = round(c.return_1w * 100, 1)
        if c.sector_etf and c.sector_diff_1w_pp is not None:
            item["sector_etf"] = c.sector_etf
            item["sector_diff_pp"] = round(c.sector_diff_1w_pp * 100, 1)
        if c.news_headlines:
            item["recent_news"] = [
                {"title": h.get("title", "")[:100], "snippet": h.get("snippet", "")[:160]}
                for h in c.news_headlines[:3]
            ]
        payload[c.ticker] = item

    user_prompt = (
        f"通过筛选的 {len(candidates)} 只科技股指标如下：\n\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
        f"输出格式（每个 ticker 一个条目）：\n"
        '{"TICKER": {"bull": "...", "bear": "..."}, ...}'
    )

    try:
        llm = ChatDeepSeek(model="deepseek-chat", temperature=0.3)
        response = llm.invoke([
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ])
    except Exception as exc:
        logger.warning("DeepSeek call failed: %s", exc)
        return {}, 0

    content = _strip_code_fence(response.content)
    try:
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            return {}, 0
    except json.JSONDecodeError as exc:
        logger.warning("DeepSeek returned non-JSON: %s\nContent: %r", exc, content[:200])
        return {}, 0

    tokens = 0
    meta = getattr(response, "usage_metadata", None) or {}
    if isinstance(meta, dict):
        tokens = int(meta.get("total_tokens", 0))

    return parsed, tokens


def _strip_code_fence(text: str) -> str:
    """Defensive — some models still wrap JSON in ```json blocks."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Drop first fence and any trailing one
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text
