"""LLM narrator — turns a divergence verdict into one-line bull/bear notes.

Same contract as v2.screening.narrator: the detector has already decided
kind + strength from the numbers; the narrator only adds qualitative color
so a human can scan faster. Hard constraint — never fabricate numbers, and
always hedge ("疑似/或"), because CMF/RSI are proxies, not tick-level flow.
"""

from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage
from v2.data.metered import ChatDeepSeek

from v2.moneyflow.models import MoneyFlowSignal

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = (
    "你是一名资深盘面分析师，解读「量价背离」信号。对每只股票给出 bull + bear 各一句判断。\n"
    "\n"
    "【背景】信号由三条轴的确定性规则算出（我已判定好，你不要推翻）：\n"
    "  - 价格轴：近期是上涨 / 横盘 / 下跌\n"
    "  - 资金流轴（CMF 代理）：净流入 / 净流出\n"
    "  - 动量轴（RSI）：所处区间（超卖/低位/中性/高位/超买）+ 是否背离\n"
    "  - 判定 kind：accumulation（疑似吸筹）或 distribution（疑似派发/出货）\n"
    "\n"
    "【关键约束 — Template Fill 模式】\n"
    "1. **严禁输出任何具体数字** —— 百分比、价格、CMF/RSI 数值全部禁止，卡片里代码会填\n"
    "2. **必须留有余地** —— 用「疑似/或/倾向」等限定词；这是量价代理推断，不是逐笔资金实锤\n"
    "3. 可用定性词：「低位换手」「筹码沉淀」「主动买盘」「高位派发」「量能背离」「承接乏力」\n"
    "\n"
    "【内容要求】\n"
    "- accumulation：bull 说明为何像吸筹（价稳/跌但资金净流入、低位动量修复）；bear 给出反面风险（也可能只是阴跌无人接、假突破前的诱多）\n"
    "- distribution：bull 给出反面可能（也可能是获利了结、洗盘）；bear 说明为何像派发（价滞涨/横盘但资金流出、高位量能背离）\n"
    "- 每句 ≤ 35 字，像「盘面边际判断」而非「百科常识」\n"
    "\n"
    "【输出格式】只输出 JSON，不要 markdown：\n"
    '{"TICKER": {"bull": "...", "bear": "..."}, ...}'
)


def narrate(signals: list[MoneyFlowSignal]) -> tuple[dict[str, dict], int]:
    """Generate bull/bear blurbs for each signal in a single batch call.

    Returns (narrations, total_tokens); narrations maps ticker ->
    {"bull": ..., "bear": ...}. Returns ({}, 0) on any failure so the
    caller can keep going with number-only cards.
    """
    if not signals:
        return {}, 0

    _ZONE_ZH = {
        "oversold": "超卖", "low": "低位", "neutral": "中性",
        "high": "高位", "overbought": "超买",
    }
    _FLOW_ZH = {"inflow": "净流入", "outflow": "净流出", "neutral": "中性"}
    _PRICE_ZH = {"up": "上涨", "flat": "横盘", "down": "下跌"}

    payload: dict = {}
    for s in signals:
        payload[s.ticker] = {
            "kind": s.kind,
            "price": _PRICE_ZH.get(s.price_state, s.price_state),
            "flow": _FLOW_ZH.get(s.flow_state, s.flow_state),
            "rsi_zone": _ZONE_ZH.get(s.rsi_zone, s.rsi_zone),
            "rsi_divergence": s.rsi_divergence,
            "strength": s.strength,
        }

    user_prompt = (
        f"检测到 {len(signals)} 只股票出现量价背离，背景标签如下：\n\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
        '输出格式（每个 ticker 一个条目）：\n'
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
        lines = text.split("\n")[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text
