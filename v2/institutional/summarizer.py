"""LLM interpretation of 13F position changes — one batch call per manager."""

from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage
from v2.data.metered import ChatDeepSeek

from v2.institutional.models import PositionChange

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = (
    "你是一名机构持仓分析师。给定 manager 的姓名/风格 和 一组本季度持仓变动，"
    "对每笔变动给出一句话解读，结合 manager 已知的投资风格、历史决策模式、"
    "以及公司基本面/行业地位。\n"
    "\n"
    "硬约束：\n"
    "1. 可以引用对 manager 风格和公司业务的常识（Buffett 价值/Burry 逆向等）\n"
    "2. 不许编造具体数字（PE、市占率、内部人持股等）\n"
    "3. 不预测未来表现，只解读「为什么这样做」\n"
    "4. 每句话 ≤ 30 字，要有信息密度\n"
    "5. 只输出 JSON，不要 markdown 代码块\n"
    "\n"
    "反例：'新进 NVDA，看好 AI'（无信息）\n"
    "正例：'罕见拥抱 AI 龙头，违背一贯不懂不投原则，可能 Todd 主导'\n"
    "\n"
    'JSON 格式：{"identifier": "解读", ...}'
)


def interpret_changes(
    manager_name: str,
    changes: list[PositionChange],
) -> tuple[dict[str, str], int]:
    """Generate one-line interpretations for all of a manager's changes.

    Returns (map keyed by ticker-or-cusip, total tokens used). Returns
    ({}, 0) on any failure — the orchestrator will gracefully skip
    interpretations and just show raw changes.
    """
    if not changes:
        return {}, 0

    items = []
    for c in changes:
        identifier = c.ticker or c.cusip
        items.append({
            "id": identifier,
            "ticker": c.ticker or "",
            "issuer": c.issuer_name[:50],
            "change_type": c.change_type,
            "current_value_M": round(c.current_value / 1e6, 1),
            "prev_value_M": round(c.prev_value / 1e6, 1),
            "current_pct_of_portfolio": (
                f"{c.current_pct:.1%}" if c.current_pct else "0%"
            ),
        })

    user_prompt = (
        f"Manager: {manager_name}\n"
        f"季度: {changes[0].quarter}\n\n"
        f"持仓变动（current_value_M 是当前持仓市值，单位百万美元）：\n"
        f"{json.dumps(items, ensure_ascii=False, indent=2)}\n\n"
        f"对每笔给出一句话解读。"
    )

    try:
        llm = ChatDeepSeek(model="deepseek-chat", temperature=0.3)
        response = llm.invoke([
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ])
    except Exception as exc:
        logger.warning("DeepSeek interpret failed for %s: %s",
                       manager_name, exc)
        return {}, 0

    content = _strip_code_fence(response.content)
    try:
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            return {}, 0
    except json.JSONDecodeError as exc:
        logger.warning("Non-JSON response for %s: %s", manager_name, exc)
        return {}, 0

    tokens = 0
    meta = getattr(response, "usage_metadata", None) or {}
    if isinstance(meta, dict):
        tokens = int(meta.get("total_tokens", 0))

    # Coerce values to strings
    return {str(k): str(v) for k, v in parsed.items()}, tokens


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text
