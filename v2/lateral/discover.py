"""LLM-driven discovery of supply-chain / peer / beneficiary neighbors.

One DeepSeek call generates raw candidates across all 4 categories for all
seeds. The output is intentionally unvalidated — `verify.py` checks ticker
existence afterward, and made-up tickers are surfaced as "hallucinations"
in the final report.
"""

from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage
from v2.data.metered import ChatDeepSeek

from v2.lateral.models import Label

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = (
    "你是一名资深科技股研究分析师。给定一组种子股票，对每只种子识别"
    "在以下 4 个维度下的相关美股上市公司：\n"
    "\n"
    "- supplier（供应商）：向种子公司提供材料、组件、设备或服务的公司\n"
    "- customer（客户）：从种子购买产品/服务且占其营收非小额的公司\n"
    "- smaller_peer（同业小市值）：与种子业务高度相似但市值明显更小的公司\n"
    "- beneficiary（间接受益方）：不在供应链中，但受益于种子相关行业趋势的公司\n"
    "\n"
    "硬约束：\n"
    "1. 每个 ticker 必须是美股上市公司（NYSE/NASDAQ），不接受外国 ADR/OTC\n"
    "2. 每个类别最多 4 个，宁缺毋滥\n"
    "3. 同一 ticker 可以出现在多个种子下，但 reason 应针对该种子\n"
    "4. 不要包含种子自己或种子列表中的其他公司\n"
    "5. 每条 reason ≤ 25 字\n"
    "6. 不确定的不要写——后续会用真实数据验证，编造的会被标为幻觉\n"
    "7. 只输出 JSON，不要 markdown 代码块或额外文字\n"
    "\n"
    "JSON 格式示例：\n"
    "{\n"
    '  "NVDA": {\n'
    '    "supplier": [{"ticker": "TSM", "reason": "唯一先进制程晶圆代工"}],\n'
    '    "customer": [{"ticker": "MSFT", "reason": "Azure 大量采购 GPU"}],\n'
    '    "smaller_peer": [{"ticker": "MRVL", "reason": "数据中心 ASIC"}],\n'
    '    "beneficiary": [{"ticker": "VST", "reason": "AI 数据中心电力需求"}]\n'
    "  }\n"
    "}"
)


def discover(seeds: list[str]) -> tuple[list[tuple[str, Label]], int]:
    """One LLM batch call to generate neighbor candidates for all seeds.

    Returns:
        pairs:  list of (ticker, Label) tuples. Same ticker may appear in
                multiple pairs (under different seeds/categories).
        tokens: total tokens used by the call.

    Returns ([], 0) on any failure — orchestrator handles gracefully.
    """
    user = (
        f"种子股票：{', '.join(seeds)}\n\n"
        f"请对每只种子识别 4 个维度下的相关公司，按上述 JSON 格式输出。"
    )

    try:
        llm = ChatDeepSeek(model="deepseek-chat", temperature=0.3)
        response = llm.invoke([
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user),
        ])
    except Exception as exc:
        logger.warning("DeepSeek discover call failed: %s", exc)
        return [], 0

    content = _strip_code_fence(response.content)
    try:
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            raise ValueError("top level is not an object")
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("DeepSeek returned non-JSON: %s\nContent: %r", exc, content[:300])
        return [], 0

    pairs: list[tuple[str, Label]] = []
    for seed, categories in parsed.items():
        if not isinstance(categories, dict):
            continue
        for category, items in categories.items():
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                ticker = str(item.get("ticker", "")).strip().upper()
                reason = str(item.get("reason", "")).strip()
                if not ticker or ticker == seed:
                    continue
                pairs.append((
                    ticker,
                    Label(seed=seed, category=category, reason=reason),
                ))

    tokens = 0
    meta = getattr(response, "usage_metadata", None) or {}
    if isinstance(meta, dict):
        tokens = int(meta.get("total_tokens", 0))

    return pairs, tokens


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text
