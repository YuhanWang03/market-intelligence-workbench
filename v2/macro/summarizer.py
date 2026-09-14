"""LLM template-fill summarizer for macro releases — Phase 4 Stage 1.

Three-layer hallucination defense (Stage 0 design ack):

**Layer 1 — System prompt:** the prompt explicitly forbids outputting
numbers and forbids forward predictions. The LLM is reduced to picking
qualitative labels from a constrained schema.

**Layer 2 — Post-parse reject:** the response is regex-scanned for
predictive verbs (``将``, ``会``, ``预期``, ``预计``, ``料`` plus
``可能.*[上下]``). Hits → fallback to neutral labels.

**Layer 3 — Tone is data-side only:** even ``tone`` is constrained to
describe the data itself, not predict Fed action. FOMC events bypass
this module entirely and route through :mod:`v2.macro.fomc_parser` +
:mod:`v2.macro.tavily_consensus`.

The fingerprint substring "宏观数据解读分析师" is registered in
:mod:`v2.observability.hooks.LLM_ROLE_FINGERPRINTS` so cron-captured
traces tag this role as ``macro_summarizer``.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from v2.macro.models import MacroRelease

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Layer 1 — system prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """你是宏观数据解读分析师。输入 JSON 含数据点，输出 JSON 含解读 label。

规则：
1. 严禁输出任何数字。所有数字都由 Python 算好了，你只需要描述事实，不重复数字。
2. 严禁预测后市方向。可说"已经发生"，不可说"将会"。
   反例：「Fed 9 月会降息」/「股市将走高」/「通胀预期回升」—— 全部禁止。
   正例：「核心通胀连续 3 月放缓」/「劳动力市场降温」—— 描述事实。
3. 输出严格 JSON，4 个字段，每个 ≤ 40 个字符：
   {
     "bull_takeaway": str | null,   // 数据中对市场偏正面的事实
     "bear_takeaway": str | null,   // 数据中对市场偏负面的事实
     "narrative":     str,          // 1 句中性描述（无方向预测）
     "tone":          "hawkish" | "dovish" | "neutral"
   }
   - tone 仅描述数据本身（例如 CPI 高于预期 = hawkish），不预测 Fed 行为。
   - bull/bear 任何一个都可以为 null（如果数据真的中性）。
   - narrative 必须非空。
"""


# ---------------------------------------------------------------------------
# Layer 2 — regex reject
# ---------------------------------------------------------------------------

# Predictive verbs / phrases that signal forward-looking opinion.
# - 将 / 会         — 中文将来时
# - 预期 / 预计 / 料 — 预测动词
# - 可能 ... 上 / 下 — 模态加方向
_REJECT_PATTERNS = re.compile(
    r"将|会(?![计算])|预期|预计|料(?![理])|可能.*[上下涨跌升降]"
)


def _violates_no_prediction(text: str) -> bool:
    """True iff ``text`` contains a Layer-2 reject pattern."""
    if not text:
        return False
    return bool(_REJECT_PATTERNS.search(text))


# Numeric leak detector — the prompt says "no numbers"; an actual digit
# in the response means the model ignored the rule. Allow short digit-
# adjacent labels like "1月" (month name) or "3 季度" via a narrow check.
_DIGIT_LEAK_RE = re.compile(r"\d+\.?\d*\s*(?:%|个百分点|bps|基点|个|亿|万亿)")


def _violates_no_numbers(text: str) -> bool:
    if not text:
        return False
    return bool(_DIGIT_LEAK_RE.search(text))


# ---------------------------------------------------------------------------
# Fallback shape
# ---------------------------------------------------------------------------

_NEUTRAL_FALLBACK: dict[str, Any] = {
    "bull_takeaway": None,
    "bear_takeaway": None,
    "narrative": "数据已发布，详见上方数值。",
    "tone": "neutral",
}


# ---------------------------------------------------------------------------
# Core summarize entry point
# ---------------------------------------------------------------------------

def summarize_release(
    release: MacroRelease,
    *,
    llm_invoke=None,
) -> dict[str, Any]:
    """Run the macro summarizer against one already-computed
    :class:`MacroRelease`.

    Args:
        release: a populated ``MacroRelease`` (Python already filled in
            the numbers + trend label). Only the qualitative labels are
            LLM-produced.
        llm_invoke: optional callable ``(system_prompt, user_text) -> str``
            used as a test seam. Default is a real ChatDeepSeek call,
            imported lazily.

    Returns:
        Dict with keys ``bull_takeaway`` / ``bear_takeaway`` / ``narrative``
        / ``tone``. Layer-2 violations and parse failures all collapse
        to :data:`_NEUTRAL_FALLBACK`.
    """
    if llm_invoke is None:
        llm_invoke = _default_llm_invoke

    user_text = _build_user_prompt(release)

    try:
        response = llm_invoke(_SYSTEM_PROMPT, user_text)
    except Exception as exc:                           # noqa: BLE001
        logger.warning("macro_summarizer LLM call failed: %s", exc)
        return dict(_NEUTRAL_FALLBACK)

    parsed = _parse_response(response)
    if parsed is None:
        return dict(_NEUTRAL_FALLBACK)
    return parsed


def _build_user_prompt(release: MacroRelease) -> str:
    """Compose the data-point block. Only numbers Python computed are
    sent in; the LLM never sees raw FRED series so it can't invent
    new data points either."""
    parts = [
        f"release:        {release.release_type} {release.period}",
        f"release_date:   {release.release_date}",
    ]
    if release.headline is not None:
        parts.append(f"headline:       {release.headline}")
    if release.core is not None:
        parts.append(f"core:           {release.core}")
    if release.mom_pct is not None:
        parts.append(f"mom_pct:        {release.mom_pct:.4f}")
    if release.yoy_pct is not None:
        parts.append(f"yoy_pct:        {release.yoy_pct:.4f}")
    if release.consensus is not None:
        parts.append(f"consensus:      {release.consensus}")
    if release.surprise_sigma is not None:
        parts.append(
            f"surprise:       {release.surprise_sigma:+.2f}σ "
            f"({release.surprise_label})"
        )
    parts.append(f"trend_3mo:      {release.trailing_3mo_trend}")
    parts.append("")
    parts.append("请按上述规则输出 JSON。")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Response parsing + reject loop
# ---------------------------------------------------------------------------

_ALLOWED_TONES = frozenset({"hawkish", "dovish", "neutral"})


def _parse_response(raw: str) -> dict[str, Any] | None:
    """Extract a JSON object from the LLM response and validate the
    fields. Returns None on any structural failure or Layer-2 violation.
    """
    if not raw or not raw.strip():
        return None

    # Tolerate ```json fences / leading prose. Find the outermost {...}.
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end < 0 or end <= start:
        logger.info("macro_summarizer: no JSON object found in response")
        return None

    try:
        obj = json.loads(raw[start : end + 1])
    except json.JSONDecodeError as exc:
        logger.info("macro_summarizer: JSON decode failed: %s", exc)
        return None

    if not isinstance(obj, dict):
        return None

    bull = obj.get("bull_takeaway")
    bear = obj.get("bear_takeaway")
    narrative = obj.get("narrative")
    tone = obj.get("tone")

    # narrative is required
    if not isinstance(narrative, str) or not narrative.strip():
        return None

    # tone must be enum
    if tone not in _ALLOWED_TONES:
        logger.info("macro_summarizer: invalid tone %r → reject", tone)
        return None

    # nullable but if present must be str
    for label, value in (("bull_takeaway", bull), ("bear_takeaway", bear)):
        if value is not None and not isinstance(value, str):
            logger.info("macro_summarizer: invalid type for %s", label)
            return None

    # Layer-2 reject — predictive language or numeric leak in any field
    for value in (bull, bear, narrative):
        if isinstance(value, str):
            if _violates_no_prediction(value):
                logger.info("macro_summarizer: predictive text rejected: %r", value)
                return None
            if _violates_no_numbers(value):
                logger.info("macro_summarizer: numeric leak rejected: %r", value)
                return None

    return {
        "bull_takeaway": bull,
        "bear_takeaway": bear,
        "narrative": narrative.strip(),
        "tone": tone,
    }


# ---------------------------------------------------------------------------
# Default LLM caller (lazy import so sandbox tests stay clean)
# ---------------------------------------------------------------------------

def _default_llm_invoke(system_prompt: str, user_text: str) -> str:
    from langchain_core.messages import HumanMessage, SystemMessage
    from v2.data.metered import ChatDeepSeek

    llm = ChatDeepSeek(model="deepseek-chat", temperature=0.0)
    msg = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_text),
    ])
    return getattr(msg, "content", "") or ""


__all__ = [
    "summarize_release",
]
