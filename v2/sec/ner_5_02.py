"""8-K item 5.02 LLM template-fill — name + title extraction only.

Item 5.02 ("Departure / Appointment of Officers / Directors") is the
ONLY 8-K item code in Phase 3 that needs LLM help — its priority swings
from P1 (independent director rotation, noise) to P0 (CEO/CFO sudden
departure, market-moving) based on **who** the document names.

Defense layers (same discipline as Phase 1 ⑧ earnings summarizer):
1. **Template-Fill**: LLM only emits structured JSON identifying
   names + titles. No prose, no editorial, no prediction.
2. **Conservative fallback**: on any failure (timeout, JSON parse error,
   schema mismatch), default ``has_senior_exec=True`` so the priority
   layer treats unknown extractions as potentially P0. Better to
   over-alert and let the user dismiss than to under-alert on a real CEO
   change.
3. **Fingerprint registration**: the prompt's distinctive phrase
   ``"8-K 5.02 高管变动信息抽取器"`` is registered in
   ``v2.observability.hooks.LLM_ROLE_FINGERPRINTS`` so the dashboard
   trace correctly tags this LLM call's role as ``sec_5_02_extractor``.

The extractor receives the 5.02 item's text block (typically 200-1000
chars; edgartools exposes it via the EightK ``items`` parsing). It does
NOT receive the entire 8-K — that would blow context budget and confuse
the model.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


# Public for observability/hooks.py fingerprint registration. Must be a
# distinctive substring (not a generic phrase) so unrelated LLM calls
# don't accidentally match it.
PROMPT_FINGERPRINT = "8-K 5.02 高管变动信息抽取器"


_SYSTEM_PROMPT = "\n".join([
    "你是一个 " + PROMPT_FINGERPRINT + "。",
    "",
    "【任务】",
    "从输入的 8-K item 5.02 文本中抽取高管 / 董事变动信息。",
    "5.02 涉及任命 (appointment) 和离职 (departure) 两类事件。",
    "",
    "【关键约束 — Template-Fill 模式】",
    "1. **只抽取实体，不写叙事**",
    '2. **严禁预测 / 解读 / 评价** — 不说"利好""利空""市场会涨"等',
    "3. **不做 inference** — 文本没说 CEO 离职就不要标 CEO",
    "4. **职位用全称** — 写 'Chief Executive Officer' 不要简写 'CEO'",
    "",
    "【字段含义】",
    "- has_senior_exec: 是否涉及 CEO / CFO / President / Chairman / COO",
    "  这些高管的变动 → true。其他独董 / VP 变动 → false。",
    "- 不确定时 has_senior_exec 输出 true（防漏 P0 升级）",
    "",
    "【输出格式】",
    "只输出 JSON，不加 markdown 代码块：",
    "{",
    '  "departures":   [{"name": "...", "title": "..."}, ...],',
    '  "appointments": [{"name": "...", "title": "..."}, ...],',
    '  "has_senior_exec": false',
    "}",
    "",
    "【反例】",
    '❌ {"departures": [{"name": "Smith", "title": "CEO"}], '
    '"narrative": "管理层突变可能利空股价"}  ← 不要加 narrative 字段',
    "❌ has_senior_exec=true 但 departures/appointments 都是普通 VP  ← 没人 senior 时应该 false",
])


# Conservative fallback for any failure — keeps the priority layer's
# escalation logic safe (treat unknown extractions as potentially P0).
_FALLBACK_RESULT: dict[str, Any] = {
    "departures": [],
    "appointments": [],
    "has_senior_exec": True,
}


def extract_5_02(item_text: str, *, llm_invoke=None) -> dict[str, Any]:
    """Run the LLM extractor against a 5.02 item text block.

    Args:
        item_text: the text content of one 8-K 5.02 item. Typically
            200-1000 characters. Caller is responsible for trimming to
            just the 5.02 section, not the whole 8-K.
        llm_invoke: test seam. If None, uses the default ChatDeepSeek
            invocation. Tests pass a callable that mocks the response.

    Returns:
        ``{"departures": [...], "appointments": [...], "has_senior_exec": bool}``.
        Always populated — failures degrade to ``_FALLBACK_RESULT`` per
        defense layer 2.
    """
    if not item_text or not item_text.strip():
        return dict(_FALLBACK_RESULT)

    if llm_invoke is None:
        llm_invoke = _default_llm_invoke

    try:
        response = llm_invoke(_SYSTEM_PROMPT, item_text[:4000])
    except Exception as exc:
        logger.warning("5.02 LLM extraction failed (network/timeout): %s", exc)
        return dict(_FALLBACK_RESULT)

    if not response:
        return dict(_FALLBACK_RESULT)

    parsed = _parse_response(response)
    if parsed is None:
        return dict(_FALLBACK_RESULT)
    return parsed


def _default_llm_invoke(system_prompt: str, user_text: str) -> str:
    """Real LLM call — DeepSeek temperature=0.0 for deterministic extraction.

    Imported lazily so unit tests can run without langchain-deepseek
    installed in the sandbox.
    """
    from langchain_core.messages import HumanMessage, SystemMessage
    from v2.data.metered import ChatDeepSeek

    llm = ChatDeepSeek(model="deepseek-chat", temperature=0.0)
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_text),
    ])
    return response.content or ""


def _parse_response(raw: str) -> dict[str, Any] | None:
    """Strip code fences, parse JSON, validate the 3 required keys.

    Returns None on any parse failure → caller falls back to safe defaults.
    """
    s = raw.strip()
    # Strip ```json … ``` fencing if model emitted it despite instructions.
    if s.startswith("```"):
        first_nl = s.find("\n")
        if first_nl >= 0:
            s = s[first_nl + 1:]
        if s.endswith("```"):
            s = s[: -3]
        s = s.strip()

    try:
        obj = json.loads(s)
    except json.JSONDecodeError as exc:
        logger.warning("5.02 LLM returned non-JSON (%s): %r", exc, raw[:120])
        return None

    if not isinstance(obj, dict):
        return None

    # Normalize — accept missing keys as empty lists, coerce flag to bool.
    departures = obj.get("departures") or []
    appointments = obj.get("appointments") or []
    has_senior_exec = bool(obj.get("has_senior_exec", False))

    if not isinstance(departures, list) or not isinstance(appointments, list):
        logger.warning("5.02 LLM returned wrong schema: %r", obj)
        return None

    return {
        "departures": [_normalize_person(p) for p in departures if isinstance(p, dict)],
        "appointments": [_normalize_person(p) for p in appointments if isinstance(p, dict)],
        "has_senior_exec": has_senior_exec,
    }


def _normalize_person(p: dict) -> dict[str, str]:
    """Strip + truncate to safe lengths for downstream card rendering."""
    return {
        "name": str(p.get("name", "") or "").strip()[:80],
        "title": str(p.get("title", "") or "").strip()[:120],
    }
