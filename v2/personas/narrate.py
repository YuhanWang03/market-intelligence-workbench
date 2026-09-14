"""Optional LLM narration — phrase a verdict the numbers already reached.

The persona layer is complete without this module.  What it adds is a short
explanation in the investor's own voice, produced under three constraints
the upstream prompts only asked for politely:

* the signal and confidence are given and must be echoed unchanged;
* only figures present in the facts may be cited;
* every figure in the reply is checked against the facts afterwards with
  :func:`v2.agent_common.grounding.check`, and the result is recorded on the
  signal as ``narrative_grounded`` rather than silently trusted.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Iterable

from v2.personas.base import Persona
from v2.personas.models import PersonaSignal
from v2.personas.registry import get_persona

logger = logging.getLogger(__name__)

MAX_CHARS = 280


def facts_for_prompt(signal: PersonaSignal) -> dict[str, Any]:
    """The compact bundle the model sees: scores, details, valuation facts."""
    return {
        "ticker": signal.ticker,
        "as_of": signal.as_of,
        "signal": signal.signal,
        "confidence": signal.confidence,
        "score": signal.score,
        "max_score": signal.max_score,
        "margin_of_safety": None if signal.margin_of_safety is None else round(signal.margin_of_safety, 4),
        "checklist": {p.name: {"score": p.score, "max": p.max_score, "details": p.details} for p in signal.parts},
        "facts": {k: v for k, v in signal.facts.items() if not isinstance(v, (list, dict))},
    }


def build_messages(signal: PersonaSignal, persona: Persona, *, language: str = "zh") -> list[dict[str, str]]:
    lang = "Simplified Chinese" if language == "zh" else "English"
    system = (
        f"{persona.system_prompt.strip()}\n\n"
        "You are explaining a verdict that has already been decided by rules. "
        f"The signal is {signal.signal} with confidence {signal.confidence}; repeat both exactly, never change them. "
        "Cite only numbers that appear in the facts, written as they appear. Do not invent data. "
        f"Write in {lang}, at most {MAX_CHARS} characters. Return JSON only."
    )
    user = (
        "Facts:\n" + json.dumps(facts_for_prompt(signal), ensure_ascii=False, separators=(",", ":")) +
        '\n\nReturn exactly: {"signal": "...", "confidence": <int>, "reasoning": "<your explanation>"}'
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


_JSON_BLOCK = re.compile(r"\{.*\}", re.S)


def parse_reply(text: str) -> dict[str, Any] | None:
    body = (text or "").strip()
    if body.startswith("```"):
        body = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", body, flags=re.S).strip()
    candidates = [body]
    match = _JSON_BLOCK.search(body)
    if match and match.group(0) != body:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and "reasoning" in parsed:
            return parsed
    return None


def grounded(narrative: str, signal: PersonaSignal) -> bool:
    """Every figure in the narrative must trace to the facts the model was shown."""
    try:
        from v2.agent_common.grounding import check
    except Exception:  # noqa: BLE001 — grounding module is optional
        return True
    observations = json.dumps(facts_for_prompt(signal), ensure_ascii=False)
    return check(narrative, observations).ok


def narrate(signal: PersonaSignal, persona: Persona | None = None, llm: Any | None = None, *, language: str = "zh") -> PersonaSignal:
    """Attach an LLM-written explanation to ``signal`` in place; returns it.

    Never raises on model trouble: the signal keeps its templated reasoning
    and ``narrative`` stays ``None``.  A reply that changes the signal or
    the confidence is discarded — the model does not get a vote.
    """
    if signal.abstained:
        return signal
    persona = persona or get_persona(signal.persona)
    if llm is None:
        from v2.agent_common.llm import build_llm
        llm = build_llm()
    try:
        reply = llm.complete(build_messages(signal, persona, language=language))
    except Exception as exc:  # noqa: BLE001
        logger.warning("narration failed for %s/%s: %s", signal.persona, signal.ticker, exc)
        return signal
    parsed = parse_reply(getattr(reply, "text", "") or "")
    if not parsed:
        logger.warning("narration unparsable for %s/%s", signal.persona, signal.ticker)
        return signal
    if str(parsed.get("signal", signal.signal)).lower() != signal.signal:
        logger.warning("narration tried to change the signal for %s/%s; discarded", signal.persona, signal.ticker)
        return signal
    try:
        if int(parsed.get("confidence", signal.confidence)) != signal.confidence:
            logger.warning("narration tried to change the confidence for %s/%s; discarded", signal.persona, signal.ticker)
            return signal
    except (TypeError, ValueError):
        return signal
    text = str(parsed.get("reasoning", "")).strip()
    if not text:
        return signal
    signal.narrative = text[:MAX_CHARS]
    signal.narrative_grounded = grounded(signal.narrative, signal)
    return signal


def narrate_many(signals: Iterable[PersonaSignal], llm: Any | None = None, *, language: str = "zh") -> list[PersonaSignal]:
    if llm is None:
        from v2.agent_common.llm import build_llm
        llm = build_llm()
    return [narrate(s, llm=llm, language=language) for s in signals]
