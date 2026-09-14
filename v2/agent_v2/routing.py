"""Request normalization and first-stage routing from the question's intent.

Routing no longer matches words: the intent (``intent.Intent``, from the
model live, from the recorded labels offline) says what kind of answer is
wanted, and the route follows from that.  The route decision carries the
intent so the planner reads the same fields.
"""

from __future__ import annotations

from typing import Any

from v2.agent_v2.entities import extract_entities
from v2.agent_v2.models import NormalizedRequest, RouteDecision, RouteKind


def normalize_request(
    text: str,
    *,
    session_id: str = "",
    allow_web: bool = False,
    metadata: dict | None = None,
) -> NormalizedRequest:
    original = text or ""
    cleaned = original.strip()
    forced = False
    if cleaned.lower().startswith("/ask-v2"):
        remainder = cleaned[len("/ask-v2") :].lstrip(" \t:：,，")
        if remainder:
            cleaned, forced = remainder, True
    return NormalizedRequest(
        original_text=original,
        text=cleaned,
        session_id=session_id,
        entities=extract_entities(cleaned),
        forced_agent=forced,
        allow_web=allow_web,
        metadata=dict(metadata or {}),
    )


def route(request: NormalizedRequest, *, intent: Any = None, classifier: Any = None) -> RouteDecision:
    """Decide the route from the intent: given, classified by ``classifier``, recorded for the text, or the default."""

    from v2.agent_v2.intent import resolve_intent

    resolved = intent if intent is not None else resolve_intent(request, classifier=classifier)
    if resolved.kind == "command" or resolved.command:
        return RouteDecision(RouteKind.COMMAND, ("command",), "explicit user-state mutation", intent=resolved)
    if resolved.kind == "lab":
        asynchronous = resolved.lab_scale == "deep"
        return RouteDecision(RouteKind.ASYNC if asynchronous else RouteKind.LAB, ("lab",), "quantitative experiment requested", asynchronous=asynchronous, intent=resolved)
    if request.forced_agent:
        return RouteDecision(RouteKind.RESEARCH, ("research", "account"), "forced by /ask-v2", intent=resolved)
    if resolved.kind == "knowledge" and not resolved.tickers and not request.entities:
        return RouteDecision(RouteKind.GENERAL_KNOWLEDGE, (), "stable conceptual question", intent=resolved)
    if resolved.kind == "research":
        packs = ("research", "account") if resolved.portfolio_scope or resolved.watchlist_scope else ("research",)
        return RouteDecision(RouteKind.RESEARCH, packs, "multi-source research or synthesis", intent=resolved)
    return RouteDecision(RouteKind.FAST_LOOKUP, ("account", "research"), "single-purpose lookup", intent=resolved)
