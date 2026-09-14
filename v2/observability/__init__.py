"""Observability SDK for the dashboard.

Production Telegram bot, scheduler, and streamer never import or call anything
from this package — it is dormant unless the dashboard backend explicitly
calls install_all() at startup. That keeps the existing services' behavior
bit-for-bit identical.

Public surface:
    Trace, Event, TRACE_CTX, current_trace, emit
    capture_trace()                — bare capture (no framing)
    capture_trace_with_framing()   — capture + auto-emit session/intent/module
                                     events so cron-produced trace renders
                                     identically to a dashboard-produced one
    install_all()
    estimate_cost(...)
"""

import time
import uuid
from contextlib import contextmanager

from v2.observability.trace import (
    Event,
    TRACE_CTX,
    Trace,
    current_trace,
    emit,
)
from v2.observability.hooks import (
    LLM_ROLE_FINGERPRINTS,
    detect_llm_role,
    install_all,
    installed_hooks,
)
from v2.observability.pricing import estimate_cost


@contextmanager
def capture_trace(session_id: str | None = None):
    """Capture all v2/ trace events emitted inside the block.

    Yields a Trace whose .events list is updated synchronously as emit()
    calls fire. Useful for low-level inspection; most schedulers should
    prefer capture_trace_with_framing so the saved trace renders fully
    in the dashboard PipelineBar + 📖 解析 disclosures.
    """
    sid = session_id or f"capture_{uuid.uuid4().hex[:12]}"
    events: list = []
    trace = Trace(session_id=sid, sink=lambda ev: events.append(ev))
    trace.events = events  # type: ignore[attr-defined]
    token = TRACE_CTX.set(trace)
    try:
        yield trace
    finally:
        TRACE_CTX.reset(token)


@contextmanager
def capture_trace_with_framing(
    agent: str,
    intent: str,
    text: str = "",
    responder_name: str = "",
):
    """capture_trace + emit the same start/end framing events the dashboard
    executor produces, so saved trace_json renders identically.

    Args:
        agent: short label for the cron job ("etf" / "anomaly" / …).
            Recorded only for logging; doesn't shape pipeline routing.
        intent: which dashboard pipeline this trace should map onto
            ("etf_view" / "explain_move" / "thirteen_f" / …).
            PipelineBar uses it to pick the pill set.
        text: trigger text shown in the trace's session_start event.
            Defaults to "(自动推送) AGENT".
        responder_name: optional fake responder name like
            "_r_etf_snapshot". When set, emits module_enter on entry
            and module_exit on exit, framing the work block the way
            dashboard responder calls do.

    The caller is responsible for emitting chat_message after rendering
    its reply text, before the with-block exits. That way the final
    trace.events list (with module_exit + session_end appended on exit)
    contains everything the dashboard needs.

    Usage:
        with capture_trace_with_framing("etf", "etf_view", text="run",
                                        responder_name="_r_etf_snapshot") as trace:
            ...work...
            reply = format(...)
            trace.emit("chat_message", role="bot", text=reply[:500])
        # session_end is now in trace.events
        notifier.send_text(reply, trace=trace, ...)
    """
    sid = f"cron_{agent}_{uuid.uuid4().hex[:8]}"
    events: list = []
    trace = Trace(session_id=sid, sink=lambda ev: events.append(ev))
    trace.events = events  # type: ignore[attr-defined]
    token = TRACE_CTX.set(trace)
    started_ms = int(time.time() * 1000)

    try:
        trace.emit(
            "session_start",
            text=text or f"(自动推送) {agent}",
            client_kind="cron",
        )
        trace.emit("intent_classified", intent=intent, args={})
        if responder_name:
            trace.emit("module_enter", name=responder_name, intent=intent)

        yield trace

        if responder_name:
            trace.emit(
                "module_exit",
                name=responder_name,
                elapsed_ms=int(time.time() * 1000) - started_ms,
            )
        trace.emit(
            "session_end",
            total_cost_usd=0.0,
            elapsed_ms=int(time.time() * 1000) - started_ms,
        )
    finally:
        TRACE_CTX.reset(token)


__all__ = [
    "Event",
    "TRACE_CTX",
    "Trace",
    "current_trace",
    "emit",
    "capture_trace",
    "capture_trace_with_framing",
    "install_all",
    "installed_hooks",
    "estimate_cost",
    "LLM_ROLE_FINGERPRINTS",
    "detect_llm_role",
]
