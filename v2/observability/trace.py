"""Trace + Event + contextvar plumbing.

A Trace is the per-query session object. It owns:
- A monotonic seq counter so the dashboard can order events deterministically
- A thread-safe callback (sink) the dashboard backend wires to an asyncio.Queue
- The session_id that all events carry

The contextvar TRACE_CTX is set by the dashboard's executor right before it
invokes a v2 responder. Every patched call site checks current_trace() and
no-ops if it's None — that's how production code paths stay clean.

Events are intentionally simple dicts (not pydantic models) for two reasons:
1. Zero coupling — anyone can emit a new event_type without schema bumps
2. Fast JSON serialization on the WS / SSE path
"""

from __future__ import annotations

import contextvars
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


EventSink = Callable[[dict], None]


@dataclass
class Event:
    """One trace event. Kept as a thin wrapper around a dict payload."""

    type: str
    session_id: str
    seq: int
    ts_ms: int
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "session_id": self.session_id,
            "seq": self.seq,
            "ts_ms": self.ts_ms,
            **self.payload,
        }


@dataclass
class Trace:
    """Per-session trace context.

    The sink callback is invoked synchronously from whichever thread emit()
    runs on. The dashboard wires it to asyncio.run_coroutine_threadsafe so
    events from v2 code (running in a thread executor) land in the main
    event loop's queue safely.
    """

    session_id: str = field(default_factory=lambda: f"sess_{uuid.uuid4().hex[:16]}")
    sink: Optional[EventSink] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    _seq: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def emit(self, event_type: str, **payload: Any) -> Event:
        with self._lock:
            self._seq += 1
            seq = self._seq
        ev = Event(
            type=event_type,
            session_id=self.session_id,
            seq=seq,
            ts_ms=int(time.time() * 1000),
            payload=payload,
        )
        sink = self.sink
        if sink is not None:
            try:
                sink(ev.to_dict())
            except Exception:
                # Sink failures must never break the underlying v2 call.
                pass
        return ev


TRACE_CTX: contextvars.ContextVar[Optional[Trace]] = contextvars.ContextVar(
    "v2_trace_ctx", default=None
)


def current_trace() -> Optional[Trace]:
    """Return the Trace bound to the current execution context, or None."""
    return TRACE_CTX.get()


def emit(event_type: str, **payload: Any) -> None:
    """Emit on the current trace, or silently drop if there is no trace bound.

    All monkey-patched wrappers call this — the no-trace case is the
    production-Telegram-bot path, which must remain a no-op.
    """
    trace = TRACE_CTX.get()
    if trace is None:
        return
    trace.emit(event_type, **payload)
