"""Ephemeral request resources: never checkpointed."""

from dataclasses import dataclass, field
import threading
import time
from typing import Any


class RunStopped(RuntimeError):
    pass


@dataclass
class RunContext:
    run_id: str
    deadline: float
    cancel_event: Any = None
    on_progress: Any = None
    usage: list[dict] = field(default_factory=list)
    lock: Any = field(default_factory=threading.Lock)

    @property
    def cancelled(self):
        return bool(self.cancel_event and self.cancel_event.is_set())

    def check(self, reserve: float = 0):
        if self.cancelled:
            raise RunStopped("cancelled")
        if time.monotonic() + reserve >= self.deadline:
            raise RunStopped("deadline")

    def remaining(self):
        return max(0, self.deadline - time.monotonic())

    def record(self, source: str, message):
        with self.lock:
            self.usage.append({"source": source, **dict(getattr(message, "usage_metadata", None) or {})})

    def emit(self, node: str):
        if self.on_progress:
            try:
                from v2.agent_v2.models import ProgressEvent, RunStatus
                self.on_progress(ProgressEvent(self.run_id, RunStatus.EXECUTING, node))
            except Exception:
                pass
