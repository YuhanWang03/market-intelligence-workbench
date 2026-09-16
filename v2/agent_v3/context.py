"""Ephemeral request resources: never checkpointed."""

from dataclasses import dataclass, field
import threading
import time
from typing import Any
from urllib.parse import urlparse


class RunStopped(RuntimeError):
    pass


def model_identity(model) -> tuple[str, str]:
    """(provider label, model name) for the usage ledger, using V2's provider rule.

    Empty provider means "do not account" — the demo brain and scripted test
    models have no provider and must never write to the real cost ledger.
    """
    name = str(getattr(model, "model_name", None) or getattr(model, "model", None) or "")
    if not name:
        return "", ""
    base = str(getattr(model, "openai_api_base", None) or getattr(model, "base_url", None) or "")
    return ("DeepSeek" if urlparse(base).hostname == "api.deepseek.com" else "Other LLM"), name


@dataclass
class RunContext:
    run_id: str
    deadline: float
    cancel_event: Any = None
    on_progress: Any = None
    usage: list[dict] = field(default_factory=list)
    lock: Any = field(default_factory=threading.Lock)
    #: Ledger identity of the model behind this run; empty provider disables accounting.
    provider: str = ""
    model: str = ""

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
        usage = dict(getattr(message, "usage_metadata", None) or {})
        with self.lock:
            self.usage.append({"source": source, **usage})
        self._account(source, message, usage)

    def _account(self, source: str, message, usage: dict) -> None:
        """Mirror Agent V2: the cost page only sees what reaches ``v2.data.usage_ledger``.

        LangChain's ``usage_metadata`` is normalised (input/output tokens,
        cache reads, reasoning tokens); the ledger's ``record`` is best-effort
        and never raises.  Runs without a provider (demo, scripted tests) are
        not accounted.
        """
        if not self.provider:
            return
        try:
            from v2.data.usage_ledger import record
        except ImportError:
            return
        meta = getattr(message, "response_metadata", None) or {}
        model = str(meta.get("model_name") or meta.get("model") or self.model)
        cached = (usage.get("input_token_details") or {}).get("cache_read")
        reasoning = (usage.get("output_token_details") or {}).get("reasoning")
        record(
            "llm", self.provider, model,
            dict(input_tokens=usage.get("input_tokens"), output_tokens=usage.get("output_tokens"), cached_tokens=cached, **({"reasoning_tokens": reasoning} if reasoning is not None else {})),
            source=f"agent_v3.{source}", endpoint="chat", usage_basis="reported" if usage else "unknown", requested_model=self.model or None,
        )

    def emit(self, node: str):
        if self.on_progress:
            try:
                from v2.agent_v2.models import ProgressEvent, RunStatus
                self.on_progress(ProgressEvent(self.run_id, RunStatus.EXECUTING, node))
            except Exception:
                pass
