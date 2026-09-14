"""Framework-independent Web facade; FastAPI can wrap this without core imports."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from v2.agent_v2.ports import AgentPort, ProgressSink


@dataclass(frozen=True)
class WebRequest:
    text: str
    session_id: str = ""
    allow_web: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class WebFacade:
    def __init__(self, agent: AgentPort) -> None:
        self.agent = agent

    def handle(self, request: WebRequest, *, on_progress: ProgressSink | None = None) -> dict[str, Any]:
        result = self.agent.run(
            request.text,
            session_id=request.session_id,
            allow_web=request.allow_web,
            on_progress=on_progress,
        )
        return result.to_dict()
