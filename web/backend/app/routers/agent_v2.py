"""Explicit Web entrypoint for the new evidence-first Agent V2."""

from __future__ import annotations

import os
import threading
from v2.usage_context import ContextThread
import uuid
from datetime import datetime, timezone
from typing import Any

from app.auth import require_owner
from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, field_validator

from v2.agent_v2.interfaces.web import WebFacade, WebRequest
from v2.agent_v2.orchestrator import AgentV2Config
from v2.agent_v2.routing import normalize_request, route
from v2.agent_v2.runtime import build_workspace_agent

router = APIRouter(
    prefix="/api/agent-v2",
    tags=["agent-v2"],
    dependencies=[Depends(require_owner)],
)

_AGENT = None
_AGENT_WEB_ENABLED = None
_AGENT_LOCK = threading.Lock()
_JOBS: dict[str, dict[str, Any]] = {}
_JOBS_LOCK = threading.Lock()
_MAX_JOBS = 50


def _enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _get_agent():
    global _AGENT, _AGENT_WEB_ENABLED
    web_enabled = _enabled("AGENT_V2_WEB_ENABLED")
    if _AGENT is None or _AGENT_WEB_ENABLED != web_enabled:
        with _AGENT_LOCK:
            if _AGENT is None or _AGENT_WEB_ENABLED != web_enabled:
                _AGENT = build_workspace_agent(
                    config=AgentV2Config(),
                    enable_web=web_enabled,
                )
                _AGENT_WEB_ENABLED = web_enabled
    return _AGENT


class AgentV2Input(BaseModel):
    text: str = Field(min_length=1, max_length=4000)
    session_id: str = Field(default="web", max_length=128)
    allow_web: bool = False
    background: bool | None = None

    @field_validator("text")
    @classmethod
    def _non_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("text must not be blank")
        return value


def _policy(body: AgentV2Input) -> dict[str, bool]:
    server_web = _enabled("AGENT_V2_WEB_ENABLED")
    return {
        "web_requested": body.allow_web,
        "web_enabled": server_web,
        "web_allowed": body.allow_web and server_web,
    }


def _execute(body: AgentV2Input, on_progress=None) -> dict[str, Any]:
    payload = WebFacade(_get_agent()).handle(
        WebRequest(
            text=body.text,
            session_id=body.session_id,
            allow_web=body.allow_web,
            metadata={"channel": "web"},
        ),
        on_progress=on_progress,
    )
    return {**payload, "interface": "web", "policy": _policy(body)}


def _job_view(job_id: str) -> dict[str, Any]:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        return dict(job) if job else {}


def _run_job(job_id: str, body: AgentV2Input) -> None:
    def progress(event) -> None:
        with _JOBS_LOCK:
            job = _JOBS.get(job_id)
            if job is not None:
                job["agent_status"] = event.status.value
                job["progress"] = event.message

    try:
        result = _execute(body, on_progress=progress)
        with _JOBS_LOCK:
            _JOBS[job_id].update(
                {
                    "status": "completed",
                    "agent_status": result.get("status", "completed"),
                    "result": result,
                    "completed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                }
            )
    except Exception as exc:  # noqa: BLE001 — job failures are returned by the poller
        with _JOBS_LOCK:
            _JOBS[job_id].update(
                {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {str(exc)[:300]}",
                    "completed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                }
            )


def _start_job(body: AgentV2Input) -> dict[str, Any]:
    job_id = uuid.uuid4().hex[:12]
    with _JOBS_LOCK:
        if len(_JOBS) >= _MAX_JOBS:
            oldest = sorted(_JOBS, key=lambda key: _JOBS[key]["created_at"])
            for key in oldest[: max(1, _MAX_JOBS // 2)]:
                if _JOBS[key].get("status") != "running":
                    _JOBS.pop(key, None)
        _JOBS[job_id] = {
            "job_id": job_id,
            "status": "running",
            "agent_status": "queued",
            "progress": "Agent V2 job queued",
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "session_id": body.session_id,
            "policy": _policy(body),
        }
    ContextThread(
        target=_run_job,
        args=(job_id, body),
        name=f"agent-v2-{job_id}",
        daemon=True,
    ).start()
    return _job_view(job_id)


@router.post("/ask")
async def ask_agent_v2(body: AgentV2Input) -> dict[str, Any]:
    decision = route(normalize_request(body.text))
    background = body.background is True or (body.background is None and decision.asynchronous)
    if background:
        return _start_job(body)
    return await run_in_threadpool(_execute, body)


@router.get("/jobs/{job_id}")
async def get_agent_v2_job(job_id: str) -> dict[str, Any]:
    job = _job_view(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Agent V2 job not found")
    return job
