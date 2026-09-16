"""Explicit Web entrypoint for the new evidence-first Agent V3."""

from __future__ import annotations

import os
import threading
from v2.usage_context import ContextThread
import uuid
from datetime import datetime, timezone
from typing import Any

from app.auth import agent_session_id, require_owner
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, field_validator

from v2.agent_v3.page_context import PageContext
from v2.agent_v3.graph import AgentV3Config
from v2.agent_v3.runtime import build_workspace_agent

router = APIRouter(
    prefix="/api/agent-v3",
    tags=["agent-v3"],
    dependencies=[Depends(require_owner)],
)

_AGENT = None
_AGENT_WEB_ENABLED = None
_AGENT_MUTATIONS_ENABLED = None
_AGENT_LOCK = threading.Lock()
_JOBS: dict[str, dict[str, Any]] = {}
_JOBS_LOCK = threading.Lock()
_MAX_JOBS = 50
_INSTANCE = uuid.uuid4().hex


def _journal():
    from pathlib import Path
    from v2.agent_v3.jobs import JobJournal
    root = Path(os.environ.get("AGENT_V3_DATA_DIR") or Path(__file__).resolve().parents[4] / "data" / "agent_v3")
    return JobJournal(root / "web_jobs.sqlite")


def _enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _get_agent():
    global _AGENT, _AGENT_WEB_ENABLED, _AGENT_MUTATIONS_ENABLED
    web_enabled = _enabled("AGENT_V3_WEB_ENABLED")
    # Writes (watchlist, alerts) stay unregistered unless the server opts in;
    # even then every write stops at the graph's confirmation node and only
    # the confirm endpoint below can resume it.
    mutations_enabled = _enabled("AGENT_V3_MUTATIONS_ENABLED")
    if _AGENT is None or (_AGENT_WEB_ENABLED, _AGENT_MUTATIONS_ENABLED) != (web_enabled, mutations_enabled):
        with _AGENT_LOCK:
            if _AGENT is None or (_AGENT_WEB_ENABLED, _AGENT_MUTATIONS_ENABLED) != (web_enabled, mutations_enabled):
                _AGENT = build_workspace_agent(
                    config=AgentV3Config(enable_web=web_enabled),
                    enable_mutations=mutations_enabled,
                )
                _AGENT_WEB_ENABLED, _AGENT_MUTATIONS_ENABLED = web_enabled, mutations_enabled
    return _AGENT


class AgentV3Input(BaseModel):
    text: str = Field(min_length=1, max_length=4000)
    session_id: str = Field(default="", max_length=128)  # empty: derived per caller, see auth.agent_session_id
    allow_web: bool = False
    background: bool | None = None
    page_context: PageContext | None = None

    @field_validator("text")
    @classmethod
    def _non_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("text must not be blank")
        return value


def _policy(body: AgentV3Input) -> dict[str, bool]:
    server_web = _enabled("AGENT_V3_WEB_ENABLED")
    return {
        "web_requested": body.allow_web,
        "web_enabled": server_web,
        "web_allowed": body.allow_web and server_web,
        "mutations_enabled": _enabled("AGENT_V3_MUTATIONS_ENABLED"),
    }


class AgentV3Confirmation(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    approve: bool


def _execute(body: AgentV3Input, on_progress=None, resume_run=None) -> dict[str, Any]:
    if resume_run:
        payload = _get_agent().recover(session_id=body.session_id,run_id=resume_run,allow_web=body.allow_web,on_progress=on_progress).to_dict()
        return {**payload, "interface":"web", "policy":_policy(body)}
    payload = _get_agent().run(
        body.text,
        session_id=body.session_id,
        allow_web=body.allow_web,
        page_context=body.page_context.model_dump(mode="json") if body.page_context else None,
        on_progress=on_progress,
    ).to_dict()
    return {**payload, "interface": "web", "policy": _policy(body)}


def _job_view(job_id: str) -> dict[str, Any]:
    with _JOBS_LOCK:
        job = dict(_JOBS.get(job_id) or {})
    if not job:
        job = _journal().get(job_id)
        if job.get("status") == "running" and job.get("instance") != _INSTANCE:
            job.update(status="failed",error="服务重启中断了任务；可重新提交。",restartable=True)
    return {key:value for key,value in job.items() if key not in {"request","instance"}}


def _run_job(job_id: str, body: AgentV3Input) -> None:
    def progress(event) -> None:
        with _JOBS_LOCK:
            job = _JOBS.get(job_id)
            if job is not None:
                if not job.get("run_id"):
                    job["run_id"] = event.run_id
                    _journal().save(job_id,job)
                job["agent_status"] = event.status.value
                job["progress"] = event.message

    try:
        result = _execute(body, on_progress=progress, resume_run=_JOBS[job_id].get("resume_run"))
        with _JOBS_LOCK:
            _JOBS[job_id].update(
                {
                    "status": "completed",
                    "agent_status": result.get("status", "completed"),
                    "result": result,
                    "completed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                }
            )
            _journal().save(job_id,_JOBS[job_id])
    except Exception as exc:  # noqa: BLE001 — job failures are returned by the poller
        with _JOBS_LOCK:
            _JOBS[job_id].update(
                {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {str(exc)[:300]}",
                    "completed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                }
            )
            _journal().save(job_id,_JOBS[job_id])


def _start_job(body: AgentV3Input, resume_run=None) -> dict[str, Any]:
    job_id = uuid.uuid4().hex[:12]
    with _JOBS_LOCK:
        if len(_JOBS) >= _MAX_JOBS:
            oldest = sorted(_JOBS, key=lambda key: _JOBS[key]["created_at"])
            for key in oldest[: max(1, _MAX_JOBS // 2)]:
                if _JOBS[key].get("status") != "running":
                    _JOBS.pop(key, None)
        if sum(job.get("status") == "running" for job in _JOBS.values()) >= 4:
            raise HTTPException(status_code=429, detail="Too many active Agent V3 jobs")
        _JOBS[job_id] = {
            "instance": _INSTANCE,
            "resume_run": resume_run,
            "request": body.model_dump(mode="json"),
            "job_id": job_id,
            "status": "running",
            "agent_status": "queued",
            "progress": "Agent V3 job queued",
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "session_id": body.session_id,
            "policy": _policy(body),
        }
        _journal().save(job_id,_JOBS[job_id])
    ContextThread(
        target=_run_job,
        args=(job_id, body),
        name=f"agent-v3-{job_id}",
        daemon=True,
    ).start()
    return _job_view(job_id)


@router.post("/ask")
async def ask_agent_v3(body: AgentV3Input, request: Request) -> dict[str, Any]:
    body = body.model_copy(update={"session_id": agent_session_id(request, body.session_id)})
    background = body.background is not False
    if background:
        return _start_job(body)
    return await run_in_threadpool(_execute, body)


@router.get("/jobs/{job_id}")
async def get_agent_v3_job(job_id: str) -> dict[str, Any]:
    job = _job_view(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Agent V3 job not found")
    return job


@router.post("/runs/{run_id}/confirm")
async def confirm_agent_v3_run(run_id: str, body: AgentV3Confirmation) -> dict[str, Any]:
    """Approve or reject the write a run stopped on; resumes the graph from its checkpoint."""
    def resume():
        return _get_agent().resume(session_id=body.session_id, run_id=run_id, approve=body.approve).to_dict()
    try:
        payload = await run_in_threadpool(resume)
    except PermissionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {**payload, "interface": "web", "policy": _policy(AgentV3Input(text="confirm", session_id=body.session_id))}


@router.post("/jobs/{job_id}/retry")
async def retry_agent_v3_job(job_id: str) -> dict[str, Any]:
    view = _job_view(job_id)
    if not view or view.get("status") != "failed":
        raise HTTPException(status_code=409,detail="Only failed or interrupted jobs can be retried")
    saved = _journal().get(job_id)
    if not saved.get("request"):
        raise HTTPException(status_code=404,detail="Saved request unavailable")
    body = AgentV3Input.model_validate(saved["request"])
    return _start_job(body,resume_run=saved["run_id"]) if saved.get("run_id") else _start_job(body)
