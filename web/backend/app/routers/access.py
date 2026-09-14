"""Browser sign-in, guest sessions and frozen public snapshots."""

from __future__ import annotations

import time
from collections import defaultdict, deque
from threading import RLock
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from app.auth import (
    SESSION_COOKIE,
    Principal,
    auth_configured,
    create_session_token,
    principal_from_request,
    require_access,
    require_owner,
    verify_admin_credentials,
)
from app.config import SETTINGS
from app.public_snapshots import latest_snapshot_time, publish_snapshot, read_snapshot


router = APIRouter(prefix="/api", tags=["access"])
_LOGIN_WINDOW_SECONDS = 15 * 60
_LOGIN_MAX_FAILURES = 8
_FAILURES: dict[str, deque[float]] = defaultdict(deque)
_FAILURE_LOCK = RLock()


class LoginInput(BaseModel):
    username: str = Field(min_length=1, max_length=128)
    password: str = Field(min_length=1, max_length=1024)


class SnapshotInput(BaseModel):
    path: str = Field(min_length=1, max_length=2048)
    payload: Any


def _client_key(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For", "").split(",", 1)[0].strip()
    return forwarded or (request.client.host if request.client else "unknown")


def _rate_limited(key: str) -> bool:
    cutoff = time.monotonic() - _LOGIN_WINDOW_SECONDS
    with _FAILURE_LOCK:
        attempts = _FAILURES[key]
        while attempts and attempts[0] < cutoff:
            attempts.popleft()
        return len(attempts) >= _LOGIN_MAX_FAILURES


def _record_failure(key: str) -> None:
    with _FAILURE_LOCK:
        _FAILURES[key].append(time.monotonic())


def _clear_failures(key: str) -> None:
    with _FAILURE_LOCK:
        _FAILURES.pop(key, None)


def _set_session(response: Response, role: str, subject: str) -> int:
    ttl = int(getattr(SETTINGS, "session_ttl_seconds", 28_800))
    response.set_cookie(
        SESSION_COOKIE,
        create_session_token(role, subject, ttl_seconds=ttl),  # type: ignore[arg-type]
        httponly=True,
        secure=bool(getattr(SETTINGS, "cookie_secure", False)),
        samesite="strict",
        path="/",
    )
    return ttl


def _status(principal: Principal) -> dict:
    return {
        "authenticated": principal.role != "anonymous",
        "role": principal.role,
        "username": principal.subject if principal.role == "owner" else "",
        "guest_enabled": bool(getattr(SETTINGS, "guest_enabled", True)),
        "auth_configured": auth_configured(),
        "snapshot_updated_at": latest_snapshot_time(),
        "session_expires_at": principal.expires_at,
    }


@router.get("/auth/status")
async def access_status(request: Request) -> dict:
    return await run_in_threadpool(_status, principal_from_request(request))


@router.post("/auth/login")
async def login(body: LoginInput, request: Request, response: Response) -> dict:
    key = _client_key(request)
    if _rate_limited(key):
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="too many login attempts; try again later")
    if not await run_in_threadpool(verify_admin_credentials, body.username, body.password):
        _record_failure(key)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid username or password")
    _clear_failures(key)
    _set_session(response, "owner", body.username.strip())
    return _status(Principal(role="owner", subject=body.username.strip()))


@router.post("/auth/guest")
async def enter_guest(response: Response) -> dict:
    if not bool(getattr(SETTINGS, "guest_enabled", True)):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="guest mode is disabled")
    _set_session(response, "guest", "public-guest")
    return _status(Principal(role="guest", subject="public-guest"))


@router.post("/auth/logout")
async def logout(response: Response) -> dict:
    response.delete_cookie(SESSION_COOKIE, path="/")
    return {"authenticated": False, "role": "anonymous"}


@router.post("/public/snapshot")
async def save_public_snapshot(body: SnapshotInput, _: Principal = Depends(require_owner)) -> dict:
    try:
        return await run_in_threadpool(publish_snapshot, body.path, body.payload)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.get("/public/snapshot")
async def get_public_snapshot(path: str, _: Principal = Depends(require_access)) -> dict:
    try:
        snapshot = await run_in_threadpool(read_snapshot, path)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    if snapshot is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="public snapshot has not been published yet")
    return snapshot
