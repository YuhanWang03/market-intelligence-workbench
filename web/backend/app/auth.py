"""Owner/guest authentication for the web application.

Browsers receive a short-lived signed HttpOnly cookie. ``WEB_OWNER_TOKEN``
is still accepted in ``X-Owner-Token`` for trusted local services, Telegram
and operational smoke tests, but is no longer intended to be copied into a
browser. Guest sessions are read-only and can only consume explicitly
published snapshots.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from dataclasses import dataclass
from typing import Literal

from fastapi import Header, HTTPException, Request, status

from app.config import SETTINGS


SESSION_COOKIE = "workbench_session"
Role = Literal["owner", "guest", "anonymous"]


@dataclass(frozen=True)
class Principal:
    role: Role
    subject: str = ""
    expires_at: int | None = None


def auth_configured() -> bool:
    return bool(getattr(SETTINGS, "owner_token", "") or getattr(SETTINGS, "admin_password_hash", ""))


def _session_secret() -> bytes:
    configured = getattr(SETTINGS, "session_secret", "")
    owner_token = getattr(SETTINGS, "owner_token", "")
    # Local development without auth remains frictionless. Production is
    # reported unhealthy when both values are missing.
    return (configured or owner_token or "market-intelligence-workbench-local-dev").encode("utf-8")


def _b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _b64decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def create_session_token(role: Literal["owner", "guest"], subject: str, *, ttl_seconds: int | None = None) -> str:
    now = int(time.time())
    ttl = ttl_seconds or int(getattr(SETTINGS, "session_ttl_seconds", 28_800))
    payload = {
        "v": 1,
        "role": role,
        "sub": subject[:128],
        "iat": now,
        "exp": now + ttl,
        "nonce": secrets.token_urlsafe(12),
    }
    encoded = _b64encode(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8"))
    signature = _b64encode(hmac.new(_session_secret(), encoded.encode("ascii"), hashlib.sha256).digest())
    return f"{encoded}.{signature}"


def decode_session_token(token: str) -> Principal | None:
    try:
        encoded, supplied = token.split(".", 1)
        expected = _b64encode(hmac.new(_session_secret(), encoded.encode("ascii"), hashlib.sha256).digest())
        if not hmac.compare_digest(supplied, expected):
            return None
        payload = json.loads(_b64decode(encoded))
        role = payload.get("role")
        expires_at = int(payload.get("exp") or 0)
        if payload.get("v") != 1 or role not in {"owner", "guest"} or expires_at <= int(time.time()):
            return None
        return Principal(role=role, subject=str(payload.get("sub") or ""), expires_at=expires_at)
    except (ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def make_password_hash(password: str, *, iterations: int = 310_000) -> str:
    """Return a portable PBKDF2 hash suitable for WEB_ADMIN_PASSWORD_HASH."""
    if len(password) < 12:
        raise ValueError("password must contain at least 12 characters")
    salt = secrets.token_bytes(18)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"pbkdf2_sha256${iterations}${_b64encode(salt)}${_b64encode(digest)}"


def _verify_hash(password: str, encoded: str) -> bool:
    try:
        algorithm, raw_iterations, raw_salt, raw_digest = encoded.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        iterations = int(raw_iterations)
        if iterations < 100_000 or iterations > 2_000_000:
            return False
        digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), _b64decode(raw_salt), iterations)
        return hmac.compare_digest(digest, _b64decode(raw_digest))
    except (ValueError, TypeError):
        return False


def verify_admin_credentials(username: str, password: str) -> bool:
    expected_username = getattr(SETTINGS, "admin_username", "owner") or "owner"
    if not hmac.compare_digest(username.strip(), expected_username):
        return False
    password_hash = getattr(SETTINGS, "admin_password_hash", "")
    if password_hash:
        return _verify_hash(password, password_hash)
    # Backwards-compatible bootstrap: until a dedicated password hash is set,
    # the existing owner token is the owner's password. It never enters
    # localStorage and should be replaced with a dedicated password later.
    owner_token = getattr(SETTINGS, "owner_token", "")
    return bool(owner_token) and hmac.compare_digest(password, owner_token)


def principal_from_request(request: Request, x_owner_token: str | None = None) -> Principal:
    if not auth_configured():
        return Principal(role="owner", subject="local-development")
    token = x_owner_token or request.headers.get("X-Owner-Token")
    owner_token = getattr(SETTINGS, "owner_token", "")
    if token and owner_token and hmac.compare_digest(token, owner_token):
        return Principal(role="owner", subject="service-token")
    session = request.cookies.get(SESSION_COOKIE, "")
    return decode_session_token(session) or Principal(role="anonymous")


def agent_session_id(request: Request, requested: str = "") -> str:
    """The conversation key an agent uses when the caller names none.

    The browser names one per tab; scripts and older clients often do not, and
    a shared default key ("web") let every such caller share one pending
    confirmation, clarification and evidence frame.  A signed-in browser
    session maps to a stable key derived from its cookie; anything else gets
    a fresh key per request.
    """
    requested = (requested or "").strip()
    if requested:
        return requested
    cookie = request.cookies.get(SESSION_COOKIE, "")
    if cookie:
        return "web:" + hashlib.sha256(cookie.encode("utf-8")).hexdigest()[:16]
    return "web:" + secrets.token_hex(6)


async def require_owner(request: Request, x_owner_token: str | None = Header(default=None)) -> Principal:
    principal = principal_from_request(request, x_owner_token)
    if principal.role != "owner":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED if principal.role == "anonymous" else status.HTTP_403_FORBIDDEN,
            detail="owner authentication required",
        )
    return principal


async def require_access(request: Request, x_owner_token: str | None = Header(default=None)) -> Principal:
    principal = principal_from_request(request, x_owner_token)
    if principal.role == "anonymous":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="sign in or choose guest mode")
    return principal
