"""Single-user auth: a static owner token in the X-Owner-Token header.

If WEB_OWNER_TOKEN is unset (local dev), auth is a no-op so the app is easy
to run without ceremony. In production, set the env var and every /api call
must present it.
"""

from __future__ import annotations

from fastapi import Header, HTTPException, status

from app.config import SETTINGS


async def require_owner(x_owner_token: str | None = Header(default=None)) -> None:
    """FastAPI dependency — 401 unless the owner token matches (or auth is off)."""
    if not SETTINGS.owner_token:
        return  # auth disabled (local dev)
    if x_owner_token != SETTINGS.owner_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid or missing X-Owner-Token",
        )
