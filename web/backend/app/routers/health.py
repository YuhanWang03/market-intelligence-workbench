"""Liveness + config sanity endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from app.config import SETTINGS

router = APIRouter(prefix="/api", tags=["meta"])


@router.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "service": "ai-hedge-fund-web",
        "auth_enabled": bool(SETTINGS.owner_token),
        "archive_db_present": SETTINGS.archive_db_path.exists(),
    }
