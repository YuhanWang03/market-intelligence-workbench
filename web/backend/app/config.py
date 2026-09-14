"""Runtime configuration for the new web backend.

Single-user by design: one owner token, no guest/budget machinery (that
lived in the old dashboard/ and was frozen). Everything is read from env
with sensible local-dev defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Repo root = .../market-intelligence-workbench (this file is web/backend/app/config.py)
_REPO_ROOT = Path(__file__).resolve().parents[3]

# Load the repo's .env so the v2 responders / broker see their credentials
# (FINANCIAL_DATASETS / DEEPSEEK / APCA_* / FRED / TAVILY). uvicorn doesn't
# do this for us. Existing os.environ values win (override=False).
load_dotenv(_REPO_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    owner_token: str            # required in prod; empty disables auth (local dev)
    archive_db_path: Path       # v2's push archive (read-only feed)
    repo_root: Path
    cors_origins: tuple[str, ...]


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def load_settings() -> Settings:
    origins = _env(
        "WEB_CORS_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173",
    )
    return Settings(
        owner_token=_env("WEB_OWNER_TOKEN"),
        archive_db_path=Path(_env("WEB_ARCHIVE_DB", str(_REPO_ROOT / "data" / "archive.db"))),
        repo_root=_REPO_ROOT,
        cors_origins=tuple(o.strip() for o in origins.split(",") if o.strip()),
    )


SETTINGS = load_settings()
