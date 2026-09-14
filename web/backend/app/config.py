"""Runtime configuration for the web backend.

The owner token remains available for trusted service-to-service calls.  The
browser uses a signed session cookie and can enter either owner or read-only
guest mode.  Empty authentication settings are only accepted as a local-dev
convenience; production should always configure a token and session secret.
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
    admin_username: str
    admin_password_hash: str
    session_secret: str
    session_ttl_seconds: int
    cookie_secure: bool
    guest_enabled: bool
    archive_db_path: Path       # v2's push archive (read-only feed)
    public_snapshot_db_path: Path
    repo_root: Path
    cors_origins: tuple[str, ...]


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _bool_env(name: str, default: bool = False) -> bool:
    value = _env(name)
    if not value:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(_env(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))


def load_settings() -> Settings:
    origins = _env(
        "WEB_CORS_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173",
    )
    return Settings(
        owner_token=_env("WEB_OWNER_TOKEN"),
        admin_username=_env("WEB_ADMIN_USERNAME", "owner"),
        admin_password_hash=_env("WEB_ADMIN_PASSWORD_HASH"),
        session_secret=_env("WEB_SESSION_SECRET"),
        session_ttl_seconds=_int_env("WEB_SESSION_TTL_SECONDS", 28_800, 900, 604_800),
        cookie_secure=_bool_env("WEB_COOKIE_SECURE", False),
        guest_enabled=_bool_env("WEB_GUEST_ENABLED", True),
        archive_db_path=Path(_env("WEB_ARCHIVE_DB", str(_REPO_ROOT / "data" / "archive.db"))),
        public_snapshot_db_path=Path(_env("WEB_PUBLIC_SNAPSHOT_DB", str(_REPO_ROOT / "data" / "public_snapshots.db"))),
        repo_root=_REPO_ROOT,
        cors_origins=tuple(o.strip() for o in origins.split(",") if o.strip()),
    )


SETTINGS = load_settings()
