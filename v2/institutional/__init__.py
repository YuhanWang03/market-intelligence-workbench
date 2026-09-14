"""Institutional 13F tracking — 玩法 ④b."""

from v2.institutional.managers import MANAGERS
from v2.institutional.models import (
    ChangeType,
    Filing,
    InstitutionalReport,
    Position,
    PositionChange,
)


def run_institutional_pipeline(*args, **kwargs):
    """Load the optional EDGAR client only for an actual 13F run."""
    from v2.institutional.orchestrator import run_institutional_pipeline as _run
    return _run(*args, **kwargs)

__all__ = [
    "ChangeType",
    "Filing",
    "InstitutionalReport",
    "MANAGERS",
    "Position",
    "PositionChange",
    "run_institutional_pipeline",
]
