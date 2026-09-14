"""Anomaly monitoring — detect price/volume events and attribute via web search.

Model types are intentionally importable without loading the optional LLM and
search stack.  The full pipeline is imported only when those entry points are
actually requested.
"""

from __future__ import annotations

from typing import Any

from v2.monitoring.models import Anomaly, MonitorConfig, NewsSource

DEFAULT_CONFIG = MonitorConfig()

__all__ = [
    "Anomaly",
    "DEFAULT_CONFIG",
    "MonitorConfig",
    "NewsSource",
    "attribute",
    "detect",
    "run_monitoring",
]


def __getattr__(name: str) -> Any:
    if name == "attribute":
        from v2.monitoring.attributor import attribute

        return attribute
    if name == "detect":
        from v2.monitoring.detectors import detect

        return detect
    if name == "run_monitoring":
        from v2.monitoring.monitor import run_monitoring

        return run_monitoring
    raise AttributeError(name)
