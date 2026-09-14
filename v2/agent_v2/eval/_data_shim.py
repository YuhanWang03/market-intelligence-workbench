"""Let the Research Engine's intelligence layer import without ``v2.data``.

``v2/data`` (FD client, price and earnings models, news provider) is
production-only and git-ignored, so a fresh checkout cannot import
``v2.research`` at all.  Offline fixture synthesis only needs the pure
intelligence layer, which never touches a provider.  This shim installs
placeholder names for the missing package *only when they are absent*; on a
machine with the real package it is a no-op.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from typing import Any


@dataclass
class _Price:
    time: str = ""
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: float = 0.0


class _Placeholder:
    """Stands in for a provider client that offline code must never call."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("v2.data is not available in this checkout; offline evaluation cannot call data providers")


class _ProviderRequestError(RuntimeError):
    error_type = "UNAVAILABLE"


def install() -> bool:
    """Return True when placeholders were installed, False when the real package exists."""

    import v2.data as data

    if hasattr(data, "CachedFDClient"):
        return False
    for name in ("CachedFDClient", "FDClient"):
        setattr(data, name, _Placeholder)
    data.ProviderRequestError = _ProviderRequestError
    data.Price = _Price
    models = types.ModuleType("v2.data.models")
    models.Price = _Price
    models.EarningsRecord = type("EarningsRecord", (object,), {})
    models.EarningsData = type("EarningsData", (object,), {})
    client = types.ModuleType("v2.data.client")
    client.FDClient = _Placeholder
    client.CachedFDClient = _Placeholder
    client.ProviderRequestError = _ProviderRequestError
    news = types.ModuleType("v2.data.news_provider")
    news.NewsProvider = _Placeholder
    news.default_news_provider = _Placeholder
    for name, module in (("v2.data.models", models), ("v2.data.client", client), ("v2.data.news_provider", news)):
        sys.modules.setdefault(name, module)
        setattr(data, name.rsplit(".", 1)[1], module)
    return True
