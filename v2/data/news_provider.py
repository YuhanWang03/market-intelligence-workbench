"""Pluggable news-search provider — currently backed by Tavily, but the
Protocol lets us swap in Marketaux / GDELT / Polygon / etc. without touching
the downstream agents.

This is the "1 hour future-proofing" investment from the review:
keep the abstraction even if we don't have a second provider today.
"""

from __future__ import annotations

import logging
import os
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class NewsProvider(Protocol):
    """Anything that can return recent news matching a query.

    Returns: list of dicts with these keys at minimum:
        title    str
        content  str   — full snippet, may be long
        url      str
    Additional keys are ignored by downstream consumers.
    """

    def search(
        self,
        query: str,
        *,
        days: int = 7,
        max_results: int = 5,
    ) -> list[dict]: ...


class TavilyNewsProvider:
    """Tavily implementation of NewsProvider.

    Reads TAVILY_API_KEY from env if not given. Empty results on any error.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = (api_key or os.environ.get("TAVILY_API_KEY", "")).strip()
        if not self._api_key:
            logger.warning("TavilyNewsProvider: TAVILY_API_KEY not set")
            self._client = None
        else:
            try:
                from v2.data.metered import TavilyClient
                self._client = TavilyClient(api_key=self._api_key)
            except ImportError:
                logger.warning("TavilyNewsProvider: tavily package not installed")
                self._client = None
        self.last_diagnostics: dict = {}

    def search(
        self,
        query: str,
        *,
        days: int = 7,
        max_results: int = 5,
    ) -> list[dict]:
        self.last_diagnostics = {"query": query, "provider": "Tavily", "raw_count": 0, "filtered_count": 0, "dedup_count": 0, "error": None, "warning": None}
        if self._client is None:
            self.last_diagnostics["warning"] = "TAVILY_API_KEY not set"
            return []
        try:
            response = self._client.search(
                query=query,
                topic="news",
                days=days,
                max_results=max_results,
                search_depth="basic",
            )
        except Exception as exc:
            logger.warning("Tavily search failed for %r: %s", query, exc)
            self.last_diagnostics["error"] = f"{type(exc).__name__}: {exc}"
            return []
        results = response.get("results", []) if response else []
        self.last_diagnostics["raw_count"] = len(results)
        self.last_diagnostics["filtered_count"] = len(results)
        self.last_diagnostics["dedup_count"] = len(results)
        if not results:
            self.last_diagnostics["warning"] = "provider returned zero results"
        return results


# Default singleton — most callers want the standard Tavily provider.
def default_news_provider() -> NewsProvider:
    """Return the default NewsProvider (currently Tavily).

    Swap implementations here when we onboard Marketaux/Polygon/etc.
    """
    return TavilyNewsProvider()
