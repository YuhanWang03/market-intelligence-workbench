"""Tavily sell-side consensus aggregator for FOMC reactions.

Layer 3 hallucination defense: instead of letting an LLM decide whether
the FOMC outcome was hawkish or dovish, we query Tavily for the day's
sell-side reactions and do a literal keyword count. The published
label is therefore a function of what Goldman / JPM / Reuters /
Bloomberg called it — not what our LLM thinks they should have called
it.

The output ``label`` is one of ``"hawkish"`` / ``"dovish"`` / ``"mixed"``;
``"mixed"`` covers exact ties and the empty-results case.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Callable

logger = logging.getLogger(__name__)


# Trusted news domains — restrict the search so model-generated blog
# spam doesn't dominate the vote. The Tavily client respects these
# via ``include_domains``.
_PREFERRED_DOMAINS: tuple[str, ...] = (
    "reuters.com",
    "bloomberg.com",
    "ft.com",
    "wsj.com",
    "cnbc.com",
    "seekingalpha.com",
    "marketwatch.com",
    "federalreserve.gov",
)


_HAWKISH_RE = re.compile(r"\bhawkish\b", re.IGNORECASE)
_DOVISH_RE = re.compile(r"\bdovish\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Default Tavily client (lazy import so sandbox stays clean)
# ---------------------------------------------------------------------------

def _default_tavily_search(query: str, *, max_results: int = 5) -> list[dict]:
    from v2.data.metered import TavilyClient
    import os

    api_key = os.environ.get("TAVILY_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY not set in env")
    client = TavilyClient(api_key=api_key)
    result = client.search(
        query=query,
        search_depth="basic",
        max_results=max_results,
        include_domains=list(_PREFERRED_DOMAINS),
    )
    return result.get("results", []) or []


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def get_fomc_consensus(
    fomc_date: str,
    *,
    search: Callable[[str, int], list[dict]] | None = None,
    max_results: int = 5,
) -> dict:
    """Query Tavily for sell-side hawkish/dovish reactions to the FOMC
    meeting on ``fomc_date``.

    Args:
        fomc_date: ISO date of the FOMC meeting, e.g. ``"2026-06-17"``.
        search: test seam. Default uses the real Tavily client.
        max_results: how many search results to scan. Tavily costs $$
            per call — 5 is the default sweet spot.

    Returns:
        ``{"hawkish_mentions": int, "dovish_mentions": int,
           "label": "hawkish" | "dovish" | "mixed",
           "sources": [str, ...], "top_headline": str | None}``

        On any sub-failure (Tavily unavailable, empty results) returns
        the ``mixed`` fallback shape so the caller doesn't have to
        special-case error vs. tie.
    """
    query = (
        f"{fomc_date} FOMC reaction hawkish dovish "
        "Goldman JPMorgan Reuters"
    )

    fn = search if search is not None else _default_tavily_search

    try:
        results = fn(query, max_results=max_results)
    except Exception as exc:                          # noqa: BLE001
        logger.warning("tavily_consensus failed: %s", exc)
        return _mixed_fallback(reason=f"tavily error: {exc}")

    if not results:
        return _mixed_fallback(reason="no results")

    hawkish_count = 0
    dovish_count = 0
    sources: list[str] = []
    headlines: list[str] = []

    for r in results:
        title = str(r.get("title") or "")
        content = str(r.get("content") or "")
        url = str(r.get("url") or "")
        blob = f"{title}\n{content}"

        h = len(_HAWKISH_RE.findall(blob))
        d = len(_DOVISH_RE.findall(blob))
        hawkish_count += h
        dovish_count += d

        if title:
            headlines.append(title)
        if url:
            host = _hostname_of(url)
            if host:
                sources.append(host)

    label = _classify(hawkish_count, dovish_count)
    return {
        "hawkish_mentions": hawkish_count,
        "dovish_mentions":  dovish_count,
        "label":            label,
        "sources":          _dedup(sources),
        "top_headline":     headlines[0] if headlines else None,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _classify(hawkish: int, dovish: int) -> str:
    if hawkish == 0 and dovish == 0:
        return "mixed"
    if hawkish > dovish:
        return "hawkish"
    if dovish > hawkish:
        return "dovish"
    return "mixed"


def _mixed_fallback(*, reason: str) -> dict:
    return {
        "hawkish_mentions": 0,
        "dovish_mentions":  0,
        "label":            "mixed",
        "sources":          [],
        "top_headline":     None,
        "_reason":          reason,
    }


def _hostname_of(url: str) -> str | None:
    """Extract the host portion of a URL without using urllib (keeps
    this module dependency-light)."""
    if "//" not in url:
        return None
    after_scheme = url.split("//", 1)[1]
    host = after_scheme.split("/", 1)[0]
    return host or None


def _dedup(items: list[str]) -> list[str]:
    seen: dict[str, None] = {}
    for it in items:
        if it not in seen:
            seen[it] = None
    return list(seen.keys())


__all__ = [
    "get_fomc_consensus",
]
