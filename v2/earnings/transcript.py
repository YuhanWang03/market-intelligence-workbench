"""Optional earnings-call transcript lookup via Tavily.

Pure best-effort — the summary card ships fine without a transcript link.
If Tavily is unconfigured or returns nothing, the caller gets ``None``.

We deliberately ask for a small page of results and pick the first that
looks like a transcript host (Seeking Alpha, Motley Fool, the company IR
page, etc.). No scraping — only the URL + Tavily's own snippet are kept.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# Host substrings whose pages we trust to actually contain transcripts.
# Ordered roughly by quality.
_TRANSCRIPT_HOST_HINTS: tuple[str, ...] = (
    "seekingalpha.com",
    "fool.com",
    "investor.",           # corporate IR subdomain
    "ir.",                 # likewise
    "rev.com",
    "earningscall.biz",
    "wallstreetzen.com",
    "transcript",          # last-resort match on the URL path
)


@dataclass(frozen=True)
class TranscriptHit:
    url: str
    snippet: str           # Tavily's short summary — never the full transcript


def find_transcript(
    ticker: str,
    report_period: str,
    *,
    max_results: int = 5,
) -> TranscriptHit | None:
    """Best-effort: find a credible earnings-call transcript URL.

    ``report_period`` is the ISO quarter end (e.g. ``"2025-06-28"``). The
    quarter label ("Q2 2025") is derived for the search query.

    Returns ``None`` if Tavily isn't configured, the call failed, or no
    result matches the transcript-host hints.
    """
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return None

    quarter_label = _quarter_label(report_period)
    query = (
        f'{ticker} {quarter_label} earnings call transcript'
        if quarter_label else
        f'{ticker} earnings call transcript'
    )

    try:
        # Local import — keeps this module importable in environments that
        # haven't installed the tavily SDK (and lets tests stub it cleanly).
        from v2.data.metered import TavilyClient
        client = TavilyClient(api_key=api_key)
        response = client.search(
            query=query,
            max_results=max_results,
            topic="general",
            search_depth="basic",
        )
    except Exception as exc:
        logger.warning("Tavily transcript search failed for %s: %s", ticker, exc)
        return None

    results = response.get("results", []) if isinstance(response, dict) else []
    return _pick_transcript(results)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _pick_transcript(results: list[dict]) -> TranscriptHit | None:
    """Return the first result whose URL host matches a transcript hint."""
    for hint in _TRANSCRIPT_HOST_HINTS:
        for item in results:
            url = (item.get("url") or "").lower()
            if hint in url:
                return TranscriptHit(
                    url=item.get("url") or "",
                    snippet=(item.get("content") or "")[:400],
                )
    return None


_QUARTER_BY_MONTH = {
    1: "Q4", 2: "Q4", 3: "Q4",     # calendar Q1 = fiscal Q4 release window
    4: "Q1", 5: "Q1", 6: "Q1",
    7: "Q2", 8: "Q2", 9: "Q2",
    10: "Q3", 11: "Q3", 12: "Q3",
}


def _quarter_label(report_period: str) -> str:
    """'2025-06-28' → 'Q2 2025'. Empty string on bad input.

    Uses calendar-quarter mapping by the report-period month: any
    report-period in months 4-6 is a Q2 release, regardless of fiscal year.
    Good enough for a search query — Tavily isn't picky.
    """
    m = re.match(r"^(\d{4})-(\d{2})", report_period or "")
    if not m:
        return ""
    year = m.group(1)
    month = int(m.group(2))
    # Map by the month the quarter ENDS in (Apr-Jun → Q2, etc.)
    if 4 <= month <= 6:
        return f"Q2 {year}"
    if 7 <= month <= 9:
        return f"Q3 {year}"
    if 10 <= month <= 12:
        return f"Q4 {year}"
    return f"Q1 {year}"
