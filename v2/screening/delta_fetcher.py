"""Dynamic delta context for the screening narrator (改进 ①).

Fetches the three delta signals that turn the narrator from a static-encyclopedia
explainer into a marginal-change analyst:

1. Recent news headlines (NewsProvider, last 7 days, top 3)
2. Earnings surprise — already on the ScreenCandidate (no fetch needed)
3. Peer-relative performance — computed locally via compute_peer_diffs

News fetching is abstracted via NewsProvider Protocol so the implementation
(Tavily today, Marketaux/Polygon tomorrow) is swappable.
"""

from __future__ import annotations

import logging
from statistics import median

from v2.data.news_provider import NewsProvider, default_news_provider
from v2.screening.models import ScreenCandidate

logger = logging.getLogger(__name__)


def fetch_news_headlines(
    ticker: str,
    max_results: int = 3,
    *,
    provider: NewsProvider | None = None,
) -> list[dict]:
    """Top news headlines for *ticker* via the supplied (or default) NewsProvider.

    Returns list of {"title": ..., "snippet": ...} — empty on any failure.
    """
    provider = provider or default_news_provider()
    results = provider.search(
        f"{ticker} stock news",
        days=7,
        max_results=max_results,
    )

    headlines: list[dict] = []
    for item in results[:max_results]:
        title = (item.get("title") or "").strip()
        content = (item.get("content") or "").strip()[:200]
        if not title:
            continue
        headlines.append({"title": title[:120], "snippet": content})
    return headlines


def compute_peer_diffs(candidates: list[ScreenCandidate]) -> None:
    """In-place: compute each candidate's 1-week return vs the cohort median.

    'Cohort' = the set of candidates that passed the screen — this is the
    natural peer group of "quality tech that survived our filter".
    """
    returns = [c.return_1w for c in candidates if c.return_1w is not None]
    if not returns:
        return
    cohort_median = median(returns)
    for c in candidates:
        if c.return_1w is not None:
            c.peer_diff_1w = c.return_1w - cohort_median


def enrich_with_delta(
    candidates: list[ScreenCandidate],
    *,
    provider: NewsProvider | None = None,
) -> int:
    """Populate news_headlines + peer_diff_1w on every candidate.

    *provider* (optional) — inject a different NewsProvider; defaults to
    Tavily via default_news_provider().

    Returns the number of news-API calls made (for cost tracking).
    """
    compute_peer_diffs(candidates)
    provider = provider or default_news_provider()
    api_calls = 0
    for c in candidates:
        c.news_headlines = fetch_news_headlines(c.ticker, provider=provider)
        api_calls += 1
    return api_calls
