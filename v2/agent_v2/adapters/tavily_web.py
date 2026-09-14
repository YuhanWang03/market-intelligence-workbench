"""Bounded Tavily search adapter for the opt-in Agent V2 web fallback."""

from __future__ import annotations

import hashlib
from typing import Any
from urllib.parse import urldefrag, urlparse

from v2.agent_v2.models import EvidenceItem, ResultStatus, ToolEnvelope


def _excerpt(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _confidence(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.55
    return min(0.9, max(0.35, score))


class TavilyWebSearchPort:
    """Use the existing news-provider boundary without enabling web by default.

    Search snippets are evidence, but are not presented as full-page reads.  A
    future fetcher can be composed behind this same WebSearchPort contract.
    """

    def __init__(
        self,
        provider=None,
        *,
        max_results: int = 5,
        max_content_chars: int = 1800,
    ) -> None:
        if provider is None:
            from v2.data.news_provider import default_news_provider

            provider = default_news_provider()
        self.provider = provider
        self.max_results = min(8, max(1, int(max_results)))
        self.max_content_chars = min(4000, max(300, int(max_content_chars)))

    def search(
        self,
        query: str,
        *,
        topic: str,
        ticker: str = "",
        recency_days: int = 30,
        run_id: str = "",
    ) -> ToolEnvelope:
        normalized = " ".join((query or "").split())
        if not normalized:
            return ToolEnvelope("web.research", ResultStatus.FAILED, errors=["query is required"])
        if len(normalized) > 500:
            return ToolEnvelope("web.research", ResultStatus.FAILED, errors=["query exceeds 500 characters"])
        days = min(3650, max(1, int(recency_days)))
        raw = self.provider.search(normalized, days=days, max_results=self.max_results) or []
        seen: set[str] = set()
        findings: list[dict[str, Any]] = []
        evidence: list[EvidenceItem] = []
        evidence_prefix = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:8]
        for row in raw:
            if not isinstance(row, dict):
                continue
            url, _ = urldefrag(str(row.get("url") or "").strip())
            parsed = urlparse(url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc or url in seen:
                continue
            seen.add(url)
            title = _excerpt(row.get("title") or parsed.netloc, 240)
            content = _excerpt(row.get("content") or row.get("snippet"), self.max_content_chars)
            if not content:
                continue
            published = str(row.get("published_date") or row.get("published_at") or "")
            finding = {
                "title": title,
                "url": url,
                "content": content,
                "published_at": published,
                "score": row.get("score"),
            }
            findings.append(finding)
            index = len(findings)
            evidence.append(
                EvidenceItem(
                    id=f"web-{evidence_prefix}-{index}",
                    entity=ticker.upper(),
                    claim=content,
                    as_of=published,
                    source_id=f"web:{parsed.netloc.lower()}",
                    source_title=title,
                    source_url=url,
                    confidence=_confidence(row.get("score")),
                    producer_run_id=run_id,
                    metadata={"evidence_type": "search_snippet", "topic": topic},
                )
            )
            if len(findings) >= self.max_results:
                break
        diagnostics = dict(getattr(self.provider, "last_diagnostics", {}) or {})
        if not evidence:
            warning = str(diagnostics.get("error") or diagnostics.get("warning") or "web search returned no usable evidence")
            return ToolEnvelope(
                "web.research",
                ResultStatus.FAILED,
                subject=ticker.upper(),
                errors=[warning],
                metadata={"topic": topic, "diagnostics": diagnostics},
            )
        return ToolEnvelope(
            "web.research",
            ResultStatus.COMPLETED,
            subject=ticker.upper(),
            summary=f"Web search returned {len(evidence)} bounded source snippet(s).",
            findings=findings,
            evidence=evidence,
            limitations=["Evidence contains search-result snippets; source pages were not fetched in this adapter."],
            metadata={
                "topic": topic,
                "query": normalized,
                "recency_days": days,
                "provider": type(self.provider).__name__,
                "diagnostics": diagnostics,
            },
        )
