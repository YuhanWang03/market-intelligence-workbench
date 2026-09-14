"""Verify that LLM-suggested neighbors are (a) real tickers and (b) actually
have the relationship the LLM claims (Step 3 of resume polish)."""

from __future__ import annotations

import logging
import os

from v2.data.client import FDClient, ProviderRequestError
from v2.lateral.models import Neighbor

logger = logging.getLogger(__name__)


# Search-query keywords per relation category (chosen for max recall on Tavily)
_RELATION_KEYWORDS = {
    "supplier":     "supplier supply chain",
    "customer":     "customer client",
    "smaller_peer": "competitor peer",
    "beneficiary":  "partnership beneficiary",
}


def verify(neighbor: Neighbor, fd: FDClient, universe: set[str]) -> int:
    """Set exists/sector/already_in_universe on *neighbor*. Returns API calls used."""
    if neighbor.ticker in universe:
        neighbor.exists = True
        neighbor.already_in_universe = True
        return 0

    try:
        facts = fd.get_company_facts(neighbor.ticker)
    except ProviderRequestError as exc:
        if exc.error_type != "EMPTY_DATA":
            raise
        facts = None
    if facts is None:
        neighbor.exists = False
        return 1

    neighbor.exists = True
    neighbor.sector = facts.sector
    return 1


# ---------------------------------------------------------------------------
# Step 3: Tavily relation verification
# ---------------------------------------------------------------------------


def verify_relation(neighbor: Neighbor) -> int:
    """Collect per-edge search evidence, without certifying LLM descriptions.

    Returns attempted Tavily searches, including failures. Directional snippets are retained
    for review; co-mention alone never sets relation_verified.
    """
    from v2.research.relationship_evidence import assess
    neighbor.relation_verified = False
    neighbor.relation_evidence_url = None
    for label in neighbor.labels:
        label.evidence_status = 'NOT_CONNECTED'
        label.evidence_url = None
        label.evidence_text = ''
        label.evidence_title = ''
    api_key = os.environ.get("TAVILY_API_KEY", "").strip()
    if not api_key or not neighbor.labels:
        return 0

    neighbor.relation_checked = True
    try:
        from v2.data.metered import TavilyClient
    except ImportError:
        logger.warning("Tavily relation verification unavailable: tavily package not installed")
        return 0
    client = TavilyClient(api_key=api_key)
    calls = 0

    for label in neighbor.labels:
        label.evidence_status = 'NO_EVIDENCE'
        keywords = _RELATION_KEYWORDS.get(label.category, "")
        query = f"{label.seed} {neighbor.ticker} {keywords}".strip()

        calls += 1  # Logical search attempts, including failures.
        try:
            response = client.search(
                query=query,
                max_results=3,
                topic="general",
                days=365,
                search_depth="basic",
            )
        except Exception:
            label.evidence_status = 'FETCH_FAILED'
            logger.warning("Tavily relation search failed for %s/%s", label.seed, neighbor.ticker)
            continue

        results = response.get("results", []) if response else []

        # Evaluate each edge separately; co-mention never verifies a relation.
        for r in results:
            evidence = assess(label.seed, neighbor.ticker, label.category, r)
            if evidence and (not label.evidence_url or evidence['status'] == 'EVIDENCE_FOUND'):
                label.evidence_status = evidence['status']
                label.evidence_url = evidence['url']
                label.evidence_text = evidence['text']
                label.evidence_title = evidence['title']
                if evidence['status'] == 'EVIDENCE_FOUND':
                    break

    return calls
