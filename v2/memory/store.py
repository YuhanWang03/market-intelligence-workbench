"""Long-term memory for anomalies via ChromaDB + OpenAI embeddings (Phase C).

When a new anomaly fires:
    1. recall(ticker, query) — find semantically similar past anomalies
       on the same ticker within the lookback window
    2. remember(anomaly) — index the current anomaly for future recalls

Storage:
    - ChromaDB persistent collection at data/chroma/
    - Embeddings via OpenAI text-embedding-3-small ($0.02 per 1M tokens)
    - Deterministic IDs (ticker_date) for natural upsert / dedup
"""

from __future__ import annotations

import logging
import os
from datetime import date, timedelta
from pathlib import Path

from v2.monitoring.models import Anomaly, HistoricalAnomaly

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CHROMA_DIR = _PROJECT_ROOT / "data" / "chroma"

_EMBED_MODEL = "text-embedding-3-small"
_COLLECTION_NAME = "anomalies"


class AnomalyMemory:
    """ChromaDB-backed similarity memory of past anomalies.

    Lazy imports of chromadb so an import error doesn't break tests that
    don't actually need memory.
    """

    def __init__(self) -> None:
        import chromadb
        from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not set — required for AnomalyMemory embedding"
            )

        _CHROMA_DIR.mkdir(parents=True, exist_ok=True)

        self._embed_fn = OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name=_EMBED_MODEL,
        )
        self._client = chromadb.PersistentClient(path=str(_CHROMA_DIR))
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            embedding_function=self._embed_fn,
        )

    # ------------------------------------------------------------------
    # Write side
    # ------------------------------------------------------------------

    def remember(self, anomaly: Anomaly, *, doc_id: str | None = None, metadata: dict | None = None) -> str:
        """Index *anomaly* in the collection. Returns the deterministic id.

        ``doc_id`` lets a retrospective attribution (the agent explaining a
        past day) sit beside the monitor's own record instead of replacing it.
        ``metadata`` adds scalar fields (governance: confidence, version,
        when it was written) next to the standard ones.
        """
        doc_text = self._build_document(anomaly)
        doc_id = doc_id or f"{anomaly.ticker}_{anomaly.date}"
        # ChromaDB 1.x requires numeric values for $gte / $lte, so we store an
        # int representation alongside the human-readable date string.
        metadata = {
            **{key: value for key, value in (metadata or {}).items() if isinstance(value, (str, int, float, bool))},
            "ticker": anomaly.ticker,
            "date": anomaly.date,                  # for display
            "date_int": _date_to_int(anomaly.date), # for numeric range filter
            "flags": ",".join(anomaly.flags),
        }
        try:
            self._collection.upsert(
                documents=[doc_text],
                metadatas=[metadata],
                ids=[doc_id],
            )
        except Exception as exc:
            logger.warning("ChromaDB upsert failed for %s: %s", doc_id, exc)
        return doc_id

    # ------------------------------------------------------------------
    # Read side
    # ------------------------------------------------------------------

    def get(self, doc_id: str) -> HistoricalAnomaly | None:
        """The record stored under ``doc_id``, with its metadata, or None."""

        try:
            found = self._collection.get(ids=[doc_id])
        except Exception as exc:
            logger.warning("ChromaDB get failed for %s: %s", doc_id, exc)
            return None
        metas = (found or {}).get("metadatas") or []
        docs = (found or {}).get("documents") or []
        if not metas or not docs or not isinstance(metas[0], dict):
            return None
        meta = metas[0]
        return HistoricalAnomaly(date=str(meta.get("date", "")), flags=str(meta.get("flags", "")), doc=str(docs[0] or ""), metadata={key: value for key, value in meta.items() if key not in {"date_int"}})

    def recall(
        self,
        ticker: str,
        query_text: str,
        *,
        lookback_days: int = 30,
        n_results: int = 3,
        exclude_date: str | None = None,
    ) -> list[HistoricalAnomaly]:
        """Find past anomalies for *ticker* semantically similar to *query_text*.

        Filters: ticker exact match + date within last *lookback_days*.
        Optionally excludes a specific date (used to skip "today" so we don't
        retrieve the anomaly we just stored).
        """
        cutoff_int = _date_to_int(
            (date.today() - timedelta(days=lookback_days)).isoformat()
        )

        # ChromaDB 1.x requires numeric operands for $gte / $ne ranges.
        where_filters: list[dict] = [
            {"ticker": ticker},
            {"date_int": {"$gte": cutoff_int}},
        ]
        if exclude_date:
            where_filters.append({"date_int": {"$ne": _date_to_int(exclude_date)}})

        where = (
            {"$and": where_filters} if len(where_filters) > 1 else where_filters[0]
        )

        try:
            results = self._collection.query(
                query_texts=[query_text],
                n_results=n_results,
                where=where,
            )
        except Exception as exc:
            logger.warning("ChromaDB query failed for %s: %s", ticker, exc)
            return []

        history: list[HistoricalAnomaly] = []
        metas = (results or {}).get("metadatas") or []
        docs = (results or {}).get("documents") or []
        if not metas or not docs:
            return []

        for meta, doc in zip(metas[0], docs[0]):
            if not isinstance(meta, dict):
                continue
            history.append(HistoricalAnomaly(
                date=str(meta.get("date", "")),
                flags=str(meta.get("flags", "")),
                doc=str(doc or ""),
                metadata={key: value for key, value in meta.items() if key not in {"date_int"}},
            ))
        return history

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_document(anomaly: Anomaly) -> str:  # noqa: D401
        """Build the embeddable text representation of an anomaly.

        Includes ticker, flags, and the Verifier-approved reasons (high/medium
        only — we don't index 'low' confidence noise). Falls back to flag-only
        if no reasons.
        """
        flag_text = " ".join(anomaly.flags)
        good_reasons = [
            r.text for r in anomaly.reasons
            if r.confidence in ("高", "中")
        ]
        reason_text = " ".join(good_reasons) if good_reasons else ""

        parts = [anomaly.ticker, flag_text, reason_text]
        return " ".join(p for p in parts if p).strip() or anomaly.ticker


def _date_to_int(d: str) -> int:
    """'2026-05-28' → 20260528 — for ChromaDB numeric range filtering."""
    try:
        return int(d.replace("-", ""))
    except (ValueError, AttributeError):
        return 0
