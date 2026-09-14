"""Evidence ledger shared by tool execution, synthesis, and verification."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace

from v2.agent_v2.models import EvidenceItem, ToolEnvelope


class EvidenceConflictError(ValueError):
    """Kept for callers that still catch it; the ledger no longer raises it."""


def same_fact(left: EvidenceItem, right: EvidenceItem) -> bool:
    """Two items state the same fact even if different runs produced them.

    The Research Engine derives evidence ids from claim content, so two runs
    for one ticker (two focuses in one plan, or a cached and a fresh run)
    legitimately reuse an id.  Only a *different* claim under the same id is
    a conflict.
    """

    return (
        left.entity == right.entity
        and left.claim == right.claim
        and left.metric == right.metric
        and left.value == right.value
        and left.unit == right.unit
        and left.period == right.period
        and left.source_id == right.source_id
    )


class EvidenceLedger:
    """Every fact the run collected, keyed by a unique citation id.

    A different claim under an id the ledger already holds is not fatal: two
    Research Engine runs for one ticker made at different times carry
    refreshed numbers under content-derived ids.  The newcomer is reissued a
    unique id (``<id>~<producer>``), both facts stay citeable, and the
    reissue is recorded in ``reissued`` so the result can disclose it.
    """

    def __init__(self) -> None:
        self._items: dict[str, EvidenceItem] = {}
        #: (original id, reissued id) pairs, in ingestion order.
        self.reissued: list[tuple[str, str]] = []

    def add(self, item: EvidenceItem) -> EvidenceItem:
        """Store ``item`` and return the stored instance (its id may be reissued)."""

        if not item.id:
            raise ValueError("evidence id is required")
        current = self._items.get(item.id)
        if current is None:
            self._items[item.id] = item
            return item
        if current == item or same_fact(current, item):
            return current  # first producer wins; the fact is identical
        suffix = item.producer_run_id or str(sum(1 for original, _ in self.reissued if original == item.id) + 2)
        candidate = f"{item.id}~{suffix}"
        sequence = 2
        while candidate in self._items and not same_fact(self._items[candidate], item):
            candidate = f"{item.id}~{suffix}-{sequence}"
            sequence += 1
        if candidate in self._items:
            return self._items[candidate]
        reissued = replace(item, id=candidate, metadata={**item.metadata, "original_evidence_id": item.id, "reissued": True})
        self._items[candidate] = reissued
        self.reissued.append((item.id, candidate))
        return reissued

    def extend(self, items: list[EvidenceItem]) -> list[EvidenceItem]:
        return [self.add(item) for item in items]

    def ingest(self, envelope: ToolEnvelope) -> None:
        """Add an envelope's evidence and rewrite it with the ids the ledger holds."""

        stored = self.extend(envelope.evidence)
        if any(kept.id != original.id for kept, original in zip(stored, envelope.evidence)):
            renamed = [f"{original.id}→{kept.id}" for kept, original in zip(stored, envelope.evidence) if kept.id != original.id]
            envelope.evidence[:] = stored
            envelope.limitations.append("证据 id 与本次其他结果冲突，已重新编号：" + ", ".join(renamed[:4]))

    def get(self, evidence_id: str) -> EvidenceItem | None:
        return self._items.get(evidence_id)

    def ids(self) -> set[str]:
        return set(self._items)

    def items(self) -> list[EvidenceItem]:
        return list(self._items.values())

    def by_entity(self) -> dict[str, list[EvidenceItem]]:
        grouped: dict[str, list[EvidenceItem]] = defaultdict(list)
        for item in self._items.values():
            grouped[item.entity].append(item)
        return dict(grouped)
