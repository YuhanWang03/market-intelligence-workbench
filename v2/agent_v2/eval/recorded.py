"""Record and replay capability envelopes as JSON.

``record_fixtures.py --live`` writes one file per capability under
``v2/agent_v2/eval/recorded/`` from the real adapters; the benchmark's
``engine`` fixture mode replays a recording when one exists for a key and
falls back to the offline synthesis in ``engine_fixtures`` otherwise.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from v2.agent_v2.models import EvidenceItem, ResultStatus, ToolEnvelope

DEFAULT_DIR = Path(__file__).resolve().parent / "recorded"


def envelope_to_dict(envelope: ToolEnvelope) -> dict[str, Any]:
    return envelope.to_dict()


def evidence_from_dict(row: dict[str, Any]) -> EvidenceItem:
    return EvidenceItem(
        id=str(row.get("id") or ""),
        entity=str(row.get("entity") or ""),
        claim=str(row.get("claim") or ""),
        metric=str(row.get("metric") or ""),
        value=row.get("value"),
        unit=str(row.get("unit") or ""),
        period=str(row.get("period") or ""),
        as_of=str(row.get("as_of") or ""),
        source_id=str(row.get("source_id") or ""),
        source_title=str(row.get("source_title") or ""),
        source_url=str(row.get("source_url") or ""),
        confidence=row.get("confidence"),
        producer_run_id=str(row.get("producer_run_id") or ""),
        metadata=dict(row.get("metadata") or {}),
    )


def envelope_from_dict(row: dict[str, Any]) -> ToolEnvelope:
    return ToolEnvelope(
        capability=str(row.get("capability") or ""),
        status=ResultStatus(str(row.get("status") or "failed")),
        subject=str(row.get("subject") or ""),
        as_of=str(row.get("as_of") or ""),
        summary=str(row.get("summary") or ""),
        metrics=dict(row.get("metrics") or {}),
        findings=list(row.get("findings") or []),
        evidence=[evidence_from_dict(item) for item in row.get("evidence") or []],
        limitations=list(row.get("limitations") or []),
        errors=list(row.get("errors") or []),
        run_id=str(row.get("run_id") or ""),
        cache_hit=bool(row.get("cache_hit")),
        elapsed_ms=int(row.get("elapsed_ms") or 0),
        metadata=dict(row.get("metadata") or {}),
    )


class RecordedStore:
    """One JSON file per capability: ``{key: envelope}``."""

    def __init__(self, directory: Path | str | None = None) -> None:
        self.directory = Path(directory) if directory else DEFAULT_DIR
        self._cache: dict[str, dict[str, Any]] = {}

    def _path(self, capability: str) -> Path:
        return self.directory / f"{capability}.json"

    def _table(self, capability: str) -> dict[str, Any]:
        if capability not in self._cache:
            path = self._path(capability)
            self._cache[capability] = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        return self._cache[capability]

    def keys(self, capability: str) -> list[str]:
        return sorted(self._table(capability))

    def load(self, capability: str, key: str) -> ToolEnvelope | None:
        row = self._table(capability).get(key)
        return envelope_from_dict(row) if isinstance(row, dict) else None

    def save(self, capability: str, key: str, envelope: ToolEnvelope) -> None:
        table = self._table(capability)
        table[key] = envelope_to_dict(envelope)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._path(capability).write_text(json.dumps(table, ensure_ascii=False, indent=1, sort_keys=True, default=str), encoding="utf-8")

    def summary(self) -> dict[str, int]:
        if not self.directory.is_dir():
            return {}
        return {path.stem: len(self._table(path.stem)) for path in sorted(self.directory.glob("*.json"))}
