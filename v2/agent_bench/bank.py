"""Frozen tool responses, keyed by question.

Live data changes by the minute, so two agents answering the same question
minutes apart already see different facts.  A ``record`` run captures every
tool call both agents make for a question (capability + canonical arguments
→ envelope) into one file per case; a ``frozen`` run serves those envelopes
back to whichever agent asks, and answers a call that was never recorded
with an explicit ``fixture_missing`` partial result — never the network.

The bank lives under ``data/agent_bench/bank/`` (git-ignored); its SHA-256
is written into every ledger row so runs can be told apart by data.
"""
from __future__ import annotations

import hashlib
import json
import threading
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from v2.agent_v2.models import ResultStatus, ToolEnvelope

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = _PROJECT_ROOT / "data" / "agent_bench" / "bank"

FAULT_MODES = ("error", "empty", "timeout")


def canonical(capability: str, arguments: Any) -> str:
    return json.dumps([capability, arguments], sort_keys=True, ensure_ascii=False, default=str)


def envelope_to_dict(envelope: ToolEnvelope) -> dict[str, Any]:
    from v2.agent_v2.eval.recorded import envelope_to_dict as convert
    return convert(envelope)


def envelope_from_dict(data: dict[str, Any]) -> ToolEnvelope:
    from v2.agent_v3.contracts import envelope_from
    return envelope_from(deepcopy(data))


def fault_envelope(fault: dict[str, Any], capability: str) -> ToolEnvelope:
    mode = fault.get("mode", "error")
    if mode == "empty":
        return ToolEnvelope(capability, ResultStatus.PARTIAL_DATA, limitations=["评测注入：供应商返回空数据。"], metadata={"injected_fault": mode})
    if mode == "timeout":
        return ToolEnvelope(capability, ResultStatus.FAILED, errors=["评测注入：供应商超时（timeout）。"], metadata={"injected_fault": mode})
    return ToolEnvelope(capability, ResultStatus.FAILED, errors=["评测注入：供应商错误（HTTP 503）。"], metadata={"injected_fault": mode})


class Bank:
    """One JSON file per case: {"case_id", "recorded_at", "records": [{"capability", "arguments", "result"}]}."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root else DEFAULT_ROOT

    def path(self, case_id: str) -> Path:
        return self.root / f"{case_id}.json"

    def load(self, case_id: str) -> list[dict[str, Any]]:
        path = self.path(case_id)
        if not path.exists():
            return []
        return list(json.loads(path.read_text(encoding="utf-8")).get("records", []))

    def save(self, case_id: str, records: list[dict[str, Any]], *, merge: bool = True) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        existing = self.load(case_id) if merge else []
        # Newer recordings win: a failed provider call recorded on a bad day must
        # not shadow the successful one recorded later for the same signature.
        incoming = {(canonical(row["capability"], row["arguments"]), row.get("version") or "") for row in records}
        kept = [row for row in existing if (canonical(row["capability"], row["arguments"]), row.get("version") or "") not in incoming]
        merged = [*kept, *records]
        path = self.path(case_id)
        path.write_text(json.dumps({"case_id": case_id, "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "records": merged}, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
        return path

    def case_ids(self) -> list[str]:
        return sorted(p.stem for p in self.root.glob("*.json")) if self.root.exists() else []

    def sha(self) -> str:
        digest = hashlib.sha256()
        for path in sorted(self.root.glob("*.json")) if self.root.exists() else []:
            digest.update(path.name.encode("utf-8"))
            digest.update(path.read_bytes())
        return digest.hexdigest()[:16]


_ENTITY_KEYS = ("ticker", "tickers", "symbol", "manager", "release_type", "section", "operation")


def _entity_key(arguments: dict[str, Any]) -> str:
    return json.dumps({k: arguments.get(k) for k in _ENTITY_KEYS if k in arguments}, sort_keys=True, ensure_ascii=False, default=str)


def _same_entity(recorded: dict[str, Any], asked: dict[str, Any]) -> bool:
    """Both calls name the same thing (ticker, manager, release…) even if optional arguments differ."""
    keys = [k for k in _ENTITY_KEYS if k in recorded or k in asked]
    if not keys:
        return True  # a capability without an entity argument (macro.overview, account.*) is one call per question
    return all(recorded.get(k) == asked.get(k) for k in keys)


class Replay:
    """Serves recorded envelopes for one case; the active case is switched between runs."""

    def __init__(self, version: str = "") -> None:
        self.version = version
        self.records: list[dict[str, Any]] = []
        self.fault: dict[str, Any] | None = None
        self.calls: list[dict[str, Any]] = []
        self.used: dict[str, int] = {}
        self.lock = threading.Lock()

    def use(self, records: list[dict[str, Any]], *, fault: dict[str, Any] | None = None, fixtures: tuple[dict[str, Any], ...] = ()) -> None:
        # The agent's own recordings first: both agents may call one capability with the
        # same arguments yet shape the envelope differently (V2's sub-agent summary lives
        # in metadata); serving the other agent's envelope would change its behaviour.
        exact = [row for row in records if row.get("version") == self.version]
        unversioned = [row for row in records if not row.get("version")]
        other = [row for row in records if row.get("version") and row.get("version") != self.version]
        with self.lock:
            self.records = [*exact, *unversioned, *other, *fixtures]
            self.fault = fault
            self.calls = []
            self.used = {}

    def missing(self) -> list[dict[str, Any]]:
        return [row for row in self.calls if not row["fixture_match"]]

    def approximate(self) -> list[dict[str, Any]]:
        return [row for row in self.calls if row.get("approximate")]

    def handler(self, capability: str) -> Callable[[dict[str, Any], Any], ToolEnvelope]:
        def call(arguments: dict[str, Any], context: Any) -> ToolEnvelope:
            if self.fault and self.fault.get("capability") == capability:
                with self.lock:
                    self.calls.append({"capability": capability, "arguments": deepcopy(arguments), "fixture_match": True, "injected_fault": self.fault.get("mode")})
                return fault_envelope(self.fault, capability)
            signature = canonical(capability, arguments)
            with self.lock:
                rows = [row for row in self.records if row["capability"] == capability and (row.get("arguments") is None or canonical(capability, row["arguments"]) == signature)]
                approximate = False
                if not rows:
                    # Same question, same capability, same entity, different optional
                    # arguments (a planner asked for top=8 instead of top=5, or phrased a
                    # search query differently): serve the recorded call for that entity
                    # and mark the match approximate rather than fail the whole answer.
                    rows = [row for row in self.records if row["capability"] == capability and row.get("arguments") is not None and _same_entity(row["arguments"], arguments)]
                    approximate = bool(rows)
                key = signature if not approximate else f"~{capability}:{_entity_key(arguments)}"
                index = self.used.get(key, 0)
                self.used[key] = index + 1
                found = index < len(rows)
                self.calls.append({"capability": capability, "arguments": deepcopy(arguments), "fixture_match": found, "approximate": found and approximate})
            if not found:
                return ToolEnvelope(capability, ResultStatus.PARTIAL_DATA, limitations=["冻结评测数据中没有录制这个工具参数；未访问真实数据源。"], metadata={"fixture_missing": True})
            served = replace(envelope_from_dict(rows[index]["result"]), elapsed_ms=0)
            if approximate:
                served.metadata = {**served.metadata, "fixture_approximate": True}
            return served
        return call


class Recorder:
    """Wraps live handlers so every call for the active case is captured for the bank."""

    def __init__(self, version: str = "") -> None:
        self.version = version
        self.records: list[dict[str, Any]] = []
        self.fault: dict[str, Any] | None = None
        self.lock = threading.Lock()

    def use(self, *, fault: dict[str, Any] | None = None) -> None:
        with self.lock:
            self.records = []
            self.fault = fault

    def wrap(self, capability: str, handler: Callable[[dict[str, Any], Any], ToolEnvelope]) -> Callable[[dict[str, Any], Any], ToolEnvelope]:
        def call(arguments: dict[str, Any], context: Any) -> ToolEnvelope:
            if self.fault and self.fault.get("capability") == capability:
                return fault_envelope(self.fault, capability)
            result = handler(arguments, context)
            if isinstance(result, ToolEnvelope) and result.status not in {ResultStatus.FAILED, ResultStatus.SKIPPED}:
                with self.lock:
                    self.records.append({"capability": capability, "arguments": deepcopy(arguments), "version": self.version, "result": envelope_to_dict(result)})
            return result
        return call
