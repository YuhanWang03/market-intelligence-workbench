"""Bridge the existing Web quantitative workspace into the Agent V2 LabPort.

The imports are intentionally lazy: Agent V2's core can run without FastAPI or
the Web backend on ``PYTHONPATH``.  The production Web process already exposes
``app.routers.workspace`` and ``app.routers.committee``; tests may inject small
bindings without importing either module.
"""

from __future__ import annotations

import importlib
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from v2.agent_v2.execution import ExecutionContext
from v2.agent_v2.models import EvidenceItem, ResultStatus, ToolEnvelope


@dataclass(frozen=True)
class LabBinding:
    """Validated input model plus an existing deterministic Lab runner."""

    input_model: Callable[..., Any]
    runner: Callable[..., dict[str, Any]]
    supports_progress: bool = False
    history_kind: str = ""


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "result"


def _short(value: Any, limit: int = 180) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _numeric_metrics(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Keep a small, stable metric surface instead of leaking a huge run."""

    preferred = (
        "n_candidates",
        "universe_size",
        "n_events",
        "n_trades",
        "n_periods",
        "total_return_pct",
        "annualized_return_pct",
        "sharpe_ratio",
        "max_drawdown_pct",
        "win_rate",
        "excess_return_pct",
        "fd_cost_usd",
    )
    metrics: dict[str, Any] = {}
    nested = payload.get("metrics")
    sources = (nested if isinstance(nested, Mapping) else {}, payload)
    for key in preferred:
        for source in sources:
            value = source.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                metrics[key] = value
                break
    for key in ("strategy", "universe", "data_source", "group_by"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            metrics[key] = value
    return metrics


def _finding_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    for key in ("candidates", "verdicts", "groups", "rows", "events", "trades"):
        value = payload.get(key)
        if isinstance(value, list):
            return [dict(row) for row in value[:12] if isinstance(row, Mapping)]
    return []


_ROW_LABELS = (("top_n", "top_n"), ("holding_days", "持有期"), ("near_high_pct", "52周高点过滤"), ("total_return_pct", "总收益%"), ("annualized_return_pct", "年化%"), ("max_drawdown_pct", "最大回撤%"), ("sharpe_ratio", "夏普"), ("win_rate", "胜率"), ("n_trades", "交易笔数"), ("benchmark_pct", "基准%"), ("excess_return_pct", "超额%"), ("start", "起"), ("end", "止"))


def _row_claim(row: Mapping[str, Any]) -> str:
    """A result row as one readable line: the known fields labelled in order, anything else as before."""

    parts = []
    for key, label in _ROW_LABELS:
        value = row.get(key)
        if value is None:
            continue
        parts.append(f"{label} {round(value, 2) if isinstance(value, float) else value}")
    if len(parts) < 2:
        return _short(row, 280)
    rest = {key: value for key, value in row.items() if key not in dict(_ROW_LABELS) and key not in ("per_trade", "n_periods", "avg_return_pct")}
    return "；".join(parts) + (f"；{_short(rest, 120)}" if rest else "")


def _limitations(payload: Mapping[str, Any]) -> list[str]:
    out: list[str] = []
    for key in ("price_failures", "skipped", "data_gaps"):
        value = payload.get(key)
        if value:
            out.append(f"{key}: {_short(value, 240)}")
    notes = payload.get("notes")
    if isinstance(notes, Mapping):
        for key in ("errors", "aborted", "no_data", "price_failures"):
            if notes.get(key):
                out.append(f"{key}: {_short(notes[key], 240)}")
    return out[:8]


def _evidence(capability: str, payload: Mapping[str, Any], run_id: str) -> list[EvidenceItem]:
    title = f"Quantitative Lab: {payload.get('kind') or capability}"
    subject = ", ".join(str(value) for value in (payload.get("tickers") or [])[:4])
    items: list[EvidenceItem] = []
    prefix = f"lab-{_slug(run_id)[-16:]}-{_slug(capability)}"
    for index, (metric, value) in enumerate(_numeric_metrics(payload).items(), 1):
        if isinstance(value, str):
            continue
        items.append(
            EvidenceItem(
                id=f"{prefix}-metric-{index}",
                entity=subject,
                claim=f"{metric} = {value}",
                metric=metric,
                value=value,
                source_id=capability,
                source_title=title,
                confidence=1.0,
                producer_run_id=run_id,
                metadata={"evidence_type": "computed_result"},
            )
        )
    for index, row in enumerate(_finding_rows(payload)[:12], 1):
        entity = str(row.get("ticker") or row.get("group") or row.get("name") or subject)
        items.append(
            EvidenceItem(
                id=f"{prefix}-finding-{index}",
                entity=entity,
                claim=_row_claim(row),
                value=dict(row),
                source_id=capability,
                source_title=title,
                confidence=1.0,
                producer_run_id=run_id,
                metadata={"evidence_type": "computed_result"},
            )
        )
    return items


def lab_envelope(capability: str, payload: Mapping[str, Any], run_id: str) -> ToolEnvelope:
    metrics = _numeric_metrics(payload)
    findings = _finding_rows(payload)
    evidence = _evidence(capability, payload, run_id)
    kind = str(payload.get("kind") or capability.removeprefix("lab."))
    subject = ", ".join(str(value) for value in (payload.get("tickers") or [])[:8])
    limitations = _limitations(payload)
    status = ResultStatus.PARTIAL_DATA if limitations else ResultStatus.COMPLETED
    result_counts = {key: len(value) for key, value in payload.items() if isinstance(value, list)}
    return ToolEnvelope(
        capability=capability,
        status=status,
        subject=subject,
        as_of=str(payload.get("date") or payload.get("as_of") or payload.get("universe_as_of") or ""),
        summary=f"{kind} completed; {len(findings)} result row(s), {len(metrics)} summary metric(s).",
        metrics=metrics,
        findings=findings,
        evidence=evidence,
        limitations=limitations,
        run_id=run_id,
        metadata={
            "kind": kind,
            "parameters": dict(payload.get("params") or {}),
            "fd_requests": dict(payload.get("fd_requests") or {}),
            "result_counts": result_counts,
        },
    )


class WorkspaceLabPort:
    """Reuse current Lab runners while preserving Agent V2 contracts."""

    def __init__(self, bindings: Mapping[str, LabBinding] | None = None) -> None:
        self._bindings = dict(bindings) if bindings is not None else None
        self._workspace: Any = None

    def _default_bindings(self) -> dict[str, LabBinding]:
        try:
            workspace = importlib.import_module("app.routers.workspace")
            committee = importlib.import_module("app.routers.committee")
        except ImportError as exc:
            backend = Path(__file__).resolve().parents[3] / "web" / "backend"
            if not backend.is_dir():
                raise RuntimeError("Web Lab backend is unavailable") from exc
            backend_path = str(backend)
            if backend_path not in sys.path:
                sys.path.insert(0, backend_path)
            try:
                workspace = importlib.import_module("app.routers.workspace")
                committee = importlib.import_module("app.routers.committee")
            except ImportError as retry_exc:
                raise RuntimeError("Web Lab backend could not be imported; inject LabBinding objects") from retry_exc
        self._workspace = workspace
        return {
            "lab.screen": LabBinding(workspace.ScreeningInput, workspace._run_screening, True, "screening"),
            "lab.backtest": LabBinding(workspace.BacktestInput, workspace._run_backtest, True, "backtest"),
            "lab.sweep": LabBinding(workspace.SweepInput, workspace._run_sweep, True, "backtest"),
            "lab.event_study": LabBinding(workspace.EventStudyInput, workspace._run_event_study, False, "event_study"),
            "lab.committee": LabBinding(committee.CommitteeInput, committee._run, False, "committee"),
        }

    @property
    def bindings(self) -> dict[str, LabBinding]:
        if self._bindings is None:
            self._bindings = self._default_bindings()
        return self._bindings

    def run(self, capability: str, arguments: dict[str, Any], context: ExecutionContext) -> ToolEnvelope:
        binding = self.bindings.get(capability)
        if binding is None:
            return ToolEnvelope(capability, ResultStatus.FAILED, errors=["unsupported Lab capability"])
        fields = getattr(binding.input_model, "model_fields", None) or {}
        if arguments.get("tickers") and not arguments.get("universe") and "universe" in fields:
            # Named tickers are the universe; the default (an index) would load hundreds of names beside them.
            arguments = {**arguments, "universe": "custom"}
        body = binding.input_model(**arguments)
        if binding.supports_progress:
            payload = binding.runner(
                body,
                on_tick=lambda completed: context.emit(
                    f"{capability}: completed {completed} work unit(s)",
                    capability=capability,
                ),
            )
        else:
            payload = binding.runner(body)
        if not isinstance(payload, Mapping):
            raise TypeError("Lab runner must return a mapping")
        if self._workspace is not None and binding.history_kind:
            remember = getattr(self._workspace, "_remember_run", None)
            if callable(remember):
                remember(binding.history_kind, dict(payload), body.model_dump())
        return lab_envelope(capability, payload, context.run_id)
