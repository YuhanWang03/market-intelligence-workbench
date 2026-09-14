"""Capability contracts and complete schema validation, separate from scheduling."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from contextvars import copy_context
from dataclasses import replace
from typing import Any

from jsonschema import Draft202012Validator

from v2.agent_v2.catalog import default_catalog
from v2.agent_v2.execution import ExecutionContext
from v2.agent_v2.models import NormalizedRequest, PlanTask, ResultStatus, ToolEnvelope
from v2.agent_v3.context import RunStopped


class Registry:
    def __init__(self, catalog=None):
        self.catalog = catalog or default_catalog()
        self.handlers = {}

    def register(self, name, handler):
        if self.catalog.get(name) is None:
            raise ValueError(f"Undeclared capability: {name}")
        self.handlers[name] = handler

    def registered(self, name):
        return name in self.handlers

    def validate(self, task: PlanTask, *, template=False):
        spec = self.catalog.get(task.capability)
        if spec is None:
            raise ValueError(f"Unknown capability: {task.capability}")
        schema = dict(spec.input_schema)
        if template and task.fan_out:
            argument = task.fan_out.get("argument")
            if argument not in schema.get("properties", {}):
                raise ValueError("fan_out target argument is not in the capability schema")
            schema["required"] = [key for key in schema.get("required", []) if key != argument]
        Draft202012Validator(schema).validate(task.arguments)
        if spec.mutating:
            operation, payload = task.arguments.get("operation"), task.arguments.get("payload", {})
            # V2's generic payload schema is tightened here before confirmation/execution.
            ticker = {"type": "string", "pattern": "^[A-Z][A-Z0-9.-]{0,7}$"}
            schemas = {
                "watchlist.add": ({"ticker": ticker}, ["ticker"]),
                "watchlist.remove": ({"ticker": ticker}, ["ticker"]),
                "alert.add": ({"ticker": ticker, "direction": {"enum": ["above", "below"]}, "target_price": {"type": "number", "exclusiveMinimum": 0}}, ["ticker", "direction", "target_price"]),
                "alert.remove": ({"alert_id": {"type": "integer", "minimum": 1}}, ["alert_id"]),
            }
            if operation not in schemas:
                raise ValueError("Unsupported state mutation")
            properties, required = schemas[operation]
            Draft202012Validator({"type": "object", "properties": properties, "required": required, "additionalProperties": False}).validate(payload)

    def execute(self, task, plan, request, run, *, approved=False, prior=(), journal=None):
        started = time.monotonic()
        try:
            run.check()
            self.validate(task)
            spec = self.catalog.get(task.capability)
            if spec.mutating and not approved:
                raise PermissionError("Mutation requires explicit plan confirmation")
            if spec.pack == "web" and not request.allow_web:
                raise PermissionError("Web research is disabled for this request")
            if task.capability not in self.handlers:
                raise LookupError(f"Capability unavailable: {task.capability}")
            context = ExecutionContext(run.run_id, request, plan.budget, allow_mutations=approved, allow_web=request.allow_web, deadline=run.deadline, cancel_event=run.cancel_event)
            object.__setattr__(context, "v3_run", run)
            for result in prior:
                context.board.post(result)
            def invoke():
                result = self.handlers[task.capability](dict(task.arguments), context)
                if not isinstance(result, ToolEnvelope):
                    raise TypeError("Capability must return ToolEnvelope")
                result = replace(result, elapsed_ms=int((time.monotonic() - started) * 1000))
                followups = context.board.take_requests()
                if followups:
                    from v2.agent_v3.contracts import plain
                    result.metadata = {**result.metadata, "followups": [plain(item) for item in followups]}
                return result
            if spec.mutating:
                if journal is None:
                    raise PermissionError("Mutations require a durable execution journal")
                return journal.mutate(f"{run.run_id}:{task.id}", task, invoke)
            return bounded_call(lambda: journal.read_completed(f"{run.run_id}:{task.id}",task,invoke),run) if journal else bounded_call(invoke, run)
        except RunStopped as exc:
            return ToolEnvelope(task.capability, ResultStatus.SKIPPED, errors=[str(exc)])
        except Exception as exc:
            return ToolEnvelope(task.capability, ResultStatus.FAILED, errors=[f"{type(exc).__name__}: {str(exc)[:250]}"], elapsed_ms=int((time.monotonic() - started) * 1000))


def bounded_call(fn, run):
    """Stop waiting on slow read-only I/O; running Python threads are cooperative."""
    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(copy_context().run, fn)
    try:
        while True:
            run.check()
            try:
                return future.result(timeout=min(0.1, run.remaining()))
            except TimeoutError:
                if future.done():
                    return future.result()  # distinguish tool TimeoutError from polling timeout
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
