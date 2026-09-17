"""LangGraph Send-based DAG scheduling; does not invoke the V2 executor."""

from __future__ import annotations

import operator
from dataclasses import replace
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Send

from v2.agent_v2.models import NormalizedRequest, PlanTask, ResultStatus, ToolEnvelope
from v2.agent_v3.context import RunContext
from v2.agent_v3.contracts import envelope_from, plain, plan_from


class ExecutionState(TypedDict, total=False):
    plan: dict
    request: dict
    approved: bool
    tasks: list[dict]
    done: dict[str, dict]
    expanded: dict[str, list[str]]
    ready: list[dict]
    updates: Annotated[list[dict], operator.add]
    consumed: int
    followup_keys: list[str]
    notes: list[str]


#: Free-text topics the classifier or planner may put where a capability wants an
#: enum ("行业与供应链风险" for research.stock's focus). First match wins.
FOCUS_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("供应链", "supply_chain"), ("supply", "supply_chain"), ("估值", "valuation"), ("valuation", "valuation"),
    ("财报", "earnings"), ("盈利", "earnings"), ("业绩", "earnings"), ("earnings", "earnings"),
    ("基本面", "fundamentals"), ("fundamental", "fundamentals"), ("申报", "filings"), ("filing", "filings"),
    ("催化", "catalysts"), ("catalyst", "catalysts"), ("机构", "ownership"), ("持仓", "ownership"), ("ownership", "ownership"),
    ("行情", "market"), ("走势", "market"), ("market", "market"), ("风险", "risk"), ("risk", "risk"),
    ("全面", "full"), ("full", "full"), ("概览", "overview"), ("overview", "overview"),
)


def coerce_enum_value(value, allowed, *, keywords=FOCUS_KEYWORDS):
    """The enum member a free-text value means, or None when nothing in it maps."""
    if value in allowed:
        return value
    text = str(value or "").strip().lower()
    if not text:
        return None
    for needle, target in keywords:
        if needle.lower() in text and target in allowed:
            return target
    return None


def coerce_plan_arguments(plan, registry):
    """Rewrite enum-typed arguments the model phrased in prose into schema values.

    A value with no mapping is dropped when the argument is optional and set to
    the schema default (its first member) when required, so a stray topic
    string never fails the whole run at validation.
    """
    from dataclasses import replace as _replace

    tasks = []
    notes = []
    for task in plan.tasks:
        spec = registry.catalog.get(task.capability)
        if spec is None:
            tasks.append(task)
            continue
        schema = spec.input_schema or {}
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))
        arguments = dict(task.arguments)
        for name, value in list(arguments.items()):
            allowed = (properties.get(name) or {}).get("enum")
            if not allowed or value in allowed:
                continue
            coerced = coerce_enum_value(value, allowed)
            if coerced is None and name in required:
                coerced = allowed[0]
            if coerced is None:
                del arguments[name]
                notes.append(f"{task.capability}.{name}={value!r} dropped (not in schema)")
            else:
                arguments[name] = coerced
                notes.append(f"{task.capability}.{name}={value!r} read as {coerced!r}")
        tasks.append(_replace(task, arguments=arguments) if arguments != dict(task.arguments) else task)
    if not notes:
        return plan
    return _replace(plan, tasks=tuple(tasks), assumptions=(*plan.assumptions, *notes))


def validate_plan(plan, registry, max_tasks=12):
    if len(plan.tasks) > max_tasks:
        raise ValueError("Task budget exceeded")
    ids = {task.id for task in plan.tasks}
    if len(ids) != len(plan.tasks):
        raise ValueError("Duplicate task IDs")
    dependencies = {}
    for task in plan.tasks:
        if "::item-" in task.id or task.id.startswith("followup-"):
            raise ValueError("Task ID uses a reserved scheduler namespace")
        registry.validate(task, template=True)
        deps = set(task.depends_on)
        if task.fan_out:
            source = task.fan_out.get("from")
            if source not in ids or source == task.id:
                raise ValueError("Invalid fan_out source")
            cap = task.fan_out.get("max", 12)
            if isinstance(cap, bool) or not isinstance(cap, int) or not 1 <= cap <= 12:
                raise ValueError("fan_out max must be 1..12")
            rank = task.fan_out.get("rank")
            if rank is not None and (not isinstance(rank, dict) or not rank.get("field") or not rank.get("key")):
                raise ValueError("Invalid fan_out ranking")
            deps.add(source)
        if not deps <= ids:
            raise ValueError("Unknown task dependency")
        dependencies[task.id] = deps
    visited = set()
    while len(visited) < len(ids):
        ready = {key for key, deps in dependencies.items() if key not in visited and deps <= visited}
        if not ready:
            raise ValueError("Task dependency cycle")
        visited.update(ready)
    mutations = [task for task in plan.tasks if registry.catalog.get(task.capability).mutating]
    if mutations and (len(plan.tasks) != 1 or mutations[0].fan_out):
        raise ValueError("A confirmed mutation must be a single non-fan-out task")


def build_executor(registry, journal=None, *, max_expanded=36, max_followups=4):
    def schedule(state: ExecutionState, runtime: Runtime[RunContext]):
        tasks = list(state.get("tasks", state["plan"].get("tasks", [])))
        done = dict(state.get("done", {}))
        expanded = dict(state.get("expanded", {}))
        notes = list(state.get("notes", []))
        keys = list(state.get("followup_keys", []))
        updates = state.get("updates", [])
        for row in updates[state.get("consumed", 0):]:
            done[row["id"]] = row["result"]
            for extra in row["result"].get("metadata", {}).get("followups", []):
                key = str((extra.get("capability"), sorted(extra.get("arguments", {}).items())))
                spec = registry.catalog.get(extra.get("capability"))
                if key in keys or len(keys) >= max_followups or len(tasks) >= max_expanded or spec is None or spec.mutating:
                    continue
                candidate = {**extra, "id": f"followup-{len(keys)}", "depends_on": [], "fan_out": None}
                try:
                    registry.validate(PlanTask(**candidate))
                except Exception:
                    notes.append("Invalid follow-up request refused")
                    continue
                keys.append(key)
                tasks.append(candidate)
        def complete(key):
            return all(child in done for child in expanded[key]) if key in expanded else key in done
        def ok(key):
            names = expanded.get(key, [key])
            return bool(names) and all(envelope_from(done[name]).ok for name in names)
        stopped = "cancelled" if runtime.context.cancelled else ("deadline" if runtime.context.remaining() <= 0 else "")
        ready = []
        for task in list(tasks):
            key = task["id"]
            if key in done or key in expanded:
                continue
            if stopped:
                done[key] = plain(ToolEnvelope(task["capability"], ResultStatus.SKIPPED, errors=[stopped]))
                continue
            fan = task.get("fan_out")
            deps = set(task.get("depends_on", [])) | ({fan["from"]} if fan else set())
            if not all(complete(dep) for dep in deps):
                continue
            if any(not ok(dep) for dep in deps) and task.get("required", True):
                done[key] = plain(ToolEnvelope(task["capability"], ResultStatus.SKIPPED, errors=["required dependency failed"]))
                continue
            if fan:
                source = done.get(fan["from"], {}).get("metadata", {})
                values = source.get(fan.get("field", "tickers"), [])
                if not isinstance(values, list):
                    values = []
                values = list(dict.fromkeys(value for value in values if isinstance(value, (str, int, float))))
                rank = fan.get("rank")
                if rank:
                    rows = [row for row in source.get(rank["field"], []) if isinstance(row, dict) and isinstance(row.get(rank["key"]), (int, float))]
                    rows.sort(key=lambda row: row[rank["key"]], reverse=bool(rank.get("descending")))
                    ordered = [row.get(fan["argument"], row.get("ticker")) for row in rows]
                    values = list(dict.fromkeys([value for value in ordered if value in values] + values))
                cap = min(fan.get("max", 12), max(0, max_expanded - len(tasks)))
                if len(values) > cap:
                    notes.append(f"{key}: selected {cap}/{len(values)} objects for investigation; selection does not mean successful coverage; remaining objects were not attempted")
                children = []
                for index, value in enumerate(values[:cap]):
                    child = {**task, "id": f"{key}::item-{index}", "arguments": {**task.get("arguments", {}), fan["argument"]: value}, "fan_out": None, "depends_on": []}
                    children.append(child["id"])
                    tasks.append(child)
                    ready.append(child)
                if children:
                    expanded[key] = children
                elif not values and not task.get("required", True):
                    done[key] = plain(ToolEnvelope(task["capability"], ResultStatus.COMPLETED, summary="No eligible objects; optional investigation not needed.", metadata={"no_work_required": True}))
                else:
                    done[key] = plain(ToolEnvelope(task["capability"], ResultStatus.SKIPPED, errors=["fan-out source has no usable values or budget exhausted"]))
            else:
                ready.append(task)
        return {"tasks": tasks, "done": done, "expanded": expanded, "ready": ready, "consumed": len(updates), "followup_keys": keys, "notes": notes}

    def dispatch(state):
        if state["ready"]:
            return [Send("capability", {"task": task, "plan": state["plan"], "request": state["request"], "approved": state.get("approved", False), "prior": list(state["done"].values())}) for task in state["ready"]]
        remaining = [task for task in state["tasks"] if task["id"] not in state["done"] and task["id"] not in state["expanded"]]
        return "schedule" if remaining else END

    def capability(state, runtime: Runtime[RunContext]):
        task = PlanTask(**state["task"])
        request = NormalizedRequest(**state["request"])
        runtime.context.emit(task.capability)
        result = registry.execute(task, plan_from(state["plan"]), request, runtime.context, approved=state.get("approved", False), prior=[envelope_from(row) for row in state["prior"]], journal=journal)
        return {"updates": [{"id": task.id, "result": plain(result)}]}

    graph = StateGraph(ExecutionState, context_schema=RunContext)
    graph.add_node("schedule", schedule)
    graph.add_node("capability", capability)
    graph.add_edge(START, "schedule")
    graph.add_conditional_edges("schedule", dispatch, ["capability", "schedule", END])
    graph.add_edge("capability", "schedule")
    return graph.compile()
