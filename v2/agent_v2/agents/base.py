"""The scaffold every sub-agent shares: a bounded JSON-action loop behind a capability.

A sub-agent is a capability whose handler lets a model choose its next
tool call for a few rounds.  What makes it safe to sit beside pure
functions is what this module owns: a hard cap on rounds and seconds,
the coordinator's remaining wall clock as an outer bound, a forced
finish when the rounds run out, and a diagnostic record of what
happened.  Domain knowledge (which tools, which prompt, how to turn a
finish into evidence) lives in the subclass.

:class:`ToolLoop` is the generic form: a sub-agent declares its tools once
(name, description, JSON schema, handler) and the loop drives the model
through native function calling, one tool call per round, with ``finish``
as a tool whose arguments are the sub-agent's report.  A model that answers
with a JSON action in text instead of a tool call is still understood.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from v2.agent_v2.execution import ExecutionContext

logger = logging.getLogger(__name__)

#: Seconds kept back from the coordinator's remaining budget so the
#: envelope can still be built and ingested after the loop stops.
BUDGET_MARGIN_SECONDS = 5.0
#: Below this much remaining time the loop does not start at all.
MINIMUM_LOOP_SECONDS = 8.0


@dataclass
class LoopLimits:
    max_rounds: int = 4
    max_seconds: float = 60.0
    #: Overrides ``max_seconds`` when the coordinator has less time left.
    outer_seconds: float | None = None
    #: Returns True when the user cancelled the run; checked before every round.
    cancelled: Callable[[], bool] | None = None

    @property
    def seconds(self) -> float:
        if self.outer_seconds is None:
            return self.max_seconds
        return max(0.0, min(self.max_seconds, self.outer_seconds - BUDGET_MARGIN_SECONDS))


def describe_action(action: dict[str, Any], limit: int = 90) -> str:
    """One line for a trace: the action's arguments, without its kind."""

    rest = {key: value for key, value in action.items() if key != "action"}
    if not rest:
        return ""
    text = json.dumps(rest, ensure_ascii=False)
    return text if len(text) <= limit else text[: limit - 1] + "…"


@dataclass
class LoopOutcome:
    """What the loop did, for the envelope's metrics and the answer's limitations."""

    finished: bool = False
    final: dict[str, Any] = field(default_factory=dict)
    rounds: int = 0
    calls: int = 0
    elapsed_ms: int = 0
    #: ``finished``, ``rounds``, ``time``, ``no_model`` or ``no_budget``.
    stop_reason: str = ""
    seconds_allowed: float = 0.0
    #: One entry per model turn: what it asked for and how long the step took.
    trace: list[dict[str, Any]] = field(default_factory=list)

    @property
    def note(self) -> str:
        return {
            "rounds": "达到轮次上限",
            "time": "达到时间上限",
            "no_model": "未配置模型",
            "no_budget": "协调者剩余时间不足，未启动",
            "cancelled": "用户取消",
        }.get(self.stop_reason, "")


def limits_for(context: ExecutionContext | None, *, max_rounds: int, max_seconds: float) -> LoopLimits:
    """The loop's limits with the coordinator's remaining wall clock as an outer bound."""

    outer = None
    if context is not None:
        try:
            outer = float(context.remaining_seconds())
        except Exception:  # noqa: BLE001 — a context without a clock imposes no bound
            outer = None
    cancelled = (lambda: bool(getattr(context, "cancelled", False))) if context is not None else None
    return LoopLimits(max_rounds=max(1, max_rounds), max_seconds=max(5.0, max_seconds), outer_seconds=outer, cancelled=cancelled)


def strip_fence(text: str) -> str:
    value = (text or "").strip()
    if value.startswith("```"):
        lines = value.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines.pop()
        value = "\n".join(lines).strip()
    start, end = value.find("{"), value.rfind("}")
    return value[start : end + 1] if start >= 0 and end > start else value


class BoundedLoop:
    #: The ledger source every model call inside ``run`` is attributed to; subclasses name themselves.
    usage_source_name = "agent_v2.sub_agent"
    #: Set by the loop for the forced-finish turn; a tool loop then offers only ``finish``.
    finish_only = False

    """Drive a model through JSON actions until it finishes or a limit stops it.

    Subclasses implement :meth:`handle` — apply one non-finish action and
    return True when it did something (a bad action returns False after
    appending a correction message).  The loop itself never reads the
    domain: it only knows ``{"action": "finish", ...}``.
    """

    def __init__(self, llm: Any, limits: LoopLimits) -> None:
        self.llm = llm
        self.limits = limits

    def handle(self, action: dict[str, Any], messages: list[dict[str, str]]) -> bool:  # pragma: no cover - abstract
        raise NotImplementedError

    def run(self, system: str, task: str, *, finish_prompt: str, preamble: list[dict[str, str]] | None = None) -> LoopOutcome:
        """Drive the loop; ``preamble`` messages (tool results gathered before the first round) follow the task."""

        outcome = LoopOutcome(seconds_allowed=self.limits.seconds)
        started = time.monotonic()
        if self.llm is None:
            outcome.stop_reason = "no_model"
            return outcome
        if self.limits.seconds < MINIMUM_LOOP_SECONDS:
            outcome.stop_reason = "no_budget"
            return outcome
        from v2.usage_context import usage_source

        with usage_source(self.usage_source_name):
            return self._run(outcome, started, system, task, finish_prompt=finish_prompt, preamble=preamble)

    def _run(self, outcome: LoopOutcome, started: float, system: str, task: str, *, finish_prompt: str, preamble: list[dict[str, str]] | None) -> LoopOutcome:
        messages: list[dict[str, str]] = [{"role": "system", "content": system}, {"role": "user", "content": task}, *(preamble or [])]
        stop = "rounds"
        for _ in range(self.limits.max_rounds):
            if time.monotonic() - started > self.limits.seconds:
                stop = "time"
                break
            if self.limits.cancelled is not None and self.limits.cancelled():
                stop = "cancelled"
                break
            outcome.rounds += 1
            outcome.calls += 1
            turn_started = time.monotonic()
            action = self.step(messages)
            if action is None:
                outcome.trace.append({"round": outcome.rounds, "action": "bad_turn", "detail": "", "ms": int((time.monotonic() - turn_started) * 1000)})
                continue
            if action.get("action") == "finish":
                if not self.accept_finish(action, messages):
                    outcome.trace.append({"round": outcome.rounds, "action": "finish_refused", "detail": self.describe_finish(action), "ms": int((time.monotonic() - turn_started) * 1000)})
                    continue
                outcome.finished, outcome.final, stop = True, action, "finished"
                outcome.trace.append({"round": outcome.rounds, "action": "finish", "detail": self.describe_finish(action), "ms": int((time.monotonic() - turn_started) * 1000)})
                break
            self.handle(action, messages)
            outcome.trace.append({"round": outcome.rounds, "action": str(action.get("action") or "?"), "detail": describe_action(action), "ms": int((time.monotonic() - turn_started) * 1000)})
        if not outcome.finished and stop == "rounds" and time.monotonic() - started <= self.limits.seconds and not (self.limits.cancelled is not None and self.limits.cancelled()):
            # One last call that may only finish: what the loop gathered is
            # not thrown away because it kept exploring.
            messages.append({"role": "user", "content": finish_prompt})
            outcome.calls += 1
            turn_started = time.monotonic()
            self.finish_only = True  # a tool loop offers only finish on this turn
            action = self.step(messages)
            if action is not None and action.get("action") == "finish":
                outcome.finished, outcome.final, stop = True, action, "finished"
            else:
                last = next((str(m.get("content") or "")[:160] for m in reversed(messages) if m.get("role") == "assistant"), "")
                logger.warning("%s: forced finish did not finish (action=%s); last reply: %r", self.usage_source_name, (action or {}).get("action"), last)
            outcome.trace.append({"round": outcome.rounds + 1, "action": "forced_finish", "detail": self.describe_finish(action) if action else "", "ms": int((time.monotonic() - turn_started) * 1000)})
            if not outcome.finished and time.monotonic() - started <= self.limits.seconds:
                # The model answered in prose or called a tool it no longer has:
                # one plain-JSON turn with no tools at all, so what the loop
                # gathered (the pages read, the filings quoted) is still reported.
                outcome.calls += 1
                turn_started = time.monotonic()
                action = self.finish_as_json(messages)
                if action is not None:
                    outcome.finished, outcome.final, stop = True, action, "finished"
                outcome.trace.append({"round": outcome.rounds + 1, "action": "forced_finish_json", "detail": self.describe_finish(action) if action else "", "ms": int((time.monotonic() - turn_started) * 1000)})
        outcome.stop_reason = stop
        outcome.elapsed_ms = int((time.monotonic() - started) * 1000)
        return outcome

    def accept_finish(self, action: dict[str, Any], messages: list[dict[str, str]]) -> bool:
        """Whether a finish may stand; a subclass that wants more work first appends why and returns False.

        The forced finish after the last round is never refused.
        """

        return True

    def finish_as_json(self, messages: list[dict[str, str]]) -> dict[str, Any] | None:
        """The finish payload as one JSON object with no tools offered; None when the model still does not comply."""

        fields = "、".join(self.finish_fields()) or "finish 的字段"
        messages.append({"role": "user", "content": f"不要调用工具，也不要写任何说明文字：只输出一个 JSON 对象，字段为 {fields}。"})
        try:
            response = self.llm.complete(messages, None)
            payload = json.loads(strip_fence(getattr(response, "text", "") or ""))
        except Exception as exc:  # noqa: BLE001 — the last resort failed; the loop reports it did not finish
            logger.warning("%s: JSON finish failed (%s): %s", self.usage_source_name, type(exc).__name__, str(exc)[:200])
            return None
        if not isinstance(payload, dict) or payload.get("action") not in (None, "finish"):
            return None  # still asking for a tool: the loop ends unfinished
        messages.append({"role": "assistant", "content": json.dumps(payload, ensure_ascii=False)})
        payload.pop("action", None)
        return {"action": "finish", **payload}

    def finish_fields(self) -> list[str]:
        """The names of the fields a finish carries; a tool loop reads them off its finish schema."""

        return []

    def describe_finish(self, action: dict[str, Any]) -> str:
        """What the finish carried, for the trace; subclasses know their own payload."""

        for key in ("reasons", "events"):
            if isinstance(action.get(key), list):
                return f"{key}={len(action[key])}"
        return ""

    def step(self, messages: list[dict[str, str]]) -> dict[str, Any] | None:
        """One model turn parsed as an action; a bad turn is answered and returns None."""

        try:
            response = self.llm.complete(messages, None)
            action = json.loads(strip_fence(response.text))
            if not isinstance(action, dict):
                raise ValueError("action must be an object")
        except Exception as exc:  # noqa: BLE001 — a bad turn is data for the envelope
            messages.append({"role": "user", "content": f"上一轮输出无法解析（{type(exc).__name__}），请只输出 JSON。"})
            return None
        messages.append({"role": "assistant", "content": json.dumps(action, ensure_ascii=False)})
        return action


# -- generic tool loop ------------------------------------------------------------


@dataclass
class Tool:
    """One tool a sub-agent may call: declared once, offered to the model as a function."""

    name: str
    description: str
    parameters: dict[str, Any]
    #: Receives the call's arguments; returns the observation text for the model.
    handler: Callable[[dict[str, Any]], str]

    def spec(self) -> dict[str, Any]:
        return {"type": "function", "function": {"name": self.name, "description": self.description, "parameters": self.parameters}}


def _schema(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": list(required or []), "additionalProperties": False}


class ToolLoop(BoundedLoop):
    """A bounded loop whose actions are declared tools, driven by native function calling.

    Subclasses give ``tools`` (the non-finish tools) and ``finish_parameters``
    (the schema of the report the finish tool carries), and may override
    :meth:`refuse_finish` to send the model back for more work once.  The
    loop turns each model turn into one action: the first tool call, or a
    JSON ``{"action": ...}`` object in text for models that do not call
    tools.  Every tool call the model makes gets a tool message back, so the
    transcript stays valid for the provider.
    """

    finish_description = "报告结果并结束。"

    def __init__(self, llm: Any, limits: LoopLimits, *, tools: list[Tool], finish_parameters: dict[str, Any], unavailable: dict[str, str] | None = None) -> None:
        super().__init__(llm, limits)
        self.tools = {tool.name: tool for tool in tools}
        self.finish_parameters = finish_parameters
        #: Tools this run does not offer (no consent, no dependency) and what the model hears if it asks for one anyway.
        self.unavailable = dict(unavailable or {})
        self._pending: list[Any] = []

    # -- what the model sees --------------------------------------------------

    def tool_specs(self) -> list[dict[str, Any]]:
        finish = {"type": "function", "function": {"name": "finish", "description": self.finish_description, "parameters": self.finish_parameters}}
        if self.finish_only:
            return [finish]
        return [tool.spec() for tool in self.tools.values()] + [finish]

    def finish_fields(self) -> list[str]:
        return list((self.finish_parameters or {}).get("properties") or {})

    def tool_lines(self) -> str:
        """The tools in one line each, for a system prompt that names them."""

        return "\n".join(f"- {name}：{tool.description}" for name, tool in self.tools.items()) + "\n- finish：" + self.finish_description

    # -- one model turn ---------------------------------------------------------

    def step(self, messages: list[dict[str, Any]]) -> dict[str, Any] | None:
        self._pending = []
        try:
            if self.finish_only:
                response = self._forced_finish_call(messages)
            else:
                response = self.llm.complete(messages, self.tool_specs())
        except Exception as exc:  # noqa: BLE001 — a failed call is a bad turn, the loop goes on
            logger.warning("%s: model call failed (%s): %s", self.usage_source_name, type(exc).__name__, str(exc)[:200])
            messages.append({"role": "user", "content": f"上一轮模型调用失败（{type(exc).__name__}），请重试：调用一个工具或 finish。"})
            return None
        calls = list(getattr(response, "tool_calls", None) or [])
        if calls:
            messages.append({"role": "assistant", "content": response.text or "", "tool_calls": [{"id": call.id, "type": "function", "function": {"name": call.name, "arguments": call.raw_arguments or json.dumps(call.arguments, ensure_ascii=False)}} for call in calls]})
            first, rest = calls[0], calls[1:]
            for extra in rest:
                # Every call needs a result or the provider rejects the next turn; only the first is executed.
                messages.append({"role": "tool", "tool_call_id": extra.id, "content": "每轮只执行一个工具调用，这一个未执行；需要的话下一轮再调用。"})
            if first.parse_error:
                messages.append({"role": "tool", "tool_call_id": first.id, "content": f"参数不是合法 JSON（{first.parse_error}），请重新调用。"})
                return None
            self._pending = [first]
            return {"action": first.name, **dict(first.arguments or {})}
        # No tool call: a JSON action in text is accepted, anything else is a bad turn.
        try:
            action = json.loads(strip_fence(response.text))
            if isinstance(action, dict) and not action.get("action") and self.finish_only:
                # The forced-finish turn asked for finish; a bare payload of its fields is that call.
                action = {"action": "finish", **action}
            if not isinstance(action, dict) or not action.get("action"):
                raise ValueError("action must be an object with an action name")
        except Exception:  # noqa: BLE001
            messages.append({"role": "assistant", "content": response.text or ""})
            messages.append({"role": "user", "content": "请调用一个工具，或调用 finish 结束；不要用普通文字回答。"})
            return None
        messages.append({"role": "assistant", "content": json.dumps(action, ensure_ascii=False)})
        return action

    def _forced_finish_call(self, messages: list[dict[str, Any]]) -> Any:
        """The forced-finish turn: the provider is told to call finish, not merely offered it.

        A provider that rejects the named ``tool_choice`` (an error, or an
        empty reply with no call) gets the same turn once more with the tools
        merely offered; the finish prompt in the messages still asks for it.
        """

        forced = {"type": "function", "function": {"name": "finish"}}
        if getattr(self.llm, "tool_choice_unsupported", False):
            return self.llm.complete(messages, self.tool_specs())
        try:
            response = self.llm.complete(messages, self.tool_specs(), tool_choice=forced)
        except TypeError:  # a client without the parameter
            return self.llm.complete(messages, self.tool_specs())
        except Exception as exc:  # noqa: BLE001 — the named choice itself may be what the provider refuses
            logger.warning("%s: forced finish with tool_choice failed (%s: %s); retrying without it", self.usage_source_name, type(exc).__name__, str(exc)[:200])
            if "tool_choice" in str(exc):
                # DeepSeek's thinking mode rejects a named choice outright; not worth a failed call per loop.
                try:
                    self.llm.tool_choice_unsupported = True
                except Exception:  # noqa: BLE001 — a client that cannot carry the flag
                    pass
            return self.llm.complete(messages, self.tool_specs())
        if not (getattr(response, "tool_calls", None) or (getattr(response, "text", "") or "").strip()):
            logger.warning("%s: forced finish with tool_choice returned nothing; retrying without it", self.usage_source_name)
            return self.llm.complete(messages, self.tool_specs())
        return response

    def _observe(self, messages: list[dict[str, Any]], content: str) -> None:
        """Return an observation to the model: as the pending call's tool result, or as a user message for a text action."""

        if self._pending:
            messages.append({"role": "tool", "tool_call_id": self._pending[0].id, "content": content})
            self._pending = []
        else:
            messages.append({"role": "user", "content": content})

    def handle(self, action: dict[str, Any], messages: list[dict[str, Any]]) -> bool:
        name = str(action.get("action") or "")
        tool = self.tools.get(name)
        if tool is None:
            self._observe(messages, self.unavailable.get(name) or f"没有叫 {name!r} 的工具；可用：{'、'.join(self.tools)}、finish。")
            return False
        arguments = {key: value for key, value in action.items() if key != "action"}
        try:
            observation = tool.handler(arguments)
        except ValueError as exc:  # a handler's own complaint about the arguments, in its words
            self._observe(messages, str(exc) or f"{name} 的参数不对。")
            return False
        except Exception as exc:  # noqa: BLE001 — a tool failing is an observation, not a crash
            self._observe(messages, f"{name} 执行失败：{type(exc).__name__}。")
            return False
        self._observe(messages, observation)
        return True

    def accept_finish(self, action: dict[str, Any], messages: list[dict[str, Any]]) -> bool:
        reason = self.refuse_finish(action)
        if reason:
            self._observe(messages, reason)
            return False
        self._pending = []
        return True

    def refuse_finish(self, action: dict[str, Any]) -> str | None:
        """Why this finish may not stand yet (the model is told and continues), or None to accept."""

        return None


def structured_call(llm: Any, system: str, payload: Any, tool: dict[str, Any]) -> dict[str, Any]:
    """One model turn that must answer through ``tool`` (a single function); its arguments are returned.

    A model that answers with JSON text instead of calling the tool is still
    understood; prose or invalid JSON raises ``ValueError``.  Used for every
    single-shot judgement (intent, judge, challenger, debater) so parsing
    failures stop being a stop reason.
    """

    content = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    response = llm.complete([{"role": "system", "content": system}, {"role": "user", "content": content}], [tool])
    calls = list(getattr(response, "tool_calls", None) or [])
    wanted = tool.get("function", {}).get("name")
    for call in calls:
        if call.name == wanted or len(calls) == 1:
            if call.parse_error:
                raise ValueError(f"tool arguments are not JSON: {call.parse_error}")
            if not isinstance(call.arguments, dict):
                raise ValueError("tool arguments must be an object")
            return dict(call.arguments)
    text = (getattr(response, "text", "") or "").strip()
    if not text:
        raise ValueError("empty model reply")
    parsed = json.loads(strip_fence(text))
    if not isinstance(parsed, dict):
        raise ValueError("reply must be a JSON object")
    return parsed
