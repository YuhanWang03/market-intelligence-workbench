"""A stand-in model that exercises every v2_llm code path without a key.

It is **not** an evaluation of any model.  It exists so the model-in-the-loop
harness (planner JSON parsing and trimming, fan-out, synthesis prompts,
verifier feedback, the repair round, the deterministic fallback, cost and
stability accounting) can be run end to end and its numbers checked before a
real model is paid for.

Behaviour:

* Planning prompts are answered with the rule planner's own plan as JSON, so
  the LLM planner's validation path runs on realistic input.
* Synthesis prompts are answered with prose that quotes evidence claims with
  citations.  With probability ``noise`` the draft also states a figure that
  is in no evidence, which the verifier must reject; the repair round then
  drops it, except that with probability ``noise / 2`` the "model" repeats
  the mistake so the fallback path runs too.
* Knowledge prompts get a fixed disclaimer sentence.

The random stream is seeded per client, and the factory advances the seed on
every call, so repeats of a case can differ the way a real model's do.
"""

from __future__ import annotations

import json
import random
from typing import Any

from v2.agent_common.llm import LLMResponse
from v2.agent_v2.models import NormalizedRequest, RouteDecision, RouteKind
from v2.agent_v2.planning import RulePlanner
from v2.agent_v2.routing import normalize_request, route

_PLANNER_MARK = "任务规划器"
_KNOWLEDGE_MARK = "通用知识回答"
_REPAIR_MARK = "校验未通过"
_BAD_FIGURE = "另外，据估算相关指标约为 4711.9。"


class SimulatedLLM:
    def __init__(self, *, noise: float = 0.2, seed: int = 0) -> None:
        self.noise = max(0.0, min(1.0, noise))
        self.random = random.Random(seed)
        self.calls = 0
        self._stubborn = False

    # -- entry point -----------------------------------------------------------

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> LLMResponse:
        self.calls += 1
        system = str(messages[0].get("content") or "") if messages else ""
        last_user = next((str(message.get("content") or "") for message in reversed(messages) if message.get("role") == "user"), "")
        if _PLANNER_MARK in system:
            text = self._plan(last_user)
        elif _KNOWLEDGE_MARK in system:
            text = "这是通用知识回答，未使用实时数据：" + last_user[:80]
        elif last_user.startswith(_REPAIR_MARK):
            text = self._repair(messages)
        else:
            text = self._synthesize(last_user)
        return LLMResponse(text=text, prompt_tokens=len(system) // 4 + len(last_user) // 4, completion_tokens=len(text) // 2, finish_reason="stop")

    # -- planning ---------------------------------------------------------------

    @staticmethod
    def _plan(payload_text: str) -> str:
        payload = json.loads(payload_text)
        request: NormalizedRequest = normalize_request(str(payload.get("query") or ""))
        decision: RouteDecision = route(request)
        if decision.kind not in {RouteKind.RESEARCH, RouteKind.LAB, RouteKind.ASYNC}:
            decision = RouteDecision(RouteKind.RESEARCH, decision.packs, decision.reason)
        plan = RulePlanner().plan(request, decision)
        allowed = {row["name"] for row in payload.get("capabilities", [])}
        tasks = [
            {
                "id": task.id,
                "capability": task.capability,
                "arguments": dict(task.arguments),
                "depends_on": list(task.depends_on),
                "required": task.required,
                "purpose": task.purpose,
                "fan_out": dict(task.fan_out) if task.fan_out else None,
            }
            for task in plan.tasks
            if task.capability in allowed
        ]
        return json.dumps({"objective": plan.objective, "tasks": tasks, "assumptions": ["simulated planner"]}, ensure_ascii=False)

    # -- synthesis ----------------------------------------------------------------

    def _synthesize(self, payload_text: str) -> str:
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError:
            return "证据不足，无法回答。"
        evidence = [row for row in payload.get("evidence", []) if row.get("id") and row.get("claim") and row.get("metadata", {}).get("citable", True)]
        results = payload.get("results", [])
        sentences: list[str] = []
        for result in results:
            summary = str(result.get("summary") or "").strip()
            if summary and not any(char.isdigit() for char in summary):
                sentences.append(summary.rstrip("。.") + "。")
        # Quote across subjects the way a capable model would, instead of
        # exhausting the first result's evidence and ignoring the rest.
        per_entity: dict[str, int] = {}
        quoted = 0
        for row in evidence:
            entity = str(row.get("entity") or "")
            if per_entity.get(entity, 0) >= 2:
                continue
            per_entity[entity] = per_entity.get(entity, 0) + 1
            claim = str(row["claim"]).strip().rstrip("。.")
            sentences.append(f"{claim}[{row['id']}]。")
            quoted += 1
            if quoted >= 14:
                break
        if not sentences:
            sentences.append("现有证据不足以给出结论。")
        self._stubborn = False
        if evidence and self.random.random() < self.noise:
            sentences.append(_BAD_FIGURE)
            self._stubborn = self.random.random() < 0.5
        return "\n".join(sentences)

    def _repair(self, messages: list[dict[str, Any]]) -> str:
        draft = next((str(message.get("content") or "") for message in reversed(messages) if message.get("role") == "assistant"), "")
        if self._stubborn:
            return draft.replace("4711.9", "4712.3")
        return "\n".join(line for line in draft.splitlines() if _BAD_FIGURE not in line) or draft


class SimulatedLLMFactory:
    """Callable factory that hands each case a differently seeded client."""

    def __init__(self, *, noise: float = 0.2, seed: int = 0) -> None:
        self.noise = noise
        self.seed = seed
        self.created = 0

    def __call__(self) -> SimulatedLLM:
        self.created += 1
        return SimulatedLLM(noise=self.noise, seed=self.seed + self.created)
