"""The debater: an adversarial pass over a research answer, bound to the run's own evidence.

The attributor's challenger tests one driver on one day.  The debater
takes the finished answer to a research question and looks for the one to
three judgements in it that the run's own evidence undermines: a
contrary figure, an ignored limitation, a number the answer stretched.
Every objection must name an evidence id from this run; one that does not
is dropped.  The debater never rewrites the answer: its objections travel
as a sub-agent record and are shown next to the answer on every surface,
so the reader sees the case against as well as the case for.
"""

from __future__ import annotations

import json
import time
from typing import Any

from v2.agent_v2.agents.base import structured_call
from v2.agent_v2.execution import ExecutionContext
from v2.agent_v2.models import EvidenceItem, ResultStatus, ToolEnvelope

DEBATE_TOOL = {"type": "function", "function": {"name": "objections", "description": "给出反方意见。", "parameters": {"type": "object", "properties": {"stance": {"type": "string", "enum": ["回答偏多", "回答偏空", "回答中性"]}, "objections": {"type": "array", "maxItems": 3, "items": {"type": "object", "properties": {"claim": {"type": "string"}, "objection": {"type": "string"}, "evidence_id": {"type": "string"}}, "required": ["objection", "evidence_id"]}}, "note": {"type": "string"}}, "required": ["stance", "objections"]}}}

_SYSTEM = """你是投研回答的反方辩手，通过 objections 工具给出结果，不回答用户问题。不要输出任何文字或分析过程，直接调用工具给出结果。
给你用户的问题、当前回答和这次运行拿到的证据（id 和内容）。任务：找出回答里最站不住的 1 到 3 个判断，
每条给出一句具体的反对理由，并且必须指向证据列表里能支持这条反对的 id（相反的数字、被回答忽略的限制、口径不符的引用）。
不能编造证据，不能用常识反驳；证据不支持反对就不要写。
字段：stance（回答偏多|回答偏空|回答中性），objections（每条 claim=被反对的回答判断，30 字内；objection=一句反对理由；evidence_id=证据 id），note（一句总评或留空）。
如果无法调用工具，就只输出同样字段的 JSON。"""

#: Seconds the debate may take; it runs after the answer is verified, on what is left of the budget.
DEBATE_MIN_SECONDS = 20.0


class Debater:
    def __init__(self, llm: Any, *, max_objections: int = 3, max_evidence: int = 40, max_answer_chars: int = 3000) -> None:
        self.llm = llm
        self.max_objections = max(1, max_objections)
        self.max_evidence = max(5, max_evidence)
        self.max_answer_chars = max(500, max_answer_chars)

    def run(self, question: str, answer: str, evidence: list[EvidenceItem], context: ExecutionContext, *, subject: str = "") -> ToolEnvelope:
        started = time.monotonic()
        by_id = {item.id: item for item in evidence if item.metadata.get("citable", True)}
        agent: dict[str, Any] = {"name": "debater", "label": "反方", "subject": subject or (question or "")[:24], "rounds": 1, "llm_calls": 0, "elapsed_ms": 0, "seconds_allowed": round(context.remaining_seconds(), 1), "stop_reason": "finished", "calls": {"objections": 0}, "yield": {"kept": 0, "dropped": 0}, "notes": [], "stance": ""}
        if self.llm is None or not by_id or not (answer or "").strip():
            agent["stop_reason"] = "no_model" if self.llm is None else "no_evidence"
            return self._envelope(subject, agent, [], "")
        rows = [{"id": item.id, "claim": item.claim[:240]} for item in list(by_id.values())[: self.max_evidence]]
        payload = json.dumps({"question": question, "answer": (answer or "")[: self.max_answer_chars], "evidence": rows}, ensure_ascii=False)
        from v2.usage_context import usage_source

        objections: list[dict[str, Any]] = []
        dropped = 0
        stance = ""
        note = ""
        try:
            agent["llm_calls"] = 1
            with usage_source("agent_v2.debater"):
                verdict = structured_call(self.llm, _SYSTEM, payload, DEBATE_TOOL)
            stance = str(verdict.get("stance") or "")[:12]
            note = str(verdict.get("note") or "")[:120]
            for row in verdict.get("objections") or []:
                if not isinstance(row, dict):
                    dropped += 1
                    continue
                evidence_id = str(row.get("evidence_id") or "")
                objection = " ".join(str(row.get("objection") or "").split())[:160]
                claim = " ".join(str(row.get("claim") or "").split())[:80]
                if evidence_id not in by_id or not objection:
                    dropped += 1
                    continue
                objections.append({"claim": claim, "objection": objection, "evidence_id": evidence_id})
                if len(objections) >= self.max_objections:
                    break
        except Exception as exc:  # noqa: BLE001 — a failed debate changes nothing about the answer
            agent["stop_reason"] = f"error:{type(exc).__name__}"
        agent["elapsed_ms"] = int((time.monotonic() - started) * 1000)
        agent["calls"] = {"objections": len(objections)}
        agent["yield"] = {"kept": len(objections), "dropped": dropped}
        agent["stance"] = stance
        agent["notes"] = [f"{row['objection']}（引 [{row['evidence_id']}]）" for row in objections]
        if not objections and agent["stop_reason"] == "finished":
            agent["notes"] = [f"未找到证据支持的反对意见{f'：{note}' if note else ''}"]
        return self._envelope(subject, agent, objections, note, stance=stance)

    @staticmethod
    def _envelope(subject: str, agent: dict[str, Any], objections: list[dict[str, Any]], note: str, *, stance: str = "") -> ToolEnvelope:
        summary = f"反方：{len(objections)} 条有证据支持的反对意见" + (f"，立场判断“{stance}”" if stance else "")
        return ToolEnvelope(
            "debate.challenge",
            ResultStatus.COMPLETED if agent["stop_reason"] == "finished" else ResultStatus.PARTIAL_DATA,
            subject=subject,
            summary=summary,
            findings=list(objections),
            limitations=[note] if note else [],
            metrics={"objections": len(objections), "dropped": agent["yield"]["dropped"], "elapsed_ms": agent["elapsed_ms"], "stop_reason": agent["stop_reason"]},
            metadata={"agent": agent, "trace": [{"round": 1, "action": "debate", "detail": stance or agent["stop_reason"], "ms": agent["elapsed_ms"]}], "objections": list(objections), "stance": stance, "citation_kind": "display"},
        )
