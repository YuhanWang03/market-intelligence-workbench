"""The claim judge: one model call that says whether an answer asserts what a rule forbids.

A wording rule used to be a regular expression over the answer ("申报内容未读取"),
and every paraphrase the model found slipped past it.  The judge takes the
rule as a sentence in plain language ("申报的正文没有被读取") and the text it
applies to, and answers whether the text asserts it, quoting the sentence
that does.  The decision of *when* a rule applies stays with the rules
(the attributor's reader did read the filings); only the language
judgement moves to the model.  All of an answer's claims go in one call.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

_SYSTEM = """你是投研回答的表述审查员，不回答用户问题。不要输出任何文字或分析过程，直接调用工具给出结果。
给你若干条目，每条有一段文本（text）和一个不允许出现的断言（claim）。text 可能分成"证据："和"回答："两部分：断言说的是回答对这条证据的表述。
逐条判断：回答是否做出了这个断言（意思相同即算，措辞不必相同；带"可能、尚未确认、不能据此"这类限定的不算断言；回答对别的证据的表述不算）。
做出了就 asserted=true 并原样引用做出断言的那句话（quote，30 字内可截断），没有就 asserted=false。
通过 verdicts 工具返回结果；如果无法调用工具，就只输出 {"verdicts":[{"id":"...","asserted":true,"quote":"..."}]}"""

JUDGE_TOOL = {"type": "function", "function": {"name": "verdicts", "description": "逐条给出审查结论。", "parameters": {"type": "object", "properties": {"verdicts": {"type": "array", "items": {"type": "object", "properties": {"id": {"type": "string"}, "asserted": {"type": "boolean"}, "quote": {"type": "string"}}, "required": ["id", "asserted"]}}}, "required": ["verdicts"]}}}


class ClaimJudge:
    """Callable: ``judge(items) -> {id: quote}`` for the items whose text asserts their claim."""

    #: Ledger source for the judge's calls.
    usage_source_name = "agent_v2.judge"

    def __init__(self, llm: Any, *, max_items: int = 24, max_text_chars: int = 1200, memo_size: int = 64) -> None:
        self.llm = llm
        self.max_items = max_items
        self.max_text_chars = max_text_chars
        #: Verdicts by content: the synthesizer judges a draft and the orchestrator judges the same final text again.
        self._memo: dict[str, dict[str, str]] = {}
        self.memo_size = memo_size

    def __call__(self, items: list[dict[str, str]]) -> dict[str, str]:
        rows = [{"id": str(item["id"]), "text": str(item["text"])[: self.max_text_chars], "claim": str(item["claim"])} for item in items[: self.max_items] if item.get("text") and item.get("claim")]
        if self.llm is None or not rows:
            return {}
        key = hashlib.sha1(json.dumps(rows, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
        if key in self._memo:
            return dict(self._memo[key])
        from v2.agent_v2.agents.base import structured_call
        from v2.usage_context import usage_source

        try:
            with usage_source(self.usage_source_name):
                verdicts = structured_call(self.llm, _SYSTEM, {"items": rows}, JUDGE_TOOL).get("verdicts") or []
        except Exception as exc:  # noqa: BLE001 — an unavailable judge means the rule is not applied, never a failed run
            logger.warning("claim judge failed: %s: %s", type(exc).__name__, exc)
            return {}
        known = {row["id"] for row in rows}
        asserted: dict[str, str] = {}
        for verdict in verdicts:
            if not isinstance(verdict, dict):
                continue
            row_id = str(verdict.get("id") or "")
            if row_id in known and verdict.get("asserted"):
                asserted[row_id] = " ".join(str(verdict.get("quote") or "").split())[:60]
        if len(self._memo) >= self.memo_size:
            self._memo.pop(next(iter(self._memo)))
        self._memo[hashlib.sha1(json.dumps(rows, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()] = dict(asserted)
        return asserted
