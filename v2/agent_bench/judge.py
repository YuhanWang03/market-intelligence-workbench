"""Grading: the rubric judge (per answer) and the pairwise judge (blind, both answers).

The rubric judge is V2's ``QualityJudge`` unchanged — it only ever sees the
question, the answer, the criteria and the forbidden assertions, so it is
already agent-agnostic.  The deterministic checks around it are written
here against the serialised result both agents produce (``to_dict()``),
with version-specific expectations applied only to that version.

The pairwise judge sees two answers labelled A and B with the version
mapping withheld and citation ids scrubbed; each pair is judged twice with
the sides swapped so position bias is measured, not assumed.
"""
from __future__ import annotations

import os
import random
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Callable

from v2.agent_bench.cases import BenchCase

_CITATION = re.compile(r"\[([A-Za-z0-9_.:\-]+)\]")

#: ``must_cite`` names a source family; each agent labels the same kind of
#: evidence differently (V2 ``web_news`` vs V3 the page URL, V2's legacy 13F
#: card vs V3 ``sec_13f_hr``), so the family is what a case can require.
SOURCE_FAMILIES: dict[str, tuple[str, ...]] = {
    "market": ("market_data",),
    "web": ("web_news", "web:", "web", "http"),
    "filings": ("sec_edgar", "sec_filings", "sec_filing", "sec_", "www.sec.gov", "http", "filings."),
    "financial": ("fd_", "sec-fin", "research_engine", "conditional_financial"),
    "13f": ("sec_13f", "institutional.manager_portfolio"),
    "ark": ("ark_daily", "etf.ark_activity"),
    "earnings": ("yf_calendar", "yfinance_earnings", "account.earnings_schedule"),
    "account": ("account.",),
}


def source_prefixes(required: tuple[str, ...]) -> tuple[str, ...]:
    """Expand family names to the prefixes both agents use; a literal prefix passes through."""
    out: list[str] = []
    for item in required:
        out.extend(SOURCE_FAMILIES.get(item, (item,)))
    return tuple(out)

Verdict = dict[str, Any]
RubricJudgeFn = Callable[[str, str, list[str], list[str]], Verdict]
PairJudgeFn = Callable[[str, list[str], str, str], Verdict]


def judge_llm() -> tuple[Any, dict[str, str]]:
    """The judge model: ``AGENT_BENCH_JUDGE_*`` when set (ideally a different family), else the agents' model."""
    from v2.agent_common.llm import OpenAICompatLLM

    model = os.environ.get("AGENT_BENCH_JUDGE_MODEL")
    base = os.environ.get("AGENT_BENCH_JUDGE_BASE_URL")
    key = os.environ.get("AGENT_BENCH_JUDGE_API_KEY")
    if model and base and key:
        return OpenAICompatLLM(model=model, base_url=base, api_key=key, thinking=os.environ.get("AGENT_BENCH_JUDGE_THINKING") or None), {"judge_model": model, "judge_base_url": base, "judge_same_as_agent": "false"}
    llm = OpenAICompatLLM()
    return llm, {"judge_model": llm.model, "judge_base_url": llm.base_url, "judge_same_as_agent": "true"}


def rubric_judge(llm: Any) -> RubricJudgeFn:
    from v2.agent_v2.eval.quality import QualityJudge

    return QualityJudge(llm)


# --- rubric grading -------------------------------------------------------------------

@dataclass
class Score:
    passed: bool
    judged: bool
    criteria_met: int
    criteria_total: int
    forbidden_hit: int
    sources_ok: bool
    status_ok: bool
    writes_ok: bool
    route_ok: bool
    length_ok: bool
    fixture_missing: int = 0
    criteria: list[dict[str, Any]] = field(default_factory=list)
    forbidden: list[dict[str, Any]] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def visible_length(answer: str) -> int:
    """Length as the reader sees it: every citation marker renders as a short footnote number."""
    return len(_CITATION.sub("[0]", answer or ""))


def cited_source_ids(answer: str, evidence: list[dict[str, Any]]) -> set[str]:
    by_id = {str(item.get("id")): str(item.get("source_id") or "") for item in evidence}
    return {by_id[ref] for ref in _CITATION.findall(answer or "") if ref in by_id}


def verdict_is_valid(verdict: Verdict | None) -> bool:
    """Both lists present and every row an object with an integer index; anything else is retried, then recorded as unjudged."""
    if not isinstance(verdict, dict):
        return False
    for key in ("criteria", "forbidden"):
        rows = verdict.get(key)
        if not isinstance(rows, list) or not all(isinstance(row, dict) and str(row.get("index", "")).lstrip("-").isdigit() for row in rows):
            return False
    return True


def grade(case: BenchCase, version: str, result: dict[str, Any], verdict: Verdict | None, *, fixture_missing: int = 0) -> Score:
    answer = str(result.get("answer") or "")
    status = str(result.get("status") or "")
    problems: list[str] = []
    # -- deterministic, version-neutral
    sources_ok = True
    if case.must_cite:
        cited = cited_source_ids(answer, result.get("evidence") or [])
        prefixes = source_prefixes(case.must_cite)
        sources_ok = any(source.startswith(prefix) for source in cited for prefix in prefixes)
        if not sources_ok:
            problems.append(f"未引用要求的来源 {case.must_cite}（引用到的：{sorted(s[:40] for s in cited) or '无'}）")
    status_ok = (status in case.expect_status) if case.expect_status else (status not in {"failed", "cancelled"} and not result.get("error"))
    if not status_ok:
        problems.append(f"状态 {status or '?'}" + (f"，期望 {case.expect_status}" if case.expect_status else "") + (f"：{str(result.get('error'))[:80]}" if result.get("error") else ""))
    completed_writes = [row.get("capability") for row in result.get("results") or [] if row.get("capability") in case.forbid_capabilities and row.get("status") in {"completed", "cached"}]
    writes_ok = not completed_writes
    if not writes_ok:
        problems.append(f"不应完成的能力被执行：{completed_writes}")
    shown = visible_length(answer)
    length_ok = not case.max_chars or shown <= case.max_chars
    if not length_ok:
        problems.append(f"答案可见长度 {shown} 字，超过上限 {case.max_chars}")
    # -- deterministic, version-scoped
    route_ok = True
    expected = case.expectations.get(version) or {}
    if expected.get("route"):
        route_ok = str((result.get("route") or {}).get("kind")) == expected["route"]
        if not route_ok:
            problems.append(f"路由 {(result.get('route') or {}).get('kind')}，期望 {expected['route']}")
    if expected.get("agents"):
        ran = {row.get("name") for row in result.get("sub_agents") or []}
        missing = [name for name in expected["agents"] if name not in ran]
        if missing:
            route_ok = False
            problems.append(f"未运行子智能体 {missing}")
    # -- judge
    judged = verdict_is_valid(verdict)
    criteria_rows: list[dict[str, Any]] = []
    forbidden_rows: list[dict[str, Any]] = []
    met = hit = 0
    if judged:
        by_index = {int(row.get("index", -1)): row for row in verdict.get("criteria") or []}
        for index, text in enumerate(case.criteria):
            row = by_index.get(index, {})
            ok = bool(row.get("met"))
            met += ok
            criteria_rows.append({"text": text, "met": ok, "quote": str(row.get("quote") or "")[:80]})
            if not ok:
                problems.append(f"未满足：{text}")
        by_index = {int(row.get("index", -1)): row for row in verdict.get("forbidden") or []}
        for index, text in enumerate(case.forbidden):
            row = by_index.get(index, {})
            asserted = bool(row.get("asserted"))
            hit += asserted
            forbidden_rows.append({"text": text, "asserted": asserted, "quote": str(row.get("quote") or "")[:80]})
            if asserted:
                problems.append(f"做出了禁止的断言：{text}")
    else:
        problems.append("未经裁判评分" if verdict is None else "裁判返回格式无效")
    passed = judged and met == len(case.criteria) and hit == 0 and sources_ok and status_ok and writes_ok and route_ok and length_ok
    return Score(passed, judged, met, len(case.criteria), hit, sources_ok, status_ok, writes_ok, route_ok, length_ok, fixture_missing, criteria_rows, forbidden_rows, problems)


# --- pairwise blind comparison ------------------------------------------------------------

_PAIR_SYSTEM = """你是投研回答的评审员，不回答用户问题，不要输出任何文字，直接调用 compare 工具。
给你用户的问题、一组评审标准（criteria，按序号）和两份回答 A、B。它们来自两个不同的系统，你不知道也不需要知道是哪两个。
逐条标准判断哪份回答做得更好（A、B 或 tie），再给出整体更好的一份（winner：A、B 或 tie）和一句理由。
只依据回答本身判断：数字有出处、限制说清楚、不把推断说成事实、不编造，比篇幅更重要。不要因为一份更长或格式更漂亮就偏向它。"""

PAIR_TOOL = {
    "type": "function",
    "function": {
        "name": "compare",
        "description": "逐条给出 A/B 比较结论。",
        "parameters": {
            "type": "object",
            "properties": {
                "criteria": {"type": "array", "items": {"type": "object", "properties": {"index": {"type": "integer"}, "better": {"type": "string", "enum": ["A", "B", "tie"]}}, "required": ["index", "better"]}},
                "winner": {"type": "string", "enum": ["A", "B", "tie"]},
                "reason": {"type": "string"},
            },
            "required": ["criteria", "winner", "reason"],
        },
    },
}


def scrub(answer: str) -> str:
    """Remove what would tell the judge which system wrote the answer."""
    text = _CITATION.sub("[#]", answer or "")
    text = re.sub(r"Agent\s*V[23]", "本系统", text, flags=re.I)
    return text[:6000]


class PairJudge:
    usage_source_name = "agent_bench.pair_judge"

    def __init__(self, llm: Any) -> None:
        self.llm = llm

    def __call__(self, question: str, criteria: list[str], answer_a: str, answer_b: str) -> Verdict:
        from v2.agent_v2.agents.base import structured_call
        from v2.usage_context import usage_source

        payload = {"question": question, "criteria": [{"index": i, "text": t} for i, t in enumerate(criteria)], "answer_A": answer_a[:6000], "answer_B": answer_b[:6000]}
        with usage_source(self.usage_source_name):
            return structured_call(self.llm, _PAIR_SYSTEM, payload, PAIR_TOOL)


def assign_sides(case_id: str, seed: int) -> dict[str, str]:
    """Which version is shown as A, decided by a seeded coin flip per case and stored, never shown."""
    coin = random.Random(f"{seed}:{case_id}").random() < 0.5
    return {"A": "v2" if coin else "v3", "B": "v3" if coin else "v2"}


def _winner_version(verdict: Verdict, sides: dict[str, str]) -> str:
    winner = str(verdict.get("winner") or "tie")
    return sides.get(winner, "tie")


def compare_pair(judge: PairJudgeFn, case: BenchCase, answers: dict[str, str], *, seed: int) -> dict[str, Any]:
    """Judge the pair both ways round; a verdict that flips with the order is reported as position-dependent."""
    sides = assign_sides(case.id, seed)
    blind = {version: scrub(text) for version, text in answers.items()}
    first = judge(case.question, list(case.criteria), blind[sides["A"]], blind[sides["B"]])
    swapped_sides = {"A": sides["B"], "B": sides["A"]}
    second = judge(case.question, list(case.criteria), blind[swapped_sides["A"]], blind[swapped_sides["B"]])
    winners = (_winner_version(first, sides), _winner_version(second, swapped_sides))
    if winners[0] == winners[1]:
        outcome = winners[0]
    elif "tie" in winners:
        outcome = "tie"
    else:
        outcome = "position_dependent"
    return {"case_id": case.id, "category": case.category, "set": case.set, "sides": sides, "outcome": outcome, "winners_by_order": list(winners),
            "reasons": [str(first.get("reason") or "")[:300], str(second.get("reason") or "")[:300]],
            "criteria": [{"text": text, "first": str((next((r for r in first.get("criteria") or [] if int(r.get("index", -1)) == i), {})).get("better") or "?"), "second": str((next((r for r in second.get("criteria") or [] if int(r.get("index", -1)) == i), {})).get("better") or "?")} for i, text in enumerate(case.criteria)]}
