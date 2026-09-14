"""LLM-backed planning and synthesis ports with deterministic fallbacks."""

from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import replace
from typing import Any

from v2.agent_common import presentation
from v2.agent_common.llm import LLMClient, LLMError
from v2.agent_v2.catalog import CapabilityCatalog, default_catalog
from v2.agent_v2.execution import task_limit
from v2.agent_v2.models import (
    AnswerMode,
    BudgetClass,
    EvidenceItem,
    ExecutionPlan,
    NormalizedRequest,
    PlanTask,
    RouteDecision,
    RouteKind,
    ToolEnvelope,
    VerificationReport,
)
from v2.agent_v2.planning import RulePlanner, intent_of, portfolio_ranking
from v2.agent_v2.synthesis import EvidenceSummarySynthesizer
from v2.agent_v2.verification import locate_number
from v2.usage_context import usage_source

logger = logging.getLogger(__name__)

_RESULT_CITATION = re.compile(r"\[results\.(metrics|limitations)([^\]]*)\]")
#: Results that are lists by nature (a parameter grid, dated findings with quotes): the answer may use lists and tables.
_DETAILED_CAPABILITIES = frozenset({"lab.sweep", "agent.investigate"})
_DETAILED_ANSWER = re.compile(r"详细|完整|全面|深度|报告|逐项|表格|清单|所有|展开")


def _response_intent(plan: ExecutionPlan, results: list[ToolEnvelope]) -> str:
    """A coarse label derived from the capabilities that actually ran."""

    capabilities = {task.capability for task in plan.tasks} | {result.capability for result in results}
    if "market.explain_move" in capabilities:
        return "move_explanation"
    if "market.performance" in capabilities:
        return "recent_performance"
    return "stock_research"


def _strip_fence(text: str) -> str:
    value = (text or "").strip()
    if value.startswith("```"):
        lines = value.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines.pop()
        value = "\n".join(lines).strip()
    start, end = value.find("{"), value.rfind("}")
    return value[start : end + 1] if start >= 0 and end > start else value


def _planner_budget(request: NormalizedRequest, route: RouteDecision) -> BudgetClass:
    if route.kind == RouteKind.ASYNC:
        return BudgetClass.DEEP
    if route.kind == RouteKind.LAB:
        return BudgetClass.LAB
    if intent_of(request, route).portfolio_scope:
        return BudgetClass.PORTFOLIO
    if len(request.entities) >= 2:
        return BudgetClass.COMPARISON
    return BudgetClass.STANDARD


class StructuredLLMPlanner:
    """Use an LLM for research/Lab planning, then validate before execution."""

    def __init__(
        self,
        llm: LLMClient,
        catalog: CapabilityCatalog,
        *,
        fallback: RulePlanner | None = None,
        max_tasks: int = 7,
    ) -> None:
        self.llm = llm
        self.catalog = catalog
        self.fallback = fallback or RulePlanner()
        self.max_tasks = max(1, max_tasks)

    def plan(self, request: NormalizedRequest, route: RouteDecision) -> ExecutionPlan:
        deterministic = self.fallback.plan(request, route)
        if deterministic.tasks and deterministic.tasks[0].capability in {"market.performance", "market.explain_move", "web.research"}:
            return deterministic
        # Rules also own a ranking of the user's holdings: the position card
        # answers it, and the model tends to fan out over every holding.
        capabilities = {task.capability for task in deterministic.tasks}
        intent = intent_of(request, route)
        if "account.portfolio" in capabilities and portfolio_ranking(intent) and capabilities <= {"account.portfolio", "account.performance", "market.explain_move", "market.performance"}:
            return deterministic
        # A follow-up the session framed (a loss since purchase) has a rule
        # plan built around that frame; the model would plan the bare words.
        if any(str(note).startswith("context_frame:") for note in deterministic.assumptions):
            return deterministic
        # A briefing is a fixed template (macro, earnings, risk, watchlist);
        # the model replaced it with one open-ended investigation.
        if intent.wants_any("briefing"):
            return deterministic
        # An investigation is a fixed brief for the investigator (task, tools by
        # job type); the model planner used to give it a vague task or none.
        if getattr(intent, "investigation", ""):
            return deterministic
        # Rules own market questions and non-thin lookups; the model gets
        # research and lab routes, plus lookups the rules could not resolve
        # beyond a scope read (the holdout wording the rules never saw).
        thin = all(task.capability in {"account.portfolio", "state.read"} for task in deterministic.tasks)
        thin_lookup = route.kind == RouteKind.FAST_LOOKUP and thin and not deterministic.direct_answer
        if route.kind not in {RouteKind.RESEARCH, RouteKind.LAB, RouteKind.ASYNC} and not thin_lookup:
            return deterministic
        budget = _planner_budget(request, route)
        limit = min(self.max_tasks, task_limit(budget))
        if route.kind in {RouteKind.LAB, RouteKind.ASYNC}:
            limit = min(limit, 2)
        allowed = self.catalog.specs(route.packs)
        capabilities = [
            {
                "name": spec.name,
                "description": spec.description,
                "input_schema": spec.input_schema,
                **({"long_running": True} if spec.long_running else {}),
            }
            for spec in allowed
            if not spec.mutating
        ]
        system = """你是投研与量化实验任务规划器，只输出 JSON，不回答用户问题。
把目标拆成最少数量的现有 capability。不要编造 capability，不要安排写操作。
相同信息只获取一次；多股票比较优先 research.compare；账户问题先取账户事实。
需要对持仓或关注列表里的每只股票分别执行某个 capability 时，先安排 account.portfolio 或 state.read，再安排一个带 fan_out 的任务：
{"id":"t2","capability":"research.stock","arguments":{"focus":"filings"},"depends_on":["t1"],"fan_out":{"from":"t1","field":"tickers","argument":"ticker","max":8}}。
标记 long_running 的 capability 是子智能体（读申报、异动归因、网页核查），每个要 10–60 秒和多次模型调用：一份计划最多安排 2 个，带 fan_out 的最多展开 3 次，只在问题确实问"为什么涨跌、最近发生了什么事、有什么新闻"时才用；
两只股票"最近为什么走势分化"这类问题，对每只分别安排 market.performance 和 market.explain_move，不要用 research.compare 代替归因。
agent.investigate 是没有固定角色的调查员：当问题需要的事实没有对口的 capability 时（某个事件的来龙去脉、某份申报里的具体条款、某个说法有没有出处），给它一句明确的 task、股票代码和需要的工具（search_news、read_page、list_filings、read_filing、recall_memory），它只报能引用原文的发现。
量化实验必须从用户原话提取参数，不要虚构参数；未提供的参数交给工具默认值。
输出格式：{"objective":"...","tasks":[{"id":"t1","capability":"...","arguments":{},"depends_on":[],"required":true,"purpose":"...","fan_out":null}],"assumptions":[]}。"""
        payload = {
            "query": request.text,
            "entities": list(request.entities),
            "intent": {key: value for key, value in intent.to_dict().items() if value not in (None, "", [], False)},
            "capabilities": capabilities,
            "maximum_tasks": limit,
            "maximum_long_running_tasks": MAX_LONG_RUNNING_TASKS,
            "maximum_long_running_fan_out": MAX_LONG_RUNNING_FAN_OUT,
        }
        try:
            with usage_source("agent_v2.planner"):
                response = self.llm.complete(
                    [
                        {"role": "system", "content": system},
                        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                    ],
                    None,
                )
            raw = json.loads(_strip_fence(response.text))
            tasks = self._tasks(raw.get("tasks"), {spec.name for spec in allowed})
            if not tasks:
                raise ValueError("planner returned no executable tasks")
            assumptions = [str(value) for value in raw.get("assumptions", []) if value]
            if thin_lookup and deterministic.tasks:
                # A thin rule plan is a floor, not a draft: keep its scope reads
                # and let the model add what the wording implies on top.
                tasks = _merge_tasks(deterministic.tasks, tasks)
            tasks = _inherit_fan_out_rank(tasks, deterministic.tasks)
            tasks, delegation_notes = _cap_long_running(tasks, self.catalog)
            assumptions.extend(delegation_notes)
            if any(_long_running(task, self.catalog) for task in tasks) and budget in {BudgetClass.DIRECT, BudgetClass.FOCUSED}:
                budget = BudgetClass.STANDARD
                limit = min(self.max_tasks, task_limit(budget))
            if any(task.capability == "research.compare" for task in tasks) and budget in {BudgetClass.DIRECT, BudgetClass.FOCUSED, BudgetClass.STANDARD}:
                budget = BudgetClass.COMPARISON
                limit = min(self.max_tasks, task_limit(budget))
            trimmed = _trim_to_budget(tasks, limit)
            if len(trimmed) < len(tasks):
                dropped = ", ".join(task.capability for task in tasks if task not in trimmed)
                assumptions.append(f"Planner trimmed {len(tasks) - len(trimmed)} task(s) to the {budget.value} budget of {limit}: {dropped}")
                tasks = trimmed
            grounded = route.kind == RouteKind.RESEARCH or any(task.capability.startswith(("research.", "market.")) for task in tasks)
            answer_mode = AnswerMode.RESEARCH_GROUNDED if grounded else AnswerMode.TOOL_GROUNDED
            return ExecutionPlan(
                objective=str(raw.get("objective") or request.text),
                route=route.kind,
                tasks=tasks,
                budget=budget,
                answer_mode=answer_mode,
                web_fallback_allowed=request.allow_web and route.kind == RouteKind.RESEARCH,
                assumptions=tuple(assumptions),
            )
        except (LLMError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            note = f"LLM planner fallback: {type(exc).__name__}"
            return replace(deterministic, assumptions=(*deterministic.assumptions, note))

    def _tasks(self, raw: Any, allowed: set[str]) -> tuple[PlanTask, ...]:
        # Over-long lists are trimmed to the budget after parsing; only an
        # absurd list is treated as a malformed plan.
        if not isinstance(raw, list) or len(raw) > 4 * self.max_tasks:
            raise ValueError("invalid task list")
        tasks: list[PlanTask] = []
        seen: set[str] = set()
        for index, row in enumerate(raw, 1):
            if not isinstance(row, dict):
                raise ValueError("task must be an object")
            capability = str(row.get("capability") or "")
            if capability not in allowed:
                raise ValueError(f"capability is not allowed: {capability}")
            task_id = str(row.get("id") or f"t{index}")
            if task_id in seen:
                raise ValueError("duplicate task id")
            arguments = row.get("arguments") or {}
            dependencies = row.get("depends_on") or []
            if not isinstance(arguments, dict) or not isinstance(dependencies, list):
                raise ValueError("invalid task arguments or dependencies")
            fan_out = row.get("fan_out") or None
            if fan_out is not None:
                if not isinstance(fan_out, dict) or not fan_out.get("from") or not fan_out.get("argument"):
                    raise ValueError("invalid fan_out")
                fan_out = {"from": str(fan_out["from"]), "field": str(fan_out.get("field") or "tickers"), "argument": str(fan_out["argument"]), "max": int(fan_out.get("max") or 8)}
                dependencies = list(dict.fromkeys([*dependencies, fan_out["from"]]))
            tasks.append(
                PlanTask(
                    id=task_id,
                    capability=capability,
                    arguments=arguments,
                    depends_on=tuple(str(value) for value in dependencies),
                    required=bool(row.get("required", True)),
                    purpose=str(row.get("purpose") or ""),
                    fan_out=fan_out,
                )
            )
            seen.add(task_id)
        if any(set(task.depends_on) - seen for task in tasks):
            raise ValueError("unknown task dependency")
        return tuple(tasks)


def _inherit_fan_out_rank(tasks: tuple[PlanTask, ...], deterministic: tuple[PlanTask, ...]) -> tuple[PlanTask, ...]:
    """Carry the rules' fan-out ordering onto the model's plan.

    The rules read the ranking intent of the wording ("跌得最多" orders by
    P/L); the model's fan-out over the same source inherits it so the cap
    keeps the relevant holdings.
    """

    ranks: dict[str, dict] = {}
    by_id = {task.id: task for task in deterministic}
    for task in deterministic:
        rank = (task.fan_out or {}).get("rank")
        source = by_id.get(str((task.fan_out or {}).get("from") or ""))
        if isinstance(rank, dict) and source is not None:
            ranks[source.capability] = rank
    if not ranks:
        return tasks
    ids = {task.id: task for task in tasks}
    updated: list[PlanTask] = []
    for task in tasks:
        fan_out = task.fan_out
        if fan_out and "rank" not in fan_out:
            source = ids.get(str(fan_out.get("from") or ""))
            rank = ranks.get(source.capability) if source is not None else None
            if rank is not None:
                fan_out = {**fan_out, "rank": dict(rank)}
                task = replace(task, fan_out=fan_out)
        updated.append(task)
    return tuple(updated)


def _merge_tasks(base: tuple[PlanTask, ...], extra: tuple[PlanTask, ...]) -> tuple[PlanTask, ...]:
    """Append model-proposed tasks that the rule plan does not already contain."""

    merged = list(base)
    seen = {(task.capability, json.dumps(task.arguments, sort_keys=True, default=str)) for task in base}
    ids = {task.id for task in base}
    for task in extra:
        key = (task.capability, json.dumps(task.arguments, sort_keys=True, default=str))
        if key in seen:
            continue
        seen.add(key)
        task_id = task.id
        while task_id in ids:
            task_id = f"{task_id}-llm"
        ids.add(task_id)
        depends = tuple(dep if dep in ids else dep for dep in task.depends_on)
        merged.append(replace(task, id=task_id, depends_on=depends))
    return tuple(merged)


#: Sub-agents the model planner may schedule in one plan, and how far one of them may fan out.
MAX_LONG_RUNNING_TASKS = 2
MAX_LONG_RUNNING_FAN_OUT = 3
#: Capabilities backed by a bounded model loop (several model calls, 10–120 s each).  The
#: research engine is long-running too, but it is governed by the task limit, not this cap.
SUB_AGENT_CAPABILITIES = frozenset({"filings.read_events", "market.attribute_move", "market.explain_move", "web.research", "agent.investigate"})


def _long_running(task: PlanTask, catalog: CapabilityCatalog) -> bool:
    return task.capability in SUB_AGENT_CAPABILITIES


def _cap_long_running(tasks: tuple[PlanTask, ...], catalog: CapabilityCatalog) -> tuple[tuple[PlanTask, ...], list[str]]:
    """Keep the model's delegation to sub-agents within the cost cap.

    The first ``MAX_LONG_RUNNING_TASKS`` long-running tasks stay; the rest
    are dropped (dependents of a dropped task go with it); a fan-out on a
    long-running task is clamped to ``MAX_LONG_RUNNING_FAN_OUT``.  Every
    change is written into the plan's assumptions.
    """

    notes: list[str] = []
    kept: list[PlanTask] = []
    seen_long = 0
    dropped: list[str] = []
    for task in tasks:
        if not _long_running(task, catalog):
            kept.append(task)
            continue
        seen_long += 1
        if seen_long > MAX_LONG_RUNNING_TASKS:
            dropped.append(task.capability)
            continue
        fan_out = task.fan_out
        if isinstance(fan_out, dict) and int(fan_out.get("max") or 0) > MAX_LONG_RUNNING_FAN_OUT:
            task = replace(task, fan_out={**fan_out, "max": MAX_LONG_RUNNING_FAN_OUT})
            notes.append(f"Planner clamped the fan-out of {task.capability} to {MAX_LONG_RUNNING_FAN_OUT} (sub-agent cost cap)")
        kept.append(task)
    if dropped:
        notes.append(f"Planner dropped {len(dropped)} sub-agent task(s) beyond the cap of {MAX_LONG_RUNNING_TASKS}: {', '.join(dropped)}")
    while True:
        ids = {task.id for task in kept}
        dangling = [task for task in kept if set(task.depends_on) - ids]
        if not dangling:
            break
        kept = [task for task in kept if task not in dangling]
    return tuple(kept), notes


def _trim_to_budget(tasks: tuple[PlanTask, ...], limit: int) -> tuple[PlanTask, ...]:
    """Drop optional tasks from the end first, then required ones, then dangling dependents."""

    kept = list(tasks)
    while len(kept) > limit:
        optional = [task for task in kept if not task.required]
        kept.remove(optional[-1] if optional else kept[-1])
    while True:
        ids = {task.id for task in kept}
        dangling = [task for task in kept if set(task.depends_on) - ids]
        if not dangling:
            return tuple(kept)
        kept = [task for task in kept if task not in dangling]


_RESEARCH_SYSTEM = """你是证据约束的中文投研助手，表达要像一位清楚、克制、有判断力的研究同事。
只能陈述输入证据支持的外部事实。每项关键事实后必须写对应的 [evidence_id]。
方括号内只能原样使用 evidence 数组中真实存在的 id；严禁把 results.*、字段路径、source_id 或占位符当作引用。
results 中的评分或限制如需引用，使用 evidence 中 citation_kind 为 metrics 或 limitations 的对应条目。
推断必须标成“推断”；数据缺失必须明确说明。不得把一个主体的数据归给另一个主体。
如果证据的方向与问题的前提相反（例如问为什么跌，证据显示今日上涨），先点明两者指的是不同区间或口径，再回答用户实际所指的那个区间；不要只否定前提，也不要用当日数据回答关于更长区间的问题。assumptions 中以 context_frame 开头的说明描述了用户追问所指的对象和区间，必须遵循。
不要把历史回测写成未来收益保证。不要输出未在证据中出现的数字。
不要向用户提内部工具、能力或角色名（归因者、申报阅读者、规划器、fan-out、capability 等）；直接说事实和来源类型（新闻、SEC 申报、盯盘记录、行情）。

严格遵循输入中的 response_style：
- brief：先用一句话直接回答用户问的那个量或对象（总额、盈亏、名单、日期、谁更强），这些被直接询问的数字和名字必须原样给出，不得因为篇幅省略；比较或排名问题必须点名每个候选并给出用来比较的数字；“有没有…”“哪几只…”这类筛选问题要先点名相对靠前的几只并按程度排序，再交代其余；开头这句里的每个数字也要紧跟它自己的 [evidence_id]，几只股票的数字来自不同证据时每个数字后各写一个 id，不能用一条引用盖住一串数字；只展开被点名的对象，未被问到的个股不要逐一复述；覆盖不全时用一句话说明未覆盖的对象。然后用 3—5 个短段落、约 300—500 个中文字完成回答，挑选最有决策价值的 3—5 条事实，只讲一个主要风险和最重要的数据缺口，最后指出接下来值得观察什么。不要使用标题、表格、分隔线、编号清单、“正面/负面/中性”标签、“必须说明”或单独的免责声明章节；不要重复同一事实。
- detailed：用户明确要求详细、完整、全面、表格或逐项展开时，才允许使用小标题与列表，但仍应合并重复内容并保持自然。

把 BULLISH、MEDIUM、forward_pe、revision_trend 等内部英文标签翻译或解释成自然中文；必要的通用缩写可以保留。不要逐项复述所有模块，也不要把工具输出改写成机械评分单。"""


class LLMEvidenceSynthesizer:
    """Generate prose from bounded evidence while preserving source identifiers."""

    supports_general_knowledge = True

    def __init__(
        self,
        llm: LLMClient,
        *,
        catalog: CapabilityCatalog | None = None,
        fallback: EvidenceSummarySynthesizer | None = None,
        max_context_chars: int = 28_000,
    ) -> None:
        self.llm = llm
        from v2.agent_v2.judge import ClaimJudge

        #: Decides the verifier's forbid_claim rules with one model call per draft.
        self.judge = ClaimJudge(llm) if llm is not None else None
        self.catalog = catalog or default_catalog()
        self.fallback = fallback or EvidenceSummarySynthesizer()
        self.max_context_chars = max(4_000, max_context_chars)
        # Per-thread diagnostics of the most recent call: the web backend and
        # the benchmark both run several answers at once on one synthesizer.
        self._diagnostics = threading.local()

    def _reset_diagnostics(self) -> None:
        self._diagnostics.outcome = "fallback"
        self._diagnostics.draft = ""
        self._diagnostics.attempts = []
        self._diagnostics.completions = []
        self._diagnostics.drafts = []

    @property
    def last_outcome(self) -> str:
        """How the most recent answer on this thread was produced: ``clean``, ``repaired``, ``fallback`` or ``knowledge``."""

        return getattr(self._diagnostics, "outcome", "")

    @last_outcome.setter
    def last_outcome(self, value: str) -> None:
        self._diagnostics.outcome = value

    @property
    def last_draft(self) -> str:
        """The first draft of the most recent answer on this thread."""

        return getattr(self._diagnostics, "draft", "")

    @last_draft.setter
    def last_draft(self, value: str) -> None:
        self._diagnostics.draft = value

    def diagnostics(self) -> dict[str, Any]:
        """What happened to the model's drafts on this thread's most recent call."""

        return {
            "outcome": self.last_outcome,
            "draft": self.last_draft[:4000],
            "attempts": [dict(attempt) for attempt in getattr(self._diagnostics, "attempts", [])],
            "citation_completions": list(getattr(self._diagnostics, "completions", [])),
        }

    def _keep_draft(self, text: str) -> None:
        drafts = getattr(self._diagnostics, "drafts", None)
        if drafts is None:
            drafts = self._diagnostics.drafts = []
        drafts.append(text or "")

    def _complete(self, answer: str, evidence, results) -> str:
        """Deterministic citation completion before the verifier sees a draft."""

        from v2.agent_v2.verification import complete_citations

        completed, notes = complete_citations(answer, evidence, results)
        if notes:
            existing = getattr(self._diagnostics, "completions", None)
            if existing is None:
                existing = self._diagnostics.completions = []
            existing.extend(notes)
        return completed

    def _record_attempt(self, stage: str, *, ok: bool, warnings=(), unknown_citations=(), ungrounded_numbers=()) -> None:
        attempts = getattr(self._diagnostics, "attempts", None)
        if attempts is None:
            attempts = self._diagnostics.attempts = []
        attempts.append(
            {
                "stage": stage,
                "ok": bool(ok),
                "warnings": [str(value) for value in warnings],
                "unknown_citations": [str(value) for value in unknown_citations],
                "ungrounded_numbers": [str(value) for value in ungrounded_numbers],
            }
        )

    def _record_report(self, stage: str, report) -> None:
        self._record_attempt(stage, ok=report.ok, warnings=report.warnings, unknown_citations=report.unknown_citations, ungrounded_numbers=report.ungrounded_numbers)

    def synthesize(self, request, plan, results, evidence) -> str:
        if plan.answer_mode == AnswerMode.GENERAL_KNOWLEDGE:
            system = """用中文回答稳定的金融概念和分析方法。明确这是通用知识回答，未使用实时数据。
不要声称知道当前股价、最新财报、近期新闻、用户持仓或其他可能变化的事实。"""
            payload = request.text
        else:
            system = self._research_system(plan, results, preferences=request.metadata.get("preferences"))
            payload = self._payload(request.text, plan, results, evidence, preferences=request.metadata.get("preferences"), conversation=request.metadata)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": payload},
        ]
        self._reset_diagnostics()
        try:
            answer = self._draft(messages, results, evidence)
            self.last_draft = answer
            if plan.answer_mode == AnswerMode.GENERAL_KNOWLEDGE:
                self.last_outcome = "knowledge"
                return answer
            from v2.agent_v2.verification import verify_answer

            answer = self._complete(answer, evidence, results)
            self._keep_draft(answer)
            report = verify_answer(answer, evidence, answer_mode=plan.answer_mode, results=results, judge=self.judge)
            self._record_report("draft", report)
            if report.ok:
                self.last_outcome = "clean"
                return self._shorten(answer, messages, plan, results, evidence, request.metadata.get("preferences"))
            # One repair round: the verifier names what failed and what must
            # stay; a second failure falls back to deterministic prose rather
            # than shipping flagged numbers to the user.
            repair = self._draft(
                [
                    *messages,
                    {"role": "assistant", "content": answer},
                    {"role": "user", "content": repair_instruction(report, evidence)},
                ],
                results,
                evidence,
            )
            repair = self._complete(repair, evidence, results)
            self._keep_draft(repair)
            repair_report = verify_answer(repair, evidence, answer_mode=plan.answer_mode, results=results, judge=self.judge)
            self._record_report("repair", repair_report)
            if repair_report.ok:
                self.last_outcome = "repaired"
                return repair
            if _problem_count(repair_report) < _problem_count(report) or (_problem_count(repair_report) == _problem_count(report) and _problem_set(repair_report) != _problem_set(report)):
                # The repair fixed what was named and tripped something else
                # (a trimmed candidate list dropped a required citation): one
                # more round, since the deterministic fallback would drop the
                # ranking or the reasoning the user asked for.
                second = self._draft(
                    [
                        *messages,
                        {"role": "assistant", "content": repair},
                        {"role": "user", "content": repair_instruction(repair_report, evidence)},
                    ],
                    results,
                    evidence,
                )
                second = self._complete(second, evidence, results)
                self._keep_draft(second)
                second_report = verify_answer(second, evidence, answer_mode=plan.answer_mode, results=results, judge=self.judge)
                self._record_report("repair2", second_report)
                if second_report.ok:
                    self.last_outcome = "repaired"
                    return second
        except (LLMError, ValueError, TypeError) as exc:
            self._record_attempt("error", ok=False, warnings=(f"{type(exc).__name__}: {str(exc)[:200]}",))
        self._log_fallback(request)
        return self.fallback.synthesize(request, plan, results, evidence)

    def _shorten(self, answer: str, messages: list[dict[str, str]], plan: ExecutionPlan, results: list[ToolEnvelope], evidence: list[EvidenceItem], preferences: list[str] | None) -> str:
        """Enforce a "回答短一点" preference: one compression round when the verified answer is over the limit; the long answer stays if the short one fails verification."""

        limit = short_answer_limit(preferences or [])
        if not limit or visible_length(answer) <= limit:
            return answer
        from v2.agent_v2.verification import verify_answer

        try:
            short = self._draft(
                [
                    *messages,
                    {"role": "assistant", "content": answer},
                    {"role": "user", "content": f"用户要求回答短一点。把上面的回答压缩到 {limit} 个字符以内、最多两段：只保留两三条最关键的事实、一个风险和一个观察点，每句关键事实后仍然写原来的 [evidence_id]，不要新增证据里没有的数字。直接输出压缩后的完整回答。"},
                ],
                results,
                evidence,
            )
            short = self._complete(short, evidence, results)
            report = verify_answer(short, evidence, answer_mode=plan.answer_mode, results=results, judge=self.judge)
            self._record_report("shorten", report)
            if report.ok and visible_length(short) <= limit * 1.2:
                return short
        except (LLMError, ValueError, TypeError) as exc:
            self._record_attempt("shorten_error", ok=False, warnings=(f"{type(exc).__name__}: {str(exc)[:200]}",))
        return answer

    def _research_system(self, plan: ExecutionPlan, results: list[ToolEnvelope], preferences: list[str] | None = None) -> str:
        system = _RESEARCH_SYSTEM
        guidance = self._guidance(plan, results)
        if guidance:
            system += "\n\n再严格遵循以下与本次所用能力对应的 response_intent 规则：\n" + guidance
        system += "\n\n每个引用只支持它紧邻的那句话。不要用一条聚合引用同时支撑价格、成交量、新闻和期权等不同事实。"
        system += "\n输入里的 user_preferences 是用户之前说过的偏好（口径、篇幅、关注点），按它组织回答，但不能因此违反证据和引用规则。"
        if preferences and short_answer_limit(preferences):
            system += f"\n用户要求回答短一点：全文不超过 {short_answer_limit(preferences)} 个字符、最多两段，只保留两三条最有决策价值的事实、一个风险和一个观察点，保留引用，不要展开成多段长文。"
        system += "\n输入里的 recent_turns 是同一会话之前的问答；previous_answer 是上一条回答的全文。用户追问上一条回答（展开某一点、问为什么、换口径）时，围绕被追问的那一点回答：先复述那一点是什么，再用证据展开或解释，不要把整条回答重说一遍；上一条里没有证据支撑的部分要明说。"
        return system

    def revise(self, request, plan, results, evidence, answer: str, objections: list[dict[str, Any]]) -> str | None:
        """One bounded rewrite that takes the debater's objections into account; None when the rewrite fails verification.

        The objections cite this run's evidence, so the rewrite can only
        move towards it: a competing candidate the answer left out, a
        limitation it skipped.  The revised draft goes through citation
        completion and the same verifier (judge included); a rewrite that
        does not pass leaves the original answer standing.
        """

        if self.llm is None or not objections or not (answer or "").strip():
            return None
        from v2.agent_v2.verification import verify_answer

        listed = "\n".join(f"- 针对“{row.get('claim') or ''}”：{row.get('objection') or ''}（证据 [{row.get('evidence_id') or ''}]）" for row in objections[:3])
        instruction = (
            "反方审阅了上面的回答，提出以下有证据支持的反对意见：\n" + listed +
            "\n请只在证据确实支持反对意见时修订回答：补上被忽略的候选解释或限制、纠正被拉高的判断，其余保持原样。"
            "保持同样的篇幅和引用格式，每句关键事实后仍然写 [evidence_id]，不要新增证据里没有的数字，不要提到反方或审阅。直接输出修订后的完整回答。"
        )
        messages = [
            {"role": "system", "content": self._research_system(plan, results)},
            {"role": "user", "content": self._payload(request.text, plan, results, evidence, preferences=request.metadata.get("preferences"))},
            {"role": "assistant", "content": answer},
            {"role": "user", "content": instruction},
        ]
        try:
            revised = self._complete(self._draft(messages, results, evidence), evidence, results)
            report = verify_answer(revised, evidence, answer_mode=plan.answer_mode, results=results, judge=self.judge)
        except (LLMError, ValueError, TypeError) as exc:
            self._record_attempt("revision_error", ok=False, warnings=(f"{type(exc).__name__}: {str(exc)[:200]}",))
            return None
        self._record_report("revision", report)
        return revised if report.ok else None

    def _log_fallback(self, request) -> None:
        """Every rejected draft, with what the verifier said, so a fallback can be diagnosed from the server log."""

        attempts = getattr(self._diagnostics, "attempts", []) or []
        drafts = getattr(self._diagnostics, "drafts", []) or []
        for index, attempt in enumerate(attempts):
            text = drafts[index] if index < len(drafts) else ""
            logger.warning(
                "agent_v2 synthesis fell back (%s) stage=%s warnings=%s unknown=%s ungrounded=%s draft=%r",
                (request.text or "")[:80],
                attempt.get("stage"),
                attempt.get("warnings"),
                attempt.get("unknown_citations"),
                attempt.get("ungrounded_numbers"),
                text[:3000],
            )

    def _guidance(self, plan: ExecutionPlan, results: list[ToolEnvelope]) -> str:
        """Collect the adapters' own answer rules for the capabilities in play."""

        names: list[str] = []
        for name in (*(task.capability for task in plan.tasks), *(result.capability for result in results)):
            if name not in names:
                names.append(name)
        lines: list[str] = []
        for name in names:
            spec = self.catalog.get(name)
            if spec is not None and spec.answer_guidance and spec.answer_guidance not in lines:
                lines.append(spec.answer_guidance)
        return "\n".join(f"- {line}" for line in lines)

    def _draft(self, messages: list[dict[str, str]], results: list[ToolEnvelope], evidence: list[EvidenceItem]) -> str:
        answer = ""
        for attempt in (1, 2):
            with usage_source("agent_v2.synthesizer"):
                response = self.llm.complete(messages, None)
            answer = presentation.strip_deliberation(response.text)
            if answer:
                break
            # A provider now and then returns nothing (a reply that was all
            # deliberation, a cut-off stream): one more call before the
            # deterministic fallback, which has no quotes and no judgement.
            logger.warning("agent_v2 synthesizer returned an empty answer (attempt %d)", attempt)
        if not answer:
            raise ValueError("synthesizer returned an empty answer")
        return _normalize_result_citations(answer, results, evidence)

    def _payload(self, query: str, plan: ExecutionPlan, results: list[ToolEnvelope], evidence, preferences: list[str] | None = None, conversation: dict | None = None) -> str:
        data = {
            "query": query[:2000],
            **({"user_preferences": [str(value)[:120] for value in preferences][:8]} if preferences else {}),
            **({"previous_question": str(conversation.get("previous_question") or "")[:300], "previous_answer": str(conversation.get("previous_answer") or "")[:4000]} if conversation and conversation.get("previous_answer") else {}),
            **({"recent_turns": [{"question": str(turn.get("question") or "")[:200], "answer_digest": str(turn.get("answer_digest") or "")[:240]} for turn in list(conversation.get("recent_turns") or [])[-2:]]} if conversation and conversation.get("recent_turns") else {}),
            "objective": plan.objective[:2000],
            "response_style": "detailed" if _DETAILED_ANSWER.search(query) or any(result.capability in _DETAILED_CAPABILITIES for result in results) else "brief",
            "response_intent": _response_intent(plan, results),
            "assumptions": list(plan.assumptions),
            "results": [_result_row(result) for result in results],
            "evidence": [_evidence_row(item) for item in _select_evidence(results, evidence, 40)],
        }
        encoded = json.dumps(data, ensure_ascii=False, default=str)
        if len(encoded) <= self.max_context_chars:
            return encoded
        # Preserve the question, summaries, and evidence heads rather than
        # allowing transport-level truncation to cut an arbitrary JSON token.
        data["evidence"] = data["evidence"][:16]
        for result in data["results"]:
            if result.get("findings"):
                result["findings"] = result["findings"][:3]
            result["summary"] = result["summary"][:1000]
        encoded = json.dumps(data, ensure_ascii=False, default=str)
        if len(encoded) <= self.max_context_chars:
            return encoded
        data["results"] = [
            {
                "capability": row["capability"],
                "status": row["status"],
                "subject": row["subject"],
                "summary": row["summary"][:400],
                **({"limitations": row["limitations"][:2]} if row.get("limitations") else {}),
            }
            for row in data["results"]
        ]
        data["evidence"] = [
            {key: (str(value)[:500] if key == "claim" else value) for key, value in row.items() if key in ("id", "entity", "claim", "value", "as_of", "source_id", "citation_kind")}
            for row in data["evidence"][:12]
        ]
        encoded = json.dumps(data, ensure_ascii=False, default=str)
        while len(encoded) > self.max_context_chars and data["evidence"]:
            data["evidence"].pop()
            encoded = json.dumps(data, ensure_ascii=False, default=str)
        if len(encoded) > self.max_context_chars:
            data = {
                "query": query[:800],
                "objective": plan.objective[:400],
                "results": [
                    {
                        "capability": row["capability"],
                        "status": row["status"],
                        "subject": row["subject"],
                        "summary": row["summary"][:120],
                    }
                    for row in data["results"]
                ],
                "evidence": [],
                "truncated": True,
            }
            encoded = json.dumps(data, ensure_ascii=False, default=str)
        return encoded


#: Evidence fields the synthesizer sees.  Titles, URLs, run ids and the rest
#: of the metadata are for the surfaces and the verifier, not for the draft:
#: the model cites by id, and every field sent is an uncached input token
#: on every call.  ``citation_kind`` stays because the system prompt names it.
_EVIDENCE_FIELDS = ("id", "entity", "claim", "metric", "value", "unit", "period", "as_of", "source_id", "confidence")


def _evidence_row(item: EvidenceItem) -> dict[str, Any]:
    row = {key: value for key in _EVIDENCE_FIELDS if (value := getattr(item, key)) not in (None, "")}
    kind = item.metadata.get("citation_kind")
    if kind:
        row["citation_kind"] = kind
    return row


def _result_row(result: ToolEnvelope) -> dict[str, Any]:
    """One result for the payload; empty metrics, findings, limitations and errors are left out."""

    row: dict[str, Any] = {"capability": result.capability, "status": result.status.value, "subject": result.subject, "as_of": result.as_of, "summary": result.summary[:2500]}
    for key, value in (("metrics", result.metrics), ("findings", result.findings[:8]), ("limitations", result.limitations[:8]), ("errors", result.errors[:3])):
        if value:
            row[key] = value
    return row


def _select_evidence(results: list[ToolEnvelope], evidence: list[EvidenceItem], cap: int) -> list[EvidenceItem]:
    """Round-robin across results so a fan-out's later holdings still reach the model.

    Taking the first ``cap`` items in ledger order starves everything after
    the first few results; six holdings' research would show the model one
    or two of them.
    """

    if len(evidence) <= cap:
        return list(evidence)
    by_id = {item.id: item for item in evidence}
    queues = [[item.id for item in result.evidence if item.id in by_id] for result in results]
    orphans = [item.id for item in evidence if not any(item.id in queue for queue in queues)]
    if orphans:
        queues.append(orphans)
    chosen: list[str] = []
    seen: set[str] = set()
    while len(chosen) < cap and any(queues):
        for queue in queues:
            while queue:
                candidate = queue.pop(0)
                if candidate not in seen:
                    seen.add(candidate)
                    chosen.append(candidate)
                    break
            if len(chosen) >= cap:
                break
    return [by_id[value] for value in chosen]


_NEARBY_UNGROUNDED = re.compile(r"^引用未支持邻近数字：([^（]+)")
_MISSING_CITATION = re.compile(r"^行情事实缺少邻近引用：“(.+?)…?”")
_FIGURE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?%?")


def _figures_in(excerpt: str, evidence: list[EvidenceItem]) -> list[tuple[str, list[str]]]:
    """The figures of an uncited excerpt with the evidence ids that carry each; figures no item carries are left out."""

    found: list[tuple[str, list[str]]] = []
    for raw in dict.fromkeys(match.group(0) for match in _FIGURE.finditer(excerpt)):
        digits = raw.replace(",", "").lstrip("+-").rstrip("%")
        if not digits or len(digits.replace(".", "")) < 2:
            continue
        ids = locate_number(digits, evidence)
        if ids:
            found.append((raw, ids))
    return found[:6]


_SHORT_WORDS = ("短一点", "简短", "简洁", "少一点", "精简", "short", "brief", "concise", "不要太长", "别太长")
SHORT_ANSWER_CHARS = 500


def visible_length(answer: str) -> int:
    """The answer's length as the user sees it: citation markers become footnote numbers on every surface."""

    from v2.agent_v2.verification import _CITATION

    return len(" ".join(_CITATION.sub("", answer or "").split()))


def short_answer_limit(preferences: list[str]) -> int:
    """The character limit a "回答短一点" preference implies, or 0 when no preference asks for brevity."""

    return SHORT_ANSWER_CHARS if any(word in str(value).lower() for value in preferences for word in _SHORT_WORDS) else 0


def _problem_set(report) -> frozenset[str]:
    """What a verification report objects to, as comparable keys (the quoted sentence stripped)."""

    from v2.agent_v2.verification import SOFT_PREFIX

    keys = set()
    if report.ungrounded_numbers:
        keys.add("number")
    if report.unknown_citations:
        keys.add("unknown")
    for warning in report.warnings:
        text = str(warning)
        if text.startswith(SOFT_PREFIX):
            continue
        # The kind of objection, not the figure or sentence it names: another
        # invented number in the same place is the same problem.
        keys.add("warning:" + text.split("：", 1)[0].split("（", 1)[0])
    return frozenset(keys)


def _problem_count(report) -> int:
    """How much a verification report found wrong: every flagged number, unknown id and blocking warning."""

    from v2.agent_v2.verification import SOFT_PREFIX

    blocking = [warning for warning in report.warnings if not str(warning).startswith(SOFT_PREFIX)]
    numbers = list(report.ungrounded_numbers)
    for warning in blocking:
        match = _NEARBY_UNGROUNDED.match(str(warning))
        if match:
            numbers.extend(value.strip() for value in match.group(1).split(",") if value.strip())
    return len(set(numbers)) + len(report.unknown_citations) + sum(1 for warning in blocking if not _NEARBY_UNGROUNDED.match(str(warning)))


def repair_instruction(report: VerificationReport, evidence: list[EvidenceItem] | None = None) -> str:
    """Tell the model exactly what failed and what must survive the rewrite.

    An ungrounded number is usually a real figure cited with the wrong id;
    naming the items that do carry it turns the repair into swapping an id
    instead of guessing.
    """

    lines = ["校验未通过，请重写完整回答。"]
    # Figures the verifier could not ground: the answer-wide list, plus the
    # ones a sentence cited with the wrong item (those arrive as warnings).
    numbers = list(report.ungrounded_numbers[:12])
    for warning in report.warnings:
        match = _NEARBY_UNGROUNDED.match(str(warning))
        if match:
            numbers.extend(value.strip() for value in match.group(1).split(",") if value.strip())
    numbers = list(dict.fromkeys(numbers))[:12]
    if numbers:
        located = []
        missing = []
        for number in numbers:
            ids = locate_number(number, list(evidence or []))
            (located if ids else missing).append((number, ids))
        if located:
            lines.append("以下数字引用的证据不含该数字，但下列证据含有它，请改用这些 id 引用：" + "；".join(f"{number} 见 " + "、".join(f"[{value}]" for value in ids) for number, ids in located) + "。")
        if missing:
            lines.append(
                "以下数字在本轮证据中找不到：" + "、".join(number for number, _ in missing) + "。"
                "只能使用证据中出现的数字；若是你自己的计算，请把算式完整写出（例如 22.4% + 18.2% = 40.6%）；无法支持的数字直接删掉，宁可省略也不要编造。"
            )
    if report.unknown_citations:
        lines.append("以下引用 id 不存在：" + "、".join(report.unknown_citations[:12]) + "。方括号内只能原样使用 evidence 数组中真实存在的 id。")
    for warning in report.warnings[:6]:
        if _NEARBY_UNGROUNDED.match(str(warning)):
            continue  # handled above, with the ids that carry the figures
        uncited = _MISSING_CITATION.match(str(warning))
        if uncited:
            # The draft said "these figures have no citeable item" twice on the
            # eval; name the items that carry them so the repair is a lookup.
            found = _figures_in(uncited.group(1), list(evidence or []))
            if found:
                lines.append(f"这句没有引用：“{uncited.group(1)}”。它里面的数字在这些证据里，请在对应数字后引用：" + "；".join(f"{number} 见 " + "、".join(f"[{value}]" for value in ids) for number, ids in found) + "。")
                continue
        lines.append(f"其他问题：{warning}。")
    if report.traced_numbers:
        lines.append("以下数字已通过校验，必须原样保留：" + "、".join(report.traced_numbers)[:400] + "。")
    lines.append("初稿里已经引用的证据 id 保留在对应的句子上，只改被指出的地方，不要在删减时把别的引用一起删掉；同一句里有几个数字来自不同证据时，每个数字后各写一个 id。")
    lines.append("请重新输出完整回答（所有段落），你的回复将完整替换初稿，是用户唯一会看到的文本。")
    return "\n".join(lines)


def _normalize_result_citations(
    answer: str,
    results: list[ToolEnvelope],
    evidence: list[EvidenceItem],
) -> str:
    """Resolve valid result-field references to their citeable derived evidence."""

    def replace_result_path(match: re.Match[str]) -> str:
        kind = match.group(1)
        suffix = match.group(2).replace("\\_", "_")
        candidates: list[EvidenceItem] = []
        for item in evidence:
            if item.metadata.get("citation_kind") != kind:
                continue
            matching_results = [result for result in results if not item.producer_run_id or result.run_id == item.producer_run_id]
            if kind == "metrics" and not any(_has_metric_path(result.metrics, suffix) for result in matching_results):
                continue
            if kind == "limitations" and (suffix or not any(result.limitations for result in matching_results)):
                continue
            candidates.append(item)
        return f"[{candidates[0].id}]" if len(candidates) == 1 else match.group(0)

    return _RESULT_CITATION.sub(replace_result_path, answer)


def _has_metric_path(metrics: dict[str, Any], suffix: str) -> bool:
    value: Any = metrics
    path = suffix.removeprefix(".")
    if not path:
        return bool(metrics)
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return False
        value = value[key]
    return value is not None
