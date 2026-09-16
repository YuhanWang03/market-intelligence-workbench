"""Explicit LangGraph orchestration, including repair cycles and interrupts."""

from __future__ import annotations

from dataclasses import dataclass, replace
from contextlib import contextmanager
import threading
import time
import uuid
import re

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command, interrupt

from v2.agent_v2.evidence import EvidenceLedger
from v2.agent_v2.models import (
    AgentResult, AnswerMode, EvidenceItem, ExecutionPlan, NormalizedRequest,
    PendingMutation, PlanTask, RouteDecision, RouteKind, RunStatus, VerificationReport,
)
from v2.agent_v2.routing import route
from v2.agent_v2.verification import complete_citations, verify_answer
from v2.agent_v3.context import RunContext, RunStopped, model_identity
from v2.agent_v3.contracts import GraphState, SemanticIntent, envelope_from, plain, plan_from
from v2.agent_v3.execution import build_executor, validate_plan
from v2.agent_v3.persistence import SessionStore


@dataclass(frozen=True)
class AgentV3Config:
    max_parallel: int = 4
    max_seconds: float = 180
    answer_reserve_seconds: float = 10
    max_repairs: int = 2
    enable_web: bool = False
    debate: bool = True
    confirmation_ttl: float = 300

    def __post_init__(self):
        if self.max_seconds <= 0 or self.max_parallel < 1 or self.max_repairs < 0 or self.answer_reserve_seconds < 0:
            raise ValueError("Invalid runtime budget")


class AgentV3:
    def __init__(self, *, registry, brain, config=None, store=None, checkpointer=None, reviewer=None, event_source=None):
        self.registry = registry
        self.brain = brain
        self.config = config or AgentV3Config()
        self.store = store or SessionStore()
        self.event_source = event_source
        self.reviewer = reviewer
        self.checkpointer = checkpointer or InMemorySaver()
        self._locks = {}
        self._locks_guard = threading.Lock()
        self.executor = build_executor(registry, self.store)
        self.graph = self._build()

    def _build(self):
        graph = StateGraph(GraphState, context_schema=RunContext)
        for name in ("resolve_context", "classify", "plan", "execute", "web_fallback", "synthesize", "verify", "repair", "fallback", "debate", "finish"):
            graph.add_node(name, self._node(name))
        graph.add_node("confirmation", self._confirmation)
        graph.add_edge(START, "resolve_context")
        graph.add_conditional_edges("resolve_context", lambda s: "finish" if s.get("status") else "classify", ["finish", "classify"])
        graph.add_conditional_edges("classify", lambda s: "finish" if s.get("status") else ("synthesize" if s.get("follow_up") or s.get("record_answer") else "plan"), ["finish", "synthesize", "plan"])
        graph.add_conditional_edges("plan", lambda s: "confirmation" if s.get("status") == "waiting_confirmation" else ("finish" if s.get("status") else "execute"), ["confirmation", "finish", "execute"])
        graph.add_conditional_edges("confirmation", lambda s: "execute" if s.get("approved") else "finish", ["execute", "finish"])
        graph.add_conditional_edges("execute", lambda s: "finish" if s.get("status") in {"failed", "cancelled"} else "web_fallback", ["finish", "web_fallback"])
        graph.add_edge("web_fallback", "synthesize")
        graph.add_conditional_edges("synthesize", lambda s: "fallback" if s.get("error") else "verify", ["fallback", "verify"])
        graph.add_conditional_edges("verify", self._after_verify, ["repair", "fallback", "debate", "finish"])
        graph.add_edge("repair", "verify")
        graph.add_edge("fallback", "finish")
        graph.add_edge("debate", "finish")
        graph.add_edge("finish", END)
        return graph.compile(checkpointer=self.checkpointer)

    def _node(self, name):
        def node(state: GraphState, runtime: Runtime[RunContext]):
            runtime.context.emit(name)
            try:
                if name not in {"fallback", "finish"}:
                    runtime.context.check()
                update = getattr(self, f"_{name}")(state, runtime.context)
            except RunStopped as exc:
                update = {"status": "cancelled" if str(exc) == "cancelled" else "partial", "stop_reason": str(exc), "error": str(exc)}
            except Exception as exc:
                update = {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}
                if name in {"classify", "plan", "execute"}:
                    update["status"] = "failed"
            return {**update, "trace": [*state.get("trace", []), name]}
        return node

    def _request(self, state):
        intent = SemanticIntent.model_validate(state["intent"]) if state.get("intent") else None
        return NormalizedRequest(original_text=state["text"], text=state["text"], session_id=state["session_id"], entities=tuple(intent.tickers) if intent else (), allow_web=state["allow_web"], metadata={"portfolio_metric":intent.portfolio_metric if intent else "", "holding_scope":bool(intent and intent.scope=="since_purchase"), "wants": intent.wants if intent else [], "page_context": state.get("page_context", {}), "experiment_arguments": intent.experiment_arguments if intent else {},"date_window":intent.date_window.model_dump() if intent and intent.date_window else None,"filing_items":intent.filing_items if intent else [],"filing_forms":intent.filing_forms if intent else []})

    def _resolve_context(self, state, run):
        page = state.get("page_context", {})
        selected = page.get("selection") or {}
        if selected.get("kind") not in {"anomaly", "price_alert"} or not selected.get("record_id"):
            return {}
        from v2.agent_v3.monitor_history import evidence
        from v2.agent_v3.tools import bounded_call
        try:
            if self.event_source is None:
                raise LookupError("Monitor archive is not configured")
            row = bounded_call(lambda: self.event_source.get(selected["record_id"], selected.get("ticker", "")), run)
        except RunStopped:
            raise
        except Exception as exc:
            return {"status": "partial", "answer": "无法回查所选监控记录，可能已清理或与股票不匹配。请重新选择记录；本轮不使用浏览器摘要代替原始记录。", "error": type(exc).__name__, "stop_reason": "monitor_record_unavailable"}
        item = evidence(row)
        selection = {**selected, "ticker": row["tickers"] or "", "occurred_at": row["ts"], "excerpt": item.claim[:4000]}
        return {"page_context": {**page, "selection": selection}, "monitor_evidence": [plain(item)]}

    def _classify(self, state, run):
        understanding = self.brain.classify(state["text"], {**state.get("history", {}), "page_context": state.get("page_context", {})}, run)
        if not isinstance(understanding, SemanticIntent):
            understanding = SemanticIntent.model_validate(understanding)
        holding=state.get("history",{}).get("portfolio_context",{})
        if understanding.market_scope=="us_broad":
            understanding=understanding.model_copy(update={"tickers":["SPY","QQQ","DIA"],"wants":["performance"],"kind":"lookup","portfolio_scope":False,"portfolio_followup":"","portfolio_metric":"","analysis_scope":"focused","refers_back":False,"clarification":""})
        if understanding.portfolio_followup=="explain_position":
            ticker=next(iter(understanding.tickers),None) or holding.get("ticker")
            if not ticker:
                return {"status":"waiting_clarification","answer":"请指定需要解释的持仓股票；上一轮没有唯一的排名对象。"}
            understanding=understanding.model_copy(update={"tickers":[ticker],"scope":"since_purchase","wants":["attribution"],"portfolio_scope":True,"date_window":None,"refers_back":False,"portfolio_metric":holding.get("metric","unrealized_percent")})
        elif understanding.portfolio_followup=="rerank":
            understanding=understanding.model_copy(update={"tickers":[],"portfolio_scope":True,"wants":["ranking"],"refers_back":False})
        if understanding.lab:
            understanding = understanding.model_copy(update={"kind":"lab", "analysis_scope":"focused", "refers_back":False})
        if understanding.analysis_scope == "company" and understanding.kind != "lab" and 'compare' not in understanding.wants:
            understanding = understanding.model_copy(update={"kind":"research", "wants":["overview", "performance", "valuation", "risk"], "refers_back":False, "investigation":""})
        if "news" in understanding.wants and set(understanding.wants) <= {"news", "catalysts"}:
            understanding = understanding.model_copy(update={"wants":["news"]})
        if (not understanding.date_window or (understanding.scope=="recent" and not understanding.date_window_explicit)) and ("news" in understanding.wants or (understanding.scope == "recent" and set(understanding.wants) & {"attribution", "drawdown", "runup"})):
            from datetime import date, timedelta
            from v2.agent_v3.contracts import DateWindow
            attribution = bool(set(understanding.wants) & {"attribution", "drawdown", "runup"})
            days = 30 if attribution else 14
            understanding = understanding.model_copy(update={"date_window":DateWindow(start=(date.today()-timedelta(days=days-1)).isoformat(),end=date.today().isoformat(),basis="event" if attribution else "publication")})
        request = NormalizedRequest(state["text"], state["text"], entities=tuple(understanding.tickers))
        decision = route(request, intent=understanding.domain())
        selected_evidence = [row for row in state.get("monitor_evidence", []) if row.get("entity") in understanding.tickers]
        update = {"intent": understanding.model_dump(), "route": decision.kind.value, "monitor_evidence": selected_evidence}
        if understanding.use_selected_record and selected_evidence:
            plan = ExecutionPlan(state["text"], RouteKind.RESEARCH, answer_mode=AnswerMode.RESEARCH_GROUNDED)
            return {**update, "record_answer": True, "route": RouteKind.RESEARCH.value, "plan": plain(plan), "results": [], "evidence": selected_evidence}
        if understanding.clarification:
            return {**update, "status": "waiting_clarification", "answer": understanding.clarification}
        if understanding.data_target == "etf_holdings" and not self.registry.registered("etf.holdings"):
            from v2.agent_v3.availability import unavailable
            return {**update, **unavailable(["etf.holdings"])}
        if understanding.kind in {"lookup", "research"} and "performance" in understanding.wants and not (understanding.tickers or understanding.portfolio_scope or understanding.watchlist_scope):
            return {**update, "status": "waiting_clarification", "answer": "你想查询哪只股票或哪个指数？请提供名称或代码。"}
        previous = state.get("history", {}).get("previous")
        previous_tickers = set((previous or {}).get("tickers", [])) or {row.get("entity") for row in (previous or {}).get("evidence", [])}
        same_subject = set(understanding.tickers) == previous_tickers
        same_permission = (previous or {}).get("allow_web", False) == state["allow_web"]
        # Only a pure rewrite may skip execution. Expansion and refreshed queries
        # need a new plan even when the model marks them as referring back.
        reuse = understanding.refers_back and understanding.follow_up_mode == "restate" and same_subject and same_permission
        if reuse and previous and not previous.get("evidence"):
            return {**update, "status": "partial", "answer": previous.get("answer") or "上一轮没有可复述的证据。", "stop_reason": "follow_up_without_evidence"}
        if reuse and previous and previous.get("evidence"):
            plan = ExecutionPlan(state["text"], decision.kind, answer_mode=AnswerMode.RESEARCH_GROUNDED)
            if understanding.follow_up_mode == "restate":
                answer = previous.get("answer", "")
                cited = [row for row in previous["evidence"] if "[" + row["id"] + "]" in answer]
                return {**update, "follow_up": True, "restatement_of": answer, "plan": plain(plan), "results": [], "evidence": cited, "inherited_partial": previous.get("status") == "partial"}
            return {**update, "follow_up": True, "plan": plain(plan), "results": previous["results"], "evidence": previous["evidence"]}
        return update

    def _plan(self, state, run):
        request = self._request(state)
        request = replace(request, metadata={**request.metadata, "data_target":state.get("intent", {}).get("data_target","auto"), "holdings_top":state.get("intent",{}).get("holdings_top",5)})
        intent = SemanticIntent.model_validate(state["intent"]).domain()
        decision = route(request, intent=intent)
        plan = self.brain.plan(request, decision, self.registry, run)
        scoped_tasks = []
        for task in plan.tasks:
            if task.capability == "filings.recent":
                arguments = dict(task.arguments)
                window = request.metadata.get("date_window")
                if window and window["basis"] == "filing":
                    arguments.update(since=window["start"],until=window["end"])
                if request.metadata.get("filing_forms"):
                    arguments["forms"] = request.metadata["filing_forms"]
                task = replace(task,arguments=arguments)
            scoped_tasks.append(task)
        plan = replace(plan,tasks=tuple(scoped_tasks))
        missing = [task.capability for task in plan.tasks if task.required and not self.registry.registered(task.capability)]
        if missing:
            from v2.agent_v3.availability import unavailable
            return {"plan": plain(plan), **unavailable(missing)}
        validate_plan(plan, self.registry)
        update = {"plan": plain(plan)}
        mutation = any(self.registry.catalog.get(task.capability).mutating for task in plan.tasks)
        if mutation:
            if not all(self.registry.registered(task.capability) for task in plan.tasks):
                raise ValueError("State mutations are not enabled in this runtime")
            if not state["session_id"]:
                raise ValueError("A mutation requires a session")
            plan = replace(plan, requires_confirmation=True)
            return {"plan": plain(plan), "status": "waiting_confirmation", "confirmation_expires": time.time() + self.config.confirmation_ttl}
        if plan.direct_answer:
            return {**update, "status": "completed" if plan.answer_mode == AnswerMode.GENERAL_KNOWLEDGE else "waiting_clarification", "answer": plan.direct_answer}
        return update

    def _confirmation(self, state: GraphState, runtime: Runtime[RunContext]):
        # Pure pre-interrupt work: this node is replayed on resume.
        response = interrupt({"kind": "confirmation", "run_id": state["run_id"], "tasks": state["plan"]["tasks"], "expires_at": state["confirmation_expires"]})
        valid = isinstance(response, dict) and response.get("run_id") == state["run_id"] and response.get("approve") is True
        if not valid or time.time() > state["confirmation_expires"] or runtime.context.cancelled:
            return {"approved": False, "status": "cancelled", "answer": "操作已取消或确认已过期，未执行修改。"}
        return {"approved": True, "status": "", "trace": [*state.get("trace", []), "confirmation"]}

    def _execute(self, state, run):
        plan = plan_from(state["plan"])
        validate_plan(plan, self.registry)
        if not plan.tasks:
            return {"results": [], "evidence": state.get("monitor_evidence", [])}
        portfolio = any(task.capability == "account.overview" for task in plan.tasks)
        reserve = min(max(self.config.answer_reserve_seconds, 45), run.remaining() * .4) if portfolio else self.config.answer_reserve_seconds
        deadline = run.deadline - reserve
        if portfolio:
            deadline = min(deadline, time.monotonic() + 60)
        tool_run = replace(run, deadline=max(time.monotonic(), deadline))
        output = self.executor.invoke({"plan": state["plan"], "request": plain(self._request(state)), "approved": state.get("approved", False), "updates": []}, context=tool_run, config={"max_concurrency": self.config.max_parallel, "recursion_limit": 150})
        ledger = EvidenceLedger()
        results = [envelope_from(row) for row in output["done"].values()]
        for item in state.get("monitor_evidence", []):
            from v2.agent_v2.models import ToolEnvelope, ResultStatus
            ledger.ingest(ToolEnvelope("monitor.record", ResultStatus.COMPLETED, evidence=[EvidenceItem(**item)]))
        for result in results:
            ledger.ingest(result)
        notes = output.get("notes", [])
        if notes:
            from v2.agent_v2.models import ResultStatus, ToolEnvelope
            note = ToolEnvelope("execution.coverage", ResultStatus.PARTIAL_DATA, limitations=notes, evidence=[EvidenceItem(id=f"coverage-{index}", entity="execution", claim=text, source_id="agent_v3", metadata={"citation_kind": "limitations"}) for index, text in enumerate(notes)])
            results.append(note)
            ledger.ingest(note)
        stopped = "cancelled" if run.cancelled else ("deadline" if any(any(error == "deadline" for error in result.errors) for result in results) else "completed")
        return {"results": [plain(row) for row in results], "evidence": [plain(row) for row in ledger.items()], "stop_reason": stopped, **({"status": "cancelled"} if run.cancelled else {})}

    def _web_fallback(self, state, run):
        plan = plan_from(state["plan"])
        results = [envelope_from(row) for row in state.get("results", [])]
        eligible = state["allow_web"] and plan.web_fallback_allowed and self.registry.registered("web.research") and state["route"] in {"research", "fast_lookup"}
        if not eligible or any(row.capability == "web.research" for row in results) or (state.get("evidence") and all(row.ok for row in results)) or run.remaining() <= self.config.answer_reserve_seconds:
            return {}
        task = PlanTask("web-fallback", "web.research", {"query": state["text"][:500], "topic": "financial_research"}, required=False)
        result = self.registry.execute(task, plan, self._request(state), replace(run, deadline=run.deadline-self.config.answer_reserve_seconds), prior=results)
        results.append(result)
        ledger = EvidenceLedger()
        for row in results:
            ledger.ingest(row)
        return {"results": [plain(row) for row in results], "evidence": [plain(row) for row in ledger.items()]}

    def _synthesize(self, state, run):
        deterministic=[row.get("metadata",{}).get("deterministic_answer") for row in state.get("results",[])]
        if len(deterministic)==1 and deterministic[0]:
            return {"answer":deterministic[0],"error":""}
        if set(state.get("intent", {}).get("wants", [])) == {"news"} and not any(row.get("metadata", {}).get("quote_located") for row in state.get("evidence", [])):
            return self._fallback(state, run)
        if state.get("approved"):
            return {**self._fallback(state, run), "fallback": False}
        if not state.get("evidence") and state.get("plan", {}).get("answer_mode") != "general_knowledge":
            return self._fallback(state, run)
        return {"answer": self.brain.draft(state, self.registry, run), "error": ""}

    def _report(self, answer, state, run):
        mode = AnswerMode(state.get("plan", {}).get("answer_mode", "insufficient_evidence"))
        evidence = [EvidenceItem(**row) for row in state.get("evidence", [])]
        results = [envelope_from(row) for row in state.get("results", [])]
        # Citation delimiters are a wire format. Normalize only exact known IDs;
        # never erase arbitrary bracketed prose or unknown identifiers.
        for item in evidence:
            for formatted in (f"【{item.id}】", f"\\[{item.id}\\]", f"［{item.id}］"):
                answer = answer.replace(formatted, f"[{item.id}]")
        known_ids = {item.id for item in evidence}
        def citation_group(match):
            ids = re.split(r"\s*[,;，；]\s*", match[1])
            return "".join(f"[{key}]" for key in ids) if all(key in known_ids for key in ids) else match[0]
        answer = re.sub(r"\[([A-Za-z0-9_.:;,，；\s-]+)\]", citation_group, answer)
        answer, _ = complete_citations(answer, evidence, results)
        objectives = set(state.get("intent", {}).get("wants", []))
        if state.get("allow_web"):
            from v2.agent_v2.models import ToolEnvelope, ResultStatus
            results.append(ToolEnvelope("permission.boundary", ResultStatus.COMPLETED, metadata={"answer_constraints":[{
                "forbid_claim":"本轮用户未授权网页检索，或网页检索权限未开启。", "warning":"本轮网页检索已授权。检索无结果或失败不等于未授权，请纠正权限说明。"}]}))
        if objectives & {"attribution", "drawdown", "runup"}:
            from v2.agent_v2.models import ToolEnvelope, ResultStatus
            results.append(ToolEnvelope('temporal.boundary',ResultStatus.COMPLETED,metadata={'answer_constraints':[{'forbid_claim':'把盘前、盘中或盘后的报道跌幅当作同日收盘跌幅，或未对齐事件日期便把某日新闻说成其他日期下跌或整个多日区间涨跌的确定原因；仅凭卖出看涨期权就判断市场押注下跌。','warning':'明确报道时间、事件日期及交易时段；单日候选报道不能解释整段收益，卖出看涨期权不足以确定看跌方向。'}]}))
            results.append(ToolEnvelope("answer.boundaries", ResultStatus.COMPLETED, metadata={"answer_constraints":[{
                "forbid_claim":"仅凭个股和基准的涨跌方向或相对表现，就排除板块、大盘等因素对个股下跌的作用，或确认任何下跌原因。",
                "warning":"相对表现不能用于确认或排除下跌原因；只陈述行情比较，原因未确认。"}]}))
            window = state.get("intent", {}).get("date_window")
            if window and window["start"] != window["end"]:
                results.append(ToolEnvelope("period.boundary", ResultStatus.COMPLETED, metadata={"answer_constraints":[{
                    "forbid_claim":"将多日区间的涨跌归因问题缩为最后一个交易日，用未找到同日催化剂替代整个区间的分析。",
                    "warning":f"问题研究范围为{window['start']}至{window['end']}，请分析该区间的候选驱动及缺口；不以同日催化剂作为整个区间的结论。"}]}))
        verification_text = answer
        # Source URLs are identifiers, not financial quantities. Only exclude
        # exact URLs already registered in evidence; preserve the displayed answer.
        for url in sorted({item.source_url for item in evidence if item.source_url},key=len,reverse=True):
            verification_text = verification_text.replace(url,"来源链接")
        report = verify_answer(verification_text, evidence, answer_mode=mode, results=results, judge=lambda rows: self.brain.judge(rows, run))
        report = plain(report)
        forbidden=[item.id for item in evidence if item.metadata.get('exclude_from_comparison') and f'[{item.id}]' in answer]
        if forbidden:
            report['ok']=False
            report.setdefault('warnings',[]).append('Unaligned financial evidence cannot be used in this comparison: '+', '.join(forbidden))
        malformed=[match[1] for match in re.finditer(r"\[([A-Za-z][A-Za-z0-9_.:~,; -]*)\](?!\()",answer) if match[1] not in known_ids]
        if malformed:
            report["ok"]=False
            report.setdefault("warnings",[]).append("Invalid citation identifiers; use one exact evidence ID per bracket: "+", ".join(malformed))
        holding_records=[row.get("metadata",{}) for row in state.get("results",[]) if row.get("capability")=="account.position_analysis" and row.get("metadata",{}).get("holding_period",{}).get("verified")]
        if holding_records and hasattr(self.brain,"audit_portfolio"):
            try:
                objections=self.brain.audit_portfolio(answer,holding_records,run)
            except RunStopped:
                raise
            except Exception:
                objections=["Holding scope audit unavailable; answer has not passed scope validation"]
            if objections:
                report["ok"]=False
                report.setdefault("warnings",[]).extend(objections)
        if state.get("restatement_of"):
            try:
                adds_facts = self.brain.check_restatement(state["restatement_of"], answer, run)
            except Exception:
                adds_facts = True
            if adds_facts:
                report["ok"] = False
                report.setdefault("warnings", []).append("Restatement adds or changes facts beyond the previous answer")
        return answer, report

    def _verify(self, state, run):
        answer, report = self._report(state.get("answer", ""), state, run)
        attempt = report if report.get("ok") else {**report, "rejected_draft": answer[:12000]}
        return {"answer": answer, "report": report, "attempts": [*state.get("attempts", []), attempt]}

    def _debate_gate(self, state):
        """Why a verified answer does not go to adversarial review; empty when it does."""
        if not self.config.debate:
            return "disabled"
        if not self.reviewer:
            return "no_reviewer"
        if state.get("follow_up"):
            return "follow_up"
        if state.get("route") != "research":
            return f"route={state.get('route') or '?'}"
        if any(row.get("metadata", {}).get("review_stage") == "source_cross_review" for row in state.get("results", [])):
            return "source_cross_review"
        return ""

    def _after_verify(self, state):
        if state.get("fallback") or state.get("status") == "cancelled":
            return "finish"
        if state.get("report", {}).get("ok") and not state.get("error"):
            return "finish" if self._debate_gate(state) else "debate"
        if state.get("error") or state.get("repairs", 0) >= self.config.max_repairs:
            return "fallback"
        return "repair"

    def _repair(self, state, run):
        return {"answer": self.brain.draft(state, self.registry, run, repair=True), "repairs": state.get("repairs", 0) + 1, "error": ""}

    def _fallback(self, state, run):
        overview = next((row.get("metadata", {}).get("portfolio_overview_answer") for row in state.get("results", []) if row.get("capability") == "account.overview" and row.get("metadata", {}).get("portfolio_overview_answer")), None)
        if overview:
            optional = [row for row in state.get("results", []) if row.get("capability") != "account.overview" and not row.get("metadata", {}).get("no_work_required")]
            succeeded = sum(row.get("status") == "completed" for row in optional)
            if optional:
                overview += f"\n\n补充研究：计划 {len(optional)} 项，成功 {succeeded} 项，未完成 {len(optional)-succeeded} 项。以上组合汇总保留；尚未完成的研究不作为结论依据。"
            return {"answer": overview, "fallback": True, "report": {"ok": False, "warnings": ["Portfolio arithmetic preserved; model interpretation unavailable or rejected"]}}
        if state.get("restatement_of"):
            return {"answer": "本轮改写未通过范围校验，保留上一条回答：\n" + state["restatement_of"], "fallback": True, "report": {"ok": False, "warnings": ["Restatement rejected; previous answer preserved"]}}
        if set(state.get("intent", {}).get("wants", [])) == {"news"}:
            subjects = set(state.get("intent", {}).get("tickers", []))
            verified = [row for row in state.get("evidence", []) if row.get("metadata", {}).get("quote_located") and (not subjects or row.get("entity") in subjects)]
            if state.get("allow_web") and verified:
                answer = "综合摘要未通过校验，以下保留已定位原文的报道内容；报道中的说法尚不代表独立交叉核实：\n" + "\n".join(f"- {row.get('as_of', '')} · {row.get('source_title') or '来源原文'}：{row['metadata'].get('quote') or row['claim']} [{row['id']}]" for row in verified[:12])
                return {"answer": answer, "fallback": True, "report": {"ok": False, "warnings": ["News synthesis rejected; located source evidence preserved"]}}
            reason = "本轮未授权网页检索。开启网页检索后可以重新查询。" if not state.get("allow_web") else "本次未取得通过原文核验的新闻内容，暂时无法提供可靠的新闻摘要。"
            return {"answer": reason, "fallback": True, "report": {"ok": False, "warnings": ["News evidence unavailable"]}}
        subjects = set(state.get("intent", {}).get("tickers", []))
        evidence = [row for row in state.get("evidence", []) if (not subjects or row.get("entity") in subjects) and not row.get('metadata',{}).get('exclude_from_comparison')]
        answer = "\n".join(f"- {row['claim']} [{row['id']}]" for row in evidence[:12])
        answer = "已获取的证据：\n" + answer if answer else "当前没有足够的可引用证据，无法完成该问题。"
        limitations = list(dict.fromkeys(note for result in state.get("results", []) for note in result.get("limitations", [])))
        if limitations:
            answer += "\n限制：\n" + "\n".join(f"- {note}" for note in limitations)
        return {"answer": answer, "fallback": True, "report": {"ok": False, "warnings": ["Deterministic evidence summary; model answer unavailable or rejected"]}}

    def _debate(self, state, run):
        """Adversarial review of an already verified answer.

        The record in ``state["debate"]`` says what happened, whichever way it
        went: the reviewer's objections, whether a revision was drafted and
        whether it passed verification.  A revision that fails verification
        keeps the original answer, and the objections are attached to its
        report as warnings so the reader still sees the review.
        """
        if run.remaining() <= self.config.answer_reserve_seconds:
            return {"debate": {"ran": False, "skipped": "time_budget"}}
        record = {"ran": True, "objections": [], "revised": False, "revised_verified": False}
        try:
            objections = list(self.reviewer(state, run) or [])
            record["objections"] = objections
            if not objections:
                return {"objections": [], "debate": record}
            material = [row for row in objections if not isinstance(row, dict) or row.get("severity", "material") == "material"]
            record["material"] = len(material)
            if not material:
                # Wording-only review: not worth a redraft and a second verification; show it instead.
                kept = dict(state.get("report", {}))
                kept["warnings"] = [*kept.get("warnings", []), *(f"审阅提示：{row.get('objection')} [{row.get('evidence_id', '?')}]" for row in objections)]
                return {"objections": objections, "debate": record, "report": kept}
            revised = self.brain.draft(state, self.registry, run, objections=material)
            record["revised"] = True
            answer, report = self._report(revised, state, run)
            record["revised_verified"] = bool(report.get("ok"))
            if report.get("ok"):
                return {"objections": objections, "debate": record, "answer": answer, "report": report, "attempts": [*state.get("attempts", []), report]}
            kept = dict(state.get("report", {}))
            kept["warnings"] = [*kept.get("warnings", []), *(f"审阅异议（修订稿未通过校验，保留原答案）：{row.get('objection', row)} [{row.get('evidence_id', '?')}]" if isinstance(row, dict) else f"审阅异议：{row}" for row in objections)]
            return {"objections": objections, "debate": record, "report": kept, "attempts": [*state.get("attempts", []), {**report, "rejected_draft": answer[:12000]}]}
        except Exception as exc:  # noqa: BLE001 — optional review must never discard the verified answer
            record["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
            return {"debate": record}

    def _finish(self, state, run):
        status = state.get("status", "")
        if run.cancelled:
            status = "cancelled"
        if status not in {"waiting_confirmation", "waiting_clarification", "cancelled", "failed"}:
            failures = any(not envelope_from(row).ok or envelope_from(row).status.value == "partial_data" or (row.get("status") == "cached" and bool(row.get("limitations"))) for row in state.get("results", []))
            status = "partial" if status == "partial" or state.get("inherited_partial") or failures or state.get("error") or state.get("fallback") or not state.get("report", {}).get("ok", bool(state.get("answer"))) else "completed"
        answer = state.get("answer", "")
        if status == "cancelled":
            answer = "已取消本次任务。" if not answer else answer
        elif not answer:
            answer = "本次任务未完成，请补充信息或稍后重试。"
        update = {"status": status, "answer": answer, "usage": list(run.usage)}
        if "debate" not in state:
            skipped = "fallback" if state.get("fallback") else ("verification_failed" if not state.get("report", {}).get("ok") else (self._debate_gate(state) or status))
            update["debate"] = {"ran": False, "skipped": skipped}
        return update

    @contextmanager
    def _session_lock(self, session_id):
        with self._locks_guard:
            entry = self._locks.setdefault(session_id, [threading.RLock(), 0])
            entry[1] += 1
        try:
            with entry[0]:
                yield
        finally:
            with self._locks_guard:
                entry[1] -= 1
                if entry[1] == 0:
                    self._locks.pop(session_id, None)

    def run(self, text, *, session_id="", allow_web=False, on_progress=None, cancel_event=None, page_context=None):
        from v2.agent_v3.page_context import PageContext
        page_context = PageContext.model_validate(page_context).model_dump(mode="json") if page_context is not None else {}
        if not text or not text.strip():
            raise ValueError("Question must not be blank")
        if len(text) > 8000 or len(session_id) > 128:
            raise ValueError("Question or session identifier exceeds size limit")
        with self._session_lock(session_id or uuid.uuid4().hex):
            history = self.store.get(session_id)
            # A new request supersedes an old confirmation. Explicit resume is a separate API.
            history.pop("pending", None)
            self.store.put(session_id, history)
            if history.get("clarification"):
                text = f"{history['clarification']['question']}\n补充信息：{text}"
            run_id = "agent-v3-" + uuid.uuid4().hex[:16]
            state = {"run_id": run_id, "session_id": session_id, "text": text.strip(), "allow_web": bool(allow_web and self.config.enable_web), "history": history, "results": [], "evidence": [], "trace": [], "attempts": [], "repairs": 0}
            state["page_context"] = page_context
            return self._invoke(state, run_id, session_id, on_progress, cancel_event)

    def recover(self, *, session_id, run_id, allow_web=False, on_progress=None):
        """Resume an interrupted read-only graph; completed tool reads are journaled."""
        with self._session_lock(session_id):
            saved = self.graph.get_state({"configurable":{"thread_id":run_id}})
            state = saved.values
            if not state or state.get("session_id") != session_id or not saved.next:
                raise ValueError("No interrupted checkpoint for this session")
            if state.get("allow_web") and not (allow_web and self.config.enable_web):
                raise PermissionError("Web permission changed; submit a new request")
            if state.get("approved") or any(self.registry.catalog.get(task["capability"]).mutating for task in state.get("plan",{}).get("tasks",[])):
                raise PermissionError("State mutations require the confirmation workflow")
            return self._invoke(None,run_id,session_id,on_progress,None)

    def resume(self, *, session_id, run_id, approve: bool, on_progress=None, cancel_event=None):
        if type(approve) is not bool:
            raise ValueError("approve must be a boolean")
        with self._session_lock(session_id):
            pending = self.store.get(session_id).get("pending", {})
            if pending.get("run_id") != run_id:
                raise PermissionError("No matching confirmation for this session and run")
            return self._invoke(Command(resume={"approve": approve, "run_id": run_id}), run_id, session_id, on_progress, cancel_event)

    def _invoke(self, input_value, run_id, session_id, progress, cancellation):
        from v2.usage_context import usage_run

        started = time.monotonic()
        provider, model_name = model_identity(getattr(self.brain, "model", None))
        run = RunContext(run_id, started + self.config.max_seconds, cancellation, progress, provider=provider, model=model_name)
        config = {"configurable": {"thread_id": run_id}, "max_concurrency": self.config.max_parallel, "recursion_limit": 80}
        # Every provider call made for this question carries the run id in the
        # usage ledger, the same attribution V2 gives its runs.
        with usage_run(run_id):
            output = self.graph.invoke(input_value, config=config, context=run)
        if output.get("__interrupt__"):
            output = dict(self.graph.get_state(config).values)
            output["status"] = "waiting_confirmation"
            task = output["plan"]["tasks"][0]
            import json
            output["answer"] = "请确认以下修改：" + json.dumps(task["arguments"], ensure_ascii=False) + "。确认前不会执行。"
        result = self._result(output, int((time.monotonic()-started)*1000), run)
        history = self.store.get(session_id)
        history.pop("pending", None)
        history.pop("clarification", None)
        if result.status == RunStatus.WAITING_CONFIRMATION:
            history["pending"] = {"run_id": run_id, "expires": output["confirmation_expires"]}
        elif result.status == RunStatus.WAITING_CLARIFICATION:
            history["clarification"] = {"question": output["text"], "clarification": result.answer}
        else:
            history["turns"] = [*history.get("turns", []), {"question": output["text"], "answer": result.answer[:1500], "intent": output.get("intent", {})}][-4:]
            history["previous"] = {"answer": result.answer, "results": output.get("results", []), "evidence": output.get("evidence", []), "tickers": list(result.request.entities), "allow_web": output.get("allow_web", False), "status": result.status.value}
            contexts=[row.get("metadata",{}).get("portfolio_context") for row in output.get("results",[])]
            contexts=[row for row in contexts if row is not None]
            if contexts:history["portfolio_context"]=contexts[-1]
            elif not output.get("intent",{}).get("portfolio_followup"):
                history.pop("portfolio_context",None)
        self.store.put(session_id, history)
        return result

    def _result(self, state, elapsed_ms, run):
        plan = plan_from(state["plan"]) if state.get("plan") else ExecutionPlan(state["text"], RouteKind(state.get("route", "fast_lookup")))
        pending = None
        if state["status"] == "waiting_confirmation":
            task = plan.tasks[0]
            pending = PendingMutation(task.arguments["operation"], task.arguments["payload"], task.purpose or task.capability)
        return AgentResult(run_id=state["run_id"], request=self._request(state), route=RouteDecision(plan.route, (), "LangGraph semantic routing"), plan=plan, status=RunStatus(state["status"]), answer=state["answer"], answer_mode=plan.answer_mode, results=[envelope_from(row) for row in state.get("results", [])], evidence=[EvidenceItem(**row) for row in state.get("evidence", [])], verification=VerificationReport(**state.get("report", {})), elapsed_ms=elapsed_ms, error=state.get("error", ""), stop_reason=state.get("stop_reason", ""), pending_mutation=pending, synthesis={"framework": "langgraph", "nodes": state.get("trace", []), "attempts": state.get("attempts", []), "outcome": "fallback" if state.get("fallback") else ("repaired" if state.get("repairs") else "model"), "usage": list(run.usage), "objections": state.get("objections", []), "debate": state.get("debate", {})})
