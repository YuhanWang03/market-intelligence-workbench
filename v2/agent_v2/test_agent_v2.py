"""Contract tests for the initial Agent V2 framework; no API keys required."""

from __future__ import annotations

import asyncio
import json
from datetime import date, datetime, timedelta
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from v2.agent_common.llm import LLMResponse, ScriptedLLM
from v2.agent_v2.adapters.lab import register_lab_capabilities
from v2.agent_v2.adapters.market import _observation_state, register_market_capabilities
from v2.agent_v2.adapters.research import register_research_capabilities
from v2.agent_v2.adapters.tavily_web import TavilyWebSearchPort
from v2.agent_v2.adapters.web import register_web_capability
from v2.agent_v2.adapters.workspace_lab import LabBinding, WorkspaceLabPort
from v2.agent_v2.catalog import default_catalog
from v2.agent_v2.eval.runner import run_suite
from v2.agent_v2.eval.scenario_cases import PORTFOLIO_CARD as _PORTFOLIO_CARD, build_drawdown_registry as _framed_registry
from v2.agent_v2.execution import CapabilityRegistry, ExecutionContext, ExecutionEngine, PlanValidationError
from v2.agent_v2.interfaces.telegram import TelegramFacade, TelegramMessage
from v2.agent_v2.interfaces.web import WebFacade, WebRequest
from v2.agent_v2.llm import LLMEvidenceSynthesizer, StructuredLLMPlanner
from v2.agent_v2.models import (
    AnswerMode,
    BudgetClass,
    EvidenceItem,
    ExecutionPlan,
    NormalizedRequest,
    PlanTask,
    ResultStatus,
    RouteKind,
    RunStatus,
    ToolEnvelope,
)
from v2.agent_v2.orchestrator import AgentV2, AgentV2Config
from v2.agent_v2.planning import RulePlanner
from v2.agent_v2.routing import normalize_request, route
from v2.agent_v2.session import ShortTermSession
from v2.agent_v2.synthesis import EvidenceSummarySynthesizer
from v2.agent_v2.verification import verify_answer


def _context() -> ExecutionContext:
    return ExecutionContext("test-run", NormalizedRequest("q", "q"), BudgetClass.STANDARD)


def test_router_separates_knowledge_research_lab_and_commands():
    assert route(normalize_request("什么是自由现金流？")).kind == RouteKind.GENERAL_KNOWLEDGE
    assert route(normalize_request("比较 NVDA 和 AMD 的风险")).kind == RouteKind.RESEARCH
    # A "which is the better buy" comparison is research: it earns the comparison budget and the debater.
    for text in ("MU和SNDK哪个更值得购买？", "NVDA 和 AMD 哪个更好", "AMD 值得买吗", "现在 ARM 值得入手吗"):
        assert route(normalize_request(text)).kind == RouteKind.RESEARCH, text
    assert route(normalize_request("AMD 今天成交量")).kind == RouteKind.FAST_LOOKUP
    assert route(normalize_request("回测 NVDA 动量策略")).kind == RouteKind.LAB
    assert route(normalize_request("把 NVDA 加入关注列表")).kind == RouteKind.COMMAND


def test_recent_stock_performance_uses_market_data_instead_of_fundamentals():
    request = normalize_request("AMD最近表现如何？")
    plan = RulePlanner().plan(request, route(request))
    assert plan.tasks[0].capability == "market.performance"
    assert plan.tasks[0].arguments == {"ticker": "AMD"}


@pytest.mark.parametrize("query", ["AMD表现如何？", "AMD股票表现怎么样？", "AMD今天成交量是不是低？", "AMD是不是放量上涨？", "AMD最近波动率多高？"])
def test_bare_stock_performance_defaults_to_recent_market_data(query):
    request = normalize_request(query)
    plan = RulePlanner().plan(request, route(request))
    assert plan.tasks[0].capability == "market.performance"


@pytest.mark.parametrize("query", ["AMD经营表现如何？", "AMD基本面表现如何？", "AMD最近财报表现如何？", "AMD技术面表现如何？"])
def test_non_price_performance_language_stays_with_research(query):
    request = normalize_request(query)
    plan = RulePlanner().plan(request, route(request))
    assert plan.tasks[0].capability == "research.stock"


def test_recent_earnings_quality_is_not_misrouted_as_price_performance():
    request = normalize_request("AMD最近的收益质量如何？")
    plan = RulePlanner().plan(request, route(request))
    assert plan.tasks[0].capability == "research.stock"


def test_move_explanation_cannot_be_overridden_by_the_llm_planner():
    llm = ScriptedLLM([LLMResponse(text='{"tasks":[{"id":"wrong","capability":"research.stock","arguments":{"ticker":"AMD"}}]}')])
    catalog = default_catalog()
    request = normalize_request("AMD今天为什么涨？")
    plan = StructuredLLMPlanner(llm, catalog).plan(request, route(request))
    assert plan.tasks[0].capability == "market.explain_move"
    assert not llm.calls


def test_market_observation_becomes_final_at_the_regular_close():
    before_close = _observation_state("2026-09-08", datetime(2026, 9, 8, 15, 59, tzinfo=ZoneInfo("America/New_York")))
    after_close = _observation_state("2026-09-08", datetime(2026, 9, 8, 16, 0, tzinfo=ZoneInfo("America/New_York")))
    assert before_close["is_intraday"] is True
    assert before_close["volume_is_final"] is False
    assert after_close["is_intraday"] is False
    assert after_close["volume_is_final"] is True


def test_catalog_exposes_only_requested_packs():
    catalog = default_catalog()
    research = catalog.names(["research"])
    assert "research.stock" in research
    assert "lab.backtest" not in research
    assert "web.research" not in research


def test_registry_blocks_unconfirmed_mutations():
    catalog = default_catalog()
    registry = CapabilityRegistry(catalog)
    registry.register("state.mutate", lambda args, context: ToolEnvelope("state.mutate", ResultStatus.COMPLETED))
    result = registry.execute(PlanTask("write", "state.mutate", {"operation": "add", "payload": {}}), _context())
    assert not result.ok
    assert "confirmation" in result.errors[0]


def test_market_performance_adapter_returns_window_and_benchmark_evidence():
    class Prices:
        def get_prices(self, ticker, start, end):
            base = 100.0
            slope = 1.0 if ticker == "AMD" else 0.2
            start_day = date(2026, 7, 1)
            return [SimpleNamespace(time=(start_day + timedelta(days=index)).isoformat(), close=base + slope * index, volume=1_000_000 + index * 10_000) for index in range(35)]

    catalog = default_catalog()
    registry = CapabilityRegistry(catalog)
    now = datetime(2026, 8, 4, 13, 0, tzinfo=ZoneInfo("America/New_York"))
    register_market_capabilities(registry, price_source_factory=Prices, move_provider=lambda ticker: None, now_factory=lambda: now)
    result = registry.execute(PlanTask("performance", "market.performance", {"ticker": "AMD"}), _context())
    assert result.ok
    assert result.metrics["returns"]["5d"] is not None
    assert result.metrics["relative_returns"]["SMH"]["5d"] is not None
    assert result.metrics["is_intraday"] is True
    assert {item.metadata["evidence_scope"] for item in result.evidence} >= {"price", "returns", "volume", "volatility", "benchmark"}
    answer = result.metadata["narrative"]
    assert "近 5 日回报" in answer
    assert "相对 SMH" in answer
    assert "盘中价格" in answer
    assert "不能据此判定是否放量或缩量" in answer
    assert verify_answer(answer, result.evidence, answer_mode=AnswerMode.TOOL_GROUNDED, results=[result]).ok


def test_market_move_adapter_splits_facts_and_causal_confidence():
    anomaly = SimpleNamespace(
        date="2026-09-08",
        price=508.71,
        price_change_pct=0.0652,
        volume_today=14_700_000,
        volume_avg_30d=22_900_000,
        volume_ratio=0.642,
        sector_etf="SMH",
        sector_return_1d=0.02,
        relative_1d_pp=0.0452,
        contrarian=False,
        reasons=[
            SimpleNamespace(text="公司发布直接利好", confidence="高", note="权威媒体同日报道"),
            SimpleNamespace(text="期权市场波动", confidence="低", note="缺少直接证据"),
        ],
        sources=[{"title": "Same-day report", "url": "https://example.test/report"}],
        next_steps=["观察 522 美元附近"],
        filtered_count=2,
    )
    catalog = default_catalog()
    registry = CapabilityRegistry(catalog)
    now = datetime(2026, 9, 8, 13, 0, tzinfo=ZoneInfo("America/New_York"))
    register_market_capabilities(registry, price_source_factory=lambda: None, move_provider=lambda ticker: anomaly, now_factory=lambda: now)
    result = registry.execute(PlanTask("move", "market.explain_move", {"ticker": "AMD"}), _context())
    scopes = [item.metadata["evidence_scope"] for item in result.evidence]
    assert scopes[:3] == ["price", "volume", "benchmark"]
    assert result.metrics["confirmed_driver_count"] == 1
    assert result.findings[0]["confirmed"] is True
    assert result.findings[1]["confirmed"] is False
    candidate_evidence = next(item for item in result.evidence if item.metadata.get("claim_role") == "candidate_driver")
    assert candidate_evidence.confidence == 0.3
    assert candidate_evidence.metadata["driver_text"] == "期权市场波动"
    assert next(item for item in result.evidence if item.metadata.get("claim_role") == "attribution_assessment")
    assert result.metrics["is_intraday"] is True
    assert "盘中累计成交量" in next(item.claim for item in result.evidence if item.metadata.get("evidence_scope") == "volume")
    answer = result.metadata["narrative"]
    assert verify_answer(answer, result.evidence, answer_mode=AnswerMode.RESEARCH_GROUNDED, results=[result]).ok
    # The sector sentence follows the sign of the relative return instead of always saying 跑赢.
    relative = result.metrics["relative_1d"]
    assert ("股价跑赢行业基准" in answer) == (relative > 0) and ("股价跑输行业基准" in answer) == (relative < 0)


def _performance_envelope():
    class Prices:
        def get_prices(self, ticker, start, end):
            slope = 1.0 if ticker == "AMD" else 0.2
            first = date(2026, 7, 1)
            return [SimpleNamespace(time=(first + timedelta(days=index)).isoformat(), close=100.0 + slope * index, volume=1_000_000 + index * 10_000) for index in range(35)]

    registry = CapabilityRegistry(default_catalog())
    now = datetime(2026, 8, 4, 18, 0, tzinfo=ZoneInfo("America/New_York"))
    register_market_capabilities(registry, price_source_factory=Prices, move_provider=lambda ticker: None, now_factory=lambda: now)
    return registry.execute(PlanTask("performance", "market.performance", {"ticker": "AMD"}), _context())


def test_market_synthesis_falls_back_to_the_adapter_narrative_when_repair_fails():
    result = _performance_envelope()
    volatility = next(item for item in result.evidence if item.metadata["evidence_scope"] == "volatility")
    llm = ScriptedLLM([LLMResponse(text=f"AMD 波动率为 99%。[{volatility.id}]"), LLMResponse(text=f"AMD 波动率为 98%。[{volatility.id}]")])
    request = normalize_request("AMD最近表现如何？")
    plan = ExecutionPlan("AMD最近表现如何？", RouteKind.FAST_LOOKUP, tasks=(PlanTask("p", "market.performance", {"ticker": "AMD"}),), answer_mode=AnswerMode.TOOL_GROUNDED)
    answer = LLMEvidenceSynthesizer(llm).synthesize(request, plan, [result], result.evidence)
    assert "99%" not in answer and "98%" not in answer
    assert "近 5 日回报" in answer
    assert answer == result.metadata["narrative"]
    assert len(llm.calls) == 2


def test_generic_synthesizer_prefers_adapter_narratives_and_skips_uncitable_evidence():
    market = _performance_envelope()
    other = ToolEnvelope("research.stock", ResultStatus.COMPLETED, subject="AMD", summary="AMD summary.", evidence=[EvidenceItem("E1", "AMD", "citable", metadata={}), EvidenceItem("E2", "AMD", "hidden", metadata={"citable": False})])
    answer = EvidenceSummarySynthesizer().synthesize(normalize_request("AMD"), ExecutionPlan("AMD", RouteKind.FAST_LOOKUP), [market, other], [*market.evidence, *other.evidence])
    assert answer.startswith(market.metadata["narrative"])
    assert "[E1]" in answer and "[E2]" not in answer


def test_executor_collects_structured_evidence():
    catalog = default_catalog()
    registry = CapabilityRegistry(catalog)

    def research(args, context):
        item = EvidenceItem("E1", args["ticker"], "supported claim")
        return ToolEnvelope("research.stock", ResultStatus.COMPLETED, subject=args["ticker"], summary=item.claim, evidence=[item])

    registry.register("research.stock", research)
    plan = ExecutionPlan("research", RouteKind.RESEARCH, (PlanTask("one", "research.stock", {"ticker": "NVDA"}),), BudgetClass.STANDARD)
    outcome = ExecutionEngine(registry).run(plan, _context())
    results, ledger = outcome.results, outcome.ledger
    assert results[0].ok
    assert ledger.get("E1").entity == "NVDA"


def test_orchestrator_runs_end_to_end_with_an_injected_capability():
    catalog = default_catalog()
    registry = CapabilityRegistry(catalog)

    def research(args, context):
        item = EvidenceItem("E-NVDA", "NVDA", "NVDA evidence")
        return ToolEnvelope("research.stock", ResultStatus.COMPLETED, subject="NVDA", summary="NVDA evidence", evidence=[item])

    registry.register("research.stock", research)
    result = AgentV2(catalog=catalog, registry=registry).run("分析 NVDA 的估值")
    assert result.status == RunStatus.COMPLETED
    assert result.verification.ok
    assert "[E-NVDA]" in result.answer


def test_command_waits_for_confirmation_without_dispatching():
    result = AgentV2().run("把 NVDA 加入关注列表")
    assert result.status == RunStatus.WAITING_CONFIRMATION
    assert not result.results


def test_research_adapter_preserves_engine_evidence():
    class FakeEngine:
        def run(self, ticker, modules=None):
            return {
                "ticker": ticker,
                "run_id": "research-1",
                "status": "COMPLETED",
                "generated_at": "2026-09-07T00:00:00Z",
                "core_thesis": "Evidence-backed thesis",
                "scores": {"valuation": 60},
                "risk_level": "Medium",
                "research_confidence": {"score": 80},
                "research_findings": [{"claim": "Revenue grew", "evidence_ids": ["ev-1"]}],
                "evidence_index": [{"id": "ev-1", "ticker": ticker, "module": "fundamental", "claim": "Revenue grew", "metrics": {"revenue_growth": 0.1}, "source_ids": ["fd_metrics"], "verified": True}],
                "sources": [{"id": "fd_metrics", "title": "Metrics", "url": "https://example.test", "published_at": "2026-09-01"}],
                "confidence_limitations": ["expectations missing forward estimates"],
                "production_diagnostics": {"modules": {"expectations": {"status": "PARTIAL", "completeness": 0.25, "missing_fields": ["forward_eps"]}}},
            }

    catalog = default_catalog()
    registry = CapabilityRegistry(catalog)
    register_research_capabilities(registry, engine_factory=FakeEngine)
    result = registry.execute(PlanTask("r", "research.stock", {"ticker": "NVDA", "focus": "fundamentals"}), _context())
    assert result.ok
    assert result.evidence[0].id == "ev-1"
    assert result.evidence[0].source_url == "https://example.test"
    metrics_evidence = next(item for item in result.evidence if item.metadata.get("citation_kind") == "metrics")
    assert metrics_evidence.id.startswith("evidence-research-metrics-")
    assert metrics_evidence.metadata["metrics"]["scores"]["valuation"] == 60
    limitation_evidence = next(item for item in result.evidence if item.metadata.get("citation_kind") == "limitations")
    assert "expectations 数据完整度 25.0%" in limitation_evidence.claim
    assert limitation_evidence.metadata["module_diagnostics"]["expectations"]["completeness"] == 0.25


def test_research_compare_keeps_the_tickers_that_worked_when_one_engine_fails(caplog):
    class FlakyEngine:
        def run(self, ticker, modules=None):
            if ticker == "SNDK":
                raise KeyError("no price history for SNDK")
            return {"ticker": ticker, "run_id": f"run-{ticker}", "status": "COMPLETED", "core_thesis": f"{ticker} 估值合理", "evidence_index": [{"id": f"ev-{ticker}", "ticker": ticker, "module": "valuation", "claim": f"{ticker} TTM 市盈率 20 倍", "source_ids": ["fd_metrics"], "verified": True}], "sources": [{"id": "fd_metrics", "title": "Metrics", "url": "https://example.test"}]}

    registry = CapabilityRegistry(default_catalog())
    register_research_capabilities(registry, engine_factory=FlakyEngine)
    with caplog.at_level("WARNING"):
        result = registry.execute(PlanTask("c", "research.compare", {"tickers": ["MU", "SNDK"], "dimensions": ["valuation"]}), _context())
    # The comparison is partial, not lost: MU's evidence stays, SNDK's failure is a limitation and an error, and it is logged.
    assert result.status == ResultStatus.PARTIAL_ERROR and [item.id for item in result.evidence] == ["ev-MU"]
    assert result.limitations == ["SNDK 的研究未完成（KeyError），比较只覆盖其余股票"] and result.errors == ["SNDK: KeyError: 'no price history for SNDK'"]
    assert result.metadata["failed_tickers"] == ["SNDK"] and "research.compare: SNDK failed: KeyError" in caplog.text

    class DownEngine:
        def run(self, ticker, modules=None):
            raise RuntimeError("provider down")

    registry = CapabilityRegistry(default_catalog())
    register_research_capabilities(registry, engine_factory=DownEngine)
    result = registry.execute(PlanTask("c", "research.compare", {"tickers": ["MU", "SNDK"]}), _context())
    assert result.status == ResultStatus.FAILED and not result.evidence and len(result.errors) == 2

    # A capability that raises is still turned into a failed envelope, and now leaves a trace in the log.
    registry.register("research.stock", lambda arguments, context: (_ for _ in ()).throw(ValueError("boom")))
    with caplog.at_level("WARNING"):
        failed = registry.execute(PlanTask("s", "research.stock", {"ticker": "MU"}), _context())
    assert failed.status == ResultStatus.FAILED and "capability research.stock failed: ValueError: boom" in caplog.text


def test_research_adapter_disambiguates_conflicting_ids_from_cached_results():
    class CachedEngine:
        def run(self, ticker, modules=None):
            return {
                "ticker": ticker,
                "run_id": "cached-research",
                "status": "COMPLETED",
                "generated_at": "2026-09-08T00:00:00Z",
                "core_thesis": "Cached thesis",
                "research_findings": [],
                "evidence_index": [
                    {"id": "legacy-id", "ticker": ticker, "module": "sec", "claim": "First filing excerpt.", "source_ids": ["sec_filings"]},
                    {"id": "legacy-id", "ticker": ticker, "module": "sec", "claim": "Second filing excerpt.", "source_ids": ["sec_filings"]},
                ],
                "sources": [{"id": "sec_filings", "title": "SEC filing", "url": "https://example.test/filing"}],
                "production_diagnostics": {"modules": {}},
            }

    catalog = default_catalog()
    registry = CapabilityRegistry(catalog)
    register_research_capabilities(registry, engine_factory=CachedEngine)
    plan = ExecutionPlan("research", RouteKind.RESEARCH, (PlanTask("one", "research.stock", {"ticker": "NVDA"}),), BudgetClass.STANDARD)
    outcome = ExecutionEngine(registry).run(plan, _context())
    results, ledger = outcome.results, outcome.ledger
    assert results[0].ok
    assert len(ledger.items()) == 2
    assert len(ledger.ids()) == 2
    repaired = next(item for item in ledger.items() if item.id != "legacy-id")
    assert repaired.metadata["original_evidence_id"] == "legacy-id"
    assert repaired.metadata["collision_disambiguated"] is True


def test_evidence_conflict_is_not_reported_as_a_valid_plan_or_verification():
    catalog = default_catalog()
    registry = CapabilityRegistry(catalog)

    def conflicting_research(args, context):
        return ToolEnvelope(
            "research.stock",
            ResultStatus.COMPLETED,
            subject=args["ticker"],
            evidence=[
                EvidenceItem("same-id", args["ticker"], "First claim"),
                EvidenceItem("same-id", args["ticker"], "Second claim"),
            ],
        )

    registry.register("research.stock", conflicting_research)
    result = AgentV2(catalog=catalog, registry=registry).run("分析 NVDA")
    # A different claim under a reused id is reissued, disclosed, and kept citeable; the run completes.
    assert result.status == RunStatus.COMPLETED, (result.status, result.error)
    ids = {item.id for item in result.evidence}
    assert "same-id" in ids and any(value.startswith("same-id~") for value in ids)
    assert {item.claim for item in result.evidence} == {"First claim", "Second claim"}
    assert any("重新编号" in value for value in result.results[0].limitations)
    assert result.verification.ok


def test_web_facade_returns_transport_neutral_dict():
    payload = WebFacade(AgentV2()).handle(WebRequest("什么是市盈率？", session_id="web-1"))
    assert payload["request"]["session_id"] == "web-1"
    assert payload["route"]["kind"] == "general_knowledge"


def test_telegram_facade_depends_only_on_a_transport_protocol():
    class Transport:
        def __init__(self):
            self.events = []

        async def typing(self, chat_id):
            self.events.append("typing")

        async def progress(self, chat_id, event):
            self.events.append(event.status.value)

        async def deliver(self, chat_id, result):
            self.events.append("delivered")

    async def scenario():
        transport = Transport()
        result = await TelegramFacade(AgentV2()).handle(TelegramMessage(7, "什么是 ROE？"), transport)
        return transport, result

    transport, result = asyncio.run(scenario())
    assert transport.events[0] == "typing"
    assert transport.events[-1] == "delivered"
    assert result.request.session_id == "7"


def test_llm_planner_accepts_only_declared_capabilities():
    response = LLMResponse(text='{"objective":"研究估值","tasks":[{"id":"t1","capability":"research.stock","arguments":{"ticker":"NVDA","focus":"valuation"}}]}')
    catalog = default_catalog()
    planner = StructuredLLMPlanner(ScriptedLLM([response]), catalog)
    request = normalize_request("分析 NVDA 的估值")
    plan = planner.plan(request, route(request))
    assert plan.tasks[0].capability == "research.stock"
    assert plan.tasks[0].arguments["focus"] == "valuation"


def test_llm_planner_falls_back_when_model_invents_a_capability():
    response = LLMResponse(text='{"tasks":[{"id":"t1","capability":"trade.execute","arguments":{}}]}')
    catalog = default_catalog()
    planner = StructuredLLMPlanner(ScriptedLLM([response]), catalog)
    request = normalize_request("分析 NVDA 的风险")
    plan = planner.plan(request, route(request))
    assert plan.tasks[0].capability == "research.stock"
    assert any("fallback" in value for value in plan.assumptions)


def test_llm_planner_preserves_explicit_lab_parameters():
    response = LLMResponse(text='{"tasks":[{"id":"lab","capability":"lab.backtest","arguments":{"strategy":"momentum","tickers":["NVDA"],"holding_days":21,"cost_bps":10}}]}')
    catalog = default_catalog()
    planner = StructuredLLMPlanner(ScriptedLLM([response]), catalog)
    request = normalize_request("回测 NVDA 动量策略，持有21天，成本10bp")
    plan = planner.plan(request, route(request))
    assert plan.tasks[0].capability == "lab.backtest"
    assert plan.tasks[0].arguments["holding_days"] == 21
    assert plan.tasks[0].arguments["cost_bps"] == 10
    assert plan.budget == BudgetClass.LAB


def test_llm_synthesizer_requires_evidence_ids_in_its_prompt_contract():
    llm = ScriptedLLM([LLMResponse(text="结论有证据支持。[E1]")])
    synthesizer = LLMEvidenceSynthesizer(llm)
    request = normalize_request("分析 NVDA")
    plan = ExecutionPlan("分析 NVDA", RouteKind.RESEARCH)
    evidence = [EvidenceItem("E1", "NVDA", "支持结论")]
    answer = synthesizer.synthesize(request, plan, [], evidence)
    assert answer.endswith("[E1]")
    system = llm.calls[0][0]["content"]
    payload = json.loads(llm.calls[0][1]["content"])
    assert "3—5 个短段落" in system
    assert payload["response_style"] == "brief"
    assert payload["response_intent"] == "stock_research"


def test_llm_synthesizer_only_requests_detailed_style_when_user_asks_for_it():
    llm = ScriptedLLM([LLMResponse(text="详细结论。[E1]")])
    synthesizer = LLMEvidenceSynthesizer(llm)
    request = normalize_request("给我一份 NVDA 的完整详细报告")
    plan = ExecutionPlan("详细分析 NVDA", RouteKind.RESEARCH)
    evidence = [EvidenceItem("E1", "NVDA", "支持结论")]
    synthesizer.synthesize(request, plan, [], evidence)
    payload = json.loads(llm.calls[0][1]["content"])
    assert payload["response_style"] == "detailed"


@pytest.mark.parametrize(
    ("capability", "intent", "guidance"),
    [("market.performance", "recent_performance", "recent_performance："), ("market.explain_move", "move_explanation", "move_explanation："), ("research.stock", "stock_research", "stock_research：")],
)
def test_llm_synthesizer_derives_intent_and_guidance_from_the_capabilities_used(capability, intent, guidance):
    llm = ScriptedLLM([LLMResponse(text="有证据的回答。[E1]")])
    synthesizer = LLMEvidenceSynthesizer(llm)
    request = normalize_request("AMD 怎么样")
    evidence = [EvidenceItem("E1", "AMD", "支持结论")]
    plan = ExecutionPlan("AMD 怎么样", RouteKind.RESEARCH, tasks=(PlanTask("t", capability, {"ticker": "AMD"}),))
    synthesizer.synthesize(request, plan, [], evidence)
    payload = json.loads(llm.calls[0][1]["content"])
    assert payload["response_intent"] == intent
    system = llm.calls[0][0]["content"]
    assert guidance in system
    assert "盘中" not in system or capability.startswith("market.")


def test_llm_synthesizer_payload_carries_only_the_fields_the_draft_uses():
    llm = ScriptedLLM([LLMResponse(text="有证据的回答。[E1]")])
    synthesizer = LLMEvidenceSynthesizer(llm)
    request = normalize_request("AMD 怎么样")
    evidence = [
        EvidenceItem("E1", "AMD", "营收增长 55.3%", metric="revenue_growth", value=0.553, unit="%", period="FY2026", as_of="2026-07-29", source_id="fd_metrics", source_title="AMD 财报", source_url="https://example.com/amd", confidence=0.9, producer_run_id="run-1", metadata={"snapshot": "a", "citable": True}),
        EvidenceItem("E2", "AMD", "评分 96/100", producer_run_id="run-1", metadata={"citation_kind": "metrics"}),
    ]
    envelope = ToolEnvelope("research.stock", ResultStatus.COMPLETED, subject="AMD", summary="研究摘要", evidence=evidence, run_id="run-1")
    plan = ExecutionPlan("AMD 怎么样", RouteKind.RESEARCH, tasks=(PlanTask("t", "research.stock", {"ticker": "AMD"}),))
    synthesizer.synthesize(request, plan, [envelope], evidence)
    payload = json.loads(llm.calls[0][1]["content"])
    # Titles, URLs, run ids and metadata are for the surfaces and the verifier; the draft cites by id. Empty result fields are left out.
    assert payload["evidence"][0] == {"id": "E1", "entity": "AMD", "claim": "营收增长 55.3%", "metric": "revenue_growth", "value": 0.553, "unit": "%", "period": "FY2026", "as_of": "2026-07-29", "source_id": "fd_metrics", "confidence": 0.9}
    assert payload["evidence"][1] == {"id": "E2", "entity": "AMD", "claim": "评分 96/100", "citation_kind": "metrics"}
    assert payload["results"][0] == {"capability": "research.stock", "status": "completed", "subject": "AMD", "as_of": "", "summary": "研究摘要"}
    assert "source_url" not in json.dumps(payload) and "producer_run_id" not in json.dumps(payload)


def test_llm_synthesizer_normalizes_valid_result_paths_to_evidence_ids():
    llm = ScriptedLLM(
        [
            LLMResponse(
                text=(
                    "基本面评分为 96/100。[results.metrics.scores.fundamental] "
                    "预期数据不足。[results.limitations]"
                )
            )
        ]
    )
    synthesizer = LLMEvidenceSynthesizer(llm)
    request = normalize_request("分析 NVDA")
    plan = ExecutionPlan("分析 NVDA", RouteKind.RESEARCH)
    result = ToolEnvelope(
        "research.stock",
        ResultStatus.COMPLETED,
        metrics={"scores": {"fundamental": 96}},
        limitations=["expectations: PARTIAL_DATA"],
        run_id="research-1",
    )
    evidence = [
        EvidenceItem(
            "evidence-research-metrics-1",
            "NVDA",
            "NVDA fundamental score is 96/100.",
            producer_run_id="research-1",
            metadata={"citation_kind": "metrics"},
        ),
        EvidenceItem(
            "evidence-research-limitations-1",
            "NVDA",
            "NVDA expectations data is incomplete.",
            producer_run_id="research-1",
            metadata={"citation_kind": "limitations"},
        ),
    ]
    answer = synthesizer.synthesize(request, plan, [result], evidence)
    assert "[results." not in answer
    assert "[evidence-research-metrics-1]" in answer
    assert "[evidence-research-limitations-1]" in answer


def test_llm_synthesizer_repairs_an_invalid_result_path_instead_of_shipping_it():
    llm = ScriptedLLM(
        [
            LLMResponse(text="虚构评分为 96。[results.metrics.scores.invented]"),
            LLMResponse(text="基本面评分为 96。[evidence-research-metrics-1]"),
        ]
    )
    synthesizer = LLMEvidenceSynthesizer(llm)
    request = normalize_request("分析 NVDA")
    plan = ExecutionPlan("分析 NVDA", RouteKind.RESEARCH, answer_mode=AnswerMode.RESEARCH_GROUNDED)
    result = ToolEnvelope(
        "research.stock",
        ResultStatus.COMPLETED,
        metrics={"scores": {"fundamental": 96}},
        run_id="research-1",
    )
    evidence = [
        EvidenceItem(
            "evidence-research-metrics-1",
            "NVDA",
            "NVDA fundamental score is 96/100.",
            producer_run_id="research-1",
            metadata={"citation_kind": "metrics"},
        )
    ]
    answer = synthesizer.synthesize(request, plan, [result], evidence)
    assert answer == "基本面评分为 96。[evidence-research-metrics-1]"
    assert len(llm.calls) == 2
    assert "results.metrics.scores.invented" in llm.calls[1][3]["content"]
    assert verify_answer(answer, evidence, answer_mode=AnswerMode.RESEARCH_GROUNDED, results=[result]).ok


def test_verifier_rejects_an_invented_number_even_with_a_valid_citation():
    evidence = [EvidenceItem("E1", "NVDA", "Revenue growth was 10%.")]
    report = verify_answer(
        "NVDA 收入增长 20%。[E1]",
        evidence,
        answer_mode=AnswerMode.RESEARCH_GROUNDED,
    )
    assert not report.ok
    assert "20" in report.ungrounded_numbers


def test_verifier_grounds_result_metrics_and_scaled_evidence_values():
    evidence = [
        EvidenceItem("evidence-999999", "NVDA", "Insider net transaction value was -349029376."),
        EvidenceItem("evidence-metrics", "NVDA", "Score 87/100 and completeness 0.857."),
    ]
    result = ToolEnvelope(
        "research.stock",
        ResultStatus.COMPLETED,
        metrics={"score": 87, "max_score": 100, "completeness": 0.857},
        findings=[{"claim": "Insiders were net sellers", "evidence_ids": ["evidence-999999"]}],
        evidence=evidence,
    )
    report = verify_answer(
        "综合评分 87/100，数据完整性 85.7%。[evidence-metrics] 内部人净卖出约 -3.49 亿美元。[evidence-999999]",
        evidence,
        answer_mode=AnswerMode.RESEARCH_GROUNDED,
        results=[result],
    )
    assert report.ok
    assert not report.ungrounded_numbers


def test_verifier_requires_nearby_citation_to_support_nearby_number():
    evidence = [
        EvidenceItem("E-REVENUE", "NVDA", "Revenue growth was 10%."),
        EvidenceItem("E-MARGIN", "NVDA", "Gross margin was 20%."),
    ]
    report = verify_answer(
        "NVDA 收入增长 20%。[E-REVENUE]",
        evidence,
        answer_mode=AnswerMode.RESEARCH_GROUNDED,
    )
    assert not report.ok
    assert any("邻近数字" in warning for warning in report.warnings)


def test_verifier_enforces_evidence_declared_forbid_unless_rules():
    rule = {"forbid": r"主要原因", "unless": r"可能", "warning": "候选归因被表述为已确认原因"}
    evidence = [EvidenceItem("C1", "AMD", "低置信度候选解释：期权市场波动。", metadata={"constraints": [rule]})]
    rejected = verify_answer("AMD 上涨的主要原因是期权市场波动。[C1]", evidence, answer_mode=AnswerMode.RESEARCH_GROUNDED)
    assert not rejected.ok
    assert any("候选归因" in warning for warning in rejected.warnings)
    hedged = verify_answer("AMD 上涨的主要原因可能是期权市场波动。[C1]", evidence, answer_mode=AnswerMode.RESEARCH_GROUNDED)
    assert hedged.ok


def test_verifier_enforces_evidence_declared_require_rules():
    rule = {"require": r"盘中|截至查询时", "warning": "盘中价格被表述为完整收盘口径"}
    evidence = [EvidenceItem("P1", "AMD", "Intraday price.", metadata={"constraints": [rule]})]
    assert not verify_answer("AMD 收盘价走强。[P1]", evidence, answer_mode=AnswerMode.TOOL_GROUNDED).ok
    assert verify_answer("AMD 盘中价格走强。[P1]", evidence, answer_mode=AnswerMode.TOOL_GROUNDED).ok


def test_verifier_enforces_result_level_caps_and_forbidden_phrases():
    evidence = [
        EvidenceItem("C1", "AMD", "Candidate one.", metadata={"claim_role": "candidate_driver"}),
        EvidenceItem("C2", "AMD", "Candidate two.", metadata={"claim_role": "candidate_driver"}),
    ]
    result = ToolEnvelope(
        "market.explain_move",
        ResultStatus.COMPLETED,
        subject="AMD",
        evidence=evidence,
        metadata={"answer_constraints": [{"max_cited": {"metadata": {"claim_role": "candidate_driver"}, "max": 1, "warning": "未确认直接驱动时展示了过多弱候选线索"}}, {"forbid": r"0\s*个", "warning": "将内部归因计数直接暴露给用户"}]},
    )
    report = verify_answer("可能与线索一相关。[C1] 也可能与线索二相关。[C2]", evidence, answer_mode=AnswerMode.RESEARCH_GROUNDED, results=[result])
    assert any("过多弱候选" in warning for warning in report.warnings)
    report = verify_answer("有 0 个驱动。[C1]", evidence, answer_mode=AnswerMode.RESEARCH_GROUNDED, results=[result])
    assert any("归因计数" in warning for warning in report.warnings)
    assert verify_answer("可能与线索一相关。[C1]", evidence, answer_mode=AnswerMode.RESEARCH_GROUNDED, results=[result]).ok


def test_verifier_rejects_citations_of_uncitable_evidence_and_uncited_market_figures():
    evidence = [EvidenceItem("C1", "AMD", "Candidate one.", metadata={"citable": False, "uncitable_warning": "展示了缺乏直接支持的过弱异动线索"}), EvidenceItem("P1", "AMD", "AMD close 100.00.")]
    result = ToolEnvelope("market.explain_move", ResultStatus.COMPLETED, subject="AMD", evidence=evidence, metadata={"require_cited_numbers": True})
    report = verify_answer("可能与该线索相关。[C1]", evidence, answer_mode=AnswerMode.RESEARCH_GROUNDED, results=[result])
    assert any("过弱异动线索" in warning for warning in report.warnings)
    report = verify_answer("AMD 收于 100.00 美元。 详情见证据。[P1]", evidence, answer_mode=AnswerMode.RESEARCH_GROUNDED, results=[result])
    assert any("邻近引用" in warning for warning in report.warnings)


def test_verifier_does_not_treat_digits_in_opaque_ids_as_observations():
    evidence = [EvidenceItem("evidence-999999", "NVDA", "Insiders were net sellers.")]
    result = ToolEnvelope(
        "research.stock",
        ResultStatus.COMPLETED,
        findings=[{"claim": "Insiders were net sellers", "evidence_ids": ["evidence-999999"]}],
        evidence=evidence,
    )
    report = verify_answer(
        "NVDA 的指标值是 999999。[evidence-999999]",
        evidence,
        answer_mode=AnswerMode.RESEARCH_GROUNDED,
        results=[result],
    )
    assert not report.ok
    assert "999999" in report.ungrounded_numbers


def test_web_and_lab_ports_can_be_injected_without_core_dependencies():
    class Search:
        def search(self, query, **kwargs):
            return ToolEnvelope(
                "web.research",
                ResultStatus.COMPLETED,
                summary="web result",
                evidence=[EvidenceItem("WEB1", "", "web result")],
            )

    class Lab:
        def run(self, capability, arguments, context):
            return ToolEnvelope(capability, ResultStatus.COMPLETED, summary="lab result")

    catalog = default_catalog()
    registry = CapabilityRegistry(catalog)
    register_web_capability(registry, Search())
    register_lab_capabilities(registry, Lab())
    web_context = ExecutionContext(
        "web",
        NormalizedRequest("q", "q", allow_web=True),
        BudgetClass.STANDARD,
        allow_web=True,
    )
    web = registry.execute(
        PlanTask("w", "web.research", {"query": "q", "topic": "company_event"}),
        web_context,
    )
    lab = registry.execute(PlanTask("l", "lab.backtest", {"strategy": "momentum"}), _context())
    assert web.ok and web.evidence[0].id == "WEB1"
    assert lab.ok


def test_short_term_session_resolves_a_follow_up_before_routing():
    catalog = default_catalog()
    registry = CapabilityRegistry(catalog)

    def research(args, context):
        ticker = args["ticker"]
        item = EvidenceItem(f"E-{ticker}", ticker, f"{ticker} evidence")
        return ToolEnvelope("research.stock", ResultStatus.COMPLETED, subject=ticker, summary=item.claim, evidence=[item])

    registry.register("research.stock", research)
    memory = ShortTermSession()
    agent = AgentV2(catalog=catalog, registry=registry, session=memory)
    first = agent.run("分析 NVDA 的风险", session_id="chat-1")
    second = agent.run("那它财报呢？", session_id="chat-1")
    assert first.status == RunStatus.COMPLETED
    assert second.request.entities == ("NVDA",)
    assert second.request.metadata["rewritten"]
    assert second.to_dict()["request"]["metadata"]["antecedent"] == "NVDA"


def test_short_term_session_carries_the_stock_an_answer_named_into_a_subjectless_follow_up():
    memory = ShortTermSession()
    agent = AgentV2(catalog=default_catalog(), registry=_framed_registry(), session=memory)
    first = agent.run("我的仓库里哪只跌的最多?", session_id="chat-2")
    assert first.request.entities == () and first.answer.startswith("按买入以来的浮动盈亏排序，最低的是 ARM")
    third = agent.run("它财报怎么样", session_id="chat-2")  # the pronoun path now finds the focus too
    assert third.request.text == "ARM财报怎么样" and third.results[0].subject == "ARM"
    assert third.request.metadata["context_frame"]["ticker"] == "ARM"
    # Questions that name their own scope or stock are left alone.
    for query in ("宏观怎么样", "我的持仓风险怎么样", "NVDA 为什么跌", "把 HPE 加到关注列表"):
        assert not memory.resolve("chat-2", query).rewritten, query
    assert not memory.resolve("chat-fresh", "什么原因跌这么多?").rewritten  # nothing to refer back to


def test_a_why_follow_up_after_a_loss_ranking_explains_the_loss_since_purchase_not_today():
    memory = ShortTermSession()
    agent = AgentV2(catalog=default_catalog(), registry=_framed_registry(), session=memory)
    agent.run("我的仓库里哪只跌的最多?", session_id="chat-3")
    resolution = memory.resolve("chat-3", "什么原因跌这么多?")
    assert resolution.frame["ticker"] == "ARM" and resolution.frame["field"] == "pl_pct" and resolution.frame["value"] == -32.22
    second = agent.run("什么原因跌这么多?", session_id="chat-3")
    assert second.request.text == "ARM 什么原因跌这么多?"
    assert [task.capability for task in second.plan.tasks] == ["account.portfolio", "market.performance", "market.drawdown", "filings.recent", "market.anomaly_history", "market.attribute_move"]
    assert second.plan.tasks[2].arguments == {"ticker": "ARM", "loss_pct": -32.22, "top": 3}
    attributor = second.plan.tasks[5]
    assert attributor.fan_out == {"from": "market-stretch", "field": "worst_dates", "argument": "date", "max": 3} and not attributor.required
    assert [result.subject for result in second.results if result.capability == "market.attribute_move"] == ["ARM"]
    assert "最相关的一条候选线索是“财报指引低于预期”，只能作为排查方向[AT-ARM-2026-08-05-filing]。" in second.answer
    assert second.plan.assumptions[0].startswith("context_frame: 用户追问的是 ARM 买入以来的浮动盈亏 -32.22%（成本价 $389.52）")
    lines = second.answer.split("\n")
    assert lines[0].startswith("你问的是 ARM 买入以来的浮动盈亏：-32.22%，成本价 $389.52[legacy-")
    assert lines[1] == "今日盘中为上涨（+0.94%），与买入以来的浮动盈亏是不同区间[P-ARM]。"
    assert lines[2] == "对照区间回报（近 5 日 +12.32%、近 1 月 -1.53%、近 3 月 -21.40%、近 1 年 +89.85%），这段跌幅大部分落在近 3 月内[W-ARM]。"
    assert lines[3] == "近 1 年 +89.85% 而该持仓仍在浮亏，说明买入点在这轮上涨之后的高位[W-ARM]。"
    assert "组合价值" not in second.answer and "近 1 月回报 -1.53%" not in second.answer  # the narrative's returns line is not repeated
    assert "相对 SMH，单日超额 +0.94%[B-ARM]。" in second.answer and "成交量尚未定型" not in second.answer
    assert "ARM 期间跌幅最大的交易日：2026-08-05 -13.21%[D-ARM-0805]。" in second.answer
    # The window return was already in the lead; the stretch block does not repeat it.
    assert "区间回报 -21.40%" not in second.answer.split("\n\n", 1)[1]
    assert "2026-03-01" not in second.answer  # a filing before the decline window is left out
    # The 08-05 watch record is superseded by that day's attribution block; the
    # filings are listed on one line because the attributor read them.
    assert "盯盘记录" not in second.answer and "ARM 这段下跌期间 1 份申报：2026-08-05 8-K[F-ARM-0805]。" in second.answer
    # Reading order is fixed: the stretch, then the filings, then the day blocks.
    positions = [second.answer.index(marker) for marker in ("从 2026-06-18 的高点", "这段下跌期间 1 份申报", "ARM 在 2026-08-05 收于")]
    assert positions == sorted(positions)
    from v2.agent_v2.synthesis import anomaly_lines, filing_line

    history = ToolEnvelope("market.anomaly_history", ResultStatus.COMPLETED, subject="ARM", evidence=[
        EvidenceItem(f"A-{day}", "ARM", f"ARM {day} 盯盘记录：{flags}；note。", metadata={"evidence_scope": "anomaly", "date": day, "flags": flags})
        for day, flags in (("2026-09-09", ""), ("2026-07-24", "retro_attribution"), ("2026-07-10", "volume_spike"), ("2026-06-23", "gap_down"), ("2026-06-01", "gap_down"))
    ])
    # Inside peak→trough only, minus attributed days and retro memories: 07-10 survives.
    assert anomaly_lines(history, "2026-06-18", "2026-07-29", frozenset({"2026-06-23"})) == "- ARM 2026-07-10 盯盘记录：volume_spike；note。 [A-2026-07-10]"
    filings = ToolEnvelope("filings.recent", ResultStatus.COMPLETED, subject="ARM", evidence=[
        EvidenceItem(f"F-{day}", "ARM", f"ARM 于 {day} 向 SEC 提交了 6-K（x）。", metadata={"evidence_scope": "filing", "date": day, "form": "6-K"})
        for day in ("2026-08-10", "2026-08-01", "2026-07-29")
    ])
    # Three days after the trough is the attributor's own reading margin; 08-10 is out.
    assert filing_line(filings, "2026-06-18", "2026-08-01") == "ARM 这段下跌期间 2 份申报：2026-08-01 6-K[F-2026-08-01]；2026-07-29 6-K[F-2026-07-29]。"
    assert "AT-ARM-2026-08-05-news" not in second.answer  # no web consent: the attributor had no news to cite
    # With web consent the same plan lets the attributor use the news.
    consenting = AgentV2(catalog=default_catalog(), registry=_framed_registry(), session=memory, config=AgentV2Config(enable_web_fallback=True))
    consenting.run("我的仓库里哪只跌的最多?", session_id="chat-4")
    with_web = consenting.run("为什么跌这么多?", session_id="chat-4", allow_web=True)
    assert [task.capability for task in with_web.plan.tasks][-1] == "market.attribute_move"
    assert "能直接支持的高置信度驱动：财报后指引令市场失望，股价大跌[AT-ARM-2026-08-05-news]。" in with_web.answer
    assert with_web.verification.ok
    assert second.verification.ok, second.verification
    assert second.status == RunStatus.COMPLETED
    # The mirror image: the best gainer, then "why did it rise so much" → the run-up chain.
    rising = AgentV2(catalog=default_catalog(), registry=_framed_registry(), session=memory)
    top = rising.run("我的仓库里哪只涨的最多?", session_id="chat-5")
    assert top.answer.startswith("按买入以来的浮动盈亏排序，最高的是 IVV（+2.03%）")
    why_up = rising.run("为什么涨这么多?", session_id="chat-5")
    assert [task.capability for task in why_up.plan.tasks] == ["account.portfolio", "market.performance", "market.runup", "filings.recent", "market.anomaly_history", "market.attribute_move"]
    assert why_up.plan.tasks[2].arguments == {"ticker": "IVV", "gain_pct": 2.03, "top": 3} and why_up.plan.tasks[-1].fan_out["field"] == "best_dates"
    assert why_up.plan.assumptions[0].startswith("context_frame: 用户追问的是 IVV 买入以来的浮动盈亏 +2.03%（成本价 $755.41）")
    assert why_up.answer.startswith("你问的是 IVV 买入以来的浮动盈亏：+2.03%，成本价 $755.41[legacy-")
    assert "这段涨幅大部分落在" in why_up.answer and "IVV 期间涨幅最大的交易日：2026-05-12 +3.50%[U-IVV-0512]。" in why_up.answer
    assert "同期行业基准 SPY 从 2026-04-07 到 2026-07-10 回报 +6.00%，IVV 比基准多涨 2.57 个百分点[U-IVV-span]。" in why_up.answer
    assert "“为什么上涨”目前还不能下定论" in why_up.answer and "为什么下跌" not in why_up.answer and "回撤" not in why_up.answer
    assert why_up.verification.ok and why_up.status == RunStatus.COMPLETED
    # Direct entry with run-up wording and no earlier turn.
    direct_up = RulePlanner().plan(normalize_request("NVDA 买入以来为什么涨了这么多"), route(normalize_request("NVDA 买入以来为什么涨了这么多")))
    assert direct_up.frame == {"kind": "runup", "ticker": "NVDA", "label": "这段涨幅", "window": ""} and direct_up.tasks[2].capability == "market.runup"
    # The frame survives the framed turn, and a question about a rise is not a drawdown question.
    assert memory.resolve("chat-3", "为什么涨").frame["ticker"] == "ARM"
    plan = RulePlanner().plan(normalize_request("ARM 为什么涨", metadata={"context_frame": resolution.frame}), route(normalize_request("ARM 为什么涨")))
    assert [task.capability for task in plan.tasks] == ["market.explain_move"]
    # The LLM planner leaves the framed plan to the rules.
    llm = ScriptedLLM([LLMResponse(text="{}")])
    framed = normalize_request("ARM 什么原因跌这么多?", metadata={"context_frame": resolution.frame})
    assert len(StructuredLLMPlanner(llm, default_catalog()).plan(framed, route(framed)).tasks) == 6 and llm.calls == []


_FILING_TEXT = """UNITED STATES SECURITIES AND EXCHANGE COMMISSION
FORM 8-K
Item 2.02 Results of Operations and Financial Condition
On July 29, 2026, Arm Holdings plc announced results for the quarter. Revenue of $1.05 billion was below the guidance range; the company now expects fiscal-year revenue growth in the low twenties.
Item 5.02 Departure of Directors or Certain Officers
On July 28, 2026, the Chief Financial Officer notified the board of his intention to resign effective September 1, 2026.
Item 9.01 Financial Statements and Exhibits
Exhibit 99.1 Press release dated July 29, 2026.
"""


class _FakeFilingSource:
    def __init__(self, refs):
        self.refs = refs
        self.reads: list[tuple[str, str]] = []

    def list_filings(self, ticker, since, until):
        return [ref for ref in self.refs if since <= ref.filing_date <= until]

    def outline(self, ref):
        from v2.agent_v2.agents.filing_reader import Section, sections_of

        return [Section(section_id, title, len(body)) for section_id, title, body in sections_of(_FILING_TEXT, ref.form)]

    def read(self, ref, section_id):
        from v2.agent_v2.agents.filing_reader import sections_of

        self.reads.append((ref.accession, section_id))
        return next((body for candidate, _, body in sections_of(_FILING_TEXT, ref.form) if candidate == section_id), "")


def test_edgar_source_appends_a_6k_exhibit_so_the_reader_can_choose_it():
    from v2.agent_v2.agents.filing_reader import EdgarFilingSource

    class Attachment:
        def __init__(self, kind, description, body):
            self.document_type, self.description, self._body = kind, description, body

        def text(self):
            return self._body

    class Raw:
        accession_number = "0001-26-000900"
        filing_date = "2026-07-29"
        form = "6-K"
        cik = "0001973239"
        homepage_url = "https://www.sec.gov/x/900/"
        attachments = [
            Attachment("6-K", "cover", "FORM 6-K Report of foreign private issuer"),
            Attachment("EX-99.1", "Press release", "<html><body><p>Arm Holdings plc reports results for the first quarter. Revenue was $1.05 billion, below the guidance range of $1.10 to $1.20 billion.</p></body></html>"),
            Attachment("EX-99.2", "Shareholder letter", "Second   exhibit   text."),
        ]

        def text(self):
            return "FORM 6-K\nReport of foreign private issuer pursuant to Rule 13a-16.\nArm Holdings plc furnishes the exhibits listed herein."

    source = EdgarFilingSource(fetch=lambda ticker, form, since, until: [Raw()] if form == "6-K" else [])
    refs = source.list_filings("ARM", "2026-07-15", "2026-08-01")
    assert [ref.form for ref in refs] == ["6-K"] and refs[0].url == "https://www.sec.gov/x/900/"
    outline = source.outline(refs[0])
    assert [section.title for section in outline][1:] == ["EXHIBIT 99.1 Press release", "EXHIBIT 99.2 Shareholder letter"]
    body = source.read(refs[0], outline[1].id)
    assert "Revenue was $1.05 billion" in body and "<p>" not in body
    assert source.read(refs[0], outline[2].id).endswith("Second exhibit text.")


def test_filing_reader_reads_the_sections_it_chooses_and_keeps_only_quoted_events():
    from v2.agent_v2.agents.filing_reader import FilingReader, FilingRef, sections_of

    parts = sections_of(_FILING_TEXT, "8-K")
    assert [part[0] for part in parts] == ["s0", "s1", "s2", "s3"] and parts[1][1].startswith("Item 2.02")
    assert parts[0][1] == "UNITED STATES SECURITIES AND EXCHANGE COMMISSION"  # the cover page stays readable
    assert [part[0] for part in sections_of("x" * 8000, "6-K")] == ["part-1", "part-2", "part-3"]
    # A long exhibit whose table headers all look like headings collapses to at most twenty sections.
    noisy = "\n".join(f"REVENUE BY SEGMENT TABLE {index}\n" + ("row of figures " * 20) for index in range(150))
    merged = sections_of(noisy, "6-K")
    assert 15 <= len(merged) <= 20 and merged[0][1].startswith("REVENUE BY SEGMENT TABLE 0 …（含后续")
    assert sum(len(body) for _, _, body in merged) >= len(noisy) - 150 * 2  # nothing is dropped, only joined
    paged = sections_of("y" * 100_000, "6-K")
    assert len(paged) == 20 and sum(len(body) for _, _, body in paged) == 100_000
    refs = [FilingRef("ARM", "8-K", "2026-07-29", "0001-26-000777", "https://www.sec.gov/x/777/"), FilingRef("ARM", "8-K", "2026-05-02", "0001-26-000500", "https://www.sec.gov/x/500/")]
    source = _FakeFilingSource(refs)
    llm = ScriptedLLM(
        [
            LLMResponse(text='{"action":"read","filing":1,"section":"s1"}'),  # the single-read form still works
            LLMResponse(text='```json\n{"action":"read","reads":[{"filing":1,"section":"s2"},{"filing":9,"section":"s1"}]}\n```'),
            LLMResponse(
                text=json.dumps(
                    {
                        "action": "finish",
                        "events": [
                            {"date": "2026-07-29", "summary": "季度营收 10.5 亿美元低于指引区间", "quote": "Revenue of $1.05 billion was below the guidance range", "filing": 1, "section": "s1"},
                            {"date": "2026-07-28", "summary": "CFO 提出辞职", "quote": "the Chief Financial Officer notified the board of his intention to resign", "filing": 1, "section": "s2"},
                            {"date": "2026-07-29", "summary": "指引下调（模型改写了标点和数字）", "quote": "the company now expects fiscal year revenue growth in the low-twenties", "filing": 1, "section": "s1"},
                            {"date": "2026-07-29", "summary": "编造的事件", "quote": "the company was acquired", "filing": 1, "section": "s1"},
                        ],
                        "note": "两节都读完了",
                    },
                    ensure_ascii=False,
                )
            ),
        ]
    )
    reader = FilingReader(llm, source, max_rounds=4)
    result = reader.run("ARM", _context(), around="2026-07-29", today=date(2026, 9, 9))
    assert result.ok and result.status == ResultStatus.COMPLETED
    assert source.reads == [("0001-26-000777", "s1"), ("0001-26-000777", "s2")]  # the May filing is outside the ±14-day window
    events = [item for item in result.evidence if item.metadata.get("evidence_scope") == "filing_event"]
    assert [item.metadata["date"] for item in events] == ["2026-07-29", "2026-07-28", "2026-07-29"]
    assert events[0].source_url == "https://www.sec.gov/x/777/" and "Revenue of $1.05 billion" in events[0].claim
    # A quote the model reworded is replaced by the filing's own words around the matching run.
    assert events[2].metadata["quote"] == "the company now expects fiscal-year revenue growth in the low twenties."
    assert {key: result.metrics[key] for key in ("filings", "sections_read", "events", "rounds", "llm_calls", "stop_reason")} == {"filings": 1, "sections_read": 2, "events": 3, "rounds": 3, "llm_calls": 3, "stop_reason": "finished"}
    assert "1 条事件的引文与已读文本不符，已丢弃" in result.limitations[0]
    assert result.metadata["narrative"].startswith("ARM 申报中读到的事件：2026-07-29 季度营收 10.5 亿美元低于指引区间[evidence-filing-event-")
    assert verify_answer(result.metadata["narrative"], result.evidence, answer_mode=AnswerMode.RESEARCH_GROUNDED, results=[result]).ok
    # A reader that keeps reading gets one forced finish so what it read is not wasted.
    forced = ScriptedLLM(
        [
            LLMResponse(text='{"action":"read","filing":1,"section":"s1"}'),
            LLMResponse(text='{"action":"read","filing":1,"section":"s2"}'),
            LLMResponse(text=json.dumps({"action": "finish", "events": [{"date": "2026-07-29", "summary": "营收低于指引", "quote": "Revenue of $1.05 billion was below the guidance range", "filing": 1, "section": "s1"}], "note": "被要求结束"}, ensure_ascii=False)),
        ]
    )
    rescued = FilingReader(forced, _FakeFilingSource(refs), max_rounds=2).run("ARM", _context(), around="2026-07-29", today=date(2026, 9, 9))
    assert rescued.status == ResultStatus.COMPLETED and {key: rescued.metrics[key] for key in ("filings", "sections_read", "events", "rounds", "llm_calls")} == {"filings": 1, "sections_read": 2, "events": 1, "rounds": 2, "llm_calls": 3}
    assert forced.calls[-1][-1]["content"].startswith("轮次已用完")
    # If even the forced finish and the plain-JSON turn after it keep reading, the cap holds and only a limitation comes back.
    endless = FilingReader(ScriptedLLM([LLMResponse(text='{"action":"read","filing":1,"section":"s1"}')] * 6), _FakeFilingSource(refs), max_rounds=2)
    capped = endless.run("ARM", _context(), around="2026-07-29", today=date(2026, 9, 9))
    assert capped.status == ResultStatus.PARTIAL_DATA and {key: capped.metrics[key] for key in ("filings", "sections_read", "events", "rounds", "llm_calls", "stop_reason")} == {"filings": 1, "sections_read": 1, "events": 0, "rounds": 2, "llm_calls": 4, "stop_reason": "rounds"} and "达到轮次上限" in capped.limitations[0]
    assert capped.evidence[0].metadata["citation_kind"] == "limitations" and "未读到与2026-07-29 附近下跌相关的事件" in capped.evidence[0].claim
    # Without a model the capability still lists the filings and says it did not read them.
    listed = FilingReader(None, _FakeFilingSource(refs)).run("ARM", _context(), around="2026-07-29", today=date(2026, 9, 9))
    assert listed.status == ResultStatus.PARTIAL_DATA and listed.metadata["filings"][0]["accession"] == "0001-26-000777" and "未配置模型" in listed.limitations[0]
    nothing = FilingReader(llm, _FakeFilingSource([])).run("ARM", _context(), around="2026-07-29", today=date(2026, 9, 9))
    assert nothing.ok and "未查到申报" in nothing.evidence[0].claim


def test_market_drawdown_locates_the_worst_days_and_the_peak_to_trough():
    class Prices:
        def get_prices(self, ticker, start, end):
            first = date(2026, 1, 5)
            rows = []
            close = 300.0
            for index in range(180):
                day = first + timedelta(days=index)
                if day.weekday() >= 5:
                    continue
                if day == date(2026, 5, 20):
                    close *= 0.80  # the crash day
                elif day == date(2026, 6, 3):
                    close *= 0.95
                elif day < date(2026, 5, 20):
                    close *= 1.002
                else:
                    close *= 0.999
                if day.isoformat() <= str(end):
                    rows.append(SimpleNamespace(time=day.isoformat(), close=round(close, 2), volume=1_000_000))
            return rows

    class SectorPrices(Prices):
        def get_prices(self, ticker, start, end):
            rows = super().get_prices("ARM", start, end)
            if ticker != "SMH":
                return rows
            # The sector fell half as much on the crash day and drifted the same way otherwise.
            close = 100.0
            out = [SimpleNamespace(time=rows[0].time, close=close, volume=1)] if rows else []
            for previous, bar in zip(rows, rows[1:]):
                step = float(bar.close) / float(previous.close)
                close *= 0.90 if step < 0.85 else step
                out.append(SimpleNamespace(time=bar.time, close=round(close, 2), volume=1))
            return out

    registry = CapabilityRegistry(default_catalog())
    now = datetime(2026, 7, 3, 18, 0, tzinfo=ZoneInfo("America/New_York"))
    register_market_capabilities(registry, price_source_factory=SectorPrices, move_provider=lambda ticker: None, now_factory=lambda: now, sector_for=lambda ticker: "SMH")
    result = registry.execute(PlanTask("d", "market.drawdown", {"ticker": "ARM", "loss_pct": -30.0, "top": 2}), _context())
    assert result.ok and result.metrics["window"] == "3m"
    span = next(item for item in result.evidence if item.metadata["evidence_scope"] == "benchmark_span")
    assert span.claim.startswith("同期行业基准 SMH 从 2026-05-19 到 ") and "ARM 比基准多跌 " in span.claim and result.metrics["benchmark_span"]["gap_pp"] > 5
    assert span.claim.rstrip("。") + f"[{span.id}]。" in result.metadata["narrative"]
    # An answer that skips the sector comparison is sent back; the narrative itself passes.
    skipped = verify_answer(f"ARM 从高点回撤 -37.40%[{result.evidence[1].id}]。", result.evidence, answer_mode=AnswerMode.RESEARCH_GROUNDED, results=[result])
    assert any(w.startswith(f"这段跌幅的回答必须引用同期行业基准对比那条证据 [{span.id}]") for w in skipped.warnings), skipped
    assert verify_answer(result.metadata["narrative"], result.evidence, answer_mode=AnswerMode.RESEARCH_GROUNDED, results=[result]).ok
    # The mirror image: a run-up from the low to the high, with the best days.
    register_market_capabilities(registry, price_source_factory=SectorPrices, move_provider=lambda ticker: None, now_factory=lambda: now, sector_for=lambda ticker: "SMH")
    rise = registry.execute(PlanTask("u", "market.runup", {"ticker": "ARM", "gain_pct": 12.0, "top": 2}), _context())
    assert rise.ok and rise.capability == "market.runup" and rise.metadata["direction"] == "up"
    # The largest rise in the series runs from the first bar to the day before the crash.
    assert rise.metrics["trough"]["date"] == "2026-01-05" and rise.metrics["peak"]["date"] == "2026-05-19" and 0.15 < rise.metrics["runup"] < 0.25
    best = [row["date"] for row in rise.metrics["best_days"]]
    assert best == sorted(best) and all(row["return"] > 0 for row in rise.metrics["best_days"]) and rise.metadata["best_dates"] == best
    rise_span = next(item for item in rise.evidence if item.metadata["evidence_scope"] == "benchmark_span")
    # The sector drifted the same way over that stretch, so the comparison says so.
    assert rise_span.claim.startswith("同期行业基准 SMH 从 2026-01-05 到 2026-05-19 回报 +") and rise_span.claim.endswith("与基准基本同步。")
    assert "上涨 +" in rise.metadata["narrative"] and "涨幅最大的交易日" in rise.metadata["narrative"]
    assert verify_answer(rise.metadata["narrative"], rise.evidence, answer_mode=AnswerMode.TOOL_GROUNDED, results=[rise]).ok
    # A big day in an earlier rise that fell back to the low is not one of this run-up's best days.
    class TwoRises:
        def get_prices(self, ticker, start, end):
            rows, close = [], 100.0
            for index in range(200):
                day = date(2026, 1, 5) + timedelta(days=index)
                if day.weekday() >= 5 or day.isoformat() > str(end):
                    continue
                if day == date(2026, 1, 20):
                    close *= 1.06  # the biggest single day, before the low
                elif day < date(2026, 2, 10):
                    close *= 1.001
                elif day < date(2026, 3, 20):
                    close *= 0.985  # down to the low
                elif day == date(2026, 4, 2):
                    close *= 1.04
                else:
                    close *= 1.003
                rows.append(SimpleNamespace(time=day.isoformat(), close=round(close, 2), volume=1))
            return rows

    register_market_capabilities(registry, price_source_factory=TwoRises, move_provider=lambda ticker: None, now_factory=lambda: datetime(2026, 5, 15, 18, 0, tzinfo=ZoneInfo("America/New_York")), sector_for=lambda ticker: "")
    two = registry.execute(PlanTask("u", "market.runup", {"ticker": "ARM", "window": "1y", "top": 3}), _context())
    assert two.metrics["trough"]["date"] == "2026-03-19" and two.metrics["peak"]["date"] == "2026-05-15"
    assert "2026-01-20" not in two.metadata["best_dates"] and "2026-04-02" in two.metadata["best_dates"]
    assert all("2026-03-19" < day <= "2026-05-15" for day in two.metadata["best_dates"])
    # No sector known: the block is simply absent, nothing fails.
    register_market_capabilities(registry, price_source_factory=Prices, move_provider=lambda ticker: None, now_factory=lambda: now, sector_for=lambda ticker: "")
    assert "benchmark_span" not in registry.execute(PlanTask("d", "market.drawdown", {"ticker": "ARM"}), _context()).metrics
    register_market_capabilities(registry, price_source_factory=Prices, move_provider=lambda ticker: None, now_factory=lambda: now)
    assert [row["date"] for row in result.metrics["worst_days"]] == ["2026-05-20", "2026-06-03"]
    assert result.metrics["peak"]["date"] == "2026-05-19" and result.metrics["drawdown"] < -0.15
    assert result.metadata["queries"] == ["why did ARM stock fall on 2026-05-20", "why did ARM stock fall on 2026-06-03"]
    narrative = result.metadata["narrative"]
    assert "2026-05-20 -20.00%" in narrative and "从 2026-05-19 的高点" in narrative
    assert verify_answer(narrative, result.evidence, answer_mode=AnswerMode.TOOL_GROUNDED, results=[result]).ok
    explicit = registry.execute(PlanTask("d", "market.drawdown", {"ticker": "ARM", "window": "1m"}), _context())
    assert explicit.metrics["window"] == "1m" and explicit.metrics["worst_days"]
    # A session still in progress is not a completed bar: the last row is dropped.
    intraday_now = datetime(2026, 7, 2, 11, 0, tzinfo=ZoneInfo("America/New_York"))
    register_market_capabilities(registry, price_source_factory=Prices, move_provider=lambda ticker: None, now_factory=lambda: intraday_now)
    assert registry.execute(PlanTask("d", "market.drawdown", {"ticker": "ARM"}), _context()).metrics["as_of"] == "2026-07-01"


def test_history_capabilities_wrap_edgar_filings_and_the_anomaly_memory():
    from v2.agent_v2.adapters.history import register_history_capabilities

    registry = CapabilityRegistry(default_catalog())
    rows = [
        SimpleNamespace(filing_date="2026-08-05", form="8-K", accession_number="0001-25-000001", cik="0001973239"),
        SimpleNamespace(filing_date="2026-08-20", form="8-K", accession_number="0001-25-000002", cik="0001973239"),
    ]
    calls: list[tuple] = []
    foreign = [SimpleNamespace(filing_date="2026-07-30", form="6-K", accession_number="0001-25-000009", cik="0001973239")]

    def fetch(ticker, form, since, until):
        calls.append((ticker, form, since, until))
        if ticker == "ARM":
            return list(rows) if form == "8-K" else []
        return list(foreign) if form == "6-K" and ticker == "TSM" else []

    recalls = SimpleNamespace(date="2026-08-05", flags="gap_down,volume_spike", doc="ARM  gapped down after earnings;   guidance missed.")
    # A record with neither flags nor content (the monitor wrote only the ticker) is dropped.
    blank = SimpleNamespace(date="2026-09-03", flags="", doc="ARM")
    register_history_capabilities(registry, filings_fetch=fetch, anomaly_recall=lambda ticker, query, days: [blank, recalls], today_factory=lambda: date(2026, 9, 9))
    filings = registry.execute(PlanTask("f", "filings.recent", {"ticker": "ARM", "forms": ["8-K", "10-Q"]}), _context())
    assert filings.ok and calls[0] == ("ARM", "8-K", "2025-09-09", "2026-09-09")
    assert [item.metadata["date"] for item in filings.evidence] == ["2026-08-20", "2026-08-05"]
    assert filings.evidence[0].source_url == "https://www.sec.gov/Archives/edgar/data/1973239/000125000002/"
    # An insider filing is labelled for the reader, not left as a bare "4".
    from v2.agent_v2.adapters.history import _form_label

    assert _form_label("4") == "Form 4（内幕交易）" and _form_label("8-K") == "8-K"
    # A NaN placeholder bar at the end of the series poisons nothing: it is dropped before any figure.
    class Trailing:
        def get_prices(self, ticker, start, end):
            rows = _attributor_prices(ticker, start, end)
            return [*rows, _Bar("2026-09-10", float("nan"), 79_204_871)]

    nan_registry = CapabilityRegistry(default_catalog())
    register_market_capabilities(nan_registry, price_source_factory=Trailing, move_provider=lambda ticker: None, now_factory=lambda: datetime(2026, 9, 9, 20, 0, tzinfo=ZoneInfo("America/New_York")), sector_for=lambda ticker: "SMH")
    performance = nan_registry.execute(PlanTask("p", "market.performance", {"ticker": "ARM"}), _context())
    assert performance.ok and performance.as_of == "2026-09-09" and all(value == value for value in performance.metrics["returns"].values())  # no NaN
    assert "nan" not in performance.metadata["narrative"]
    assert "2026-08-20 8-K[" in filings.metadata["narrative"]
    empty = registry.execute(PlanTask("f", "filings.recent", {"ticker": "ARM", "forms": ["10-Q"]}), _context())
    assert empty.ok and empty.evidence[0].metadata["citation_kind"] == "limitations" and "未查到 10-Q 申报" in empty.evidence[0].claim
    # A foreign private issuer has no 8-K; with no explicit form the adapter looks at 6-K before saying none.
    calls.clear()
    tsm = registry.execute(PlanTask("f", "filings.recent", {"ticker": "TSM"}), _context())
    assert [call[1] for call in calls] == ["8-K", "6-K"]
    assert tsm.ok and tsm.evidence[0].metadata["form"] == "6-K" and "2026-07-30 6-K[" in tsm.metadata["narrative"]
    calls.clear()
    none = registry.execute(PlanTask("f", "filings.recent", {"ticker": "XYZ"}), _context())
    assert [call[1] for call in calls] == ["8-K", "6-K"] and "未查到 8-K、6-K 申报" in none.evidence[0].claim
    anomalies = registry.execute(PlanTask("a", "market.anomaly_history", {"ticker": "ARM", "lookback_days": 365}), _context())
    assert anomalies.ok and [item.claim for item in anomalies.evidence] == ["ARM 2026-08-05 盯盘记录：gap_down,volume_spike；ARM gapped down after earnings; guidance missed."]
    register_history_capabilities(registry, filings_fetch=fetch, anomaly_recall=lambda *args: (_ for _ in ()).throw(RuntimeError("chroma down")))
    broken = registry.execute(PlanTask("a", "market.anomaly_history", {"ticker": "ARM"}), _context())
    assert not broken.ok and "anomaly memory unavailable" in broken.errors[0]


@pytest.mark.parametrize(
    ("query", "window", "is_drawdown"),
    [
        ("ARM 从 6 月高点为什么跌了这么多?", "", True),
        ("ARM 我买入以来为什么亏了 30%", "", True),
        ("QCOM 这几个月为什么一路跌", "", True),
        ("ARM 今年为什么回撤这么多", "1y", True),
        ("ARM 这个月为什么跌这么狠", "1m", True),
        ("NVDA 今天为什么跌这么多", "", False),
        ("NVDA 为什么涨了这么多", "", "up"),
        ("NVDA 今天为什么涨了这么多", "", False),
        ("TSLA 为什么跌，内部人在卖吗，财报什么时候", "", False),
    ],
)
def test_rule_planner_opens_a_drawdown_from_the_wording_alone(query, window, is_drawdown):
    plan = _plan(query)
    capabilities = [task.capability for task in plan.tasks]
    if is_drawdown == "up":
        assert capabilities == ["account.portfolio", "market.performance", "market.runup", "filings.recent", "market.anomaly_history", "market.attribute_move"]
        assert "gain_pct" not in plan.tasks[2].arguments and plan.frame["kind"] == "runup" and "从低点以来或这段时间的涨幅" in plan.assumptions[0]
    elif is_drawdown:
        assert capabilities == ["account.portfolio", "market.performance", "market.drawdown", "filings.recent", "market.anomaly_history", "market.attribute_move"]
        drawdown = plan.tasks[2].arguments
        assert "loss_pct" not in drawdown and drawdown.get("window", "") == window
        assert plan.assumptions[0].startswith("context_frame: 用户问的是") and "从高点以来或这段时间的跌幅" in plan.assumptions[0]
        llm = ScriptedLLM([LLMResponse(text="{}")])
        request = normalize_request(query)
        assert len(StructuredLLMPlanner(llm, default_catalog()).plan(request, route(request)).tasks) == 6 and llm.calls == []
    else:
        assert "market.drawdown" not in capabilities and "market.runup" not in capabilities and "market.attribute_move" not in capabilities


def test_decline_timing_reads_the_return_windows():
    from v2.agent_v2.synthesis import catalyst_lines, decline_timing

    item = EvidenceItem("W", "ARM", "区间回报", metadata={"evidence_scope": "returns"})
    result = ToolEnvelope("market.performance", ResultStatus.COMPLETED, subject="ARM", evidence=[item])
    assert decline_timing(-32.0, {"5d": 0.12, "1m": -0.015, "3m": -0.214, "1y": -0.351}, result)[0].endswith("这段跌幅大部分落在近 3 月内[W]。")
    assert decline_timing(-32.0, {"5d": -0.20, "1m": -0.25}, result)[0].endswith("这段跌幅大部分落在近 5 日内[W]。")
    sentences = decline_timing(-32.0, {"5d": 0.01, "1m": -0.02, "3m": -0.05, "1y": 0.90}, result)
    assert sentences[0].endswith("这段跌幅主要发生在近 1 年以前[W]。") and sentences[1].startswith("近 1 年 +90.00% 而该持仓仍在浮亏")
    assert decline_timing(5.0, {"1m": -0.02}, result) == ["对照区间回报（近 1 月 -2.00%），这段涨幅主要发生在近 1 月以前[W]。", "近 1 月 -2.00% 而该持仓仍在浮盈，说明买入点在这轮下跌之后的低位[W]。"]
    assert decline_timing(14.0, {"5d": 0.01, "1m": 0.09, "3m": 0.20}, result)[0] == "对照区间回报（近 5 日 +1.00%、近 1 月 +9.00%、近 3 月 +20.00%），这段涨幅大部分落在近 1 月内[W]。"
    assert decline_timing(-32.0, {}, result) == [] and decline_timing(0, {"1m": 0.01}, result) == []
    undated = ToolEnvelope("research.stock", ResultStatus.COMPLETED, subject="ARM", evidence=[EvidenceItem("F", "ARM", "TTM P/E is 377.2x.")], limitations=["expectations: 34/100"])
    assert catalyst_lines(undated) == "ARM 期间未查到可核对的催化剂（财报、公告或新闻）。\n数据限制：expectations: 34/100"


def test_sub_agent_loop_is_bounded_by_the_coordinators_remaining_time():
    import time

    from v2.agent_v2.agents.base import BoundedLoop, LoopLimits, limits_for
    from v2.agent_v2.agents.filing_reader import FilingReader, FilingRef

    # limits_for: the coordinator's remaining clock, minus a margin, caps the loop.
    roomy = ExecutionContext("run", NormalizedRequest("q", "q"), BudgetClass.PORTFOLIO, deadline=time.monotonic() + 600)
    assert limits_for(roomy, max_rounds=6, max_seconds=90).seconds == 90
    tight = ExecutionContext("run", NormalizedRequest("q", "q"), BudgetClass.PORTFOLIO, deadline=time.monotonic() + 30)
    assert 24 <= limits_for(tight, max_rounds=6, max_seconds=90).seconds <= 25
    assert limits_for(None, max_rounds=6, max_seconds=90).seconds == 90

    class Echo(BoundedLoop):
        def handle(self, action, messages):
            messages.append({"role": "user", "content": "ok"})
            return True

    # No budget left: the loop does not start and no model call is made.
    llm = ScriptedLLM([LLMResponse(text='{"action":"finish","events":[]}')])
    outcome = Echo(llm, LoopLimits(max_rounds=3, max_seconds=60, outer_seconds=4)).run("sys", "task", finish_prompt="finish")
    assert outcome.stop_reason == "no_budget" and outcome.calls == 0 and llm.calls == []
    assert Echo(None, LoopLimits()).run("sys", "task", finish_prompt="finish").stop_reason == "no_model"
    # A normal finish records rounds, calls and the final action.
    done = Echo(ScriptedLLM([LLMResponse(text='{"action":"read"}'), LLMResponse(text='{"action":"finish","x":1}')]), LoopLimits(max_rounds=3)).run("sys", "task", finish_prompt="finish")
    assert done.finished and done.final == {"action": "finish", "x": 1} and (done.rounds, done.calls, done.stop_reason) == (2, 2, "finished")
    assert [(step["round"], step["action"], step["detail"]) for step in done.trace] == [(1, "read", ""), (2, "finish", "")]
    exhausted = Echo(ScriptedLLM([LLMResponse(text='{"action":"read","ids":["a"]}'), LLMResponse(text='{"action":"finish","events":[1,2]}')]), LoopLimits(max_rounds=1)).run("sys", "task", finish_prompt="finish")
    assert exhausted.finished and [(step["action"], step["detail"]) for step in exhausted.trace] == [("read", '{"ids": ["a"]}'), ("forced_finish", "events=2")]

    # The engine hands the deadline to handlers: a reader called with almost no time left says so instead of reading.
    refs = [FilingRef("ARM", "8-K", "2026-07-29", "0001-26-000777", "https://www.sec.gov/x/777/")]
    reader = FilingReader(ScriptedLLM([LLMResponse(text='{"action":"finish","events":[]}')]), _FakeFilingSource(refs))
    starved = ExecutionContext("run", NormalizedRequest("q", "q"), BudgetClass.PORTFOLIO, deadline=time.monotonic() + 3)
    result = reader.run("ARM", starved, around="2026-07-29", today=date(2026, 9, 9))
    assert result.metrics["stop_reason"] == "no_budget" and result.metrics["llm_calls"] == 0 and "协调者剩余时间不足" in result.limitations[0]
    registry = CapabilityRegistry(default_catalog())
    seen: dict[str, float] = {}

    def probe(arguments, context):
        seen["remaining"] = context.remaining_seconds()
        return ToolEnvelope("account.portfolio", ResultStatus.COMPLETED, subject="portfolio", summary="x", evidence=[EvidenceItem("P", "portfolio", "x")])

    registry.register("account.portfolio", probe)
    context = ExecutionContext("run", NormalizedRequest("q", "q"), BudgetClass.DIRECT)  # DIRECT allows 30 s
    ExecutionEngine(registry).run(ExecutionPlan("q", RouteKind.FAST_LOOKUP, tasks=(PlanTask("p", "account.portfolio"),), budget=BudgetClass.DIRECT), context)
    assert 0 < seen["remaining"] <= 30 and context.deadline is not None


class _Bar:
    def __init__(self, time, close, volume=1_000_000):
        self.time, self.close, self.volume = time, close, volume


def _attributor_prices(ticker, start, end):
    first = date(2026, 1, 5)
    rows = []
    close = 300.0 if ticker == "ARM" else 100.0
    for index in range(260):
        day = first + timedelta(days=index)
        if day.weekday() >= 5 or day.isoformat() > str(end):
            continue
        if day == date(2026, 7, 29):
            close *= 0.92 if ticker == "ARM" else 0.99
        else:
            close *= 1.001
        rows.append(_Bar(day.isoformat(), round(close, 2), 3_000_000 if day == date(2026, 7, 29) and ticker == "ARM" else 1_000_000))
    return rows


def test_move_attributor_explains_a_past_day_from_sources_it_fetched_and_remembers_it():
    from v2.agent_v2.agents.move_attributor import MoveAttributor, day_facts

    prices = _attributor_prices("ARM", "2025-07-01", "2026-09-09")
    facts = day_facts("ARM", "2026-07-29", prices, "SMH", _attributor_prices("SMH", "2025-07-01", "2026-09-09"))
    assert facts.date == "2026-07-29" and abs(facts.change + 0.08) < 0.001
    assert facts.volume_ratio == 3.0 and facts.sector_return_1d is not None and facts.relative_1d < 0

    news_calls: list[str] = []

    def news(query, day):
        news_calls.append(query)
        return [
            {"title": "Arm falls as guidance disappoints", "url": "https://example.com/arm-guidance", "content": "Arm Holdings shares slid 8% on Wednesday after the company's revenue guidance came in below Wall Street expectations.", "published_date": "2026-07-29"},
            {"title": "Unrelated chip story", "url": "https://example.com/other", "content": "Nvidia rallied on strong demand.", "published_date": "2026-07-29"},
        ]

    class Reader:
        def run(self, ticker, context, *, around, today):
            return ToolEnvelope("filings.read_events", ResultStatus.COMPLETED, subject=ticker, evidence=[EvidenceItem("E-ARM-0729", "ARM", "ARM 2026-07-29：季度营收低于指引区间（6-K 2026-07-29 s1：“Revenue was below the guidance range”）。", as_of="2026-07-29", source_url="https://www.sec.gov/x/114/", metadata={"evidence_scope": "filing_event", "date": "2026-07-29", "quote": "Revenue was below the guidance range"})])

    remembered: list[tuple] = []
    memory = [SimpleNamespace(date="2026-07-29", flags="gap_down", doc="ARM gap_down 财报后跳空低开")]
    llm = ScriptedLLM(
        [
            LLMResponse(text='{"action":"news","query":"Arm Holdings stock July 29 2026 falls"}'),
            LLMResponse(text='{"action":"filing_events"}'),
            LLMResponse(text='{"action":"memory","query":"ARM 下跌"}'),
            LLMResponse(
                text=json.dumps(
                    {
                        "action": "finish",
                        "reasons": [
                            {"text": "营收指引低于华尔街预期", "confidence": "高", "source": {"kind": "news", "url": "https://example.com/arm-guidance"}, "quote": "revenue guidance came in below Wall Street expectations"},
                            {"text": "申报显示营收低于指引区间", "confidence": "中", "source": {"kind": "filing", "id": "E-ARM-0729"}, "quote": "Revenue was below the guidance range"},
                            {"text": "盯盘记录显示财报后跳空低开", "confidence": "高", "source": {"kind": "memory", "date": "2026-07-29"}, "quote": "财报后跳空低开"},
                            {"text": "编造：被收购传闻", "confidence": "高", "source": {"kind": "news", "url": "https://example.com/nowhere"}, "quote": "takeover rumours"},
                        ],
                        "next_steps": ["关注下季指引"],
                        "note": "新闻与申报一致",
                    },
                    ensure_ascii=False,
                )
            ),
        ]
    )
    attributor = MoveAttributor(llm, price_source_factory=lambda: SimpleNamespace(get_prices=_attributor_prices), news=news, filing_reader=Reader(), memory_recall=lambda ticker, query, days: memory, memory_remember=lambda facts, reasons: remembered.append((facts.date, [(r["text"], r["confidence"]) for r in reasons])) or "ARM_2026-07-29_retro", sector_for=lambda ticker: "SMH")
    context = ExecutionContext("run", NormalizedRequest("q", "q"), BudgetClass.PORTFOLIO, allow_web=True)
    result = attributor.run("ARM", context, day="2026-07-29", today=date(2026, 9, 9))
    assert result.ok and result.status == ResultStatus.COMPLETED and news_calls == ["Arm Holdings stock July 29 2026 falls"]
    drivers = [item for item in result.evidence if item.metadata.get("claim_role") == "confirmed_driver"]
    candidates = [item for item in result.evidence if item.metadata.get("claim_role") == "candidate_driver"]
    assert [item.metadata["driver_text"] for item in drivers] == ["营收指引低于华尔街预期"] and drivers[0].source_url == "https://example.com/arm-guidance"
    assert [(item.metadata["driver_text"], item.metadata["causal_confidence"]) for item in candidates] == [("申报显示营收低于指引区间", "中"), ("盯盘记录显示财报后跳空低开", "中")]  # memory-only support is capped at 中
    assert any(item.id == "E-ARM-0729" for item in result.evidence)  # the reader's event travels with the attribution
    assert "1 条原因没有可核对的来源，已丢弃" in result.limitations[0]
    assert result.metrics["confirmed_driver_count"] == 1 and result.metrics["news_calls"] == 1 and result.metrics["reader_calls"] == 1 and result.metrics["memory_calls"] == 1
    assert result.metrics["remembered_as"] == "ARM_2026-07-29_retro" and remembered == [("2026-07-29", [("营收指引低于华尔街预期", "高"), ("申报显示营收低于指引区间", "中"), ("盯盘记录显示财报后跳空低开", "中")])]
    narrative = result.metadata["narrative"]
    assert narrative.startswith("ARM 在 2026-07-29 收于") and "能直接支持的高置信度驱动：营收指引低于华尔街预期[" in narrative and "跑输行业基准 SMH" in narrative
    assert verify_answer(narrative, result.evidence, answer_mode=AnswerMode.RESEARCH_GROUNDED, results=[result]).ok
    assert "申报读到：2026-07-29 季度营收低于指引区间[E-ARM-0729]。" in narrative
    # An answer that calls the read filings unread is sent back by the verifier: the rule is a sentence the judge decides, in any wording.
    seen_claims: list[dict] = []

    def judge(items):
        seen_claims.extend(items)
        return {item["id"]: "正文未读取" for item in items if "没有被读取" in item["claim"]}

    unread = verify_answer(f"ARM 当天提交了 6-K，日期和表格类型可见，正文未读取[{result.evidence[0].id}]。", result.evidence, answer_mode=AnswerMode.RESEARCH_GROUNDED, results=[result], judge=judge)
    assert any(warning.startswith("2026-07-29 附近的申报已由申报阅读者读取并摘出事件，回答却称申报内容未读取") and warning.endswith("（“正文未读取”）") for warning in unread.warnings)
    assert any(item["claim"].startswith("申报的正文或内容没有被读取") and item["id"].startswith("answer:market.attribute_move") for item in seen_claims)
    # Without a judge (no model) the claim rules are simply not applied.
    assert verify_answer(f"ARM 当天提交了 6-K，正文未读取[{result.evidence[0].id}]。", result.evidence, answer_mode=AnswerMode.RESEARCH_GROUNDED, results=[result]).ok
    compact = result.metadata["narrative_compact"]
    assert compact.startswith("2026-07-29 ARM ") and "驱动：营收指引低于华尔街预期[" in compact and "跑输 SMH" in compact
    assert "申报读到" not in compact  # a confirmed driver keeps the phone version to one lead
    assert "\n" not in compact and len(compact) < len(narrative)
    assert verify_answer(compact, result.evidence, answer_mode=AnswerMode.RESEARCH_GROUNDED, results=[result]).ok
    # Without web consent the news action is refused and the loop is told so.
    refused = ScriptedLLM([LLMResponse(text='{"action":"news","query":"x"}'), LLMResponse(text='{"action":"finish","reasons":[],"note":"无新闻"}')])
    quiet = MoveAttributor(refused, price_source_factory=lambda: SimpleNamespace(get_prices=_attributor_prices), news=news, filing_reader=Reader(), memory_recall=None, memory_remember=None, sector_for=None)
    result = quiet.run("ARM", ExecutionContext("run", NormalizedRequest("q", "q"), BudgetClass.PORTFOLIO), day="2026-07-29", today=date(2026, 9, 9))
    assert result.status == ResultStatus.PARTIAL_DATA and result.metrics["news_calls"] == 0 and len(news_calls) == 1
    assert "用户未授权网页搜索" in refused.calls[1][-1]["content"] and "归因未使用新闻" in " ".join(result.limitations)


def test_locate_quote_tolerates_punctuation_and_rejects_invention():
    from v2.agent_v2.agents.filing_reader import locate_quote

    text = "Revenue of $1,050 million was below the guidance range; the company now expects fiscal-year revenue growth in the “low twenties”. Nothing else."
    assert locate_quote("Revenue of $1,050 million was below the guidance range", text) == "Revenue of $1,050 million was below the guidance range"
    assert locate_quote('the company now expects fiscal-year revenue growth in the “low twenties”', text) == 'the company now expects fiscal-year revenue growth in the "low twenties"'
    assert locate_quote("the company now expects fiscal year revenue growth in the low-twenties", text) == 'the company now expects fiscal-year revenue growth in the "low twenties".'
    assert locate_quote("the company was acquired by a competitor last week", text) is None
    assert locate_quote("", text) is None and locate_quote("anything", "") is None


def test_framed_answer_renders_web_results_by_headline_and_date_and_coverage_as_one_line():
    from v2.agent_v2.synthesis import web_lines

    result = ToolEnvelope(
        "web.research",
        ResultStatus.COMPLETED,
        subject="ARM",
        evidence=[
            EvidenceItem("W1", "ARM", "Arm shares slid 8% after guidance came in below expectations.", as_of="2026-07-30T12:00:00", source_title="Arm falls on soft outlook", source_url="https://example.com/a", metadata={"evidence_type": "search_snippet"}),
            EvidenceItem("W2", "ARM", "An older story.", as_of="2026-03-01", source_title="Arm rallies", source_url="https://example.com/b", metadata={"evidence_type": "search_snippet"}),
            EvidenceItem("W3", "ARM", "Undated aggregator page " * 20, source_title="Stock page", source_url="https://example.com/c", metadata={"evidence_type": "search_snippet"}),
        ],
        limitations=["Evidence contains search-result snippets; source pages were not fetched in this adapter."],
    )
    lines = web_lines(result, "2026-06-10").split("\n")
    assert lines[0] == "- Arm falls on soft outlook（2026-07-30）：Arm shares slid 8% after guidance came in below expectations. [W1]"
    assert "[W2]" not in "\n".join(lines) and lines[1].startswith("- Stock page（日期未知）：") and lines[1].endswith("… [W3]")
    assert web_lines(ToolEnvelope("web.research", ResultStatus.COMPLETED, subject="ARM"), "2026-06-10") == "ARM 网页搜索未返回落在区间内的报道。"


def test_frame_lead_drops_a_sentence_its_own_verifier_rejects():
    from v2.agent_v2.adapters.legacy import _wrap
    from v2.agent_v2.synthesis import frame_lead

    portfolio = _wrap("account.portfolio", "portfolio", _PORTFOLIO_CARD)
    # A price item whose rule no generated wording can satisfy: the aside must be dropped, the rest kept.
    price = EvidenceItem("P", "ARM", "ARM 盘中 +0.94%。", metadata={"evidence_scope": "price", "constraints": [{"require": "永远不会出现的标记", "warning": "rule"}]})
    windows = EvidenceItem("W", "ARM", "ARM 区间回报：1d +0.94%，1m -1.53%，3m -21.40%。", metadata={"evidence_scope": "returns"})
    performance = ToolEnvelope("market.performance", ResultStatus.COMPLETED, subject="ARM", metrics={"returns": {"1d": 0.0094, "1m": -0.0153, "3m": -0.2140}}, evidence=[price, windows])
    frame = {"kind": "position", "ticker": "ARM", "field": "pl_pct", "text": "pl_pct_text", "label": "买入以来的浮动盈亏"}
    lead = frame_lead(frame, [portfolio, performance])
    assert "今日" not in lead and "这段跌幅大部分落在近 3 月内[W]" in lead


def test_agent_v2_seed_eval_passes_offline():
    report = run_suite()
    assert report.passed == report.total


def test_every_telegram_handler_keeps_its_owner_guard():
    """A helper inserted between @authorized_only and its handler once stole the decorator; never again."""

    import re
    from pathlib import Path

    source = Path(__file__).resolve().parents[1].joinpath("bot", "commands.py").read_text(encoding="utf-8")
    handlers = re.findall(r"^(@authorized_only\n)?async def (cmd_\w+)\(", source, re.M)
    assert handlers, "no handlers found"
    assert [name for decorator, name in handlers if not decorator] == []
    assert "from v2.agent import" not in source


def _require_telegram() -> None:
    """Skip when python-telegram-bot is absent or its native deps fail to load (a sandbox, not a bug)."""

    import importlib

    try:
        importlib.import_module("telegram")
    except BaseException as exc:  # noqa: BLE001 — pyo3 raises a PanicException, not ImportError
        pytest.skip(f"telegram unavailable: {type(exc).__name__}")


def test_telegram_plain_messages_keep_v2_after_legacy_retirement(monkeypatch):
    _require_telegram()
    from v2.bot import agent_v2_bridge, commands

    called: list[dict] = []

    async def handle(update, context, text, *, allow_web=False):
        called.append({"text": text, "allow_web": allow_web})

    class Message:
        def __init__(self, text):
            self.text = text
            self.replies = []

        async def reply_html(self, text, **kwargs):
            self.replies.append(text)
            return self

        async def edit_text(self, text, **kwargs):
            self.replies.append(text)

    class Chat:
        id = 7

    def update_for(text):
        return type("Update", (), {"message": Message(text), "effective_chat": Chat()})()

    monkeypatch.setenv("TELEGRAM_CHAT_ID", "7")
    monkeypatch.delenv("TELEGRAM_FREE_TEXT_AGENT", raising=False)
    monkeypatch.setattr(agent_v2_bridge, "handle_agent_v2", handle)
    monkeypatch.delenv("TELEGRAM_WEB_DEFAULT", raising=False)
    update = update_for("为什么跌这么狠")
    asyncio.run(commands.cmd_nl(update, object()))
    assert called == [{"text": "为什么跌这么狠", "allow_web": True}] and update.message.replies == []
    asyncio.run(commands.cmd_nl(update_for("为什么跌这么狠 --noweb"), object()))
    assert called[-1] == {"text": "为什么跌这么狠", "allow_web": False}
    handled = len(called)
    # Retired configuration must not silently reactivate a removed runtime.
    monkeypatch.setenv("TELEGRAM_FREE_TEXT_AGENT", "v1")
    asyncio.run(commands.cmd_nl(update_for("为什么跌这么狠"), object()))
    assert len(called) == handled + 1
    from pathlib import Path
    main = Path(commands.__file__).with_name("main.py").read_text(encoding="utf-8")
    assert 'CommandHandler("ask", commands.cmd_agent_v2, block=False)' in main


def test_telegram_ask_v2_command_is_explicit_and_parses_web_consent(monkeypatch):
    _require_telegram()
    from v2.bot import agent_v2_bridge, commands

    called = {}

    async def handle(update, context, text, *, allow_web=False):
        called.update({"text": text, "allow_web": allow_web})

    class Message:
        async def reply_html(self, text, **kwargs):
            pytest.fail(f"unexpected usage response: {text}")

    class Chat:
        id = 7

    class Update:
        message = Message()
        effective_chat = Chat()

    class Context:
        args = ["--noweb", "比较", "NVDA", "和", "AMD"]

    monkeypatch.setenv("TELEGRAM_CHAT_ID", "7")
    monkeypatch.setattr(agent_v2_bridge, "handle_agent_v2", handle)
    asyncio.run(commands.cmd_agent_v2(Update(), Context()))
    assert called == {"text": "比较 NVDA 和 AMD", "allow_web": False}


def test_telegram_web_consent_defaults_on_with_an_opt_out(monkeypatch):
    from v2.bot.agent_v2_bridge import split_web_consent

    monkeypatch.delenv("TELEGRAM_WEB_DEFAULT", raising=False)
    assert split_web_consent("为什么跌这么狠") == ("为什么跌这么狠", True)
    assert split_web_consent("为什么跌这么狠 --noweb") == ("为什么跌这么狠", False)
    assert split_web_consent("--WEB 为什么跌这么狠 --noweb") == ("为什么跌这么狠", False)
    monkeypatch.setenv("TELEGRAM_WEB_DEFAULT", "0")
    assert split_web_consent("为什么跌这么狠") == ("为什么跌这么狠", False)
    assert split_web_consent("--web 为什么跌这么狠") == ("为什么跌这么狠", True)


def _telegram_result(answer: str, *, outcome: str = "fallback", warnings: tuple[str, ...] = ()):
    from v2.agent_v2.models import AgentResult, AnswerMode, ExecutionPlan, NormalizedRequest, ResultStatus, RouteDecision, RouteKind, RunStatus, ToolEnvelope, VerificationReport

    price = EvidenceItem("evidence-market-price-42f5c41951e36ef1", "ARM", "ARM 在 2026-07-29 收于 149.35，当日 -13.21%。", source_id="market_data")
    news = EvidenceItem("evidence-news-1", "ARM", "Arm Holdings slides after guidance disappoints; the stock fell 13%.", source_id="web_news", source_title="Arm slides on soft guidance", source_url="https://example.com/arm")
    full = "ARM 在 2026-07-29 收于 149.35，当日 -13.21%[evidence-market-price-42f5c41951e36ef1]。当日成交量为 30 日均量的 4 倍。\n\n能直接支持的高置信度驱动：营收指引低于预期[evidence-news-1]。\n\n从盘面看，当天跑输行业基准 SMH 约 10.00%。"
    compact = "2026-07-29 ARM -13.21%[evidence-market-price-42f5c41951e36ef1]，跑输 SMH 约 10.00%。驱动：营收指引低于预期[evidence-news-1]。"
    envelope = ToolEnvelope("market.attribute_move", ResultStatus.COMPLETED, subject="ARM", evidence=[price, news], metadata={"narrative": full, "narrative_compact": compact})
    request = NormalizedRequest("为什么跌这么狠", "为什么跌这么狠")
    return AgentResult(
        "run", request, RouteDecision(RouteKind.RESEARCH, ("research",), "why"), ExecutionPlan(objective="q", route=RouteKind.RESEARCH),
        RunStatus.COMPLETED, answer.replace("{full}", full), AnswerMode.RESEARCH_GROUNDED, results=[envelope], evidence=[price, news],
        verification=VerificationReport(ok=not warnings, warnings=warnings),
        synthesis={"outcome": outcome, "attempts": [{"stage": "draft", "ok": False, "warnings": ["行情事实缺少邻近引用"]}, {"stage": "repair", "ok": False, "unknown_citations": ["results.metrics"]}] if outcome == "fallback" else []},
    )


def test_telegram_delivery_numbers_citations_and_compacts_worst_days(monkeypatch):
    from v2.agent_v2.interfaces import telegram_format
    from v2.bot.agent_v2_bridge import TelegramBotTransport

    result = _telegram_result("ARM 自买入以来浮亏 20%[evidence-market-price-42f5c41951e36ef1]。\n\n{full}", warnings=("未确认直接驱动时展示了过多弱候选线索",))
    numbered = telegram_format.number_citations(telegram_format.compact_attributions(result.answer, result), result.evidence)
    assert numbered.ids == ("evidence-market-price-42f5c41951e36ef1", "evidence-news-1")
    assert numbered.text == "ARM 自买入以来浮亏 20%[1]。\n\n2026-07-29 ARM -13.21%[1]，跑输 SMH 约 10.00%。驱动：营收指引低于预期[2]。"
    # Brackets that are not evidence ids are left alone.
    assert telegram_format.number_citations("ARM [2026-07-29] 跌 [evidence-news-1]", result.evidence).text == "ARM [2026-07-29] 跌 [1]"
    # Further text continues the same numbering: known ids keep theirs, a new id gets the next number.
    extra = EvidenceItem("evidence-risk-9", "ARM", "风险因素变化 15 项。", source_id="sec_filings")
    order = list(numbered.ids)
    note = telegram_format.number_citations("反对（引 [evidence-risk-9]）和（引 [evidence-news-1]）", [*result.evidence, extra], order=order)
    assert note.text == "反对（引 [3]）和（引 [2]）" and order == ["evidence-market-price-42f5c41951e36ef1", "evidence-news-1", "evidence-risk-9"]
    labelled = telegram_format.source_entries(tuple(order), [*result.evidence, extra])
    assert (labelled[-1].numbers, labelled[-1].label) == ("3", "SEC 申报（EDGAR）")
    engine = EvidenceItem("e-1", "NVDA", "x", metadata={"module": "valuation"})
    assert telegram_format.source_entries(("e-1",), [engine])[0].label == "研究引擎·valuation"
    web = EvidenceItem("w-1", "NVDA", "x", source_id="web:reuters.com")
    assert telegram_format.source_entries(("w-1",), [web])[0].label == "网页（reuters.com）"
    entries = telegram_format.source_entries(numbered.ids, result.evidence)
    # A linked page shows its title (and date), not the claim the answer already quotes.
    assert [(entry.numbers, entry.label, entry.url) for entry in entries] == [
        ("1", "日线行情", ""),
        ("2", "Arm slides on soft guidance", "https://example.com/arm"),
    ]
    dated = EvidenceItem("w-2", "ARM", "x", as_of="2026-09-04", source_title="Opinions on Recent Earnings", source_url="https://example.com/q")
    assert telegram_format.source_entries(("w-2",), [dated])[0].label == "Opinions on Recent Earnings（2026-09-04）"
    # Several claims from the same page share one line, numbered as a range.
    same_page = [EvidenceItem(f"q{i}", "ARM", f"claim {i}", as_of="2026-09-04", source_title="Opinions on Recent Earnings", source_url="https://example.com/q") for i in range(1, 4)]
    other = EvidenceItem("q4", "ARM", "other", source_title="Other", source_url="https://example.com/o")
    merged = telegram_format.source_entries(("q1", "q2", "q4", "q3"), [*same_page, other])
    assert [(entry.numbers, entry.label) for entry in merged] == [("1–2、4", "Opinions on Recent Earnings（2026-09-04）"), ("3", "Other")]
    # Unlinked items are one line per origin, with their numbers as ranges.
    many = [EvidenceItem(f"m{i}", "ARM", f"row {i}", source_id="market_data") for i in range(1, 8)]
    many[3] = EvidenceItem("m4", "ARM", "card\n━━━\nrow", source_title="Existing deterministic responder")
    grouped = telegram_format.source_entries(tuple(item.id for item in many), many)
    assert [(entry.numbers, entry.label) for entry in grouped] == [("1–3、5–7", "日线行情"), ("4", "账户卡片")]
    filing = EvidenceItem("f1", "ARM", "ARM 于 2026-07-29 向 SEC 提交了 6-K（0001）。", source_id="sec_edgar", source_title="ARM 6-K 2026-07-29", source_url="https://www.sec.gov/x")
    assert telegram_format.source_entries(("f1",), [filing])[0].label == "ARM 6-K 2026-07-29"

    class Placeholder:
        sent: list[str] = []

        async def edit_text(self, text, **kwargs):
            self.sent.append(text)

    monkeypatch.setenv("AGENT_V2_WEB_ENABLED", "1")
    placeholder = Placeholder()
    transport = TelegramBotTransport(object(), placeholder, web_requested=False)
    asyncio.run(transport.deliver(7, result))
    (message,) = placeholder.sent
    header, _, body = message.partition("\n\n")
    assert "合成：兜底摘要" in header and "校验：有警告（1）" in header and "网页：已关闭（去掉 --noweb 可用新闻归因）" in header
    assert "<i>⚠ 校验：未确认直接驱动时展示了过多弱候选线索</i>" in header and "<i>兜底原因：初稿：行情事实缺少邻近引用；修正稿：未知引用 results.metrics</i>" in header
    assert "[evidence-" not in body and "[1]。" in body and "跑输 SMH" in body and "当日成交量" not in body
    assert body.endswith('<b>来源</b>\n1. 日线行情\n2. <a href="https://example.com/arm">Arm slides on soft guidance</a>')
    # A debater note citing an id the answer never used gets the next number and a source line.
    debated = _telegram_result("模型自己的话[evidence-news-1]。", outcome="clean")
    risk = EvidenceItem("evidence-risk-9", "ARM", "风险因素变化 15 项。", source_id="sec_filings")
    debated.evidence.append(risk)
    debated.results.append(ToolEnvelope("debate.challenge", ResultStatus.COMPLETED, subject="ARM", metadata={"agent": {"name": "debater", "label": "反方", "subject": "ARM", "rounds": 1, "llm_calls": 1, "elapsed_ms": 1500, "stop_reason": "finished", "calls": {"objections": 1}, "notes": ["该证据只说明数量变化，未给出方向（引 [evidence-risk-9]）"]}, "trace": [], "citation_kind": "display"}))
    asyncio.run(transport.deliver(7, debated))
    delivered = placeholder.sent[-1]
    assert "模型自己的话[1]。" in delivered and "<b>来源</b>\n1. <a href" in delivered and "\n2. SEC 申报（EDGAR）" in delivered
    assert "反方 ARM：1 轮 · 1.5s · 反对 1 · 完成\n  · 该证据只说明数量变化，未给出方向（引 [2]）" in delivered
    # A model-written answer never contains the narrative verbatim and is delivered as written.
    clean = _telegram_result("模型自己的话[evidence-news-1]。", outcome="clean")
    transport = TelegramBotTransport(object(), placeholder, web_requested=True)
    asyncio.run(transport.deliver(7, clean))
    assert "合成：模型回答 · 校验：通过 · 网页：已启用" in placeholder.sent[-1] and "模型自己的话[1]。" in placeholder.sent[-1]
    clean.synthesis["citation_completions"] = ["439.46 → [x]", "12.52 → [y]"]
    asyncio.run(transport.deliver(7, clean))
    assert "合成：模型回答，引用补全 2 处 · 校验：通过" in placeholder.sent[-1]
    repaired = _telegram_result("模型自己的话[evidence-news-1]。", outcome="repaired")
    repaired.synthesis["attempts"] = [{"stage": "draft", "ok": False, "warnings": ["回撤回答必须引用同期行业基准对比那条证据 [D-span]（…）"]}, {"stage": "repair", "ok": True}]
    asyncio.run(transport.deliver(7, repaired))
    assert "合成：模型回答（修正一轮）" in placeholder.sent[-1] and "<i>修正原因：初稿：回撤回答必须引用同期行业基准对比那条证据 [D-span]（…）</i>" in placeholder.sent[-1]
    assert "兜底原因" not in placeholder.sent[-1]
    assert "⚠ 校验" not in placeholder.sent[-1] and "兜底原因" not in placeholder.sent[-1]
    monkeypatch.setenv("AGENT_V2_WEB_ENABLED", "0")
    asyncio.run(transport.deliver(7, clean))
    assert "网页：未启用（服务端 AGENT_V2_WEB_ENABLED 未开）" in placeholder.sent[-1]


def test_workspace_lab_port_reuses_an_injected_runner_and_builds_evidence():
    progress = []

    class Input:
        def __init__(self, **values):
            self.values = values

    def runner(body, on_tick=None):
        assert body.values["strategy"] == "momentum"
        on_tick(3)
        return {
            "kind": "backtest",
            "strategy": "momentum",
            "tickers": ["NVDA"],
            "metrics": {"total_return_pct": 0.12, "n_trades": 8},
            "trades": [{"ticker": "NVDA", "return_pct": 0.03}],
        }

    context = ExecutionContext(
        "lab-run",
        NormalizedRequest("回测", "回测"),
        BudgetClass.LAB,
        on_progress=lambda event: progress.append(event.message),
    )
    port = WorkspaceLabPort({"lab.backtest": LabBinding(Input, runner, supports_progress=True)})
    result = port.run("lab.backtest", {"strategy": "momentum"}, context)
    assert result.ok
    assert result.metrics["total_return_pct"] == 0.12
    assert any(item.metric == "n_trades" for item in result.evidence)
    assert progress == ["lab.backtest: completed 3 work unit(s)"]


def test_tavily_web_adapter_bounds_and_deduplicates_search_evidence():
    class Provider:
        last_diagnostics = {"provider": "fake"}

        def search(self, query, *, days, max_results):
            assert query == "NVDA latest product"
            assert days == 30 and max_results == 2
            return [
                {
                    "title": "Source A",
                    "content": "A" * 600,
                    "url": "https://example.test/a#section",
                    "score": 0.8,
                },
                {
                    "title": "Duplicate",
                    "content": "duplicate",
                    "url": "https://example.test/a",
                },
                {"title": "Unsafe", "content": "ignored", "url": "file:///tmp/a"},
            ]

    result = TavilyWebSearchPort(Provider(), max_results=2, max_content_chars=300).search("NVDA latest product", topic="company_event", ticker="NVDA")
    assert result.ok
    assert len(result.evidence) == 1
    assert result.evidence[0].source_url == "https://example.test/a"
    assert len(result.evidence[0].claim) == 300
    assert result.evidence[0].metadata["evidence_type"] == "search_snippet"


def test_web_fallback_requires_runtime_and_per_request_opt_in():
    catalog = default_catalog()
    registry = CapabilityRegistry(catalog)
    calls = []

    def failed_research(args, context):
        return ToolEnvelope("research.stock", ResultStatus.FAILED, errors=["provider down"])

    def web(args, context):
        calls.append(args["query"])
        item = EvidenceItem("WEB1", "NVDA", "A current source supports the event.")
        return ToolEnvelope(
            "web.research",
            ResultStatus.COMPLETED,
            summary=item.claim,
            evidence=[item],
        )

    registry.register("research.stock", failed_research)
    registry.register("web.research", web)
    agent = AgentV2(
        catalog=catalog,
        registry=registry,
        config=AgentV2Config(enable_web_fallback=True),
    )
    disabled = agent.run("分析 NVDA 的最新事件", allow_web=False)
    enabled = agent.run("分析 NVDA 的最新事件", allow_web=True)
    assert all(result.capability != "web.research" for result in disabled.results)
    assert calls == ["分析 NVDA 的最新事件"]
    # A two-stock question hands both tickers to the fallback.
    registry.register("research.compare", lambda arguments, context: ToolEnvelope("research.compare", ResultStatus.FAILED, errors=["provider down"]))
    seen: list[dict] = []
    registry.register("web.research", lambda args, context: seen.append(dict(args)) or web(args, context))
    agent.run("MU和SNDK哪个更值得购买？", allow_web=True)
    assert seen and seen[-1]["ticker"] == "MU" and seen[-1]["tickers"] == ["MU", "SNDK"]
    assert enabled.answer_mode == AnswerMode.WEB_GROUNDED
    assert enabled.plan.tasks[-1].capability == "web.research"
    assert "[WEB1]" in enabled.answer


def test_router_requires_a_user_state_object_before_treating_english_verbs_as_commands():
    assert route(normalize_request("AVGO 的 total addressable market 有多大")).kind != RouteKind.COMMAND
    assert route(normalize_request("NVDA 加入标普指数会怎样")).kind != RouteKind.COMMAND
    assert route(normalize_request("add NVDA to my watchlist")).kind == RouteKind.COMMAND
    assert route(normalize_request("set an alert for AAPL")).kind == RouteKind.COMMAND
    assert route(normalize_request("删除 TSLA 提醒")).kind == RouteKind.COMMAND


@pytest.mark.parametrize(
    ("query", "entities"),
    [
        ("分析 nvda 的估值", ("NVDA",)),
        ("分析英伟达的估值", ("NVDA",)),
        ("比较阿里巴巴和拼多多", ("BABA", "PDD")),
        ("V 最近表现怎么样", ("V",)),
        ("BRK.B 估值高吗", ("BRK.B",)),
        ("what is the cost now", ()),
        ("t+1 结算规则", ()),
        ("NVDA 的 EPS 和 ROE", ("NVDA",)),
    ],
)
def test_entities_resolve_aliases_and_known_symbols(query, entities):
    assert normalize_request(query).entities == entities


def test_llm_planner_trims_an_over_budget_plan_instead_of_failing():
    rows = [
        {"id": "t1", "capability": "research.stock", "arguments": {"ticker": "NVDA", "focus": "valuation"}},
        {"id": "t2", "capability": "research.stock", "arguments": {"ticker": "NVDA", "focus": "earnings"}},
        {"id": "t3", "capability": "research.stock", "arguments": {"ticker": "NVDA", "focus": "risk"}},
        {"id": "t4", "capability": "market.performance", "arguments": {"ticker": "NVDA"}, "required": False},
        {"id": "t5", "capability": "market.explain_move", "arguments": {"ticker": "NVDA"}, "depends_on": ["t4"]},
        {"id": "t6", "capability": "research.changes", "arguments": {"ticker": "NVDA"}},
    ]
    llm = ScriptedLLM([LLMResponse(text=json.dumps({"objective": "x", "tasks": rows}))])
    catalog = default_catalog()
    request = normalize_request("深入分析 NVDA 的估值、财报和风险")
    plan = StructuredLLMPlanner(llm, catalog).plan(request, route(request))
    prompt = json.loads(llm.calls[0][1]["content"])
    assert prompt["maximum_tasks"] == 5
    assert plan.budget == BudgetClass.STANDARD
    assert [task.id for task in plan.tasks] == ["t1", "t2", "t3", "t6"]
    assert any("trimmed" in value for value in plan.assumptions)
    ExecutionEngine(CapabilityRegistry(catalog)).run(plan, ExecutionContext("run", request, plan.budget))


def _research_fixture():
    request = normalize_request("分析 NVDA 的增长")
    plan = ExecutionPlan("分析 NVDA 的增长", RouteKind.RESEARCH, answer_mode=AnswerMode.RESEARCH_GROUNDED)
    evidence = [EvidenceItem("E1", "NVDA", "NVDA revenue growth was 10%.")]
    results = [ToolEnvelope("research.stock", ResultStatus.COMPLETED, subject="NVDA", summary="NVDA revenue growth was 10%.", evidence=evidence)]
    return request, plan, results, evidence


def test_llm_synthesizer_repairs_an_ungrounded_draft_once():
    llm = ScriptedLLM([LLMResponse(text="NVDA 收入增长 20%。[E1]"), LLMResponse(text="NVDA 收入增长 10%。[E1]")])
    request, plan, results, evidence = _research_fixture()
    answer = LLMEvidenceSynthesizer(llm).synthesize(request, plan, results, evidence)
    assert answer == "NVDA 收入增长 10%。[E1]"
    assert len(llm.calls) == 2
    repair = llm.calls[1]
    assert repair[2] == {"role": "assistant", "content": "NVDA 收入增长 20%。[E1]"}
    assert "20" in repair[3]["content"]
    assert "完整回答" in repair[3]["content"]


def test_llm_synthesizer_falls_back_to_deterministic_prose_when_repair_still_fails(caplog):
    import logging

    llm = ScriptedLLM([LLMResponse(text="NVDA 收入增长 20%。[E1]"), LLMResponse(text="NVDA 收入增长 25%。[E1]")])
    request, plan, results, evidence = _research_fixture()
    with caplog.at_level(logging.WARNING, logger="v2.agent_v2.llm"):
        answer = LLMEvidenceSynthesizer(llm).synthesize(request, plan, results, evidence)
    assert "20%" not in answer and "25%" not in answer
    # Both rejected drafts are in the server log with what the verifier said, so a fallback can be diagnosed later.
    logged = [record.getMessage() for record in caplog.records if "synthesis fell back" in record.getMessage()]
    assert len(logged) == 2 and "stage=draft" in logged[0] and "NVDA 收入增长 20%。[E1]" in logged[0] and "stage=repair" in logged[1] and "25%" in logged[1]
    assert "[E1]" in answer
    assert verify_answer(answer, evidence, answer_mode=plan.answer_mode, results=results).ok


def _mutation_agent(applied: list):
    catalog = default_catalog()
    registry = CapabilityRegistry(catalog)

    def mutate(arguments, context):
        applied.append(arguments)
        return ToolEnvelope("state.mutate", ResultStatus.COMPLETED, subject=arguments["operation"], summary="已将 NVDA 加入关注列表。", evidence=[EvidenceItem("M1", "NVDA", "已将 NVDA 加入关注列表。")])

    registry.register("state.mutate", mutate)
    return AgentV2(catalog=catalog, registry=registry, session=ShortTermSession())


def test_command_is_parsed_held_for_confirmation_and_applied_only_after_confirm():
    applied: list = []
    agent = _mutation_agent(applied)
    first = agent.run("把 NVDA 加入关注列表", session_id="chat-1")
    assert first.status == RunStatus.WAITING_CONFIRMATION
    assert first.pending_mutation is not None
    assert first.pending_mutation.operation == "watchlist.add"
    assert first.pending_mutation.payload == {"ticker": "NVDA"}
    assert "确认" in first.answer
    assert not applied
    second = agent.run("确认", session_id="chat-1")
    assert second.status == RunStatus.COMPLETED
    assert applied == [{"operation": "watchlist.add", "payload": {"ticker": "NVDA"}}]
    assert second.results[0].capability == "state.mutate"
    assert second.verification.ok
    third = agent.run("确认", session_id="chat-1")
    assert third.status != RunStatus.COMPLETED or not third.results
    assert len(applied) == 1


def test_command_cancel_or_new_question_drops_the_pending_mutation():
    applied: list = []
    agent = _mutation_agent(applied)
    agent.run("NVDA 涨到 200 美元提醒我", session_id="chat-2")
    cancelled = agent.run("取消", session_id="chat-2")
    assert cancelled.status == RunStatus.CANCELLED
    assert agent.run("确认", session_id="chat-2").results == []
    agent.run("把 AMD 加入关注列表", session_id="chat-3")
    moved_on = agent.run("什么是自由现金流？", session_id="chat-3")
    assert moved_on.route.kind == RouteKind.GENERAL_KNOWLEDGE
    assert agent.run("确认", session_id="chat-3").results == []
    assert not applied


def test_command_without_a_session_or_with_missing_parameters_does_not_wait_forever():
    applied: list = []
    agent = _mutation_agent(applied)
    no_session = agent.run("把 NVDA 加入关注列表")
    assert no_session.status == RunStatus.WAITING_CONFIRMATION
    assert "无法接收确认" in no_session.answer
    incomplete = agent.run("取消 NVDA 的提醒", session_id="chat-4")
    assert incomplete.status == RunStatus.WAITING_CLARIFICATION  # the next message is read as the alert id
    assert "提醒编号" in incomplete.answer
    assert incomplete.pending_mutation is None
    assert not applied


def test_state_mutate_adapter_maps_operations_onto_bot_state(monkeypatch):
    import sys
    from types import ModuleType

    from v2.agent_v2.adapters.legacy import register_legacy_capabilities

    calls: list = []
    fake = ModuleType("v2.bot.state")
    fake.watchlist_add = lambda ticker, note="": calls.append(("add", ticker)) or True
    fake.watchlist_remove = lambda ticker: calls.append(("remove", ticker)) or False
    fake.alert_add = lambda ticker, direction, target: calls.append(("alert", ticker, direction, target)) or 7
    fake.alert_remove = lambda alert_id: calls.append(("unalert", alert_id)) or True
    monkeypatch.setitem(sys.modules, "v2.bot.state", fake)
    registry = CapabilityRegistry(default_catalog())
    register_legacy_capabilities(registry)
    context = ExecutionContext("run", NormalizedRequest("q", "q"), BudgetClass.DIRECT, allow_mutations=True)
    added = registry.execute(PlanTask("m", "state.mutate", {"operation": "watchlist.add", "payload": {"ticker": "nvda"}}), context)
    assert added.ok and "加入关注列表" in added.summary and added.evidence
    removed = registry.execute(PlanTask("m", "state.mutate", {"operation": "watchlist.remove", "payload": {"ticker": "AMD"}}), context)
    assert "不在关注列表" in removed.summary
    alert = registry.execute(PlanTask("m", "state.mutate", {"operation": "alert.add", "payload": {"ticker": "AAPL", "direction": "below", "target_price": 150}}), context)
    assert "#7" in alert.summary and "跌到" in alert.summary
    unalert = registry.execute(PlanTask("m", "state.mutate", {"operation": "alert.remove", "payload": {"alert_id": 7}}), context)
    assert "已取消提醒 #7" in unalert.summary
    assert calls == [("add", "NVDA"), ("remove", "AMD"), ("alert", "AAPL", "below", 150.0), ("unalert", 7)]
    blocked = registry.execute(PlanTask("m", "state.mutate", {"operation": "watchlist.add", "payload": {"ticker": "NVDA"}}), _context())
    assert not blocked.ok and "confirmation" in blocked.errors[0]


def test_executor_enforces_the_wall_clock_budget():
    import time as _time

    catalog = default_catalog()
    registry = CapabilityRegistry(catalog)

    def slow(arguments, context):
        _time.sleep(0.5)
        return ToolEnvelope("research.stock", ResultStatus.COMPLETED, subject="NVDA", evidence=[EvidenceItem("S1", "NVDA", "slow")])

    def fast(arguments, context):
        return ToolEnvelope("account.portfolio", ResultStatus.COMPLETED, subject="portfolio", evidence=[EvidenceItem("F1", "portfolio", "fast")])

    registry.register("research.stock", slow)
    registry.register("account.portfolio", fast)
    plan = ExecutionPlan(
        "q",
        RouteKind.RESEARCH,
        tasks=(
            PlanTask("slow", "research.stock", {"ticker": "NVDA"}),
            PlanTask("fast", "account.portfolio"),
            PlanTask("after", "account.risk", depends_on=("slow",)),
        ),
        budget=BudgetClass.PORTFOLIO,
    )
    context = ExecutionContext("run", NormalizedRequest("q", "q"), BudgetClass.PORTFOLIO, deadline=_time.monotonic() + 0.1)
    outcome = ExecutionEngine(registry).run(plan, context)
    by_capability = {result.capability: result for result in outcome.results}
    assert outcome.stop_reason == "deadline"
    assert by_capability["account.portfolio"].ok
    assert by_capability["research.stock"].status == ResultStatus.FAILED and "timed out" in by_capability["research.stock"].errors[0]
    assert by_capability["account.risk"].status == ResultStatus.SKIPPED
    assert outcome.ledger.ids() == {"F1"}


def test_a_comparison_plan_gets_the_comparison_budget_on_any_route():
    from v2.agent_v2.execution import time_limit

    request = normalize_request("MU和SNDK哪个更值得购买？")
    plan = RulePlanner().plan(request, route(request))
    assert [task.capability for task in plan.tasks] == ["research.compare"] and plan.budget == BudgetClass.COMPARISON and time_limit(plan.budget) >= 240
    # The budget follows the task, not the route: a fast lookup that plans a compare still gets it.
    from v2.agent_v2.models import RouteDecision

    lookup = RulePlanner().plan(request, RouteDecision(RouteKind.FAST_LOOKUP, ("account", "research"), "single-purpose lookup"))
    assert lookup.tasks[0].capability == "research.compare" and lookup.budget == BudgetClass.COMPARISON


def test_telegram_header_names_the_task_the_deadline_cut():
    from v2.agent_v2.interfaces import telegram_format
    from v2.agent_v2.models import ExecutionPlan, ResultStatus, ToolEnvelope

    result = _telegram_result("x[evidence-news-1]。", outcome="clean")
    assert telegram_format.budget_line(result) == ""
    result.stop_reason = "deadline"
    result.plan = ExecutionPlan(objective="q", route=RouteKind.FAST_LOOKUP, budget=BudgetClass.FOCUSED)
    result.results = [
        ToolEnvelope("research.compare", ResultStatus.FAILED, errors=["timed out after 60s wall-clock budget"]),
        ToolEnvelope("account.risk", ResultStatus.SKIPPED, errors=["wall-clock budget exhausted"]),
        ToolEnvelope("web.research", ResultStatus.COMPLETED),
    ]
    assert telegram_format.budget_line(result) == "预算用尽：research.compare 未在 60 秒内完成；account.risk 未开始"
    result.results = []
    assert telegram_format.budget_line(result) == "预算用尽：60 秒预算已耗尽"


def test_orchestrator_surfaces_deadline_as_partial_with_a_stop_reason():
    import time as _time

    catalog = default_catalog()
    registry = CapabilityRegistry(catalog)

    def slow(arguments, context):
        _time.sleep(0.3)
        return ToolEnvelope("research.stock", ResultStatus.COMPLETED, subject="NVDA", evidence=[EvidenceItem("S1", "NVDA", "slow")])

    registry.register("research.stock", slow)
    agent = AgentV2(catalog=catalog, registry=registry, config=AgentV2Config(max_seconds=0.05))
    result = agent.run("分析 NVDA 的风险")
    assert result.status == RunStatus.PARTIAL
    assert result.stop_reason == "deadline"
    assert result.to_dict()["stop_reason"] == "deadline"
    assert "timed out" in result.results[0].errors[0]


def test_async_lab_requests_execute_inline_and_half_finished_lab_result_is_gone():
    assert default_catalog().get("lab.result") is None
    catalog = default_catalog()
    registry = CapabilityRegistry(catalog)
    registry.register("lab.sweep", lambda arguments, context: ToolEnvelope("lab.sweep", ResultStatus.COMPLETED, subject="sp500", summary="sweep done", evidence=[EvidenceItem("L1", "sp500", "sweep done")]))
    result = AgentV2(catalog=catalog, registry=registry).run("对标普全部股票做十年参数扫描")
    assert result.route.kind == RouteKind.ASYNC and result.route.asynchronous
    assert result.status == RunStatus.COMPLETED
    assert result.results[0].capability == "lab.sweep"


def _plan(query: str) -> ExecutionPlan:
    request = normalize_request(query)
    return RulePlanner().plan(request, route(request))


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("最近 CPI", {"macro.release"}),
        ("宏观怎么样，还有最近 CPI", {"macro.overview", "macro.release"}),
        ("巴菲特最新持仓", {"institutional.manager_portfolio"}),
        ("巴菲特买了什么，ARKK 又买了什么", {"institutional.manager_portfolio", "etf.ark_activity"}),
        ("推送阈值是多少", {"state.read"}),
        ("我关注了哪些股票", {"state.read"}),
        ("未来两周谁要发财报", {"account.earnings_schedule"}),
        ("我这周亏的钱今天补回来了吗", {"account.performance"}),
        ("我的当日盈亏和组合风险", {"account.performance", "account.risk", "account.portfolio"}),
        ("TSLA 和 PLTR 哪个逆势更严重", {"market.explain_move"}),
        ("NVDA 涨了吗？资金流呢？", {"market.explain_move", "research.stock"}),
        ("NVDA 和 AMD 谁的财报更好", {"research.compare"}),
        ("AAPL 财报怎么样，另外内部人有没有在卖", {"research.stock"}),
        ("我的组合和 ARKK 有重叠吗", {"account.portfolio", "etf.ark_activity"}),
        ("现在是加仓的好时候吗", {"macro.overview", "account.risk"}),
        ("帮我看看要不要减仓", {"macro.overview", "account.risk", "account.portfolio"}),
        ("CRWD 占仓多少，超没超过集中度阈值", {"account.risk", "state.read", "research.stock"}),
        ("我的仓库里哪只跌的最多?", {"account.portfolio"}),
        ("我的仓库里今天哪只跌的最多?", {"account.portfolio", "market.explain_move"}),
        ("仓库里哪个亏最多", {"account.performance", "account.portfolio"}),
    ],
)
def test_rule_planner_covers_the_capabilities_the_v1_benchmark_needs(query, expected):
    plan = _plan(query)
    assert {task.capability for task in plan.tasks} == expected, [task.capability for task in plan.tasks]


def test_rule_planner_fans_per_ticker_topics_out_over_holdings_and_watchlist():
    plan = _plan("我持仓里有没有内部人在卖")
    assert plan.tasks[0].capability == "account.portfolio"
    template = next(task for task in plan.tasks if task.fan_out)
    assert template.capability == "research.stock" and template.arguments == {"focus": "ownership"}
    assert template.fan_out["from"] == "account-portfolio" and template.depends_on == ("account-portfolio",)
    assert plan.budget == BudgetClass.PORTFOLIO
    watch = _plan("关注列表里那几只最近怎么样")
    assert watch.tasks[0].capability == "state.read" and watch.tasks[0].arguments == {"section": "watchlist"}
    assert watch.tasks[1].capability == "market.explain_move" and watch.tasks[1].fan_out["from"] == "state-watchlist"
    assert _plan("TSLA 什么时候发财报").tasks[0].arguments == {"ticker": "TSLA", "focus": "earnings"}


def test_rule_planner_answers_help_directly_and_asks_for_missing_command_details():
    plan = _plan("你能帮我做什么")
    assert not plan.tasks and "持仓" in plan.direct_answer
    result = AgentV2().run("你能帮我做什么")
    assert result.status == RunStatus.COMPLETED and result.answer == plan.direct_answer
    clarification = AgentV2().run("取消 NVDA 的提醒")
    assert clarification.status == RunStatus.WAITING_CLARIFICATION and "提醒编号" in clarification.answer


def test_executor_expands_fan_out_tasks_from_the_source_result():
    catalog = default_catalog()
    registry = CapabilityRegistry(catalog)
    registry.register("account.portfolio", lambda a, c: ToolEnvelope("account.portfolio", ResultStatus.COMPLETED, subject="portfolio", evidence=[EvidenceItem("P", "portfolio", "holdings")], metadata={"tickers": ["NVDA", "AMD", "CRWD"]}))
    registry.register("research.stock", lambda a, c: ToolEnvelope("research.stock", ResultStatus.COMPLETED, subject=a["ticker"], evidence=[EvidenceItem(f"R-{a['ticker']}", a["ticker"], f"{a['ticker']} {a['focus']}")]))
    registry.register("account.risk", lambda a, c: ToolEnvelope("account.risk", ResultStatus.COMPLETED, subject="portfolio", evidence=[EvidenceItem("K", "portfolio", "risk")]))
    plan = ExecutionPlan(
        "q",
        RouteKind.RESEARCH,
        tasks=(
            PlanTask("holdings", "account.portfolio"),
            PlanTask("each", "research.stock", {"focus": "filings"}, depends_on=("holdings",), fan_out={"from": "holdings", "field": "tickers", "argument": "ticker", "max": 2}),
            PlanTask("after", "account.risk", depends_on=("each",)),
        ),
        budget=BudgetClass.PORTFOLIO,
    )
    outcome = ExecutionEngine(registry).run(plan, ExecutionContext("run", NormalizedRequest("q", "q"), BudgetClass.PORTFOLIO))
    subjects = [result.subject for result in outcome.results]
    # Three holdings under a cap of two: the engine records what it skipped.
    assert subjects == ["portfolio", "each", "NVDA", "AMD", "portfolio"]
    note = outcome.results[1]
    assert note.status == ResultStatus.PARTIAL_DATA and "未覆盖：CRWD" in note.limitations[0]
    assert outcome.ledger.ids() == {"P", "fan-out-coverage-each", "R-NVDA", "R-AMD", "K"}
    empty = ExecutionPlan("q", RouteKind.RESEARCH, tasks=(PlanTask("risk", "account.risk"), PlanTask("each", "research.stock", {"focus": "risk"}, depends_on=("risk",), fan_out={"from": "risk", "argument": "ticker"})), budget=BudgetClass.FOCUSED)
    outcome = ExecutionEngine(registry).run(empty, ExecutionContext("run", NormalizedRequest("q", "q"), BudgetClass.FOCUSED))
    assert outcome.results[1].status == ResultStatus.SKIPPED
    bad = ExecutionPlan("q", RouteKind.RESEARCH, tasks=(PlanTask("each", "research.stock", {}, fan_out={"from": "missing", "argument": "ticker"}),))
    with pytest.raises(PlanValidationError):
        ExecutionEngine(registry).run(bad, ExecutionContext("run", NormalizedRequest("q", "q"), BudgetClass.DIRECT))


def test_llm_planner_accepts_fan_out_tasks_and_adds_the_source_dependency():
    rows = [
        {"id": "t1", "capability": "account.portfolio", "arguments": {}},
        {"id": "t2", "capability": "research.stock", "arguments": {"focus": "filings"}, "fan_out": {"from": "t1", "argument": "ticker"}},
    ]
    llm = ScriptedLLM([LLMResponse(text=json.dumps({"tasks": rows}))])
    request = normalize_request("研究一下我持仓里每只的 SEC 申报")
    plan = StructuredLLMPlanner(llm, default_catalog()).plan(request, route(request))
    assert plan.tasks[1].fan_out == {"from": "t1", "field": "tickers", "argument": "ticker", "max": 8}
    assert plan.tasks[1].depends_on == ("t1",)


def test_ledger_accepts_the_same_fact_from_another_run_but_rejects_a_different_claim():
    from v2.agent_v2.evidence import EvidenceConflictError, EvidenceLedger

    ledger = EvidenceLedger()
    first = EvidenceItem("evidence-1", "NVDA", "Revenue growth is +55.3%.", metric="revenue_growth", value=0.553, producer_run_id="run-a", metadata={"snapshot": "a"})
    ledger.add(first)
    ledger.add(EvidenceItem("evidence-1", "NVDA", "Revenue growth is +55.3%.", metric="revenue_growth", value=0.553, producer_run_id="run-b", metadata={"snapshot": "b"}))
    assert ledger.get("evidence-1").producer_run_id == "run-a"
    reissued = ledger.add(EvidenceItem("evidence-1", "NVDA", "Revenue growth is +12.0%.", metric="revenue_growth", value=0.12, producer_run_id="run-c"))
    assert reissued.id == "evidence-1~run-c" and reissued.metadata["original_evidence_id"] == "evidence-1"
    assert ledger.get("evidence-1").claim == "Revenue growth is +55.3%." and ledger.get("evidence-1~run-c").claim == "Revenue growth is +12.0%."
    assert ledger.reissued == [("evidence-1", "evidence-1~run-c")]
    assert isinstance(EvidenceConflictError(), ValueError)


def test_result_level_citation_caps_count_each_results_own_evidence():
    first = [EvidenceItem("A1", "NVDA", "candidate a", metadata={"claim_role": "candidate_driver"})]
    second = [EvidenceItem("B1", "AMD", "candidate b", metadata={"claim_role": "candidate_driver"})]
    cap = {"max_cited": {"metadata": {"claim_role": "candidate_driver"}, "max": 1, "warning": "过多弱候选线索"}}
    results = [
        ToolEnvelope("market.explain_move", ResultStatus.COMPLETED, subject="NVDA", evidence=first, metadata={"answer_constraints": [cap]}),
        ToolEnvelope("market.explain_move", ResultStatus.COMPLETED, subject="AMD", evidence=second, metadata={"answer_constraints": [cap]}),
    ]
    report = verify_answer("NVDA 可能与线索 a 相关。[A1] AMD 可能与线索 b 相关。[B1]", [*first, *second], answer_mode=AnswerMode.RESEARCH_GROUNDED, results=results)
    assert report.ok, report.warnings


def test_llm_planner_extends_thin_fast_lookup_plans_without_dropping_the_scope_read():
    response = LLMResponse(text='{"tasks":[{"id":"t1","capability":"account.performance","arguments":{"period":"month"}}]}')
    catalog = default_catalog()
    request = normalize_request("我这个月比上个月表现好还是差？")
    llm = ScriptedLLM([response])
    plan = StructuredLLMPlanner(llm, catalog).plan(request, route(request))
    assert llm.calls and [task.capability for task in plan.tasks] == ["account.portfolio", "account.performance"]
    request = normalize_request("我的持仓有哪些？")
    llm = ScriptedLLM([LLMResponse(text=response.text)])
    plan = StructuredLLMPlanner(llm, catalog).plan(request, route(request))
    assert [task.capability for task in plan.tasks] == ["account.portfolio", "account.performance"]
    request = normalize_request("AMD最近表现如何？")
    llm = ScriptedLLM([LLMResponse(text=response.text)])
    plan = StructuredLLMPlanner(llm, catalog).plan(request, route(request))
    assert not llm.calls and plan.tasks[0].capability == "market.performance"


def test_benchmark_fixture_makes_a_failed_card_citeable():
    from v2.agent_v2.eval.benchmark_fixtures import build_benchmark_registry

    registry, _ = build_benchmark_registry()
    agent = AgentV2(catalog=registry.catalog, registry=registry)
    result = agent.run("SMCI 最近有什么 8-K")
    research = next(item for item in result.results if item.capability == "research.stock")
    assert research.status == ResultStatus.PARTIAL_DATA and "timed out" in research.limitations[0]
    assert any(item.metadata.get("citation_kind") == "limitations" for item in result.evidence)
    assert result.verification.ok




def test_legacy_wrap_strips_card_html_and_parses_positions():
    from v2.agent_v2.adapters.legacy import _wrap

    envelope = _wrap("account.portfolio", "portfolio", _PORTFOLIO_CARD)
    assert "<b>" not in envelope.summary and "<code>" not in envelope.evidence[0].claim
    assert envelope.metadata["tickers"] == ["IVV", "BRK.B", "ARM", "MRVL"]
    rows = {row["ticker"]: row for row in envelope.metadata["positions"]}
    assert rows["ARM"]["pl_pct"] == -32.22 and rows["ARM"]["pl"] == -628.0 and rows["ARM"]["pl_pct_text"] == "-32.22%"
    assert rows["IVV"]["pl"] == 1073.0 and rows["IVV"]["market_value"] == 53951.0
    assert envelope.metrics["positions"][0]["ticker"] == "IVV"
    assert envelope.metadata["rankable"][0]["field"] == "pl_pct"


def test_fallback_synthesizer_answers_a_ranking_question_from_the_position_table():
    from v2.agent_v2.adapters.legacy import _wrap

    portfolio = _wrap("account.portfolio", "portfolio", _PORTFOLIO_CARD)
    request = normalize_request("我的仓库里哪只跌的最多?")
    plan = ExecutionPlan(request.text, RouteKind.RESEARCH, tasks=(PlanTask("p", "account.portfolio"),), answer_mode=AnswerMode.TOOL_GROUNDED)
    answer = EvidenceSummarySynthesizer().synthesize(request, plan, [portfolio], portfolio.evidence)
    assert answer == f"按买入以来的浮动盈亏排序，最低的是 ARM（-32.22%），其次是 MRVL（-27.33%）、BRK.B（-0.17%）[{portfolio.evidence[0].id}]。"
    assert "组合价值" not in answer  # the card stays in the evidence list, not the answer
    assert verify_answer(answer, portfolio.evidence, answer_mode=AnswerMode.TOOL_GROUNDED, results=[portfolio]).ok
    winners = EvidenceSummarySynthesizer().synthesize(normalize_request("持仓里哪只赚得最多"), plan, [portfolio], portfolio.evidence)
    assert winners.startswith("按买入以来的浮动盈亏排序，最高的是 IVV（+2.03%）")
    by_size = EvidenceSummarySynthesizer().synthesize(normalize_request("哪只仓位最大"), plan, [portfolio], portfolio.evidence)
    assert by_size.startswith("按市值排序，最高的是 IVV（$53,951）")
    plain = EvidenceSummarySynthesizer().synthesize(normalize_request("看看我的持仓"), plan, [portfolio], portfolio.evidence)
    assert not plain.startswith("按")
    # "仓位" names the size rule, but the direction word belongs to P/L: fall through to it.
    mixed = EvidenceSummarySynthesizer().synthesize(normalize_request("仓位里哪只跌得最多"), plan, [portfolio], portfolio.evidence)
    assert mixed.startswith("按买入以来的浮动盈亏排序，最低的是 ARM（-32.22%）")


def test_executor_orders_a_ranked_fan_out_by_the_source_table_and_discloses_the_cut():
    from v2.agent_v2.adapters.legacy import _wrap

    catalog = default_catalog()
    registry = CapabilityRegistry(catalog)
    registry.register("account.portfolio", lambda a, c: _wrap("account.portfolio", "portfolio", _PORTFOLIO_CARD))
    registry.register("market.explain_move", lambda a, c: ToolEnvelope("market.explain_move", ResultStatus.COMPLETED, subject=a["ticker"], summary=f"{a['ticker']} moved", evidence=[EvidenceItem(f"M-{a['ticker']}", a["ticker"], f"{a['ticker']} moved")]))
    plan = ExecutionPlan(
        "q",
        RouteKind.RESEARCH,
        tasks=(
            PlanTask("holdings", "account.portfolio"),
            PlanTask("each", "market.explain_move", {}, depends_on=("holdings",), fan_out={"from": "holdings", "field": "tickers", "argument": "ticker", "max": 2, "rank": {"field": "positions", "key": "pl_pct", "descending": False}}),
        ),
        budget=BudgetClass.PORTFOLIO,
    )
    outcome = ExecutionEngine(registry).run(plan, ExecutionContext("run", NormalizedRequest("q", "q"), BudgetClass.PORTFOLIO))
    subjects = [result.subject for result in outcome.results]
    assert subjects == ["portfolio", "each", "ARM", "MRVL"]
    note = outcome.results[1]
    assert "按相关性排序后" in note.limitations[0] and "未覆盖：BRK.B, IVV" in note.limitations[0]
    assert note.evidence[0].metadata["citation_kind"] == "limitations"
    # The ranking answer: conclusion, one line per named holding, what was not covered.
    request = normalize_request("我持仓里今天哪只跌得最多")
    answer = EvidenceSummarySynthesizer().synthesize(request, plan, outcome.results, outcome.ledger.items())
    lines = answer.split("\n")
    assert lines[0].startswith("按买入以来的浮动盈亏排序，最低的是 ARM（-32.22%）")
    assert lines[1:3] == ["ARM moved [M-ARM]", "MRVL moved [M-MRVL]"]
    assert lines[-1] == "market.explain_move 未覆盖：BRK.B、IVV [fan-out-coverage-each]。"
    assert "组合价值" not in answer and len(lines) == 4
    assert verify_answer(answer, outcome.ledger.items(), answer_mode=AnswerMode.TOOL_GROUNDED, results=outcome.results).ok
    # A compound question keeps the other results it asked for.
    risk = ToolEnvelope("account.risk", ResultStatus.COMPLETED, subject="portfolio", summary="集中度 54.7%", evidence=[EvidenceItem("K", "portfolio", "集中度 54.7%")])
    compound = EvidenceSummarySynthesizer().synthesize(normalize_request("我持仓里今天哪只跌得最多，组合风险怎么样"), plan, [*outcome.results, risk], [*outcome.ledger.items(), *risk.evidence])
    assert compound.startswith(answer) and compound.endswith("集中度 54.7% [K]")
    bad = ExecutionPlan("q", RouteKind.RESEARCH, tasks=(PlanTask("h", "account.portfolio"), PlanTask("e", "market.explain_move", {}, depends_on=("h",), fan_out={"from": "h", "argument": "ticker", "rank": {"field": "positions"}})))
    with pytest.raises(PlanValidationError):
        ExecutionEngine(registry).run(bad, ExecutionContext("run", NormalizedRequest("q", "q"), BudgetClass.DIRECT))


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("我的仓库里今天哪只跌的最多?", {"field": "positions", "key": "pl_pct", "descending": False}),
        ("我持仓里最近哪只涨得最多", {"field": "positions", "key": "pl_pct", "descending": True}),
        ("我持仓里跌得最狠的那只是什么原因", {"field": "positions", "key": "pl_pct", "descending": False}),
        ("我持仓里每只最近怎么样", None),
    ],
)
def test_rule_planner_ranks_portfolio_fan_out_by_direction(query, expected):
    request = normalize_request(query)
    plan = RulePlanner().plan(request, route(request))
    template = next(task for task in plan.tasks if task.fan_out)
    assert template.fan_out.get("rank") == expected


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("我的仓库里哪只跌的最多?", ["account.portfolio"]),
        ("我持仓里哪只跌得最多", ["account.portfolio"]),
        ("持仓里谁赚得最多", ["account.performance", "account.portfolio"]),
    ],
)
def test_rule_planner_answers_a_portfolio_ranking_from_the_card_alone(query, expected):
    plan = _plan(query)
    assert [task.capability for task in plan.tasks] == expected
    assert not any(task.fan_out for task in plan.tasks)
    llm = ScriptedLLM([LLMResponse(text="{}")])
    request = normalize_request(query)
    assert [task.capability for task in StructuredLLMPlanner(llm, default_catalog()).plan(request, route(request)).tasks] == expected
    assert llm.calls == []  # the rules own it; the model is not consulted


def test_llm_planner_inherits_the_rules_fan_out_rank():
    rows = [
        {"id": "t1", "capability": "account.portfolio", "arguments": {}},
        {"id": "t2", "capability": "market.performance", "arguments": {}, "fan_out": {"from": "t1", "argument": "ticker"}},
    ]
    llm = ScriptedLLM([LLMResponse(text=json.dumps({"tasks": rows}))])
    request = normalize_request("帮我研究一下我持仓里跌得最多的几只，财报和估值怎么样")
    plan = StructuredLLMPlanner(llm, default_catalog()).plan(request, route(request))
    assert len(llm.calls) == 1
    template = next(task for task in plan.tasks if task.fan_out)
    assert template.fan_out["rank"] == {"field": "positions", "key": "pl_pct", "descending": False}


def test_llm_synthesizer_reports_each_verification_attempt_per_thread():
    import threading

    result = _performance_envelope()
    volatility = next(item for item in result.evidence if item.metadata["evidence_scope"] == "volatility")
    llm = ScriptedLLM([LLMResponse(text=f"AMD 波动率为 99%。[{volatility.id}]"), LLMResponse(text=f"AMD 波动率为 98%。[{volatility.id}]")])
    request = normalize_request("AMD最近表现如何？")
    plan = ExecutionPlan(request.text, RouteKind.FAST_LOOKUP, tasks=(PlanTask("p", "market.performance", {"ticker": "AMD"}),), answer_mode=AnswerMode.TOOL_GROUNDED)
    synthesizer = LLMEvidenceSynthesizer(llm)
    synthesizer.synthesize(request, plan, [result], result.evidence)
    diagnostics = synthesizer.diagnostics()
    assert diagnostics["outcome"] == "fallback" and "99%" in diagnostics["draft"]
    assert [attempt["stage"] for attempt in diagnostics["attempts"]] == ["draft", "repair"]
    assert not diagnostics["attempts"][0]["ok"] and diagnostics["attempts"][0]["warnings"]
    seen: dict[str, str] = {}

    def other_thread() -> None:
        seen["outcome"] = synthesizer.last_outcome

    worker = threading.Thread(target=other_thread)
    worker.start()
    worker.join()
    assert seen["outcome"] == ""  # another thread's run never sees this one's diagnostics


def test_agent_result_carries_synthesis_diagnostics():
    registry = CapabilityRegistry(default_catalog())
    registry.register("account.portfolio", lambda a, c: ToolEnvelope("account.portfolio", ResultStatus.COMPLETED, subject="portfolio", summary="holdings", evidence=[EvidenceItem("P", "portfolio", "holdings")]))
    agent = AgentV2(catalog=registry.catalog, registry=registry)
    payload = agent.run("我的持仓").to_dict()
    assert payload["synthesis"] == {"outcome": "deterministic", "draft": "", "attempts": []}


def test_yfinance_price_source_uses_the_dash_share_class_spelling():
    from v2.data.price_source import YFinancePriceSource

    requested: list[str] = []

    class _Ticker:
        def history(self, **kwargs):
            return None

    def factory(symbol: str):
        requested.append(symbol)
        return _Ticker()

    YFinancePriceSource(ticker_factory=factory).get_prices("BRK.B", "2026-01-01", "2026-01-10")
    assert requested == ["BRK-B"]
    assert YFinancePriceSource.yfinance_symbol("nvda") == "NVDA"


def test_repair_instruction_points_at_the_evidence_that_carries_each_number():
    from v2.agent_v2.llm import repair_instruction
    from v2.agent_v2.models import VerificationReport

    peak = EvidenceItem("D-peak", "ARM", "ARM 从 2026-06-18 的高点 439.46 美元到 2026-07-29 的低点 224.89 美元回撤 -48.83%。", value=-0.4883)
    price = EvidenceItem("AT-price", "ARM", "ARM 在 2026-07-29 收于 224.89 美元，较前一交易日 -8.11%。")
    report = VerificationReport(ok=False, ungrounded_numbers=("439.46", "224.89", "12.34"))
    text = repair_instruction(report, [peak, price])
    assert "439.46 见 [D-peak]；224.89 见 [D-peak]、[AT-price]" in text
    assert "以下数字在本轮证据中找不到：12.34。" in text and "439.46" not in text.split("找不到")[1]
    assert repair_instruction(report).count("找不到：439.46、224.89、12.34") == 1  # without evidence, the old wording
    # A sentence that cited the wrong item reports its figures as a warning; those get the same hint.
    nearby = VerificationReport(ok=False, warnings=("引用未支持邻近数字：439.46, 224.89, -48.8（“ARM 从高点 439.46 美元跌到…”）", "行情事实缺少邻近引用：“近 5 日 +12.52%，近 3 月 -18.66%。”"))
    text = repair_instruction(nearby, [peak, price])
    assert "439.46 见 [D-peak]；224.89 见 [D-peak]、[AT-price]；-48.8 见 [D-peak]" in text
    assert "其他问题：行情事实缺少邻近引用：“近 5 日 +12.52%，近 3 月 -18.66%。”。" in text and "其他问题：引用未支持" not in text


def test_attributor_lead_text_keeps_a_quoted_lead_in_one_sentence():
    from v2.agent_v2.agents.move_attributor import lead_text

    lead = lead_text("当日 ARM 大跌主要受芯片股抛售拖累。其 2026 年已累计上涨 235%，市盈率 431 倍。 获利了结压力放大跌幅。")
    assert lead == "当日 ARM 大跌主要受芯片股抛售拖累；其 2026 年已累计上涨 235%，市盈率 431 倍；获利了结压力放大跌幅"
    # A figure-bearing lead quoted inside one cited sentence keeps its citation.
    item = EvidenceItem("lead-1", "ARM", lead, metadata={"claim_role": "candidate_driver"})
    sentence = f"最相关的一条候选线索是“{lead}”，只能作为排查方向[lead-1]。"
    report = verify_answer(sentence, [item], answer_mode=AnswerMode.RESEARCH_GROUNDED, results=[ToolEnvelope("market.attribute_move", ResultStatus.COMPLETED, evidence=[item], metadata={"require_cited_numbers": True})])
    assert report.ok, report


def _drawdown_citation_fixture():
    peak = EvidenceItem("D-peak", "ARM", "ARM 从 2026-06-18 的高点 439.46 美元到 2026-07-29 的低点 224.89 美元回撤 -48.83%。", value=-0.4883)
    price = EvidenceItem("AT-price", "ARM", "ARM 在 2026-07-29 收于 224.89 美元，较前一交易日 -8.11%。", value=-0.0811)
    perf = EvidenceItem("W-ARM", "ARM", "ARM 区间回报：1d +1.03%，5d +12.52%，1m -1.35%，3m -18.66%，1y +89.90%。")
    bench = EvidenceItem("B-SMH", "ARM", "同期基准 SMH 回报：1d +0.10%（ARM 相对 +0.93%），5d +5.33%（ARM 相对 +7.19%）。")
    hidden = EvidenceItem("H-1", "ARM", "内部：ARM 目标价 439.46。", metadata={"citable": False})
    evidence = [peak, price, perf, bench, hidden]
    results = [ToolEnvelope("market.drawdown", ResultStatus.COMPLETED, subject="ARM", evidence=evidence, metadata={"require_cited_numbers": True})]
    return evidence, results


def test_complete_citations_adds_the_one_item_that_carries_a_misattributed_figure():
    from v2.agent_v2.verification import complete_citations

    evidence, results = _drawdown_citation_fixture()
    draft = "ARM 从高点 439.46 美元跌到低点 224.89 美元，回撤 -48.8%[AT-price]。\n近 5 日 +12.52%，跑赢 SMH[B-SMH]。 三只合计 -1,335 美元[AT-price]。"
    completed, notes = complete_citations(draft, evidence, results)
    # The ids go next to the existing citation, before the closing punctuation.
    assert completed.split("\n")[0] == "ARM 从高点 439.46 美元跌到低点 224.89 美元，回撤 -48.8%[AT-price][D-peak]。"
    assert "跑赢 SMH[B-SMH][W-ARM]。" in completed
    # A figure the model computed itself has no carrier and is left for the repair round.
    assert "三只合计 -1,335 美元[AT-price]。" in completed and notes == ["439.46 → [D-peak]", "-48.8 → [D-peak]", "12.52 → [W-ARM]"]
    report = verify_answer(completed, evidence, answer_mode=AnswerMode.RESEARCH_GROUNDED, results=results)
    assert report.warnings == ("引用未支持邻近数字：-1,335（“三只合计 -1,335 美元。”）",)
    # Vague figures and figures several items carry are not completed.
    untouched = "ARM 跌了 3 天，2026 年表现[B-SMH]。 收于 224.89 美元[B-SMH]。"
    assert complete_citations(untouched, evidence, results) == (untouched, [])
    # A stray figure in the sentence does not block the ones that can be placed.
    partial, partial_notes = complete_citations("从高点 439.46 跌到低点 224.89，回撤 -48.8%，成交 9755737183 股[AT-price]。", evidence, results)
    assert partial == "从高点 439.46 跌到低点 224.89，回撤 -48.8%，成交 9755737183 股[AT-price][D-peak]。" and partial_notes == ["439.46 → [D-peak]", "-48.8 → [D-peak]"]
    # A sentence with no citation is completed when every precise figure has one carrier.
    uncited, uncited_notes = complete_citations("近 5 日 +12.52%，近 3 月 -18.66%。", evidence, results)
    assert uncited == "近 5 日 +12.52%，近 3 月 -18.66%[W-ARM]。" and uncited_notes == ["12.52 → [W-ARM]", "-18.66 → [W-ARM]"]
    assert complete_citations("近 5 日 +12.52%，成交 9755737183 股。", evidence, results)[0] == "近 5 日 +12.52%，成交 9755737183 股。"
    # Two carriers that describe the same stock on the same day are as good as one; two on different days are not.
    best = EvidenceItem("U-0827", "NVDA", "NVDA 2026-08-27 单日 +8.74%，收盘 181.60 美元。", metadata={"date": "2026-08-27"})
    same_day = EvidenceItem("AT-0827-price", "NVDA", "NVDA 在 2026-08-27 收于 181.60 美元，较前一交易日 +8.74%。", as_of="2026-08-27")
    other_day = EvidenceItem("U-0310", "NVDA", "NVDA 2026-03-10 单日 +8.74%，收盘 120.00 美元。", metadata={"date": "2026-03-10"})
    agreeing = ToolEnvelope("market.runup", ResultStatus.COMPLETED, evidence=[best, same_day], metadata={"require_cited_numbers": True})
    assert complete_citations("涨幅最大的一天是 +8.74%[W-ARM]。", [best, same_day, evidence[2]], [agreeing]) == ("涨幅最大的一天是 +8.74%[W-ARM][U-0827]。", ["8.74 → [U-0827]"])
    assert complete_citations("涨幅最大的一天是 +8.74%[W-ARM]。", [best, other_day, evidence[2]], [agreeing])[1] == []
    assert complete_citations("", evidence, results) == ("", [])


def test_llm_synthesizer_completes_citations_before_verifying_a_draft():
    evidence, results = _drawdown_citation_fixture()
    request = normalize_request("ARM 为什么跌这么多")
    plan = ExecutionPlan("ARM 为什么跌这么多", RouteKind.RESEARCH, answer_mode=AnswerMode.RESEARCH_GROUNDED)
    llm = ScriptedLLM([LLMResponse(text="ARM 从高点 439.46 美元跌到低点 224.89 美元，回撤 -48.8%[AT-price]。近 5 日 +12.52%[B-SMH]。")])
    synthesizer = LLMEvidenceSynthesizer(llm)
    answer = synthesizer.synthesize(request, plan, results, evidence)
    assert answer == "ARM 从高点 439.46 美元跌到低点 224.89 美元，回撤 -48.8%[AT-price][D-peak]。近 5 日 +12.52%[B-SMH][W-ARM]。"
    assert len(llm.calls) == 1  # no repair round was needed
    diagnostics = synthesizer.diagnostics()
    assert diagnostics["outcome"] == "clean" and diagnostics["citation_completions"] == ["439.46 → [D-peak]", "-48.8 → [D-peak]", "12.52 → [W-ARM]"]
    assert verify_answer(answer, evidence, answer_mode=plan.answer_mode, results=results).ok


def test_move_attributor_searches_the_news_once_before_settling_for_memory():
    from v2.agent_v2.agents.move_attributor import MoveAttributor

    calls: list[str] = []

    def news(query, day):
        calls.append(query)
        return []

    finish = LLMResponse(text='{"action":"finish","reasons":[],"next_steps":[],"note":""}')
    llm = ScriptedLLM([finish, LLMResponse(text='{"action":"news","query":"NVDA stock September 9 2026"}'), finish])
    attributor = MoveAttributor(llm, price_source_factory=lambda: SimpleNamespace(get_prices=_attributor_prices), news=news, filing_reader=None, memory_recall=None, memory_remember=None, sector_for=lambda ticker: "SMH")
    result = attributor.run("ARM", ExecutionContext("run", NormalizedRequest("q", "q"), BudgetClass.PORTFOLIO, allow_web=True), day="2026-07-29", today=date(2026, 9, 9))
    assert calls == ["NVDA stock September 9 2026"] and [step["action"] for step in result.metadata["trace"]] == ["finish_refused", "news", "finish"]
    assert "还没有搜过新闻" in llm.calls[1][-1]["content"]
    # Without consent there is no news to insist on.
    quiet = ScriptedLLM([finish])
    MoveAttributor(quiet, price_source_factory=lambda: SimpleNamespace(get_prices=_attributor_prices), news=news, filing_reader=None, memory_recall=None, memory_remember=None, sector_for=lambda ticker: "SMH").run("ARM", ExecutionContext("run", NormalizedRequest("q", "q"), BudgetClass.PORTFOLIO), day="2026-07-29", today=date(2026, 9, 9))
    assert len(quiet.calls) == 1 and calls == ["NVDA stock September 9 2026"]


def test_move_attributor_reads_a_filing_dated_just_before_the_day_without_being_asked():
    from v2.agent_v2.agents.filing_reader import FilingRef
    from v2.agent_v2.agents.move_attributor import MoveAttributor

    listed: list[tuple[str, str, str]] = []

    class Source:
        def list_filings(self, ticker, since, until):
            listed.append((ticker, since, until))
            return [FilingRef(ticker, "8-K", "2026-07-28", "0001-26-000001", "https://www.sec.gov/x/1/")]

    class Reader:
        source = Source()
        calls = 0

        def run(self, ticker, context, *, around, today):
            Reader.calls += 1
            return ToolEnvelope("filings.read_events", ResultStatus.COMPLETED, subject=ticker, evidence=[EvidenceItem("E-ARM-0728", "ARM", "ARM 2026-07-28：季度营收低于指引区间（8-K 2026-07-28 s1：“Revenue was below the guidance range”）。", metadata={"evidence_scope": "filing_event", "date": "2026-07-28", "quote": "Revenue was below the guidance range", "text": "Revenue was below the guidance range for the quarter."})])

    # The model finishes at once, quoting the filing it was handed; it never asked for filing_events.
    llm = ScriptedLLM([LLMResponse(text=json.dumps({"action": "finish", "reasons": [{"text": "申报显示营收低于指引区间", "confidence": "中", "source": {"kind": "filing", "id": "E-ARM-0728"}, "quote": "Revenue was below the guidance range"}], "next_steps": [], "note": ""}, ensure_ascii=False))])
    attributor = MoveAttributor(llm, price_source_factory=lambda: SimpleNamespace(get_prices=_attributor_prices), news=None, filing_reader=Reader(), memory_recall=None, memory_remember=None, sector_for=lambda ticker: "SMH")
    result = attributor.run("ARM", ExecutionContext("run", NormalizedRequest("q", "q"), BudgetClass.PORTFOLIO), day="2026-07-29", today=date(2026, 9, 9))
    assert listed == [("ARM", "2026-07-26", "2026-07-29")] and Reader.calls == 1 and result.metrics["reader_calls"] == 1
    # The filing result was in front of the model before its first action.
    first_call = llm.calls[0]
    assert first_call[-1]["role"] == "user" and first_call[-1]["content"].startswith("当日或前 3 天内有申报，已先读取。申报阅读者的结果：")
    assert [item.metadata["driver_text"] for item in result.evidence if item.metadata.get("claim_role") == "candidate_driver"] == ["申报显示营收低于指引区间"]
    # No filing in the three days before: nothing is read up front.
    Source.list_filings = lambda self, ticker, since, until: []
    Reader.calls = 0
    quiet = ScriptedLLM([LLMResponse(text='{"action":"finish","reasons":[],"next_steps":[],"note":""}')])
    MoveAttributor(quiet, price_source_factory=lambda: SimpleNamespace(get_prices=_attributor_prices), news=None, filing_reader=Reader(), memory_recall=None, memory_remember=None, sector_for=lambda ticker: "SMH").run("ARM", ExecutionContext("run", NormalizedRequest("q", "q"), BudgetClass.PORTFOLIO), day="2026-07-29", today=date(2026, 9, 9))
    assert Reader.calls == 0 and len(quiet.calls[0]) == 2


def test_todays_move_goes_to_the_attributor_with_intraday_wording_when_the_session_is_open():
    from v2.agent_v2.agents.move_attributor import register_move_attributor
    from v2.agent_v2.interfaces import telegram_format
    from v2.agent_v2.models import sub_agent_summaries

    class Source:
        def list_filings(self, ticker, since, until):
            return []

    remembered = []
    finish = LLMResponse(text=json.dumps({"action": "finish", "reasons": [], "next_steps": [], "note": "盘中无新闻"}, ensure_ascii=False))
    registry = CapabilityRegistry(default_catalog())
    register_market_capabilities(registry, price_source_factory=lambda: SimpleNamespace(get_prices=_attributor_prices), move_provider=lambda ticker: None, now_factory=lambda: datetime(2026, 9, 9, 11, 0, tzinfo=ZoneInfo("America/New_York")))
    open_session = datetime(2026, 9, 9, 11, 0, tzinfo=ZoneInfo("America/New_York"))
    register_move_attributor(registry, ScriptedLLM([finish]), price_source_factory=lambda: SimpleNamespace(get_prices=_attributor_prices), news=None, filing_source=Source(), memory_recall=None, memory_remember=lambda facts, reasons: remembered.append(facts.date) or "x", sector_for=lambda ticker: "SMH", today_factory=lambda: date(2026, 9, 9), now_factory=lambda: open_session)
    result = registry.execute(PlanTask("m", "market.explain_move", {"ticker": "ARM"}), _context())
    assert result.capability == "market.explain_move" and result.metadata["is_intraday"] is True and result.metadata["date"] == "2026-09-09"
    price = next(item for item in result.evidence if item.metadata["evidence_scope"] == "price")
    assert price.claim.startswith("ARM 截至 2026-09-09 11:00 ET 盘中报 ") and price.metadata["constraints"]
    assert "当日未收盘，价格、成交量和归因都以收盘后为准" in result.metadata["narrative"] and "盘中 " in result.metadata["narrative_compact"]
    assert verify_answer(result.metadata["narrative"], result.evidence, answer_mode=AnswerMode.RESEARCH_GROUNDED, results=[result]).ok
    assert remembered == []  # nothing is remembered while the bar is not final
    # The run is visible as a sub-agent with its trace, on every surface.
    (summary,) = sub_agent_summaries([result])
    assert summary["name"] == "move_attributor" and summary["intraday"] is True and summary["stop_reason"] == "finished" and [step["action"] for step in summary["trace"]] == ["finish"]
    telegram = _telegram_result("x[evidence-news-1]。", outcome="clean")
    telegram.results = [result]
    assert telegram_format.agent_lines(telegram) == [f"异动归因 ARM 2026-09-09：1 轮 · {summary['elapsed_ms'] / 1000:.1f}s · 完成（盘中）"]
    # A challenge verdict and a memory decision get their own indented lines.
    result.metadata["agent"]["challenge"] = {"called": True, "source": "model", "objection": "引文只说股价跟随指引下跌，无法解释 8% 的跌幅", "downgraded": True}
    result.metadata["agent"]["memory"] = {"written": False, "conflict": True, "note": "记忆中已有更高置信度的归因（高，2026-09-09写入），本次结论未覆盖"}
    result.metadata["agent"]["reader_runs"] = [{"filings": 0, "sections_read": 0, "events": 0, "rounds": 0, "stop_reason": "no_filings", "elapsed_ms": 0, "trace": []}]
    lines = telegram_format.agent_lines(telegram)
    assert lines[1] == "  · 反方降级：引文只说股价跟随指引下跌，无法解释 8% 的跌幅" and lines[2].startswith("  · 记忆冲突：记忆中已有更高置信度的归因") and lines[3] == "  ↳ 申报阅读：0 轮 · 0.0s · 无申报"
    assert telegram.to_dict()["sub_agents"][0]["label"] == "异动归因"
    # After the close the same capability reads as a completed bar and is remembered.
    closed = datetime(2026, 9, 9, 18, 0, tzinfo=ZoneInfo("America/New_York"))
    register_move_attributor(registry, ScriptedLLM([finish]), price_source_factory=lambda: SimpleNamespace(get_prices=_attributor_prices), news=None, filing_source=Source(), memory_recall=None, memory_remember=lambda facts, reasons: remembered.append(facts.date) or "x", sector_for=lambda ticker: "SMH", today_factory=lambda: date(2026, 9, 9), now_factory=lambda: closed)
    settled = registry.execute(PlanTask("m", "market.explain_move", {"ticker": "ARM"}), _context())
    assert settled.metadata["is_intraday"] is False and "收于" in settled.evidence[0].claim and remembered == ["2026-09-09"]
    # Without a model the V1 explainer stays registered.
    plain = CapabilityRegistry(default_catalog())
    register_market_capabilities(plain, price_source_factory=lambda: None, move_provider=lambda ticker: None, now_factory=lambda: closed)
    register_move_attributor(plain, None, price_source_factory=lambda: None, news=None, filing_source=Source(), memory_recall=None, memory_remember=None)
    assert plain.execute(PlanTask("m", "market.explain_move", {"ticker": "ARM"}), _context()).errors == ["no recent move data"]


def test_news_checker_reports_dated_events_with_quotes_it_located():
    from v2.agent_v2.agents.news_checker import NewsChecker
    from v2.agent_v2.synthesis import web_lines

    page = "Arm Holdings shares slid 8% on Wednesday, July 29, after the company's revenue guidance came in below Wall Street expectations. Analysts had expected stronger smartphone royalty growth."
    searches: list[str] = []

    def search(query, *, days, max_results):
        searches.append(query)
        return [
            {"title": "Arm falls as guidance disappoints", "url": "https://example.com/arm-guidance#top", "content": "Arm shares slid after guidance came in below expectations.", "published_date": "2026-07-29", "raw_content": page},
            {"title": "Arm at 2030: a long-term view", "url": "https://example.com/opinion", "content": "Why Arm could double by 2030.", "published_date": "2026-07-28"},
            {"title": "junk", "url": "ftp://nope", "content": "x"},
        ]

    llm = ScriptedLLM(
        [
            LLMResponse(text='{"action":"search","query":"Arm Holdings stock July 29 2026 falls"}'),
            LLMResponse(text='{"action":"read","ids":["r1"]}'),
            LLMResponse(
                text=json.dumps(
                    {
                        "action": "finish",
                        "events": [
                            {"date": "2026-07-29", "text": "营收指引低于华尔街预期，股价下跌 8%", "source": "r1", "quote": "revenue guidance came in below Wall Street expectations"},
                            {"date": "2026-07-28", "text": "看多到 2030 年的观点", "source": "r2", "quote": "Why Arm could double by 2030"},
                            {"date": "", "text": "没有日期的事件", "source": "r1", "quote": "shares slid 8% on Wednesday"},
                            {"date": "2026-07-29", "text": "编造", "source": "r1", "quote": "takeover rumours swirled"},
                        ],
                        "note": "一篇正文一篇摘要",
                    },
                    ensure_ascii=False,
                )
            ),
        ]
    )
    checker = NewsChecker(llm, search)
    result = checker.run("ARM", _context(), query="ARM 为什么在 7 月 29 日大跌", topic="company_event", recency_days=60, today=date(2026, 9, 9))
    assert result.capability == "web.research" and result.ok and searches == ["Arm Holdings stock July 29 2026 falls"]
    events = [item for item in result.evidence if item.metadata.get("evidence_type") == "news_event"]
    assert [(item.as_of, item.source_url, item.metadata["read"]) for item in events] == [("2026-07-29", "https://example.com/arm-guidance", True), ("2026-07-28", "https://example.com/opinion", False)]
    # A comparison names several stocks: the task, subject and label carry all of them, claims are not prefixed with one.
    llm.calls.clear()
    llm.responses = list(llm.responses) if hasattr(llm, "responses") else llm.responses
    both = NewsChecker(ScriptedLLM([LLMResponse(text='{"action":"search","query":"Micron SanDisk"}'), LLMResponse(text='{"action":"finish","events":[{"date":"2026-07-29","text":"指引低于预期","source":"r1","quote":"revenue guidance came in below Wall Street expectations"}],"note":""}')]), search)
    pair = both.run("MU,SNDK", _context(), query="MU 和 SNDK 哪个更值得买", topic="company_event", recency_days=60, today=date(2026, 9, 9))
    assert pair.subject == "MU、SNDK" and pair.metadata["agent"]["subject"].startswith("MU、SNDK ")
    assert [item.claim.startswith("2026-07-29：") for item in pair.evidence if item.metadata.get("evidence_type") == "news_event"] == [True]
    assert events[0].claim == "ARM 2026-07-29：营收指引低于华尔街预期，股价下跌 8%（新闻：“revenue guidance came in below Wall Street expectations”）。"
    assert result.metrics["events"] == 2 and result.metrics["reads"] == 1 and "2 条事件没有日期或引文与正文不符，已丢弃" in result.limitations[0]
    assert verify_answer(result.metadata["narrative"], result.evidence, answer_mode=AnswerMode.RESEARCH_GROUNDED, results=[result]).ok
    assert result.metadata["agent"]["name"] == "news_checker" and [step["action"] for step in result.metadata["trace"]] == ["search", "read", "finish"]
    assert web_lines(result, "2026-07-01").startswith("- Arm falls as guidance disappoints（2026-07-29）：ARM 2026-07-29：营收指引低于华尔街预期")
    # A broad question: one search is refused once, then a second angle is required before finishing.
    two_angles = ScriptedLLM([
        LLMResponse(text='{"action":"search","query":"Arm news September 2026"}'),
        LLMResponse(text='{"action":"finish","events":[],"note":"够了"}'),
        LLMResponse(text='{"action":"search","query":"Arm insider selling September 2026"}'),
        LLMResponse(text='{"action":"finish","events":[],"note":"两个角度都搜了"}'),
    ])
    broad = NewsChecker(two_angles, search).run("ARM", _context(), query="ARM 最近有什么新闻", today=date(2026, 9, 9), min_searches=2)
    assert broad.metrics["searches"] == 2 and [step["action"] for step in broad.metadata["trace"]] == ["search", "finish_refused", "search", "finish"]
    assert "请换一个角度" in two_angles.calls[2][-1]["content"]
    # No events at all: a limitations item says so and the envelope is partial, never empty.
    silent = NewsChecker(ScriptedLLM([LLMResponse(text='{"action":"finish","events":[],"note":"没有找到"}')]), search)
    none = silent.run("ARM", _context(), query="q", today=date(2026, 9, 9))
    assert none.status == ResultStatus.PARTIAL_DATA and none.evidence[0].metadata["citation_kind"] == "limitations" and "未找到可核实、带日期的事件" in none.evidence[0].claim


def test_live_registry_puts_the_news_checker_behind_web_research_when_a_model_is_present():
    from v2.agent_v2.runtime import build_live_registry

    class Provider:
        def search(self, query, *, days, max_results):
            return [{"title": "t", "url": "https://example.com/a", "content": "Arm shares slid after guidance came in below expectations on July 29.", "published_date": "2026-07-29"}]

    class Port:
        provider = Provider()

        def search(self, query, *, topic, ticker="", recency_days=30, run_id=""):
            raise AssertionError("the one-shot adapter must not be used when a model is present")

    llm = ScriptedLLM([LLMResponse(text='{"action":"search","query":"Arm July 29"}'), LLMResponse(text=json.dumps({"action": "finish", "events": [{"date": "2026-07-29", "text": "指引不及预期", "source": "r1", "quote": "guidance came in below expectations on July 29"}]}, ensure_ascii=False))])
    registry = build_live_registry(default_catalog(), lab=None, web_search=Port(), llm=llm)
    consenting = ExecutionContext("run", NormalizedRequest("q", "q"), BudgetClass.STANDARD, allow_web=True)
    result = registry.execute(PlanTask("w", "web.research", {"query": "why did ARM fall", "topic": "company_event", "ticker": "ARM", "recency_days": 60}), consenting)
    assert result.ok and result.metadata["agent"]["name"] == "news_checker" and result.metadata["dates"] == ["2026-07-29"]
    # No model: the snippet adapter answers, as before.
    class SnippetPort(Port):
        def search(self, query, *, topic, ticker="", recency_days=30, run_id=""):
            return ToolEnvelope("web.research", ResultStatus.COMPLETED, subject=ticker, evidence=[EvidenceItem("W", ticker, "snippet")])

    plain = build_live_registry(default_catalog(), lab=None, web_search=SnippetPort(), llm=None)
    assert plain.execute(PlanTask("w", "web.research", {"query": "q", "topic": "general", "ticker": "ARM"}), consenting).evidence[0].id == "W"


def test_a_news_question_plans_web_filings_and_memory_under_a_real_budget():
    import re

    from v2.agent_v2.planning import _budget

    for text in ("ARM最近有什么新闻？", "NVDA 最近有什么消息", "英伟达有什么新闻", "ARM 最近有什么动态"):
        request = normalize_request(text, allow_web=True)
        plan = RulePlanner().plan(request, route(request))
        assert [task.capability for task in plan.tasks] == ["web.research", "filings.recent", "market.anomaly_history"], text
        web = plan.tasks[0]
        assert web.arguments["ticker"] in {"ARM", "NVDA"} and web.arguments["topic"] == "company_event" and web.arguments["recency_days"] == 14 and web.arguments["min_searches"] == 2 and not web.required
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", plan.tasks[1].arguments["since"]) and plan.tasks[1].arguments["forms"] == ["8-K", "6-K", "4", "424B5"] and plan.tasks[2].arguments["lookback_days"] == 30
        assert plan.budget == BudgetClass.STANDARD and plan.assumptions[0].startswith("news: ") and "网页已授权并已搜索" in plan.assumptions[0]
        # The model planner leaves it to the rules.
        llm = ScriptedLLM([LLMResponse(text="{}")])
        assert [task.capability for task in StructuredLLMPlanner(llm, default_catalog()).plan(request, route(request)).tasks][0] == "web.research" and llm.calls == []
    without = normalize_request("ARM最近有什么新闻？")
    assert "网页未授权" in RulePlanner().plan(without, route(without)).assumptions[0]
    # "Why did it move" still wins over the news wording.
    why = normalize_request("NVDA 今天为什么跌，有什么消息")
    assert RulePlanner().plan(why, route(why)).tasks[0].capability == "market.explain_move"
    # A lone research-engine task or sub-agent never gets the 30-second lookup budget.
    assert _budget([PlanTask("r", "research.stock", {"ticker": "ARM", "focus": "catalysts"})]) == BudgetClass.STANDARD  # a cold research run overran 60 s
    assert _budget([PlanTask("m", "market.explain_move", {"ticker": "ARM"})]) == BudgetClass.FOCUSED
    assert _budget([PlanTask("p", "account.portfolio")]) == BudgetClass.DIRECT


def test_web_fallback_gets_a_grace_when_the_internal_step_spent_the_budget():
    import time

    from v2.agent_v2.orchestrator import WEB_FALLBACK_GRACE_SECONDS

    seen: dict[str, float] = {}
    registry = CapabilityRegistry(default_catalog())

    def exhausted(arguments, context):
        object.__setattr__(context, "deadline", time.monotonic() - 1)  # the engine's clock has run out
        return ToolEnvelope("research.stock", ResultStatus.FAILED, subject="ARM", errors=["timed out after 60s wall-clock budget"])

    def web(arguments, context):
        seen["remaining"] = context.remaining_seconds()
        return ToolEnvelope("web.research", ResultStatus.COMPLETED, subject="ARM", evidence=[EvidenceItem("W-1", "ARM", "ARM 2026-09-03：宣布新产品（新闻：“Arm announced a new product on September 3”）。", as_of="2026-09-03", source_url="https://example.com/a")])

    registry.register("research.stock", exhausted)
    registry.register("web.research", web)
    agent = AgentV2(catalog=default_catalog(), registry=registry, config=AgentV2Config(enable_web_fallback=True))
    result = agent.run("分析 ARM 的估值", allow_web=True)
    assert 40 <= seen["remaining"] <= WEB_FALLBACK_GRACE_SECONDS and "W-1" in [item.id for item in result.evidence]
    assert any("grace" in note for note in result.plan.assumptions) and result.plan.tasks[-1].capability == "web.research"
    # With time to spare the grace is not applied.
    seen.clear()
    registry.register("research.stock", lambda arguments, context: ToolEnvelope("research.stock", ResultStatus.FAILED, subject="ARM", errors=["boom"]))
    result = agent.run("分析 ARM 的估值", allow_web=True)
    assert seen["remaining"] > WEB_FALLBACK_GRACE_SECONDS and not any("grace" in note for note in result.plan.assumptions)


def test_sub_agent_runs_are_ledgered_and_reported(tmp_path, monkeypatch):
    from v2.agent_v2.eval import subagent_report
    from v2.agent_v2.eval.subagent_ledger import aggregate, read_rows, record_runs, render, rows_for

    ledger = tmp_path / "subagents.jsonl"
    monkeypatch.setenv("AGENT_V2_SUBAGENT_LEDGER", str(ledger))
    attributed = ToolEnvelope("market.attribute_move", ResultStatus.COMPLETED, subject="ARM", as_of="2026-07-29", metadata={
        "agent": {"name": "move_attributor", "label": "异动归因", "subject": "ARM 2026-07-29", "rounds": 4, "llm_calls": 5, "elapsed_ms": 12300, "seconds_allowed": 120, "stop_reason": "finished", "calls": {"news": 2, "filing_events": 1, "memory": 1}, "yield": {"kept": 2, "dropped": 1, "confirmed": 1}, "reader_runs": [{"filings": 2, "sections_read": 4, "events": 1, "rounds": 3, "stop_reason": "finished", "elapsed_ms": 6100, "trace": []}]},
        "trace": [{"round": 1, "action": "news", "detail": "", "ms": 900}],
    })
    checker = ToolEnvelope("web.research", ResultStatus.PARTIAL_DATA, subject="ARM", metadata={"agent": {"name": "news_checker", "label": "新闻核查", "subject": "ARM", "rounds": 6, "llm_calls": 7, "elapsed_ms": 40000, "seconds_allowed": 75, "stop_reason": "rounds", "calls": {"search": 3, "read": 3}, "yield": {"kept": 0, "dropped": 2}}, "trace": []})
    result = _telegram_result("x[evidence-news-1]。", outcome="clean")
    result.results = [attributed, checker]
    result.request = NormalizedRequest("ARM 为什么跌", "ARM 为什么跌", metadata={"channel": "telegram"})
    rows = rows_for(result)
    assert [(row["agent"], row["nested"], row["channel"]) for row in rows] == [("move_attributor", False, "telegram"), ("filing_reader", True, "telegram"), ("news_checker", False, "telegram")]
    assert record_runs(result) == 3 and record_runs(result) == 3 and len(read_rows(ledger)) == 6
    summary = aggregate(read_rows(ledger))
    assert summary["move_attributor"]["runs"] == 2 and summary["move_attributor"]["rounds_avg"] == 4.0 and summary["move_attributor"]["seconds_avg"] == 12.3
    assert summary["move_attributor"]["kept_per_run"] == 2.0 and summary["move_attributor"]["drop_rate"] == 0.333 and summary["move_attributor"]["confirmed"] == 2
    assert summary["news_checker"]["empty_runs"] == 2 and summary["news_checker"]["stop_reasons"] == {"rounds": 2} and summary["news_checker"]["calls"] == {"search": 6, "read": 6}
    assert summary["filing_reader"]["runs"] == 2 and summary["filing_reader"]["kept"] == 2
    text = render(summary, {}, since_days=7)
    assert "| move_attributor | 2 | 4.0 | 12.3 |" in text and "| news_checker | 2 |" in text and "用量账本不可用" in text
    from v2.agent_v2.eval.subagent_ledger import TOKEN_WEIGHTS, cost_label, token_equivalent, usage_totals

    # The standard token equivalent is price-period invariant: uncached input 1, cached input 1/30, output 3.
    assert TOKEN_WEIGHTS == {"input": 1.0, "cached_input": 1.0 / 30.0, "output": 3.0}
    assert token_equivalent(3000, 1500, 200) == 1500 + 50 + 600 and token_equivalent(100, 500, 0) == 100 / 30 and token_equivalent(None, None, None) == 0
    unpriced = {"agent_v2.move_attributor": {"calls": 3, "input_tokens": 3000, "cached_tokens": 1500, "output_tokens": 200, "equivalent": 2150.0, "cost": {}, "unpriced": 3, "unpriced_reasons": {"缺少价格版本": 3}, "failed": 0},
                "agent_v2.synthesizer": {"calls": 2, "input_tokens": 10000, "cached_tokens": 0, "output_tokens": 1000, "equivalent": 13000.0, "cost": {}, "unpriced": 2, "unpriced_reasons": {"缺少价格版本": 2}, "failed": 1}}
    from v2.agent_v2.eval.subagent_ledger import question_count

    assert question_count(read_rows(ledger)) == 1 and question_count([]) == 0
    text = render(summary, unpriced, since_days=None, questions=2)
    # Tokens first; per run uses the sub-agent's ledger runs (2), the synthesizer has no run count; per question divides by the ledger's distinct runs; no money column without a price.
    assert "| 来源 | 模型调用 | 未缓存输入 | 缓存输入 | 输出 | 其中推理 | 标准当量 | 当量/调用 | 当量/次运行 | 当量/问题 | 缓存命中 | 失败 |" in text and "估算成本" not in text and "待定价" not in text
    assert "账本里有 2 个用到子智能体的问题。" in text
    assert "| agent_v2.move_attributor | 3 | 1,500 | 1,500 | 200 | 0% | 2,150 | 717 | 1,075 | 1,075 | 50% | 0 |" in text
    assert "| agent_v2.synthesizer | 2 | 10,000 | 0 | 1,000 | 0% | 13,000 | 6,500 | — | 6,500 | 0% | 1 |" in text
    assert "| 合计 | 5 | 11,500 | 1,500 | 1,200 | 0% | 15,150 | 3,030 | — | 7,575 | 12% | 1 |" in text and "标准当量 = 未缓存输入 × 1 + 缓存输入 × 1/30 + 输出 × 3" in text
    assert "| 合计 | 5 | 11,500 | 1,500 | 1,200 | 0% | 15,150 | 3,030 | — | — | 12% | 1 |" in render(summary, unpriced, since_days=None)
    assert usage_totals(unpriced)["unpriced_reasons"] == {"缺少价格版本": 5}
    priced = {"agent_v2.move_attributor": {**unpriced["agent_v2.move_attributor"], "cost": {"CNY": 0.0123}, "unpriced": 1, "unpriced_reasons": {"缺少价格版本": 1}}}
    assert cost_label(priced["agent_v2.move_attributor"]) == "0.0123 CNY、1 次待定价（缺少价格版本）" and cost_label({"cost": {}, "unpriced": 0}) == "—"
    text = render(summary, priced, since_days=None)
    assert "| 缓存命中 | 失败 | 估算成本 |" in text and "| 200 | 0% | 2,150 | 717 | 1,075 | — | 50% | 0 | 0.0123 CNY、1 次待定价（缺少价格版本） |" in text and "| 合计 | 3 |" in text
    # Per-run usage from the ledger's run ids: exact question counts and the most expensive questions.
    from v2.agent_v2.eval.subagent_ledger import top_questions, usage_by_run, usage_by_source

    run_rows = read_rows(ledger)
    run_id = run_rows[0]["run_id"]
    events = [
        ({"source": "agent_v2.planner", "run_id": run_id, "occurred_at": "2026-09-09T15:00:00+00:00", "usage": {"input_tokens": 1000, "cached_tokens": 0, "output_tokens": 100}, "state": "success"}, None),
        ({"source": "agent_v2.synthesizer", "run_id": run_id, "occurred_at": "2026-09-09T15:00:20+00:00", "usage": {"input_tokens": 3000, "cached_tokens": 1500, "output_tokens": 200}, "state": "success"}, None),
        ({"source": "agent_v2.synthesizer", "run_id": "agent-v2-other", "occurred_at": "2026-09-09T16:00:00+00:00", "usage": {"input_tokens": 500, "cached_tokens": 0, "output_tokens": 50}, "state": "success"}, None),
        ({"source": "agent_v2.synthesizer", "occurred_at": "2026-09-08T16:00:00+00:00", "usage": {"input_tokens": 500, "cached_tokens": 0, "output_tokens": 50}, "state": "success"}, None),
    ]
    runs = usage_by_run(events=events)
    assert set(runs) == {run_id, "agent-v2-other"} and runs[run_id] == {"calls": 2, "equivalent": 3450.0, "sources": {"agent_v2.planner": 1300.0, "agent_v2.synthesizer": 2150.0}, "at": "2026-09-09T15:00:20+00:00"}
    ranked = top_questions(runs, run_rows)
    assert [(entry["question"], entry["equivalent"], entry["top_sources"][0]) for entry in ranked] == [("ARM 为什么跌", 3450.0, ("synthesizer", 2150.0)), ("", 650.0, ("synthesizer", 650.0))]
    by_source = usage_by_source(events=events)
    text = render(summary, by_source, since_days=None, questions=99, runs=runs, rows=run_rows)
    # The synthesizer's per-question figure divides the two tagged calls (2,150 + 650) by 2 runs; the untagged call stays in the totals only.
    assert "用量账本里有 2 个问题（按 run_id 计）。" in text and "| agent_v2.synthesizer | 3 | 2,500 | 1,500 | 300 | 0% | 3,450 | 1,150 | — | 1,400 | 38% | 0 |" in text
    assert "| agent_v2.planner | 1 | 1,000 | 0 | 100 | 0% | 1,300 | 1,300 | — | 650 | 0% | 0 |" in text and "| 合计 | 4 | 3,500 | 1,500 | 400 | 0% | 4,750 | 1,188 | — | 2,050 | 30% | 0 |" in text
    assert "另有 1 次调用没有 run_id" in text and "按用量账本里的 run_id 数算" in text
    assert "| ARM 为什么跌 | 2026-09-09 15:00 | 2 | 3,450 | synthesizer 2,150、planner 1,300 |" in text and "| （无子智能体记录） | 2026-09-09 16:00 | 1 | 650 | synthesizer 650 |" in text
    assert subagent_report.main(["--path", str(ledger), "--json"]) == 0
    # An orchestrator run with a sub-agent envelope writes the ledger by itself; the flag turns it off.
    registry = CapabilityRegistry(default_catalog())
    registry.register("market.explain_move", lambda arguments, context: ToolEnvelope("market.explain_move", ResultStatus.COMPLETED, subject="ARM", evidence=[EvidenceItem("M", "ARM", "ARM 今日 -1.00%")], metadata={"narrative": "ARM 今日 -1.00%[M]。", "agent": {"name": "move_attributor", "rounds": 1, "llm_calls": 1, "elapsed_ms": 10, "stop_reason": "finished", "calls": {}, "yield": {"kept": 0, "dropped": 0}}, "trace": []}))
    before = len(read_rows(ledger))
    AgentV2(catalog=default_catalog(), registry=registry).run("ARM 今天为什么跌")
    assert len(read_rows(ledger)) == before + 1 and read_rows(ledger)[-1]["question"] == "ARM 今天为什么跌"
    AgentV2(catalog=default_catalog(), registry=registry, config=AgentV2Config(record_sub_agents=False)).run("ARM 今天为什么跌")
    assert len(read_rows(ledger)) == before + 1


def test_provider_calls_carry_the_run_id_of_the_question(monkeypatch):
    from v2.usage_context import current_run

    seen: list[str] = []
    registry = CapabilityRegistry(default_catalog())
    registry.register("market.performance", lambda arguments, context: seen.append(current_run("")) or ToolEnvelope("market.performance", ResultStatus.COMPLETED, subject="ARM", evidence=[EvidenceItem("P", "ARM", "ARM 近 30 天 +1.00%")], metadata={"narrative": "ARM 近 30 天 +1.00%[P]。"}))
    result = AgentV2(catalog=default_catalog(), registry=registry).run("ARM 最近30天表现")
    assert seen == [result.run_id] and current_run("") == ""


def test_provider_calls_inside_a_sub_agent_are_attributed_to_it():
    from v2.usage_context import current_source, usage_source

    class Probe(ScriptedLLM):
        sources: list[str] = []

        def complete(self, messages, tools=None):
            Probe.sources.append(current_source("agent.chat"))
            return super().complete(messages, tools)

    from v2.agent_v2.agents.base import BoundedLoop, LoopLimits

    class Echo(BoundedLoop):
        usage_source_name = "agent_v2.probe"

        def handle(self, action, messages):
            return True

    assert current_source("agent.chat") == "agent.chat"
    Echo(Probe([LLMResponse(text='{"action":"finish"}')]), LoopLimits(max_rounds=2)).run("s", "t", finish_prompt="f")
    assert Probe.sources == ["agent_v2.probe"] and current_source("agent.chat") == "agent.chat"
    with usage_source("agent_v2.synthesizer"):
        assert current_source() == "agent_v2.synthesizer"


def test_memory_governance_keeps_a_confirmed_attribution_and_records_drift():
    from v2.agent_v2.agents.move_attributor import govern_memory

    class Stored:
        def __init__(self, **metadata):
            self.metadata = metadata

    high = [{"text": "财报指引不及预期", "confidence": "高"}]
    candidates = [{"text": "板块回调", "confidence": "中"}]
    # Nothing stored: written as version 1 with its best confidence and top reason.
    fresh = govern_memory(None, high, today="2026-09-10")
    assert fresh["write"] and fresh["metadata"] == {"confidence": "高", "confidence_rank": 3, "top_reason": "财报指引不及预期", "version": 1, "written_at": "2026-09-10"} and not fresh["conflict"]
    stored = Stored(**fresh["metadata"])
    # A later run with only candidates does not erase a confirmed driver.
    kept = govern_memory(stored, candidates, today="2026-09-11")
    assert not kept["write"] and "记忆中已有更高置信度的归因" in kept["note"] and kept["conflict"]
    # An equal or better result overwrites, versioned, and a different top reason is recorded as drift.
    changed = govern_memory(stored, [{"text": "出口管制放宽", "confidence": "高"}], today="2026-09-12")
    assert changed["write"] and changed["conflict"] and changed["metadata"]["version"] == 2 and changed["metadata"]["previous_reason"] == "财报指引不及预期"
    assert changed["note"].startswith("与上次归因不同（上次：财报指引不及预期）")
    same = govern_memory(stored, high, today="2026-09-12")
    assert same["write"] and not same["conflict"] and same["note"] == ""
    # The same event reworded is not drift.
    from v2.agent_v2.agents.move_attributor import same_reason

    assert same_reason("Arm 财报虽超预期，但公司指引下一季度智能手机版税收入将下滑，引发市场对核心授权业务前景的担忧，股价承压下跌", "财报营收与 EPS 超指引上限，但公司指引下季智能手机版税收入下滑，引发对核心授权业务的担忧")
    assert same_reason("当日为韩国主导的芯片与AI板块普跌，ARM 作为高估值半导体股被同步抛售，且此前12个月涨幅巨大引发获利了结。", "韩国主导的芯片与 AI 板块普跌、叠加前期大涨后的获利了结")
    assert not same_reason("财报指引不及预期", "美国商务部放开对阿联酋的出口管制")
    reworded = govern_memory(Stored(**{**fresh["metadata"], "top_reason": "公司财报指引不及预期，股价下跌"}), [{"text": "财报指引不及预期", "confidence": "高"}], today="2026-09-12")
    assert reworded["write"] and not reworded["conflict"]
    # Empty results never overwrite anything that exists.
    assert not govern_memory(stored, [], today="2026-09-12")["write"]


def test_attributor_surfaces_the_memory_decision():
    from v2.agent_v2.agents.move_attributor import MoveAttributor
    from v2.agent_v2.models import sub_agent_summaries

    finish = LLMResponse(text=json.dumps({"action": "finish", "reasons": [], "next_steps": [], "note": ""}, ensure_ascii=False))
    decisions = [{"id": "ARM_2026-07-29_retro", "written": False, "note": "记忆中已有更高置信度的归因（高，2026-09-09写入），本次结论未覆盖", "conflict": True}]
    attributor = MoveAttributor(ScriptedLLM([finish]), price_source_factory=lambda: SimpleNamespace(get_prices=_attributor_prices), news=None, filing_reader=None, memory_recall=None, memory_remember=lambda facts, reasons: decisions[0], sector_for=lambda ticker: "SMH")
    result = attributor.run("ARM", ExecutionContext("run", NormalizedRequest("q", "q"), BudgetClass.PORTFOLIO), day="2026-07-29", today=date(2026, 9, 9))
    assert result.metrics["remembered_as"] == "" and "本次结论未覆盖" in " ".join(result.limitations)
    (summary,) = sub_agent_summaries([result])
    assert summary["memory"] == {"written": False, "conflict": True, "note": decisions[0]["note"]}
    # The plain string form (older callers, tests) still works.
    legacy = MoveAttributor(ScriptedLLM([finish]), price_source_factory=lambda: SimpleNamespace(get_prices=_attributor_prices), news=None, filing_reader=None, memory_recall=None, memory_remember=lambda facts, reasons: "ARM_2026-07-29_retro", sector_for=lambda ticker: "SMH")
    assert legacy.run("ARM", ExecutionContext("run", NormalizedRequest("q", "q"), BudgetClass.PORTFOLIO), day="2026-07-29", today=date(2026, 9, 9)).metrics["remembered_as"] == "ARM_2026-07-29_retro"


def test_model_planner_delegation_to_sub_agents_is_capped():
    from v2.agent_v2.llm import MAX_LONG_RUNNING_FAN_OUT, MAX_LONG_RUNNING_TASKS, _cap_long_running

    catalog = default_catalog()
    tasks = (
        PlanTask("t1", "account.portfolio"),
        PlanTask("t2", "market.explain_move", {"ticker": "NVDA"}),
        PlanTask("t3", "market.attribute_move", {"ticker": "NVDA"}, depends_on=("t1",), fan_out={"from": "t1", "field": "tickers", "argument": "ticker", "max": 8}),
        PlanTask("t4", "web.research", {"query": "q", "topic": "general"}),
        PlanTask("t5", "research.stock", {"ticker": "NVDA", "focus": "overview"}, depends_on=("t4",)),
    )
    kept, notes = _cap_long_running(tasks, catalog)
    assert [task.id for task in kept] == ["t1", "t2", "t3"]  # t4 beyond the cap, t5 depended on it
    assert kept[2].fan_out["max"] == MAX_LONG_RUNNING_FAN_OUT and MAX_LONG_RUNNING_TASKS == 2
    assert notes == [f"Planner clamped the fan-out of market.attribute_move to {MAX_LONG_RUNNING_FAN_OUT} (sub-agent cost cap)", "Planner dropped 1 sub-agent task(s) beyond the cap of 2: web.research"]
    # End to end: the model plans per-ticker attribution for a divergence question and the catalog payload flags sub-agents.
    rows = [
        {"id": "p1", "capability": "market.performance", "arguments": {"ticker": "NVDA"}},
        {"id": "p2", "capability": "market.performance", "arguments": {"ticker": "AMD"}},
        {"id": "m1", "capability": "market.explain_move", "arguments": {"ticker": "NVDA"}},
        {"id": "m2", "capability": "market.explain_move", "arguments": {"ticker": "AMD"}},
        {"id": "m3", "capability": "filings.read_events", "arguments": {"ticker": "NVDA", "around": "2026-09-01"}},
    ]
    llm = ScriptedLLM([LLMResponse(text=json.dumps({"objective": "x", "tasks": rows}))])
    request = normalize_request("比较 NVDA 和 AMD 最近为什么走势分化")
    plan = StructuredLLMPlanner(llm, catalog).plan(request, route(request))
    payload = json.loads(llm.calls[0][1]["content"])
    assert payload["maximum_long_running_tasks"] == 2 and any(spec.get("long_running") for spec in payload["capabilities"] if spec["name"] == "market.explain_move")
    assert [task.capability for task in plan.tasks] == ["market.performance", "market.performance", "market.explain_move", "market.explain_move"]
    assert any("dropped 1 sub-agent task" in note for note in plan.assumptions) and plan.budget in {BudgetClass.STANDARD, BudgetClass.COMPARISON}


def test_challenger_downgrades_a_driver_the_sector_explains_or_the_model_refutes():
    from v2.agent_v2.agents.move_attributor import MoveAttributor
    from v2.agent_v2.models import sub_agent_summaries

    def sector_prices(ticker, start, end):
        rows = []
        close = 300.0 if ticker == "ARM" else 100.0
        for index in range(260):
            day = date(2026, 1, 5) + timedelta(days=index)
            if day.weekday() >= 5 or day.isoformat() > str(end):
                continue
            close *= 0.92 if day == date(2026, 7, 29) else 1.001  # the sector fell as much as the stock
            rows.append(_Bar(day.isoformat(), round(close, 2)))
        return rows

    def news(query, day):
        return [{"title": "Arm falls as guidance disappoints", "url": "https://example.com/arm-guidance", "content": "Arm Holdings shares slid 8% after the company's revenue guidance came in below Wall Street expectations.", "published_date": "2026-07-29"}]

    confirmed_finish = LLMResponse(text=json.dumps({"action": "finish", "reasons": [{"text": "营收指引低于华尔街预期", "confidence": "高", "source": {"kind": "news", "url": "https://example.com/arm-guidance"}, "quote": "revenue guidance came in below Wall Street expectations"}], "next_steps": [], "note": ""}, ensure_ascii=False))
    context = ExecutionContext("run", NormalizedRequest("q", "q"), BudgetClass.PORTFOLIO, allow_web=True)
    # The sector fell the same way and by the same amount: the driver is a contributor at most, no model call needed.
    same_sector = MoveAttributor(ScriptedLLM([LLMResponse(text='{"action":"news","query":"arm"}'), confirmed_finish]), price_source_factory=lambda: SimpleNamespace(get_prices=sector_prices), news=news, filing_reader=None, memory_recall=None, memory_remember=None, sector_for=lambda ticker: "SMH")
    result = same_sector.run("ARM", context, day="2026-07-29", today=date(2026, 9, 9))
    roles = [item.metadata["claim_role"] for item in result.evidence if item.metadata.get("claim_role") in {"confirmed_driver", "candidate_driver"}]
    assert roles == ["candidate_driver"] and result.metrics["confirmed_driver_count"] == 0
    assert "反方意见：当日行业基准 SMH 同向 -8.00%" in " ".join(result.limitations) and "（已降为中置信度）" in " ".join(result.limitations)
    (summary,) = sub_agent_summaries([result])
    assert summary["challenge"]["source"] == "sector" and summary["challenge"]["downgraded"] is True
    # The stock fell far more than the sector: one adversarial model call; a specific objection downgrades.
    refuting = ScriptedLLM([LLMResponse(text='{"action":"news","query":"arm"}'), confirmed_finish, LLMResponse(text='{"objection":"引文只说股价跟随指引下跌，没有说明指引下调幅度，无法解释 8% 的跌幅","downgrade":true}')])
    result = MoveAttributor(refuting, price_source_factory=lambda: SimpleNamespace(get_prices=_attributor_prices), news=news, filing_reader=None, memory_recall=None, memory_remember=None, sector_for=lambda ticker: "SMH").run("ARM", context, day="2026-07-29", today=date(2026, 9, 9))
    assert result.metrics["confirmed_driver_count"] == 0 and refuting.calls[-1][0]["content"].startswith("你是异动归因的反方") and [step["action"] for step in result.metadata["trace"]][-1] == "challenge"
    assert sub_agent_summaries([result])[0]["challenge"] == {"called": True, "source": "model", "objection": "引文只说股价跟随指引下跌，没有说明指引下调幅度，无法解释 8% 的跌幅", "downgraded": True}
    # No objection: the driver stands, and the challenge is still on record.
    agreeing = ScriptedLLM([LLMResponse(text='{"action":"news","query":"arm"}'), confirmed_finish, LLMResponse(text='{"objection":"","downgrade":false}')])
    result = MoveAttributor(agreeing, price_source_factory=lambda: SimpleNamespace(get_prices=_attributor_prices), news=news, filing_reader=None, memory_recall=None, memory_remember=None, sector_for=lambda ticker: "SMH").run("ARM", context, day="2026-07-29", today=date(2026, 9, 9))
    assert result.metrics["confirmed_driver_count"] == 1 and "反方意见" not in " ".join(result.limitations) and sub_agent_summaries([result])[0]["challenge"]["downgraded"] is False


def test_debater_objections_must_cite_the_runs_own_evidence():
    from v2.agent_v2.agents.debater import Debater
    from v2.agent_v2.interfaces import telegram_format
    from v2.agent_v2.models import sub_agent_summaries

    evidence = [
        EvidenceItem("R-growth", "NVDA", "Revenue growth is +83.4% on the latest available basis."),
        EvidenceItem("R-valuation", "NVDA", "Forward P/E 45.2; EV/Sales 28.1, top decile of the sector."),
        EvidenceItem("R-hidden", "NVDA", "internal score", metadata={"citable": False}),
    ]
    verdict = {"stance": "回答偏多", "objections": [
        {"claim": "估值合理", "objection": "前瞻市盈率 45 倍、EV/Sales 处于板块前十分位，回答没有把估值风险计入", "evidence_id": "R-valuation"},
        {"claim": "增长可持续", "objection": "没有证据", "evidence_id": "R-nowhere"},
        {"claim": "评分高", "objection": "内部评分不可引用", "evidence_id": "R-hidden"},
    ], "note": "回答只讲增长不讲估值"}
    llm = ScriptedLLM([LLMResponse(text=json.dumps(verdict, ensure_ascii=False))])
    context = ExecutionContext("run", NormalizedRequest("q", "q"), BudgetClass.STANDARD)
    result = Debater(llm).run("分析 NVDA 的估值", "NVDA 增长强劲，估值合理[R-growth]。", evidence, context, subject="NVDA")
    assert result.capability == "debate.challenge" and result.ok and result.metrics == {"objections": 1, "dropped": 2, "elapsed_ms": result.metrics["elapsed_ms"], "stop_reason": "finished"}
    assert result.metadata["objections"] == [{"claim": "估值合理", "objection": "前瞻市盈率 45 倍、EV/Sales 处于板块前十分位，回答没有把估值风险计入", "evidence_id": "R-valuation"}]
    payload = json.loads(llm.calls[0][-1]["content"])
    assert [row["id"] for row in payload["evidence"]] == ["R-growth", "R-valuation"]  # the uncitable item is not offered
    (summary,) = sub_agent_summaries([result])
    assert summary["name"] == "debater" and summary["stance"] == "回答偏多" and summary["notes"] == ["前瞻市盈率 45 倍、EV/Sales 处于板块前十分位，回答没有把估值风险计入（引 [R-valuation]）"]
    telegram = _telegram_result("x[evidence-news-1]。", outcome="clean")
    telegram.results = [result]
    assert telegram_format.agent_lines(telegram) == [f"反方 NVDA：1 轮 · {result.metrics['elapsed_ms'] / 1000:.1f}s · 反对 1 · 完成", "  · 前瞻市盈率 45 倍、EV/Sales 处于板块前十分位，回答没有把估值风险计入（引 [R-valuation]）"]
    # A long note is cut before its citation, never through it, so the bridge can still number the id.
    long_note = "同一评分体系里技术分项仅 20 分、供应链覆盖度 14/100，" * 4 + "不能据此判定明显偏低（引 [evidence-research-limitations-4]）"
    line = telegram_format._note_line(long_note, 110)
    assert line.endswith("…（引 [evidence-research-limitations-4]）") and len(line) <= 110 + len("（引 [evidence-research-limitations-4]）")
    assert telegram_format._note_line("短说明", 110) == "短说明"
    # Nothing to object to: the record says so, and the answer is untouched.
    quiet = Debater(ScriptedLLM([LLMResponse(text='{"stance":"回答中性","objections":[],"note":"结论与证据一致"}')])).run("q", "答[R-growth]。", evidence, context)
    assert quiet.metadata["agent"]["notes"] == ["未找到证据支持的反对意见：结论与证据一致"] and quiet.metrics["objections"] == 0
    # No model: no call, a partial display record.
    assert Debater(None).run("q", "答", evidence, context).metrics["stop_reason"] == "no_model"


def test_orchestrator_runs_the_debate_on_research_answers_only():
    from v2.agent_v2.synthesis import EvidenceSummarySynthesizer

    registry = CapabilityRegistry(default_catalog())
    registry.register("research.stock", lambda arguments, context: ToolEnvelope("research.stock", ResultStatus.COMPLETED, subject="NVDA", summary="NVDA revenue growth was 10%.", evidence=[EvidenceItem("E1", "NVDA", "NVDA revenue growth was 10%.")]))
    verdict = LLMResponse(text='{"stance":"回答偏多","objections":[{"claim":"增长强","objection":"10% 的增长在证据里只是最新一期，不能外推","evidence_id":"E1"}],"note":""}')

    class Synth(EvidenceSummarySynthesizer):
        def __init__(self, llm):
            self.llm = llm

    llm = ScriptedLLM([verdict])
    agent = AgentV2(catalog=default_catalog(), registry=registry, synthesizer=Synth(llm), config=AgentV2Config(record_sub_agents=False))
    result = agent.run("分析 NVDA 的增长")
    debate = [envelope for envelope in result.results if envelope.capability == "debate.challenge"]
    assert len(debate) == 1 and debate[0].metadata["objections"][0]["evidence_id"] == "E1" and len(llm.calls) == 1
    assert "反对" not in result.answer and result.verification.ok and result.to_dict()["sub_agents"][0]["name"] == "debater"
    # Off by configuration, and never on a lookup.
    quiet = ScriptedLLM([verdict])
    off = AgentV2(catalog=default_catalog(), registry=registry, synthesizer=Synth(quiet), config=AgentV2Config(record_sub_agents=False, debate=False)).run("分析 NVDA 的增长")
    assert not [envelope for envelope in off.results if envelope.capability == "debate.challenge"] and quiet.calls == []
    lookup = AgentV2(catalog=default_catalog(), registry=_framed_registry(), synthesizer=Synth(quiet), config=AgentV2Config(record_sub_agents=False)).run("我的仓库里哪只跌的最多?")
    assert not [envelope for envelope in lookup.results if envelope.capability == "debate.challenge"] and quiet.calls == []


def test_intent_drives_routing_and_planning_with_the_model_the_labels_or_the_default(tmp_path, monkeypatch):
    from v2.agent_v2 import intent as intent_mod
    from v2.agent_v2.eval import intent_report
    from v2.agent_v2.intent import Intent, IntentClassifier, RecordedIntents, default_intent, parse_intent, read_decisions, record_decision, decision_row, render, resolve_intent, summarize

    # Off-vocabulary values are dropped, an unknown kind is an error, commands are typed.
    parsed = parse_intent({"kind": "Research", "scope": "yesterday", "direction": "down", "wants": ["valuation", "moon", "compare", "compare"], "tickers": ["mu", "sndk", "not a ticker"], "portfolio_scope": "yes", "confidence": 1.7, "command": {"operation": "alert.add", "ticker": "nvda", "direction": "below", "price": "150"}, "managers": ["buffett", "nobody"], "ark_etfs": ["arkk", "spy"], "window": "1m", "focus": ["valuation", "x"], "lab": "backtest", "strategy": "pead", "periods": ["week", "year"], "rank": "low"})
    assert parsed.kind == "research" and parsed.scope == "none" and parsed.wants == ("valuation", "compare") and parsed.tickers == ("MU", "SNDK") and parsed.portfolio_scope and parsed.confidence == 1.0
    assert parsed.command == {"operation": "alert.add", "ticker": "NVDA", "direction": "below", "price": 150.0} and parsed.managers == ("buffett",) and parsed.ark_etfs == ("ARKK",) and parsed.window == "1m" and parsed.focus == ("valuation",) and parsed.lab == "backtest" and parsed.strategy == "pead" and parsed.periods == ("week",) and parsed.rank == "low"
    with pytest.raises(ValueError):
        parse_intent({"kind": "chat"})

    # The model classifies through a function tool; JSON text is accepted; prose is a failure that falls back.
    from v2.agent_common.llm import ToolCall

    llm = ScriptedLLM([
        LLMResponse(tool_calls=[ToolCall(id="c1", name="classify", arguments={"kind": "research", "scope": "today", "direction": "down", "wants": ["attribution", "news"], "tickers": ["AMD"], "portfolio_scope": False, "confidence": 0.9}, raw_arguments="{}")]),
        LLMResponse(text='{"kind":"knowledge","scope":"none","direction":"none","wants":["valuation"],"tickers":[],"portfolio_scope":false,"confidence":0.8}'),
        LLMResponse(text="I think it is research."),
    ])
    classifier = IntentClassifier(llm)
    english = normalize_request("why did AMD drop today")
    intent = classifier.classify(english)
    assert intent.kind == "research" and intent.scope == "today" and intent.wants == ("attribution", "news") and intent.tickers == ("AMD",) and intent.source == "model"
    assert llm.calls[0] and json.loads(llm.calls[0][1]["content"])["text"] == "why did AMD drop today"
    assert classifier.classify(normalize_request("市盈率和市销率有什么区别？")).kind == "knowledge"
    assert classifier.classify(normalize_request("x")) is None and IntentClassifier(None).classify(english) is None

    # Resolution order: the model, then the recorded label for the exact text, then the default.
    recorded = RecordedIntents(tmp_path / "labels.json")
    (tmp_path / "labels.json").write_text(json.dumps({"ARM 最近有哪些新闻？": {"intent": {"kind": "lookup", "scope": "recent", "direction": "none", "wants": ["news"], "tickers": ["ARM"], "portfolio_scope": False}}}), encoding="utf-8")
    assert resolve_intent(normalize_request("ARM 最近有哪些新闻？"), recorded=recorded).source == "recorded"
    fallback = resolve_intent(normalize_request("ARM 一句没人标过的话"), recorded=recorded)
    assert fallback.source == "default" and fallback.kind == "research" and fallback.wants == ("overview",) and fallback.tickers == ("ARM",)
    assert default_intent(normalize_request("没有代码的话")).kind == "lookup"

    # The route and the plan follow the intent, not the wording: an English question routes as research and plans today's attribution.
    decision = route(english, intent=intent)
    assert decision.kind == RouteKind.RESEARCH and decision.intent is intent
    plan = RulePlanner().plan(english, decision)
    assert [task.capability for task in plan.tasks] == ["market.explain_move"] and plan.answer_mode == AnswerMode.RESEARCH_GROUNDED
    knowledge = route(normalize_request("市盈率和市销率有什么区别？"), intent=Intent(kind="knowledge", wants=("valuation",)))
    assert knowledge.kind == RouteKind.GENERAL_KNOWLEDGE
    watch = Intent(kind="lookup", scope="recent", wants=("watchlist", "ranking", "performance"), watchlist_scope=True, source="model")
    plan = RulePlanner().plan(normalize_request("我关注的股票里有没有最近在放量的？"), route(normalize_request("我关注的股票里有没有最近在放量的？"), intent=watch))
    assert [(task.capability, bool(task.fan_out)) for task in plan.tasks] == [("state.read", False), ("market.performance", True)]  # volume is a performance look, not an attribution
    briefing = Intent(kind="lookup", scope="today", wants=("briefing", "macro"), source="model")
    plan = RulePlanner().plan(normalize_request("美股今天有啥注意的？"), route(normalize_request("美股今天有啥注意的？"), intent=briefing))
    assert [task.capability for task in plan.tasks] == ["macro.overview", "account.earnings_schedule", "account.risk", "state.read"]
    command = Intent(kind="command", command={"operation": "alert.add", "ticker": "NVDA", "direction": "below", "price": 150.0}, tickers=("NVDA",), source="model")
    plan = RulePlanner().plan(normalize_request("NVDA 跌到 150 提醒我"), route(normalize_request("NVDA 跌到 150 提醒我"), intent=command))
    assert plan.requires_confirmation and plan.tasks[0].arguments == {"operation": "alert.add", "payload": {"ticker": "NVDA", "direction": "below", "target_price": 150.0}}
    default_plan = RulePlanner().plan(normalize_request("ARM 一句没人标过的话"), route(normalize_request("ARM 一句没人标过的话"), intent=fallback))
    assert [task.capability for task in default_plan.tasks] == ["research.stock"] and any(note.startswith("intent: 未分类") for note in default_plan.assumptions)

    # The decision ledger and its report.
    ledger = tmp_path / "intents.jsonl"
    monkeypatch.setenv("AGENT_V2_INTENT_LEDGER", str(ledger))
    record_decision(decision_row(english, intent, plan, run_id="run-1", elapsed_ms=900, channel="telegram"))
    record_decision(decision_row(normalize_request("ARM 一句没人标过的话"), fallback, default_plan, run_id="run-2"))
    unsure = Intent(kind="lookup", wants=("overview",), tickers=("ARM",), confidence=0.4, source="model")
    record_decision(decision_row(normalize_request("ARM这波是不是该跑了？"), unsure, default_plan, run_id="run-3"))
    rows = read_decisions(ledger)
    summary = summarize(rows)
    assert summary["rows"] == 3 and summary["sources"] == {"model": 2, "default": 1} and summary["kinds"]["research"] == 2 and [entry["text"] for entry in summary["unsure"]] == ["ARM这波是不是该跑了？"] and [entry["text"] for entry in summary["unclassified"]] == ["ARM 一句没人标过的话"]
    text = render(summary, since_days=1)
    assert "来源：default 1、model 2" in text and "| ARM这波是不是该跑了？ | 0.40 |" in text and "| ARM 一句没人标过的话 |" in text
    assert intent_report.main(["--path", str(ledger), "--json"]) == 0

    # The orchestrator uses its classifier for the route and ledgers the decision when configured.
    registry = CapabilityRegistry(default_catalog())
    registry.register("market.explain_move", lambda arguments, context: ToolEnvelope("market.explain_move", ResultStatus.COMPLETED, subject="AMD", evidence=[EvidenceItem("M", "AMD", "AMD 今日 -1.00%")], metadata={"narrative": "AMD 今日 -1.00%[M]。"}))
    live = IntentClassifier(ScriptedLLM([LLMResponse(text='{"kind":"research","scope":"today","direction":"down","wants":["attribution"],"tickers":["AMD"],"portfolio_scope":false,"confidence":0.9}')]))
    before = len(read_decisions(ledger))
    result = AgentV2(catalog=default_catalog(), registry=registry, classifier=live, config=AgentV2Config(record_intents=True)).run("why did AMD drop today")
    assert result.route.kind == RouteKind.RESEARCH and [item.capability for item in result.results] == ["market.explain_move"]
    assert len(read_decisions(ledger)) == before + 1 and read_decisions(ledger)[-1]["intent"]["source"] == "model" and read_decisions(ledger)[-1]["run_id"] == result.run_id
    AgentV2(catalog=default_catalog(), registry=registry).run("why did AMD drop today")
    assert len(read_decisions(ledger)) == before + 1  # no ledger without the flag


def test_workspace_agent_records_intents_even_with_an_explicit_default_config(monkeypatch):
    """The bot and the web pass ``AgentV2Config()``; live model runs must still ledger their decisions."""

    from v2.agent_v2 import runtime

    captured: dict = {}

    def fake_llm_agent(*, config, llm, lab, web_search):
        captured["config"] = config
        return object()

    monkeypatch.setattr(runtime, "build_llm_agent", fake_llm_agent)
    monkeypatch.setattr(runtime, "WorkspaceLabPort", lambda: None)
    runtime.build_workspace_agent(config=AgentV2Config(), enable_web=False, use_llm=True)
    assert captured["config"].record_intents is True
    runtime.build_workspace_agent(config=AgentV2Config(record_intents=False), enable_web=False, use_llm=True)
    assert captured["config"].record_intents is False


def test_tool_loop_drives_declared_tools_through_native_function_calling():
    from v2.agent_common.llm import ToolCall
    from v2.agent_v2.agents.base import LoopLimits, Tool, ToolLoop, _schema

    seen: list[str] = []

    class Probe(ToolLoop):
        usage_source_name = "agent_v2.probe"

        def __init__(self, llm):
            self.refused = False
            super().__init__(
                llm,
                LoopLimits(max_rounds=8, max_seconds=30),
                tools=[Tool("look", "看一眼。", _schema({"at": {"type": "string"}}, ["at"]), self._look)],
                finish_parameters=_schema({"answer": {"type": "string"}}, ["answer"]),
                unavailable={"web": "本次未授权网页。"},
            )

        def _look(self, arguments):
            if not arguments.get("at"):
                raise ValueError("look 需要 at。")
            seen.append(arguments["at"])
            return f"看到了 {arguments['at']}"

        def refuse_finish(self, action):
            if not self.refused:
                self.refused = True
                return "再看一眼。"
            return None

    def call(name, arguments, *, id="c1", raw=None, parse_error=""):
        return ToolCall(id=id, name=name, arguments=arguments, raw_arguments=raw if raw is not None else json.dumps(arguments), parse_error=parse_error)

    llm = ScriptedLLM([
        LLMResponse(tool_calls=[call("look", {"at": "a"}, id="c1"), call("look", {"at": "b"}, id="c2")]),  # only the first executes; the second gets a result anyway
        LLMResponse(tool_calls=[call("look", {}, id="c3", raw="{bad", parse_error="unterminated")]),  # bad arguments
        LLMResponse(tool_calls=[call("look", {"at": ""}, id="c4")]),  # the handler's own complaint
        LLMResponse(tool_calls=[call("web", {"q": "x"}, id="c5")]),  # known but not offered
        LLMResponse(text="我觉得答案是 a"),  # prose instead of a tool: sent back
        LLMResponse(tool_calls=[call("finish", {"answer": "a"}, id="c6")]),  # refused once
        LLMResponse(tool_calls=[call("finish", {"answer": "a!"}, id="c7")]),
    ])
    loop = Probe(llm)
    outcome = loop.run("system", "task", finish_prompt="finish now")
    assert outcome.finished and outcome.final == {"action": "finish", "answer": "a!"} and outcome.stop_reason == "finished" and seen == ["a"]
    # Every call carried the declared tools, finish included.
    specs = llm.calls and [tool["function"]["name"] for tool in loop.tool_specs()]
    assert specs == ["look", "finish"] and loop.tool_specs()[1]["function"]["parameters"]["required"] == ["answer"]
    transcript = llm.calls[-1]
    roles = [(m["role"], m.get("tool_call_id", "")) for m in transcript]
    assert roles[:2] == [("system", ""), ("user", "")]
    # Round 1: assistant with two calls, a tool result for each (the second says it was not executed).
    assert transcript[2]["role"] == "assistant" and [c["id"] for c in transcript[2]["tool_calls"]] == ["c1", "c2"]
    assert (transcript[3]["tool_call_id"], transcript[3]["content"]) == ("c2", "每轮只执行一个工具调用，这一个未执行；需要的话下一轮再调用。")
    assert (transcript[4]["tool_call_id"], transcript[4]["content"]) == ("c1", "看到了 a")
    by_id = {m.get("tool_call_id"): m["content"] for m in transcript if m["role"] == "tool"}
    assert "参数不是合法 JSON" in by_id["c3"] and by_id["c4"] == "look 需要 at。" and by_id["c5"] == "本次未授权网页。" and by_id["c6"] == "再看一眼。"
    prose = next(i for i, m in enumerate(transcript) if m["role"] == "assistant" and m.get("content") == "我觉得答案是 a")
    assert transcript[prose + 1] == {"role": "user", "content": "请调用一个工具，或调用 finish 结束；不要用普通文字回答。"}
    assert [step["action"] for step in outcome.trace] == ["look", "bad_turn", "look", "web", "bad_turn", "finish_refused", "finish"]
    assert loop.tool_lines() == "- look：看一眼。\n- finish：报告结果并结束。"


def test_news_checker_works_through_tool_calls_and_answers_every_call():
    from v2.agent_common.llm import ToolCall
    from v2.agent_v2.agents.news_checker import NewsChecker

    page = "Arm Holdings shares slid 8% on Wednesday, July 29, after the company's revenue guidance came in below Wall Street expectations."

    def search(query, *, days, max_results):
        return [{"title": "Arm falls as guidance disappoints", "url": "https://example.com/arm-guidance", "content": "Arm shares slid after guidance came in below expectations.", "published_date": "2026-07-29", "raw_content": page}]

    def call(name, arguments, id):
        return ToolCall(id=id, name=name, arguments=arguments, raw_arguments=json.dumps(arguments))

    llm = ScriptedLLM([
        LLMResponse(tool_calls=[call("search", {"query": "Arm Holdings July 29 2026"}, "s1")]),
        LLMResponse(tool_calls=[call("read", {"ids": ["r1"]}, "r1")]),
        LLMResponse(tool_calls=[call("finish", {"events": [{"date": "2026-07-29", "text": "营收指引低于华尔街预期", "source": "r1", "quote": "revenue guidance came in below Wall Street expectations"}], "note": ""}, "f1")]),
    ])
    checker = NewsChecker(llm, search)
    result = checker.run("ARM", _context(), query="ARM 为什么在 7 月 29 日大跌", topic="company_event", recency_days=60, today=date(2026, 9, 9))
    events = [item for item in result.evidence if item.metadata.get("evidence_type") == "news_event"]
    assert result.ok and len(events) == 1 and events[0].metadata["read"] is True and result.metrics["searches"] == 1 and result.metrics["reads"] == 1
    # The transcript pairs every tool call with a tool message, and the tools offered are search, read, finish.
    transcript = llm.calls[-1]
    assert [m.get("tool_call_id") for m in transcript if m["role"] == "tool"] == ["s1", "r1"]
    assert [step["action"] for step in result.metadata["trace"]] == ["search", "read", "finish"]
    assert [tool["function"]["name"] for tool in checker_loop_specs(llm)] == ["search", "read", "finish"]


def checker_loop_specs(llm):
    from v2.agent_v2.agents.base import LoopLimits
    from v2.agent_v2.agents.news_checker import _CheckLoop

    return _CheckLoop(llm, LoopLimits(), search=lambda *a, **k: [], days=30, max_searches=3, max_reads=3, max_chars=5000).tool_specs()


def test_read_filings_are_stamped_and_unread_claims_are_caught_in_every_wording():
    from v2.agent_v2.agents.move_attributor import _UNREAD_RULE, clip, mark_read_filings
    from v2.agent_v2.interfaces.telegram_format import _one_line

    # The rule is a sentence for the judge, not a pattern over wordings.
    assert _UNREAD_RULE["forbid_claim"].startswith("申报的正文或内容没有被读取") and "forbid" not in _UNREAD_RULE

    bare = EvidenceItem("evidence-filing-a", "ARM", "ARM 于 2026-07-29 向 SEC 提交了 6-K（0001）。", source_id="sec_edgar", source_url="https://www.sec.gov/Archives/edgar/data/1973239/0001/", metadata={"evidence_scope": "filing", "date": "2026-07-29", "form": "6-K", "accession": "0001"})
    other = EvidenceItem("evidence-filing-b", "ARM", "ARM 于 2026-07-29 向 SEC 提交了 6-K（0002）。", source_id="sec_edgar", source_url="https://www.sec.gov/Archives/edgar/data/1973239/0002/", metadata={"evidence_scope": "filing", "date": "2026-07-29", "form": "6-K", "accession": "0002"})
    by_date = EvidenceItem("evidence-filing-c", "ARM", "ARM 于 2026-07-28 向 SEC 提交了 8-K（0003）。", source_id="sec_edgar", metadata={"evidence_scope": "filing", "date": "2026-07-28", "form": "8-K", "accession": "0003"})
    recent = ToolEnvelope("filings.recent", ResultStatus.COMPLETED, subject="ARM", evidence=[bare, other, by_date])
    event = EvidenceItem("evidence-filing-event-1", "ARM", "ARM 2026-07-29：营收指引低于预期（6-K 2026-07-29 s1：“…”）。", as_of="2026-07-29", source_url="https://www.sec.gov/Archives/edgar/data/1973239/0001", metadata={"evidence_scope": "filing_event", "date": "2026-07-29", "form": "6-K", "filing_date": "2026-07-29"})
    dated = EvidenceItem("evidence-filing-event-2", "ARM", "ARM 2026-07-28：高管变动（8-K 2026-07-28 s2：“…”）。", as_of="2026-07-28", metadata={"evidence_scope": "filing_event", "date": "2026-07-28", "form": "8-K", "filing_date": "2026-07-28"})
    attributed = ToolEnvelope("market.attribute_move", ResultStatus.COMPLETED, subject="ARM", evidence=[event, dated])
    assert mark_read_filings([recent, attributed]) == 2
    # Matched by URL (trailing slash ignored) and by form + date; the untouched record stays bare; a second pass changes nothing.
    assert bare.claim == "ARM 于 2026-07-29 向 SEC 提交了 6-K（0001）；申报阅读者已读取正文，读到的事件见 [evidence-filing-event-1]。" and bare.metadata["read_by"] == ["evidence-filing-event-1"]
    assert by_date.metadata["read_by"] == ["evidence-filing-event-2"] and "read_by" not in other.metadata
    assert mark_read_filings([recent, attributed]) == 0 and mark_read_filings([recent]) == 0

    # Clipping lands on punctuation and never leaves a bracket open.
    long = "当日高增长半导体股遭整体抛售、纳指跌 2.60%，ARM 作为估值极高的芯片股被同向抛压，跌幅（约 -9.4% 至收盘 224.89 美元）与板块相当"
    assert clip(long, 60) == "当日高增长半导体股遭整体抛售、纳指跌 2.60%，ARM 作为估值极高的芯片股被同向抛压…"
    assert clip("短句", 60) == "短句" and clip("没有标点的一长串文字" * 10, 30).endswith("…") and len(clip("没有标点的一长串文字" * 10, 30)) <= 30
    assert _one_line("  多  空格 ", 80) == "多 空格" and _one_line(long, 60) == clip(long, 60)


def test_claim_judge_decides_wording_rules_in_one_call_and_the_synthesizer_uses_it():
    from v2.agent_v2.adapters.market import _CANDIDATE_RULE, _INTRADAY_VOLUME_RULE
    from v2.agent_v2.judge import ClaimJudge

    # The judge: one call, verdicts by id, quotes trimmed, unknown ids ignored, failure means no verdicts.
    llm = ScriptedLLM([LLMResponse(text='{"verdicts":[{"id":"a","asserted":true,"quote":"AMD 属于缩量上涨"},{"id":"b","asserted":false},{"id":"zzz","asserted":true}]}'), LLMResponse(text="not json")])
    judge = ClaimJudge(llm)
    items = [{"id": "a", "text": "AMD 属于缩量上涨。", "claim": "用盘中成交量断定缩量"}, {"id": "b", "text": "AMD 可能缩量。", "claim": "用盘中成交量断定缩量"}]
    assert judge(items) == {"a": "AMD 属于缩量上涨"} and len(llm.calls) == 1 and json.loads(llm.calls[0][1]["content"])["items"][0]["claim"] == "用盘中成交量断定缩量"
    # The same items again are answered from the memo (the orchestrator re-judges the synthesizer's accepted text); different items go to the model.
    assert judge(items) == {"a": "AMD 属于缩量上涨"} and len(llm.calls) == 1
    other = [{"id": "c", "text": "别的文本。", "claim": "别的断言"}]
    assert judge(other) == {} and len(llm.calls) == 2 and ClaimJudge(None)(items) == {} and judge([]) == {}

    # The verifier collects every forbid_claim (per cited evidence and per answer) into one judge call and warns with the quote.
    volume = EvidenceItem("V1", "AMD", "AMD 截至查询时的盘中累计成交量为 1,000 股。", metadata={"constraints": [_INTRADAY_VOLUME_RULE]})
    candidate = EvidenceItem("C1", "AMD", "中置信度候选解释：期权市场波动。", metadata={"claim_role": "candidate_driver", "constraints": [_CANDIDATE_RULE]})
    result = ToolEnvelope("market.explain_move", ResultStatus.COMPLETED, subject="AMD", evidence=[volume, candidate], metadata={"answer_constraints": [{"forbid_claim": "把内部计数说给用户", "warning": "将内部归因计数直接暴露给用户"}]})
    calls: list[list[dict]] = []

    def scripted(items):
        calls.append(items)
        return {item["id"]: "期权市场波动是主要原因" for item in items if "候选" in item["claim"]}

    answer = "AMD 上涨。[V1] 期权市场波动是主要原因。[C1] 成交量放大，但当日未收盘不能据此判断。[V1]"
    report = verify_answer(answer, [volume, candidate], answer_mode=AnswerMode.RESEARCH_GROUNDED, results=[result], judge=scripted)
    assert len(calls) == 1 and sorted(item["claim"][:6] for item in calls[0]) == sorted(["用尚未收盘的", "把“证据”里", "把内部计数说"])
    by_claim = {item["claim"][:6]: item for item in calls[0]}
    assert by_claim["用尚未收盘的"]["text"] == "证据：AMD 截至查询时的盘中累计成交量为 1,000 股。\n回答：AMD 上涨。 成交量放大，但当日未收盘不能据此判断。" and by_claim["把内部计数说"]["text"].startswith("AMD 上涨。 期权市场波动是主要原因。")
    assert list(report.warnings) == ["候选归因被表述为已确认原因（“期权市场波动是主要原因”）"] and not report.ok

    # The model synthesizer owns a judge and repairs a draft the judge rejected.
    synth_llm = ScriptedLLM([
        LLMResponse(text="期权市场波动是主要原因。[C1]"),  # draft
        LLMResponse(text='{"verdicts":[{"id":"evidence:C1:0","asserted":true,"quote":"期权市场波动是主要原因"}]}'),  # judge on the draft
        LLMResponse(text="期权市场波动可能是原因之一，尚未确认。[C1]"),  # repair
        LLMResponse(text='{"verdicts":[{"id":"evidence:C1:0","asserted":false}]}'),  # judge on the repair
    ])
    synthesizer = LLMEvidenceSynthesizer(synth_llm)
    assert synthesizer.judge is not None
    plan = ExecutionPlan("AMD 为什么涨", RouteKind.RESEARCH, tasks=(PlanTask("t", "market.explain_move", {"ticker": "AMD"}),), answer_mode=AnswerMode.RESEARCH_GROUNDED)
    result_plain = ToolEnvelope("market.explain_move", ResultStatus.COMPLETED, subject="AMD", evidence=[candidate])
    text = synthesizer.synthesize(normalize_request("AMD 为什么涨"), plan, [result_plain], [candidate])
    assert text == "期权市场波动可能是原因之一，尚未确认。[C1]" and synthesizer.last_outcome == "repaired"
    assert "候选归因被表述为已确认原因（“期权市场波动是主要原因”）" in synth_llm.calls[2][-1]["content"]


def test_recorded_intents_reproduce_every_legacy_plan():
    """The labels were taken from the regex pipeline before it was removed; the intent planner must plan the same tasks from them.

    Each fixture row keeps the legacy route and capabilities; the diff is the
    regression test for the templates.  Rows that are not questions (test
    prose harvested along the way) are skipped.
    """

    from v2.agent_v2.intent import _RECORDED

    rows = _RECORDED._load()
    assert len(rows) > 300
    mismatches = []
    checked = 0
    for text, row in rows.items():
        legacy = row.get("legacy") or {}
        if not legacy or "→" in text or "[" in text or text.endswith(("。", "：", "，")):
            continue  # hand-labelled rows (the regex pipeline was wrong there) carry no legacy plan
        request = normalize_request(text)
        decision = route(request)
        plan = RulePlanner().plan(request, decision)
        checked += 1
        got = (decision.kind.value, [task.capability for task in plan.tasks], plan.answer_mode.value, plan.requires_confirmation)
        want = (legacy["route"], legacy["capabilities"], legacy["answer_mode"], legacy["requires_confirmation"])
        if got != want:
            mismatches.append((text, got, want))
    assert checked > 300 and not mismatches, mismatches[:5]



def test_this_rounds_fixes_watchlist_performance_confirmed_commands_forced_finish_and_labels(monkeypatch):
    from v2.agent_common.llm import ToolCall
    from v2.agent_v2.agents.base import LoopLimits, Tool, ToolLoop, _schema
    from v2.agent_v2.intent import Intent
    from v2.agent_v2.interfaces import telegram_format

    # 1. A watchlist performance question fans market.performance out over the names even without a ranking word.
    text = "我关注的股票里有没有最近在放量的？"
    watch = Intent(kind="lookup", scope="recent", wants=("watchlist", "performance"), watchlist_scope=True, source="model")
    plan = RulePlanner().plan(normalize_request(text), route(normalize_request(text), intent=watch))
    assert [(task.capability, bool(task.fan_out)) for task in plan.tasks] == [("state.read", False), ("market.performance", True)]
    mine = Intent(kind="lookup", scope="none", wants=("performance",), portfolio_scope=True, periods=("month",), source="model")
    plan = RulePlanner().plan(normalize_request("我这个月赚了多少？"), route(normalize_request("我这个月赚了多少？"), intent=mine))
    assert [task.capability for task in plan.tasks] == ["account.performance"]  # the account card answers; no per-holding fan-out

    # 2. A confirmed command is answered from the store's message, never by the model synthesizer.
    applied: list[dict] = []
    registry = CapabilityRegistry(default_catalog())
    registry.register("state.mutate", lambda arguments, context: applied.append(dict(arguments)) or ToolEnvelope("state.mutate", ResultStatus.COMPLETED, subject="alert.add", summary="提醒 #6 已记录：NVDA 涨到 240 美元时通知。", evidence=[EvidenceItem("S1", "NVDA", "提醒 #6 已记录：NVDA 涨到 240 美元时通知。", source_id="state.mutate")]))
    model = ScriptedLLM([LLMResponse(text="它在你确认之前不会生效。[S1]")])
    from v2.agent_v2.intent import IntentClassifier

    classifier = IntentClassifier(ScriptedLLM([LLMResponse(text='{"kind":"command","scope":"none","direction":"none","wants":["alerts"],"tickers":["NVDA"],"portfolio_scope":false,"confidence":0.95,"command":{"operation":"alert.add","ticker":"NVDA","direction":"above","price":240}}')]))
    agent = AgentV2(catalog=default_catalog(), registry=registry, synthesizer=LLMEvidenceSynthesizer(model), session=ShortTermSession(), classifier=classifier)
    first = agent.run("NVDA涨到240时提醒我", session_id="cmd-1")
    assert first.status == RunStatus.WAITING_CONFIRMATION
    done = agent.run("确认", session_id="cmd-1")
    assert done.status == RunStatus.COMPLETED and applied and "提醒 #6 已记录" in done.answer and "确认之前" not in done.answer and model.calls == []

    # 3. The forced-finish turn tells the provider to call finish.
    class Lazy(ToolLoop):
        def __init__(self, llm):
            super().__init__(llm, LoopLimits(max_rounds=1, max_seconds=30), tools=[Tool("look", "看。", _schema({}), lambda a: "看到了")], finish_parameters=_schema({"answer": {"type": "string"}}, ["answer"]))

    llm = ScriptedLLM([LLMResponse(tool_calls=[ToolCall(id="c1", name="look", arguments={}, raw_arguments="{}")]), LLMResponse(tool_calls=[ToolCall(id="c2", name="finish", arguments={"answer": "x"}, raw_arguments="{}")])])
    outcome = Lazy(llm).run("s", "t", finish_prompt="finish now")
    assert outcome.finished and llm.tool_choices == [None, {"type": "function", "function": {"name": "finish"}}]
    assert [tool["function"]["name"] for tool in Lazy(ScriptedLLM([])).tool_specs()] == ["look", "finish"]

    # 4. The max_cited warning names the candidate to keep and the ones to drop.
    candidates = [EvidenceItem(f"C{i}", "AMD", f"候选 {i}", metadata={"claim_role": "candidate_driver"}) for i in (1, 2, 3)]
    result = ToolEnvelope("market.attribute_move", ResultStatus.COMPLETED, subject="AMD 2026-07-02", evidence=candidates, metadata={"answer_constraints": [{"max_cited": {"metadata": {"claim_role": "candidate_driver"}, "max": 1, "warning": "未确认直接驱动时展示了过多弱候选线索"}}]})
    report = verify_answer("一 [C1]。二 [C2]。三 [C3]。又一 [C1]。", candidates, answer_mode=AnswerMode.RESEARCH_GROUNDED, results=[result])
    assert list(report.warnings) == ["（提示）未确认直接驱动时展示了过多弱候选线索（AMD 2026-07-02：只保留 [C1]，去掉 [C2]、[C3]）"] and report.ok  # soft: reported, not a rejection

    # 6. Legacy cards are labelled by capability.
    macro = EvidenceItem("legacy-1", "macro", "VIX 15.74", source_id="macro.overview", source_title="Existing deterministic responder")
    calendar = EvidenceItem("legacy-2", "", "无标的发财报", source_id="account.earnings_schedule", source_title="Existing deterministic responder")
    assert [entry.label for entry in telegram_format.source_entries(("legacy-1", "legacy-2"), [macro, calendar])] == ["宏观面板", "财报日历"]


# -- the six framework items -----------------------------------------------------------------


def test_quality_loop_grades_answers_records_runs_and_reports(tmp_path, monkeypatch):
    from v2.agent_v2.eval import quality
    from v2.agent_v2.eval.quality import QualityJudge, grade, read_rows, render, run_cases, summarize
    from v2.agent_v2.eval.quality_cases import QUALITY_CASES, QualityCase, from_feedback

    assert len(QUALITY_CASES) >= 12 and len({case.id for case in QUALITY_CASES}) == len(QUALITY_CASES)
    # The model judge answers through the grade tool; parsed per index.
    judge_llm = ScriptedLLM([LLMResponse(text='{"criteria":[{"index":0,"met":true,"quote":"AAPL 今天上涨 3.18%"},{"index":1,"met":false}],"forbidden":[{"index":0,"asserted":false}]}')])
    judge = QualityJudge(judge_llm)
    verdict = judge("q", "a", ["有涨跌幅", "有基准"], ["候选说成原因"])
    assert verdict["criteria"][0]["met"] is True and json.loads(judge_llm.calls[0][1]["content"])["criteria"][1]["text"] == "有基准"

    case = QualityCase("t1", "AAPL今天为什么涨？", criteria=("有涨跌幅", "有基准"), forbidden=("候选说成原因",), must_cite=("market_data",), expected_route=RouteKind.RESEARCH, expected_agents=("move_attributor",))
    result = _telegram_result("AAPL 今天上涨 3.18%[evidence-market-price-42f5c41951e36ef1]。", outcome="clean")
    result.results[0].metadata["agent"] = {"name": "move_attributor", "rounds": 1, "llm_calls": 1, "elapsed_ms": 10, "stop_reason": "finished", "calls": {}, "yield": {"kept": 1, "dropped": 0}}
    score = grade(case, result, lambda q, a, c, f: {"criteria": [{"index": 0, "met": True, "quote": "AAPL 今天上涨"}, {"index": 1, "met": False}], "forbidden": [{"index": 0, "asserted": False}]})
    assert not score.passed and score.criteria_met == 1 and score.criteria_total == 2 and score.route_ok and score.agents_ok and score.sources_ok and score.problems == ["未满足：有基准"]
    full = grade(case, result, lambda q, a, c, f: {"criteria": [{"index": 0, "met": True}, {"index": 1, "met": True}], "forbidden": [{"index": 0, "asserted": False}]})
    assert full.passed
    hit = grade(case, result, lambda q, a, c, f: {"criteria": [{"index": 0, "met": True}, {"index": 1, "met": True}], "forbidden": [{"index": 0, "asserted": True, "quote": "主要原因是"}]})
    assert not hit.passed and hit.forbidden_hit == 1 and hit.problems == ["出现禁止断言：候选说成原因（“主要原因是”）"]
    unjudged = grade(case, result, None)
    assert not unjudged.passed and "未评分" in unjudged.problems
    wrong_agent = grade(QualityCase("t2", "x", criteria=(), expected_agents=("news_checker",), must_cite=("web",)), result, None)
    assert not wrong_agent.passed and "missing agents: news_checker" in wrong_agent.problems and "cited sources lack web" in wrong_agent.problems

    # A run through an agent records one row per case; the report compares runs.
    ledger = tmp_path / "quality.jsonl"
    registry = CapabilityRegistry(default_catalog())
    registry.register("market.explain_move", lambda arguments, context: ToolEnvelope("market.explain_move", ResultStatus.COMPLETED, subject="AMD", evidence=[EvidenceItem("M", "AMD", "AMD 今日 +3.18%", source_id="market_data")], metadata={"narrative": "AMD 今日 +3.18%[M]。", "agent": {"name": "move_attributor", "rounds": 1, "llm_calls": 1, "elapsed_ms": 10, "stop_reason": "finished", "calls": {}, "yield": {"kept": 1, "dropped": 0}}}))
    agent = AgentV2(catalog=default_catalog(), registry=registry, config=AgentV2Config(record_sub_agents=False))
    passing = lambda q, a, c, f: {"criteria": [{"index": i, "met": True} for i in range(len(c))], "forbidden": [{"index": i, "asserted": False} for i in range(len(f))]}
    failing = lambda q, a, c, f: {"criteria": [{"index": i, "met": i == 0} for i in range(len(c))], "forbidden": [{"index": i, "asserted": False} for i in range(len(f))]}
    cases = (QualityCase("t1", "AMD今天为什么涨？", criteria=("有涨跌幅", "有基准"), must_cite=("market_data",), expected_route=RouteKind.RESEARCH, expected_agents=("move_attributor",)),)
    rows = run_cases(agent, cases, failing, label="before", path=ledger)
    rows += run_cases(agent, cases, passing, label="after", path=ledger)
    assert [row["score"]["passed"] for row in rows] == [False, True] and rows[0]["sub_agents"] == ["move_attributor"] and rows[0]["route"] == "research"
    summary = summarize(read_rows(ledger))
    assert [(run["label"], run["passed"], run["cases"]) for run in summary["runs"]] == [("before", 0, 1), ("after", 1, 1)] and summary["missed"] == [("t1: 有基准", 1)]
    text = render(summary)
    assert "| before |" in text and "| after |" in text and "| t1 | ✗ 1/2 | ✓ 2/2 |  |" in text  # the last column is the latest run's problems
    monkeypatch.setenv("AGENT_V2_QUALITY_LEDGER", str(ledger))
    assert quality.main(["report", "--json"]) == 0
    # Negative feedback becomes a case.
    fb = from_feedback([{"kind": "feedback", "verdict": "bad", "question": "ARM买入以来跌了这么多，是什么原因？", "note": "7/29 是财报日，不是无事件"}, {"kind": "feedback", "verdict": "good", "question": "x"}])
    assert len(fb) == 1 and fb[0].origin == "feedback" and fb[0].criteria == ("回答不再出现用户指出的问题：7/29 是财报日，不是无事件",)


def test_run_board_shares_results_and_schedules_bounded_follow_ups():
    from v2.agent_v2.execution import RunBoard

    board = RunBoard(max_requests=2)
    assert board.request("filings.read_events", {"ticker": "ARM", "around": "2026-07-29"}, requested_by="news_checker")
    assert not board.request("filings.read_events", {"ticker": "ARM", "around": "2026-07-29"})  # the same work twice
    assert board.request("market.performance", {"ticker": "ARM"}) and not board.request("market.performance", {"ticker": "MU"})  # the cap
    assert [task.capability for task in board.take_requests()] == ["filings.read_events", "market.performance"] and board.take_requests() == [] and board.refused

    # The engine adopts a follow-up a task requested and refuses a mutation; tasks see earlier results on the board.
    catalog = default_catalog()
    registry = CapabilityRegistry(catalog)
    seen: dict[str, Any] = {}

    def news(arguments, context):
        context.board.request("filings.read_events", {"ticker": "ARM", "around": "2026-07-29"}, purpose="read the filing the story named", requested_by="news_checker")
        context.board.request("state.mutate", {"operation": "watchlist.add", "payload": {"ticker": "ARM"}})
        return ToolEnvelope("web.research", ResultStatus.COMPLETED, subject="ARM", evidence=[EvidenceItem("W1", "ARM", "story")])

    def reader(arguments, context):
        seen["board_results"] = [r.capability for r in context.board.results]
        seen["evidence"] = [item.id for item in context.board.evidence_where(entity="ARM")]
        return ToolEnvelope("filings.read_events", ResultStatus.COMPLETED, subject="ARM", evidence=[EvidenceItem("E1", "ARM", "event", metadata={"evidence_scope": "filing_event", "date": "2026-07-29"})])

    registry.register("web.research", news)
    registry.register("filings.read_events", reader)
    plan = ExecutionPlan("q", RouteKind.RESEARCH, tasks=(PlanTask("news", "web.research", {"query": "ARM", "topic": "company_event"}),), budget=BudgetClass.STANDARD)
    context = ExecutionContext("run", NormalizedRequest("q", "q", allow_web=True), BudgetClass.STANDARD, allow_web=True)
    outcome = ExecutionEngine(registry).run(plan, context)
    assert [r.capability for r in outcome.results] == ["web.research", "filings.read_events"] and outcome.stop_reason == "completed"
    assert seen["board_results"] == ["web.research"] and seen["evidence"] == ["W1"] and outcome.ledger.get("E1") is not None
    assert context.board.refused == ["state.mutate"] or "state.mutate" in context.board.refused

    # The news checker asks the board to read a filing a story named.
    from v2.agent_v2.agents.news_checker import NewsChecker

    page = "Arm said in an 8-K filed on July 29, 2026 that revenue guidance came in below Wall Street expectations."
    search = lambda query, *, days, max_results: [{"title": "Arm files", "url": "https://example.com/a", "content": "Arm 8-K", "published_date": "2026-07-29", "raw_content": page}]
    llm = ScriptedLLM([LLMResponse(text='{"action":"search","query":"Arm 8-K"}'), LLMResponse(text='{"action":"read","ids":["r1"]}'), LLMResponse(text=json.dumps({"action": "finish", "events": [{"date": "2026-07-29", "text": "指引低于预期", "source": "r1", "quote": "revenue guidance came in below Wall Street expectations"}], "note": "", "filing_to_read": "2026-07-29"}, ensure_ascii=False))])
    context = ExecutionContext("run", NormalizedRequest("q", "q"), BudgetClass.STANDARD)
    result = NewsChecker(llm, search).run("ARM", context, query="ARM 新闻", topic="company_event", recency_days=30, today=date(2026, 9, 9))
    assert result.metadata["agent"]["follow_up"] == {"filings_around": "2026-07-29", "accepted": True} and [task.capability for task in context.board.take_requests()] == ["filings.read_events"]
    assert "已请申报阅读者读取" in result.metadata["agent"]["notes"][0]


def test_cancellation_stops_the_engine_the_loops_and_marks_the_run(monkeypatch):
    import threading

    from v2.agent_v2.agents.base import LoopLimits, Tool, ToolLoop, _schema, limits_for

    # A loop stops between rounds once the run is cancelled.
    event = threading.Event()
    context = ExecutionContext("run", NormalizedRequest("q", "q"), BudgetClass.STANDARD, cancel_event=event)
    assert not context.cancelled
    limits = limits_for(context, max_rounds=5, max_seconds=30)
    calls = {"n": 0}

    def look(arguments):
        calls["n"] += 1
        event.set()
        return "看到了"

    loop = ToolLoop(ScriptedLLM([LLMResponse(text='{"action":"look"}'), LLMResponse(text='{"action":"look"}')]), limits, tools=[Tool("look", "看。", _schema({}), look)], finish_parameters=_schema({}))
    outcome = loop.run("s", "t", finish_prompt="finish")
    assert outcome.stop_reason == "cancelled" and calls["n"] == 1 and outcome.note == "用户取消"

    # The engine skips what is left and the orchestrator reports the run as cancelled with the partial answer.
    registry = CapabilityRegistry(default_catalog())
    cancel = threading.Event()

    def slow(arguments, context):
        cancel.set()
        return ToolEnvelope("account.portfolio", ResultStatus.COMPLETED, subject="me", evidence=[EvidenceItem("P1", "me", "持仓 12 只")])

    registry.register("account.portfolio", slow)
    registry.register("account.risk", lambda arguments, context: ToolEnvelope("account.risk", ResultStatus.COMPLETED, subject="me"))
    plan = ExecutionPlan("q", RouteKind.FAST_LOOKUP, tasks=(PlanTask("p", "account.portfolio"), PlanTask("r", "account.risk", depends_on=("p",))), budget=BudgetClass.STANDARD)
    context = ExecutionContext("run", NormalizedRequest("q", "q"), BudgetClass.STANDARD, cancel_event=cancel)
    outcome = ExecutionEngine(registry).run(plan, context)
    assert outcome.stop_reason == "cancelled" and [(r.capability, r.status) for r in outcome.results] == [("account.portfolio", ResultStatus.COMPLETED), ("account.risk", ResultStatus.SKIPPED)]
    cancel = threading.Event()
    registry.register("research.stock", lambda arguments, context: ToolEnvelope("research.stock", ResultStatus.COMPLETED, subject="x"))
    result = AgentV2(catalog=default_catalog(), registry=registry).run("我的持仓里哪只风险最高？", cancel_event=cancel)  # the fan-out wave after the card is skipped
    assert result.status == RunStatus.CANCELLED and result.stop_reason == "cancelled"

    # Two runs in flight on one agent: each execution context carries its own
    # cancel event. Run A is held in planning while run B starts and finishes
    # with a different event; A's tools must still see A's event.
    seen = {}
    gate = threading.Event()
    hold = threading.Event()
    registry = CapabilityRegistry(default_catalog())
    def observe(arguments, context):
        seen[context.run_id] = context.cancel_event  # handlers run on the engine's pool threads
        return ToolEnvelope("research.stock", ResultStatus.COMPLETED, subject="x")
    registry.register("research.stock", observe)
    agent = AgentV2(catalog=default_catalog(), registry=registry)
    planner = agent.planner
    class Holding:
        def plan(self, request, decision):
            if threading.current_thread().name == "run-a":
                gate.set()
                hold.wait(5)
            return planner.plan(request, decision)
    agent.planner = Holding()
    event_a, event_b = threading.Event(), threading.Event()
    finished = {}
    thread_a = threading.Thread(target=lambda: finished.__setitem__("a", agent.run("分析一下NVDA", cancel_event=event_a)), name="run-a")
    thread_a.start()
    assert gate.wait(5)
    thread_b = threading.Thread(target=lambda: finished.__setitem__("b", agent.run("分析一下AMD", cancel_event=event_b)), name="run-b")
    thread_b.start()
    thread_b.join(10)
    hold.set()
    thread_a.join(10)
    assert seen[finished["a"].run_id] is event_a and seen[finished["b"].run_id] is event_b
    assert not hasattr(agent, "_cancel_event")

    # Telegram: the cancel word, the active-run registry and the header line.
    from v2.agent_v2.interfaces import telegram_format
    from v2.bot import agent_v2_bridge as bridge

    assert bridge.is_cancel_word("取消") and bridge.is_cancel_word("stop。") and not bridge.is_cancel_word("取消提醒 3")
    assert not bridge.cancel_active_run(1)
    bridge._ACTIVE_RUNS[1] = threading.Event()
    assert bridge.cancel_active_run(1) and bridge._ACTIVE_RUNS[1].is_set()
    bridge._ACTIVE_RUNS.pop(1)
    from v2.agent_v2.models import ProgressEvent

    assert telegram_format.progress_line(ProgressEvent("r", RunStatus.EXECUTING, "explain", capability="market.explain_move"), elapsed=12.4) == "执行中：解释今日涨跌 · 12s"
    assert telegram_format.progress_line(ProgressEvent("r", RunStatus.PLANNED, "planned 3 task(s)")) == "已规划：3 个任务"
    assert telegram_format.progress_line(ProgressEvent("r", RunStatus.EXECUTING, "follow-up: read", capability="filings.read_events")) == "执行中：追加：读申报"
    assert telegram_format._STOP_LABELS["cancelled"] == "用户取消"


def test_debate_revision_rewrites_once_and_only_keeps_a_verified_rewrite():
    from v2.agent_v2.interfaces import telegram_format

    registry = CapabilityRegistry(default_catalog())
    evidence = [EvidenceItem("R1", "NVDA", "NVDA TTM 市盈率 28.2 倍", source_id="fd_metrics"), EvidenceItem("R2", "NVDA", "90 天内部人净卖出 2.4 亿美元", source_id="fd_insiders")]
    registry.register("research.stock", lambda arguments, context: ToolEnvelope("research.stock", ResultStatus.COMPLETED, subject="NVDA", summary="估值不低", evidence=evidence))
    llm = ScriptedLLM([
        LLMResponse(text="NVDA 市盈率 28.2 倍[R1]，估值不算高。"),  # draft
        LLMResponse(text='{"stance":"回答偏多","objections":[{"claim":"估值不算高","objection":"内部人净卖出 2.4 亿美元被忽略","evidence_id":"R2"}],"note":""}'),  # debater
        LLMResponse(text="NVDA 市盈率 28.2 倍[R1]，估值不算高，但 90 天内部人净卖出 2.4 亿美元[R2]是需要计入的风险。"),  # revision
    ])
    agent = AgentV2(catalog=default_catalog(), registry=registry, synthesizer=LLMEvidenceSynthesizer(llm))
    result = agent.run("分析 NVDA 的估值")
    assert "[R2]" in result.answer and result.synthesis["debate_revision"] == {"applied": True, "objections": 1} and result.verification.ok
    assert [r.capability for r in result.results][-1] == "debate.challenge"
    assert telegram_format.synthesis_label(result) == "模型回答，按反方修订"
    # A rewrite that fails verification is dropped and the original stands, with the reason shown.
    llm = ScriptedLLM([
        LLMResponse(text="NVDA 市盈率 28.2 倍[R1]，估值不算高。"),
        LLMResponse(text='{"stance":"回答偏多","objections":[{"claim":"估值不算高","objection":"内部人净卖出被忽略","evidence_id":"R2"}],"note":""}'),
        LLMResponse(text="NVDA 市盈率 31 倍[R1]，内部人净卖出 5 亿美元[R2]。"),  # invents numbers
    ])
    agent = AgentV2(catalog=default_catalog(), registry=registry, synthesizer=LLMEvidenceSynthesizer(llm))
    result = agent.run("分析 NVDA 的估值")
    assert result.answer == "NVDA 市盈率 28.2 倍[R1]，估值不算高。" and result.synthesis["debate_revision"] == {"applied": False, "objections": 1, "reason": "修订稿未通过校验"}
    assert telegram_format.synthesis_label(result) == "模型回答，反方意见未采纳（修订稿未通过校验）"
    # Off by configuration: objections stay display-only.
    llm = ScriptedLLM([LLMResponse(text="NVDA 市盈率 28.2 倍[R1]。"), LLMResponse(text='{"stance":"回答中性","objections":[{"claim":"x","objection":"y","evidence_id":"R2"}]}')])
    result = AgentV2(catalog=default_catalog(), registry=registry, synthesizer=LLMEvidenceSynthesizer(llm), config=AgentV2Config(debate_revision=False)).run("分析 NVDA 的估值")
    assert "debate_revision" not in result.synthesis and len(llm.calls) == 2


def test_toolbox_tools_are_shared_and_the_investigator_reports_only_quoted_findings():
    from v2.agent_common.llm import ToolCall
    from v2.agent_v2.agents.filing_reader import FilingRef, Section
    from v2.agent_v2.agents.toolbox import Investigator, Toolbox, register_investigator

    page = "On July 29, 2026 Arm Holdings said revenue guidance came in below Wall Street expectations, sending shares down 13%."
    search = lambda query, *, days, max_results: [{"title": "Arm falls", "url": "https://example.com/arm", "content": "Arm guidance", "published_date": "2026-07-29", "raw_content": page}]

    class Source:
        def list_filings(self, ticker, since, until):
            return [FilingRef(ticker=ticker, form="6-K", filing_date="2026-07-29", accession="0001", url="https://www.sec.gov/x/1/")]

        def outline(self, ref):
            return [Section("s1", "Exhibit 99.1", 120)]

        def read(self, ref, section_id):
            return "Revenue was below the guidance range, the company said in its quarterly update."

    recalls = []
    toolbox = Toolbox(search=search, filing_source=Source(), recall=lambda ticker, query, days: recalls.append(query) or [], today=date(2026, 9, 9))
    assert toolbox.available() == ("search_news", "read_page", "list_filings", "read_filing", "recall_memory")
    names = [tool.name for tool in toolbox.tools(("read_filing", "search_news", "bogus"))]
    assert names == ["read_filing", "search_news"]
    tools = {tool.name: tool for tool in toolbox.tools()}
    assert "id=p1" in tools["search_news"].handler({"query": "Arm July 29"}) and "Exhibit 99.1" in tools["list_filings"].handler({"ticker": "ARM"})
    assert tools["read_page"].handler({"id": "p1"}).startswith("[p1] Arm falls（2026-07-29）") and "guidance range" in tools["read_filing"].handler({"filing": 1, "section": "s1"})
    with pytest.raises(ValueError):
        tools["read_page"].handler({"id": "p9"})
    assert toolbox.state.text_for("p1")[0] == page[:5000] and toolbox.state.text_for("f1:s1")[2] == "6-K 2026-07-29" and toolbox.state.text_for("zz") == ("", "", "")

    def call(name, arguments, id):
        return ToolCall(id=id, name=name, arguments=arguments, raw_arguments=json.dumps(arguments))

    llm = ScriptedLLM([
        LLMResponse(tool_calls=[call("search_news", {"query": "Arm guidance July 2026"}, "c1")]),
        LLMResponse(tool_calls=[call("read_page", {"id": "p1"}, "c2")]),
        LLMResponse(tool_calls=[call("list_filings", {"ticker": "ARM", "since": "2026-07-01"}, "c3")]),
        LLMResponse(tool_calls=[call("read_filing", {"filing": 1, "section": "s1"}, "c4")]),
        LLMResponse(tool_calls=[call("finish", {"findings": [
            {"date": "2026-07-29", "text": "指引低于华尔街预期", "source": "p1", "quote": "revenue guidance came in below Wall Street expectations"},
            {"date": "2026-07-29", "text": "季度更新称营收低于指引区间", "source": "f1:s1", "quote": "Revenue was below the guidance range"},
            {"date": "2026-07-29", "text": "编造", "source": "p1", "quote": "takeover rumours swirled around the company"},
            {"date": "2026-07-29", "text": "来源不存在", "source": "p7", "quote": "revenue guidance came in below Wall Street expectations"},
        ], "note": "两条有出处"}, "c5")]),
    ])
    investigator = Investigator(llm, lambda: Toolbox(search=search, filing_source=Source(), recall=None, today=date(2026, 9, 9)))
    consenting = ExecutionContext("test-run", NormalizedRequest("q", "q"), BudgetClass.STANDARD, allow_web=True)
    result = investigator.run("查 ARM 7 月 29 日大跌的申报和报道", consenting, ticker="ARM", tools=("search_news", "read_page", "list_filings", "read_filing"))
    assert result.capability == "agent.investigate" and result.ok and result.metrics["findings"] == 2 and result.metrics["dropped"] == 2
    kinds = [(item.source_id, item.source_url) for item in result.evidence]
    assert kinds == [("web:example.com", "https://example.com/arm"), ("sec_edgar", "https://www.sec.gov/x/1/")]
    assert result.metadata["agent"]["name"] == "investigator" and result.metadata["agent"]["tools"] == ["search_news", "read_page", "list_filings", "read_filing"] and result.metadata["agent"]["calls"] == {"search_news": 1, "read_page": 1, "list_filings": 1, "read_filing": 1}
    assert "2 条发现没有可核对的引文" in result.limitations[0] and "[evidence-investigate-" in result.metadata["narrative"]
    assert [rule["quote_of"] for rule in result.metadata["answer_constraints"]] == [item.id for item in result.evidence] and result.metadata["answer_constraints"][0]["require"].startswith("revenue")
    assert [step["action"] for step in result.metadata["trace"]] == ["search_news", "read_page", "list_filings", "read_filing", "finish"]
    # Registered as a capability the model planner may delegate to.
    registry = CapabilityRegistry(default_catalog())
    register_investigator(registry, ScriptedLLM([]), search=search, filing_source=Source(), recall=None, today_factory=lambda: date(2026, 9, 9))
    spec = default_catalog().get("agent.investigate")
    assert spec is not None and spec.long_running and registry.registered("agent.investigate")
    from v2.agent_v2.llm import SUB_AGENT_CAPABILITIES

    assert "agent.investigate" in SUB_AGENT_CAPABILITIES
    none = Investigator(None, lambda: Toolbox(search=search, today=date(2026, 9, 9))).run("x", _context(), ticker="ARM")
    assert none.status == ResultStatus.PARTIAL_DATA and none.metadata["agent"]["stop_reason"] == "no_model"
    # Without the user's web consent the web tools are withheld, whatever the planner asked for, and the answer is told.
    withheld = Investigator(ScriptedLLM([LLMResponse(tool_calls=[call("finish", {"findings": [], "note": "无"}, "c9")])]), lambda: Toolbox(search=search, filing_source=Source(), today=date(2026, 9, 9)))
    result = withheld.run("查 ARM 的申报", _context(), ticker="ARM", tools=("search_news", "read_page", "list_filings", "read_filing"))
    assert result.metadata["agent"]["tools"] == ["list_filings", "read_filing"] and result.metadata["agent"]["withheld"] == ["search_news", "read_page"] and "网页未授权，未搜索新闻" in result.limitations[0]


def test_investigation_intents_get_a_fixed_brief_per_job_and_the_model_planner_keeps_it():
    from v2.agent_v2.intent import INTENT_TOOL, INVESTIGATIONS, Intent, parse_intent, resolve_intent
    from v2.agent_v2.planning import INVESTIGATION_JOBS

    assert INVESTIGATIONS == ("event_story", "filing_terms", "claim_source") and INTENT_TOOL["function"]["parameters"]["properties"]["investigation"]["enum"] == ["", *INVESTIGATIONS]
    assert parse_intent({"kind": "research", "investigation": "filing_terms", "tickers": ["TSLA"]}, source="model").investigation == "filing_terms"
    assert parse_intent({"kind": "research", "investigation": "gossip"}, source="model").investigation == ""
    planner = RulePlanner()

    def plan_for(text, job, tickers, *, allow_web=True):
        request = normalize_request(text, allow_web=allow_web)
        return planner.plan(request, route(request, intent=Intent(kind="research", scope="recent", wants=("news",), tickers=tickers, investigation=job, source="model")))

    # The story of an event: every tool, the monitor's memory alongside.
    plan = plan_for("英特尔被美国政府入股那件事的来龙去脉是什么？", "event_story", ("INTC",))
    first = plan.tasks[0]
    assert first.capability == "agent.investigate" and first.required and first.arguments["ticker"] == "INTC" and first.arguments["tools"] == list(INVESTIGATION_JOBS["event_story"]["tools"]) and first.arguments["recency_days"] == 120
    assert first.arguments["task"].endswith("英特尔被美国政府入股那件事的来龙去脉是什么？") and first.arguments["task"].startswith("梳理这件事的来龙去脉")
    assert [task.capability for task in plan.tasks] == ["agent.investigate", "market.anomaly_history"] and plan.budget == BudgetClass.STANDARD and plan.answer_mode == AnswerMode.RESEARCH_GROUNDED
    assert plan.assumptions[0].startswith("investigation: 按时间顺序") and "网页已授权" in plan.assumptions[0] and not plan.web_fallback_allowed
    # A filing's terms: only the filing tools, a year back, the dated list beside it.
    plan = plan_for("特斯拉最新的 10-K 里对 FSD 的风险具体是怎么写的？", "filing_terms", ("TSLA",), allow_web=False)
    assert plan.tasks[0].arguments["tools"] == ["list_filings", "read_filing"] and plan.tasks[0].arguments["recency_days"] == 400 and [task.capability for task in plan.tasks] == ["agent.investigate", "filings.recent"]
    assert "网页未授权" in plan.assumptions[0]
    # A claim's source: the news and the filings, nothing else; no ticker is fine.
    plan = plan_for("有说法称美国要对所有进口芯片加征关税，有出处吗？", "claim_source", ())
    assert [task.capability for task in plan.tasks] == ["agent.investigate"] and "ticker" not in plan.tasks[0].arguments and plan.tasks[0].arguments["tools"] == ["search_news", "read_page", "list_filings", "read_filing"]
    # The plan validates against the catalog and the model planner keeps it as is.
    ExecutionEngine(CapabilityRegistry(default_catalog())).validate(plan)
    llm = ScriptedLLM([LLMResponse(text='{"tasks":[{"id":"t1","capability":"agent.investigate","arguments":{"task":"查一下"}}]}')])
    request = normalize_request("英特尔被美国政府入股那件事的来龙去脉是什么？")
    kept = StructuredLLMPlanner(llm, default_catalog()).plan(request, route(request, intent=Intent(kind="research", tickers=("INTC",), investigation="event_story", source="model")))
    assert kept.tasks[0].arguments["tools"] == list(INVESTIGATION_JOBS["event_story"]["tools"]) and not llm.calls
    # The three development cases are labelled by hand and expect the investigator.
    from v2.agent_v2.eval.quality_cases import QUALITY_CASES

    cases = [case for case in QUALITY_CASES if "investigate" in case.tags]
    assert [case.expected_agents for case in cases] == [("investigator",)] * 3
    recorded = [resolve_intent(normalize_request(case.question)) for case in cases]
    assert [(intent.source, intent.investigation) for intent in recorded] == [("recorded", "event_story"), ("recorded", "filing_terms"), ("recorded", "claim_source")]


def test_a_tool_loop_that_will_not_call_finish_reports_as_plain_json_before_giving_up():
    from v2.agent_common.llm import ToolCall
    from v2.agent_v2.agents.base import LoopLimits, Tool, ToolLoop, _schema

    class Reader(ToolLoop):
        def __init__(self, llm):
            super().__init__(llm, LoopLimits(max_rounds=1, max_seconds=30), tools=[Tool("look", "看。", _schema({}), lambda a: "看到了")], finish_parameters=_schema({"findings": {"type": "array"}, "note": {"type": "string"}}, ["findings"]))

    look = LLMResponse(tool_calls=[ToolCall(id="c1", name="look", arguments={}, raw_arguments="{}")])
    # The forced finish comes back as prose; the plain-JSON turn carries the report.
    llm = ScriptedLLM([look, LLMResponse(text="我读到了两条发现，但不想调用 finish。"), LLMResponse(text='```json\n{"findings":[{"date":"2026-07-29"}],"note":"补报"}\n```')])
    outcome = Reader(llm).run("s", "t", finish_prompt="finish now")
    assert outcome.finished and outcome.final == {"action": "finish", "findings": [{"date": "2026-07-29"}], "note": "补报"} and outcome.calls == 3
    assert [step["action"] for step in outcome.trace] == ["look", "forced_finish", "forced_finish_json"] and llm.calls[-1][-1]["content"].startswith("不要调用工具") and "findings、note" in llm.calls[-1][-1]["content"]
    assert llm.tool_choices[-1] is None
    # Still asking for a tool on that turn: the loop ends unfinished, as before.
    llm = ScriptedLLM([look, LLMResponse(text="再看一次"), LLMResponse(text='{"action":"look"}')])
    outcome = Reader(llm).run("s", "t", finish_prompt="finish now")
    assert not outcome.finished and outcome.stop_reason == "rounds" and outcome.calls == 3


def test_the_attributor_asks_the_board_for_the_news_of_a_filing_day_it_never_searched():
    from v2.agent_v2.agents.filing_reader import FilingRef
    from v2.agent_v2.agents.move_attributor import MoveAttributor

    class Source:
        def list_filings(self, ticker, since, until):
            return [FilingRef(ticker, "8-K", "2026-07-28", "0001-26-000001", "https://www.sec.gov/x/1/")]

    class Reader:
        source = Source()

        def run(self, ticker, context, *, around, today):
            return ToolEnvelope("filings.read_events", ResultStatus.COMPLETED, subject=ticker, evidence=[EvidenceItem("E-ARM-0728", "ARM", "ARM 2026-07-28：季度营收低于指引区间（8-K 2026-07-28 s1：“Revenue was below the guidance range”）。", metadata={"evidence_scope": "filing_event", "date": "2026-07-28", "quote": "Revenue was below the guidance range", "text": "Revenue was below the guidance range for the quarter."})])

    finish = json.dumps({"action": "finish", "reasons": [{"text": "申报显示营收低于指引区间", "confidence": "中", "source": {"kind": "filing", "id": "E-ARM-0728"}, "quote": "Revenue was below the guidance range"}], "next_steps": [], "note": ""}, ensure_ascii=False)
    searched = []
    # The model finishes on the filing alone, twice (the first finish is refused because the news was never searched).
    llm = ScriptedLLM([LLMResponse(text=finish), LLMResponse(text=finish)])
    attributor = MoveAttributor(llm, price_source_factory=lambda: SimpleNamespace(get_prices=_attributor_prices), news=lambda query, day: searched.append((query, day)) or [], filing_reader=Reader(), memory_recall=None, memory_remember=None, sector_for=lambda ticker: "SMH")
    context = ExecutionContext("run", NormalizedRequest("q", "q"), BudgetClass.PORTFOLIO, allow_web=True)
    result = attributor.run("ARM", context, day="2026-07-29", today=date(2026, 9, 9))
    assert not searched and result.metrics["news_calls"] == 0 and result.metadata["agent"]["follow_up"] == {"news_around": "2026-07-29", "accepted": True}
    requested = context.board.take_requests()
    assert [task.capability for task in requested] == ["web.research"] and requested[0].arguments == {"query": "ARM stock news 2026-07-29", "topic": "company_event", "ticker": "ARM", "recency_days": 49, "min_searches": 1} and not requested[0].required
    assert "已请新闻核查者补查" in result.metadata["agent"]["notes"][0]
    # The same request twice on one board is refused; without consent there is no request at all.
    assert not context.board.request("web.research", requested[0].arguments)
    quiet = ExecutionContext("run", NormalizedRequest("q", "q"), BudgetClass.PORTFOLIO)
    result = MoveAttributor(ScriptedLLM([LLMResponse(text=finish)]), price_source_factory=lambda: SimpleNamespace(get_prices=_attributor_prices), news=lambda query, day: [], filing_reader=Reader(), memory_recall=None, memory_remember=None, sector_for=lambda ticker: "SMH").run("ARM", quiet, day="2026-07-29", today=date(2026, 9, 9))
    assert "follow_up" not in result.metadata["agent"] and not quiet.board.take_requests()


def test_user_memory_feedback_and_preferences_reach_the_answer(tmp_path, monkeypatch):
    from v2.agent_v2.memory import UserMemory, parse_feedback, render_feedback

    assert parse_feedback("不对") == {"kind": "feedback", "verdict": "bad", "note": ""}
    assert parse_feedback("不对，7/29 是财报日") == {"kind": "feedback", "verdict": "bad", "note": "7/29 是财报日"}
    assert parse_feedback("反馈：口径应该用收盘价") == {"kind": "feedback", "verdict": "bad", "note": "口径应该用收盘价"}
    assert parse_feedback("👍") == {"kind": "feedback", "verdict": "good", "note": ""} and parse_feedback("对。") == {"kind": "feedback", "verdict": "good", "note": ""}
    assert parse_feedback("记住：回答短一点") == {"kind": "preference", "note": "回答短一点"} and parse_feedback("以后都用收盘价口径") == {"kind": "preference", "note": "以后都用收盘价口径"}
    assert parse_feedback("ARM 对不对") is None and parse_feedback("AAPL今天为什么涨？") is None and parse_feedback("") is None

    memory = UserMemory(tmp_path / "memory.jsonl")
    assert memory.record_feedback("chat-1", "bad", "x") is None  # nothing answered yet
    memory.remember_answer("chat-1", question="ARM买入以来跌了这么多，是什么原因？", answer="回答…", run_id="run-1")
    row = memory.record_feedback("chat-1", "bad", "7/29 是财报日")
    assert row["question"] == "ARM买入以来跌了这么多，是什么原因？" and row["run_id"] == "run-1" and row["verdict"] == "bad"
    memory.add_preference("chat-1", "回答短一点")
    memory.add_preference("chat-1", "用收盘价口径")
    memory.add_preference("chat-1", "回答短一点")
    assert memory.preferences("chat-1") == ["回答短一点", "用收盘价口径"] and memory.preferences("chat-2") == [] and len(memory.feedback()) == 1
    text = render_feedback(memory.rows())
    assert "反馈 1 条：好 0、不对 1；偏好 2 条。" in text and "| ARM买入以来跌了这么多，是什么原因？ | 7/29 是财报日 |" in text and "偏好：回答短一点；用收盘价口径" in text

    # The orchestrator hands the session's preferences to the synthesizer and remembers the answer.
    registry = CapabilityRegistry(default_catalog())
    registry.register("market.performance", lambda arguments, context: ToolEnvelope("market.performance", ResultStatus.COMPLETED, subject="ARM", evidence=[EvidenceItem("P", "ARM", "ARM 近 30 天 +1.00%")], metadata={"narrative": "ARM 近 30 天 +1.00%[P]。"}))
    llm = ScriptedLLM([LLMResponse(text="ARM 近 30 天 +1.00%[P]。")])
    agent = AgentV2(catalog=default_catalog(), registry=registry, synthesizer=LLMEvidenceSynthesizer(llm), session=ShortTermSession(), memory=memory)
    result = agent.run("ARM 最近30天表现", session_id="chat-1")
    assert result.request.metadata["preferences"] == ["回答短一点", "用收盘价口径"]
    assert json.loads(llm.calls[0][1]["content"])["user_preferences"] == ["回答短一点", "用收盘价口径"] and "user_preferences" in llm.calls[0][0]["content"]
    assert memory.last_answer("chat-1") == {"question": "ARM 最近30天表现", "answer_digest": "ARM 近 30 天 +1.00%[P]。", "run_id": result.run_id}

    # Telegram feedback commands answer through the bridge without running the agent.
    from v2.bot import agent_v2_bridge as bridge

    monkeypatch.setattr(bridge, "_get_agent", lambda: agent)
    assert bridge.feedback_reply(1, "不对，口径错了") == "这个会话里还没有可以评价的回答。"
    memory.remember_answer("1", question="ARM 最近30天表现", answer="…", run_id="run-9")
    assert bridge.feedback_reply(1, "不对，口径错了").startswith("收到，已记为有误：口径错了")
    assert bridge.feedback_reply(1, "记住：只看半导体") == "记下了：只看半导体。之后的回答会照这个来。"
    assert bridge.feedback_reply(1, "AAPL今天为什么涨？") is None
    assert memory.preferences("1") == ["只看半导体"] and memory.feedback("1")[-1]["note"] == "口径错了"


def test_base1_follow_ups_nested_reader_bare_cancel_remember_comma_briefing_template_forced_finish_retry_and_soft_rules(tmp_path):
    from v2.agent_common.llm import ToolCall
    from v2.agent_v2.adapters.market import _COUNT_LEAK_RULE
    from v2.agent_v2.agents.base import LoopLimits, Tool, ToolLoop, _schema
    from v2.agent_v2.eval.quality import agents_ran, grade
    from v2.agent_v2.eval.quality_cases import QualityCase
    from v2.agent_v2.intent import Intent
    from v2.agent_v2.memory import parse_feedback
    from v2.agent_v2.verification import SOFT_PREFIX

    # 1. The reader the attributor called counts as having run.
    result = _telegram_result("ARM 跌了 [evidence-market-price-42f5c41951e36ef1]。", outcome="clean")
    result.results[0].metadata["agent"] = {"name": "move_attributor", "rounds": 2, "llm_calls": 2, "elapsed_ms": 10, "stop_reason": "finished", "calls": {}, "reader_runs": [{"rounds": 1, "elapsed_ms": 5, "stop_reason": "finished", "filings": 1, "sections_read": 2, "events": 1}]}
    assert agents_ran(result) == {"move_attributor", "filing_reader"}
    nested = grade(QualityCase("t3", "x", criteria=(), expected_agents=("move_attributor", "filing_reader"), must_cite=("market_data",)), result, None)
    assert nested.agents_ok and "missing agents" not in " ".join(nested.problems)

    # 2. A bare "取消" with nothing pending is answered, not turned into an alert removal.
    applied: list = []
    agent = _mutation_agent(applied)
    nothing = agent.run("取消", session_id="chat-9")
    assert nothing.status == RunStatus.COMPLETED and nothing.answer == "当前没有待确认的操作，也没有正在处理的问题。" and nothing.results == [] and nothing.route.reason == "nothing to cancel"
    agent.run("把 NVDA 加入关注列表", session_id="chat-9")
    assert agent.run("取消", session_id="chat-9").status == RunStatus.CANCELLED and not applied

    # 3. "记住" takes a comma, a colon or a space before the instruction.
    assert parse_feedback("记住，回答短一点") == {"kind": "preference", "note": "回答短一点"}
    assert parse_feedback("记住, 用收盘价口径") == {"kind": "preference", "note": "用收盘价口径"} and parse_feedback("记住 只看半导体") == {"kind": "preference", "note": "只看半导体"}
    assert parse_feedback("记住：回答短一点") == {"kind": "preference", "note": "回答短一点"} and parse_feedback("记住") is None

    # 4. The model planner does not replace the briefing template.
    llm = ScriptedLLM([LLMResponse(text='{"tasks":[{"id":"t1","capability":"agent.investigate","arguments":{"question":"美股今天有啥注意的"}}]}')])
    request = normalize_request("美股今天有啥注意的？")
    plan = StructuredLLMPlanner(llm, default_catalog()).plan(request, route(request, intent=Intent(kind="lookup", scope="today", wants=("briefing", "macro"), source="model")))
    assert [task.capability for task in plan.tasks] == ["macro.overview", "account.earnings_schedule", "account.risk", "state.read"] and not llm.calls

    # 5. A provider that returns nothing for the named tool_choice gets the turn again without it.
    class Lazy(ToolLoop):
        def __init__(self, llm):
            super().__init__(llm, LoopLimits(max_rounds=1, max_seconds=30), tools=[Tool("look", "看。", _schema({}), lambda a: "看到了")], finish_parameters=_schema({"answer": {"type": "string"}}, ["answer"]))

    llm = ScriptedLLM([LLMResponse(tool_calls=[ToolCall(id="c1", name="look", arguments={}, raw_arguments="{}")]), LLMResponse(text=""), LLMResponse(tool_calls=[ToolCall(id="c2", name="finish", arguments={"answer": "x"}, raw_arguments="{}")])])
    outcome = Lazy(llm).run("s", "t", finish_prompt="finish now")
    assert outcome.finished and llm.tool_choices == [None, {"type": "function", "function": {"name": "finish"}}, None]

    class Refusing(ScriptedLLM):
        def complete(self, messages, tools=None, tool_choice=None):
            if tool_choice is not None:
                self.tool_choices.append(tool_choice)
                raise RuntimeError("HTTP 400: tool_choice not supported")
            return super().complete(messages, tools, tool_choice)

    llm = Refusing([LLMResponse(tool_calls=[ToolCall(id="c1", name="look", arguments={}, raw_arguments="{}")]), LLMResponse(tool_calls=[ToolCall(id="c2", name="finish", arguments={"answer": "x"}, raw_arguments="{}")])])
    assert Lazy(llm).run("s", "t", finish_prompt="finish now").finished and len(llm.calls) == 2

    # 6. The count-leak rule is soft: reported with a prefix, not grounds to reject the draft.
    assert _COUNT_LEAK_RULE["soft"] is True
    item = EvidenceItem("A1", "ARM", "ARM 三个下跌日的归因。")
    result = ToolEnvelope("market.explain_move", ResultStatus.COMPLETED, subject="ARM", evidence=[item], metadata={"answer_constraints": [dict(_COUNT_LEAK_RULE), {"forbid_claim": "把候选说成原因", "warning": "候选归因被表述为已确认原因"}]})
    soft_only = verify_answer("三个下跌日都没有确认的高置信度驱动 [A1]。", [item], answer_mode=AnswerMode.RESEARCH_GROUNDED, results=[result], judge=lambda items: {i["id"]: "" for i in items if i["claim"].startswith("把归因系统")})
    assert soft_only.ok and list(soft_only.warnings) == [SOFT_PREFIX + "将内部归因计数直接暴露给用户"]
    both = verify_answer("三个下跌日都没有确认的高置信度驱动 [A1]。", [item], answer_mode=AnswerMode.RESEARCH_GROUNDED, results=[result], judge=lambda items: {i["id"]: "" for i in items})
    assert not both.ok and sorted(both.warnings) == sorted([SOFT_PREFIX + "将内部归因计数直接暴露给用户", "候选归因被表述为已确认原因"])


def test_telegram_queues_a_chat_and_lets_cancel_through_and_forced_finish_remembers_a_refusing_provider(monkeypatch):
    from v2.agent_common.llm import ToolCall
    from v2.agent_v2.agents.base import LoopLimits, Tool, ToolLoop, _schema
    from v2.bot import agent_v2_bridge as bridge

    # The bridge runs one question of a chat at a time; a second one is told it queued; a cancel word bypasses the queue.
    started, release = asyncio.Event(), asyncio.Event()
    seen: list[str] = []

    class FakeFacade:
        def __init__(self, agent):
            pass

        async def handle(self, message, transport, *, allow_web=False, cancel_event=None):
            seen.append(message.text)
            if message.text == "slow":
                started.set()
                await release.wait()
                return "slow-done"
            return "fast-done"

    class Message:
        def __init__(self, text):
            self.text, self.replies, self.message_id = text, [], 1

        async def reply_html(self, text, **kwargs):
            self.replies.append(text)
            return self

        async def edit_text(self, text, **kwargs):
            self.replies.append(text)

    def update_for(text):
        return type("Update", (), {"message": Message(text), "effective_chat": type("Chat", (), {"id": 5})()})()

    monkeypatch.setattr(bridge, "TelegramFacade", FakeFacade)
    monkeypatch.setattr(bridge, "_get_agent", lambda: object())
    monkeypatch.setattr(bridge, "TelegramBotTransport", lambda context, placeholder, web_requested=False: None)
    monkeypatch.setattr(bridge, "feedback_reply", lambda chat_id, text: None)
    bridge._CHAT_LOCKS.pop(5, None)

    async def scenario():
        first, second, cancel = update_for("slow"), update_for("next"), update_for("取消")
        task1 = asyncio.create_task(bridge.handle_agent_v2(first, None, "slow"))
        await started.wait()
        task2 = asyncio.create_task(bridge.handle_agent_v2(second, None, "next"))
        await asyncio.sleep(0)
        assert second.message.replies == [bridge.QUEUED_NOTICE] and seen == ["slow"]
        assert await bridge.handle_agent_v2(cancel, None, "取消") is None and cancel.message.replies == ["正在停止当前的处理…"] and bridge._ACTIVE_RUNS[5].is_set()
        release.set()
        assert await task1 == "slow-done" and await task2 == "fast-done" and seen == ["slow", "next"] and 5 not in bridge._ACTIVE_RUNS

    asyncio.run(scenario())

    # A provider that rejects the named tool_choice is not asked again; a bare finish payload in text is the finish call.
    class Lazy(ToolLoop):
        def __init__(self, llm):
            super().__init__(llm, LoopLimits(max_rounds=1, max_seconds=30), tools=[Tool("look", "看。", _schema({}), lambda a: "看到了")], finish_parameters=_schema({"answer": {"type": "string"}}, ["answer"]))

    class Refusing(ScriptedLLM):
        def complete(self, messages, tools=None, tool_choice=None):
            if tool_choice is not None:
                self.tool_choices.append(tool_choice)
                raise RuntimeError('HTTP 400: {"error":{"message":"Thinking mode does not support this tool_choice"}}')
            return super().complete(messages, tools, tool_choice)

    look = LLMResponse(tool_calls=[ToolCall(id="c1", name="look", arguments={}, raw_arguments="{}")])
    llm = Refusing([look, LLMResponse(text='{"answer": "x"}'), look, LLMResponse(text='{"answer": "y"}')])
    first = Lazy(llm).run("s", "t", finish_prompt="finish now")
    assert first.finished and first.final == {"action": "finish", "answer": "x"} and llm.tool_choice_unsupported is True and len(llm.tool_choices) == 3
    second = Lazy(llm).run("s", "t", finish_prompt="finish now")
    assert second.finished and second.final["answer"] == "y" and len(llm.tool_choices) == 5 and llm.tool_choices[-2:] == [None, None]


def test_market_level_questions_second_repair_on_progress_and_quality_show(tmp_path):
    from v2.agent_v2.eval.quality import QualityJudge, grade, read_rows, record, render_case
    from v2.agent_v2.eval.quality_cases import QualityCase
    from v2.agent_v2.intent import MARKET_TICKERS, Intent

    # "今天美股行情如何" is the market, not the account: the macro board plus the index ETFs.
    market = Intent(kind="lookup", scope="today", wants=("market", "performance"), source="model")
    request = normalize_request("今天美股行情如何？")
    plan = RulePlanner().plan(request, route(request, intent=market))
    assert [task.capability for task in plan.tasks] == ["macro.overview", "market.performance", "market.performance", "market.performance"]
    assert [task.arguments["ticker"] for task in plan.tasks[1:]] == list(MARKET_TICKERS) and plan.route == RouteKind.FAST_LOOKUP
    offline = RulePlanner().plan(request, route(request))  # the recorded label plans the same without a model
    assert [task.capability for task in offline.tasks] == [task.capability for task in plan.tasks]

    # A repair that fixed part of the draft earns a second round; one that did not still falls back.
    llm = ScriptedLLM([LLMResponse(text="NVDA 收入增长 20%，利润率 30%。[E1]"), LLMResponse(text="NVDA 收入增长 10%，利润率 30%。[E1]"), LLMResponse(text="NVDA 收入增长 10%。[E1]")])
    request, plan, results, evidence = _research_fixture()
    synthesizer = LLMEvidenceSynthesizer(llm)
    answer = synthesizer.synthesize(request, plan, results, evidence)
    assert answer == "NVDA 收入增长 10%。[E1]" and len(llm.calls) == 3 and synthesizer.last_outcome == "repaired"
    assert [attempt["stage"] for attempt in synthesizer.diagnostics()["attempts"]] == ["draft", "repair", "repair2"]
    llm = ScriptedLLM([LLMResponse(text="NVDA 收入增长 20%。[E1]"), LLMResponse(text="NVDA 收入增长 25%。[E1]"), LLMResponse(text="NVDA 收入增长 10%。[E1]")])
    answer = LLMEvidenceSynthesizer(llm).synthesize(request, plan, results, evidence)
    assert "20%" not in answer and "25%" not in answer and len(llm.calls) == 2  # no progress: no second round

    # quality show: the answer and the verdict of one recorded case.
    case = QualityCase("t9", "MU和SNDK哪个更值得购买？", criteria=("同口径比较", "指出缺口"), forbidden=("无条件买入",), expected_route=RouteKind.RESEARCH)
    result = _telegram_result("MU 比 SNDK 便宜 [evidence-market-price-42f5c41951e36ef1]。", outcome="clean")
    score = grade(case, result, lambda q, a, c, f: {"criteria": [{"index": 0, "met": False}, {"index": 1, "met": True, "quote": "缺口"}], "forbidden": [{"index": 0, "asserted": False}]})
    record(case, result, score, label="t-run", path=tmp_path / "q.jsonl")
    text = render_case(read_rows(tmp_path / "q.jsonl"), "t9")
    assert text.startswith("# t9 · t-run") and "✗ 同口径比较" in text and "✓ 指出缺口（“缺口”）" in text and "## 回答" in text and "MU 比 SNDK 便宜" in text
    assert render_case([], "t9").startswith("没有 t9 的记录") and render_case(read_rows(tmp_path / "q.jsonl"), "t9", label="other").startswith("没有")


def test_research_with_failed_core_modules_is_partial_data():
    from v2.agent_v2.adapters.research import _envelope

    hollow = {"ticker": "SNDK", "status": "COMPLETED", "run_id": "r1", "scores": {"overall": 36}, "production_diagnostics": {"modules": {"valuation": {"status": "FAILED", "completeness": 0.0}, "fundamental": {"status": "FAILED", "completeness": 0.0}, "expectations": {"status": "COMPLETED"}}}}
    result = _envelope(hollow, "research.stock")
    assert result.status == ResultStatus.PARTIAL_DATA and any(value.startswith("核心模块失败（fundamental、valuation）") for value in result.limitations)
    fine = _envelope({**hollow, "production_diagnostics": {"modules": {"valuation": {"status": "COMPLETED"}}}}, "research.stock")
    assert fine.status == ResultStatus.COMPLETED and not any("核心模块失败" in value for value in fine.limitations)


def test_quality_repeat_majority_preceding_turns_length_bound_and_case_sets(tmp_path, monkeypatch):
    from v2.agent_v2.eval import quality
    from v2.agent_v2.eval.quality import _fold_attempts, read_rows, render, run_cases, summarize
    from v2.agent_v2.eval.quality_cases import QUALITY_CASES, QualityCase
    from v2.agent_v2.eval.quality_holdout import HOLDOUT_CASES
    from v2.agent_v2.memory import UserMemory

    # Every case plans offline without a model, ids are unique across both sets, the hold-out set is marked.
    ids = [case.id for case in (*QUALITY_CASES, *HOLDOUT_CASES)]
    assert len(ids) == len(set(ids)) and len(QUALITY_CASES) >= 35 and len(HOLDOUT_CASES) >= 10
    assert all(case.set == "holdout" for case in HOLDOUT_CASES) and all(case.set == "dev" for case in QUALITY_CASES)
    assert sum(1 for case in QUALITY_CASES if "quick" in case.tags) >= 5
    for case in (*QUALITY_CASES, *HOLDOUT_CASES):
        for text in (*case.preceding, case.question):
            request = normalize_request(text)
            RulePlanner().plan(request, route(request))

    # Repeated attempts fold into a majority verdict; the preceding turns run in the same session; a preference goes to memory.
    ledger = tmp_path / "quality.jsonl"
    registry = CapabilityRegistry(default_catalog())
    seen: list[tuple[str, str]] = []

    def news(arguments, context):
        seen.append((context.request.session_id if hasattr(context, "request") else "", arguments.get("ticker", "")))
        return ToolEnvelope("research.stock", ResultStatus.COMPLETED, subject=arguments.get("ticker", ""), evidence=[EvidenceItem("W", arguments.get("ticker", ""), "news", source_id="web:x")], metadata={"narrative": "新闻 [W]。"})

    registry.register("web.research", news)
    registry.register("research.stock", news)  # offline the rewritten follow-up plans a research read; the ticker is what matters
    registry.register("market.explain_move", lambda arguments, context: ToolEnvelope("market.explain_move", ResultStatus.COMPLETED, subject="AAPL", evidence=[EvidenceItem("M", "AAPL", "AAPL 今日 +1%", source_id="market_data")], metadata={"narrative": "AAPL 今日 +1%[M]。"}))
    memory = UserMemory(tmp_path / "memory.jsonl")
    agent = AgentV2(catalog=default_catalog(), registry=registry, session=ShortTermSession(), memory=memory, config=AgentV2Config(record_sub_agents=False))
    flaky = iter([True, False, True])
    judge = lambda q, a, c, f: {"criteria": [{"index": i, "met": next(flaky, True)} for i in range(len(c))], "forbidden": []}
    short = QualityCase("t_pref", "AAPL今天为什么涨？", criteria=("有涨跌幅",), preceding=("记住，回答短一点",), max_chars=5, allow_web=False)
    rows = run_cases(agent, (short,), judge, label="r", path=ledger, repeat=3)
    assert [row["attempt"] for row in rows] == [1, 2, 3] and [row["score"]["passed"] for row in rows] == [False, False, False]  # every attempt breaks the 5-char bound
    assert all(any(problem.startswith("answer ") and "> 5" in problem for problem in row["score"]["problems"]) for row in rows)
    assert memory.preferences("quality-r-t_pref-1") == ["回答短一点"]
    verdict = _fold_attempts(rows)
    assert verdict["attempts"] == 3 and verdict["passes"] == 0 and not verdict["passed"]
    majority = _fold_attempts([{"score": {"passed": True}}, {"score": {"passed": False, "problems": ["x"]}}, {"score": {"passed": True}}])
    assert majority["passed"] and majority["passes"] == 2 and majority["problems"] == ["x"]
    assert not _fold_attempts([{"score": {"passed": True}}, {"score": {"passed": False}}])["passed"]  # a tie is not a pass

    follow = QualityCase("t_follow", "那它最近有什么新闻？", criteria=(), preceding=("AAPL今天为什么涨？",), allow_web=True)
    rows = run_cases(agent, (follow,), None, label="r2", path=ledger)
    assert rows[0]["score"]["passed"] and rows[0]["question"] == "那它最近有什么新闻？"
    assert seen and seen[-1][1] == "AAPL"  # the pronoun resolved against the preceding turn in the same session

    summary = summarize(read_rows(ledger), runs=2)
    assert [(run["label"], run["cases"], run["attempts"], run["passed"]) for run in summary["runs"]] == [("r", 1, 3, 0), ("r2", 1, 1, 1)]
    text = render(summary)
    assert "| 每题次数 |" in text and "| t_pref | ✗ 0/3次 | — |" in text and "| t_follow | — | ✓ 0/0 |" in text

    # The CLI picks the set and the smoke subset.
    picked: dict[str, tuple[str, ...]] = {}

    def fake_run_cases(agent_, cases, judge_, **kwargs):
        picked["ids"] = tuple(case.id for case in cases)
        picked["kwargs"] = kwargs
        return []

    monkeypatch.setattr(quality, "run_cases", fake_run_cases)
    monkeypatch.setattr(quality, "_live_agent", lambda **kwargs: type("A", (), {"synthesizer": None})())
    monkeypatch.setenv("AGENT_V2_QUALITY_LEDGER", str(ledger))
    quality.main(["run", "--label", "x", "--set", "holdout", "--repeat", "2", "--parallel", "2"])
    assert picked["ids"] == tuple(case.id for case in HOLDOUT_CASES) and picked["kwargs"]["repeat"] == 2 and picked["kwargs"]["parallel"] == 2
    quality.main(["run", "--label", "x", "--quick"])
    assert picked["ids"] == tuple(case.id for case in QUALITY_CASES if "quick" in case.tags)
    quality.main(["run", "--label", "x", "--set", "all"])
    assert len(picked["ids"]) == len(QUALITY_CASES) + len(HOLDOUT_CASES)


def test_p1_first_pass_rate_restated_figures_compare_rows_filings_plan_short_preference_and_risk_modules():
    from v2.agent_v2.adapters.research import _FOCUS_MODULES, _comparison_table
    from v2.agent_v2.intent import Intent
    from v2.agent_v2.llm import _problem_set, short_answer_limit

    # 1. A figure grounded by an earlier cited sentence may be restated later without a citation.
    peak = EvidenceItem("PK", "TSLA", "TSLA 从 2025-12-16 高点 489.88 美元到 2026-07-29 低点 298.32 美元回撤 -39.10%。", source_id="market_data")
    ret = EvidenceItem("RT", "TSLA", "TSLA 近 1 月 +11.58%。", source_id="market_data")
    result = ToolEnvelope("market.drawdown", ResultStatus.COMPLETED, subject="TSLA", evidence=[peak, ret], metadata={"require_cited_numbers": True})
    restated = "特斯拉这轮回撤是 -39.10% [PK]。近 1 月 +11.58% [RT]。所以 -39.10% 说的是去年 12 月到今年 7 月那段，不是现在。"
    assert verify_answer(restated, [peak, ret], answer_mode=AnswerMode.RESEARCH_GROUNDED, results=[result]).ok
    wrong_cite = "特斯拉这轮回撤是 -39.10% [PK]。近 1 月 +11.58%，回撤 -39.10% [RT]。"
    assert verify_answer(wrong_cite, [peak, ret], answer_mode=AnswerMode.RESEARCH_GROUNDED, results=[result]).ok  # 39.10 was grounded a sentence earlier
    invented = "特斯拉这轮回撤是 -39.10% [PK]。所以 -42.00% 是那段的跌幅。"
    report = verify_answer(invented, [peak, ret], answer_mode=AnswerMode.RESEARCH_GROUNDED, results=[result])
    assert not report.ok and any("行情事实缺少邻近引用" in warning for warning in report.warnings)

    # 2. The problem set compares kinds: another invented number is the same problem; a different kind is progress.
    class R:
        def __init__(self, numbers=(), unknown=(), warnings=()):
            self.ungrounded_numbers, self.unknown_citations, self.warnings = tuple(numbers), tuple(unknown), tuple(warnings)

    assert _problem_set(R(numbers=("20",))) == _problem_set(R(numbers=("25",)))
    assert _problem_set(R(warnings=("未确认直接驱动时展示了过多弱候选线索（AMD：只保留 [a]）",))) != _problem_set(R(warnings=("这段涨幅的回答必须引用同期行业基准对比那条证据 [b]",)))
    assert _problem_set(R(warnings=("（提示）将内部归因计数直接暴露给用户",))) == frozenset()

    # 3. Compare rows: one citeable side-by-side item per metric two or more tickers carry.
    mu = ToolEnvelope("research.stock", ResultStatus.COMPLETED, subject="MU", evidence=[EvidenceItem("m1", "MU", "MU 市盈率 12.3 倍", metadata={"metrics": {"pe_ratio": 12.3}}), EvidenceItem("m2", "MU", "MU 营收增速 44%", metadata={"metrics": {"revenue_growth": 0.44}})])
    sndk = ToolEnvelope("research.stock", ResultStatus.COMPLETED, subject="SNDK", evidence=[EvidenceItem("s1", "SNDK", "SNDK 市盈率 30.1 倍", metadata={"metrics": {"pe_ratio": 30.1}}), EvidenceItem("s2", "SNDK", "SNDK 毛利率 27%", metadata={"metrics": {"gross_margin": 0.27}})])
    rows = _comparison_table([mu, sndk])
    assert [row.claim for row in rows] == ["同口径对照 市盈率（TTM）：MU 12.3、SNDK 30.1。"] and rows[0].metadata["citation_kind"] == "metrics" and rows[0].metadata["from"] == ["m1", "s1"]
    assert _comparison_table([mu]) == []

    # 4. A filings question lists the dated filings first, then the engine's filings module.
    request = normalize_request("MU最近有什么SEC申报？")
    plan = RulePlanner().plan(request, route(request, intent=Intent(kind="lookup", scope="recent", wants=("filings",), tickers=("MU",), source="model")))
    assert [(task.capability, task.arguments.get("focus")) for task in plan.tasks] == [("filings.recent", None), ("filings.read_events", None), ("research.stock", "filings")] and plan.tasks[0].arguments["forms"][0] == "8-K"
    assert plan.tasks[1].depends_on == ("filings-recent-MU",) and not plan.tasks[1].required and plan.tasks[1].arguments["max_filings"] == 2
    assert [task.capability for task in RulePlanner().plan(request, route(request)).tasks] == ["filings.recent", "filings.read_events", "research.stock"]  # the recorded label

    # 5. "回答短一点" becomes a character limit and a compression round; a long answer stays when the short one fails.
    assert short_answer_limit(["回答短一点"]) == 500 and short_answer_limit(["用收盘价口径"]) == 0 and short_answer_limit(["keep it brief"]) == 500
    request, plan, results, evidence = _research_fixture()
    long_answer = "NVDA 收入增长 10%。[E1] " + "这是一段解释。" * 80
    llm = ScriptedLLM([LLMResponse(text=long_answer), LLMResponse(text="NVDA 收入增长 10%。[E1] 简短版。")])
    request.metadata["preferences"] = ["回答短一点"]
    synthesizer = LLMEvidenceSynthesizer(llm)
    answer = synthesizer.synthesize(request, plan, results, evidence)
    assert answer == "NVDA 收入增长 10%。[E1] 简短版。" and "不超过 500 个字符" in llm.calls[0][0]["content"] and "压缩到 500 个字符以内" in llm.calls[1][-1]["content"]
    assert [attempt["stage"] for attempt in synthesizer.diagnostics()["attempts"]] == ["draft", "shorten"]
    llm = ScriptedLLM([LLMResponse(text=long_answer), LLMResponse(text="NVDA 收入增长 99%。[E1]")])
    assert LLMEvidenceSynthesizer(llm).synthesize(request, plan, results, evidence) == long_answer  # the short one invented a figure

    # 6. The risk focus no longer pulls the engine's supply-chain discovery.
    assert "supply_chain" not in _FOCUS_MODULES["risk"] and "risk" not in _FOCUS_MODULES["risk"] and "sec" in _FOCUS_MODULES["risk"]


def test_performance_fan_out_gets_a_ranking_table_and_compare_rows_read_module_metrics():
    from v2.agent_v2.adapters.research import _comparison_table, _envelope
    from v2.agent_v2.execution import _fan_out_table
    from v2.agent_v2.intent import Intent
    from v2.agent_v2.llm import visible_length

    def perf(ticker, returns, intraday=False):
        return ToolEnvelope("market.performance", ResultStatus.COMPLETED, subject=ticker, metrics={"returns": returns, "is_intraday": intraday}, evidence=[EvidenceItem(f"r-{ticker}", ticker, "x", source_id="market_data")])

    table = _fan_out_table("performance-each", [perf("AAPL", {"1d": 0.01, "5d": 0.021, "1m": 0.10}), perf("PLTR", {"1d": -0.02, "5d": -0.0815, "1m": 0.03}), perf("AMD", {"1d": 0.005, "5d": 0.1317}, intraday=True)])
    assert table is not None and table.metadata["fan_out_table"] and [item.metadata["window"] for item in table.evidence] == ["1d", "5d", "1m"]
    five = next(item for item in table.evidence if item.metadata["window"] == "5d")
    assert five.claim == "近 5 日回报排序：AMD +13.17%、AAPL +2.10%、PLTR -8.15%；最高 AMD，最低 PLTR。" and five.id == "evidence-ranking-5d-performance-each"
    assert "盘中口径" in table.evidence[0].claim and _fan_out_table("p", [perf("AAPL", {"5d": 0.1})]) is None

    # The engine appends the table after the fan-out; the week's ranking fans out over every holding without the since-purchase order.
    registry = CapabilityRegistry(default_catalog())
    card = ToolEnvelope("account.portfolio", ResultStatus.COMPLETED, subject="me", evidence=[EvidenceItem("C", "me", "card", source_id="account.portfolio")], metadata={"tickers": ["AAPL", "PLTR", "AMD"], "positions": [{"ticker": "AAPL", "pl_pct": 0.3}, {"ticker": "PLTR", "pl_pct": -0.2}, {"ticker": "AMD", "pl_pct": 0.1}]})
    registry.register("account.portfolio", lambda a, c: card)
    registry.register("market.performance", lambda a, c: perf(a["ticker"], {"5d": {"AAPL": 0.02, "PLTR": -0.08, "AMD": 0.13}[a["ticker"]]}))
    week = Intent(kind="lookup", scope="recent", wants=("portfolio", "ranking", "performance"), portfolio_scope=True, rank="high", source="model")
    request = normalize_request("持仓里这周谁涨得最好？")
    plan = RulePlanner().plan(request, route(request, intent=week))
    each = next(task for task in plan.tasks if task.capability == "market.performance")
    assert "rank" not in each.fan_out and each.fan_out["max"] == 12
    outcome = ExecutionEngine(registry).run(plan, ExecutionContext("run", request, BudgetClass.PORTFOLIO))
    tables = [r for r in outcome.results if r.metadata.get("fan_out_table")]
    assert len(tables) == 1 and "最高 AMD，最低 PLTR" in tables[0].evidence[0].claim and outcome.ledger.get(tables[0].evidence[0].id) is not None
    since = Intent(kind="lookup", scope="none", wants=("portfolio", "ranking"), portfolio_scope=True, rank="low", source="model")
    plan = RulePlanner().plan(request, route(request, intent=since))
    assert [task.capability for task in plan.tasks] == ["account.portfolio"]  # the card alone answers "哪只跌得最惨"

    # Compare rows read the engine's own metric names and the modules' metrics dicts.
    mu = _envelope({"ticker": "MU", "status": "COMPLETED", "run_id": "r1", "evidence_index": [{"id": "e-mu-1", "ticker": "MU", "module": "valuation", "claim": "TTM P/E is 12.3x.", "metrics": {"pe_ttm": 12.3}, "source_ids": ["fd_metrics"], "verified": True}], "sources": [{"id": "fd_metrics", "title": "m", "url": ""}], "modules": {"valuation": {"metrics": {"forward_pe": 9.8, "ev_ebitda": 6.1, "note": "x"}}, "earnings": {"metrics": {"beat_rate": 0.75}}}, "production_diagnostics": {"modules": {}}}, "research.stock")
    sndk = _envelope({"ticker": "SNDK", "status": "COMPLETED", "run_id": "r2", "evidence_index": [{"id": "e-s-1", "ticker": "SNDK", "module": "valuation", "claim": "TTM P/E is 30.1x.", "metrics": {"pe_ttm": 30.1}, "source_ids": ["fd_metrics"], "verified": True}], "sources": [{"id": "fd_metrics", "title": "m", "url": ""}], "modules": {"valuation": {"metrics": {"forward_pe": 22.0}}}, "production_diagnostics": {"modules": {}}}, "research.stock")
    assert mu.metadata["module_metrics"] == {"valuation": {"forward_pe": 9.8, "ev_ebitda": 6.1}, "earnings": {"beat_rate": 0.75}}
    claims = [row.claim for row in _comparison_table([mu, sndk])]
    assert claims == ["同口径对照 市盈率（TTM）：MU 12.3、SNDK 30.1。", "同口径对照 前瞻市盈率：MU 9.8、SNDK 22.0。"]

    # The visible length ignores citation markers.
    assert visible_length("AAPL 涨了 [evidence-market-price-42f5c41951e36ef1]。") == len("AAPL 涨了 。")


def test_capability_ledger_and_health_report_and_known_data_fixes(tmp_path, monkeypatch):
    from v2.agent_v2.eval import capability_ledger as ledger
    from v2.sec.eight_k_parser import get_item_text

    # One row per capability result; the report groups by capability, marks hollow results and names the worst first.
    registry = CapabilityRegistry(default_catalog())
    registry.register("market.performance", lambda a, c: ToolEnvelope("market.performance", ResultStatus.COMPLETED, subject="ARM", evidence=[EvidenceItem("P", "ARM", "ARM 近 30 天 +1.00%", source_id="market_data")], metadata={"narrative": "ARM 近 30 天 +1.00%[P]。"}))
    registry.register("research.stock", lambda a, c: ToolEnvelope("research.stock", ResultStatus.PARTIAL_DATA, subject="ARM", limitations=["核心模块失败（valuation）：估值和基本面数字缺失"], errors=[]))
    agent = AgentV2(catalog=default_catalog(), registry=registry, config=AgentV2Config(record_sub_agents=False, record_capabilities=True))
    monkeypatch.setenv("AGENT_V2_CAPABILITY_LEDGER", str(tmp_path / "cap.jsonl"))
    result = agent.run("ARM 最近30天表现", session_id="quality-x-q1-1")
    rows = ledger.read_rows(tmp_path / "cap.jsonl")
    assert [row["capability"] for row in rows] == ["market.performance"] and rows[0]["channel"] == "quality" and rows[0]["ok"] and not rows[0]["hollow"] and rows[0]["run_id"] == result.run_id
    request = normalize_request("分析 ARM 估值")
    research = agent.run("分析 ARM 估值", session_id="chat-1")
    rows = ledger.read_rows(tmp_path / "cap.jsonl")
    hollow = [row for row in rows if row["capability"] == "research.stock"]
    assert hollow and hollow[0]["hollow"] and hollow[0]["status"] == "partial_data" and hollow[0]["limitation"].startswith("核心模块失败") and hollow[0]["channel"] == ""
    summary = ledger.summarize(rows)
    by_name = {row["capability"]: row for row in summary["capabilities"]}
    assert by_name["research.stock"]["hollow"] == 1 and by_name["research.stock"]["ok"] == 0 and by_name["market.performance"]["ok"] == 1
    assert summary["capabilities"][0]["capability"] == "research.stock"  # the least reliable first
    text = ledger.render(summary, since_days=1)
    assert "| research.stock | 1 | 0 | 1 |" in text and "| 需要看的能力 | 正常率 | 最近一次异常 |" in text and "核心模块失败" in text
    assert ledger.render(ledger.summarize([])).startswith("# 数据源健康报告") and ledger.main(["--since", "1", "--path", str(tmp_path / "cap.jsonl")]) == 0
    assert ledger.read_rows(tmp_path / "cap.jsonl", channel="quality") and not ledger.read_rows(tmp_path / "cap.jsonl", channel="web")
    monkeypatch.setenv("AGENT_V2_CAPABILITY_LEDGER", str(tmp_path / "off.jsonl"))
    AgentV2(catalog=default_catalog(), registry=registry, config=AgentV2Config(record_sub_agents=False, record_capabilities=False)).run("ARM 最近30天表现")
    assert not (tmp_path / "off.jsonl").exists()

    # edgartools exposes EightK.text() as a method now; the parser calls it instead of matching the bound method.
    class Modern:
        def text(self):
            return "Item 2.02 Results of Operations\nRevenue was $1.0B.\nItem 9.01 Exhibits"

    class Legacy:
        text = "Item 2.02 Results of Operations\nRevenue was $1.0B.\nItem 9.01 Exhibits"

    assert "Revenue was $1.0B." in get_item_text(Modern(), "2.02") and "Revenue was $1.0B." in get_item_text(Legacy(), "2.02") and get_item_text(Modern(), "5.02") == ""

    # ETFs are not asked for an earnings calendar; a ticker whose calendar came back empty is not asked again for a day.
    pytest.importorskip("yfinance")
    from v2.earnings import calendar as earnings_calendar

    assert not earnings_calendar.is_supported_ticker("IVV") and not earnings_calendar.is_supported_ticker("NUGT") and earnings_calendar.is_supported_ticker("AAPL")
    earnings_calendar._EMPTY_CALENDAR.clear()
    monkeypatch.setattr(earnings_calendar, "_fetch_one", lambda ticker: None)
    batch = earnings_calendar.get_upcoming_batch(["ZZZZ", "IVV"])
    assert batch.skipped_empty == ["ZZZZ"] and batch.skipped_unsupported == ["IVV"]
    assert not earnings_calendar.is_supported_ticker("ZZZZ")  # remembered as empty
    earnings_calendar._EMPTY_CALENDAR.clear()
    assert earnings_calendar.is_supported_ticker("ZZZZ")


def test_merge_gate_verdict_and_offline_steps(monkeypatch, capsys):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("agent_v2_gate", Path(__file__).resolve().parents[2] / "scripts" / "agent_v2_gate.py")
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    assert gate.verdict([("a", True, ""), ("b", True, "")]) and not gate.verdict([("a", True, ""), ("b", False, "x")])
    ok, detail = gate.run_offline_eval()
    assert ok and detail == "39/39"
    monkeypatch.setattr(gate, "run_unit_tests", lambda: (True, "1 passed"))
    monkeypatch.setattr(gate, "run_quick_quality", lambda label, min_pass: (False, "4/6 passed, need 5", {}))
    assert gate.main(["--json"]) == 0
    out = capsys.readouterr().out
    assert "[PASS] unit tests: 1 passed" in out and "[PASS] offline eval: 39/39" in out and out.strip().endswith("gate: PASS (2 steps, 0s)")
    assert gate.main(["--live", "--skip-eval"]) == 1 and "gate: FAIL" in capsys.readouterr().out


def test_warm_imports_runs_once_and_tolerates_missing_libraries(monkeypatch):
    from v2.agent_v2 import warmup

    warmup._done.clear()
    missing = warmup.warm_imports(("json", "no_such_library_xyz"))
    assert missing == ["no_such_library_xyz"] and warmup._done == {"json", "no_such_library_xyz"}
    assert warmup.warm_imports(("json", "no_such_library_xyz")) == []  # already attempted: not retried
    warmup._done.clear()


def test_low_confidence_classification_asks_one_question_and_reads_the_next_message_as_the_answer():
    from v2.agent_v2.intent import Intent, parse_intent

    assert parse_intent({"kind": "lookup", "confidence": 0.4, "clarification": "  是想看 TSLA 的行情、新闻还是研究？ "}).clarification == "是想看 TSLA 的行情、新闻还是研究？"

    class Unsure:
        def __init__(self):
            self.texts: list[str] = []

        def classify(self, request):
            self.texts.append(request.text)
            if "补充" in request.text:
                return Intent(kind="lookup", scope="recent", wants=("news",), tickers=("TSLA",), confidence=0.9, source="model")
            return Intent(kind="research", wants=("overview",), tickers=("TSLA",), confidence=0.35, source="model", clarification="是想看 TSLA 的行情、新闻还是研究？")

    registry = CapabilityRegistry(default_catalog())
    registry.register("web.research", lambda a, c: ToolEnvelope("web.research", ResultStatus.COMPLETED, subject="TSLA", evidence=[EvidenceItem("W", "TSLA", "news", source_id="web:x")], metadata={"narrative": "新闻 [W]。"}))
    registry.register("filings.recent", lambda a, c: ToolEnvelope("filings.recent", ResultStatus.COMPLETED, subject="TSLA"))
    registry.register("market.anomaly_history", lambda a, c: ToolEnvelope("market.anomaly_history", ResultStatus.COMPLETED, subject="TSLA"))
    classifier = Unsure()
    agent = AgentV2(catalog=default_catalog(), registry=registry, classifier=classifier, session=ShortTermSession(), config=AgentV2Config(record_sub_agents=False, record_capabilities=False))
    asked = agent.run("特斯拉最近", session_id="c1")
    assert asked.status == RunStatus.WAITING_CLARIFICATION and asked.answer == "是想看 TSLA 的行情、新闻还是研究？" and asked.results == [] and asked.plan.assumptions[0].startswith("clarification: confidence 0.35")
    answered = agent.run("新闻", session_id="c1")
    assert classifier.texts[-1] == "特斯拉最近（补充：新闻）" and answered.request.metadata["clarified"] is True and answered.request.metadata["clarification"] == asked.answer
    assert answered.status in {RunStatus.COMPLETED, RunStatus.PARTIAL} and {r.capability for r in answered.results} >= {"web.research", "filings.recent"}  # partial: the web is off in this test
    assert agent.session.pop_clarification("c1") is None  # consumed

    # A merged turn is never asked about again, even when the classifier stays unsure.
    class StillUnsure(Unsure):
        def classify(self, request):
            self.texts.append(request.text)
            return Intent(kind="research", wants=("overview",), tickers=("TSLA",), confidence=0.3, source="model", clarification="哪方面？")

    registry.register("research.stock", lambda a, c: ToolEnvelope("research.stock", ResultStatus.COMPLETED, subject="TSLA", evidence=[EvidenceItem("R", "TSLA", "研究", source_id="research_engine")], metadata={"narrative": "研究 [R]。"}))
    agent = AgentV2(catalog=default_catalog(), registry=registry, classifier=StillUnsure(), session=ShortTermSession(), config=AgentV2Config(record_sub_agents=False, record_capabilities=False))
    assert agent.run("特斯拉最近", session_id="c2").status == RunStatus.WAITING_CLARIFICATION
    second = agent.run("随便", session_id="c2")
    assert second.status != RunStatus.WAITING_CLARIFICATION and second.results

    # "取消" after a question drops it; knowledge questions and recorded labels are never asked about; the threshold can be turned off.
    agent = AgentV2(catalog=default_catalog(), registry=registry, classifier=Unsure(), session=ShortTermSession(), config=AgentV2Config(record_sub_agents=False, record_capabilities=False))
    agent.run("特斯拉最近", session_id="c3")
    dropped = agent.run("取消", session_id="c3")
    assert dropped.status == RunStatus.CANCELLED and dropped.answer == "已取消，那个问题不再处理。" and agent.session.pop_clarification("c3") is None
    off = AgentV2(catalog=default_catalog(), registry=registry, classifier=Unsure(), session=ShortTermSession(), config=AgentV2Config(record_sub_agents=False, record_capabilities=False, clarify_below=0))
    assert off.run("特斯拉最近", session_id="c4").status != RunStatus.WAITING_CLARIFICATION
    assert not agent._should_clarify(Intent(kind="knowledge", confidence=0.2, clarification="?", source="model")) and not agent._should_clarify(Intent(kind="lookup", confidence=0.2, clarification="?", source="recorded"))

    # A command missing its price asks, and the next message completes the command (offline through the recorded labels).
    applied: list = []
    agent = _mutation_agent(applied)
    first = agent.run("给AMD设个提醒", session_id="c5")
    assert first.status == RunStatus.WAITING_CLARIFICATION and "目标价" in first.answer
    second = agent.run("跌到150的时候", session_id="c5")
    assert second.status == RunStatus.WAITING_CONFIRMATION and second.pending_mutation.payload == {"ticker": "AMD", "direction": "below", "target_price": 150.0}
    assert agent.run("确认", session_id="c5").status == RunStatus.COMPLETED and applied and applied[0]["payload"]["target_price"] == 150.0
    no_session = _mutation_agent([]).run("给AMD设个提醒")
    assert no_session.status == RunStatus.WAITING_CLARIFICATION and "没有会话" in no_session.answer


def test_follow_up_about_the_previous_answer_is_written_from_its_evidence_without_new_calls():
    from v2.agent_v2.intent import Intent, IntentClassifier, parse_intent

    assert parse_intent({"kind": "lookup", "confidence": 0.9, "refers_back": True}).refers_back is True and not parse_intent({"kind": "lookup", "confidence": 0.9}).refers_back

    calls: list[str] = []

    def research(arguments, context):
        calls.append(arguments["ticker"])
        return ToolEnvelope("research.stock", ResultStatus.COMPLETED, subject="QCOM", evidence=[EvidenceItem("R1", "QCOM", "QCOM 苹果基带订单将在 2027 年前逐步流失", source_id="research_engine"), EvidenceItem("R2", "QCOM", "QCOM 手机业务占营收 60%", source_id="fd_metrics")], metadata={"narrative": "风险一：苹果基带订单流失 [R1]。风险二：手机依赖 [R2]。"})

    class Scripted:
        def __init__(self):
            self.payloads: list[dict] = []

        def classify(self, request):
            self.payloads.append(dict(request.metadata))
            if "展开" in request.text:
                return Intent(kind="research", tickers=("QCOM",), confidence=0.9, source="model", refers_back=True)
            return Intent(kind="research", wants=("risk",), tickers=("QCOM",), confidence=0.9, source="model")

    registry = CapabilityRegistry(default_catalog())
    registry.register("research.stock", research)
    llm = ScriptedLLM([LLMResponse(text="风险一：苹果基带订单流失 [R1]。风险二：手机依赖，手机业务占营收 60% [R2]。"), LLMResponse(text="第二点是手机依赖：QCOM 手机业务占营收 60% [R2]，所以基带订单流失 [R1] 会直接影响收入。")])
    classifier = Scripted()
    agent = AgentV2(catalog=default_catalog(), registry=registry, classifier=classifier, session=ShortTermSession(), synthesizer=LLMEvidenceSynthesizer(llm), config=AgentV2Config(record_sub_agents=False, record_capabilities=False, debate=False))
    first = agent.run("QCOM有什么风险？", session_id="m1")
    assert first.status in {RunStatus.COMPLETED, RunStatus.PARTIAL} and calls == ["QCOM"]
    previous = agent.session.previous_turn("m1")
    assert previous["question"] == "QCOM有什么风险？" and [item.id for item in previous["evidence"]] == ["R1", "R2"] and previous["run_id"] == first.run_id
    assert agent.session.recent_turns("m1") == [{"question": "QCOM有什么风险？", "answer_digest": first.answer[:240], "tickers": ["QCOM"]}]

    second = agent.run("上面第二点展开讲", session_id="m1")
    assert calls == ["QCOM"]  # no new capability call
    assert second.status == RunStatus.COMPLETED and second.answer.startswith("第二点是手机依赖") and [item.id for item in second.evidence] == ["R1", "R2"]
    assert second.synthesis["follow_up"] == {"previous_run_id": first.run_id, "evidence": 2} and second.plan.assumptions[0].startswith("follow_up:")
    assert classifier.payloads[-1]["recent_turns"][0]["question"] == "QCOM有什么风险？"  # the classifier saw the conversation
    sent = json.loads(llm.calls[-1][1]["content"])
    assert sent["previous_answer"] == first.answer and sent["previous_question"] == "QCOM有什么风险？" and sent["recent_turns"][0]["question"] == "QCOM有什么风险？" and "previous_answer" in llm.calls[-1][0]["content"]
    assert agent.session.previous_turn("m1")["run_id"] == second.run_id  # the follow-up becomes the previous turn

    # Without a previous turn the flag is ignored and the planner runs as usual; the live classifier sends recent_turns.
    fresh = AgentV2(catalog=default_catalog(), registry=registry, classifier=Scripted(), session=ShortTermSession(), config=AgentV2Config(record_sub_agents=False, record_capabilities=False))
    assert fresh.run("上面第二点展开讲", session_id="m2").results and calls == ["QCOM", "QCOM"]
    seen: list[dict] = []

    class Recording:
        def complete(self, messages, tools=None, tool_choice=None):
            seen.append(json.loads(messages[1]["content"]))
            return LLMResponse(text='{"kind":"lookup","scope":"none","direction":"none","wants":[],"tickers":[],"portfolio_scope":false,"confidence":0.9}')

    IntentClassifier(Recording()).classify(normalize_request("那它呢", metadata={"recent_turns": [{"question": "AAPL今天为什么涨？", "answer_digest": "涨了", "tickers": ["AAPL"]}]}))
    assert seen[0]["recent_turns"][0]["tickers"] == ["AAPL"]


def test_session_memory_is_bounded_and_internal_limitations_stay_internal():
    from v2.agent_v2 import session as session_module
    from v2.agent_v2.synthesis import user_facing_limitations

    registry = CapabilityRegistry(default_catalog())
    heavy = ToolEnvelope("research.stock", ResultStatus.COMPLETED, subject="ARM", evidence=[EvidenceItem(f"E{i}", "ARM", f"claim {i}", source_id="research_engine") for i in range(50)], metadata={"narrative": "x", "trace": [{"round": 1}] * 50, "agent": {"name": "move_attributor"}, "module_metrics": {}})
    registry.register("research.stock", lambda a, c: heavy)
    session = ShortTermSession()
    agent = AgentV2(catalog=default_catalog(), registry=registry, session=session, config=AgentV2Config(record_sub_agents=False, record_capabilities=False, debate=False))
    for index in range(session_module.MAX_REMEMBERED_SESSIONS + 4):
        last = agent.run("分析 ARM 估值", session_id=f"s{index}")
    assert len(session._previous) == session_module.MAX_REMEMBERED_SESSIONS and session.previous_turn("s0") is None and session.previous_turn(f"s{session_module.MAX_REMEMBERED_SESSIONS + 3}") is not None
    last.results.append(ToolEnvelope("debate.challenge", ResultStatus.COMPLETED, subject="ARM", metadata={"objections": [{"text": "x"}]}))
    session.record(last)  # a debate envelope is never carried into a follow-up
    kept = session.previous_turn(f"s{session_module.MAX_REMEMBERED_SESSIONS + 3}")["results"]
    assert all(item.capability != "debate.challenge" for item in kept) and "trace" not in kept[0].metadata and "agent" not in kept[0].metadata and kept[0].metadata.get("narrative") == "x" and len(kept[0].evidence) == 40

    assert user_facing_limitations(["Legacy formatted output; structured field-level evidence is not yet available.", "核心模块失败（valuation）"]) == ["核心模块失败（valuation）"]
    mutate = ToolEnvelope("state.mutate", ResultStatus.COMPLETED, subject="alert.add", summary="已设置提醒 #7。", evidence=[EvidenceItem("M1", "AMD", "已设置提醒 #7。")], limitations=["Legacy formatted output; structured field-level evidence is not yet available."])
    text = EvidenceSummarySynthesizer().synthesize(normalize_request("确认"), ExecutionPlan("确认", RouteKind.COMMAND, tasks=(PlanTask("m", "state.mutate", {"operation": "alert.add", "payload": {}}),)), [mutate], mutate.evidence)
    assert "Legacy formatted output" not in text and "已设置提醒 #7" in text


def test_earnings_dates_never_fan_the_research_engine_over_the_holdings_and_slow_responders_are_cached(monkeypatch):
    from v2.agent_v2.adapters import legacy
    from v2.agent_v2.intent import Intent

    # "谁要出财报" with each=True is the calendar, not twelve research runs.
    request = normalize_request("接下来两周我的持仓里谁要出财报？")
    each = Intent(kind="lookup", scope="recent", wants=("portfolio", "earnings"), portfolio_scope=True, each=True, source="model")
    plan = RulePlanner().plan(request, route(request, intent=each))
    assert [task.capability for task in plan.tasks] == ["account.earnings_schedule", "account.portfolio"]  # the card for the holdings' names, no research fan-out
    research = Intent(kind="research", wants=("earnings",), tickers=("NVDA",), source="model")
    plan = RulePlanner().plan(normalize_request("分析 NVDA 财报"), route(normalize_request("分析 NVDA 财报"), intent=research))
    assert any(task.capability == "research.stock" and task.arguments.get("focus") == "earnings" for task in plan.tasks)

    # The macro board and the 13F responders are served from a cache within their window.
    legacy.clear_cache()
    calls: list[str] = []

    def produce():
        calls.append("x")
        return "宏观面板"

    clock = {"now": 100.0}
    assert legacy.cached_value("macro.overview", "macro", produce, now=clock["now"]) == ("宏观面板", False)
    assert legacy.cached_value("macro.overview", "macro", produce, now=clock["now"] + 30) == ("宏观面板", True) and calls == ["x"]
    assert legacy.cached_value("macro.overview", "macro", produce, now=clock["now"] + 700) == ("宏观面板", False) and calls == ["x", "x"]
    assert legacy.cached_value("account.portfolio", "portfolio", produce, now=clock["now"]) == ("宏观面板", False) and calls == ["x", "x", "x"]  # never cached
    legacy.clear_cache()
    registry = CapabilityRegistry(default_catalog())
    monkeypatch.setattr(legacy, "_resolve", lambda path: (lambda *a, **k: "VIX 15.8，10 年期 4.83%"))
    legacy.register_legacy_capabilities(registry)
    context = ExecutionContext("run", request, BudgetClass.STANDARD)
    first = registry.execute(PlanTask("m", "macro.overview"), context)
    second = registry.execute(PlanTask("m", "macro.overview"), context)
    assert first.status == ResultStatus.COMPLETED and not first.cache_hit and second.cache_hit and second.status == ResultStatus.CACHED and second.evidence[0].claim == first.evidence[0].claim
    legacy.clear_cache()


def test_a_time_framed_holdings_ranking_fans_out_performance_without_the_performance_want():
    from v2.agent_v2.intent import Intent

    request = normalize_request("持仓里这周谁涨得最好？")
    week = Intent(kind="lookup", scope="recent", wants=("portfolio", "ranking"), portfolio_scope=True, rank="high", source="model")
    plan = RulePlanner().plan(request, route(request, intent=week))
    assert [task.capability for task in plan.tasks] == ["account.portfolio", "market.performance"] and plan.tasks[1].fan_out["max"] == 12 and "rank" not in plan.tasks[1].fan_out
    since = Intent(kind="lookup", scope="none", wants=("portfolio", "ranking"), portfolio_scope=True, rank="low", source="model")
    assert [task.capability for task in RulePlanner().plan(request, route(request, intent=since)).tasks] == ["account.portfolio"]


def test_repair_names_the_items_that_carry_an_uncited_sentence_s_figures():
    from v2.agent_v2.llm import _figures_in, repair_instruction
    from v2.agent_v2.models import VerificationReport

    price = EvidenceItem("evidence-attribute-price-1", "PLTR", "PLTR 在 2026-09-11 收于 167.23 美元，较前一交易日 +0.83%。", source_id="market_data")
    bench = EvidenceItem("evidence-attribute-benchmark-1", "PLTR", "PLTR 2026-09-11 行业基准 XLK 单日回报为 +1.32%，PLTR 相对回报为 -0.50%。", source_id="market_data")
    assert _figures_in("PLTR 收在 167.23 美元、较前一交易日 +0.83%，同日 XLK 涨 1.32%", [price, bench]) == [("167.23", ["evidence-attribute-price-1"]), ("+0.83%", ["evidence-attribute-price-1"]), ("1.32%", ["evidence-attribute-benchmark-1"])]
    report = VerificationReport(ok=False, warnings=("行情事实缺少邻近引用：“PLTR 收在 167.23 美元、较前一交易日 +0.83%…”",))
    text = repair_instruction(report, [price, bench])
    assert "这句没有引用" in text and "167.23 见 [evidence-attribute-price-1]" in text and "其他问题：行情事实缺少邻近引用" not in text
    unknown = repair_instruction(VerificationReport(ok=False, warnings=("行情事实缺少邻近引用：“成交量只有均量的 0.38 倍…”",)), [price])
    assert "其他问题：行情事实缺少邻近引用" in unknown  # nothing carries the figure: the plain warning stands


def test_durable_session_state_survives_a_restart_and_the_bot_tells_interrupted_chats(tmp_path, monkeypatch):
    import asyncio

    from v2.agent_v2.session_store import SqliteSessionState, plan_from_dict, plan_to_dict
    from v2.bot import agent_v2_bridge as bridge

    db = tmp_path / "session.sqlite"

    def agent_over(store):
        applied: list = []
        catalog = default_catalog()
        registry = CapabilityRegistry(catalog)
        registry.register("state.mutate", lambda a, c: (applied.append(a), ToolEnvelope("state.mutate", ResultStatus.COMPLETED, subject=a["operation"], summary="已设置提醒。", evidence=[EvidenceItem("M1", "AMD", "已设置提醒。")]))[1])
        registry.register("market.performance", lambda a, c: ToolEnvelope("market.performance", ResultStatus.COMPLETED, subject=a["ticker"], evidence=[EvidenceItem("P", a["ticker"], f"{a['ticker']} 近 30 天 +1.00%")], metadata={"narrative": f"{a['ticker']} 近 30 天 +1.00%[P]。"}))
        return AgentV2(catalog=catalog, registry=registry, session=ShortTermSession(durable=store), config=AgentV2Config(record_sub_agents=False, record_capabilities=False)), applied

    # Plans round-trip through JSON.
    plan = ExecutionPlan("NVDA 涨到 200 提醒我", RouteKind.COMMAND, tasks=(PlanTask("mutation", "state.mutate", {"operation": "alert.add", "payload": {"ticker": "NVDA", "direction": "above", "target_price": 200.0}}, purpose="提醒"),), requires_confirmation=True, assumptions=("x",))
    assert plan_from_dict(plan_to_dict(plan)) == plan

    # Before the "restart": a pending write, a clarification on another session, a turn about ARM.
    first, applied_before = agent_over(SqliteSessionState(db))
    assert first.run("NVDA 涨到 200 美元提醒我", session_id="chat-a").status == RunStatus.WAITING_CONFIRMATION
    assert first.run("给AMD设个提醒", session_id="chat-b").status == RunStatus.WAITING_CLARIFICATION
    first.run("ARM 最近30天表现", session_id="chat-c")
    assert not applied_before

    # After: a new session object over the same file — the process memory is gone.
    second, applied_after = agent_over(SqliteSessionState(db))
    confirmed = second.run("确认", session_id="chat-a")
    assert confirmed.status == RunStatus.COMPLETED and applied_after and applied_after[0]["payload"]["target_price"] == 200.0
    completed = second.run("跌到150的时候", session_id="chat-b")
    assert completed.status == RunStatus.WAITING_CONFIRMATION and completed.pending_mutation.payload["target_price"] == 150.0
    assert second.session.recent_turns("chat-c")[0]["question"].startswith("ARM") and second.session.recent_turns("chat-c")[0]["tickers"] == ["ARM"]
    followed = second.run("为什么跌这么多", session_id="chat-c")  # the subject-less follow-up still finds ARM after the restart
    assert followed.request.text.startswith("ARM") and followed.request.metadata.get("rewritten")
    assert second.session.pop_pending("chat-a") is None and second.session.pop_clarification("chat-b") is None  # consumed once

    # Expired rows are ignored; clear removes a session's rows.
    stale = SqliteSessionState(db, ttl_seconds=0.0)
    stale.set_pending("chat-d", plan)
    assert SqliteSessionState(db).pop_pending("chat-d") is None
    store = SqliteSessionState(db)
    store.set_clarification("chat-e", "q", "?")
    store.clear("chat-e")
    assert store.pop_clarification("chat-e") is None

    # The bot records in-flight runs durably and, at startup, tells the chats the restart cut off.
    store.mark_active("7", "ARM买入以来跌了这么多，是什么原因？")
    store.mark_active("8", "分析NVDA估值")
    store.clear_active("8")
    sent: list[tuple[int, str]] = []

    class Bot:
        async def send_message(self, *, chat_id, text):
            sent.append((chat_id, text))

    class Agent:
        session = ShortTermSession(durable=store)

    monkeypatch.setattr(bridge, "_get_agent", lambda: Agent())
    assert asyncio.run(bridge.notify_interrupted_runs(Bot())) == 1 and sent == [(7, bridge.INTERRUPTED_NOTICE.format(question="ARM买入以来跌了这么多，是什么原因？"))]
    assert asyncio.run(bridge.notify_interrupted_runs(Bot())) == 0  # told once


def test_lab_cases_route_to_the_lab_and_form_their_own_set(monkeypatch):
    from v2.agent_v2.eval import quality
    from v2.agent_v2.eval.quality_cases import QUALITY_CASES
    from v2.agent_v2.eval.quality_holdout import HOLDOUT_CASES
    from v2.agent_v2.eval.quality_lab import LAB_CASES

    ids = [case.id for case in (*QUALITY_CASES, *HOLDOUT_CASES, *LAB_CASES)]
    assert len(ids) == len(set(ids)) and all(case.set == "lab" and case.expected_route == RouteKind.LAB and "quick" not in case.tags for case in LAB_CASES)
    for case in LAB_CASES:
        request = normalize_request(case.question)
        decision = route(request)
        plan = RulePlanner().plan(request, decision)
        assert decision.kind == RouteKind.LAB and plan.tasks[0].capability.startswith("lab.") and plan.budget == BudgetClass.LAB, (case.id, decision.kind, [task.capability for task in plan.tasks])
    picked: dict = {}
    monkeypatch.setattr(quality, "run_cases", lambda agent_, cases, judge_, **kwargs: picked.setdefault("ids", tuple(case.id for case in cases)) and [])
    monkeypatch.setattr(quality, "_live_agent", lambda **kwargs: type("A", (), {"synthesizer": None})())
    quality.main(["run", "--label", "x", "--set", "lab"])
    assert picked["ids"] == tuple(case.id for case in LAB_CASES)


def test_the_investigator_can_list_an_annual_report_and_read_the_passages_around_a_keyword():
    from v2.agent_v2.agents.filing_reader import EdgarFilingSource, FilingRef, Section
    from v2.agent_v2.agents.toolbox import Toolbox, passages_around

    # The EDGAR source lists the forms it is asked for; without forms it stops at the current reports.
    asked = []

    def fetch(ticker, form, since, until):
        asked.append(form)
        return [SimpleNamespace(accession_number=f"{form}-1", filing_date="2026-01-30" if form == "10-K" else "2026-07-23", form=form, homepage_url=f"https://www.sec.gov/{form}/")]

    source = EdgarFilingSource(fetch=fetch)
    refs = source.list_filings("TSLA", "2025-01-01", "2026-09-09", ("10-K", "10-Q"))
    assert asked == ["10-K", "10-Q"] and [(ref.form, ref.filing_date) for ref in refs] == [("10-Q", "2026-07-23"), ("10-K", "2026-01-30")]
    asked.clear()
    assert [ref.form for ref in source.list_filings("TSLA", "2025-01-01", "2026-09-09")] == ["8-K"] and asked == ["8-K"]
    # Passages around keywords: merged, ordered, bounded; nothing when absent.
    text = "A" * 2000 + " Full Self-Driving (FSD) may not be delivered on time. " + "B" * 2000 + " Autopilot claims are pending. " + "C" * 2000
    found = passages_around(text, "FSD, Autopilot", limit=5000, radius=100)
    assert found.count("……") == 1 and "Full Self-Driving (FSD)" in found and "Autopilot claims" in found and len(found) < 700
    assert passages_around(text, "robotaxi", limit=5000) == "" and len(passages_around(text, "FSD", limit=120, radius=300)) == 120

    class Source:
        listed = []

        def list_filings(self, ticker, since, until, forms=None):
            Source.listed.append(forms)
            return [FilingRef(ticker=ticker, form="10-K", filing_date="2026-01-30", accession="0001", url="https://www.sec.gov/x/1/")]

        def outline(self, ref):
            return [Section("s3", "ITEM 1A. RISK FACTORS", len(text))]

        def read(self, ref, section_id):
            return text

    toolbox = Toolbox(filing_source=Source(), today=date(2026, 9, 9), max_chars=5000)
    tools = {tool.name: tool for tool in toolbox.tools()}
    assert "10-K | 2026-01-30" in tools["list_filings"].handler({"ticker": "TSLA", "forms": ["10-K"], "since": "2025-06-01"}) and Source.listed == [("10-K",)]
    # A plain read returns the head and says the section is longer; a keyword read returns the passages.
    head = tools["read_filing"].handler({"filing": 1, "section": "s3"})
    assert head.startswith("申报 f1 章节 s3 的文本（章节共 6086 字，只返回 5000 字）") and "FSD" not in head[:100]
    windows = tools["read_filing"].handler({"filing": 1, "section": "s3", "find": "FSD, Autopilot"})
    assert windows.startswith("申报 f1 章节 s3 里含“FSD, Autopilot”的段落") and "Autopilot claims are pending" in windows
    missing = tools["read_filing"].handler({"filing": 1, "section": "s3", "find": "robotaxi"})
    assert missing.startswith("申报 f1 章节 s3 里没有“robotaxi”；章节开头")
    # Both reads count against the cap and a quote from either read can be checked.
    assert toolbox.state.calls["read_filing"] == 3 and "Autopilot claims are pending" in toolbox.state.text_for("f1:s3")[0] and toolbox.state.text_for("f1:s3")[0].startswith("A" * 100)
    # A source without the forms parameter still lists.
    class Plain:
        def list_filings(self, ticker, since, until):
            return [FilingRef(ticker=ticker, form="8-K", filing_date="2026-07-29", accession="0002", url="")]

        def outline(self, ref):
            return []

    plain = Toolbox(filing_source=Plain(), today=date(2026, 9, 9))
    assert "8-K | 2026-07-29" in {tool.name: tool for tool in plain.tools()}["list_filings"].handler({"ticker": "ARM", "forms": ["10-K"]})


def test_lab_rows_read_as_labelled_lines_and_lab_answers_carry_their_guidance_and_the_detailed_style():
    from v2.agent_v2.adapters.workspace_lab import _evidence
    from v2.agent_v2.catalog import _LAB_GUIDANCE, _SWEEP_GUIDANCE

    rows = [{"top_n": 10, "holding_days": days, "near_high_pct": None, "per_trade": 10000.0, "n_trades": 30 + days, "n_periods": 9, "total_return_pct": 12.3456, "annualized_return_pct": 4.1, "sharpe_ratio": 0.62, "max_drawdown_pct": -8.25, "win_rate": 0.55, "avg_return_pct": 1.1, "benchmark_pct": 9.0, "excess_return_pct": 3.3456, "start": "2021-09-10", "end": "2026-09-09"} for days in (10, 21, 42)] * 3
    payload = {"kind": "sweep", "strategy": "momentum", "tickers": ["NVDA"], "rows": rows[:9]}
    items = _evidence("lab.sweep", payload, "run-1")
    findings = [item for item in items if "-finding-" in item.id]
    assert len(findings) == 9 and findings[0].claim == "top_n 10；持有期 10；总收益% 12.35；年化% 4.1；最大回撤% -8.25；夏普 0.62；胜率 0.55；交易笔数 40；基准% 9.0；超额% 3.35；起 2021-09-10；止 2026-09-09"
    assert findings[0].value["holding_days"] == 10 and [item.claim for item in items if "-metric-" in item.id] == []
    specs = default_catalog()
    assert specs.get("lab.backtest").answer_guidance == _LAB_GUIDANCE and specs.get("lab.sweep").answer_guidance.endswith(_SWEEP_GUIDANCE) and "事件窗口" in specs.get("lab.event_study").answer_guidance
    # The sweep's grid and an investigation's dated findings are lists by nature: the detailed style, whatever the wording.
    synthesizer = LLMEvidenceSynthesizer(ScriptedLLM([]))
    plan = ExecutionPlan(objective="x", route=RouteKind.LAB)
    sweep = ToolEnvelope("lab.sweep", ResultStatus.COMPLETED, evidence=items)
    assert json.loads(synthesizer._payload("对 NVDA 做参数扫描", plan, [sweep], items))["response_style"] == "detailed"
    assert json.loads(synthesizer._payload("对 NVDA 做参数扫描", plan, [ToolEnvelope("lab.backtest", ResultStatus.COMPLETED, evidence=items)], items))["response_style"] == "brief"


def test_an_investigation_finding_must_be_quoted_verbatim_where_it_is_cited_and_named_tickers_are_the_lab_universe():
    from v2.agent_v2.agents.toolbox import quote_rule
    from v2.agent_v2.eval.quality import render_case

    rule = quote_rule("evidence-investigate-abc", "revenue guidance came in below Wall Street expectations, sending shares down 13%.")
    item = EvidenceItem("evidence-investigate-abc", "ARM", "ARM 2026-07-29：指引低于预期（Arm falls：“revenue guidance came in below Wall Street expectations”）。", metadata={"evidence_type": "investigation"})
    envelope = ToolEnvelope("agent.investigate", ResultStatus.COMPLETED, evidence=[item], metadata={"answer_constraints": [rule]})
    paraphrased = verify_answer("2026-07-29：Arm 的指引低于华尔街预期（Arm falls）[evidence-investigate-abc]。", [item], answer_mode=AnswerMode.RESEARCH_GROUNDED, results=[envelope])
    assert not paraphrased.ok and any("没有照抄它的引文" in warning for warning in paraphrased.warnings)
    # The quote once, anywhere: a summary sentence that cites the finding again without it is fine.
    quoted = verify_answer("2026-07-29：Arm 的指引低于华尔街预期（Arm falls：“revenue guidance came  in below Wall Street expectations”）[evidence-investigate-abc]。\n\n最值得跟踪的是指引本身[evidence-investigate-abc]。", [item], answer_mode=AnswerMode.RESEARCH_GROUNDED, results=[envelope])
    assert quoted.ok, quoted.warnings
    # Not cited at all: the rule does not apply.
    assert not any("引文" in warning for warning in verify_answer("没有相关发现。", [item], answer_mode=AnswerMode.RESEARCH_GROUNDED, results=[envelope]).warnings)
    assert quote_rule("e1", "")["require"] == "" and quote_rule("e1", "")["quote_of"] == "e1"

    # The lab port: named tickers make a custom universe unless one was asked for.
    # (A stand-in for the pydantic input model: the port reads model_fields and model_dump; CI has no pydantic.)
    class Input:
        model_fields = {"universe": None, "tickers": None, "holding_days_list": None}

        def __init__(self, universe="sp500", tickers=(), holding_days_list=(21,)):
            self.universe, self.tickers, self.holding_days_list = universe, list(tickers), list(holding_days_list)

        def model_dump(self):
            return {"universe": self.universe, "tickers": self.tickers, "holding_days_list": self.holding_days_list}

    seen = []
    port = WorkspaceLabPort({"lab.sweep": LabBinding(Input, lambda body: seen.append(body.model_dump()) or {"kind": "sweep", "tickers": body.tickers, "rows": []})})
    context = ExecutionContext("lab-run", NormalizedRequest("扫描", "扫描"), BudgetClass.LAB)
    port.run("lab.sweep", {"tickers": ["NVDA"], "holding_days_list": [10, 21, 42]}, context)
    port.run("lab.sweep", {"tickers": ["NVDA"], "universe": "tech30"}, context)
    port.run("lab.sweep", {}, context)
    assert [row["universe"] for row in seen] == ["custom", "tech30", "sp500"]

    # `show --attempt` picks one attempt of a repeated run.
    rows = [{"case_id": "q_x", "label": "p6b", "attempt": 1, "question": "q", "answer": "first", "score": {"passed": False}}, {"case_id": "q_x", "label": "p6b", "attempt": 2, "question": "q", "answer": "second", "score": {"passed": True}}]
    assert render_case(rows, "q_x").endswith("second") and render_case(rows, "q_x", attempt=1).endswith("first") and "未通过" in render_case(rows, "q_x", attempt=1)
    assert render_case(rows, "q_x", attempt=3) == "没有 q_x 的记录（第 3 次）。"


def test_an_empty_synthesizer_reply_is_asked_for_once_more_before_the_fallback():
    synthesizer = LLMEvidenceSynthesizer(ScriptedLLM([LLMResponse(text=""), LLMResponse(text="NVDA 市盈率 28.2 倍[R1]。")]))
    assert synthesizer._draft([{"role": "user", "content": "x"}], [], []) == "NVDA 市盈率 28.2 倍[R1]。" and len(synthesizer.llm.calls) == 2
    twice = LLMEvidenceSynthesizer(ScriptedLLM([LLMResponse(text=""), LLMResponse(text="   ")]))
    with pytest.raises(ValueError):
        twice._draft([{"role": "user", "content": "x"}], [], [])
    assert len(twice.llm.calls) == 2
