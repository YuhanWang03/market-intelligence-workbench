"""Small seed set for routing and capability acquisition, independent of tool names from V1."""

from __future__ import annotations

from dataclasses import dataclass

from v2.agent_v2.models import AnswerMode, RouteKind, RunStatus


@dataclass(frozen=True)
class EvalCase:
    id: str
    query: str
    expected_route: RouteKind
    required_capabilities: tuple[str, ...] = ()
    forbidden_capabilities: tuple[str, ...] = ()
    expected_statuses: tuple[RunStatus, ...] = (RunStatus.COMPLETED,)
    expected_answer_mode: AnswerMode | None = None
    category: str = "routing"


CASES = (
    EvalCase("k01", "什么是自由现金流？", RouteKind.GENERAL_KNOWLEDGE, expected_answer_mode=AnswerMode.GENERAL_KNOWLEDGE, category="knowledge"),
    EvalCase("f01", "我的持仓有哪些？", RouteKind.FAST_LOOKUP, ("account.portfolio",), category="account"),
    EvalCase("f02", "我这个月赚了多少？", RouteKind.FAST_LOOKUP, ("account.performance",), category="account"),
    EvalCase("f03", "查看我的关注列表", RouteKind.FAST_LOOKUP, ("state.read",), category="account"),
    EvalCase("r01", "分析 NVDA 的估值", RouteKind.RESEARCH, ("research.stock",), expected_answer_mode=AnswerMode.RESEARCH_GROUNDED, category="research"),
    EvalCase("r02", "比较 NVDA 和 AMD 的风险", RouteKind.RESEARCH, ("research.compare",), forbidden_capabilities=("research.stock",), category="comparison"),
    EvalCase("r03", "NVDA 的研究结论相比上次有什么变化？", RouteKind.RESEARCH, ("research.changes",), category="research_history"),
    EvalCase("r04", "我的持仓里哪只风险最高？", RouteKind.RESEARCH, ("account.portfolio", "account.risk"), category="portfolio_research"),
    EvalCase("l01", "回测 NVDA 动量策略", RouteKind.LAB, ("lab.backtest",), category="lab"),
    EvalCase("l02", "对 AAPL 和 MSFT 做财报事件研究", RouteKind.LAB, ("lab.event_study",), category="lab"),
    EvalCase("c01", "把 NVDA 加入关注列表", RouteKind.COMMAND, expected_statuses=(RunStatus.WAITING_CONFIRMATION,), category="command"),
    EvalCase("a01", "对标普全部股票做十年参数扫描", RouteKind.ASYNC, ("lab.sweep",), category="async"),
    EvalCase("m01", "AMD最近表现如何？", RouteKind.FAST_LOOKUP, ("market.performance",), forbidden_capabilities=("research.stock",), category="market"),
    EvalCase("m02", "AMD今天为什么涨？", RouteKind.RESEARCH, ("market.explain_move",), forbidden_capabilities=("research.stock",), category="market"),
    EvalCase("m03", "AMD经营表现如何？", RouteKind.FAST_LOOKUP, ("research.stock",), forbidden_capabilities=("market.performance",), category="market"),
    EvalCase("m04", "AMD最近的收益质量如何？", RouteKind.FAST_LOOKUP, ("research.stock",), forbidden_capabilities=("market.performance",), category="market"),
    EvalCase("e01", "分析 nvda 的估值", RouteKind.RESEARCH, ("research.stock",), category="entities"),
    EvalCase("e02", "比较阿里巴巴和拼多多", RouteKind.RESEARCH, ("research.compare",), category="entities"),
    EvalCase("c02", "AVGO 的 total addressable market 有多大", RouteKind.FAST_LOOKUP, ("research.stock",), category="command"),
)
