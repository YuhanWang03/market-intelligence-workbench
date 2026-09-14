"""Answer-discipline cases: recorded market envelopes plus answers the verifier must judge.

Each case is one behaviour that was patched by hand in the market path.  The
envelopes are produced by the real adapter against fixed fixtures, so the
constraints under test are the ones production evidence actually carries.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from types import SimpleNamespace
from typing import Callable
from zoneinfo import ZoneInfo

from v2.agent_v2.adapters.market import register_market_capabilities
from v2.agent_v2.catalog import default_catalog
from v2.agent_v2.execution import CapabilityRegistry, ExecutionContext
from v2.agent_v2.models import AnswerMode, BudgetClass, NormalizedRequest, PlanTask, ToolEnvelope

_ET = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class AnswerCase:
    id: str
    description: str
    envelope: Callable[[], ToolEnvelope]
    answer: Callable[[ToolEnvelope], str]
    expect_ok: bool
    answer_mode: AnswerMode = AnswerMode.TOOL_GROUNDED
    expected_warning: str = ""
    #: ``judge(items) -> {id: quote}`` standing in for the model that decides forbid_claim rules; None leaves those rules unapplied.
    judge: Callable[[list[dict[str, str]]], dict[str, str]] | None = None


def asserting_judge(*claim_words: str) -> Callable[[list[dict[str, str]]], dict[str, str]]:
    """A scripted judge: every item whose claim mentions one of ``claim_words`` counts as asserted (the model's verdict, replayed)."""

    def judge(items: list[dict[str, str]]) -> dict[str, str]:
        return {item["id"]: item["text"][:30] for item in items if any(word in item["claim"] for word in claim_words)}

    return judge


class _Prices:
    def get_prices(self, ticker, start, end):
        slope = 1.0 if ticker == "AMD" else 0.2
        first = date(2026, 7, 1)
        return [SimpleNamespace(time=(first + timedelta(days=index)).isoformat(), close=100.0 + slope * index, volume=1_000_000 + index * 10_000) for index in range(35)]


def _context() -> ExecutionContext:
    return ExecutionContext("eval-answer", NormalizedRequest("q", "q"), BudgetClass.FOCUSED)


def performance_envelope(*, intraday: bool) -> ToolEnvelope:
    registry = CapabilityRegistry(default_catalog())
    now = datetime(2026, 8, 4, 13, 0, tzinfo=_ET) if intraday else datetime(2026, 8, 4, 18, 0, tzinfo=_ET)
    register_market_capabilities(registry, price_source_factory=_Prices, move_provider=lambda ticker: None, now_factory=lambda: now)
    return registry.execute(PlanTask("performance", "market.performance", {"ticker": "AMD"}), _context())


def move_envelope(*, confirmed: bool, weak_note: str = "缺少直接证据", intraday: bool = True) -> ToolEnvelope:
    reasons = [SimpleNamespace(text="期权市场波动", confidence="低", note=weak_note), SimpleNamespace(text="行业轮动", confidence="中", note="")]
    if confirmed:
        reasons.insert(0, SimpleNamespace(text="公司发布直接利好", confidence="高", note="权威媒体同日报道"))
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
        reasons=reasons,
        sources=[{"title": "Same-day report", "url": "https://example.test/report"}],
        next_steps=[],
        filtered_count=2,
    )
    registry = CapabilityRegistry(default_catalog())
    now = datetime(2026, 9, 8, 13, 0, tzinfo=_ET) if intraday else datetime(2026, 9, 8, 18, 0, tzinfo=_ET)
    register_market_capabilities(registry, price_source_factory=lambda: None, move_provider=lambda ticker: anomaly, now_factory=lambda: now)
    return registry.execute(PlanTask("move", "market.explain_move", {"ticker": "AMD"}), _context())


def _scoped(envelope: ToolEnvelope, scope: str):
    return next(item for item in envelope.evidence if item.metadata.get("evidence_scope") == scope)


def _role(envelope: ToolEnvelope, role: str):
    return [item for item in envelope.evidence if item.metadata.get("claim_role") == role]


ANSWER_CASES: tuple[AnswerCase, ...] = (
    AnswerCase(
        "p_narrative_ok",
        "adapter narrative for an intraday performance envelope passes verification",
        lambda: performance_envelope(intraday=True),
        lambda env: str(env.metadata["narrative"]),
        True,
    ),
    AnswerCase(
        "p_closed_narrative_ok",
        "adapter narrative for a closed-session performance envelope passes verification",
        lambda: performance_envelope(intraday=False),
        lambda env: str(env.metadata["narrative"]),
        True,
    ),
    AnswerCase(
        "p_intraday_price_as_close",
        "an intraday price must not be written as a closing price",
        lambda: performance_envelope(intraday=True),
        lambda env: f"AMD 收盘价为 {env.metrics['close']:.2f} 美元。[{_scoped(env, 'price').id}]",
        False,
        expected_warning="盘中价格",
        judge=asserting_judge("盘中价格"),
    ),
    AnswerCase(
        "p_intraday_volume_conclusion",
        "cumulative intraday volume cannot support a volume verdict",
        lambda: performance_envelope(intraday=True),
        lambda env: f"AMD 属于缩量上涨。[{_scoped(env, 'volume').id}]",
        False,
        expected_warning="未收盘成交量",
        judge=asserting_judge("成交量"),
    ),
    AnswerCase(
        "p_invented_number",
        "a figure absent from the evidence is rejected even with a valid citation",
        lambda: performance_envelope(intraday=False),
        lambda env: f"AMD 近 21 日年化波动率为 99.00%。[{_scoped(env, 'volatility').id}]",
        False,
    ),
    AnswerCase(
        "p_uncited_market_fact",
        "a market fact with a figure needs a nearby citation",
        lambda: performance_envelope(intraday=False),
        lambda env: f"AMD 收盘价为 {env.metrics['close']:.2f} 美元。",
        False,
        expected_warning="邻近引用",
    ),
    AnswerCase(
        "m_narrative_ok",
        "adapter narrative for a move envelope without a confirmed driver passes verification",
        lambda: move_envelope(confirmed=False),
        lambda env: str(env.metadata["narrative"]),
        True,
        AnswerMode.RESEARCH_GROUNDED,
    ),
    AnswerCase(
        "m_confirmed_narrative_ok",
        "adapter narrative for a move envelope with a confirmed driver passes verification",
        lambda: move_envelope(confirmed=True),
        lambda env: str(env.metadata["narrative"]),
        True,
        AnswerMode.RESEARCH_GROUNDED,
    ),
    AnswerCase(
        "m_candidate_as_cause",
        "a candidate driver may not be written as the confirmed cause",
        lambda: move_envelope(confirmed=False),
        lambda env: f"AMD 上涨的主要原因是行业轮动。[{_role(env, 'candidate_driver')[1].id}]",
        False,
        AnswerMode.RESEARCH_GROUNDED,
        expected_warning="候选归因",
        judge=asserting_judge("候选解释"),
    ),
    AnswerCase(
        "m_too_many_candidates",
        "without a confirmed driver at most one candidate may be shown",
        lambda: move_envelope(confirmed=False, weak_note=""),
        lambda env: "可能与期权市场波动相关。[{}] 也可能与行业轮动相关。[{}]".format(*(item.id for item in _role(env, "candidate_driver"))),
        False,
        AnswerMode.RESEARCH_GROUNDED,
        expected_warning="过多弱候选",
    ),
    AnswerCase(
        "m_rejected_candidate_shown",
        "a candidate the attribution step rejected is not citable",
        lambda: move_envelope(confirmed=False),
        lambda env: f"可能与期权市场波动相关。[{_role(env, 'candidate_driver')[0].id}]",
        False,
        AnswerMode.RESEARCH_GROUNDED,
        expected_warning="过弱异动线索",
    ),
    AnswerCase(
        "m_count_leak",
        "internal attribution counts reaching the user is a soft warning: reported, not a rejection",
        lambda: move_envelope(confirmed=False),
        lambda env: f"归因结果：0 个直接驱动。[{_role(env, 'attribution_assessment')[0].id}]",
        True,
        AnswerMode.RESEARCH_GROUNDED,
        expected_warning="（提示）将内部归因计数",
        judge=asserting_judge("内部统计口径"),
    ),
)
