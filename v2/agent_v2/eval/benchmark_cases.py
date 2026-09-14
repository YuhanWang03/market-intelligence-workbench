"""The Agent V1 evaluation set, ported onto Agent V2's capability vocabulary.

V1 labelled 89 development questions and 15 held-out questions against its 24
responder tools.  The questions, the answer keys (facts quotable from the
recorded fixtures, behaviours, forbidden misattributions) and the categories
carry over unchanged; only the tool names are translated, because V2 exposes
capabilities rather than cards.  Several V1 cards collapse into one V2
capability (``research.stock``); a V1 tool with no V2 equivalent maps to
``""`` and is recorded on the case as a gap rather than hidden.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from v2.agent_eval.cases import CASES as V1_CASES
from v2.agent_eval.cases import EvalCase as V1Case
from v2.agent_eval.holdout import HOLDOUT as V1_HOLDOUT

#: V1 responder tool -> V2 capability.  ``""`` marks a capability V2 lacks.
TOOL_MAP: dict[str, str] = {
    "portfolio_view": "account.portfolio",
    "pnl_view": "account.performance",
    "pnl_period": "account.performance",
    "risk_view": "account.risk",
    "earnings_calendar": "account.earnings_schedule",
    "institutional_13f": "institutional.manager_portfolio",
    "etf_view": "etf.ark_activity",
    "release_check": "macro.release",
    "explain_move": "market.explain_move",
    "watchlist_view": "state.read",
    "alert_list": "state.read",
    "settings_view": "state.read",
    "watchlist_add": "state.mutate",
    "watchlist_remove": "state.mutate",
    "alert_set": "state.mutate",
    "alert_remove": "state.mutate",
    "summary": "research.stock",
    "earnings_view": "research.stock",
    "eight_k_view": "research.stock",
    "insider_view": "research.stock",
    "holders": "research.stock",
    "moneyflow_view": "research.stock",
    "chain": "research.stock",
    "macro_view": "macro.overview",
}


@dataclass(frozen=True)
class BenchmarkCase:
    id: str
    query: str
    category: str
    #: V2 capabilities the answer needs; V1 ``must_call`` translated and de-duplicated.
    must_call: tuple[str, ...]
    #: V1 tools that have no V2 capability; a case with any of these cannot
    #: reach 100% tool recall and is reported as a capability gap.
    unmapped_tools: tuple[str, ...]
    wasteful: tuple[str, ...]
    facts: tuple[tuple[str, ...], ...]
    behaviors: tuple[tuple[str, ...], ...]
    forbidden: tuple[str, ...]
    max_tool_calls: int
    v1: V1Case = field(repr=False, compare=False)
    holdout: bool = False

    @property
    def v1_must_call(self) -> tuple[str, ...]:
        return self.v1.must_call

    @property
    def extra(self) -> dict[str, Any]:
        return dict(self.v1.extra)


def _translate(tools: tuple[str, ...]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    mapped: list[str] = []
    unmapped: list[str] = []
    for tool in tools:
        target = TOOL_MAP.get(tool)
        if target is None or target == "":
            unmapped.append(tool)
        elif target not in mapped:
            mapped.append(target)
    return tuple(mapped), tuple(unmapped)


def port(case: V1Case, *, holdout: bool = False) -> BenchmarkCase:
    must_call, unmapped = _translate(case.must_call)
    wasteful, _ = _translate(case.wasteful_tools)
    # A tool that is required under another name is not waste.
    wasteful = tuple(name for name in wasteful if name not in must_call)
    return BenchmarkCase(
        id=case.id,
        query=case.query,
        category=case.category,
        must_call=must_call,
        unmapped_tools=unmapped,
        wasteful=wasteful,
        facts=tuple(tuple(forms) for forms in case.facts),
        behaviors=tuple(tuple(forms) for forms in case.behaviors),
        forbidden=tuple(case.forbidden),
        max_tool_calls=case.max_tool_calls,
        v1=case,
        holdout=holdout,
    )


DEV_CASES: tuple[BenchmarkCase, ...] = tuple(port(case) for case in V1_CASES)
HOLDOUT_CASES: tuple[BenchmarkCase, ...] = tuple(port(case, holdout=True) for case in V1_HOLDOUT)


def by_category(cases: tuple[BenchmarkCase, ...]) -> dict[str, list[BenchmarkCase]]:
    grouped: dict[str, list[BenchmarkCase]] = {}
    for case in cases:
        grouped.setdefault(case.category, []).append(case)
    return grouped
