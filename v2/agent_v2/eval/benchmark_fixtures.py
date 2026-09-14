"""Serve Agent V1's recorded observations through Agent V2 capability envelopes.

Account, state, macro, ETF and 13F capabilities are thin wrappers over the
same responders V1 recorded, so their cards port verbatim as legacy-style
evidence.  ``research.stock`` has no recorded Research Engine output yet; for
this stage it is composed from the per-ticker V1 cards (summary, earnings,
insiders, 8-K, money flow, holders, chain) so the V1 answer keys stay
checkable.  Stage two replaces that composition with recorded engine runs.
"""

from __future__ import annotations

import hashlib
from typing import Any, Callable

from v2.agent_eval.fixtures import EVAL_FIXTURES
from v2.agent_eval.base_fixtures import SimulatedToolFailure
from v2.agent_v2.catalog import default_catalog
from v2.agent_v2.entities import extract_entities
from v2.agent_v2.execution import CapabilityRegistry, ExecutionContext
from v2.agent_v2.models import EvidenceItem, ResultStatus, ToolEnvelope

#: research focus -> V1 cards that carry the same information.
FOCUS_CARDS: dict[str, tuple[str, ...]] = {
    "overview": ("summary", "earnings_view", "insider_view", "eight_k_view", "moneyflow_view"),
    "fundamentals": ("summary", "earnings_view"),
    "valuation": ("summary", "moneyflow_view"),
    "earnings": ("earnings_view", "eight_k_view", "summary"),
    "market": ("moneyflow_view", "summary"),
    "ownership": ("holders", "insider_view"),
    "catalysts": ("eight_k_view", "earnings_view", "summary"),
    "filings": ("eight_k_view",),
    "supply_chain": ("chain",),
    "risk": ("summary", "insider_view", "eight_k_view", "moneyflow_view"),
    "full": ("summary", "earnings_view", "insider_view", "eight_k_view", "moneyflow_view", "holders", "chain"),
}


class RecordedCalls:
    """What the registry was asked to run, for tool-recall scoring."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def names(self) -> list[str]:
        return [name for name, _ in self.calls]

    def reset(self) -> None:
        self.calls.clear()


def _card(tool: str, key: str) -> str:
    """Look a V1 card up the way V1's FixtureExecutor does."""

    entry = EVAL_FIXTURES.get(tool)
    if entry is None:
        return f"(no fixture for {tool})"
    if callable(entry):
        return str(entry({"ticker": key}))
    if isinstance(entry, dict):
        return str(entry.get(key, entry.get("_", f"(no fixture for {tool}/{key})")))
    return str(entry)


def _evidence(capability: str, subject: str, text: str, module: str = "") -> EvidenceItem:
    digest = hashlib.sha256(f"{capability}:{subject}:{module}:{text}".encode("utf-8")).hexdigest()[:16]
    return EvidenceItem(
        id=f"fixture-{digest}",
        entity=subject,
        claim=text,
        source_id=module or capability,
        source_title="Recorded V1 observation",
        metadata={"legacy_formatted_output": True, "module": module},
    )


def _wrap(capability: str, subject: str, text: str) -> ToolEnvelope:
    metadata = {"tickers": list(extract_entities(text))} if capability in {"account.portfolio", "state.read"} else {}
    return ToolEnvelope(capability, ResultStatus.COMPLETED, subject=subject, summary=text, evidence=[_evidence(capability, subject, text)], metadata=metadata)


def _research_envelope(ticker: str, focus: str) -> ToolEnvelope:
    cards = FOCUS_CARDS.get(focus, FOCUS_CARDS["overview"])
    evidence: list[EvidenceItem] = []
    limitations: list[str] = []
    texts: list[str] = []
    for tool in cards:
        try:
            text = _card(tool, ticker)
        except SimulatedToolFailure as exc:
            limitations.append(f"{tool}: {exc}")
            continue
        texts.append(text)
        evidence.append(_evidence("research.stock", ticker, text, module=tool))
    status = ResultStatus.PARTIAL_DATA if limitations else ResultStatus.COMPLETED
    if limitations:
        # Production's research adapter makes limitations citeable; a result
        # with no source evidence would otherwise be impossible to write about.
        joined = "；".join(limitations)
        evidence.append(
            EvidenceItem(
                id=f"fixture-limitations-{hashlib.sha256(f'{ticker}:{joined}'.encode('utf-8')).hexdigest()[:16]}",
                entity=ticker,
                claim=f"{ticker} 研究数据限制：{joined}",
                source_id="research_engine",
                source_title="Research data limitations",
                metadata={"citation_kind": "limitations", "limitations": list(limitations), "verified": True},
            )
        )
    return ToolEnvelope(
        "research.stock",
        status,
        subject=ticker,
        summary="\n".join(texts),
        evidence=evidence,
        limitations=limitations,
        metadata={"requested_modules": list(cards), "focus": focus},
    )


FIXTURE_MODES = ("v1", "engine")


def build_benchmark_registry(calls: RecordedCalls | None = None, *, fixtures: str = "v1") -> tuple[CapabilityRegistry, RecordedCalls]:
    """A V2 registry that answers from recorded observations.

    ``fixtures="v1"`` serves V1's cards for everything, so V1's answer keys
    stay checkable.  ``fixtures="engine"`` serves research and market
    capabilities from engine-shaped envelopes instead: a live recording under
    ``eval/recorded/`` when one exists, else the offline synthesis in
    ``engine_fixtures``.  Account, state, macro, ETF and 13F stay on V1 cards
    in both modes because they wrap the same responders.
    """

    if fixtures not in FIXTURE_MODES:
        raise ValueError(f"unknown fixture mode: {fixtures}")
    recorded = calls or RecordedCalls()
    registry = CapabilityRegistry(default_catalog())

    def register(name: str, handler: Callable[[dict[str, Any], ExecutionContext], ToolEnvelope]) -> None:
        def wrapped(arguments: dict[str, Any], context: ExecutionContext) -> ToolEnvelope:
            recorded.calls.append((name, dict(arguments)))
            return handler(arguments, context)

        registry.register(name, wrapped)

    register("account.portfolio", lambda a, c: _wrap("account.portfolio", "portfolio", _card("portfolio_view", "_")))
    register("account.risk", lambda a, c: _wrap("account.risk", "portfolio", _card("risk_view", "_")))
    register("account.earnings_schedule", lambda a, c: _wrap("account.earnings_schedule", "portfolio", _card("earnings_calendar", "_")))

    def performance(arguments: dict[str, Any], context: ExecutionContext) -> ToolEnvelope:
        period = str(arguments.get("period") or "day")
        text = _card("pnl_view", "_") if period == "day" else _card("pnl_period", period)
        return _wrap("account.performance", "portfolio", text)

    register("account.performance", performance)
    register("institutional.manager_portfolio", lambda a, c: _wrap("institutional.manager_portfolio", str(a.get("manager") or ""), _card("institutional_13f", str(a.get("manager") or "").lower())))
    register("etf.ark_activity", lambda a, c: _wrap("etf.ark_activity", str(a.get("symbol") or ""), _card("etf_view", str(a.get("symbol") or "").upper())))
    register("macro.overview", lambda a, c: _wrap("macro.overview", "macro", _card("macro_view", "_")))
    register("macro.release", lambda a, c: _wrap("macro.release", str(a.get("release_type") or ""), _card("release_check", str(a.get("release_type") or "").lower())))
    if fixtures == "engine":
        from v2.agent_v2.eval.engine_fixtures import synthesize_market_envelope, synthesize_research_envelope
        from v2.agent_v2.eval.recorded import RecordedStore

        store = RecordedStore()

        def research_envelope(ticker: str, focus: str) -> ToolEnvelope:
            return store.load("research.stock", f"{ticker}:{focus}") or synthesize_research_envelope(ticker, focus)

        def market_handler(capability: str):
            def handler(arguments: dict[str, Any], context: ExecutionContext) -> ToolEnvelope:
                ticker = str(arguments.get("ticker") or "").upper()
                return store.load(capability, ticker) or synthesize_market_envelope(capability, ticker)

            return handler

        register("market.explain_move", market_handler("market.explain_move"))
        register("market.performance", market_handler("market.performance"))
    else:
        research_envelope = _research_envelope
        register("market.explain_move", lambda a, c: _wrap("market.explain_move", str(a.get("ticker") or "").upper(), _card("explain_move", str(a.get("ticker") or "").upper())))
        register("market.performance", lambda a, c: _wrap("market.performance", str(a.get("ticker") or "").upper(), _card("explain_move", str(a.get("ticker") or "").upper())))

    def state_read(arguments: dict[str, Any], context: ExecutionContext) -> ToolEnvelope:
        section = str(arguments.get("section") or "watchlist")
        tool = {"watchlist": "watchlist_view", "alerts": "alert_list", "settings": "settings_view"}[section]
        return _wrap("state.read", section, _card(tool, "_"))

    register("state.read", state_read)
    register("state.mutate", lambda a, c: _wrap("state.mutate", str(a.get("operation") or ""), f"已执行 {a.get('operation')}（fixture）。"))
    register("research.stock", lambda a, c: research_envelope(str(a.get("ticker") or "").upper(), str(a.get("focus") or "overview")))

    def compare(arguments: dict[str, Any], context: ExecutionContext) -> ToolEnvelope:
        tickers = [str(value).upper() for value in arguments.get("tickers", [])][:4]
        dimensions = arguments.get("dimensions") or ["overview"]
        parts = [research_envelope(ticker, str(dimensions[0])) for ticker in tickers]
        return ToolEnvelope(
            "research.compare",
            ResultStatus.COMPLETED if all(part.ok for part in parts) else ResultStatus.PARTIAL_ERROR,
            subject=",".join(tickers),
            summary="\n".join(f"{part.subject}: {part.summary}" for part in parts),
            evidence=[item for part in parts for item in part.evidence],
            limitations=[value for part in parts for value in part.limitations],
        )

    register("research.compare", compare)
    register("research.changes", lambda a, c: ToolEnvelope("research.changes", ResultStatus.PARTIAL_DATA, subject=str(a.get("ticker") or ""), limitations=["at least two research snapshots are required"]))
    for name in ("lab.screen", "lab.backtest", "lab.sweep", "lab.event_study", "lab.committee"):
        register(name, lambda a, c, name=name: ToolEnvelope(name, ResultStatus.COMPLETED, summary=f"{name} completed (fixture)", evidence=[_evidence(name, ",".join(a.get("tickers", [])), f"{name} completed (fixture)")]))
    return registry, recorded
