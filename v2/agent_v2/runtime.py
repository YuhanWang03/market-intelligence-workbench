"""Construction helpers for live and explicitly empty Agent V2 runtimes."""

from __future__ import annotations

import logging

from dataclasses import replace

from v2.agent_v2.adapters import (
    register_history_capabilities,
    register_lab_capabilities,
    register_legacy_capabilities,
    register_market_capabilities,
    register_research_capabilities,
    register_web_capability,
    TavilyWebSearchPort,
    WorkspaceLabPort,
)
from v2.agent_v2.agents.filing_reader import register_filing_reader
from v2.agent_v2.agents.news_checker import register_news_checker
from v2.agent_v2.agents.move_attributor import register_move_attributor
from v2.agent_v2.catalog import CapabilityCatalog, default_catalog
from v2.agent_v2.execution import CapabilityRegistry
from v2.agent_v2.llm import LLMEvidenceSynthesizer, StructuredLLMPlanner
from v2.agent_v2.orchestrator import AgentV2, AgentV2Config
from v2.agent_v2.session import ShortTermSession


def _search_via(web_search):
    """A search callable for the news checker from a WebSearchPort's provider, raw page text included when it can."""

    provider = getattr(web_search, "provider", None)
    client = getattr(provider, "_client", None)

    def search(query: str, *, days: int, max_results: int) -> list:
        if client is not None:
            recent = days <= 30
            response = client.search(query=query, max_results=max_results, topic="news" if recent else "general", days=days, search_depth="basic", include_raw_content=True)
            return list(response.get("results", []))
        if provider is not None:
            return list(provider.search(query, days=days, max_results=max_results) or [])
        envelope = web_search.search(query, topic="general", recency_days=days)
        return [dict(finding) for finding in envelope.findings]

    return search


def build_live_registry(
    catalog: CapabilityCatalog | None = None,
    *,
    lab=None,
    web_search=None,
    llm=None,
) -> CapabilityRegistry:
    """Register live capabilities; Lab and Web remain explicit injections.

    ``llm`` powers the sub-agent capabilities; without it they still
    register and answer with a "no model configured" limitation.
    """
    registry = CapabilityRegistry(catalog or default_catalog())
    register_research_capabilities(registry)
    register_legacy_capabilities(registry)
    register_market_capabilities(registry)
    register_history_capabilities(registry)
    register_filing_reader(registry, llm)
    register_move_attributor(registry, llm)
    if llm is not None:
        from v2.agent_v2.agents.move_attributor import _default_recall
        from v2.agent_v2.agents.toolbox import register_investigator

        register_investigator(registry, llm, search=_search_via(web_search) if web_search is not None else None, recall=_default_recall)
    if lab is not None:
        register_lab_capabilities(registry, lab)
    if web_search is not None:
        if llm is not None:
            # With a model, the web step is a bounded sub-agent: it searches,
            # reads the pages that matter and reports dated events with
            # located quotes.  Without one, the one-shot snippet adapter.
            register_news_checker(registry, llm, search=_search_via(web_search))
        else:
            register_web_capability(registry, web_search)
    return registry


def build_empty_registry(catalog: CapabilityCatalog | None = None) -> CapabilityRegistry:
    """A deterministic test/development registry with no external dependencies."""
    return CapabilityRegistry(catalog or default_catalog())


def build_live_agent(*, config: AgentV2Config | None = None, lab=None, web_search=None) -> AgentV2:
    """Build the in-process starter runtime; Lab, Web, and channel wiring stay opt-in."""
    catalog = default_catalog()
    return AgentV2(
        catalog=catalog,
        registry=build_live_registry(catalog, lab=lab, web_search=web_search),
        session=ShortTermSession(),
        config=config,
    )


def build_llm_agent(*, config: AgentV2Config | None = None, llm=None, lab=None, web_search=None) -> AgentV2:
    """Build V2 with the existing OpenAI-compatible client for planning/synthesis."""
    if llm is None:
        from v2.agent_common.llm import build_llm

        llm = build_llm()
    catalog = default_catalog()
    registry = build_live_registry(catalog, lab=lab, web_search=web_search, llm=llm)
    from v2.agent_v2.intent import IntentClassifier
    from v2.agent_v2.memory import UserMemory

    return AgentV2(
        catalog=catalog,
        registry=registry,
        planner=StructuredLLMPlanner(llm, catalog),
        synthesizer=LLMEvidenceSynthesizer(llm, catalog=catalog),
        session=ShortTermSession(durable=durable_session_state()),
        config=config,
        classifier=IntentClassifier(llm),
        memory=UserMemory(),
    )


def durable_session_state():
    """The sqlite session state the live runtime uses; None when it cannot be opened (the session then lives in memory)."""

    from v2.agent_v2.session_store import SqliteSessionState

    try:
        return SqliteSessionState()
    except Exception as exc:  # noqa: BLE001 — a read-only disk must not stop the bot
        logging.getLogger(__name__).warning("durable session state unavailable: %s", exc)
        return None


def build_workspace_agent(
    *,
    config: AgentV2Config | None = None,
    llm=None,
    use_llm: bool = True,
    enable_web: bool = False,
    news_provider=None,
) -> AgentV2:
    """Compose existing Research + Web Lab tools, with optional Tavily fallback.

    This constructor is intended for the Web backend, where ``app.routers`` is
    importable.  Passing ``enable_web=True`` is the explicit network opt-in;
    individual calls must still pass ``allow_web=True``.
    """
    from v2.agent_v2.warmup import warm_imports

    warm_imports()  # the executor's threads must not be the first to import numpy and yfinance

    effective_config = config or AgentV2Config()
    if enable_web and not effective_config.enable_web_fallback:
        effective_config = replace(effective_config, enable_web_fallback=True)
    if use_llm and effective_config.record_intents is None:
        # Live runs ledger every routing decision for the intent report; a config that says False keeps it off.
        effective_config = replace(effective_config, record_intents=True)
    lab = WorkspaceLabPort()
    web_search = TavilyWebSearchPort(news_provider) if enable_web else None
    if use_llm:
        return build_llm_agent(
            config=effective_config,
            llm=llm,
            lab=lab,
            web_search=web_search,
        )
    return build_live_agent(config=effective_config, lab=lab, web_search=web_search)
