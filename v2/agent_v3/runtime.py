"""Live composition with isolated V3 state and explicit model configuration."""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver

from v2.agent_v3.adapters import register_account, register_user_state
from v2.agent_v3.brain import ModelBrain
from v2.agent_v3.graph import AgentV3, AgentV3Config
from v2.agent_v3.persistence import SessionStore
from v2.agent_v3.specialists import make_reviewer, register_specialists
from v2.agent_v3.tools import Registry


def build_model():
    from langchain_openai import ChatOpenAI
    # A key is never paired with an inferred provider URL. Configure all three explicitly.
    model = os.environ.get("AGENT_V3_MODEL") or os.environ.get("AGENT_LLM_MODEL")
    base_url = os.environ.get("AGENT_V3_BASE_URL") or os.environ.get("AGENT_LLM_BASE_URL")
    key = os.environ.get("AGENT_V3_API_KEY") or os.environ.get("AGENT_LLM_API_KEY")
    if not (model and base_url and key):
        raise ValueError("Set AGENT_V3_MODEL, AGENT_V3_BASE_URL and AGENT_V3_API_KEY (or their AGENT_LLM equivalents)")
    thinking = os.environ.get("AGENT_V3_THINKING")
    if thinking not in {None, "disabled", "enabled"}:
        raise ValueError("AGENT_V3_THINKING must be enabled or disabled")
    extra_body = {"thinking": {"type": thinking}} if thinking else None
    return ChatOpenAI(model=model, base_url=base_url.rstrip("/"), api_key=key.strip(), timeout=45, max_retries=1, temperature=0, extra_body=extra_body)


def build_workspace_agent(*, model=None, config=None, data_dir=None, enable_mutations=False, lab=None, search=None, filing_source=None, recall=None):
    from v2.agent_v3.research import register_research_capabilities
    from v2.agent_v3.market import register_market_capabilities
    from v2.agent_v3.sec import register_history_capabilities, filing_source as default_filing_source

    model = model if model is not None else build_model()
    config = config or AgentV3Config()
    from v2.agent_v3.monitor_history import MonitorArchive, register_monitor_history
    archive = MonitorArchive(os.environ.get("AGENT_V3_ARCHIVE_DB") or os.environ.get("WEB_ARCHIVE_DB"))
    recall = recall if recall is not None else archive.recall
    registry = Registry()
    register_research_capabilities(registry)
    register_market_capabilities(registry)
    register_history_capabilities(registry)
    register_monitor_history(registry, archive)
    register_account(registry)
    register_user_state(registry, enable_mutations=enable_mutations)
    from v2.agent_v3.data_extensions import register_data_extensions
    register_data_extensions(registry)
    from v2.agent_v3.institutional import register_institutional
    register_institutional(registry)
    from v2.agent_v3.ark import register_ark
    register_ark(registry)
    from v2.agent_v3.earnings import register_earnings
    register_earnings(registry)
    primary_search = None
    if config.enable_web and search is None:
        key = os.environ.get("TAVILY_API_KEY")
        if key:
            from tavily import TavilyClient
            client = TavilyClient(api_key=key)
            def primary_search(query, *, days, max_results):
                return client.search(query=query,topic="general",search_depth="advanced",max_results=max_results,include_raw_content=True,exclude_domains=["x.com","twitter.com","facebook.com","reddit.com"]).get("results",[])
            def search(query, *, days, max_results):
                from datetime import datetime
                from email.utils import parsedate_to_datetime
                rows = client.search(query=query, topic="news", search_depth="advanced", days=days, max_results=max_results, include_raw_content=True, exclude_domains=["x.com","twitter.com","facebook.com","reddit.com"]).get("results", [])
                for row in rows:
                    value = row.get("published_date")
                    if value:
                        try:
                            try:
                                stamp = datetime.fromisoformat(value.replace("Z","+00:00"))
                            except ValueError:
                                stamp = parsedate_to_datetime(value)
                            row["published_date"] = stamp.date().isoformat()
                        except (ValueError, TypeError):
                            row["published_date"] = ""
                return rows
    register_specialists(registry, model, search=search, filing_source=filing_source or default_filing_source(), recall=recall)
    from v2.agent_v3.news_research import register_news_research
    register_news_research(registry,model,search,primary_search)
    from v2.agent_v3.portfolio_analysis import register_portfolio_workflows
    from v2.agent_v3.news_research import investigated_news
    def holding_news(ticker,request,run):
        if search is None:
            from v2.agent_v2.models import ToolEnvelope, ResultStatus
            return ToolEnvelope("web.research",ResultStatus.PARTIAL_DATA,limitations=["搜索服务未配置。"])
        return investigated_news(model,search,request.text,ticker,request.metadata["date_window"],run,attribution=True,primary_search=primary_search,event_dates=request.metadata.get("holding_event_dates",[]))
    registry.holding_news=holding_news
    register_portfolio_workflows(registry)
    if lab is None:
        from v2.agent_v3.lab import V3LabPort
        lab = V3LabPort()
    if lab is not None:
        from v2.agent_v2.adapters.lab import register_lab_capabilities
        register_lab_capabilities(registry, lab)
    root = Path(data_dir or os.environ.get("AGENT_V3_DATA_DIR") or Path(__file__).resolve().parents[2] / "data" / "agent_v3")
    root.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(root / "checkpoints.sqlite", check_same_thread=False)
    try:
        agent = AgentV3(registry=registry, brain=ModelBrain(model), config=config, store=SessionStore(root / "sessions.sqlite"), checkpointer=SqliteSaver(conn), reviewer=make_reviewer(model), event_source=archive)
    except BaseException:
        conn.close()
        raise
    agent.checkpoint_connection = conn
    return agent
