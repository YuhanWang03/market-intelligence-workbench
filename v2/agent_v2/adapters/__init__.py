"""Adapters from existing project services to Agent V2 capability contracts."""

from v2.agent_v2.adapters.history import register_history_capabilities
from v2.agent_v2.adapters.lab import register_lab_capabilities
from v2.agent_v2.adapters.legacy import register_legacy_capabilities
from v2.agent_v2.adapters.market import register_market_capabilities
from v2.agent_v2.adapters.research import register_research_capabilities
from v2.agent_v2.adapters.tavily_web import TavilyWebSearchPort
from v2.agent_v2.adapters.web import register_web_capability
from v2.agent_v2.adapters.workspace_lab import LabBinding, WorkspaceLabPort

__all__ = [
    "register_history_capabilities",
    "register_lab_capabilities",
    "register_legacy_capabilities",
    "register_market_capabilities",
    "register_research_capabilities",
    "register_web_capability",
    "LabBinding",
    "TavilyWebSearchPort",
    "WorkspaceLabPort",
]
