"""Build both agents under one set of conditions, and rewire their tools.

``live`` / ``record`` / ``frozen`` build the real workspace agents with the
same model, temperature 0, the same wall-clock budget and web allowance;
``record`` wraps every tool with a recorder, ``frozen`` swaps every tool for
the bank replay.  ``offline`` builds the model-free agents (V2 rules planner,
V3 demo brain) for the bench's own tests.

Writes are left registered on purpose: a command must stop at the
confirmation step, and the runner never confirms, so nothing is written.
Lab capabilities are removed from both — an experiment is not a question.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from v2.agent_bench.bank import Recorder, Replay

MODES = ("live", "record", "frozen", "offline")


def configure_model() -> dict[str, str]:
    """One model for both agents and (unless overridden) the judge; the settings are returned for the ledger."""
    from dotenv import load_dotenv
    from v2.agent_common.llm import OpenAICompatLLM

    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)
    defaults = OpenAICompatLLM()
    model = os.environ.get("AGENT_V3_MODEL") or defaults.model
    base = os.environ.get("AGENT_V3_BASE_URL") or defaults.base_url
    key = os.environ.get("AGENT_V3_API_KEY") or defaults.api_key
    if not key:
        raise ValueError("Model API credentials are not configured (AGENT_V3_API_KEY / AGENT_LLM_API_KEY / DEEPSEEK_API_KEY)")
    if urlparse(base).hostname == "api.deepseek.com":
        os.environ.setdefault("AGENT_V3_THINKING", "disabled")
        os.environ.setdefault("AGENT_LLM_THINKING", "disabled")
    for prefix in ("AGENT_V3_", "AGENT_LLM_"):
        for name, value in (("MODEL", model), ("BASE_URL", base), ("API_KEY", key)):
            os.environ[prefix + name] = value
    return {"model": model, "base_url": base, "thinking": os.environ.get("AGENT_V3_THINKING") or "provider_default", "temperature": 0}


def handlers_of(agent: Any, version: str) -> dict[str, Any]:
    return agent.registry._handlers if version == "v2" else agent.registry.handlers


def capability_names(agent: Any) -> list[str]:
    return list(agent.registry.catalog.names())


def _drop_lab(agent: Any, version: str) -> None:
    handlers = handlers_of(agent, version)
    for name in list(handlers):
        if name.startswith("lab."):
            del handlers[name]


def build_agent(version: str, *, mode: str, seconds: float, workdir: Path, debate: bool = False) -> Any:
    if version not in {"v2", "v3"} or mode not in MODES:
        raise ValueError(f"unknown version/mode: {version}/{mode}")
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    if mode == "offline":
        return _offline_agent(version, seconds=seconds, workdir=workdir)
    settings = configure_model()
    os.environ["AGENT_V2_SESSION_DB"] = str(workdir / "v2-sessions.sqlite")
    os.environ["AGENT_V2_USER_MEMORY"] = str(workdir / "v2-memory.jsonl")
    if version == "v2":
        from v2.agent_common.llm import OpenAICompatLLM
        from v2.agent_v2.orchestrator import AgentV2Config
        from v2.agent_v2.runtime import build_workspace_agent

        agent = build_workspace_agent(llm=OpenAICompatLLM(model=settings["model"], base_url=settings["base_url"]), enable_web=True,
                                      config=AgentV2Config(max_seconds=seconds, allow_mutations=False, debate=debate, record_sub_agents=False, record_capabilities=False, record_intents=False))
    else:
        from v2.agent_v3.graph import AgentV3Config
        from v2.agent_v3.runtime import build_workspace_agent

        agent = build_workspace_agent(config=AgentV3Config(max_seconds=seconds, enable_web=True, debate=debate), data_dir=workdir / "v3", enable_mutations=True, recall=lambda *args: [])
    _drop_lab(agent, version)
    return agent


def _offline_agent(version: str, *, seconds: float, workdir: Path) -> Any:
    if version == "v2":
        from v2.agent_v2.catalog import default_catalog
        from v2.agent_v2.execution import CapabilityRegistry
        from v2.agent_v2.orchestrator import AgentV2, AgentV2Config
        from v2.agent_v2.session import ShortTermSession

        catalog = default_catalog()
        return AgentV2(catalog=catalog, registry=CapabilityRegistry(catalog), session=ShortTermSession(),
                       config=AgentV2Config(max_seconds=seconds, allow_mutations=False, debate=False, record_sub_agents=False, record_capabilities=False, record_intents=False))
    from v2.agent_v3.demo import build_demo_agent

    return build_demo_agent()


def attach_replay(agent: Any, version: str, replay: Replay) -> None:
    """Every declared capability answers from the bank; nothing reaches a provider."""
    handlers = handlers_of(agent, version)
    for name in capability_names(agent):
        if name.startswith("lab."):
            continue
        handlers[name] = replay.handler(name)


def attach_recorder(agent: Any, version: str, recorder: Recorder) -> None:
    handlers = handlers_of(agent, version)
    for name, handler in list(handlers.items()):
        handlers[name] = recorder.wrap(name, handler)


def set_preference(agent: Any, version: str, session_id: str, note: str) -> bool:
    """A "记住，…" turn is a stored preference, not a question; each agent keeps it differently."""
    if version == "v2":
        memory = getattr(agent, "memory", None)
        if memory is None:
            return False
        memory.add_preference(session_id, note)
        return True
    store = getattr(agent, "store", None)
    if store is None:
        return False
    history = store.get(session_id)
    history["preferences"] = [*history.get("preferences", []), note]
    store.put(session_id, history)
    return True


def close_agent(agent: Any) -> None:
    for closer in (getattr(getattr(agent, "store", None), "close", None), getattr(getattr(agent, "checkpoint_connection", None), "close", None)):
        if closer:
            try:
                closer()
            except Exception:  # noqa: BLE001
                pass
