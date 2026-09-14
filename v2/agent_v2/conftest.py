"""Every Agent V2 ledger points at a temporary directory during tests.

The orchestrator appends sub-agent runs, capability outcomes and routing
decisions to ``data/*.jsonl`` by default; a test suite run on the server
was writing its fixtures ("provider down", "capability adapter is not
registered") into the production ledgers the reports read.
"""

from __future__ import annotations

import pytest

_LEDGER_VARS = (
    "AGENT_V2_SUBAGENT_LEDGER",
    "AGENT_V2_CAPABILITY_LEDGER",
    "AGENT_V2_INTENT_LEDGER",
    "AGENT_V2_QUALITY_LEDGER",
    "AGENT_V2_USER_MEMORY",
    "AGENT_V2_SESSION_DB",
)


@pytest.fixture(autouse=True)
def _isolated_ledgers(tmp_path, monkeypatch):
    for name in _LEDGER_VARS:
        monkeypatch.setenv(name, str(tmp_path / (name.lower() + (".sqlite" if name.endswith("_DB") else ".jsonl"))))
    yield
