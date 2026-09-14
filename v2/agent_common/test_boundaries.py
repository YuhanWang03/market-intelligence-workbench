"""Prove current agents work even when the legacy package is unavailable."""
import subprocess
import sys
from pathlib import Path


def test_current_agents_do_not_load_legacy_package():
    code = '''
import importlib.abc
import sys
class BlockLegacy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "v2.agent" or fullname.startswith("v2.agent."):
            raise AssertionError("Legacy dependency: " + fullname)
sys.meta_path.insert(0, BlockLegacy())
from v2.agent_v2.runtime import build_workspace_agent
from v2.agent_v2.eval.benchmark import _v2_agent
from v2.agent_v3.runtime import build_workspace_agent as build_v3
from v2.agent_v3.demo import build_demo_agent, DEMO_QUESTION
from v2.agent_common.telegram_delivery import split_for_telegram
from v2.bot import agent_v2_bridge, agent_v3_bridge
agent, *_ = _v2_agent("v2_rules", None, "v1")
result = agent.run("NVDA现在多少钱？")
assert result.answer
result = build_demo_agent().run(DEMO_QUESTION)
assert result.answer and "120" in result.answer
assert len(split_for_telegram("x" * 8000)) == 3
assert not any(n == "v2.agent" or n.startswith("v2.agent.") for n in sys.modules)
'''
    result = subprocess.run([sys.executable, '-c', code], cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr


def test_legacy_directory_is_retired():
    assert not (Path(__file__).resolve().parents[1] / 'agent').exists()


def test_retired_benchmark_mode_is_rejected():
    import pytest
    from v2.agent_v2.eval.benchmark import run_mode
    with pytest.raises(ValueError, match='unknown mode'):
        run_mode('v1_baseline', ())
