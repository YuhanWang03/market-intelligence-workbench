"""V3 evaluation entry and offline benchmark-contract tests.

pytest does not use live credentials. Execute as a module with --live to run
the external, version-neutral benchmark; V3 itself never invokes V2.
"""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from v2.agent_v3.evaluation_cases import CASES
from v2.agent_comparison import observe, report, run_turns


@pytest.mark.parametrize('case', CASES, ids=lambda case: case.id)
def test_cases_trace_to_v2_contract_questions(case):
    source = Path(__file__).resolve().parents[2] / 'agent_v2/test_agent_v2.py'
    functions = {node.name for node in ast.parse(source.read_text(encoding='utf-8')).body if isinstance(node, ast.FunctionDef)}
    assert case.source_test in functions
    assert case.question and case.criteria


def test_case_ids_are_unique():
    assert len({case.id for case in CASES}) == len(CASES)


def test_multiturn_runs_preceding_questions_with_same_session():
    calls = []
    class Agent:
        def run(self, text, **kwargs):
            calls.append((text, kwargs))
            return SimpleNamespace(to_dict=lambda: {'answer': text})
    case = next(case for case in CASES if case.id == 'followup')
    results = run_turns(Agent(), case)
    assert len(results) == 2 and calls[-1][0] == case.question
    assert calls[0][1] == calls[1][1]
    assert calls[-1][1]['allow_web'] is True


def test_verification_success_does_not_become_quality_success():
    result = observe({'status': 'completed', 'answer': 'unsupported assertion', 'verification': {'ok': True}})
    assert result['quality_verdict'] == 'pending_review'


def test_report_keeps_timeouts_in_denominator(tmp_path):
    report(tmp_path, [{'case': 'knowledge', 'version': 'v3', 'repeat': 1,
                      'execution': 'process_timeout', 'artifact': 'missing'}])
    text = (tmp_path / 'report.md').read_text(encoding='utf-8')
    assert '失败/超时 1' in text and 'process_timeout' in text


def test_fast_failed_answer_is_not_counted_as_a_fast_success(tmp_path):
    report(tmp_path, [{'case':'knowledge','version':'v3','repeat':1,'execution':'returned',
                      'artifact':'missing','observations':observe({'status':'failed','elapsed_ms':10})}])
    text=(tmp_path/'report.md').read_text(encoding='utf-8')
    assert '失败/超时 1' in text and '耗时中位数' not in text


if __name__ == '__main__':
    from v2.agent_original_comparison import main
    main()
