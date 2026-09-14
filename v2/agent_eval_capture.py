"""Capture original V2 test inputs and fixture responses without rewriting tests.

Python call boundaries, not language heuristics, define the inventory. Run the
original pytest suite so parametrization, loops and preceding turns survive.
"""
import argparse
import ast
from dataclasses import asdict, is_dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
import sys
import threading
import weakref


def plain(value):
    if hasattr(value, 'to_dict'):
        return plain(value.to_dict())
    if is_dataclass(value):
        return plain(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return {'unserialized_type': type(value).__name__}


class Capture:
    def __init__(self):
        from v2.agent_v2.orchestrator import AgentV2
        from v2.agent_v2.routing import normalize_request
        from v2.agent_v2.models import NormalizedRequest
        from v2.agent_v2.execution import CapabilityRegistry
        from v2.agent_v2.session import ShortTermSession
        from v2.agent_v2.planning import RulePlanner
        self.codes = {AgentV2.run.__code__: 'agent', AgentV2._execute.__code__: 'execution_plan', normalize_request.__code__: 'normalize',
                      NormalizedRequest.__init__.__code__: 'request', CapabilityRegistry.execute.__code__: 'tool',
                      ShortTermSession.resolve.__code__: 'session_resolve', RulePlanner.plan.__code__: 'plan'}
        self.rows, self.outcomes, self.pending = [], [], {}
        self.node = ''
        self.object_sequence = 0
        self.object_ids = {}
        self.old_profile = None
        self.old_thread_profile = None

    def object_key(self, obj):
        previous = self.object_ids.get(id(obj))
        if previous is None or previous[0]() is not obj:
            self.object_sequence += 1
            self.object_ids[id(obj)] = (weakref.ref(obj), self.object_sequence)
        return self.object_ids[id(obj)][1]

    def profile(self, frame, event, result):
        kind = self.codes.get(frame.f_code)
        if not kind or event not in {'call', 'return'}:
            return
        if event == 'call':
            local = frame.f_locals
            stack, parent = [], frame.f_back
            inside_agent = False
            while parent:
                inside_agent |= self.codes.get(parent.f_code) == 'agent'
                if parent.f_code.co_filename.endswith('test_agent_v2.py'):
                    stack.append({'line': parent.f_lineno, 'function': parent.f_code.co_name})
                parent = parent.f_back
            row = {'entry': kind, 'test': self.node, 'source_stack': stack, 'inside_agent': inside_agent}
            if kind in {'agent', 'normalize', 'request', 'session_resolve'}:
                row.update({k: plain(local[k]) for k in ('text', 'original_text', 'session_id', 'allow_web', 'metadata') if k in local})
                if kind in {'agent', 'session_resolve'}:
                    row['object'] = self.object_key(local['self'])
                if kind == 'agent':
                    row['available_capabilities'] = sorted(local['self'].registry._handlers)
                    session = local['self'].session
                    durable = getattr(session, 'durable', None)
                    row['continuity_key'] = ('durable:' + str(durable.path)) if getattr(durable, 'path', None) else ('session:' + str(self.object_key(session))) if session else ('agent:' + str(row['object']))
            elif kind == 'plan':
                row.update(request=plain(local['request']), route=plain(local['route']))
            elif kind == 'execution_plan':
                row.update(plan=plain(local['plan']), run_id=local['run_id'])
            else:
                row.update(task=plain(local['task']), run_id=local['context'].run_id,
                           allow_web=local['context'].allow_web, allow_mutations=local['context'].allow_mutations)
            row['sequence'] = len(self.rows)
            self.rows.append(row)
            self.pending[id(frame)] = row
        else:
            row = self.pending.pop(id(frame), None)
            if row is not None:
                row['result'] = plain(result) if kind != 'execution_plan' else None
                if kind == 'agent' and result is not None:
                    row['intent'] = plain(getattr(result.route, 'intent', None))
                    row['config'] = plain(frame.f_locals['self'].config)

    def pytest_runtest_setup(self, item):
        self.node = item.nodeid

    def pytest_runtest_logreport(self, report):
        if report.when == 'call' or report.outcome != 'passed':
            self.outcomes.append({'test': report.nodeid, 'phase': report.when, 'outcome': report.outcome})

    def start(self):
        self.old_profile = sys.getprofile()
        self.old_thread_profile = threading.getprofile()
        sys.setprofile(self.profile)
        threading.setprofile(self.profile)

    def stop(self):
        sys.setprofile(self.old_profile)
        threading.setprofile(self.old_thread_profile)


def build_inventory(source, capture, exit_code):
    text = source.read_text(encoding='utf-8')
    tests = [node for node in ast.parse(text).body if isinstance(node, ast.FunctionDef) and node.name.startswith('test_')]
    tools = {}
    initial_plans = {}
    for row in capture.rows:
        if row['entry'] == 'tool':
            tools.setdefault(row['run_id'], []).append(row)
        elif row['entry'] == 'execution_plan':
            initial_plans[row['run_id']] = row['plan']
    episodes = {}
    standalone = []
    for row in capture.rows:
        if row['entry'] == 'agent':
            # A blank session means no conversation, even on the same instance.
            key = (row['test'], row.get('continuity_key', row['object']), row.get('session_id') or f"single-{row['sequence']}")
            result = row.get('result') or {}
            episodes.setdefault(key, []).append({**row, 'tool_records': tools.get(result.get('run_id'), []),
                                                'initial_plan': initial_plans.get(result.get('run_id'))})
        elif row['entry'] in {'normalize', 'request', 'session_resolve'} and not row['inside_agent']:
            standalone.append(row)
    coverage = []
    for test in tests:
        observed = [row for row in capture.rows if any(site['function'] == test.name for site in row['source_stack'])]
        # Keep every original test function. Zero observation is never a pass.
        coverage.append({'test': test.name, 'line': test.lineno, 'observed_boundaries': len(observed),
                         'classification': 'observed_input_or_tool_contract' if observed else 'internal_contract_or_unobserved',
                         'original_assertions': [ast.get_source_segment(text, node) for node in ast.walk(test) if isinstance(node, ast.Assert)]})
    return {'source': str(source), 'source_sha256': hashlib.sha256(text.encode()).hexdigest(),
            'pytest_exit_code': exit_code, 'outcomes': capture.outcomes, 'coverage': coverage,
            'episodes': [{'id': f'v2-original-{i:04d}', 'test': key[0], 'turns': turns}
                         for i, (key, turns) in enumerate(episodes.items(), 1)],
            'standalone_boundaries': standalone,
            'planning_cases': [row for row in capture.rows if row['entry']=='plan' and not row['inside_agent']],
            'scope': 'Actual original test execution; unresolved/internal boundaries retained, not silently converted into end-to-end questions.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error('Use a new output path; preserve prior capture')
    root = Path(__file__).resolve().parents[1]
    source = root / 'v2/agent_v2/test_agent_v2.py'
    import pytest
    capture = Capture()
    capture.start()
    try:
        code = pytest.main([str(source), '-q'], plugins=[capture])
    finally:
        capture.stop()
    inventory = build_inventory(source, capture, int(code))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'episodes': len(inventory['episodes']), 'standalone_boundaries': len(inventory['standalone_boundaries']),
                      'original_tests': len(inventory['coverage']), 'pytest_exit_code': int(code)}))
    raise SystemExit(code)


if __name__ == '__main__':
    main()
