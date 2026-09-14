"""Original V2 questions, isolated agents, identical captured tool responses.

Contract mode injects original plans/intents and measures execution contracts,
not language understanding. --live-model evaluates each native planner/answer
against the same frozen tool bank; never enables live business tools.
"""
import argparse
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
import subprocess
import threading
import time

from v2.agent_comparison import observe


def key(capability, arguments):
    return json.dumps([capability, arguments], sort_keys=True, ensure_ascii=False)


class FrozenTools:
    def __init__(self, episode):
        self.episode = episode
        self.turn = 0
        self.calls = []
        self.used = {}
        self.lock = threading.Lock()

    def handler(self, capability):
        def call(arguments, context):
            from v2.agent_v3.contracts import envelope_from
            from v2.agent_v2.models import ToolEnvelope, ResultStatus
            signature = key(capability, arguments)
            with self.lock:
                rows = [row for row in self.episode['turns'][self.turn].get('tool_records', [])
                        if key(row['task']['capability'], row['task']['arguments']) == signature]
                slot = (self.turn, signature)
                index = self.used.get(slot, 0)
                self.used[slot] = index + 1
                found = index < len(rows)
                self.calls.append({'turn': self.turn, 'capability': capability, 'arguments': deepcopy(arguments),
                                   'fixture_match': found, 'mutation_authorized': context.allow_mutations})
            if not found:
                return ToolEnvelope(capability, ResultStatus.PARTIAL_DATA,
                    limitations=['冻结测试数据中未录制此轮的该工具参数；未访问真实数据源。'],
                    metadata={'fixture_missing': True})
            result = envelope_from(deepcopy(rows[index]['result']))
            return replace(result, elapsed_ms=0)
        return call


def original_plan(turn):
    from v2.agent_v3.contracts import plan_from
    return plan_from(turn.get('initial_plan') or turn['result']['plan'])


def intent_dict(turn):
    value = dict(turn.get('intent') or {})
    if not value:
        # A cancelled/confirmed turn may not classify; explicit command
        # protocol is handled by each version adapter, not language guessing.
        value = {'kind': 'lookup', 'tickers': turn.get('result', {}).get('request', {}).get('entities', [])}
    return value


def frozen_answer(evidence):
    return '\n'.join(f'{item.claim} [{item.id}]' for item in evidence) or '冻结工具未提供可引用证据。'


class ContractBrain:
    def __init__(self, tools):
        self.tools = tools

    def classify(self, text, history, run):
        from v2.agent_v3.contracts import SemanticIntent
        data = intent_dict(self.tools.episode['turns'][self.tools.turn])
        data = {k: v for k, v in data.items() if k in SemanticIntent.model_fields and v is not None}
        return SemanticIntent.model_validate(data)

    def plan(self, *args):
        return original_plan(self.tools.episode['turns'][self.tools.turn])

    def draft(self, state, registry, run, **kwargs):
        from v2.agent_v2.models import EvidenceItem
        return frozen_answer([EvidenceItem(**item) for item in state.get('evidence', [])])

    def judge(self, *args):
        return {}


def build_pair_member(version, episode, live_model=False, seconds=60):
    from v2.agent_v2.catalog import default_catalog
    from v2.agent_v2.execution import CapabilityRegistry
    from v2.agent_v3.tools import Registry
    bank = FrozenTools(episode)
    registry = CapabilityRegistry(default_catalog()) if version == 'v2' else Registry()
    # Same capability contracts and availability on both sides; all handlers
    # return deep-copied fixture envelopes, including state/lab operations.
    names = set()
    for turn in episode['turns']:
        names.update(turn.get('available_capabilities', []))
        names.update(row['task']['capability'] for row in turn.get('tool_records', []))
        names.update(task['capability'] for task in turn.get('result', {}).get('plan', {}).get('tasks', []))
    for name in names:
        if registry.catalog.get(name):
            registry.register(name, bank.handler(name))
    if live_model:
        from dotenv import load_dotenv
        from v2.agent_common.llm import OpenAICompatLLM
        from urllib.parse import urlparse
        load_dotenv(Path(__file__).resolve().parents[1] / '.env', override=False)
        default = OpenAICompatLLM()
        model = os.environ.get('AGENT_V3_MODEL') or default.model
        base = os.environ.get('AGENT_V3_BASE_URL') or default.base_url
        credential = os.environ.get('AGENT_V3_API_KEY') or default.api_key
        if not credential:
            raise ValueError('Model credential missing')
        os.environ.update(AGENT_V3_MODEL=model, AGENT_V3_BASE_URL=base, AGENT_V3_API_KEY=credential)
        if urlparse(base).hostname == 'api.deepseek.com':
            os.environ.setdefault('AGENT_V3_THINKING', 'disabled')
        llm = OpenAICompatLLM(model=model, base_url=base, api_key=credential)
    if version == 'v2':
        from v2.agent_v2.orchestrator import AgentV2, AgentV2Config
        from v2.agent_v2.session import ShortTermSession
        from v2.agent_v2.intent import Intent, parse_intent
        class Planner:
            def plan(self, *args):
                return original_plan(episode['turns'][bank.turn])
        class Synthesizer:
            supports_general_knowledge = True
            def synthesize(self, request, plan, results, evidence):
                return frozen_answer(evidence)
        class Classifier:
            def classify(self, request):
                return parse_intent(intent_dict(episode['turns'][bank.turn]), source='recorded')
        planner, synth, classifier = Planner(), Synthesizer(), Classifier()
        if live_model:
            from v2.agent_v2.llm import StructuredLLMPlanner, LLMEvidenceSynthesizer
            from v2.agent_v2.intent import IntentClassifier
            planner, synth, classifier = StructuredLLMPlanner(llm, registry.catalog), LLMEvidenceSynthesizer(llm, catalog=registry.catalog), IntentClassifier(llm)
        agent = AgentV2(registry=registry, planner=planner, synthesizer=synth, classifier=classifier,
                        session=ShortTermSession(), config=AgentV2Config(max_seconds=seconds, enable_web_fallback=True,
                        record_sub_agents=False, record_capabilities=False, record_intents=False, debate=False))
        class FixtureMemory:
            def preferences(self, session_id):
                return episode['turns'][bank.turn].get('result', {}).get('request', {}).get('metadata', {}).get('preferences', [])
            def remember_answer(self, *args, **kwargs):
                pass
        agent.memory = FixtureMemory()
    else:
        from v2.agent_v3.graph import AgentV3, AgentV3Config
        brain = ContractBrain(bank)
        if live_model:
            from v2.agent_v3.runtime import build_model
            from v2.agent_v3.brain import ModelBrain
            brain = ModelBrain(build_model())
        agent = AgentV3(registry=registry, brain=brain, config=AgentV3Config(max_seconds=seconds,
            answer_reserve_seconds=min(10, seconds / 3), enable_web=True, debate=False))
    return agent, bank


def run_episode(version, episode, live_model=False, seconds=60):
    agent, bank = build_pair_member(version, episode, live_model, seconds)
    output = []
    pending = None
    try:
        for index, turn in enumerate(episode['turns']):
            bank.turn = index
            runtime_web = turn.get('config', {}).get('enable_web_fallback', True)
            agent.config = replace(agent.config, **({'enable_web_fallback':runtime_web} if version=='v2' else {'enable_web':runtime_web}))
            session = turn.get('session_id', '')
            text = turn['text']
            if version == 'v3':
                history = agent.store.get(session)
                history['preferences'] = turn.get('result', {}).get('request', {}).get('metadata', {}).get('preferences', [])
                agent.store.put(session, history)
            try:
                # Explicit confirmation control messages map to V3's UI API.
                # The logged input remains the exact original text.
                if version == 'v3' and pending and text in {'确认', '取消'}:
                    result = agent.resume(session_id=session, run_id=pending, approve=text == '确认')
                else:
                    result = agent.run(text, session_id=session, allow_web=turn.get('allow_web', False))
                pending = result.run_id if result.status.value == 'waiting_confirmation' else None
                data = result.to_dict()
                from v2.agent_v2.verification import verify_answer
                shared_check = verify_answer(result.answer, result.evidence, answer_mode=result.answer_mode, results=result.results)
                calls = [row for row in bank.calls if row['turn'] == index]
                expected = turn.get('result', {})
                output.append({'input': {'text': text, 'session_id': session, 'allow_web': turn.get('allow_web', False)},
                    'runtime_web_enabled': runtime_web,
                    'result': data, 'observations': observe(data), 'calls': calls,
                    'fixture_complete': all(row['fixture_match'] for row in calls),
                    'reference_status': expected.get('status'), 'status_matches_reference': data['status'] == expected.get('status'),
                    'entities_match_reference': data['request']['entities'] == expected.get('request', {}).get('entities'),
                    'common_checks': {'grounding_ok': shared_check.ok,
                        'unknown_citations': list(shared_check.unknown_citations),
                        'ungrounded_numbers': list(shared_check.ungrounded_numbers),
                        'expected_mutation_calls': sum(row['task']['capability']=='state.mutate' for row in turn.get('tool_records',[])),
                        'actual_mutation_calls': sum(row['capability']=='state.mutate' for row in calls),
                        'tool_requests_match_reference': sorted(key(row['capability'],row['arguments']) for row in calls)==sorted(key(row['task']['capability'],row['task']['arguments']) for row in turn.get('tool_records',[])),
                        'no_unapproved_mutation': all(row['mutation_authorized'] for row in calls if row['capability']=='state.mutate')},
                    'semantic_quality': 'pending_review' if live_model else 'not_evaluated_contract_mode'})
            except Exception as exc:
                import traceback
                trace=traceback.extract_tb(exc.__traceback__)
                output.append({'input': {'text': text, 'session_id': session}, 'error_type': type(exc).__name__,
                               'error_location':f'{Path(trace[-1].filename).name}:{trace[-1].lineno}' if trace else '',
                               'semantic_quality': 'not_evaluated', 'fixture_complete': False})
    finally:
        if version == 'v3':
            agent.store.close()
    return {'version': version, 'episode': episode['id'], 'test': episode['test'], 'turns': output,
            'mode': 'live_model_frozen_tools' if live_model else 'recorded_plan_contract'}


def compare_planning(inventory):
    """All actual planner inputs, including parameterized and imported questions.

Both receive the captured request/context and semantic labels. This tests
planning, not model classification. No response fixture is invented.
"""
    from v2.agent_v2.models import NormalizedRequest, RouteDecision, RouteKind
    from v2.agent_v2.intent import parse_intent
    from v2.agent_v2.planning import RulePlanner, intent_of
    from v2.agent_v3.tools import Registry
    from v2.agent_v3.brain import ModelBrain
    from v2.agent_v3.context import RunContext
    from v2.agent_eval_capture import plain
    rows = []
    registry = Registry()
    for spec in registry.catalog.specs():
        registry.register(spec.name, lambda args, ctx: None)  # planning only; never executed
    for index, row in enumerate(inventory.get('planning_cases', [])):
        item = {'id': f'original-plan-{index+1:04d}', 'test': row['test'], 'request': row['request'],
                'source_stack': row['source_stack'], 'reference_plan': row.get('result')}
        for version in ('v2', 'v3'):
            try:
                raw = row['request']
                request = NormalizedRequest(**{**raw, 'entities': tuple(raw.get('entities', []))})
                decision = RouteDecision(RouteKind(row['route']['kind']), tuple(row['route']['packs']), row['route']['reason'],
                                         intent=parse_intent(row['route']['intent'], source='recorded') if row['route'].get('intent') else None)
                if decision.intent is None:
                    decision=replace(decision,intent=intent_of(request,decision))
                plan = RulePlanner().plan(request, decision) if version=='v2' else ModelBrain(None).plan(request, decision, registry, RunContext('planning', time.monotonic()+20))
                item[version] = plain(plan)
            except Exception as exc:
                item[version] = {'error_type': type(exc).__name__}
        rows.append(item)
    return rows


def save_report(output, records, inventory):
    (output / 'summary.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# V2 原题、同数据对比', '',
        '参考状态一致只表示与原测试轨迹一致，不是回答质量合格；fixture_missing 是测评数据覆盖缺口。', '',
        f"原始端到端会话 {len(inventory['episodes'])}；本次版本结果 {len(records)}。独立输入边界 {len(inventory['standalone_boundaries'])} 另存 coverage.json，不冒充已执行问答。", '',
        '| 会话 | 版本 | 原题 | 状态 | 与参考状态一致 | 数据覆盖 |', '|---|---|---|---|---|---|']
    for record in records:
        for turn in record['turns']:
            text = turn['input']['text'].replace('|', '\\|').replace('\n', '<br>')
            lines.append(f"| {record['episode']} | {record['version']} | {text} | {turn.get('result', {}).get('status', turn.get('error_type'))} | {turn.get('status_matches_reference')} | {turn['fixture_complete']} |")
    lines.extend(['', '## 完整答案与证据'])
    for record in records:
        lines.append(f"\n[{record['episode']} / {record['version']}]({record['episode']}-{record['version']}.json)")
    lines.extend(['', '[原始规划问题与两版计划](planning.json)', '', '## 同题答案'])
    grouped = {}
    for record in records:
        grouped.setdefault(record['episode'], {})[record['version']] = record
    for eid, versions in grouped.items():
        longest = max(len(record['turns']) for record in versions.values())
        for index in range(longest):
            first = next(record['turns'][index] for record in versions.values() if index < len(record['turns']))
            lines.extend(['', f"### {eid} / 第 {index+1} 轮", '', first['input']['text']])
            for version, record in versions.items():
                if index < len(record['turns']):
                    turn = record['turns'][index]
                    lines.extend(['', f'**{version.upper()}**', '', turn.get('result', {}).get('answer') or turn.get('error_type', '未返回答案')])
    (output / 'report.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--episodes', nargs='+')
    parser.add_argument('--versions', nargs='+', choices=['v2', 'v3'], default=['v2', 'v3'])
    parser.add_argument('--live-model', action='store_true')
    parser.add_argument('--seconds', type=float, default=60)
    parser.add_argument('--worker-version', choices=['v2', 'v3'], help=argparse.SUPPRESS)
    args = parser.parse_args()
    inventory = json.loads(Path(args.corpus).read_text(encoding='utf-8'))
    source = Path(__file__).resolve().parent / 'agent_v2/test_agent_v2.py'
    if hashlib.sha256(source.read_text(encoding='utf-8').encode()).hexdigest() != inventory['source_sha256']:
        parser.error('Source test file changed; recapture original inputs first')
    if args.seconds <= 0:
        parser.error('seconds must be positive')
    if inventory['pytest_exit_code']:
        parser.error('Original source tests did not pass; inspect capture before comparison')
    episodes = [e for e in inventory['episodes'] if not args.episodes or e['id'] in args.episodes]
    if not episodes or (args.episodes and set(args.episodes) - {e['id'] for e in episodes}):
        parser.error('Unknown episode IDs')
    output = Path(args.output_dir)
    if args.worker_version:
        if not args.live_model or len(episodes) != 1:
            parser.error('Worker requires one episode and --live-model')
        result = run_episode(args.worker_version, episodes[0], True, args.seconds)
        (output / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        os._exit(0)
    if output.exists() and any(output.iterdir()):
        parser.error('Use a new empty output directory')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'conditions.json').write_text(json.dumps({
        'mode':'live_model_frozen_tools' if args.live_model else 'recorded_plan_contract',
        'source_sha256':inventory['source_sha256'],
        'corpus_sha256':hashlib.sha256(Path(args.corpus).read_bytes()).hexdigest(),
        'business_tools':'captured fixture responses only; unmatched calls never reach live providers',
        'same_inputs':['exact original text','original session grouping','per-turn web consent','recorded user preferences','per-turn tool argument/response fixtures'],
        'common_budget_seconds':args.seconds,
        'not_reproduced':['original scripted answer drafts','provider timing/fault injection','real process restart','all internal implementation assertions'],
        'quality_rubric':['回答原题，正确承接或切换对象','数字与所引证据、期间、单位一致','区分事实、报道和推断','缺失证据明确说明，不伪造数据','写操作必须经过确认且不重复执行','不把模拟证据声称为实时查询'],
        'quality_scoring':'manual review or separate semantic evaluation; contract status is not a quality grade'
    },ensure_ascii=False,indent=2),encoding='utf-8')
    (output / 'planning.json').write_text(json.dumps(compare_planning(inventory), ensure_ascii=False, indent=2), encoding='utf-8')
    (output / 'coverage.json').write_text(json.dumps({k: inventory[k] for k in ('source_sha256', 'coverage', 'outcomes', 'standalone_boundaries')}, ensure_ascii=False, indent=2), encoding='utf-8')
    records = []
    for index, episode in enumerate(episodes):
        for version in (args.versions if index % 2 == 0 else list(reversed(args.versions))):
            print('START', episode['id'], version, flush=True)
            if args.live_model:
                target = output / f"worker-{episode['id']}-{version}"
                target.mkdir()
                with (target / 'worker.log').open('w', encoding='utf-8') as log:
                    try:
                        child = subprocess.run([sys.executable, '-m', 'v2.agent_original_comparison', '--corpus', str(Path(args.corpus).resolve()),
                            '--output-dir', str(target.resolve()), '--episodes', episode['id'], '--live-model', '--worker-version', version,
                            '--seconds', str(args.seconds)], stdout=log, stderr=log, timeout=(args.seconds+20)*len(episode['turns'])+30)
                        error = 'worker_error' if child.returncode else ''
                    except subprocess.TimeoutExpired:
                        error = 'process_timeout'
                if not error and (target / 'result.json').exists():
                    result = json.loads((target / 'result.json').read_text(encoding='utf-8'))
                else:
                    result = {'episode':episode['id'], 'test':episode['test'], 'version':version, 'mode':'live_model_frozen_tools',
                        'turns':[{'input':{'text':turn['text'],'session_id':turn.get('session_id','')}, 'error_type':error or 'worker_error',
                                  'fixture_complete':False,'semantic_quality':'not_evaluated'} for turn in episode['turns']]}
            else:
                result = run_episode(version, episode, False, args.seconds)
            (output / f"{episode['id']}-{version}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
            records.append(result)
            save_report(output, records, inventory)
    print('DONE', len(records), 'version/episode results; no live business tools', flush=True)


if __name__ == '__main__':
    main()
