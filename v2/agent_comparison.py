"""External V2/V3 benchmark driver; neither Agent invokes the other.

Default exports a reviewable question manifest without network calls.
--live executes native read-only stacks in isolated bounded subprocesses.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from v2.agent_v3.evaluation_cases import CASES, OFFLINE_ONLY


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def observe(result):
    """Diagnostics, not semantic correctness scores."""
    return {
        'status': result.get('status'), 'seconds': result.get('elapsed_ms', 0) / 1000,
        'answer_present': bool(result.get('answer', '').strip()),
        'verification_ok': result.get('verification', {}).get('ok'),
        'evidence_count': len(result.get('evidence', [])),
        'tools': [row['capability'] for row in result.get('results', [])],
        'synthesis': result.get('synthesis', {}).get('outcome'),
        'quality_verdict': 'pending_review',
    }


def run_turns(agent, case):
    return [agent.run(text, session_id='benchmark-' + case.id, allow_web=case.allow_web).to_dict()
            for text in (*case.preceding, case.question)]


def build_agent(version, output, seconds):
    from dotenv import load_dotenv
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / '.env', override=False)
    sys.path.insert(0, str(root / 'web/backend'))
    from v2.agent_common.llm import OpenAICompatLLM
    defaults = OpenAICompatLLM()
    model = os.environ.get('AGENT_V3_MODEL') or defaults.model
    base = os.environ.get('AGENT_V3_BASE_URL') or defaults.base_url
    key = os.environ.get('AGENT_V3_API_KEY') or defaults.api_key
    if not key:
        raise ValueError('Model API credentials are not configured')
    from urllib.parse import urlparse
    if urlparse(base).hostname == 'api.deepseek.com':
        # Same compatibility setting as the deployed V3 service: forced
        # structured tool selection is rejected by this provider's thinking mode.
        os.environ.setdefault('AGENT_V3_THINKING', 'disabled')
    for prefix in ('AGENT_V3_', 'AGENT_LLM_'):
        for name, value in [('MODEL', model), ('BASE_URL', base), ('API_KEY', key)]:
            os.environ[prefix + name] = value
    from v2.data import cost_ledger
    cost_ledger._DB_PATH = output / 'costs.sqlite'
    os.environ['AGENT_V2_SESSION_DB'] = str(output / 'v2-sessions.sqlite')
    os.environ['AGENT_V2_USER_MEMORY'] = str(output / 'v2-memory.json')
    if version == 'v2':
        from v2.agent_v2.runtime import build_workspace_agent
        from v2.agent_v2.orchestrator import AgentV2Config
        agent = build_workspace_agent(llm=OpenAICompatLLM(model=model, base_url=base, api_key=key),
            enable_web=True, config=AgentV2Config(max_seconds=seconds, allow_mutations=False,
                record_sub_agents=False, record_capabilities=False, record_intents=False))
        agent.memory = None
        handlers = agent.registry._handlers
    else:
        from v2.agent_v3.runtime import build_workspace_agent
        from v2.agent_v3.graph import AgentV3Config
        agent = build_workspace_agent(config=AgentV3Config(max_seconds=seconds, enable_web=True),
                                      data_dir=output / 'state', enable_mutations=False, recall=lambda *args: [])
        agent.event_source = None
        handlers = agent.registry.handlers
    for name in list(handlers):
        if agent.registry.catalog.get(name).mutating or name.startswith(('account.', 'state.', 'lab.')):
            del handlers[name]
    write(output / 'conditions.json', {'model': model, 'version': version, 'max_seconds_per_turn': seconds,
        'temperature': 0, 'mode': 'native_stack_live',
        'thinking': os.environ.get('AGENT_V3_THINKING') if version == 'v3' else 'provider_default',
        'limitations': ['Native tool stacks, retrieval and caches differ; this is not a controlled framework-only experiment.',
                        'Live data can change between runs; alternating order reduces but does not remove cache/order bias.',
                        'SDK retries, thinking defaults and token budgets can differ; no equal-cost claim.',
                        'Public market archive recall may differ; no account/state/lab handlers are enabled.']})
    return agent


def worker(args, case):
    output = Path(args.output_dir)
    try:
        agent = build_agent(args.worker, output, args.seconds)
        turns = run_turns(agent, case)
        write(output / 'result.json', {'case': case.to_dict(), 'turns': turns, 'observations': observe(turns[-1])})
        code = 0
    except Exception as exc:
        write(output / 'error.json', {'error_type': type(exc).__name__})
        code = 1
    # Do not leave timed-out provider threads running after results are saved.
    os._exit(code)


def report(output, records):
    write(output / 'summary.json', records)
    lines = ['# V2 / V3 测评对比', '', '执行及引用校验不是答案质量分数；未人工评审项统一为 pending_review。', '',
             '| 问题 | 版本 | 重复 | 执行 | Agent 状态 | 秒 | 证据数 | 引用校验 |', '|---|---|---|---|---|---:|---:|---|']
    for row in records:
        metrics = row.get('observations', {})
        lines.append(f"| {row['case']} | {row['version']} | {row['repeat']} | {row['execution']} | {metrics.get('status', '')} | {metrics.get('seconds', '')} | {metrics.get('evidence_count', '')} | {metrics.get('verification_ok', '')} |")
    for version in ('v2', 'v3'):
        selected = [row for row in records if row['version'] == version]
        returned = [row for row in selected if row.get('observations')]
        failed = [row for row in selected if not row.get('observations') or row['observations']['status'] in {'failed', 'cancelled'}]
        times = [row['observations']['seconds'] for row in returned if row not in failed]
        if selected:
            lines.extend(['', f"{version}：总任务 {len(selected)}，取得结果 {len(returned)}，失败/超时 {len(failed)}。" +
                          (f' 非失败返回样本耗时中位数 {statistics.median(times):.2f} 秒；其中可能有 partial 或澄清，不代表答案合格，不能单独用来评优。' if times else '')])
    lines.extend(['', '## 逐题答案与评审标准'])
    for row in records:
        path = output / row['artifact'] / 'result.json'
        if not path.exists():
            continue
        item = json.loads(path.read_text(encoding='utf-8'))
        lines.extend(['', f"### {row['case']} / {row['version']} / {row['repeat']}", '', item['case']['question'], '',
                      '验收：' + '；'.join(item['case']['criteria']), '', item['turns'][-1]['answer'], '',
                      f"[完整多轮结果与证据]({row['artifact']}/result.json)", '', '人工结论：待评审。'])
    (output / 'report.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--versions', nargs='+', choices=['v2', 'v3'], default=['v2', 'v3'])
    parser.add_argument('--cases', nargs='+')
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--seconds', type=int, default=120)
    parser.add_argument('--worker', choices=['v2', 'v3'], help=argparse.SUPPRESS)
    args = parser.parse_args()
    selected = [case for case in CASES if not args.cases or case.id in args.cases]
    if not selected or (args.cases and set(args.cases) - {case.id for case in CASES}):
        parser.error('Unknown or empty case selection')
    if args.repeat < 1 or args.seconds < 1:
        parser.error('repeat and seconds must be positive')
    output = Path(args.output_dir).resolve()
    if args.worker:
        if not args.live or len(selected) != 1:
            parser.error('Worker requires --live and exactly one case')
        worker(args, selected[0])
    if output.exists() and any(output.iterdir()):
        parser.error('Use a new empty output directory to preserve existing results')
    output.mkdir(parents=True, exist_ok=True)
    write(output / 'cases.json', {'created_at': datetime.now(timezone.utc).isoformat(),
          'cases': [case.to_dict() for case in selected], 'offline_only': OFFLINE_ONLY})
    if not args.live:
        print(f'Exported {len(selected)} cases. No model/provider calls. Add --live with a new output directory to execute.')
        return
    records = []
    for repeat in range(args.repeat):
        for index, case in enumerate(selected):
            versions = list(dict.fromkeys(args.versions))
            if (repeat + index) % 2:
                versions.reverse()
            for version in versions:
                artifact = f'{case.id}/{repeat+1}/{version}'
                target = output / artifact
                target.mkdir(parents=True)
                started = time.monotonic()
                print(f'START {artifact}', flush=True)
                with (target / 'worker.log').open('w', encoding='utf-8') as log:
                    try:
                        process = subprocess.run([sys.executable, '-m', 'v2.agent_comparison', '--live', '--worker', version,
                            '--cases', case.id, '--seconds', str(args.seconds), '--output-dir', str(target)],
                            stdout=log, stderr=log, timeout=(args.seconds + 20) * (len(case.preceding) + 1) + 45)
                        execution = 'returned' if process.returncode == 0 and (target / 'result.json').exists() else 'worker_error'
                    except subprocess.TimeoutExpired:
                        execution = 'process_timeout'
                row = {'case': case.id, 'version': version, 'repeat': repeat+1, 'execution': execution,
                       'artifact': artifact, 'wall_seconds': time.monotonic() - started}
                if execution == 'returned':
                    row['observations'] = json.loads((target / 'result.json').read_text(encoding='utf-8'))['observations']
                records.append(row)
                report(output, records)
                print(f'END {artifact}: {execution}', flush=True)


if __name__ == '__main__':
    main()
