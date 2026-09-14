"""Auditable model mappings and account quota snapshots, separate from events."""
import json
import hashlib
import math
import os
import uuid
from datetime import datetime, timezone

import requests

SCHEMA = '''
CREATE TABLE IF NOT EXISTS billing_aliases(id TEXT PRIMARY KEY, payload TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS tavily_quota(month TEXT PRIMARY KEY, payload TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS billing_audit(id TEXT PRIMARY KEY, payload TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS billing_sync_state(id TEXT PRIMARY KEY, payload TEXT NOT NULL);
'''


# DeepSeek documents these names as the same Flash billing family across the
# V4/V4.1 transition.  Keep this explicit and narrow: unrelated requested
# models must never inherit a price merely because the provider returned a
# different model name.
DEEPSEEK_FLASH_FAMILY = (
    'deepseek-flash',
    'deepseek-v4-flash',
    'deepseek-v4-flash-vision-exp',
    'deepseek-chat',
    'deepseek-reasoner',
)


def deepseek_model_family(model):
    model = str(model or '').strip()
    if model in DEEPSEEK_FLASH_FAMILY:
        return (model,) + tuple(item for item in DEEPSEEK_FLASH_FAMILY if item != model)
    return (model,) if model else ()


def same_deepseek_family(left, right):
    left, right = str(left or '').strip(), str(right or '').strip()
    return bool(left and right and right in deepseek_model_family(left))


def key_tag():
    key = os.environ.get('TAVILY_API_KEY', '').strip()
    return hashlib.sha256(key.encode()).hexdigest() if key else None


def number(value):
    if isinstance(value, bool):
        raise ValueError('invalid number')
    n = float(value)
    if not math.isfinite(n) or n < 0:
        raise ValueError('invalid number')
    return n


def configure_alias(data):
    from v2.data.usage_ledger import _conn, init, timestamp, now_iso
    source, target = str(data['model']).strip(), str(data['target']).strip()
    evidence = str(data['source']).strip()
    start, end = timestamp(data['effective_at']), timestamp(data['review_after'])
    if not source or not target or source == target or max(len(source), len(target)) > 120 or not evidence or len(evidence) > 500 or start >= end or data.get('confirmed') is not True:
        raise ValueError('需确认映射依据和有效时间')
    item = dict(id=uuid.uuid4().hex, provider='DeepSeek', model=source, target=target,
                source=evidence, effective_at=start, review_after=end, created_at=now_iso())
    with _conn() as c:
        init(c)
        if not c.execute('SELECT 1 FROM usage_prices WHERE provider=? AND model=?', ('DeepSeek', target)).fetchone():
            raise ValueError('目标模型没有价格版本')
        c.execute('INSERT INTO billing_aliases VALUES (?,?)', (item['id'], json.dumps(item)))
    return item


def resolve_alias(c, provider, model, at):
    if provider != 'DeepSeek':
        return model, None
    rows = [json.loads(r[0]) for r in c.execute('SELECT payload FROM billing_aliases')]
    rows = [r for r in rows if r['model'] == model and r['effective_at'] <= at < r['review_after']]
    if not rows:
        return model, None
    alias = max(rows, key=lambda r: (r['effective_at'], r['created_at']))
    return alias['target'], alias


def quota_status():
    from v2.data.usage_ledger import _conn, init, now_iso
    month = now_iso()[:7]
    with _conn() as c:
        init(c)
        row = c.execute('SELECT payload FROM tavily_quota WHERE month=?', (month,)).fetchone()
        last = c.execute("SELECT payload FROM billing_sync_state WHERE id='tavily'").fetchone()
    sync = json.loads(last[0]) if last else {}
    if not row:
        return dict(status='unconfigured', month=month, timezone='UTC', message=sync.get('error') or '本月账户额度尚未同步或校准，不能假定还有 1000 免费 credits')
    q = json.loads(row[0])
    return {**q, 'free_remaining': max(0, 1000-q['used']), 'paid_credits_estimate': max(0, q['used']-1000),
            'message': sync.get('error') or '账户快照＋本地后续用量估算；账户可能还有其他应用用量。不是本项目历史账单。'}


def save_quota(used, source, observed_at, official=None):
    from v2.data.usage_ledger import _conn, init, now_iso, timestamp
    at = timestamp(observed_at)
    used = number(used)
    month = at[:7]
    if month != now_iso()[:7]:
        raise ValueError('仅允许校准当前 UTC 自然月')
    with _conn() as c:
        init(c)
        c.execute('BEGIN IMMEDIATE')
        old = c.execute('SELECT payload FROM tavily_quota WHERE month=?', (month,)).fetchone()
        previous = json.loads(old[0]) if old else None
        # A delayed provider response must never give previously consumed credits back.
        if official and previous and previous.get('key_tag') == key_tag():
            used = max(used, previous['used'])
        q = dict(id=uuid.uuid4().hex, status='ok', month=month, timezone='UTC', used=used,
                 observed_at=at, last_event_at=at, source=source, official=official,
                 free_limit=1000, unit_price_usd=.008)
        q['key_tag'] = key_tag()
        c.execute('INSERT INTO billing_audit VALUES (?,?)', (q['id'], json.dumps({'type': 'quota_calibration', 'previous': previous, 'new': q})))
        c.execute('INSERT OR REPLACE INTO tavily_quota VALUES (?,?)', (month, json.dumps(q)))
    return quota_status()


def sync_tavily():
    from v2.data.usage_ledger import _conn, init, now_iso
    at = now_iso()
    with _conn() as c:
        init(c)
        c.execute('BEGIN IMMEDIATE')
        row = c.execute("SELECT payload FROM billing_sync_state WHERE id='tavily'").fetchone()
        old = json.loads(row[0]) if row else {}
        if old.get('key_tag') == key_tag() and old.get('attempted_at') and (datetime.fromisoformat(at)-datetime.fromisoformat(old['attempted_at'])).total_seconds() < 60:
            throttled = True
        else:
            throttled = False
            c.execute('INSERT OR REPLACE INTO billing_sync_state VALUES (?,?)', ('tavily', json.dumps({'attempted_at': at, 'key_tag': key_tag()})))
    if throttled:
        return quota_status()
    result = _fetch_tavily()
    with _conn() as c:
        c.execute('UPDATE billing_sync_state SET payload=? WHERE id=?', (json.dumps({'attempted_at': at, 'key_tag': key_tag(), 'error': result['message'] if result['status'] == 'error' else None}), 'tavily'))
    return result


def _fetch_tavily():
    from v2.data.usage_ledger import now_iso
    q = quota_status()
    if q.get('official') and q.get('key_tag') == key_tag() and (datetime.now(timezone.utc)-datetime.fromisoformat(q['observed_at'])).total_seconds() < 60:
        return q
    key = os.environ.get('TAVILY_API_KEY', '').strip()
    if not key:
        return {**q, 'status': 'error', 'message': '未配置 Tavily API Key；可手动校准本月账户用量'}
    try:
        response = requests.get('https://api.tavily.com/usage', headers={'Authorization': 'Bearer '+key}, timeout=12, allow_redirects=False)
        response.raise_for_status()
        if response.status_code != 200:
            raise ValueError('redirect')
        account = response.json()['account']
        plan = str(account['current_plan'])
        limit = number(account['plan_limit'])
        if limit != 1000 or plan.lower() not in ('researcher', 'free'):
            from v2.data.usage_ledger import _conn, init
            with _conn() as c:
                init(c)
                row = c.execute('SELECT payload FROM tavily_quota WHERE month=?', (q['month'],)).fetchone()
                if row:
                    invalid = json.loads(row[0])
                    invalid['status'] = 'unsupported_plan'
                    c.execute('UPDATE tavily_quota SET payload=? WHERE month=?', (json.dumps(invalid), q['month']))
            return {**q, 'status': 'error', 'message': '账户不是已配置的每月 1000 credits 免费套餐；未自动套用此计价规则'}
        used, paid = number(account['plan_usage']), number(account['paygo_usage'])
        total = used+paid
        if used > limit:
            # Researcher can report cumulative plan_usage, including PAYG.
            # Accept only when the separately reported excess corroborates it.
            if not math.isclose(used-limit, paid, abs_tol=1e-6, rel_tol=0):
                raise ValueError('inconsistent cumulative usage')
            total = used
        official = dict(plan=plan, plan_usage=used, plan_limit=limit, paygo_usage=paid)
        return save_quota(total, 'Tavily /usage 官方账户快照', now_iso(), official)
    except Exception:
        return {**q, 'status': 'error', 'message': '账户同步失败或返回格式无法识别，未覆盖已有校准；可稍后重试'}


def allocate_tavily(c, event):
    at = event['occurred_at']
    row = c.execute('SELECT payload FROM tavily_quota WHERE month=?', (at[:7],)).fetchone()
    if not row:
        return None, '本月账户免费额度未同步或校准'
    q = json.loads(row[0])
    if q['status'] != 'ok' or q.get('key_tag') != key_tag():
        return None, '账户套餐或 API Key 已变化，需要重新同步额度'
    if (datetime.fromisoformat(at)-datetime.fromisoformat(q['observed_at'])).total_seconds() > 86400:
        return None, '账户额度快照超过24小时，需要重新同步'
    if at < q['observed_at'] or at < q['last_event_at']:
        return None, '调用早于账户校准或已分配记录，历史免费额度无法确定'
    units = event['usage'].get('units')
    if units is None:
        return None, '未返回 credits 用量'
    units = number(units)
    free = min(units, max(0, 1000-q['used']))
    paid = units-free
    detail = dict(quota_id=q['id'], source=q['source'], observed_at=q['observed_at'],
                  account_used_before=q['used'], free_credits=free, paid_credits=paid,
                  gross_amount=units*.008, currency='USD', unit_price=.008)
    q['used'] += units
    q['last_event_at'] = at
    c.execute('UPDATE tavily_quota SET payload=? WHERE month=?', (json.dumps(q), q['month']))
    event['quota'] = detail
    event['price'] = dict(id='tavily-researcher-paygo', source='用户确认：每月1000 credits，超额$0.008/credit', rates={'unit': .008}, currency='USD')
    return paid*.008, ''
