"""Versioned usage estimates. No keys, prompts or search content are stored."""
from __future__ import annotations

import json
import logging
import math
import uuid
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from v2.data.cost_ledger import _conn, _prices, _PATH_TO_ENDPOINT
from v2.usage_context import current_channel, current_run

logger = logging.getLogger(__name__)
ET = ZoneInfo('America/New_York')
TAVILY_UNIT_PRICE_USD = .008
RECENT_DETAIL_HOURS = 24
SCHEMA = """
CREATE TABLE IF NOT EXISTS usage_prices (
 id TEXT PRIMARY KEY, provider TEXT NOT NULL, model TEXT NOT NULL,
 effective_at TEXT NOT NULL, review_after TEXT NOT NULL, created_at TEXT NOT NULL,
 payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS usage_events (
 id TEXT PRIMARY KEY, occurred_at TEXT NOT NULL, category TEXT NOT NULL,
 provider TEXT NOT NULL, model TEXT NOT NULL, cost_usd REAL,
 status TEXT NOT NULL, payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS usage_time ON usage_events(occurred_at);
CREATE TABLE IF NOT EXISTS usage_rollups (
 day TEXT NOT NULL, category TEXT NOT NULL, provider TEXT NOT NULL,
 currency TEXT NOT NULL, payload TEXT NOT NULL,
 PRIMARY KEY(day, category, provider, currency)
);
"""


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def init(conn):
    conn.executescript(SCHEMA)
    from v2.data.billing_rules import SCHEMA as BILLING_SCHEMA
    conn.executescript(BILLING_SCHEMA)


def timestamp(value):
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        raise ValueError('时间必须包含时区')
    return parsed.astimezone(timezone.utc).isoformat()


def add_price(data):
    currency = data.get('currency', 'USD')
    if currency not in ('USD', 'CNY'):
        raise ValueError('仅支持 USD 和 CNY')
    provider = data.get('provider')
    if provider not in ('DeepSeek', 'Tavily', 'Financial Datasets', 'Other LLM'):
        raise ValueError('不支持的供应商')
    model = str(data.get('model', '')).strip()
    if not model or len(model) > 120:
        raise ValueError('请填写模型名或端点名')
    effective = timestamp(data['effective_at'])
    review = timestamp(data['review_after'])
    if review <= effective:
        raise ValueError('价格复核期限必须晚于生效时间')
    rates = {}
    required = ('input', 'cached_input', 'output') if provider in ('DeepSeek', 'Other LLM') else ('unit',)
    for key in required:
        value = float(data['rates'][key])
        if not math.isfinite(value) or value < 0 or value > 100000:
            raise ValueError('单价必须为有限非负数')
        rates[key] = value
    source = str(data.get('source', '')).strip()
    if not source or len(source) > 500:
        raise ValueError('请填写价格来源／套餐说明')
    record = dict(id=uuid.uuid4().hex, provider=provider, model=model,
                  effective_at=effective, review_after=review, created_at=now_iso(),
                  rates=rates, source=source, currency=currency)
    with _conn() as conn:
        init(conn)
        conn.execute('INSERT INTO usage_prices VALUES (?,?,?,?,?,?,?)',
                     (record['id'], provider, model, effective, review, record['created_at'], json.dumps(record)))
    return record


def prices():
    with _conn() as conn:
        init(conn)
        return [{k: v for k, v in json.loads(r[0]).items() if k != 'page_snapshot'} for r in conn.execute('SELECT payload FROM usage_prices ORDER BY effective_at DESC, created_at DESC LIMIT 100')]


def quote_event(conn, event):
    """Price one event inside the same transaction used to persist it."""
    from v2.data.billing_rules import (
        deepseek_model_family,
        resolve_alias,
        same_deepseek_family,
    )
    at, provider, model = event['occurred_at'], event['provider'], event['model']
    category, usage = event['category'], event['usage']
    target, alias = resolve_alias(conn, provider, model, at)
    # A historical confirmation must not override a different recorded request.
    requested = event.get('requested_model')
    if alias and requested and requested not in (model, target):
        target, alias = model, None
    row = conn.execute('SELECT payload FROM usage_prices WHERE provider=? AND model=? AND effective_at<=? ORDER BY effective_at DESC, created_at DESC LIMIT 1', (provider, target, at)).fetchone()
    basis = 'confirmed_alias' if alias else 'response_model'
    # A request to a priced exact model is an estimate basis, not a global alias.
    if not row and requested and requested != model:
        row = conn.execute('SELECT payload FROM usage_prices WHERE provider=? AND model=? AND effective_at<=? ORDER BY effective_at DESC, created_at DESC LIMIT 1', (provider, requested, at)).fetchone()
        if row:
            target, basis, alias = requested, 'requested_model', None
    # DeepSeek may return a canonical model name that differs from the legacy
    # request name.  Use another official name only when both names belong to
    # the explicitly documented Flash family; never alias an unrelated model.
    if not row and provider == 'DeepSeek' and requested and same_deepseek_family(model, requested):
        for candidate in deepseek_model_family(model):
            if candidate in (target, requested):
                continue
            candidate_row = conn.execute(
                'SELECT payload FROM usage_prices WHERE provider=? AND model=? AND effective_at<=? ORDER BY effective_at DESC, created_at DESC LIMIT 1',
                (provider, candidate, at),
            ).fetchone()
            if candidate_row:
                row, target, basis, alias = candidate_row, candidate, 'official_model_family', None
                break
    # Owner policy: an unrecognized DeepSeek model name uses the official
    # deepseek-flash price that was valid when the call occurred.  The
    # original/requested names remain on the event for auditability.
    if not row and category == 'llm' and provider == 'DeepSeek':
        fallback = 'deepseek-flash'
        fallback_row = conn.execute(
            'SELECT payload FROM usage_prices WHERE provider=? AND model=? AND effective_at<=? ORDER BY effective_at DESC, created_at DESC LIMIT 1',
            (provider, fallback, at),
        ).fetchone()
        if fallback_row:
            row, target, basis, alias = fallback_row, fallback, 'deepseek_flash_fallback', None
    price = json.loads(row[0]) if row else None
    if price:
        price.pop('page_snapshot', None)
        if price.get('schedule'):
            schedule = price['schedule']
            local = datetime.fromisoformat(at).astimezone(ZoneInfo(schedule['timezone']))
            minute = local.hour * 60 + local.minute
            peak = local.weekday() in schedule['weekdays'] and any(start <= minute < end for start, end in schedule['windows'])
            price['applied_period'] = '高峰时段' if peak else '空闲时段'
            if peak:
                price['rates'] = schedule['peak_rates']
    if category == 'search' and provider == 'Tavily' and model == 'search':
        price = dict(
            id='tavily-flat-credit-rate',
            rates={'unit': TAVILY_UNIT_PRICE_USD},
            source='项目计费规则：所有 Tavily credits 统一按 $0.008/credit 估算，不抵扣免费额度',
            currency='USD',
        )
        event.pop('quota', None)
        event['quota_note'] = '不抵扣每月免费额度；所有 credits 统一按 $0.008/credit 估算'
    elif not price and category == 'data':
        price = dict(id='fd-config-snapshot', rates={'unit': _prices().get(event.get('endpoint'), .02)}, source='FD_PRICES / existing default', currency='USD')
    event.update(price=price, pricing_model=target, pricing_basis=basis, alias=alias, breakdown=None)
    cost, reason = None, '模型未匹配价格版本' if category == 'llm' else '缺少价格版本'
    if event.get('state') != 'success':
        reason = '请求失败，用量及计费待核对'
    elif category == 'search' and provider == 'Tavily' and model == 'search':
        if usage.get('units') is not None:
            cost = usage['units'] * TAVILY_UNIT_PRICE_USD
        else:
            reason = '未返回 credits 用量'
    elif price and price.get('review_after', at) < at:
        reason = '价格已到复核期限'
    elif price:
        rates = price['rates']
        if category == 'llm':
            inp, out, cached = (usage.get(k) for k in ('input_tokens', 'output_tokens', 'cached_tokens'))
            observed = []
            if inp is not None:
                if cached is not None and 0 <= cached <= inp:
                    observed.extend([('input', inp-cached), ('cached_input', cached)])
                else:
                    observed.append(('input', inp))
                    event['usage_note'] = '缓存 Token 未返回或不一致；已将实际记录的输入 Token 按未缓存输入单价估算'
            elif cached is not None:
                observed.append(('cached_input', cached))
                event['usage_note'] = '总输入 Token 未返回；仅核算实际记录的缓存 Token'
            if out is not None:
                observed.append(('output', out))
            if observed:
                event['breakdown'] = {
                    key: {'tokens': count, 'rate': rates[key], 'amount': count*rates[key]/1_000_000}
                    for key, count in observed
                }
                cost = sum(item['amount'] for item in event['breakdown'].values())
            reason = '没有返回可核算的 Token 用量'
        elif usage.get('units') is not None:
            cost = usage['units'] * rates['unit']
    currency = price.get('currency', 'USD') if price else None
    event.update(amount=cost, currency=currency, cost_usd=cost if currency == 'USD' else None,
                 status='estimated' if cost is not None else 'pending', reason='' if cost is not None else reason)
    return event


def record(category, provider, model, usage, *, endpoint='', ticker=None,
           source='', usage_basis='reported', state='success', occurred_at=None, requested_model=None):
    """Best effort: accounting must not break a provider response or retry it."""
    try:
        at = timestamp(occurred_at) if occurred_at else now_iso()
        usage = dict(usage)
        for key in ('input_tokens', 'output_tokens', 'cached_tokens', 'units'):
            if key in usage and usage[key] is not None:
                value = float(usage[key])
                usage[key] = value if math.isfinite(value) and value >= 0 else None
        with _conn() as conn:
            init(conn)
            conn.execute('BEGIN IMMEDIATE')
            event = dict(id=uuid.uuid4().hex, occurred_at=at, category=category, provider=provider,
                         model=model, endpoint=endpoint, ticker=ticker, source=source, run_id=current_run() or None, usage=usage, channel=current_channel(),
                         usage_basis=usage_basis, state=state, requested_model=requested_model)
            quote_event(conn, event)
            conn.execute('INSERT INTO usage_events VALUES (?,?,?,?,?,?,?,?)',
                         (event['id'], at, category, provider, model, event['cost_usd'], event['status'], json.dumps(event)))
    except Exception:
        logger.warning('Usage accounting failed; provider response retained', exc_info=False)


def record_fd(path, params=None):
    endpoint = _PATH_TO_ENDPOINT.get(path, path.strip('/').replace('/', '_') or 'unknown')
    record('data', 'Financial Datasets', endpoint, {'units': 1}, endpoint=endpoint,
           ticker=str((params or {}).get('ticker') or '').upper() or None, source='FD HTTP response')


def record_llm(data, model, provider='DeepSeek', source=''):
    if not isinstance(data, dict):
        data = {}
    usage = data.get('usage') or {}
    if not isinstance(usage, dict):
        usage = {}
    cached = usage.get('prompt_cache_hit_tokens')
    if cached is None:
        details = usage.get('prompt_tokens_details') or {}
        cached = details.get('cached_tokens') if isinstance(details, dict) else None
    # Reasoning tokens are billed as output; kept apart so the report can show how much of the output is thinking.
    completion_details = usage.get('completion_tokens_details') or {}
    reasoning = completion_details.get('reasoning_tokens') if isinstance(completion_details, dict) else None
    if reasoning is None:
        reasoning = usage.get('reasoning_tokens')
    record('llm', provider, data.get('model') or model,
           dict(input_tokens=usage.get('prompt_tokens'), output_tokens=usage.get('completion_tokens'), cached_tokens=cached, **({'reasoning_tokens': reasoning} if reasoning is not None else {})),
           source=source, endpoint='chat', usage_basis='reported' if usage else 'unknown', requested_model=model)


def reconcile_pending():
    """Only reprice LLM pending rows with an applicable historic price; keep audit."""
    updated = 0
    with _conn() as conn:
        init(conn)
        conn.execute('BEGIN IMMEDIATE')
        rows = conn.execute("SELECT payload FROM usage_events WHERE status='pending' AND category='llm' ORDER BY occurred_at").fetchall()
        for row in rows:
            before = json.loads(row[0])
            after = quote_event(conn, json.loads(row[0]))
            if after['status'] != 'estimated':
                continue
            after['reconciled_at'] = now_iso()
            conn.execute('INSERT INTO billing_audit VALUES (?,?)', (uuid.uuid4().hex, json.dumps({'type': 'reconcile', 'before': before, 'after': after})))
            conn.execute('UPDATE usage_events SET payload=?, cost_usd=?, status=? WHERE id=?', (json.dumps(after), after['cost_usd'], after['status'], after['id']))
            updated += 1
    return {'updated': updated, 'message': f'已补算 {updated} 条 LLM 记录；未找到历史价格或完整用量的记录不变。'}


def reconcile_tavily_flat_rate():
    """Apply the owner's flat per-credit policy to all stored Tavily usage."""
    updated = 0
    with _conn() as conn:
        init(conn)
        conn.execute('BEGIN IMMEDIATE')
        rows = conn.execute(
            "SELECT payload FROM usage_events WHERE category='search' AND provider='Tavily' AND model='search' ORDER BY occurred_at"
        ).fetchall()
        for row in rows:
            before = json.loads(row[0])
            after = quote_event(conn, json.loads(row[0]))
            if after == before:
                continue
            after['reconciled_at'] = now_iso()
            conn.execute('INSERT INTO billing_audit VALUES (?,?)', (uuid.uuid4().hex, json.dumps({'type': 'tavily_flat_rate', 'before': before, 'after': after})))
            conn.execute('UPDATE usage_events SET payload=?, cost_usd=?, status=? WHERE id=?',
                         (json.dumps(after), after['cost_usd'], after['status'], after['id']))
            updated += 1
    return {'updated': updated, 'message': f'已按 $0.008/credit 统一核算 {updated} 条 Tavily 历史记录。'}


def _rollup_for_event(event):
    usage = event.get('usage') or {}
    currency = event['currency']
    cached = usage.get('cached_tokens')
    input_tokens = usage.get('input_tokens')
    valid_cache = input_tokens is not None and cached is not None and 0 <= cached <= input_tokens
    token_costs = {'input': 0., 'cached_input': 0., 'output': 0.}
    for key, item in (event.get('breakdown') or {}).items():
        if key in token_costs:
            token_costs[key] += item.get('amount') or 0
    return {
        'day': datetime.fromisoformat(event['occurred_at'].replace('Z', '+00:00')).astimezone(ET).date().isoformat(),
        'category': event['category'], 'provider': event['provider'], 'currency': currency,
        'requests': 1, 'amount': event['amount'],
        'input_tokens': input_tokens or 0, 'output_tokens': usage.get('output_tokens') or 0,
        'cached_tokens': cached if valid_cache else 0,
        'uncached_tokens': input_tokens-cached if valid_cache else 0,
        'unclassified_requests': 1 if event['category'] == 'llm' and not valid_cache else 0,
        'credits': (usage.get('units') or 0) if event['category'] == 'search' else 0,
        'token_costs': token_costs,
    }


def _merge_rollup(left, right):
    merged = dict(left or right)
    for key in ('requests', 'amount', 'input_tokens', 'output_tokens', 'cached_tokens', 'uncached_tokens', 'unclassified_requests', 'credits'):
        merged[key] = (left or {}).get(key, 0) + right.get(key, 0)
    merged['token_costs'] = {
        key: (left or {}).get('token_costs', {}).get(key, 0) + right.get('token_costs', {}).get(key, 0)
        for key in ('input', 'cached_input', 'output')
    }
    return merged


def rollup_and_prune(retention_hours=RECENT_DETAIL_HOURS):
    """Keep aggregate spend forever while irreversibly deleting old details and all pending rows."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=max(1, int(retention_hours)))
    rollups = {}
    usage_ids, legacy_ids, pending_deleted = [], [], 0
    with _conn() as conn:
        init(conn)
        conn.execute('BEGIN IMMEDIATE')
        for row in conn.execute('SELECT id, payload FROM usage_events'):
            event = json.loads(row['payload'])
            event.setdefault('amount', event.get('cost_usd'))
            event.setdefault('currency', (event.get('price') or {}).get('currency') or ('USD' if event['amount'] is not None else None))
            pending = event.get('status') == 'pending' or event.get('amount') is None or event.get('currency') not in ('CNY', 'USD')
            if pending:
                usage_ids.append((row['id'],))
                pending_deleted += 1
                continue
            occurred = datetime.fromisoformat(event['occurred_at'].replace('Z', '+00:00')).astimezone(timezone.utc)
            if occurred < cutoff:
                item = _rollup_for_event(event)
                key = (item['day'], item['category'], item['provider'], item['currency'])
                rollups[key] = _merge_rollup(rollups.get(key), item)
                usage_ids.append((row['id'],))
        for row in conn.execute('SELECT * FROM query_costs'):
            occurred = datetime.fromisoformat(row['occurred_at'].replace('Z', '+00:00')).astimezone(timezone.utc)
            if occurred >= cutoff:
                continue
            event = {**dict(row), 'category': 'data', 'model': row['endpoint'], 'status': 'estimated',
                     'amount': row['cost_usd'], 'currency': 'USD', 'usage': {'units': 1},
                     'usage_basis': 'legacy', 'state': 'success', 'source': '旧 FD 账本', 'price': None}
            item = _rollup_for_event(event)
            key = (item['day'], item['category'], item['provider'], item['currency'])
            rollups[key] = _merge_rollup(rollups.get(key), item)
            legacy_ids.append((row['id'],))
        for key, item in rollups.items():
            old = conn.execute('SELECT payload FROM usage_rollups WHERE day=? AND category=? AND provider=? AND currency=?', key).fetchone()
            merged = _merge_rollup(json.loads(old[0]) if old else None, item)
            conn.execute('INSERT OR REPLACE INTO usage_rollups VALUES (?,?,?,?,?)', (*key, json.dumps(merged)))
        conn.executemany('DELETE FROM usage_events WHERE id=?', usage_ids)
        conn.executemany('DELETE FROM query_costs WHERE id=?', legacy_ids)
    return {
        'cutoff': cutoff.isoformat(), 'rolled_up': len(usage_ids)-pending_deleted+len(legacy_ids),
        'pending_deleted': pending_deleted, 'details_deleted': len(usage_ids)+len(legacy_ids),
    }


def report(limit=100, recent_filter='all', offset=0, recent_hours=None):
    from v2.data.price_sync import sync_status
    # Old FD entries remain immutable and are included once. New entries live
    # in usage_events, allowing unknown cost to be NULL, never fictitious zero.
    with _conn() as conn:
        init(conn)
        events = [json.loads(r[0]) for r in conn.execute('SELECT payload FROM usage_events')]
        rollups = [json.loads(r[0]) for r in conn.execute('SELECT payload FROM usage_rollups')]
        for row in conn.execute('SELECT * FROM query_costs'):
            events.append({**dict(row), 'category': 'data', 'model': row['endpoint'], 'status': 'estimated',
                           'usage': {'units': 1}, 'usage_basis': 'legacy', 'state': 'success',
                           'source': '旧 FD 账本', 'price': None, 'reason': ''})
    today = datetime.now(ET).date().isoformat()
    totals = dict(today_cost_usd=0., month_cost_usd=0., total_cost_usd=0., total_requests=len(events)+sum(row['requests'] for row in rollups), pending_requests=0)
    currencies = {code: dict(currency=code, today_amount=0., month_amount=0., total_amount=0.) for code in ('CNY', 'USD')}
    groups = {}
    pending_reasons = {}
    for event in events:
        event.setdefault('channel', 'unknown')  # Never infer the source of old records.
        event.setdefault('amount', event.get('cost_usd'))
        event.setdefault('currency', (event.get('price') or {}).get('currency') or ('USD' if event['amount'] is not None else None))
        day = datetime.fromisoformat(event['occurred_at']).astimezone(ET).date().isoformat()
        group = groups.setdefault((event['category'], event['provider']), dict(category=event['category'], provider=event['provider'], cost_usd=0., amounts={'CNY': 0., 'USD': 0.}, requests=0, pending=0, input_tokens=0, output_tokens=0, cached_tokens=0, uncached_tokens=0, unclassified_requests=0, credits=0, free_credits=0, paid_credits=0, token_costs={'CNY': {'input': 0., 'cached_input': 0., 'output': 0.}, 'USD': {'input': 0., 'cached_input': 0., 'output': 0.}}))
        group['requests'] += 1
        u = event['usage']
        group['input_tokens'] += u.get('input_tokens') or 0
        group['output_tokens'] += u.get('output_tokens') or 0
        if event['category'] == 'llm':
            if u.get('input_tokens') is not None and u.get('cached_tokens') is not None and 0 <= u['cached_tokens'] <= u['input_tokens']:
                group['cached_tokens'] += u['cached_tokens']
                group['uncached_tokens'] += u['input_tokens']-u['cached_tokens']
            else:
                group['unclassified_requests'] += 1
            for key, item in (event.get('breakdown') or {}).items():
                if event['currency'] in group['token_costs']:
                    group['token_costs'][event['currency']][key] += item['amount']
        group['free_credits'] += (event.get('quota') or {}).get('free_credits', 0)
        group['paid_credits'] += (event.get('quota') or {}).get('paid_credits', 0)
        if event['category'] == 'search':
            group['credits'] += u.get('units') or 0
        if event['amount'] is None or event['currency'] not in currencies:
            totals['pending_requests'] += 1
            group['pending'] += 1
            reason = event.get('reason') or '历史记录缺少核算依据'
            pending_reasons[reason] = pending_reasons.get(reason, 0)+1
            continue
        cost = event['amount']
        currency = event['currency']
        bucket = currencies[currency]
        group['amounts'][currency] += cost
        bucket['total_amount'] += cost
        if day == today:
            bucket['today_amount'] += cost
        if day[:7] == today[:7]:
            bucket['month_amount'] += cost
        if currency == 'USD':
            group['cost_usd'] += cost
    for item in rollups:
        group = groups.setdefault((item['category'], item['provider']), dict(category=item['category'], provider=item['provider'], cost_usd=0., amounts={'CNY': 0., 'USD': 0.}, requests=0, pending=0, input_tokens=0, output_tokens=0, cached_tokens=0, uncached_tokens=0, unclassified_requests=0, credits=0, free_credits=0, paid_credits=0, token_costs={'CNY': {'input': 0., 'cached_input': 0., 'output': 0.}, 'USD': {'input': 0., 'cached_input': 0., 'output': 0.}}))
        group['requests'] += item['requests']
        for key in ('input_tokens', 'output_tokens', 'cached_tokens', 'uncached_tokens', 'unclassified_requests', 'credits'):
            group[key] += item.get(key, 0)
        currency, amount = item['currency'], item['amount']
        group['amounts'][currency] += amount
        if currency == 'USD':
            group['cost_usd'] += amount
        for key, value in item.get('token_costs', {}).items():
            group['token_costs'][currency][key] += value
        bucket = currencies[currency]
        bucket['total_amount'] += amount
        if item['day'] == today:
            bucket['today_amount'] += amount
        if item['day'][:7] == today[:7]:
            bucket['month_amount'] += amount
    # Compatibility fields are USD-only, never a cross-currency total.
    for name in ('today', 'month', 'total'):
        totals[name + '_cost_usd'] = currencies['USD'][name + '_amount']
    detail_hours = max(1, int(recent_hours)) if recent_hours is not None else None
    detail_cutoff = datetime.now(timezone.utc) - timedelta(hours=detail_hours) if detail_hours is not None else None
    detail_events = events if detail_cutoff is None else [
        event for event in events
        if datetime.fromisoformat(event['occurred_at']).astimezone(timezone.utc) >= detail_cutoff
    ]
    if recent_filter == 'pending':
        # Match the aggregate definition above so the count and the detail
        # list can never disagree because of a stale legacy status field.
        filtered_events = [event for event in detail_events if event.get('amount') is None or event.get('currency') not in currencies]
    elif recent_filter in {'data', 'llm', 'search'}:
        filtered_events = [event for event in detail_events if event.get('category') == recent_filter]
    else:
        filtered_events = detail_events
        recent_filter = 'all'
    filtered_events.sort(key=lambda event: event['occurred_at'], reverse=True)
    page_limit = max(1, min(int(limit), 500))
    page_offset = max(0, int(offset))
    recent_total = len(filtered_events)
    recent = filtered_events[page_offset:page_offset + page_limit]
    return {**totals, 'currencies': list(currencies.values()), 'timezone': 'America/New_York', 'basis': 'usage_estimates',
            'by_provider': list(groups.values()), 'prices': prices(), 'price_sync': sync_status(), 'pending_reasons': pending_reasons,
            'recent': recent, 'recent_filter': recent_filter, 'recent_total': recent_total,
            'recent_offset': page_offset, 'recent_limit': page_limit,
            'recent_has_more': page_offset + len(recent) < recent_total,
            'recent_detail_hours': detail_hours, 'recent_cutoff': detail_cutoff.isoformat() if detail_cutoff else None}
