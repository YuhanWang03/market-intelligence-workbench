import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from v2.data import price_sync as sync, usage_ledger as ledger


def page():
    rows = ['<tr><td colspan="3">模型</td><td>deepseek-test</td></tr>']
    for i, label in enumerate(['百万tokens输入（缓存命中）', '百万tokens输入（缓存未命中）', '百万tokens输出']):
        rows.append(f'<tr>{"<td rowspan=6>价格</td>" if i == 0 else ""}<td rowspan=2>{label}</td><td>空闲时段</td><td>1元</td></tr>')
        rows.append('<tr><td>高峰时段</td><td>2元</td></tr>')
    return '<table>' + ''.join(rows) + '</table>高峰时段为北京时间周一至周五9:00-12:00、14:00-18:00（其余为空闲时段）'


def test_parse_and_reject():
    result = sync.parse_prices(page())[0]
    assert result['currency'] == 'CNY'
    assert result['schedule']['windows'] == [[540, 720], [840, 1080]]
    for bad in [page().replace('元', '美元'), page().replace('周一至周五', '每天'), page().replace('2元', '-2元')]:
        with pytest.raises(ValueError):
            sync.parse_prices(bad)


def test_parse_current_docusaurus_model_footnotes():
    current = page().replace('deepseek-test', 'deepseek-flash <sup>(1)</sup>')
    result = sync.parse_prices(current)[0]
    assert result['model'] == 'deepseek-flash'
    assert result['schedule']['timezone'] == 'Asia/Shanghai'


def test_sync_and_failure_preserve_prices(monkeypatch):
    calls = []
    def get(*args, **kwargs):
        calls.append(args)
        assert kwargs['headers'] == sync.REQUEST_HEADERS
        return SimpleNamespace(status_code=200, content=page().encode())
    monkeypatch.setattr(sync.requests, 'get', get)
    assert sync.sync_prices()['status'] == 'ok'
    sync.sync_prices(force=True)
    assert len(calls) == 1
    prices = ledger.prices()
    assert len(prices) == 1 and 'page_snapshot' not in prices[0]
    with ledger._conn() as conn:
        conn.execute('DELETE FROM price_sync_state')
    monkeypatch.setattr(sync.requests, 'get', lambda *a, **k: SimpleNamespace(status_code=500))
    assert sync.sync_prices()['status'] == 'error'
    assert ledger.prices() == prices


def test_successful_sync_bridges_gap_and_reconciles_flash_family(monkeypatch):
    now = datetime.now(timezone.utc)
    effective = (now - timedelta(days=10)).isoformat()
    expired = (now - timedelta(days=2)).isoformat()
    event_at = (now - timedelta(days=1)).isoformat()
    old = ledger.add_price(dict(
        provider='DeepSeek', model='deepseek-v4-flash', currency='CNY',
        effective_at=effective, review_after=expired, source='previous official snapshot',
        rates={'input': 1, 'cached_input': .1, 'output': 2},
    ))
    ledger.record(
        'llm', 'DeepSeek', 'deepseek-flash',
        {'input_tokens': 1_000_000, 'cached_tokens': 0, 'output_tokens': 0},
        occurred_at=event_at, requested_model='deepseek-chat',
    )
    assert ledger.report()['recent'][0]['status'] == 'pending'

    current = page().replace('deepseek-test', 'deepseek-flash <sup>(1)</sup>')
    monkeypatch.setattr(sync.requests, 'get', lambda *a, **k: SimpleNamespace(status_code=200, content=current.encode()))
    result = sync.sync_prices(force=True)
    assert result['status'] == 'ok' and result['extended_versions'] == 1
    assert ledger.reconcile_pending()['updated'] == 1
    event = ledger.report()['recent'][0]
    assert event['amount'] == 1
    assert event['pricing_model'] == 'deepseek-v4-flash'
    assert event['pricing_basis'] == 'official_model_family'
    previous = next(item for item in ledger.prices() if item['id'] == old['id'])
    assert previous['review_after'] == result['synced_at']
    assert previous['continuity_basis']
    current_price = next(item for item in ledger.prices() if item['model'] == 'deepseek-flash')
    assert datetime.fromisoformat(current_price['review_after']) - datetime.fromisoformat(current_price['effective_at']) == timedelta(days=7)


@pytest.mark.parametrize('at,rate', [('2026-09-07T00:59:00+00:00', 1), ('2026-09-07T01:00:00+00:00', 2), ('2026-09-07T04:00:00+00:00', 1), ('2026-09-07T06:00:00+00:00', 2), ('2026-09-07T10:00:00+00:00', 1), ('2026-09-06T01:00:00+00:00', 1)])
def test_schedule_boundaries(at, rate):
    p = {**sync.parse_prices(page())[0], 'id': 'test', 'effective_at': '2026-01-01T00:00:00+00:00', 'review_after': '2099-01-01T00:00:00+00:00', 'page_snapshot': 'large html'}
    with ledger._conn() as conn:
        ledger.init(conn)
        conn.execute('INSERT INTO usage_prices VALUES (?,?,?,?,?,?,?)', ('test', 'DeepSeek', p['model'], p['effective_at'], p['review_after'], p['effective_at'], json.dumps(p)))
    ledger.record('llm', 'DeepSeek', p['model'], {'input_tokens': 1000000, 'cached_tokens': 0, 'output_tokens': 0}, occurred_at=at)
    event = ledger.report()['recent'][0]
    assert event['amount'] == rate and event['currency'] == 'CNY'
    assert 'page_snapshot' not in event['price']
