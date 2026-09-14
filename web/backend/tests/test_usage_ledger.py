from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from v2.data import usage_ledger as ledger
from v2.data.metered import LLMProxy, SearchProxy


def price(provider='DeepSeek', model='m', **overrides):
    return ledger.add_price(dict(provider=provider, model=model,
        effective_at='2026-01-01T00:00:00+00:00', review_after='2099-01-01T00:00:00+00:00',
        source='test verified rate', rates={'input': 1, 'cached_input': .1, 'output': 2} if provider == 'DeepSeek' else {'unit': .008}, **overrides))


def test_version_snapshot_and_cache_tokens():
    old = price()
    ledger.record_llm({'usage': {'prompt_tokens': 1000, 'completion_tokens': 100, 'prompt_cache_hit_tokens': 500}}, 'm')
    event = ledger.report()['recent'][0]
    assert event['cost_usd'] == pytest.approx(.00075)
    assert event['price']['id'] == old['id']
    ledger.add_price({**old, 'effective_at': '2026-02-01T00:00:00+00:00', 'rates': {'input': 9, 'cached_input': 9, 'output': 9}})
    assert ledger.report()['recent'][0]['cost_usd'] == event['cost_usd']


def test_llm_charges_only_observed_tokens_when_cache_split_is_missing():
    ledger.record_llm({'usage': {'prompt_tokens': 100, 'completion_tokens': 20}}, 'missing')
    price()
    ledger.record_llm({'usage': {'prompt_tokens': 100, 'completion_tokens': 20}}, 'm')
    report = ledger.report()
    assert report['pending_requests'] == 1
    priced = next(event for event in report['recent'] if event['status'] == 'estimated')
    assert priced['amount'] == pytest.approx(.00014)
    assert set(priced['breakdown']) == {'input', 'output'}
    assert '实际记录的输入 Token' in priced['usage_note']


def test_unknown_deepseek_model_uses_flash_fallback_with_observed_usage():
    price(model='deepseek-flash')
    ledger.record(
        'llm', 'DeepSeek', 'unconfirmed-response-name',
        {'input_tokens': 100, 'cached_tokens': 0, 'output_tokens': 10},
        requested_model='unconfirmed-request-name',
    )
    event = ledger.report()['recent'][0]
    assert event['status'] == 'estimated'
    assert event['pricing_model'] == 'deepseek-flash'
    assert event['pricing_basis'] == 'deepseek_flash_fallback'
    assert event['amount'] == pytest.approx(.00012)


def test_llm_missing_token_components_are_not_invented():
    price()
    ledger.record('llm', 'DeepSeek', 'm', {'output_tokens': 20})
    event = ledger.report()['recent'][0]
    assert event['amount'] == pytest.approx(.00004)
    assert set(event['breakdown']) == {'output'}


def test_expired_and_future_prices():
    p = price()
    ledger.add_price({**p, 'effective_at': '2026-02-01T00:00:00+00:00', 'review_after': '2026-03-01T00:00:00+00:00'})
    ledger.record('llm', 'DeepSeek', 'm', {'input_tokens': 10, 'output_tokens': 2, 'cached_tokens': 0}, occurred_at='2026-04-01T00:00:00+00:00')
    assert ledger.report()['recent'][0]['reason'] == '价格已到复核期限'
    ledger.record('llm', 'DeepSeek', 'm', {}, occurred_at='2025-01-01T00:00:00+00:00')
    assert ledger.report()['pending_requests'] == 2


def test_legacy_preserved_and_fd_not_duplicated(monkeypatch):
    from v2.data.cost_ledger import record_fd_request
    monkeypatch.setenv('FD_PRICES', '{"news":0.03}')
    record_fd_request('/news/', {'ticker': 'MU'})
    ledger.record_fd('/news/', {'ticker': 'MU'})
    report = ledger.report()
    assert report['total_requests'] == 2
    assert report['total_cost_usd'] == pytest.approx(.06)
    monkeypatch.setenv('FD_PRICES', '{"news":99}')
    assert ledger.report()['total_cost_usd'] == pytest.approx(.06)


def test_search_actual_usage_and_failure():
    price('Tavily', 'search')
    def search(**kwargs):
        assert kwargs['include_usage'] is True
        return {'usage': {'credits': 2}, 'auto_parameters': {'search_depth': 'advanced'}}
    SearchProxy(SimpleNamespace(search=search), 'test').search(query='private query', auto_parameters=True)
    event = ledger.report()['recent'][0]
    assert event['cost_usd'] == .016 and event['usage']['units'] == 2
    assert 'private query' not in str(event)
    def fail(**kwargs):
        raise RuntimeError('secret key')
    with pytest.raises(RuntimeError):
        SearchProxy(SimpleNamespace(search=fail), 'test').search(query='x')
    assert ledger.report()['pending_requests'] == 1
    assert 'secret key' not in str(ledger.report())


def test_llm_records_before_business_parser_and_without_trace():
    price()
    response = SimpleNamespace(content='not JSON', usage_metadata={'input_tokens': 100, 'output_tokens': 10, 'input_token_details': {'cache_read': 0}}, response_metadata={'model_name': 'm'})
    proxy = LLMProxy(SimpleNamespace(invoke=lambda *a, **kw: response), 'm', 'test')
    assert proxy.invoke('private prompt') is response
    report = ledger.report()
    assert report['total_requests'] == 1 and report['pending_requests'] == 0
    assert 'private prompt' not in str(report)


@pytest.mark.parametrize('value', [-1, float('inf'), float('nan')])
def test_invalid_price_rejected(value):
    p = price()
    with pytest.raises(ValueError):
        ledger.add_price({**p, 'rates': {'input': value, 'cached_input': 0, 'output': 1}})


def test_balance_redacts_and_caches(monkeypatch):
    from v2.data import provider_balance as balance
    monkeypatch.setattr(balance, '_cache', None)
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'secret')
    calls = []
    def get(url, **kwargs):
        calls.append(url)
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {'balance_infos': [{'currency': 'USD', 'total_balance': '3.05', 'granted_balance': '0', 'topped_up_balance': '3.05'}]})
    monkeypatch.setattr(balance.requests, 'get', get)
    assert balance.deepseek_balance()['balances'][0]['total_balance'] == '3.05'
    assert 'secret' not in str(balance.deepseek_balance())
    assert len(calls) == 1
    assert ledger.report()['total_requests'] == 0


def test_cost_api_routes_and_validation(monkeypatch):
    from fastapi.testclient import TestClient
    from app.main import app
    from v2.data import price_sync
    monkeypatch.setattr(price_sync, 'sync_prices', lambda force=False: {'status': 'ok'})
    client = TestClient(app)
    response = client.get('/api/costs')
    assert response.status_code == 200
    assert response.json()['recent_detail_hours'] == 24
    assert client.get('/api/costs?filter=pending').status_code == 200
    assert client.get('/api/costs?filter=invalid').status_code == 422
    response = client.post('/api/costs/refresh')
    assert response.status_code == 200
    assert response.json()['refresh_result']['retention']['pending_deleted'] == 0
    assert client.post('/api/costs/prices', json={}).status_code == 400


def test_recent_filter_is_applied_before_limit_and_supports_pagination():
    price()
    now = datetime.now(timezone.utc)
    pending_times = [(now-timedelta(minutes=value)).isoformat() for value in (20, 10)]
    ledger.record('llm', 'DeepSeek', 'missing', {}, occurred_at=pending_times[0])
    ledger.record('llm', 'DeepSeek', 'missing', {}, occurred_at=pending_times[1])
    for value in range(5):
        ledger.record('llm', 'DeepSeek', 'm', {'input_tokens': 10, 'cached_tokens': 0, 'output_tokens': 2}, occurred_at=(now-timedelta(minutes=value)).isoformat())

    assert all(event['status'] != 'pending' for event in ledger.report(limit=2)['recent'])
    first = ledger.report(limit=1, recent_filter='pending')
    second = ledger.report(limit=1, recent_filter='pending', offset=1)
    assert first['recent_total'] == 2 and first['recent_has_more'] is True
    assert first['recent'][0]['occurred_at'] == pending_times[1]
    assert second['recent'][0]['occurred_at'] == pending_times[0]
    assert second['recent_has_more'] is False


def test_recent_details_are_limited_to_24_hours_but_totals_are_retained():
    price()
    now = datetime.now(timezone.utc)
    ledger.record('llm', 'DeepSeek', 'm', {'input_tokens': 100, 'cached_tokens': 0, 'output_tokens': 10}, occurred_at=(now-timedelta(days=3)).isoformat())
    ledger.record('llm', 'DeepSeek', 'm', {'input_tokens': 100, 'cached_tokens': 0, 'output_tokens': 10}, occurred_at=(now-timedelta(hours=1)).isoformat())
    result = ledger.report(recent_hours=24)
    assert result['total_requests'] == 2
    assert result['recent_total'] == len(result['recent']) == 1
    assert result['recent_detail_hours'] == 24


def test_rollup_prunes_old_details_and_all_pending_without_losing_resolved_totals():
    price()
    now = datetime.now(timezone.utc)
    ledger.record('llm', 'DeepSeek', 'm', {'input_tokens': 100, 'cached_tokens': 0, 'output_tokens': 10}, occurred_at=(now-timedelta(days=2)).isoformat())
    ledger.record('llm', 'DeepSeek', 'm', {'input_tokens': 100, 'cached_tokens': 0, 'output_tokens': 10}, occurred_at=(now-timedelta(hours=1)).isoformat())
    ledger.record('llm', 'DeepSeek', 'missing', {}, occurred_at=(now-timedelta(minutes=30)).isoformat())

    result = ledger.rollup_and_prune(24)
    assert result['rolled_up'] == 1
    assert result['pending_deleted'] == 1
    assert result['details_deleted'] == 2
    with ledger._conn() as conn:
        assert conn.execute('SELECT count(*) FROM usage_events').fetchone()[0] == 1
        assert conn.execute('SELECT count(*) FROM usage_rollups').fetchone()[0] == 1

    report = ledger.report(recent_hours=24)
    assert report['total_requests'] == 2
    assert report['pending_requests'] == 0
    assert report['recent_total'] == 1
    assert report['currencies'][1]['total_amount'] == pytest.approx(.00024)
    assert ledger.rollup_and_prune(24)['details_deleted'] == 0


def test_report_uses_eastern_calendar(monkeypatch):
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 9, 9, 16, tzinfo=timezone.utc).astimezone(tz)
    monkeypatch.setattr(ledger, 'datetime', Clock)
    for at in ('2026-09-09T02:00:00+00:00', '2026-09-09T06:00:00+00:00'):
        ledger.record('data', 'Financial Datasets', 'news', {'units': 1}, endpoint='news', occurred_at=at)
    assert ledger.report()['total_cost_usd'] == pytest.approx(.04)
    assert ledger.report()['today_cost_usd'] == pytest.approx(.02)


def test_agent_paid_response_with_invalid_choices_still_counted(monkeypatch):
    import json
    from v2.agent_common.llm import OpenAICompatLLM, LLMError
    price()
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self): return json.dumps({'usage': {'prompt_tokens': 100, 'completion_tokens': 20, 'prompt_cache_hit_tokens': 0}, 'choices': []}).encode()
    monkeypatch.setattr('urllib.request.urlopen', lambda *args, **kwargs: Response())
    with pytest.raises(LLMError):
        OpenAICompatLLM(model='m', api_key='test', max_retries=1).complete([])
    report = ledger.report()
    assert report['total_requests'] == 1
    assert report['pending_requests'] == 0


def test_cost_endpoints_require_owner_when_configured(monkeypatch):
    from fastapi.testclient import TestClient
    from app.main import app
    from app import auth
    monkeypatch.setattr(auth, 'SETTINGS', SimpleNamespace(owner_token='owner-test'))
    client = TestClient(app)
    assert client.get('/api/costs').status_code == 401
    assert client.get('/api/costs/deepseek-balance').status_code == 401
    assert client.post('/api/costs/prices', json={}).status_code == 401
    assert client.post('/api/costs/prices/sync').status_code == 401
    assert client.post('/api/costs/refresh').status_code == 401
    assert client.get('/api/costs', headers={'X-Owner-Token': 'owner-test'}).status_code == 200


def test_cny_and_usd_are_never_added_or_converted():
    p = price()
    ledger.add_price({**p, 'currency': 'CNY', 'effective_at': '2026-02-01T00:00:00+00:00'})
    ledger.record('llm', 'DeepSeek', 'm', {'input_tokens': 1_000_000, 'cached_tokens': 0, 'output_tokens': 0}, occurred_at='2026-01-15T00:00:00+00:00')
    ledger.record('llm', 'DeepSeek', 'm', {'input_tokens': 2_000_000, 'cached_tokens': 0, 'output_tokens': 0}, occurred_at='2026-02-15T00:00:00+00:00')
    result = ledger.report()
    assert {r['currency']: r['total_amount'] for r in result['currencies']} == {'CNY': 2., 'USD': 1.}
    assert result['by_provider'][0]['amounts'] == {'CNY': 2., 'USD': 1.}
    assert result['total_cost_usd'] == 1.
    cny = next(e for e in result['recent'] if e['currency'] == 'CNY')
    assert cny['amount'] == 2. and cny['cost_usd'] is None
    assert result['pending_requests'] == 0
    with ledger._conn() as conn:
        assert conn.execute('SELECT cost_usd FROM usage_events WHERE id=?', (cny['id'],)).fetchone()[0] is None


def test_legacy_usd_and_unknown_currency():
    from v2.data.cost_ledger import record_fd_request
    record_fd_request('/news/')
    ledger.record_llm({}, 'unconfigured')
    result = ledger.report()
    legacy = next(e for e in result['recent'] if e['usage_basis'] == 'legacy')
    pending = next(e for e in result['recent'] if e['status'] == 'pending')
    assert legacy['currency'] == 'USD' and legacy['amount'] == legacy['cost_usd']
    assert pending['currency'] is None and pending['amount'] is None
    assert result['currencies'][0]['total_amount'] == 0


def test_currency_validation_and_price_snapshot():
    p = price()
    with pytest.raises(ValueError):
        ledger.add_price({**p, 'currency': 'EUR'})
    ledger.record_llm({'usage': {'prompt_tokens': 100, 'completion_tokens': 10, 'prompt_cache_hit_tokens': 0}}, 'm')
    ledger.add_price({**p, 'currency': 'CNY', 'effective_at': '2026-02-01T00:00:00+00:00'})
    assert ledger.report()['recent'][0]['currency'] == 'USD'
