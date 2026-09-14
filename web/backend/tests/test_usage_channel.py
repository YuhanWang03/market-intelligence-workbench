import asyncio
from types import SimpleNamespace

from v2.data.usage_ledger import record_fd, report
from v2.usage_context import ContextExecutor, ContextThread, current_channel, usage_channel


def test_concurrent_channels_nested_workers_and_reset(monkeypatch):
    monkeypatch.delenv('USAGE_CHANNEL', raising=False)
    def work():
        with ContextExecutor(max_workers=1) as nested:
            nested.submit(record_fd, '/news/').result()
        return current_channel()
    with ContextExecutor(max_workers=2) as pool:
        with usage_channel('web'):
            web = pool.submit(work)
        with usage_channel('telegram'):
            telegram = pool.submit(work)
        assert web.result() == 'web'
        assert telegram.result() == 'telegram'
        assert pool.submit(current_channel).result() == 'unknown'
    events = report()['recent']
    assert {e['channel'] for e in events} == {'web', 'telegram'}
    assert report()['total_requests'] == 2
    assert current_channel() == 'unknown'


def test_detached_thread_keeps_origin_after_request_ends():
    with usage_channel('web'):
        thread = ContextThread(target=record_fd, args=('/news/',))
    thread.start()
    thread.join(timeout=10)
    assert not thread.is_alive()
    assert report()['recent'][0]['channel'] == 'web'


def test_http_boundary_tags_threadpool_and_overrides_fallback(monkeypatch):
    from fastapi import FastAPI
    from fastapi.concurrency import run_in_threadpool
    from fastapi.testclient import TestClient
    from app.main import billing_channel
    monkeypatch.setenv('USAGE_CHANNEL', 'background')
    app = FastAPI()
    app.middleware('http')(billing_channel)
    @app.get('/record')
    async def route():
        await run_in_threadpool(record_fd, '/news/')
        return {}
    assert TestClient(app).get('/record').status_code == 200
    assert report()['recent'][0]['channel'] == 'web'
    assert current_channel() == 'background'


def test_telegram_authorized_boundary_and_executor(monkeypatch):
    from v2.bot import commands
    monkeypatch.setattr(commands, '_allowed_chat_id', lambda: 123)
    @commands.authorized_only
    async def handler(update, context):
        # The production async helper must copy ContextVars into the executor.
        await commands._run_blocking(record_fd, '/news/')
    asyncio.run(handler(SimpleNamespace(effective_chat=SimpleNamespace(id=123)), None))
    assert report()['recent'][0]['channel'] == 'telegram'


def test_legacy_origin_not_inferred():
    from v2.data.cost_ledger import record_fd_request
    with usage_channel('web'):
        record_fd_request('/news/')
    assert report()['recent'][0]['channel'] == 'unknown'
