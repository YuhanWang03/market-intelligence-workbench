"""Explicit provider adapters, independent of optional observability tracing."""
import sys
from v2.data.usage_ledger import record, record_llm


class LLMProxy:
    def __init__(self, client, model, source):
        self.client, self.model, self.source = client, model, source

    def __getattr__(self, name):
        return getattr(self.client, name)

    def account(self, result):
        try:
            self._account(result)
        except Exception:
            # Never retry a paid completion because usage metadata is malformed.
            record('llm', 'DeepSeek', self.model, {}, source=self.source, usage_basis='unknown')

    def _account(self, result):
        meta = getattr(result, 'response_metadata', None) or {}
        usage = meta.get('token_usage') or meta.get('usage') or {}
        standard = getattr(result, 'usage_metadata', None) or {}
        if standard:
            for incoming, outgoing in [('input_tokens', 'prompt_tokens'), ('output_tokens', 'completion_tokens')]:
                if standard.get(incoming) is not None:
                    usage[outgoing] = standard[incoming]
            cached = (standard.get('input_token_details') or {}).get('cache_read')
            if cached is not None:
                usage['prompt_cache_hit_tokens'] = cached
        record_llm({'usage': usage, 'model': meta.get('model_name') or meta.get('model')}, self.model, source=self.source)

    def invoke(self, *args, **kwargs):
        try:
            result = self.client.invoke(*args, **kwargs)
        except Exception:
            record('llm', 'DeepSeek', self.model, {}, source=self.source, state='failed', usage_basis='unknown')
            raise
        self.account(result)
        return result

    async def ainvoke(self, *args, **kwargs):
        try:
            result = await self.client.ainvoke(*args, **kwargs)
        except Exception:
            record('llm', 'DeepSeek', self.model, {}, source=self.source, state='failed', usage_basis='unknown')
            raise
        self.account(result)
        return result


def ChatDeepSeek(*args, **kwargs):
    from langchain_deepseek import ChatDeepSeek as Client
    return LLMProxy(Client(*args, **kwargs), kwargs.get('model', 'deepseek-chat'), sys._getframe(1).f_globals.get('__name__', ''))


class SearchProxy:
    def __init__(self, client, source):
        self.client, self.source = client, source

    def __getattr__(self, name):
        return getattr(self.client, name)

    def search(self, *args, **kwargs):
        kwargs['include_usage'] = True
        try:
            result = self.client.search(*args, **kwargs)
        except Exception:
            record('search', 'Tavily', 'search', {}, source=self.source, state='failed', usage_basis='unknown')
            raise
        credits = (result.get('usage') or {}).get('credits')
        # Do not infer credits from requested depth: auto_parameters can change it.
        record('search', 'Tavily', 'search', {'units': credits,
               'search_depth': (result.get('auto_parameters') or {}).get('search_depth') or kwargs.get('search_depth', 'basic')},
               source=self.source, endpoint='search', usage_basis='reported' if credits is not None else 'unknown')
        return result


def TavilyClient(*args, **kwargs):
    from tavily import TavilyClient as Client
    return SearchProxy(Client(*args, **kwargs), sys._getframe(1).f_globals.get('__name__', ''))
