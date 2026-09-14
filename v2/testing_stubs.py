"""Sandbox stand-ins for the git-ignored ``v2.data`` package (test support only).

``v2/data/`` (FDClient, CachedFDClient, models, yfinance client) exists only on
the production VPS. :func:`install_data_stubs` registers minimal modules under
the same names when the real package is absent, so tests that only exercise
logic around the data layer can be collected and run in a bare checkout. It is
a no-op when the real package imports.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_DATA_DIR = _REPO / "v2" / "data"


def install_data_stubs() -> bool:
    """Return True when stubs were installed (real package missing)."""
    try:  # real production package present → do nothing
        import v2.data.client  # noqa: F401
        return False
    except ImportError:
        pass
    if True:
        class _Unavailable(RuntimeError):
            pass

        class FDClient:  # noqa: D401 — stub
            """Stand-in for the production Financial Datasets client."""

            def __init__(self, *args, **kwargs) -> None:
                pass

            def __enter__(self):
                return self

            def __exit__(self, *exc) -> bool:
                return False

            def __getattr__(self, name):
                raise _Unavailable(f"v2.data.FDClient.{name} is production-only; monkeypatch the caller")

        class CachedFDClient(FDClient):
            pass

        class ProviderRequestError(RuntimeError):
            pass

        @dataclass
        class Price:
            open: float
            close: float
            high: float
            low: float
            volume: int
            time: str

        class _Flexible:
            """Accepts any fields (the real models are pydantic); unknown reads are None."""

            def __init__(self, **fields) -> None:
                self.__dict__.update(fields)

            def __getattr__(self, name):
                if name.startswith("__"):
                    raise AttributeError(name)
                return None

        class EarningsRecord(_Flexible):
            pass

        class EarningsData(_Flexible):
            pass

        class NewsProvider:  # noqa: D401 — stub
            def get_news(self, *args, **kwargs):
                return []

        def default_news_provider() -> NewsProvider:
            return NewsProvider()

        class YFinanceClient(FDClient):
            pass

        KNOWN_ADRS: set[str] = set()

        pkg = types.ModuleType("v2.data")
        pkg.__path__ = [str(_DATA_DIR)]  # keep the real price_source.py importable as a submodule
        pkg.FDClient, pkg.CachedFDClient = FDClient, CachedFDClient
        pkg.Price, pkg.EarningsRecord, pkg.EarningsData = Price, EarningsRecord, EarningsData

        client = types.ModuleType("v2.data.client")
        client.FDClient, client.ProviderRequestError = FDClient, ProviderRequestError
        models = types.ModuleType("v2.data.models")
        models.Price, models.EarningsRecord, models.EarningsData = Price, EarningsRecord, EarningsData
        news = types.ModuleType("v2.data.news_provider")
        news.NewsProvider, news.default_news_provider = NewsProvider, default_news_provider
        yf = types.ModuleType("v2.data.yfinance_client")
        yf.YFinanceClient, yf.KNOWN_ADRS = YFinanceClient, KNOWN_ADRS

        for name, module in (("v2.data", pkg), ("v2.data.client", client), ("v2.data.models", models),
                             ("v2.data.news_provider", news), ("v2.data.yfinance_client", yf)):
            sys.modules[name] = module
        pkg.client, pkg.models, pkg.news_provider, pkg.yfinance_client = client, models, news, yf
    return True
