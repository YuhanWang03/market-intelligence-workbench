"""Secret-safe provider health probes and error semantics."""

from __future__ import annotations

import os
import threading
import time
from datetime import datetime, timezone
from typing import Any

import requests


PROVIDERS = {
    "Financial Datasets": "FINANCIAL_DATASETS_API_KEY",
    "SEC": None,
    "Tavily": "TAVILY_API_KEY",
    "DeepSeek": "DEEPSEEK_API_KEY",
    "FRED": "FRED_API_KEY",
    "Yahoo Finance": None,
}


def classify_provider_error(error: Any = None, status_code: int | None = None, *, empty: bool = False) -> str:
    if empty:
        return "EMPTY_DATA"
    if status_code in {401, 403}:
        return "AUTH_ERROR"
    if status_code == 429:
        return "RATE_LIMITED"
    if isinstance(error, (requests.Timeout, TimeoutError)):
        return "TIMEOUT"
    if isinstance(error, requests.ConnectionError):
        return "UNREACHABLE"
    return "DEGRADED"


class ProviderHealthService:
    _lock = threading.Lock()
    _cached_at = 0.0
    _cached: list[dict] = []

    def __init__(self, timeout: float = 12.0) -> None:
        self.timeout = timeout

    @staticmethod
    def _configured(provider: str) -> bool:
        variable = PROVIDERS[provider]
        return True if variable is None else bool(os.environ.get(variable, "").strip())

    @staticmethod
    def _row(provider: str, configured: bool, reachable: bool, authenticated: bool, status: str, error_type: str | None = None) -> dict:
        return {"provider": provider, "configured": configured, "reachable": reachable, "authenticated": authenticated,
                "status": status, "last_checked_at": datetime.now(timezone.utc).isoformat(), "last_error_type": error_type}

    def _request(self, provider: str, method: str, url: str, **kwargs: Any) -> dict:
        configured = self._configured(provider)
        if not configured:
            return self._row(provider, False, False, False, "NOT_CONFIGURED", "NOT_CONFIGURED")
        try:
            response = requests.request(method, url, timeout=self.timeout, **kwargs)
            if response.status_code in {401, 403, 429}:
                kind = classify_provider_error(status_code=response.status_code)
                return self._row(provider, True, True, False, kind, kind)
            if response.status_code >= 400:
                return self._row(provider, True, True, False, "DEGRADED", f"HTTP_{response.status_code}")
            empty = not response.content or response.text.strip() in {"", "{}", "[]", "null"}
            if empty:
                return self._row(provider, True, True, True, "DEGRADED", "EMPTY_DATA")
            return self._row(provider, True, True, True, "HEALTHY")
        except (requests.Timeout, TimeoutError) as exc:
            kind = classify_provider_error(exc)
            return self._row(provider, True, False, False, kind, kind)
        except requests.ConnectionError as exc:
            kind = classify_provider_error(exc)
            return self._row(provider, True, False, False, kind, kind)
        except requests.RequestException:
            return self._row(provider, True, False, False, "UNREACHABLE", "UNREACHABLE")

    def check(self, provider: str) -> dict:
        if provider == "Financial Datasets":
            return self._request(provider, "GET", "https://api.financialdatasets.ai/company/facts/",
                                 params={"ticker": "AAPL"}, headers={"X-API-Key": os.environ.get("FINANCIAL_DATASETS_API_KEY", "").strip()})
        if provider == "SEC":
            return self._request(provider, "GET", "https://data.sec.gov/submissions/CIK0000320193.json",
                                 headers={"User-Agent": os.environ.get("SEC_USER_AGENT", "AI Hedge Fund research@example.com")})
        if provider == "Tavily":
            return self._request(provider, "POST", "https://api.tavily.com/search", json={"api_key": os.environ.get("TAVILY_API_KEY", "").strip(), "query": "Apple investor relations", "max_results": 1, "search_depth": "basic"})
        if provider == "DeepSeek":
            return self._request(provider, "GET", "https://api.deepseek.com/models", headers={"Authorization": f"Bearer {os.environ.get('DEEPSEEK_API_KEY', '').strip()}"})
        if provider == "FRED":
            return self._request(provider, "GET", "https://api.stlouisfed.org/fred/series/observations", params={"series_id": "DGS10", "api_key": os.environ.get("FRED_API_KEY", "").strip(), "file_type": "json", "limit": 1, "sort_order": "desc"})
        if provider == "Yahoo Finance":
            return self._request(provider, "GET", "https://query1.finance.yahoo.com/v8/finance/chart/AAPL", params={"range": "1d", "interval": "1d"}, headers={"User-Agent": "Mozilla/5.0"})
        raise ValueError(f"unknown provider: {provider}")

    def check_all(self, *, force: bool = False) -> list[dict]:
        now = time.time()
        with self._lock:
            if not force and self._cached and now - self._cached_at < 300:
                return [dict(row) for row in self._cached]
        rows = [self.check(provider) for provider in PROVIDERS]
        with self._lock:
            self.__class__._cached = rows
            self.__class__._cached_at = now
        return [dict(row) for row in rows]

