"""Data access for the personas: one protocol, one HTTP client, one adapter.

The thirteen upstream agents each called ``financialdatasets.ai`` directly.
Here every persona reads from a :class:`~v2.personas.snapshot.PersonaSnapshot`
built through the :class:`PersonaDataClient` protocol, so the same rules run
against the production ``FDClient`` on the VPS, this stand-alone
:class:`FinancialDatasetsClient`, or a fake in tests.

Nothing in this module imports ``v2.data`` — that package is production-only
and absent from the repo checkout.  :func:`adapt_client` duck-types whatever
client it is handed and fills the gaps from the HTTP client when an API key
is configured.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime
from typing import Any, Protocol

from v2.personas.models import Record, as_records

logger = logging.getLogger(__name__)

FD_BASE_URL = "https://api.financialdatasets.ai"
#: Cloudflare in front of financialdatasets.ai blocks urllib's default
#: "Python-urllib/3.x" agent with a 403 whose body is a Cloudflare problem
#: document. Identify ourselves like an ordinary HTTP client instead.
USER_AGENT = "ai-hedge-fund-altdata/2026 (+https://github.com/YuhanWang03/ai-hedge-fund-altdata; python)"
#: financialdatasets.ai rejects large /news/ pages with "Invalid limit"; step down until accepted
NEWS_LIMIT_STEPS = (100, 50, 20, 10)
NEWS_MAX_LIMIT = NEWS_LIMIT_STEPS[0]
_INVALID_ITEMS = re.compile(r"Invalid line items?:\s*([A-Za-z0-9_,\s]+)")


def _invalid_line_items(error: Exception) -> list[str]:
    """Names financialdatasets.ai rejected, parsed from its 400 body."""
    match = _INVALID_ITEMS.search(str(error))
    if not match:
        return []
    return [n.strip() for n in match.group(1).split(",") if n.strip()]


def _with_smaller_news_limit(fn, limit: int):
    """Call ``fn(limit)`` stepping the page size down on failure."""
    last: Exception | None = None
    for step in [limit] + [n for n in NEWS_LIMIT_STEPS if n < limit]:
        try:
            return fn(step)
        except Exception as exc:  # noqa: BLE001
            last = exc
            if "limit" not in str(exc).lower() and "400" not in str(exc):
                raise
    raise last if last else RuntimeError("news fetch failed")

#: Union of every line item any persona reads. Fetched once per ticker and
#: period, then shared, so a 13-persona run costs two line-item calls, not 13.
ALL_LINE_ITEMS: tuple[str, ...] = (
    "revenue",
    "gross_profit",
    "gross_margin",
    "operating_income",
    "operating_margin",
    "operating_expense",
    "net_income",
    "earnings_per_share",
    "ebit",
    "ebitda",
    "interest_expense",
    "research_and_development",
    "free_cash_flow",
    "capital_expenditure",
    "depreciation_and_amortization",
    "dividends_and_other_cash_distributions",
    "issuance_or_purchase_of_equity_shares",
    "outstanding_shares",
    "total_assets",
    "total_liabilities",
    "current_assets",
    "current_liabilities",
    "cash_and_equivalents",
    "total_debt",
    "shareholders_equity",
    "book_value_per_share",
    "goodwill_and_intangible_assets",
    "return_on_invested_capital",
    "debt_to_equity",
)


class PersonaDataClient(Protocol):
    """Everything :func:`~v2.personas.snapshot.build_snapshot` needs."""

    def get_financial_metrics(self, ticker: str, end_date: str, *, period: str = "ttm", limit: int = 10) -> list[Any]: ...

    def search_line_items(self, ticker: str, line_items: list[str], end_date: str, *, period: str = "ttm", limit: int = 10) -> list[Any]: ...

    def get_market_cap(self, ticker: str, end_date: str) -> float | None: ...

    def get_insider_trades(self, ticker: str, end_date: str, *, start_date: str | None = None, limit: int = 1000) -> list[Any]: ...

    def get_company_news(self, ticker: str, end_date: str, *, start_date: str | None = None, limit: int = 100) -> list[Any]: ...

    def get_prices(self, ticker: str, start_date: str, end_date: str) -> list[Any]: ...


def _iso(value: Any) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)[:10]


class FinancialDatasetsClient:
    """Minimal ``financialdatasets.ai`` client — stdlib only, memoised per instance.

    Ported from the upstream ``src/tools/api.py`` with three changes: urllib
    instead of ``requests``, a short exponential backoff on 429 instead of a
    60-second sleep, and every result returned as :class:`Record`.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = FD_BASE_URL,
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        self.api_key = (api_key or os.environ.get("FINANCIAL_DATASETS_API_KEY") or "").strip()
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self._cache: dict[str, Any] = {}
        self.calls = 0

    # -- transport -----------------------------------------------------------

    def _request(self, path: str, *, params: dict[str, Any] | None = None, body: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        if params:
            url += "?" + urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
        key = url + ("|" + json.dumps(body, sort_keys=True) if body else "")
        if key in self._cache:
            return self._cache[key]

        headers = {"Accept": "application/json", "User-Agent": USER_AGENT}
        if self.api_key:
            headers["X-API-KEY"] = self.api_key
        data = None
        if body is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(body).encode("utf-8")

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            request = urllib.request.Request(url, data=data, headers=headers, method="POST" if body is not None else "GET")
            try:
                self.calls += 1
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = json.loads(response.read().decode("utf-8") or "{}")
                from v2.data.usage_ledger import record_fd
                record_fd(path, params)
                self._cache[key] = payload
                return payload
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")[:200]
                hint = ""
                if exc.code == 403 and "cloudflare" in detail.lower():
                    hint = " [blocked by Cloudflare in front of the API — check the User-Agent / IP reputation]"
                elif exc.code in (401, 403):
                    hint = " [check FINANCIAL_DATASETS_API_KEY and whether the plan includes this endpoint]"
                last_error = RuntimeError(f"HTTP {exc.code} for {path}: {detail}{hint}")
                if exc.code == 429 and attempt < self.max_retries:
                    time.sleep(2.0 ** attempt)
                    continue
                if exc.code == 404:
                    self._cache[key] = {}
                    return {}
                break
            except Exception as exc:  # noqa: BLE001 — network flakiness
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(2.0 ** attempt)
                    continue
        raise RuntimeError(f"financialdatasets request failed: {last_error}") from last_error

    # -- endpoints -----------------------------------------------------------

    def get_financial_metrics(self, ticker: str, end_date: str, *, period: str = "ttm", limit: int = 10) -> list[Record]:
        payload = self._request(
            "/financial-metrics/",
            params={"ticker": ticker, "report_period_lte": _iso(end_date), "limit": limit, "period": period},
        )
        return as_records(payload.get("financial_metrics") or [])

    def search_line_items(self, ticker: str, line_items: list[str], end_date: str, *, period: str = "ttm", limit: int = 10) -> list[Record]:
        wanted = list(dict.fromkeys(line_items))
        for _ in range(4):  # the API names only the first offenders; a few rounds settle it
            try:
                payload = self._request(
                    "/financials/search/line-items",
                    body={"tickers": [ticker], "line_items": wanted, "end_date": _iso(end_date), "period": period, "limit": limit},
                )
            except RuntimeError as exc:
                bad = [n for n in _invalid_line_items(exc) if n in wanted]
                if not bad:
                    raise
                logger.warning("financialdatasets rejected line items %s; retrying without them", bad)
                wanted = [n for n in wanted if n not in bad]
                if not wanted:
                    return []
                continue
            return as_records(payload.get("search_results") or [])[:limit]
        return []

    def get_market_cap(self, ticker: str, end_date: str) -> float | None:
        end = _iso(end_date)
        if end == date.today().isoformat():
            payload = self._request("/company/facts/", params={"ticker": ticker})
            facts = payload.get("company_facts") or {}
            value = facts.get("market_cap")
            if value:
                return float(value)
        metrics = self.get_financial_metrics(ticker, end, period="ttm", limit=1)
        value = metrics[0].market_cap if metrics else None
        return float(value) if value else None

    def get_insider_trades(self, ticker: str, end_date: str, *, start_date: str | None = None, limit: int = 1000) -> list[Record]:
        params: dict[str, Any] = {"ticker": ticker, "filing_date_lte": _iso(end_date), "limit": limit}
        if start_date:
            params["filing_date_gte"] = _iso(start_date)
        payload = self._request("/insider-trades/", params=params)
        return as_records(payload.get("insider_trades") or [])

    def get_company_news(self, ticker: str, end_date: str, *, start_date: str | None = None, limit: int = 100) -> list[Record]:
        def fetch(page: int) -> list[Record]:
            params: dict[str, Any] = {"ticker": ticker, "end_date": _iso(end_date), "limit": page}
            if start_date:
                params["start_date"] = _iso(start_date)
            return as_records(self._request("/news/", params=params).get("news") or [])

        return _with_smaller_news_limit(fetch, min(limit, NEWS_MAX_LIMIT))

    def get_prices(self, ticker: str, start_date: str, end_date: str) -> list[Record]:
        payload = self._request(
            "/prices/",
            params={"ticker": ticker, "interval": "day", "interval_multiplier": 1, "start_date": _iso(start_date), "end_date": _iso(end_date)},
        )
        return as_records(payload.get("prices") or [])


class _Adapted:
    """Wrap a foreign client (e.g. the production ``FDClient``) behind the protocol.

    Each method tries the wrapped object first, tolerating the positional /
    keyword spellings the production client uses, then the fallback client.
    A method neither side provides raises ``NotImplementedError`` so the
    snapshot builder can record a data gap rather than crash.
    """

    def __init__(self, primary: Any, fallback: PersonaDataClient | None) -> None:
        self.primary = primary
        self.fallback = fallback

    def _call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        fn = getattr(self.primary, name, None)
        if fn is not None:
            try:
                return fn(*args, **kwargs)
            except TypeError:
                # Production FDClient spells some parameters positionally.
                return fn(*args, *kwargs.values())
        if self.fallback is not None:
            return getattr(self.fallback, name)(*args, **kwargs)
        raise NotImplementedError(f"data client has no {name}()")

    def get_financial_metrics(self, ticker: str, end_date: str, *, period: str = "ttm", limit: int = 10) -> list[Record]:
        fn = getattr(self.primary, "get_financial_metrics", None)
        if fn is not None:
            try:
                rows = fn(ticker, end_date, period=period, limit=limit)
            except TypeError:
                rows = fn(ticker, end_date, limit=limit)
            return as_records(rows)
        if self.fallback is not None:
            return self.fallback.get_financial_metrics(ticker, end_date, period=period, limit=limit)
        raise NotImplementedError("data client has no get_financial_metrics()")

    def search_line_items(self, ticker: str, line_items: list[str], end_date: str, *, period: str = "ttm", limit: int = 10) -> list[Record]:
        return as_records(self._call("search_line_items", ticker, line_items, end_date, period=period, limit=limit))

    def get_market_cap(self, ticker: str, end_date: str) -> float | None:
        fn = getattr(self.primary, "get_market_cap", None)
        if fn is not None:
            # The production client's get_market_cap has been seen to raise from
            # inside (a CompanyFacts model without market_cap); treat any failure
            # as "derive it another way" rather than as a snapshot gap.
            for args in ((ticker, end_date), (ticker,)):
                try:
                    value = fn(*args)
                except TypeError:
                    continue
                except Exception as exc:  # noqa: BLE001
                    logger.debug("primary get_market_cap failed for %s: %s", ticker, exc)
                    break
                if value:
                    return float(value)
                break
        # Cheap derivation the production client always supports.
        try:
            rows = self.get_financial_metrics(ticker, end_date, period="ttm", limit=1)
        except Exception:  # noqa: BLE001
            rows = []
        value = rows[0].market_cap if rows else None
        if value:
            return float(value)
        if self.fallback is not None:
            return self.fallback.get_market_cap(ticker, end_date)
        return None

    def get_insider_trades(self, ticker: str, end_date: str, *, start_date: str | None = None, limit: int = 1000) -> list[Record]:
        return as_records(self._call("get_insider_trades", ticker, end_date, start_date=start_date, limit=limit))

    def get_company_news(self, ticker: str, end_date: str, *, start_date: str | None = None, limit: int = 100) -> list[Record]:
        fn = getattr(self.primary, "get_company_news", None) or getattr(self.primary, "get_news", None)
        limit = min(limit, NEWS_MAX_LIMIT)
        if fn is not None:
            def fetch(page: int) -> list[Record]:
                try:
                    return as_records(fn(ticker, end_date, start_date=start_date, limit=page))
                except TypeError:
                    return as_records(fn(ticker, end_date, start_date, page))

            return _with_smaller_news_limit(fetch, limit)
        if self.fallback is not None:
            return self.fallback.get_company_news(ticker, end_date, start_date=start_date, limit=limit)
        raise NotImplementedError("data client has no get_company_news()")

    def get_prices(self, ticker: str, start_date: str, end_date: str) -> list[Record]:
        return as_records(self._call("get_prices", ticker, start_date, end_date))


def adapt_client(client: Any | None = None, *, api_key: str | None = None) -> PersonaDataClient:
    """Return something satisfying :class:`PersonaDataClient`.

    * ``None`` → a fresh :class:`FinancialDatasetsClient` (needs the env key).
    * A client that already speaks the protocol → returned as is.
    * Anything else (production ``FDClient``, a yfinance shim) → wrapped, with
      the HTTP client filling in whatever methods it lacks when a key exists.
    """
    if client is None:
        return FinancialDatasetsClient(api_key)
    if isinstance(client, (FinancialDatasetsClient, _Adapted)):
        return client
    needed = ("get_financial_metrics", "search_line_items", "get_market_cap", "get_insider_trades", "get_company_news", "get_prices")
    if all(callable(getattr(client, name, None)) for name in needed):
        return client
    key = (api_key or os.environ.get("FINANCIAL_DATASETS_API_KEY") or "").strip()
    fallback = FinancialDatasetsClient(key) if key else None
    return _Adapted(client, fallback)
