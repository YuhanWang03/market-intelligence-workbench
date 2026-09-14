from __future__ import annotations

from types import SimpleNamespace

import pytest

from v2.research.cache import ResearchCache
from v2.research.engine import MODULES, ResearchEngine, normalize_ticker
from v2.research.services import ResearchServices


def ns(**kwargs):
    return SimpleNamespace(**kwargs)


class FakeFD:
    def __init__(self, *, negative_pe: bool = False, missing: bool = False):
        self.negative_pe = negative_pe
        self.missing = missing
        self.calls: dict[str, int] = {}

    def _call(self, name):
        self.calls[name] = self.calls.get(name, 0) + 1

    def get_company_facts(self, ticker):
        self._call("company")
        return None if self.missing else ns(ticker=ticker, name=f"{ticker} Corp", sector="Technology", industry="Software", exchange="NASDAQ", location="US", sec_filings_url="https://sec.example/company")

    def get_financial_metrics(self, ticker, end_date, period="ttm", limit=20):
        self._call("metrics")
        if self.missing:
            return []
        return [ns(report_period=f"202{i}-12-31", market_cap=1e11, price_to_earnings_ratio=-5 if self.negative_pe else 25 + i, price_to_book_ratio=7, price_to_sales_ratio=8, enterprise_value_to_ebitda_ratio=20, enterprise_value_to_revenue_ratio=7, free_cash_flow_yield=.035, peg_ratio=1.8, gross_margin=.62, operating_margin=.3, net_margin=.24, return_on_equity=.32, return_on_assets=.15, return_on_invested_capital=.25, current_ratio=1.8, quick_ratio=1.4, cash_ratio=.7, operating_cash_flow_ratio=.3, debt_to_equity=.4, debt_to_assets=.2, interest_coverage=16, revenue_growth=.2, earnings_growth=.24, earnings_per_share_growth=.22, free_cash_flow_growth=.18, operating_income_growth=.2, ebitda_growth=.2) for i in range(5, 0, -1)]

    def get_earnings_history(self, ticker, limit=32):
        self._call("earnings")
        if self.missing:
            return []
        rows = []
        for i in range(8):
            q = ns(revenue=10e9 + i * 1e9, estimated_revenue=9.8e9 + i * 1e9, revenue_surprise="BEAT", earnings_per_share=2 + i * .1, estimated_earnings_per_share=1.9 + i * .1, eps_surprise="BEAT", net_income=2e9, gross_profit=6e9, operating_income=3e9, weighted_average_shares=1e9, weighted_average_shares_diluted=1e9 + i * 1e6, free_cash_flow=2.2e9, cash_and_equivalents=20e9, total_debt=5e9, total_assets=60e9, total_liabilities=20e9, shareholders_equity=40e9, net_cash_flow_from_operations=3e9, capital_expenditure=-.8e9, revenue_chg=.2 - i * .005, net_income_chg=.18, operating_income_chg=.2, gross_profit_chg=.17, free_cash_flow_chg=.19)
            rows.append(ns(ticker=ticker, report_period=f"202{6 - i // 4}-{12 - (i % 4) * 3:02d}-31", source_type="10-Q", filing_date=f"202{6 - i // 4}-01-30", filing_url="https://sec.example/filing", quarterly=q))
        return rows

    def get_insider_trades(self, *args, **kwargs):
        self._call("insiders")
        return []

    def get_news(self, *args, **kwargs):
        self._call("news")
        return []


class FakePrices:
    def get_prices(self, ticker, start, end):
        return [ns(time=f"2026-01-{(i % 28) + 1:02d}", open=100 + i * .1, high=102 + i * .1, low=99 + i * .1, close=101 + i * .1, volume=1_000_000 + i * 1000) for i in range(240)]


def fake_services():
    return ResearchServices(
        earnings_sec=ns(collect=lambda ticker, earnings, upcoming: {
            "expectations": {"upcoming_earnings": None, "revision_history": []},
            "filings": [{"form": "10-Q", "filing_date": "2026-01-30", "url": "https://sec.example/filing", "source": "Financial Datasets"}],
            "warnings": [],
        }),
        institutional=ns(collect=lambda ticker, insiders: {
            "institutional_holdings": [], "13f_changes": [], "insider_transactions": [],
            "ownership_changes": [], "etf_exposure": [], "warnings": [],
        }),
        macro=ns(collect=lambda today: {"snapshot": {"vix": 18.0, "dgs10": 4.1}, "upcoming_events": [], "warnings": []}),
        supply_chain=ns(collect=lambda ticker, fd: {"date": "2026-09-05", "neighbors": [], "llm_tokens": 0, "api_calls": 0, "tavily_calls": 0}),
    )


@pytest.mark.parametrize("ticker", ["NVDA", "AAPL", "JPM", "XOM", "TSLA"])
def test_research_engine_produces_uniform_result_for_company_types(tmp_path, ticker):
    fd = FakeFD()
    engine = ResearchEngine(fd=fd, prices=FakePrices(), cache=ResearchCache(tmp_path / "research.db"), services=fake_services())

    result = engine.run(ticker)

    assert result["ticker"] == ticker
    assert result["status"] in {"PARTIAL_DATA", "PARTIAL_ERROR"}
    assert set(MODULES) == set(result["modules"])
    assert result["modules"]["fundamental"]["score"] is not None
    assert result["investment_thesis"]
    assert result["thesis_invalidation"]
    assert all(fd.calls[name] == 1 for name in ("company", "metrics", "earnings", "insiders", "news"))


def test_negative_pe_is_na_and_missing_segments_are_explicit(tmp_path):
    result = ResearchEngine(fd=FakeFD(negative_pe=True), prices=FakePrices(), cache=ResearchCache(tmp_path / "research.db"), services=fake_services()).run("TSLA")

    assert result["modules"]["valuation"]["metrics"]["pe_ttm"] is None
    assert result["modules"]["fundamental"]["details"]["company"]["segments"] == []


def test_core_data_failure_marks_run_failed_without_crashing(tmp_path):
    result = ResearchEngine(fd=FakeFD(missing=True), prices=ns(get_prices=lambda *_: []), cache=ResearchCache(tmp_path / "research.db"), services=fake_services()).run("NVDA")

    assert result["status"] == "FAILED"
    assert result["modules"]["fundamental"]["status"] == "PARTIAL_DATA"
    assert result["modules"]["valuation"]["status"] == "PARTIAL_DATA"
    assert result["modules"]["earnings"]["provider_errors"][0]["type"] == "EMPTY_DATA"


def test_research_cache_prevents_duplicate_provider_calls(tmp_path):
    fd = FakeFD()
    engine = ResearchEngine(fd=fd, prices=FakePrices(), cache=ResearchCache(tmp_path / "research.db"), services=fake_services())
    engine.run("AAPL")
    cached = engine.run("AAPL")

    assert cached["from_cache"] is True
    assert fd.calls["metrics"] == 1


def test_optional_service_failure_yields_partial_aggregate(tmp_path):
    services = fake_services()
    services.supply_chain = ns(collect=lambda ticker, fd: (_ for _ in ()).throw(RuntimeError("provider down")))

    result = ResearchEngine(
        fd=FakeFD(),
        prices=FakePrices(),
        cache=ResearchCache(tmp_path / "research.db"),
        services=services,
    ).run("AAPL", refresh=True)

    assert result["status"] == "PARTIAL_ERROR"
    assert result["modules"]["supply_chain"]["status"] == "PARTIAL_ERROR"
    assert result["modules"]["fundamental"]["status"] != "FAILED"
    assert result["modules"]["risk"]["status"] in {"COMPLETED", "PARTIAL_DATA", "PARTIAL_ERROR"}


def test_ticker_validation():
    assert normalize_ticker(" brk.b ") == "BRK.B"
    with pytest.raises(ValueError):
        normalize_ticker("not a ticker")
