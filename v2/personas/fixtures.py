"""Synthetic snapshots for tests and demos — no network, no key.

Two archetypes: a compounding ``quality`` business Buffett would like, and a
``distressed`` one carrying debt, shrinking earnings and dilution.  Both are
plausible enough for every persona's rules to reach a verdict.
"""

from __future__ import annotations

from datetime import date, timedelta

from v2.personas.models import Record
from v2.personas.snapshot import PersonaSnapshot


def _periods(n: int, as_of: str) -> list[str]:
    end = date.fromisoformat(as_of)
    return [(end - timedelta(days=91 * i)).isoformat() for i in range(n)]


def quality_snapshot(ticker: str = "QLTY", as_of: str = "2026-06-30", n: int = 10) -> PersonaSnapshot:
    """A high-ROE, net-cash compounder growing ~12% a year at a full price."""
    periods = _periods(n, as_of)
    metrics, items = _quality_rows(ticker, periods, "ttm", per_period=4)
    metrics_a, items_a = _quality_rows(ticker, _years(n, as_of), "annual", per_period=1)
    prices = _prices(as_of, start=150.0, drift=0.0009, wobble=0.01)
    return PersonaSnapshot(
        ticker=ticker, as_of=as_of,
        metrics_ttm=metrics, metrics_annual=metrics_a,
        line_items_ttm=items, line_items_annual=items_a,
        market_cap=250_000e6,
        insider_trades=[Record(ticker=ticker, transaction_shares=20_000, transaction_date=periods[0], filing_date=periods[0], name="CEO"),
                        Record(ticker=ticker, transaction_shares=5_000, transaction_date=periods[1], filing_date=periods[1], name="CFO")],
        news=[Record(ticker=ticker, title=f"{ticker} beats estimates", sentiment="positive", date=periods[0], source="wire", url="")] * 3,
        prices=prices, fetched_at="2026-06-30T00:00:00",
    )


def _years(n: int, as_of: str) -> list[str]:
    end = date.fromisoformat(as_of)
    return [(end - timedelta(days=365 * i)).isoformat() for i in range(n)]


def _quality_rows(ticker: str, periods: list[str], label: str, *, per_period: int) -> tuple[list[Record], list[Record]]:
    n = len(periods)
    metrics, items = [], []
    for i, p in enumerate(periods):
        g = 1.12 ** (-(i / per_period))  # ~12% annual growth, latest first
        revenue = 40_000e6 * g
        net_income = 10_000e6 * g
        equity = 60_000e6 * g
        shares = 1_000e6 * (1 - 0.01 * (n - i) / n)  # slowly shrinking share count
        metrics.append(Record(
            ticker=ticker, report_period=p, period=label, currency="USD",
            market_cap=250_000e6, enterprise_value=240_000e6,
            price_to_earnings_ratio=25.0, price_to_book_ratio=4.2, price_to_sales_ratio=6.2,
            enterprise_value_to_ebitda_ratio=17.0, free_cash_flow_yield=0.05, peg_ratio=1.9,
            gross_margin=0.62 + 0.002 * (n - i), operating_margin=0.30, net_margin=0.25,
            return_on_equity=0.24, return_on_assets=0.15, return_on_invested_capital=0.22,
            asset_turnover=0.7, current_ratio=2.1, quick_ratio=1.8, debt_to_equity=0.25,
            debt_to_assets=0.15, interest_coverage=30.0, revenue_growth=0.12, earnings_growth=0.13,
            book_value_growth=0.11, earnings_per_share_growth=0.14, free_cash_flow_growth=0.12,
            payout_ratio=0.3, earnings_per_share=net_income / shares, book_value_per_share=equity / shares,
            free_cash_flow_per_share=12_000e6 * g / shares, ev_to_ebit=16.0, beta=0.9,
        ))
        items.append(Record(
            ticker=ticker, report_period=p, period=label, currency="USD",
            revenue=revenue, gross_profit=revenue * 0.62, gross_margin=0.62 + 0.002 * (n - i),
            operating_income=revenue * 0.30, operating_margin=0.30, operating_expense=revenue * 0.32,
            net_income=net_income, earnings_per_share=net_income / shares, ebit=revenue * 0.30,
            ebitda=revenue * 0.34, interest_expense=200e6, research_and_development=revenue * 0.12,
            free_cash_flow=12_000e6 * g, capital_expenditure=-2_500e6 * g, depreciation_and_amortization=1_800e6 * g,
            dividends_and_other_cash_distributions=-3_000e6 * g, issuance_or_purchase_of_equity_shares=-4_000e6 * g,
            outstanding_shares=shares, total_assets=110_000e6 * g, total_liabilities=50_000e6 * g,
            current_assets=45_000e6 * g, current_liabilities=21_000e6 * g, cash_and_equivalents=30_000e6 * g,
            total_debt=15_000e6 * g, shareholders_equity=equity, book_value_per_share=equity / shares,
            goodwill_and_intangible_assets=8_000e6, intangible_assets=3_000e6,
            return_on_invested_capital=0.22, debt_to_equity=0.25,
        ))
    return metrics, items


def distressed_snapshot(ticker: str = "DSTR", as_of: str = "2026-06-30", n: int = 10) -> PersonaSnapshot:
    """A leveraged, shrinking business burning cash and issuing shares."""
    periods = _periods(n, as_of)
    metrics, items = _distressed_rows(ticker, periods, "ttm", per_period=4)
    metrics_a, items_a = _distressed_rows(ticker, _years(n, as_of), "annual", per_period=1)
    prices = _prices(as_of, start=40.0, drift=-0.0015, wobble=0.03)
    return PersonaSnapshot(
        ticker=ticker, as_of=as_of,
        metrics_ttm=metrics, metrics_annual=metrics_a,
        line_items_ttm=items, line_items_annual=items_a,
        market_cap=6_000e6,
        insider_trades=[Record(ticker=ticker, transaction_shares=-150_000, transaction_date=periods[0], filing_date=periods[0], name="CEO"),
                        Record(ticker=ticker, transaction_shares=-40_000, transaction_date=periods[1], filing_date=periods[1], name="Director")],
        news=[Record(ticker=ticker, title=f"{ticker} cuts guidance", sentiment="negative", date=periods[0], source="wire", url="")] * 6,
        prices=prices, fetched_at="2026-06-30T00:00:00",
    )


def _distressed_rows(ticker: str, periods: list[str], label: str, *, per_period: int) -> tuple[list[Record], list[Record]]:
    n = len(periods)
    metrics, items = [], []
    for i, p in enumerate(periods):
        g = 0.9 ** (-(i / per_period))  # shrinking ~10% a year, latest first (older periods larger)
        revenue = 8_000e6 * g
        net_income = -300e6 * (1 + 0.1 * (n - i))
        equity = 2_000e6 * g
        shares = 500e6 * (1 + 0.03 * (n - i))  # dilution
        metrics.append(Record(
            ticker=ticker, report_period=p, period=label, currency="USD",
            market_cap=6_000e6, enterprise_value=14_000e6,
            price_to_earnings_ratio=None, price_to_book_ratio=3.0, price_to_sales_ratio=0.75,
            enterprise_value_to_ebitda_ratio=35.0, free_cash_flow_yield=-0.04, peg_ratio=None,
            gross_margin=0.22 - 0.003 * (n - i), operating_margin=-0.02, net_margin=-0.04,
            return_on_equity=-0.15, return_on_assets=-0.03, return_on_invested_capital=-0.02,
            asset_turnover=0.6, current_ratio=0.9, quick_ratio=0.6, debt_to_equity=4.5,
            debt_to_assets=0.7, interest_coverage=0.8, revenue_growth=-0.1, earnings_growth=-0.3,
            book_value_growth=-0.2, earnings_per_share_growth=-0.35, free_cash_flow_growth=-0.5,
            payout_ratio=0.0, earnings_per_share=net_income / shares, book_value_per_share=equity / shares,
            free_cash_flow_per_share=-0.5, ev_to_ebit=None, beta=1.8,
        ))
        items.append(Record(
            ticker=ticker, report_period=p, period=label, currency="USD",
            revenue=revenue, gross_profit=revenue * 0.22, gross_margin=0.22 - 0.003 * (n - i),
            operating_income=-revenue * 0.02, operating_margin=-0.02, operating_expense=revenue * 0.24,
            net_income=net_income, earnings_per_share=net_income / shares, ebit=-revenue * 0.02,
            ebitda=revenue * 0.03, interest_expense=600e6, research_and_development=revenue * 0.02,
            free_cash_flow=-250e6 * g, capital_expenditure=-900e6 * g, depreciation_and_amortization=400e6 * g,
            dividends_and_other_cash_distributions=0.0, issuance_or_purchase_of_equity_shares=400e6 * g,
            outstanding_shares=shares, total_assets=13_000e6 * g, total_liabilities=11_000e6 * g,
            current_assets=2_500e6 * g, current_liabilities=2_800e6 * g, cash_and_equivalents=700e6 * g,
            total_debt=9_000e6 * g, shareholders_equity=equity, book_value_per_share=equity / shares,
            goodwill_and_intangible_assets=3_000e6, intangible_assets=1_500e6,
            return_on_invested_capital=-0.02, debt_to_equity=4.5,
        ))
    return metrics, items


def _prices(as_of: str, *, start: float, drift: float, wobble: float, days: int = 260) -> list[Record]:
    """Deterministic pseudo-random daily closes, ascending by date."""
    end = date.fromisoformat(as_of)
    out = []
    price = start
    seed = 12345
    for i in range(days):
        seed = (seed * 1103515245 + 12345) % (2 ** 31)
        noise = ((seed / 2 ** 31) - 0.5) * 2 * wobble
        price = max(1.0, price * (1 + drift + noise))
        d = end - timedelta(days=days - i)
        out.append(Record(
            time=d.isoformat(), open=price * 0.995, high=price * 1.01, low=price * 0.99, close=price,
            volume=int(5e6 * (1 + abs(noise) * 10)),
        ))
    return out


def empty_snapshot(ticker: str = "NODATA", as_of: str = "2026-06-30") -> PersonaSnapshot:
    return PersonaSnapshot(ticker=ticker, as_of=as_of, gaps=["fundamentals: no metrics or line items returned"])
