"""yfinance client — fallback data source for tickers FD doesn't cover well.

Used specifically for foreign ADRs (TSM, ASML, STM, etc.) where FD has
gaps in financial-metrics or pricing data. Mirrors the relevant parts of
the FDClient interface so build_candidate works identically.

Only implements the methods we actually use from the DataClient protocol:
get_prices, get_financial_metrics, get_company_facts.
"""

from __future__ import annotations

import logging

import yfinance as yf

from v2.data.models import CompanyFacts, FinancialMetrics, Price

logger = logging.getLogger(__name__)


# Tickers where FD has known coverage gaps; the orchestrator will pass
# YFinanceClient as a fallback for these.
KNOWN_ADRS: frozenset[str] = frozenset({
    "TSM",   # Taiwan Semiconductor
    "ASML",  # ASML Holding
    "STM",   # STMicroelectronics
})


class YFinanceClient:
    """Minimal yfinance adapter — get_prices / get_financial_metrics / get_company_facts."""

    def get_prices(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        **_,
    ) -> list[Price]:
        """Daily OHLCV via yf.Ticker.history(). Returns [] on any failure."""
        try:
            hist = yf.Ticker(ticker).history(
                start=start_date,
                end=end_date,
                auto_adjust=False,
            )
        except Exception as exc:
            logger.warning("yfinance get_prices(%s) failed: %s", ticker, exc)
            return []

        if hist is None or hist.empty:
            return []

        prices: list[Price] = []
        for idx, row in hist.iterrows():
            try:
                prices.append(Price(
                    open=float(row["Open"]),
                    close=float(row["Close"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    volume=int(row["Volume"]),
                    time=idx.strftime("%Y-%m-%d"),
                ))
            except (KeyError, ValueError, TypeError):
                continue
        return prices

    def get_financial_metrics(
        self,
        ticker: str,
        end_date: str,
        period: str = "ttm",
        limit: int = 10,
    ) -> list[FinancialMetrics]:
        """One TTM snapshot from yf.Ticker.info — yfinance doesn't expose history."""
        try:
            info = yf.Ticker(ticker).info
        except Exception as exc:
            logger.warning("yfinance get_info(%s) failed: %s", ticker, exc)
            return []

        if not info:
            return []

        return [FinancialMetrics(
            ticker=ticker,
            report_period=end_date,
            period="ttm",
            currency=info.get("currency") or "USD",
            market_cap=_safe_float(info.get("marketCap")),
            revenue_growth=_safe_float(info.get("revenueGrowth")),
            gross_margin=_safe_float(info.get("grossMargins")),
            net_margin=_safe_float(info.get("profitMargins")),
            return_on_equity=_safe_float(info.get("returnOnEquity")),
            debt_to_equity=_safe_float(info.get("debtToEquity")),
        )]

    def get_company_facts(self, ticker: str) -> CompanyFacts | None:
        """Build a CompanyFacts from yf.Ticker.info — used if FD doesn't have it."""
        try:
            info = yf.Ticker(ticker).info
        except Exception:
            return None

        if not info or not info.get("symbol"):
            return None

        return CompanyFacts(
            ticker=ticker,
            name=info.get("longName") or info.get("shortName"),
            sector=info.get("sector"),
            industry=info.get("industry"),
            exchange=info.get("exchange"),
            is_active=True,
        )


def _safe_float(v) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
        # yfinance occasionally returns "Infinity" string or actual inf
        if f != f or f == float("inf") or f == float("-inf"):
            return None
        return f
    except (ValueError, TypeError):
        return None
