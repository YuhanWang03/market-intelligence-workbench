"""Forward-return backfill — the honest way to score an LLM-free persona.

Every committee run stores each persona's vote with its date.  Once a month
(or three) has passed, this module looks up what the stock actually did and
writes ``fwd_1m`` / ``fwd_3m`` next to the vote.  ``PersonaStore.persona_scoreboard()``
then turns those into a per-persona hit rate with no look-ahead: the vote
was cast before the return existed.

Prices come from any object with ``get_prices(ticker, start, end)`` returning
rows with ``close`` and ``time`` (the v2 ``PriceSource`` protocol, or the
personas' own Financial Datasets client).  Run it from the scheduler, the
CLI (``python -m v2.personas.forward``) or the Lab endpoint.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

from v2.personas.store import PersonaStore

logger = logging.getLogger(__name__)

#: calendar days after the vote at which each horizon is measured
HORIZONS: dict[str, int] = {"fwd_1m": 30, "fwd_3m": 91}
#: how many trading days of slack to allow when the exact date has no bar
_WINDOW_DAYS = 6


@dataclass
class BackfillReport:
    checked: int = 0
    filled: int = 0
    skipped_no_price: int = 0
    errors: dict[str, str] = field(default_factory=dict)
    by_column: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"checked": self.checked, "filled": self.filled, "skipped_no_price": self.skipped_no_price, "errors": dict(self.errors), "by_column": dict(self.by_column)}


def _default_price_source() -> Any:
    try:
        from v2.data.price_source import default_price_source

        return default_price_source()
    except Exception:  # noqa: BLE001 — production-only package absent
        from v2.personas.data import FinancialDatasetsClient

        return FinancialDatasetsClient()


def _close(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if f == f and f > 0 else None


def _row_date(row: Any) -> str:
    for name in ("time", "date"):
        value = getattr(row, name, None) if not isinstance(row, dict) else row.get(name)
        if value:
            return str(value)[:10]
    return ""


class _PriceLookup:
    """One fetch per ticker, then close-on-or-after lookups by date."""

    def __init__(self, source: Any) -> None:
        self.source = source
        self._series: dict[str, list[tuple[str, float]]] = {}

    def load(self, ticker: str, start: date, end: date) -> None:
        rows = self.source.get_prices(ticker, start.isoformat(), end.isoformat()) or []
        series = []
        for row in rows:
            d = _row_date(row)
            c = _close(getattr(row, "close", None) if not isinstance(row, dict) else row.get("close"))
            if d and c is not None:
                series.append((d, c))
        series.sort()
        self._series[ticker] = series

    def close_on_or_after(self, ticker: str, day: date, *, within: int = _WINDOW_DAYS) -> float | None:
        limit = (day + timedelta(days=within)).isoformat()
        for d, c in self._series.get(ticker, []):
            if day.isoformat() <= d <= limit:
                return c
        return None

    def close_on_or_before(self, ticker: str, day: date, *, within: int = _WINDOW_DAYS) -> float | None:
        floor = (day - timedelta(days=within)).isoformat()
        best = None
        for d, c in self._series.get(ticker, []):
            if floor <= d <= day.isoformat():
                best = c
        return best


def backfill_forward_returns(store: PersonaStore, price_source: Any | None = None, *, today: date | None = None, columns: tuple[str, ...] = ("fwd_1m", "fwd_3m")) -> BackfillReport:
    """Fill every due ``fwd_*`` cell. Idempotent; safe to run daily."""
    today = today or date.today()
    source = price_source or _default_price_source()
    lookup = _PriceLookup(source)
    report = BackfillReport()

    for column in columns:
        days = HORIZONS[column]
        pending = store.signals_awaiting_forward_returns(older_than_days=days, column=column, today=today)
        report.checked += len(pending)
        by_ticker: dict[str, list[dict[str, Any]]] = {}
        for row in pending:
            by_ticker.setdefault(row["ticker"], []).append(row)
        for ticker, rows in by_ticker.items():
            dates = sorted(date.fromisoformat(r["as_of"]) for r in rows)
            try:
                lookup.load(ticker, dates[0] - timedelta(days=_WINDOW_DAYS), min(today, dates[-1] + timedelta(days=days + _WINDOW_DAYS)))
            except Exception as exc:  # noqa: BLE001 — one ticker's outage must not stop the sweep
                logger.warning("forward backfill: prices for %s failed: %s", ticker, exc)
                report.errors[ticker] = f"{type(exc).__name__}: {str(exc)[:120]}"
                continue
            for r in rows:
                start_day = date.fromisoformat(r["as_of"])
                base = _close(r.get("price_at")) or lookup.close_on_or_before(ticker, start_day)
                later = lookup.close_on_or_after(ticker, start_day + timedelta(days=days))
                if base is None or later is None:
                    report.skipped_no_price += 1
                    continue
                store.set_forward_return(int(r["id"]), column=column, value=later / base - 1.0)
                report.filled += 1
                report.by_column[column] = report.by_column.get(column, 0) + 1
    logger.info("forward backfill: checked %d, filled %d, no price %d, errors %d", report.checked, report.filled, report.skipped_no_price, len(report.errors))
    return report


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(prog="python -m v2.personas.forward", description="Back-fill forward returns for stored persona votes.")
    parser.add_argument("--db", default=None, help="path to personas.db (default: data/personas.db)")
    parser.add_argument("--scoreboard", action="store_true", help="print the per-persona hit rate afterwards")
    args = parser.parse_args(argv)
    store = PersonaStore(args.db)
    report = backfill_forward_returns(store)
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    if args.scoreboard:
        print(json.dumps(store.persona_scoreboard(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
