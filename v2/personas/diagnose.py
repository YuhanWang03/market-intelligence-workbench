"""``python -m v2.personas.diagnose AAPL`` — try every data path a snapshot needs and say what came back.

Run it on the box where the personas run (the VPS) to see, per input,
whether the production ``FDClient`` or the built-in HTTP client served it
and what the error was if not.  No LLM, no scoring.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date, timedelta

from v2.personas.data import ALL_LINE_ITEMS, FinancialDatasetsClient, adapt_client


def _load_env() -> None:
    """Pick up the repo's .env like the services do (systemd EnvironmentFile)."""
    try:
        from pathlib import Path

        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    except Exception:  # noqa: BLE001 — python-dotenv is a dependency, but stay quiet if not
        pass


def _try(label: str, fn, *args, **kwargs) -> None:
    started = time.time()
    try:
        out = fn(*args, **kwargs)
        n = len(out) if isinstance(out, list) else out
        print(f"  ok    {label:<18} {str(n)[:60]:<60} {time.time() - started:5.2f}s")
    except Exception as exc:  # noqa: BLE001
        print(f"  FAIL  {label:<18} {type(exc).__name__}: {str(exc)[:110]}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m v2.personas.diagnose")
    parser.add_argument("ticker", nargs="?", default="AAPL")
    parser.add_argument("--as-of", default=date.today().isoformat())
    parser.add_argument("--http-only", action="store_true", help="skip the production FDClient, test only the built-in HTTP client")
    args = parser.parse_args(argv)
    _load_env()
    ticker, end = args.ticker.upper(), args.as_of
    start = (date.fromisoformat(end) - timedelta(days=365)).isoformat()
    key = os.environ.get("FINANCIAL_DATASETS_API_KEY", "")
    print(f"ticker {ticker} · as of {end} · FINANCIAL_DATASETS_API_KEY {'set' if key else 'MISSING'}")

    clients = []
    if not args.http_only:
        try:
            from v2.data import CachedFDClient  # type: ignore[import-not-found]

            raw = CachedFDClient()
            clients.append(("production FDClient (adapted)", adapt_client(raw)))
            print(f"production FDClient methods: {sorted(m for m in dir(raw) if not m.startswith('_') and callable(getattr(raw, m)))}")
        except Exception as exc:  # noqa: BLE001
            print(f"production FDClient unavailable: {type(exc).__name__}: {exc}")
    clients.append(("built-in HTTP client", FinancialDatasetsClient(key)))

    for name, client in clients:
        print(f"\n== {name}")
        _try("metrics ttm", client.get_financial_metrics, ticker, end, period="ttm", limit=10)
        _try("metrics annual", client.get_financial_metrics, ticker, end, period="annual", limit=10)
        _try("line items ttm", client.search_line_items, ticker, list(ALL_LINE_ITEMS), end, period="ttm", limit=10)
        _try("line items annual", client.search_line_items, ticker, list(ALL_LINE_ITEMS), end, period="annual", limit=10)
        _try("market cap", client.get_market_cap, ticker, end)
        _try("insider trades", client.get_insider_trades, ticker, end, start_date=start, limit=1000)
        _try("news", client.get_company_news, ticker, end, start_date=start, limit=100)
        _try("prices", client.get_prices, ticker, start, end)
    return 0


if __name__ == "__main__":
    sys.exit(main())
