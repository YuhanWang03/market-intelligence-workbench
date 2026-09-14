"""Run PEAD on a small universe and push results to Telegram.

Usage:
    poetry run python scripts/pead_to_telegram.py
"""

from __future__ import annotations

from dotenv import load_dotenv

from v2.backtesting import BacktestEngine, PEADStrategy
from v2.data import FDClient
from v2.reporting import (
    TelegramNotifier,
    format_backtest_summary,
    render_equity_curve,
)

load_dotenv()

TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "META",
    "AMZN", "TSLA", "AMD", "NFLX", "CRM",
]
HOLDING_DAYS = 5
CAPITAL = 100_000.0
PER_TRADE = 10_000.0


def main() -> None:
    print(f"Running PEAD on {len(TICKERS)} tickers...")
    with FDClient() as fd:
        strategy = PEADStrategy(holding_days=HOLDING_DAYS)
        engine = BacktestEngine(capital=CAPITAL, per_trade=PER_TRADE)
        result = engine.run(strategy, TICKERS, fd)

    if not result.trades:
        print("No trades generated — nothing to send.")
        return

    print(f"{len(result.trades)} trades — pushing to Telegram...")
    notifier = TelegramNotifier()
    notifier.send_text(format_backtest_summary(
        result, strategy_name="PEAD", universe_size=len(TICKERS),
    ))
    notifier.send_photo(
        render_equity_curve(result, title="PEAD · Equity Curve"),
        caption=f"<i>{len(result.trades)} trades · {HOLDING_DAYS}-day hold</i>",
    )
    print("Done.")


if __name__ == "__main__":
    main()
