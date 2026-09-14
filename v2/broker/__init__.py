"""Broker adapters — paper / live account access for portfolio + P&L queries.

Currently only Alpaca is supported. The abstraction is kept minimal: we
expose `get_portfolio()` and `get_pnl()` that return plain dicts ready for
the bot's formatter. Adding IBKR / Schwab later means implementing the same
two functions against a different client.
"""

from v2.broker.alpaca_client import (
    AlpacaConfig,
    AlpacaUnavailable,
    get_portfolio,
    get_portfolio_history,
    get_pnl,
)

__all__ = [
    "AlpacaConfig",
    "AlpacaUnavailable",
    "get_portfolio",
    "get_portfolio_history",
    "get_pnl",
]
