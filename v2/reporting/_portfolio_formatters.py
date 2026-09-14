"""Compatibility re-export for the public v2.reporting namespace.

Single source of truth lives in ``v2/portfolio/_bot_cards.py`` — see
that module for the rationale (v2.reporting's package init pulls in
matplotlib + v2.lateral + v2.backtesting, which transitively requires
the production-only v2.data; keeping the implementation in
``v2/portfolio/`` lets the byte-equal tests stay sandbox-runnable).

The public names use the ``format_portfolio_*`` prefix to match the
Phase 1 ``format_earnings_*`` convention.
"""

from v2.portfolio._bot_cards import (
    format_pnl_period as format_portfolio_pnl_period,
    format_risk_card as format_portfolio_risk_card,
    format_risk_view as format_portfolio_risk_view,
    format_weekly_card as format_portfolio_weekly_card,
)

__all__ = [
    "format_portfolio_pnl_period",
    "format_portfolio_risk_card",
    "format_portfolio_risk_view",
    "format_portfolio_weekly_card",
]
