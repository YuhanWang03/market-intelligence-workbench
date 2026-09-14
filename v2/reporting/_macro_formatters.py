"""Compatibility re-export for the public v2.reporting namespace.

Single source of truth lives in ``v2/macro/_bot_cards.py`` — see that
module for the rationale (v2.reporting's package init pulls in
matplotlib + v2.lateral + v2.backtesting, which transitively requires
the production-only v2.data; keeping the implementation in
``v2/macro/`` lets the byte-equal tests stay sandbox-runnable).

The public names use the ``format_macro_*`` prefix to match the
Phase 1 ``format_earnings_*`` / Phase 2 ``format_portfolio_*`` /
Phase 3 ``format_sec_*`` conventions.
"""

from v2.macro._bot_cards import (
    format_macro_claims_card,
    format_macro_daily_snapshot,
    format_macro_dashboard,
    format_macro_fomc_card,
    format_macro_release_card,
    format_macro_weekly_recap,
)

__all__ = [
    "format_macro_claims_card",
    "format_macro_daily_snapshot",
    "format_macro_dashboard",
    "format_macro_fomc_card",
    "format_macro_release_card",
    "format_macro_weekly_recap",
]
