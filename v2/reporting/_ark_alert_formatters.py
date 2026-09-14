"""Compatibility re-export for ARK alert card formatters.

Single source of truth lives in ``v2/etf/_ark_alert_cards.py`` — see
that module for the rationale (v2.reporting's package init pulls
matplotlib + v2.lateral + v2.backtesting, which require the
production-only v2.data; keeping the implementation in ``v2/etf/``
lets the byte-equal tests stay sandbox-runnable).

The public names use the ``format_ark_*`` prefix consistent with the
Phase 1 ``format_earnings_*`` / Phase 2 ``format_portfolio_*`` /
Phase 3 ``format_sec_*`` / Phase 4 ``format_macro_*`` conventions.
"""

from v2.etf._ark_alert_cards import (
    format_ark_alert,
    format_ark_summary,
)

__all__ = [
    "format_ark_alert",
    "format_ark_summary",
]
