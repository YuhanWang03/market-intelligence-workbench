"""Compatibility re-export for the public v2.reporting namespace.

Single source of truth lives in ``v2/sec/_bot_cards.py`` — see that
module for the rationale (v2.reporting's package init pulls in
matplotlib + v2.lateral + v2.backtesting, which transitively requires
the production-only v2.data; keeping the implementation in ``v2/sec/``
lets the byte-equal tests stay sandbox-runnable).

The public names use the ``format_sec_*`` prefix to match the
Phase 1 ``format_earnings_*`` and Phase 2 ``format_portfolio_*``
conventions.
"""

from v2.sec._bot_cards import (
    format_sec_8k_card,
    format_sec_8k_view,
    format_sec_form4_cluster_card,
    format_sec_form4_individual_card,
    format_sec_form4_view,
    format_sec_insider_digest,
)

__all__ = [
    "format_sec_8k_card",
    "format_sec_8k_view",
    "format_sec_form4_cluster_card",
    "format_sec_form4_individual_card",
    "format_sec_form4_view",
    "format_sec_insider_digest",
]
