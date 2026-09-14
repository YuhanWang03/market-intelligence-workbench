"""Compatibility re-export for the public v2.reporting namespace.

Single source of truth lives in ``v2/earnings/_bot_cards.py`` since Stage 5
— see that file for the rationale (v2.reporting's package init pulls in
matplotlib + v2.backtesting + v2.monitoring, which transitively requires
the production-only v2.data; keeping the implementation here would force
all unit tests to either ship v2.data or bypass the package init via
importlib.util).
"""

from v2.earnings._bot_cards import (
    format_earnings_calendar,
    format_earnings_pending,
    format_earnings_reminder,
    format_earnings_summary,
    format_earnings_view,
)

__all__ = [
    "format_earnings_calendar",
    "format_earnings_pending",
    "format_earnings_reminder",
    "format_earnings_summary",
    "format_earnings_view",
]
