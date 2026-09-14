"""Cost estimation for LLM and search API calls.

Numbers reflect public list prices as of the dashboard build. They are
intentionally simple — the dashboard uses them for two things:
1. Per-session running total displayed in the trace panel.
2. Global daily budget enforcement for guest visitors.

Real per-month operating cost on the live VPS is dominated by FD subscription
(flat) and DeepSeek tokens; per-query cost almost always lands under $0.01.
"""

from __future__ import annotations


# Per-1K-token prices in USD.
_DEEPSEEK_INPUT_PER_1K = 0.00014
_DEEPSEEK_OUTPUT_PER_1K = 0.00028

# Tavily charges per search request on paid tiers; the README cites the
# project sitting in the free tier. Use the lowest paid tier for accounting
# so the dashboard's budget gauge is conservative.
_TAVILY_PER_SEARCH = 0.005

# OpenAI text-embedding-3-small.
_OPENAI_EMBED_PER_1K = 0.00002


def deepseek_cost(input_tokens: int, output_tokens: int) -> float:
    return (
        input_tokens * _DEEPSEEK_INPUT_PER_1K / 1000.0
        + output_tokens * _DEEPSEEK_OUTPUT_PER_1K / 1000.0
    )


def tavily_cost(num_searches: int = 1) -> float:
    return num_searches * _TAVILY_PER_SEARCH


def openai_embed_cost(tokens: int) -> float:
    return tokens * _OPENAI_EMBED_PER_1K / 1000.0


# Rough per-intent estimates used at POST /api/query time to reserve budget
# before the actual run finishes. Tuned from production logs.
INTENT_ESTIMATES_USD: dict[str, float] = {
    "explain_move": 0.012,       # FD prices + Tavily + 2x DeepSeek
    "summary": 0.008,            # FD multi-endpoint + 1x DeepSeek narration
    "chain": 0.015,              # DeepSeek expand + Tavily verify per neighbor
    "thirteen_f": 0.003,         # EDGAR + 1x small DeepSeek
    "holders_view": 0.001,       # Pure SQLite lookup
    "etf_view": 0.001,           # Local ETF db
    "alert_set": 0.0005,
    "alert_list": 0.0005,
    "alert_remove": 0.0005,
    "portfolio_view": 0.001,
    "pnl_view": 0.001,
    "watchlist_view": 0.0005,
    "find_anomalies": 0.002,
    "unknown": 0.001,
}


def estimate_cost(intent_name: str) -> float:
    """Return a budget-reservation estimate for the given intent.

    Unknown intents fall back to a small positive number so they still
    debit the rate counter; never returns 0.
    """
    return INTENT_ESTIMATES_USD.get(intent_name, 0.005)
