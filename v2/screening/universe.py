"""Hardcoded ticker universes for screening.

We don't pull S&P 500 membership dynamically because FD doesn't expose
that endpoint and we want the universe to be deterministic / auditable.

Update this list quarterly or when adding sectors.
"""

# 29 large/mid-cap US tech tickers — mix of mega-cap, semis, software, internet
TECH_30: list[str] = [
    # Mega-cap (7)
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA",
    # Semiconductors (6 — ASML dropped, not covered by FD)
    "AMD", "AVGO", "QCOM", "INTC", "TXN", "MU",
    # Software / SaaS (9)
    "ORCL", "CRM", "ADBE", "NOW", "SNOW", "PLTR", "CRWD", "PANW", "DDOG",
    # Internet / consumer tech (3)
    "NFLX", "SHOP", "UBER",
    # Other tech (4)
    "IBM", "CSCO", "INTU", "ANET",
]
