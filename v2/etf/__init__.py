"""ARK Invest daily ETF holdings tracker.

Why this module: 13F filings are quarterly + 45-day delayed. ARK publishes
holdings every trading day via public CSV. Combining the two gives:

  - Quarterly archaeology (13F): "what did ARK build over Q1?"
  - Daily live signal (this module): "did Cathie just sell PLTR?"

Data flow:
  client.py     fetch CSV from ARK's CDN, parse defensively
  tracker.py    SQLite snapshots — one row per (fund, date, cusip)
  detector.py   diff today vs yesterday → ETFChange records
  models.py     ETFHolding, ETFChange
"""

from v2.etf.client import SUPPORTED_FUNDS, fetch_holdings
from v2.etf.detector import compute_daily_changes
from v2.etf.models import ETFChange, ETFHolding
from v2.etf.tracker import (
    get_latest_snapshot_before,
    get_snapshot,
    save_snapshot,
)

__all__ = [
    "ETFChange",
    "ETFHolding",
    "SUPPORTED_FUNDS",
    "compute_daily_changes",
    "fetch_holdings",
    "get_latest_snapshot_before",
    "get_snapshot",
    "save_snapshot",
]
