"""Streamer service — intraday polling for price-alert triggers.

Runs as a separate systemd service (hedge-fund-streamer.service). Polls
every minute during US market hours, fetches the current price for each
ticker that has open alerts, fires triggered alerts via Telegram, and
marks them in the SQLite alerts table.

Designed to be:
  - SAFE: read-mostly. Only writes to alerts.fired_at via the atomic
          UPDATE … WHERE fired_at IS NULL guard.
  - CHEAP: only polls tickers that actually have open alerts (none =
          one SELECT per minute, zero FD/Alpaca calls).
  - GRACEFUL: outside market hours it sleeps without polling.
"""

from v2.streamer.runner import run_streamer

__all__ = ["run_streamer"]
