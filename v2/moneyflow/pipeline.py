"""Money-flow divergence orchestration: universe -> OHLCV -> detect -> narrate.

Daily-level (⑱). Prices come from Financial Datasets daily bars because CMF
needs volume on every bar (the yfinance EOD price_source used by the screener
isn't guaranteed to carry volume). The multi-day divergence read tolerates
FD's few-day settle lag on the most recent bar.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

from v2.data.client import FDClient
from v2.moneyflow.detector import detect_divergence
from v2.moneyflow.models import DivergenceConfig, MoneyFlowResult, MoneyFlowSignal
from v2.moneyflow.narrator import narrate

logger = logging.getLogger(__name__)

# ~120 calendar days comfortably yields the ~60 trading bars we want
# (cmf_window + price_window + rsi warmup, with headroom).
_HISTORY_CALENDAR_DAYS = 120


def run_moneyflow(
    tickers: list[str],
    fd_client: FDClient,
    config: DivergenceConfig,
    *,
    narrate_fn=narrate,
) -> MoneyFlowResult:
    """Scan *tickers* for price/money-flow/RSI divergence, narrate, and return.

    ``narrate_fn`` is injectable so tests can bypass the LLM. On any per-ticker
    data error we skip that ticker and keep going — a single bad symbol never
    sinks the batch.
    """
    today = date.today()
    today_str = today.isoformat()
    history_start = (today - timedelta(days=_HISTORY_CALENDAR_DAYS)).isoformat()

    initial_misses = getattr(fd_client, "misses", None)
    invocation_count = 0

    signals: list[MoneyFlowSignal] = []
    for ticker in tickers:
        invocation_count += 1
        try:
            prices = fd_client.get_prices(ticker, history_start, today_str)
        except Exception as exc:
            logger.warning("get_prices failed for %s: %s", ticker, exc)
            continue
        if not prices:
            continue
        signal = detect_divergence(ticker, prices, config)
        if signal is not None:
            signals.append(signal)

    llm_tokens = 0
    if signals:
        narrations, llm_tokens = narrate_fn(signals)
        for s in signals:
            note = narrations.get(s.ticker) or {}
            s.bull = note.get("bull", "")
            s.bear = note.get("bear", "")

    if initial_misses is not None:
        fd_calls = int(fd_client.misses) - int(initial_misses)
    else:
        fd_calls = invocation_count

    return MoneyFlowResult(
        date=today_str,
        universe_size=len(tickers),
        signals=signals,
        fd_calls=fd_calls,
        llm_tokens=llm_tokens,
    )
