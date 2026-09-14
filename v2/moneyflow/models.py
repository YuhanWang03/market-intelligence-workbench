"""Pydantic models for the money-flow divergence pipeline (⑱).

Divergence = money flow (proxied by Chaikin Money Flow) disagreeing with
price, filtered/graded by RSI position + momentum divergence. All numbers
are computed in Python; the LLM narrator only fills qualitative bull/bear
slots — same anti-hallucination contract as the screening narrator.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class DivergenceConfig(BaseModel):
    """Thresholds for the three-axis (price / CMF / RSI) divergence detector.

    Defaults are industry-standard starting points — expect to re-calibrate
    after a few live days. All windows are in trading days.
    """

    cmf_window: int = 20            # Chaikin Money Flow lookback
    rsi_window: int = 14            # Wilder RSI period
    price_window: int = 20          # window for the price trend + divergence

    # Money-flow axis — CMF sits in [-1, 1].
    cmf_inflow_threshold: float = 0.05    # CMF ≥ this → 净流入
    cmf_outflow_threshold: float = -0.05  # CMF ≤ this → 净流出

    # Price axis — |return_over_window| < flat_band → 横盘.
    flat_band: float = 0.03

    # Momentum axis — RSI zones (0-100).
    rsi_low: float = 45.0           # < this (and > oversold) → 低位区
    rsi_high: float = 55.0          # > this (and < overbought) → 高位区
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    rsi_divergence_delta: float = 3.0     # min RSI move to call a divergence

    min_history: int = 40           # need at least this many bars to judge


class MoneyFlowReading(BaseModel):
    """The three-axis read for one ticker, independent of any verdict.

    Always populated when there's enough data — used by the bot's on-demand
    query so a user who asks about a specific ticker sees the raw price /
    money-flow / momentum picture even when no divergence verdict fires.
    """

    ticker: str
    price: float

    price_return: float             # over price_window, fraction
    price_state: str                # "up" | "flat" | "down"

    cmf: float                      # [-1, 1]
    flow_state: str                 # "inflow" | "outflow" | "neutral"

    rsi: float                      # 0-100
    rsi_zone: str                   # oversold|low|neutral|high|overbought
    rsi_divergence: str             # "none" | "bullish" | "bearish"


class MoneyFlowSignal(MoneyFlowReading):
    """A reading that resolved to an accumulation/distribution verdict.

    Numbers come from the detector; bull/bear come from the narrator.
    """

    kind: str                       # "accumulation" | "distribution"
    strength: str                   # "strong" | "moderate"

    bull: str = ""                  # filled by narrator (pure logic, no numbers)
    bear: str = ""                  # filled by narrator (pure logic, no numbers)


class MoneyFlowResult(BaseModel):
    """Top-level result returned by run_moneyflow()."""

    date: str
    universe_size: int
    signals: list[MoneyFlowSignal] = Field(default_factory=list)
    fd_calls: int = 0
    llm_tokens: int = 0
