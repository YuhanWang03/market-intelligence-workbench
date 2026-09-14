"""Money-flow divergence (⑱) — CMF/RSI vs price, daily-level.

Detects accumulation (疑似吸筹) / distribution (疑似派发) by comparing a
volume-weighted money-flow proxy (Chaikin Money Flow) against price trend,
graded by RSI position + momentum divergence. Verdict + numbers are pure
Python; the LLM narrator only fills qualitative bull/bear slots.
"""

from v2.moneyflow.cards import format_signal_card, format_view_card
from v2.moneyflow.detector import detect_divergence, read_axes
from v2.moneyflow.indicators import (
    chaikin_money_flow,
    rsi,
    rsi_series,
    window_return,
)
from v2.moneyflow.models import (
    DivergenceConfig,
    MoneyFlowReading,
    MoneyFlowResult,
    MoneyFlowSignal,
)
from v2.moneyflow.narrator import narrate
from v2.moneyflow.pipeline import run_moneyflow

DEFAULT_CONFIG = DivergenceConfig()

__all__ = [
    "DEFAULT_CONFIG",
    "DivergenceConfig",
    "MoneyFlowReading",
    "MoneyFlowResult",
    "MoneyFlowSignal",
    "chaikin_money_flow",
    "detect_divergence",
    "format_signal_card",
    "format_view_card",
    "narrate",
    "read_axes",
    "rsi",
    "rsi_series",
    "run_moneyflow",
    "window_return",
]
