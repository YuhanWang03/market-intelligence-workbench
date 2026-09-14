"""v2 backtesting — pluggable strategies and simulation engine."""

from v2.backtesting.engine import BacktestEngine, yearly_breakdown
from v2.backtesting.models import (
    BacktestResult,
    PerformanceMetrics,
    Trade,
    TradeSignal,
)
from v2.backtesting.strategies import STRATEGY_KEYS, BacktestData, CommitteeStrategy, InsiderClusterStrategy, MomentumStrategy, PriceCache
from v2.backtesting.strategy import PEADStrategy, Strategy

__all__ = [
    "BacktestData",
    "BacktestEngine",
    "CommitteeStrategy",
    "InsiderClusterStrategy",
    "MomentumStrategy",
    "PriceCache",
    "STRATEGY_KEYS",
    "BacktestResult",
    "PerformanceMetrics",
    "PEADStrategy",
    "Strategy",
    "Trade",
    "TradeSignal",
    "yearly_breakdown",
]
