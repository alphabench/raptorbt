"""Class-based strategy contract for RaptorBT.

Strategies subclass :class:`Strategy`, override lifecycle hooks
(``on_start``, ``on_bar``, ``on_stop``) and order/position event hooks, and
emit order intents (:meth:`Strategy.enter`, :meth:`Strategy.close_position`).
The engine simulates fills and routes resulting events back into the hooks.

Run one with :func:`run_strategy_backtest`, which returns the same
``PyBacktestResult`` as the array-based runners.
"""

from raptorbt.strategy.base import Strategy
from raptorbt.strategy.config import StrategyConfig
from raptorbt.strategy.context import Bar, StrategyContext
from raptorbt.strategy.orders import ClosePosition, MarketOrder
from raptorbt.strategy.runner import run_strategy_backtest

__all__ = [
    "Bar",
    "ClosePosition",
    "MarketOrder",
    "Strategy",
    "StrategyConfig",
    "StrategyContext",
    "run_strategy_backtest",
]
