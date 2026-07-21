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
from raptorbt.strategy.context import Bar, CompositeBar, StrategyContext
from raptorbt.strategy.orders import ClosePosition, MarketOrder
from raptorbt.strategy.portfolio_runner import PortfolioContext, run_portfolio_strategy
from raptorbt.strategy.runner import run_strategy_backtest

__all__ = [
    "Bar",
    "ClosePosition",
    "CompositeBar",
    "MarketOrder",
    "Strategy",
    "StrategyConfig",
    "PortfolioContext",
    "StrategyContext",
    "run_portfolio_strategy",
    "run_strategy_backtest",
]
