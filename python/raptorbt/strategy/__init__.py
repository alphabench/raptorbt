"""Class-based strategy contract for RaptorBT.

Strategies subclass :class:`Strategy`, override lifecycle hooks
(``on_start``, ``on_bar``, ``on_stop``) and order/position event hooks, and
emit order intents (:meth:`Strategy.enter`, :meth:`Strategy.close_position`).
The engine simulates fills and routes resulting events back into the hooks.

Run one with :func:`run_strategy_backtest`, which returns the same
``BacktestResult`` as the array-based runners.
"""

from raptorbt.strategy.accounts import AccountSpec
from raptorbt.strategy.continuous import RollEvent, RollEvents, continuous_series
from raptorbt.strategy.exits import ExitEvents, ExitHandle, ExitRule
from raptorbt.strategy.order_check import OrderProposal, OrderRefusal
from raptorbt.strategy.api import BarHistory, Timeframe
from raptorbt.strategy.base import Strategy
from raptorbt.strategy.tick_runner import run_tick_strategy
from raptorbt.strategy.tick_stream import TickStrategyStream
from raptorbt.strategy.live_node import LiveNode
from raptorbt.strategy.config import StrategyConfig
from raptorbt.strategy.context import (
    Bar,
    CompositeBar,
    QuoteTick,
    StrategyContext,
    TradeTick,
)
from raptorbt.strategy.orders import ClosePosition, MarketOrder
from raptorbt.strategy.multi_runner import Window, WindowDone, run_strategies, run_strategies_windowed
from raptorbt.strategy.portfolio_runner import run_portfolio_strategy
from raptorbt.strategy.requirements import (
    Bars,
    ByDelta,
    Chain,
    Duration,
    Expiry,
    Future,
    Holding,
    Instrument,
    InstrumentPlan,
    Issue,
    LegContracts,
    LegPlan,
    Legs,
    MaxPosition,
    OptionLeg,
    ParamSchema,
    ParamSpec,
    Requirements,
    RequirementsError,
    RolePlan,
    StrikeWindow,
    SymbolPlan,
    Underlying,
    Universe,
    check_requirements,
    param,
    params_schema,
    plan_instruments,
    requirements_hash,
    resolve,
    resolve_legs,
)
from raptorbt.strategy.runner import run_strategy_backtest

__all__ = [
    "Bars",
    "ByDelta",
    "Chain",
    "Duration",
    "Expiry",
    "Future",
    "Holding",
    "Instrument",
    "InstrumentPlan",
    "Issue",
    "LegContracts",
    "LegPlan",
    "Legs",
    "MaxPosition",
    "OptionLeg",
    "ParamSchema",
    "ParamSpec",
    "Requirements",
    "RequirementsError",
    "RolePlan",
    "StrikeWindow",
    "SymbolPlan",
    "Underlying",
    "Universe",
    "check_requirements",
    "param",
    "params_schema",
    "plan_instruments",
    "requirements_hash",
    "resolve",
    "resolve_legs",
    "AccountSpec",
    "ExitEvents",
    "ExitHandle",
    "ExitRule",
    "OrderProposal",
    "OrderRefusal",
    "RollEvent",
    "RollEvents",
    "continuous_series",
    "Bar",
    "BarHistory",
    "ClosePosition",
    "CompositeBar",
    "QuoteTick",
    "TradeTick",
    "run_tick_strategy",
    "TickStrategyStream",
    "LiveNode",
    "MarketOrder",
    "Strategy",
    "StrategyConfig",
    "StrategyContext",
    "Timeframe",
    "run_portfolio_strategy",
    "run_strategies",
    "run_strategies_windowed",
    "run_strategy_backtest",
    "Window",
    "WindowDone",
]


def __getattr__(name):
    # PortfolioContext was the portfolio runner's context class; that runner
    # now runs on the shared runtime, whose context is RuntimeContext.
    if name == "PortfolioContext":
        import warnings

        from raptorbt._raptorbt import RuntimeContext

        warnings.warn(
            "raptorbt.strategy.PortfolioContext is deprecated, an alias kept for old code; "
            "portfolio strategies now receive a raptorbt.RuntimeContext",
            DeprecationWarning,
            stacklevel=2,
        )
        return RuntimeContext
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
