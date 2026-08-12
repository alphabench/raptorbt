"""
RaptorBT - High-performance Rust backtesting engine.

Provides Python bindings for a Rust-based backtesting engine built for
production quantitative trading:
- Sub-millisecond execution on thousands of bars
- Disk footprint: <10MB, startup latency: <10ms
- 100% deterministic execution (no JIT cache)
- Native parallelism via Rayon + explicit SIMD
- Full tick-level simulation (no bar resampling required)
"""

import warnings

from raptorbt._raptorbt import (
    # Session lengths (minutes) for PyBacktestConfig(session_minutes=...)
    SESSION_NSE,
    SESSION_MCX,
    SESSION_CDS,
    SESSION_CONTINUOUS,
    IST_OFFSET_NS,
    # Config classes
    PyBacktestConfig,
    PyInstrumentConfig,
    PyStopConfig,
    PyTargetConfig,
    # Result classes
    PyBacktestResult,
    PyBacktestMetrics,
    PyTrade,
    PyPortfolioResult,
    PyInstrumentSummary,
    # Backtest functions
    run_single_backtest,
    run_basket_backtest,
    run_portfolio_backtest,
    run_options_backtest,
    run_pairs_backtest,
    run_multi_backtest,
    run_spread_backtest,
    run_tick_backtest,
    # Batch backtest
    PyBatchSpreadItem,
    batch_spread_backtest,
    # Monte Carlo simulation
    simulate_portfolio_mc,
    # Portfolio math (covariance, optimizer, factor panels, risk
    # contributions, rebalance simulation, cost schedule)
    RiskModel,
    PyOptimizerConfig,
    PyOptimizationResult,
    PyRiskContributions,
    PyOptimizeItem,
    PyRebalanceSimResult,
    PyRankIc,
    estimate_covariance,
    optimize_portfolio,
    batch_optimize_portfolios,
    compute_risk_contributions,
    winsorize_panel,
    zscore_panel,
    rank_panel,
    momentum_panel,
    composite_scores,
    rank_ic,
    simulate_rebalance_policy,
    indian_cost_schedule,
    # Tick signal functions
    compute_tick_entry_signals,
    compute_tick_exit_signals,
    # Tick feature functions
    tick_spread_pct,
    buy_sell_imbalance_delta,
    return_window,
    realized_vol_rolling,
    oi_position_pct,
    tick_velocity,
    # Indicator functions
    sma,
    ema,
    rsi,
    macd,
    stochastic,
    atr,
    bollinger_bands,
    adx,
    vwap,
    supertrend,
    rolling_min,
    rolling_max,
    # Instrument market definitions
    InstrumentSpec,
    # Streaming indicators
    Indicator,
    # Bar aggregation
    BarAggregator,
    aggregate_bars,
    bars_from_ticks,
    # Per-bar strategy session (class-based strategy contract)
    PyKernelSession,
    PyEngineEvent,
    PyPositionSnapshot,
    resolve_atr_period,
)
from raptorbt.strategy import (
    Bar,
    PortfolioContext,
    run_portfolio_strategy,
    run_tick_strategy,
    TickStrategyStream,
    ClosePosition,
    MarketOrder,
    Strategy,
    StrategyConfig,
    StrategyContext,
    run_strategy_backtest,
)

try:  # Python 3.8+
    from importlib.metadata import PackageNotFoundError, version as _pkg_version

    __version__ = _pkg_version("raptorbt")
except Exception:  # pragma: no cover - source checkout without install metadata
    __version__ = "unknown"

# Tell the log, at most once a day, if this install is behind the latest
# release. Runs on a daemon thread and swallows every failure, so it cannot
# delay or break this import. Opt out with RAPTORBT_NO_VERSION_CHECK=1.
from raptorbt.version_check import check_for_update as _check_for_update

_check_for_update(__version__)

__all__ = [
    # Session lengths
    "SESSION_NSE",
    "SESSION_MCX",
    "SESSION_CDS",
    "SESSION_CONTINUOUS",
    "IST_OFFSET_NS",
    # Config classes
    "PyBacktestConfig",
    "PyInstrumentConfig",
    "PyStopConfig",
    "PyTargetConfig",
    # Result classes
    "PyPortfolioResult",
    "PyInstrumentSummary",
    "PyBacktestResult",
    "PyBacktestMetrics",
    "PyTrade",
    # Backtest functions
    "run_single_backtest",
    "run_basket_backtest",
    "run_portfolio_backtest",
    "run_options_backtest",
    "run_pairs_backtest",
    "run_multi_backtest",
    "run_spread_backtest",
    "run_tick_backtest",
    # Batch backtest
    "PyBatchSpreadItem",
    "batch_spread_backtest",
    # Portfolio math
    "RiskModel",
    "PyOptimizerConfig",
    "PyOptimizationResult",
    "PyRiskContributions",
    "PyOptimizeItem",
    "PyRebalanceSimResult",
    "PyRankIc",
    "estimate_covariance",
    "optimize_portfolio",
    "batch_optimize_portfolios",
    "compute_risk_contributions",
    "winsorize_panel",
    "zscore_panel",
    "rank_panel",
    "momentum_panel",
    "composite_scores",
    "rank_ic",
    "simulate_rebalance_policy",
    "indian_cost_schedule",
    # Monte Carlo simulation
    "simulate_portfolio_mc",
    # Tick signal functions
    "compute_tick_entry_signals",
    "compute_tick_exit_signals",
    # Tick feature functions
    "tick_spread_pct",
    "buy_sell_imbalance_delta",
    "return_window",
    "realized_vol_rolling",
    "oi_position_pct",
    "tick_velocity",
    # Indicator functions
    "sma",
    "ema",
    "rsi",
    "macd",
    "stochastic",
    "atr",
    "bollinger_bands",
    "adx",
    "vwap",
    "supertrend",
    "rolling_min",
    "rolling_max",
    # Per-bar strategy session (class-based strategy contract)
    "InstrumentSpec",
    "Indicator",
    "BarAggregator",
    "aggregate_bars",
    "bars_from_ticks",
    "PyKernelSession",
    "PyEngineEvent",
    "PyPositionSnapshot",
    "resolve_atr_period",
    # Class-based strategy contract
    "Bar",
    "ClosePosition",
    "MarketOrder",
    "Strategy",
    "StrategyConfig",
    "StrategyContext",
    "PortfolioContext",
    "run_portfolio_strategy",
    "run_tick_strategy",
    "TickStrategyStream",
    "run_strategy_backtest",
]


# --- Deprecated names --------------------------------------------------------
# Every class below was exposed to Python with a ``Py`` prefix. That prefix is a
# Rust-side disambiguator -- the crate has its own ``RiskModel``, ``Trade`` and
# ``BacktestConfig`` in ``src/core`` and two Rust types cannot share a name --
# and it was never meant to cross into Python, where "the Python one" describes
# every object in the library. The clean name is canonical from 0.7.0; the old
# one keeps working for this minor version and is removed in 0.8.0.
#
# Deliberately absent from ``__all__``: old names still resolve, but they are no
# longer advertised. Pinned by tests/python/test_deprecated_names.py.
_RENAMED = {
    "PyRiskModel": "RiskModel",
}


def __getattr__(name: str):
    """Resolve a pre-0.7.0 ``Py``-prefixed name, warning that it is going away."""
    new_name = _RENAMED.get(name)
    if new_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    warnings.warn(
        f"raptorbt.{name} is deprecated; use raptorbt.{new_name}. "
        f"The old name is removed in 0.8.0.",
        DeprecationWarning,
        stacklevel=2,
    )
    return globals()[new_name]


def __dir__() -> list[str]:
    return sorted([*__all__, *_RENAMED])
