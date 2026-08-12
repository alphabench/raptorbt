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
    # Session lengths (minutes) for BacktestConfig(session_minutes=...)
    SESSION_NSE,
    SESSION_MCX,
    SESSION_CDS,
    SESSION_CONTINUOUS,
    IST_OFFSET_NS,
    # Config classes
    BacktestConfig,
    InstrumentConfig,
    StopConfig,
    TargetConfig,
    # Result classes
    BacktestResult,
    BacktestMetrics,
    Trade,
    PortfolioResult,
    InstrumentSummary,
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
    BatchSpreadItem,
    batch_spread_backtest,
    # Monte Carlo simulation
    simulate_portfolio_mc,
    # Portfolio math (covariance, optimizer, factor panels, risk
    # contributions, rebalance simulation, cost schedule)
    RiskModel,
    OptimizerConfig,
    OptimizationResult,
    RiskContributions,
    OptimizeItem,
    RebalanceSimResult,
    RankIC,
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
    KernelSession,
    PortfolioSession,
    EngineEvent,
    PositionSnapshot,
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
    "BacktestConfig",
    "InstrumentConfig",
    "StopConfig",
    "TargetConfig",
    # Result classes
    "PortfolioResult",
    "InstrumentSummary",
    "BacktestResult",
    "BacktestMetrics",
    "Trade",
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
    "BatchSpreadItem",
    "batch_spread_backtest",
    # Portfolio math
    "RiskModel",
    "OptimizerConfig",
    "OptimizationResult",
    "RiskContributions",
    "OptimizeItem",
    "RebalanceSimResult",
    "RankIC",
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
    "KernelSession",
    "PortfolioSession",
    "EngineEvent",
    "PositionSnapshot",
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
# Rust-side disambiguator -- the crate has its own ``BacktestConfig``, ``Trade``
# and ``BacktestResult`` in ``src/core`` and two Rust types cannot share a name
# -- and it was never meant to cross into Python, where "the Python one"
# describes every object in the library. The clean name is canonical from
# 0.7.0; the old one keeps working for this minor version, and is removed in
# 0.8.0.
#
# Deliberately absent from ``__all__``: old names still resolve, but they are no
# longer advertised. Pinned by tests/python/test_deprecated_names.py.
_RENAMED = {
    "PyBacktestConfig": "BacktestConfig",
    "PyBacktestMetrics": "BacktestMetrics",
    "PyBacktestResult": "BacktestResult",
    "PyBatchSpreadItem": "BatchSpreadItem",
    "PyEngineEvent": "EngineEvent",
    "PyInstrumentConfig": "InstrumentConfig",
    "PyInstrumentSummary": "InstrumentSummary",
    "PyKernelSession": "KernelSession",
    "PyOptimizationResult": "OptimizationResult",
    "PyOptimizeItem": "OptimizeItem",
    "PyOptimizerConfig": "OptimizerConfig",
    "PyPortfolioResult": "PortfolioResult",
    "PyPortfolioSession": "PortfolioSession",
    "PyPositionSnapshot": "PositionSnapshot",
    "PyRankIc": "RankIC",
    "PyRebalanceSimResult": "RebalanceSimResult",
    "PyRiskContributions": "RiskContributions",
    "PyRiskModel": "RiskModel",
    "PyStopConfig": "StopConfig",
    "PyTargetConfig": "TargetConfig",
    "PyTrade": "Trade",
}


def _warn_renamed(module: str, name: str, new_name: str) -> None:
    warnings.warn(
        f"{module}.{name} is deprecated; use raptorbt.{new_name}. "
        f"The old name is removed in 0.8.0.",
        DeprecationWarning,
        stacklevel=3,
    )


def __getattr__(name: str):
    """Resolve a pre-0.7.0 ``Py``-prefixed name, warning that it is going away."""
    new_name = _RENAMED.get(name)
    if new_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    _warn_renamed(__name__, name, new_name)
    return globals()[new_name]


def _install_extension_shim() -> None:
    """Extend the deprecation shim to ``raptorbt._raptorbt``.

    Some code reaches past the package into the compiled module -- notably for
    ``PortfolioSession``, which has never been re-exported at top level, so a
    deep import was the *only* way to get it. Those imports must keep working
    for the same window as the public ones, or the rename breaks people who had
    no supported alternative.

    PyO3 builds the extension as a plain module object, which has no class-level
    ``__getattr__`` to override, so the module-level hook is installed here.

    ``__dir__`` is installed alongside it. A name that resolves but does not
    appear in ``dir()`` is invisible to tooling: the consuming backend has a
    guard comparing ``_raptorbt.pyi`` against ``dir(_raptorbt)``, and the stub's
    alias block looked to it like 21 declarations for symbols the engine had
    dropped -- the exact "type-checks clean, AttributeError in prod" shape that
    guard exists to catch. The aliases are real, so they should be listed.
    """
    from raptorbt import _raptorbt

    def _ext_getattr(name: str):
        new_name = _RENAMED.get(name)
        if new_name is None:
            raise AttributeError(
                f"module {_raptorbt.__name__!r} has no attribute {name!r}"
            )
        _warn_renamed(_raptorbt.__name__, name, new_name)
        return getattr(_raptorbt, new_name)

    _base_dir = sorted(vars(_raptorbt))

    def _ext_dir() -> list[str]:
        return sorted({*_base_dir, *_RENAMED})

    _raptorbt.__getattr__ = _ext_getattr
    _raptorbt.__dir__ = _ext_dir


_install_extension_shim()


def __dir__() -> list[str]:
    return sorted([*__all__, *_RENAMED])
