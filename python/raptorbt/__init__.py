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

import raptorbt._raptorbt as _native
from raptorbt import _guard
from raptorbt._bands import BollingerBands

# Money faults leave the extension as a panic; make them the structured
# ``money_overflow`` error at every entry point.
_guard.install(_native)

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
    AccountResult,
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
    BatchSingleItem,
    BatchSpreadItem,
    batch_single_backtest,
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
    vwap_session,
    supertrend,
    rolling_min,
    rolling_max,
    # Instrument market definitions
    InstrumentSpec,
    SessionSchedule,
    CorporateAction,
    adjust_history,
    # Option chain (pricing lives in raptorbt.pricing)
    OptionChain,
    # Fees as data and margin models
    FeeSchedule,
    RateCard,
    MarginModel,
    # Streaming indicators
    Indicator,
    # Bar aggregation
    BarAggregator,
    aggregate_bars,
    bars_from_ticks,
    # Resampling: ticks to bars, bars to coarser bars
    Resampler,
    ResampledBar,
    resample_bars,
    resample_trades,
    # Per-bar strategy session (class-based strategy contract)
    KernelSession,
    PortfolioSession,
    EngineEvent,
    PositionSnapshot,
    resolve_atr_period,
)
from raptorbt.strategy import (
    AccountSpec,
    Bar,
    run_portfolio_strategy,
    run_strategies,
    run_strategies_windowed,
    run_tick_strategy,
    TickStrategyStream,
    Window,
    WindowDone,
    LiveNode,
    ClosePosition,
    MarketOrder,
    Strategy,
    StrategyConfig,
    StrategyContext,
    run_strategy_backtest,
)
from raptorbt import errors
from raptorbt import report, research, sampling
from raptorbt.report import full_report
from raptorbt.capabilities import capabilities
from raptorbt.expressions import col, expr
from raptorbt.errors import (
    RaptorBTError,
    InvalidParameter,
    InvalidData,
    InvalidOrder,
    InvalidSubscription,
    InvalidType,
    UnsupportedOperation,
    StateError,
    SessionFinished,
    UnknownSymbol,
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

# Opt-in, off by default; when enabled it reports to the maintainers' own server unless
# you name another: see raptorbt.crash_report and the README.
from raptorbt.crash_report import enable_crash_reporting

__all__ = [
    # Research statistics and the full report
    "research",
    "report",
    "capabilities",
    "expr",
    "col",
    "full_report",
    "enable_crash_reporting",
    # Structured errors
    "errors",
    "RaptorBTError",
    "InvalidParameter",
    "InvalidData",
    "InvalidOrder",
    "InvalidSubscription",
    "InvalidType",
    "UnsupportedOperation",
    "StateError",
    "SessionFinished",
    "UnknownSymbol",
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
    "AccountResult",
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
    "BatchSingleItem",
    "BatchSpreadItem",
    "batch_single_backtest",
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
    "BollingerBands",
    "adx",
    "vwap",
    "vwap_session",
    "supertrend",
    "rolling_min",
    "rolling_max",
    # Per-bar strategy session (class-based strategy contract)
    "InstrumentSpec",
    "OptionChain",
    "SessionSchedule",
    "CorporateAction",
    "adjust_history",
    "FeeSchedule",
    "RateCard",
    "MarginModel",
    "Indicator",
    "BarAggregator",
    "aggregate_bars",
    "bars_from_ticks",
    "Resampler",
    "ResampledBar",
    "resample_bars",
    "resample_trades",
    "sampling",
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
    "AccountSpec",
    "run_portfolio_strategy",
    "run_strategies",
    "run_strategies_windowed",
    "Window",
    "WindowDone",
    "run_tick_strategy",
    "TickStrategyStream",
    "LiveNode",
    "run_strategy_backtest",
]



def __getattr__(name):
    if name == "PortfolioContext":
        from raptorbt import strategy

        return strategy.__getattr__(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
