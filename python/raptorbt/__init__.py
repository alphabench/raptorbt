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

from raptorbt._raptorbt import (
    # Session lengths (minutes) for PyBacktestConfig(session_minutes=...)
    SESSION_NSE,
    SESSION_MCX,
    SESSION_CDS,
    SESSION_CONTINUOUS,
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
)

try:  # Python 3.8+
    from importlib.metadata import PackageNotFoundError, version as _pkg_version

    __version__ = _pkg_version("raptorbt")
except Exception:  # pragma: no cover - source checkout without install metadata
    __version__ = "unknown"

__all__ = [
    # Session lengths
    "SESSION_NSE",
    "SESSION_MCX",
    "SESSION_CDS",
    "SESSION_CONTINUOUS",
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
]
