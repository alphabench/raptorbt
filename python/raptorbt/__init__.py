"""
RaptorBT - High-performance Rust backtesting engine.

This module provides Python bindings for the Rust-based backtesting engine,
offering significant performance improvements over vectorbt:
- Disk footprint: <10MB (vs vectorbt's ~450MB)
- Startup latency: <10ms (vs 200-600ms)
- 100% deterministic execution (no JIT cache)
- Native parallelism via Rayon + explicit SIMD
"""

from raptorbt._raptorbt import (
    # Config classes
    PyBacktestConfig,
    PyStopConfig,
    PyTargetConfig,
    # Result classes
    PyBacktestResult,
    PyBacktestMetrics,
    PyTrade,
    # Backtest functions
    run_single_backtest,
    run_basket_backtest,
    run_options_backtest,
    run_pairs_backtest,
    run_multi_backtest,
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
)

__version__ = "0.2.0"

__all__ = [
    # Config classes
    "PyBacktestConfig",
    "PyStopConfig",
    "PyTargetConfig",
    # Result classes
    "PyBacktestResult",
    "PyBacktestMetrics",
    "PyTrade",
    # Backtest functions
    "run_single_backtest",
    "run_basket_backtest",
    "run_options_backtest",
    "run_pairs_backtest",
    "run_multi_backtest",
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
]
