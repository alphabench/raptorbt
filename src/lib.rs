// Suppress warning from PyO3 macro expansion (fixed in newer PyO3 versions)
#![allow(non_local_definitions)]

//! RaptorBT - High-performance Rust backtesting engine.
//!
//! This crate provides a complete backtesting solution with:
//! - Technical indicators (SMA, EMA, RSI, MACD, etc.)
//! - Portfolio simulation engine
//! - Multiple strategy types (single, basket, options, pairs, multi)
//! - Stop-loss and take-profit mechanisms
//! - Streaming metrics calculation

use pyo3::prelude::*;

pub mod accounts;
pub mod core;
pub mod data;
pub mod execution;
pub mod indicators;
pub mod instruments;
pub mod metrics;
pub mod portfolio;
pub mod python;
pub mod signals;
pub mod stops;
pub mod strategies;

/// Python module entry point
#[pymodule]
fn _raptorbt(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    // Session lengths in minutes, for BacktestConfig(session_minutes=...).
    // Intraday annualization scales with session length, so using the NSE
    // default on MCX data understates Sharpe by ~1.5x.
    m.add("SESSION_NSE", 375.0)?; // 09:15-15:30 equity / F&O
    m.add("SESSION_MCX", 870.0)?; // 09:00-23:30 commodity
    m.add("SESSION_CDS", 480.0)?; // 09:00-17:00 currency
    m.add("SESSION_CONTINUOUS", 0.0)?; // 24x7, annualize on calendar time

    // Timezone offset (ns) for session-aligned day/week/month/year bars.
    m.add("IST_OFFSET_NS", crate::data::IST_OFFSET_NS)?;

    // Register config classes
    m.add_class::<python::bindings::PyBacktestConfig>()?;
    m.add_class::<python::bindings::PyInstrumentConfig>()?;
    m.add_class::<python::bindings::PyStopConfig>()?;
    m.add_class::<python::bindings::PyTargetConfig>()?;

    // Register result classes
    m.add_class::<python::bindings::PyBacktestResult>()?;
    m.add_class::<python::bindings::PyBacktestMetrics>()?;
    m.add_class::<python::bindings::PyTrade>()?;
    m.add_class::<python::bindings::PyPortfolioResult>()?;
    m.add_class::<python::bindings::PyInstrumentSummary>()?;

    // Register backtest functions
    m.add_function(wrap_pyfunction!(python::bindings::run_single_backtest, m)?)?;
    m.add_function(wrap_pyfunction!(python::bindings::run_basket_backtest, m)?)?;
    m.add_function(wrap_pyfunction!(python::bindings::run_portfolio_backtest, m)?)?;
    m.add_function(wrap_pyfunction!(python::bindings::run_options_backtest, m)?)?;
    m.add_function(wrap_pyfunction!(python::bindings::run_pairs_backtest, m)?)?;
    m.add_function(wrap_pyfunction!(python::bindings::run_multi_backtest, m)?)?;
    m.add_function(wrap_pyfunction!(python::bindings::run_spread_backtest, m)?)?;
    m.add_function(wrap_pyfunction!(python::bindings::run_tick_backtest, m)?)?;

    // Register instrument market definitions
    m.add_class::<python::instrument_bindings::PyInstrumentSpec>()?;

    // Register streaming indicators
    m.add_class::<python::indicator_bindings::PyIndicator>()?;

    // Register bar aggregation
    m.add_class::<python::data_bindings::PyBarAggregator>()?;
    m.add_function(wrap_pyfunction!(python::data_bindings::aggregate_bars, m)?)?;
    m.add_function(wrap_pyfunction!(python::data_bindings::bars_from_ticks, m)?)?;

    // Register the per-bar strategy session (class-based strategy contract)
    m.add_class::<python::strategy_bindings::PyKernelSession>()?;
    m.add_class::<python::session_bindings::PyPortfolioSession>()?;
    m.add_class::<python::strategy_bindings::PyEngineEvent>()?;
    m.add_class::<python::strategy_bindings::PyPositionSnapshot>()?;
    m.add_function(wrap_pyfunction!(python::strategy_bindings::resolve_atr_period, m)?)?;

    // Register batch spread backtest
    m.add_class::<python::bindings::PyBatchSpreadItem>()?;
    m.add_function(wrap_pyfunction!(python::bindings::batch_spread_backtest, m)?)?;

    // Register Monte Carlo simulation
    m.add_function(wrap_pyfunction!(python::bindings::simulate_portfolio_mc, m)?)?;

    // Register portfolio math (covariance, optimizer, factor panels, risk
    // contributions, rebalance simulation, cost schedule export)
    m.add_class::<python::portfolio_bindings::PyRiskModel>()?;
    m.add_class::<python::portfolio_bindings::PyOptimizerConfig>()?;
    m.add_class::<python::portfolio_bindings::PyOptimizationResult>()?;
    m.add_class::<python::portfolio_bindings::PyRiskContributions>()?;
    m.add_class::<python::portfolio_bindings::PyOptimizeItem>()?;
    m.add_class::<python::portfolio_bindings::PyRebalanceSimResult>()?;
    m.add_class::<python::portfolio_bindings::PyRankIc>()?;
    m.add_function(wrap_pyfunction!(python::portfolio_bindings::estimate_covariance, m)?)?;
    m.add_function(wrap_pyfunction!(python::portfolio_bindings::optimize_portfolio, m)?)?;
    m.add_function(wrap_pyfunction!(python::portfolio_bindings::batch_optimize_portfolios, m)?)?;
    m.add_function(wrap_pyfunction!(python::portfolio_bindings::compute_risk_contributions, m)?)?;
    m.add_function(wrap_pyfunction!(python::portfolio_bindings::winsorize_panel, m)?)?;
    m.add_function(wrap_pyfunction!(python::portfolio_bindings::zscore_panel, m)?)?;
    m.add_function(wrap_pyfunction!(python::portfolio_bindings::rank_panel, m)?)?;
    m.add_function(wrap_pyfunction!(python::portfolio_bindings::momentum_panel, m)?)?;
    m.add_function(wrap_pyfunction!(python::portfolio_bindings::composite_scores, m)?)?;
    m.add_function(wrap_pyfunction!(python::portfolio_bindings::rank_ic, m)?)?;
    m.add_function(wrap_pyfunction!(python::portfolio_bindings::simulate_rebalance_policy, m)?)?;
    m.add_function(wrap_pyfunction!(python::portfolio_bindings::indian_cost_schedule, m)?)?;

    // Register tick signal functions
    m.add_function(wrap_pyfunction!(python::bindings::compute_tick_entry_signals, m)?)?;
    m.add_function(wrap_pyfunction!(python::bindings::compute_tick_exit_signals, m)?)?;

    // Register tick feature functions
    m.add_function(wrap_pyfunction!(python::bindings::tick_spread_pct, m)?)?;
    m.add_function(wrap_pyfunction!(python::bindings::buy_sell_imbalance_delta, m)?)?;
    m.add_function(wrap_pyfunction!(python::bindings::return_window, m)?)?;
    m.add_function(wrap_pyfunction!(python::bindings::realized_vol_rolling, m)?)?;
    m.add_function(wrap_pyfunction!(python::bindings::oi_position_pct, m)?)?;
    m.add_function(wrap_pyfunction!(python::bindings::tick_velocity, m)?)?;

    // Register indicator functions
    m.add_function(wrap_pyfunction!(python::bindings::sma, m)?)?;
    m.add_function(wrap_pyfunction!(python::bindings::ema, m)?)?;
    m.add_function(wrap_pyfunction!(python::bindings::rsi, m)?)?;
    m.add_function(wrap_pyfunction!(python::bindings::macd, m)?)?;
    m.add_function(wrap_pyfunction!(python::bindings::stochastic, m)?)?;
    m.add_function(wrap_pyfunction!(python::bindings::atr, m)?)?;
    m.add_function(wrap_pyfunction!(python::bindings::bollinger_bands, m)?)?;
    m.add_function(wrap_pyfunction!(python::bindings::adx, m)?)?;
    m.add_function(wrap_pyfunction!(python::bindings::vwap, m)?)?;
    m.add_function(wrap_pyfunction!(python::bindings::supertrend, m)?)?;
    m.add_function(wrap_pyfunction!(python::bindings::rolling_min, m)?)?;
    m.add_function(wrap_pyfunction!(python::bindings::rolling_max, m)?)?;

    Ok(())
}
