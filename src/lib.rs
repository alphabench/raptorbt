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

pub mod core;
pub mod execution;
pub mod indicators;
pub mod metrics;
pub mod portfolio;
pub mod python;
pub mod signals;
pub mod stops;
pub mod strategies;

/// Python module entry point
#[pymodule]
fn _raptorbt(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    // Register config classes
    m.add_class::<python::bindings::PyBacktestConfig>()?;
    m.add_class::<python::bindings::PyInstrumentConfig>()?;
    m.add_class::<python::bindings::PyStopConfig>()?;
    m.add_class::<python::bindings::PyTargetConfig>()?;

    // Register result classes
    m.add_class::<python::bindings::PyBacktestResult>()?;
    m.add_class::<python::bindings::PyBacktestMetrics>()?;
    m.add_class::<python::bindings::PyTrade>()?;

    // Register backtest functions
    m.add_function(wrap_pyfunction!(python::bindings::run_single_backtest, m)?)?;
    m.add_function(wrap_pyfunction!(python::bindings::run_basket_backtest, m)?)?;
    m.add_function(wrap_pyfunction!(python::bindings::run_options_backtest, m)?)?;
    m.add_function(wrap_pyfunction!(python::bindings::run_pairs_backtest, m)?)?;
    m.add_function(wrap_pyfunction!(python::bindings::run_multi_backtest, m)?)?;
    m.add_function(wrap_pyfunction!(python::bindings::run_spread_backtest, m)?)?;

    // Register Monte Carlo simulation
    m.add_function(wrap_pyfunction!(python::bindings::simulate_portfolio_mc, m)?)?;

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
