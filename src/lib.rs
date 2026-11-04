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
pub mod capabilities;
pub mod core;
pub mod data;
pub mod execution;
pub mod indicators;
pub mod instruments;
pub mod journal;
pub mod math;
pub mod metrics;
pub mod portfolio;
pub mod pricing;
pub mod python;
pub mod research;
pub mod runtime;
pub mod signals;
pub mod strategies;

/// The source file and line of the most recent raptorbt engine panic in this process, from any
/// thread (a panic in a worker thread is re-raised on the caller's). Never the panic message.
/// Written only for a panic that is a crash: not by the money / fee-schedule faults, which are
/// structured errors. Taken (cleared) on read.
static LAST_PANIC_LOCATION: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);

/// `file:line` of the most recent engine panic, or None; clears it, so a later panic that is not
/// raptorbt's cannot inherit it. Read by `raptorbt.crash_report`.
#[pyfunction]
fn _last_panic_location() -> Option<String> {
    LAST_PANIC_LOCATION.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).take()
}

/// A money fault is a structured error in Python (`money_overflow`), as is a fee
/// schedule gap (`fee_schedule_gap`), each raised through a panic only to unwind out of the kernel. Its panic notice would be
/// a second, unstructured report of the same failure on stderr, so the hook
/// skips it and leaves every other panic to the hook that was there before.
fn silence_money_fault_notices() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            let payload = info.payload();
            let message = payload
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| payload.downcast_ref::<&str>().copied());
            if message.is_some_and(|m| {
                m.starts_with(accounts::money::MONEY_FAULT_PREFIX)
                    || m.starts_with(execution::fees::FEE_GAP_PREFIX)
            }) {
                return;
            }
            // Where, never what: the opt-in crash reporter reads this file:line, and the
            // message (which can hold prices or symbols) is not kept. Only a panic that is a
            // crash gets here. The lock is never held across code that can panic.
            if let Some(location) = info.location() {
                *LAST_PANIC_LOCATION.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) =
                    Some(format!("{}:{}", location.file(), location.line()));
            }
            previous(info);
        }));
    });
}

/// Python module entry point
#[pymodule]
fn _raptorbt(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    silence_money_fault_notices();
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
    m.add_class::<python::bindings::PyOrder>()?;
    m.add_class::<python::bindings::PyPortfolioResult>()?;
    m.add_class::<python::account_bindings::PyAccountResult>()?;
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
    m.add_class::<python::schedule_bindings::PySessionSchedule>()?;
    m.add_class::<python::lifecycle_bindings::PyCorporateAction>()?;
    m.add_function(wrap_pyfunction!(python::lifecycle_bindings::adjust_history, m)?)?;
    m.add_class::<python::costs_bindings::PyRateCard>()?;
    m.add_class::<python::costs_bindings::PyFeeSchedule>()?;
    m.add_class::<python::margin_bindings::PyMarginModel>()?;
    m.add_function(wrap_pyfunction!(python::margin_bindings::defined_risk_py, m)?)?;

    // Register streaming indicators
    m.add_class::<python::indicator_bindings::PyIndicator>()?;

    // Register bar aggregation
    m.add_class::<python::data_bindings::PyBarAggregator>()?;
    m.add_function(wrap_pyfunction!(python::data_bindings::aggregate_bars, m)?)?;
    m.add_function(wrap_pyfunction!(python::data_bindings::bars_from_ticks, m)?)?;

    // Register the per-bar strategy session (class-based strategy contract)
    m.add_class::<python::strategy_bindings::PyKernelSession>()?;
    m.add_class::<python::session_bindings::PyPortfolioSession>()?;
    m.add_class::<python::runtime_bindings::PyMarketData>()?;
    m.add_class::<python::runtime_bindings::PyRuntime>()?;
    m.add_class::<python::runtime_bindings::PyRuntimeContext>()?;
    m.add_class::<python::runtime_bindings::PyStrategyView>()?;
    m.add_class::<python::runtime_bindings::PyCommandQueue>()?;
    m.add_class::<python::runtime_bindings::PyKernelDriver>()?;
    m.add_class::<python::execution_bindings::PyOrderRequest>()?;
    m.add_class::<python::execution_bindings::PyExecutionReport>()?;
    m.add_class::<python::journal_bindings::PyJournal>()?;
    m.add_class::<python::journal_bindings::PyRecord>()?;
    m.add_function(wrap_pyfunction!(python::journal_bindings::py_chain_step, m)?)?;
    m.add_function(wrap_pyfunction!(python::journal_bindings::py_chain_start, m)?)?;
    m.add_function(wrap_pyfunction!(python::journal_bindings::py_canonical, m)?)?;
    m.add_class::<python::strategy_bindings::PyEngineEvent>()?;
    m.add_class::<python::strategy_bindings::PyPositionSnapshot>()?;
    m.add_function(wrap_pyfunction!(python::strategy_bindings::resolve_atr_period, m)?)?;

    // Register batch spread backtest
    m.add_class::<python::bindings::PyBatchSpreadItem>()?;
    m.add_function(wrap_pyfunction!(python::bindings::batch_spread_backtest, m)?)?;
    m.add_class::<python::bindings::PyBatchSingleItem>()?;
    m.add_function(wrap_pyfunction!(python::bindings::batch_single_backtest, m)?)?;

    // Register the research numerics and the full report
    python::research_bindings::register(m)?;
    python::report_bindings::register(m)?;
    python::resample_bindings::register(m)?;

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

    // Register indicator functions. The names are recorded as they are
    // registered, so `capabilities()` lists exactly these.
    let mut indicator_functions: Vec<String> = Vec::new();
    macro_rules! indicator {
        ($f:path) => {{
            let f = wrap_pyfunction!($f, m)?;
            indicator_functions.push(f.getattr("__name__")?.extract()?);
            m.add_function(f)?;
        }};
    }
    indicator!(python::bindings::sma);
    indicator!(python::bindings::ema);
    indicator!(python::bindings::rsi);
    indicator!(python::bindings::macd);
    indicator!(python::bindings::stochastic);
    indicator!(python::bindings::atr);
    indicator!(python::bindings::bollinger_bands);
    indicator!(python::bindings::adx);
    indicator!(python::bindings::vwap);
    indicator!(python::bindings::supertrend);
    indicator!(python::bindings::rolling_min);
    indicator!(python::bindings::rolling_max);
    m.add("_INDICATOR_FUNCTIONS", indicator_functions)?;

    m.add_function(wrap_pyfunction!(python::capabilities_bindings::capabilities_json, m)?)?;
    m.add_function(wrap_pyfunction!(_last_panic_location, m)?)?;

    // Register option pricing, Greeks, implied volatility and the option chain
    python::pricing_bindings::register(m)?;

    Ok(())
}
