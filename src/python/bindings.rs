//! PyO3 function bindings for RaptorBT.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use std::collections::HashMap;

use crate::core::types::{
    BacktestConfig, CompiledSignals, Direction, InstrumentConfig, OhlcvData, StopConfig,
    TargetConfig,
};
use crate::indicators;
use crate::signals::synchronizer::SyncMode;
use crate::strategies::basket::{BasketBacktest, BasketConfig};
use crate::strategies::multi::{CombineMode, MultiStrategyBacktest, MultiStrategyConfig};
use crate::strategies::options::{
    OptionType, OptionsBacktest, OptionsConfig, SizeType, StrikeSelection,
};
use crate::strategies::pairs::{PairsBacktest, PairsConfig};
use crate::strategies::single::SingleBacktest;

/// One instrument's arrays as Python hands them to `run_basket_backtest`:
/// `(timestamps, open, high, low, close, volume, entries, exits, direction,
/// weight, symbol)`.
type PyInstrumentArrays<'py> = (
    PyReadonlyArray1<'py, i64>,
    PyReadonlyArray1<'py, f64>,
    PyReadonlyArray1<'py, f64>,
    PyReadonlyArray1<'py, f64>,
    PyReadonlyArray1<'py, f64>,
    PyReadonlyArray1<'py, f64>,
    PyReadonlyArray1<'py, bool>,
    PyReadonlyArray1<'py, bool>,
    i32,
    f64,
    String,
);

/// One strategy's signals for `run_multi_backtest`:
/// `(entries, exits, direction, weight, name)`.
/// Parse a `direction` argument, refusing anything that is not `1` or `-1`.
///
/// Through 0.6.4 every call site did `Direction::from_int(d).unwrap_or(Long)`.
/// A book encoded `0`/`1` instead of `-1`/`1` -- a natural "flat or long"
/// convention -- therefore backtested entirely long, flipping the sign of the
/// P&L on every short with a perfectly well-formed equity curve to show for it.
/// For the basket and portfolio runners the parse happens per instrument, so a
/// single bad row silently turned one leg of a market-neutral book into a
/// doubled long.
fn parse_direction(direction: i32) -> PyResult<Direction> {
    Direction::from_int(direction).ok_or_else(|| {
        PyValueError::new_err(format!("direction must be 1 (long) or -1 (short), got {direction}"))
    })
}

type PyStrategySignals<'py> =
    (PyReadonlyArray1<'py, bool>, PyReadonlyArray1<'py, bool>, i32, f64, String);
use crate::strategies::spreads::{
    LegConfig, OptionType as SpreadOptionType, SpreadBacktest, SpreadConfig, SpreadType,
};
use crate::strategies::tick::{TickBacktest, TickBacktestConfig};

use super::numpy_bridge::*;

// ============================================================================
// Configuration Classes
// ============================================================================

/// Python-exposed backtest configuration.
#[pyclass(name = "BacktestConfig")]
#[derive(Debug, Clone)]
pub struct PyBacktestConfig {
    #[pyo3(get, set)]
    pub initial_capital: f64,
    #[pyo3(get, set)]
    pub fees: f64,
    #[pyo3(get, set)]
    pub slippage: f64,
    #[pyo3(get, set)]
    pub upon_bar_close: bool,
    /// Whether `slippage` is applied to fills. Ignored entirely before 0.5.0.
    #[pyo3(get, set)]
    pub apply_slippage: bool,
    /// Explicit annualization factor; `None` infers it from bar spacing.
    #[pyo3(get, set)]
    pub periods_per_year: Option<f64>,
    /// Annual risk-free rate as a fraction.
    #[pyo3(get, set)]
    pub risk_free_rate: f64,
    /// Trading minutes per session for intraday annualization.
    ///
    /// NSE equity 375, MCX 870, CDS 480. `0` marks a 24x7 market.
    /// `None` defaults to NSE.
    #[pyo3(get, set)]
    pub session_minutes: Option<f64>,
    /// Itemized Indian cost segment, e.g. "NSE", "NFO-OPT", "MCX-FUT".
    ///
    /// When set, the engine charges the real regulatory schedule instead of
    /// the flat `fees` fraction, and reports the breakdown per trade.
    #[pyo3(get, set)]
    pub fee_segment: Option<String>,
    /// Maximum concurrent open positions. `None` is unlimited.
    #[pyo3(get, set)]
    pub max_positions: Option<usize>,
    /// Drawdown percent that halts new entries. `None` disables.
    #[pyo3(get, set)]
    pub max_drawdown_pct: Option<f64>,
    /// Reproduce pre-0.5.0 annualization constants.
    #[pyo3(get, set)]
    pub legacy_annualization: bool,
    /// Probability a marketable resting limit order fills (1.0 = always).
    #[pyo3(get, set)]
    pub fill_prob_limit: f64,
    /// Probability a stop/market fill slips one tick against the trader.
    #[pyo3(get, set)]
    pub fill_prob_slippage: f64,
    /// Fill resting limits from observed queue position (needs depth data).
    #[pyo3(get, set)]
    pub queue_fill_model: bool,
    /// Offset for the trading date DAY orders expire on (0 = UTC).
    #[pyo3(get, set)]
    pub session_tz_offset_ns: i64,
    /// Squareoff time as minutes from local midnight; `None` disables.
    ///
    /// Set through the `squareoff_time` "HH:MM" argument, not directly.
    #[pyo3(get)]
    pub squareoff_time_minutes: Option<u32>,
    /// Adverse adjustment on limit fills, as a fraction of the limit price.
    #[pyo3(get, set)]
    pub limit_slippage: f64,
    /// Force-close positions on a margin call instead of only halting entries.
    #[pyo3(get, set)]
    pub liquidate_on_margin_call: bool,
    /// Seed for the stochastic-fill RNG.
    #[pyo3(get, set)]
    pub fill_seed: u64,
    /// Infer intra-bar high/low order from candle geometry on stop/target
    /// conflicts; false keeps the legacy stop-first assumption.
    #[pyo3(get, set)]
    pub bar_path_adaptive: bool,
    stop_config: StopConfig,
    target_config: TargetConfig,
}

#[pymethods]
impl PyBacktestConfig {
    // New parameters are appended so existing positional calls keep working.
    #[new]
    #[pyo3(signature = (
        initial_capital=100000.0,
        fees=0.001,
        slippage=0.0,
        upon_bar_close=true,
        apply_slippage=true,
        periods_per_year=None,
        risk_free_rate=0.0,
        session_minutes=None,
        fee_segment=None,
        max_positions=None,
        max_drawdown_pct=None,
        legacy_annualization=false,
        fill_prob_limit=1.0,
        fill_prob_slippage=0.0,
        fill_seed=0,
        bar_path_adaptive=false,
        queue_fill_model=false,
        session_tz_offset_ns=0,
        limit_slippage=0.0,
        liquidate_on_margin_call=false,
        squareoff_time=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        initial_capital: f64,
        fees: f64,
        slippage: f64,
        upon_bar_close: bool,
        apply_slippage: bool,
        periods_per_year: Option<f64>,
        risk_free_rate: f64,
        session_minutes: Option<f64>,
        fee_segment: Option<String>,
        max_positions: Option<usize>,
        max_drawdown_pct: Option<f64>,
        legacy_annualization: bool,
        fill_prob_limit: f64,
        fill_prob_slippage: f64,
        fill_seed: u64,
        bar_path_adaptive: bool,
        queue_fill_model: bool,
        session_tz_offset_ns: i64,
        limit_slippage: f64,
        liquidate_on_margin_call: bool,
        squareoff_time: Option<String>,
    ) -> PyResult<Self> {
        let squareoff_time_minutes = parse_squareoff_time(squareoff_time.as_deref())?;
        Ok(Self {
            initial_capital,
            fees,
            slippage,
            upon_bar_close,
            apply_slippage,
            periods_per_year,
            risk_free_rate,
            session_minutes,
            fee_segment,
            max_positions,
            max_drawdown_pct,
            legacy_annualization,
            fill_prob_limit,
            queue_fill_model,
            session_tz_offset_ns,
            limit_slippage,
            liquidate_on_margin_call,
            fill_prob_slippage,
            fill_seed,
            bar_path_adaptive,
            stop_config: StopConfig::None,
            target_config: TargetConfig::None,
            squareoff_time_minutes,
        })
    }

    /// Set fixed percentage stop-loss.
    fn set_fixed_stop(&mut self, percent: f64) {
        self.stop_config = StopConfig::Fixed { percent };
    }

    /// Set ATR-based stop-loss.
    fn set_atr_stop(&mut self, multiplier: f64, period: usize) {
        self.stop_config = StopConfig::Atr { multiplier, period };
    }

    /// Set trailing stop-loss.
    fn set_trailing_stop(&mut self, percent: f64) {
        self.stop_config = StopConfig::Trailing { percent };
    }

    /// Set fixed percentage take-profit.
    fn set_fixed_target(&mut self, percent: f64) {
        self.target_config = TargetConfig::Fixed { percent };
    }

    /// Set ATR-based take-profit.
    fn set_atr_target(&mut self, multiplier: f64, period: usize) {
        self.target_config = TargetConfig::Atr { multiplier, period };
    }

    /// Set risk-reward based take-profit.
    fn set_risk_reward_target(&mut self, ratio: f64) {
        self.target_config = TargetConfig::RiskReward { ratio };
    }
}

/// Parse a "HH:MM" local squareoff time into minutes from midnight.
///
/// Refuses anything it cannot read rather than guessing. A silently ignored
/// squareoff is the exact defect this argument exists to fix: the backend
/// passed `session_aware=True` to an engine with no such setting for months,
/// the call was swallowed by a `hasattr` guard, and every intraday backtest
/// held positions overnight while reporting profit no user could have earned.
/// Refusing loudly is the whole point -- do not add a fallback here.
fn parse_squareoff_time(value: Option<&str>) -> PyResult<Option<u32>> {
    let Some(raw) = value else {
        return Ok(None);
    };
    let text = raw.trim();

    let invalid = || {
        PyValueError::new_err(format!(
            "invalid squareoff_time {raw:?}; expected 24-hour \"HH:MM\" local \
             time, e.g. \"15:25\" for NSE. Pass None to disable squareoff."
        ))
    };

    let (h, m) = text.split_once(':').ok_or_else(invalid)?;
    let hours: u32 = h.trim().parse().map_err(|_| invalid())?;
    let minutes: u32 = m.trim().parse().map_err(|_| invalid())?;

    // 24:00 is rejected too: a squareoff at midnight local time would never
    // fire inside a session, so accepting it would be accepting a no-op.
    if hours > 23 || minutes > 59 {
        return Err(invalid());
    }

    Ok(Some(hours * 60 + minutes))
}

impl From<&PyBacktestConfig> for BacktestConfig {
    fn from(py_config: &PyBacktestConfig) -> Self {
        BacktestConfig {
            initial_capital: py_config.initial_capital,
            fees: py_config.fees,
            slippage: py_config.slippage,
            stop: py_config.stop_config,
            target: py_config.target_config,
            upon_bar_close: py_config.upon_bar_close,
            apply_slippage: py_config.apply_slippage,
            periods_per_year: py_config.periods_per_year,
            risk_free_rate: py_config.risk_free_rate,
            session_minutes: py_config.session_minutes,
            squareoff_time_minutes: py_config.squareoff_time_minutes,
            fee_segment: py_config.fee_segment.clone(),
            max_positions: py_config.max_positions,
            max_drawdown_pct: py_config.max_drawdown_pct,
            legacy_annualization: py_config.legacy_annualization,
            fill_prob_limit: py_config.fill_prob_limit,
            queue_fill_model: py_config.queue_fill_model,
            session_tz_offset_ns: py_config.session_tz_offset_ns,
            limit_slippage: py_config.limit_slippage,
            liquidate_on_margin_call: py_config.liquidate_on_margin_call,
            fill_prob_slippage: py_config.fill_prob_slippage,
            fill_seed: py_config.fill_seed,
            bar_path_adaptive: py_config.bar_path_adaptive,
        }
    }
}

/// Python-exposed per-instrument configuration.
#[pyclass(name = "InstrumentConfig")]
#[derive(Debug, Clone)]
pub struct PyInstrumentConfig {
    #[pyo3(get, set)]
    pub lot_size: Option<f64>,
    #[pyo3(get, set)]
    pub alloted_capital: Option<f64>,
    #[pyo3(get, set)]
    pub existing_qty: Option<f64>,
    #[pyo3(get, set)]
    pub avg_price: Option<f64>,
    stop_config: Option<StopConfig>,
    target_config: Option<TargetConfig>,
}

#[pymethods]
impl PyInstrumentConfig {
    #[new]
    #[pyo3(signature = (lot_size=None, alloted_capital=None, existing_qty=None, avg_price=None))]
    fn new(
        lot_size: Option<f64>,
        alloted_capital: Option<f64>,
        existing_qty: Option<f64>,
        avg_price: Option<f64>,
    ) -> Self {
        Self {
            lot_size,
            alloted_capital,
            existing_qty,
            avg_price,
            stop_config: None,
            target_config: None,
        }
    }

    /// Set fixed percentage stop-loss override.
    fn set_fixed_stop(&mut self, percent: f64) {
        self.stop_config = Some(StopConfig::Fixed { percent });
    }

    /// Set ATR-based stop-loss override.
    fn set_atr_stop(&mut self, multiplier: f64, period: usize) {
        self.stop_config = Some(StopConfig::Atr { multiplier, period });
    }

    /// Set trailing stop-loss override.
    fn set_trailing_stop(&mut self, percent: f64) {
        self.stop_config = Some(StopConfig::Trailing { percent });
    }

    /// Set fixed percentage take-profit override.
    fn set_fixed_target(&mut self, percent: f64) {
        self.target_config = Some(TargetConfig::Fixed { percent });
    }

    /// Set ATR-based take-profit override.
    fn set_atr_target(&mut self, multiplier: f64, period: usize) {
        self.target_config = Some(TargetConfig::Atr { multiplier, period });
    }

    /// Set risk-reward based take-profit override.
    fn set_risk_reward_target(&mut self, ratio: f64) {
        self.target_config = Some(TargetConfig::RiskReward { ratio });
    }

    fn __repr__(&self) -> String {
        format!(
            "InstrumentConfig(lot_size={:?}, alloted_capital={:?})",
            self.lot_size, self.alloted_capital
        )
    }
}

impl From<&PyInstrumentConfig> for InstrumentConfig {
    fn from(py_config: &PyInstrumentConfig) -> Self {
        InstrumentConfig {
            lot_size: py_config.lot_size,
            alloted_capital: py_config.alloted_capital,
            stop: py_config.stop_config,
            target: py_config.target_config,
            existing_qty: py_config.existing_qty,
            avg_price: py_config.avg_price,
        }
    }
}

/// Python-exposed stop configuration.
#[pyclass(name = "StopConfig")]
#[derive(Debug, Clone)]
pub struct PyStopConfig {
    #[pyo3(get, set)]
    pub stop_type: String,
    #[pyo3(get, set)]
    pub percent: Option<f64>,
    #[pyo3(get, set)]
    pub multiplier: Option<f64>,
    #[pyo3(get, set)]
    pub period: Option<usize>,
}

#[pymethods]
impl PyStopConfig {
    #[new]
    fn new() -> Self {
        Self { stop_type: "none".to_string(), percent: None, multiplier: None, period: None }
    }

    #[staticmethod]
    fn fixed(percent: f64) -> Self {
        Self {
            stop_type: "fixed".to_string(),
            percent: Some(percent),
            multiplier: None,
            period: None,
        }
    }

    #[staticmethod]
    fn atr(multiplier: f64, period: usize) -> Self {
        Self {
            stop_type: "atr".to_string(),
            percent: None,
            multiplier: Some(multiplier),
            period: Some(period),
        }
    }

    #[staticmethod]
    fn trailing(percent: f64) -> Self {
        Self {
            stop_type: "trailing".to_string(),
            percent: Some(percent),
            multiplier: None,
            period: None,
        }
    }
}

/// Python-exposed target configuration.
#[pyclass(name = "TargetConfig")]
#[derive(Debug, Clone)]
pub struct PyTargetConfig {
    #[pyo3(get, set)]
    pub target_type: String,
    #[pyo3(get, set)]
    pub percent: Option<f64>,
    #[pyo3(get, set)]
    pub multiplier: Option<f64>,
    #[pyo3(get, set)]
    pub period: Option<usize>,
    #[pyo3(get, set)]
    pub ratio: Option<f64>,
}

#[pymethods]
impl PyTargetConfig {
    #[new]
    fn new() -> Self {
        Self {
            target_type: "none".to_string(),
            percent: None,
            multiplier: None,
            period: None,
            ratio: None,
        }
    }

    #[staticmethod]
    fn fixed(percent: f64) -> Self {
        Self {
            target_type: "fixed".to_string(),
            percent: Some(percent),
            multiplier: None,
            period: None,
            ratio: None,
        }
    }

    #[staticmethod]
    fn atr(multiplier: f64, period: usize) -> Self {
        Self {
            target_type: "atr".to_string(),
            percent: None,
            multiplier: Some(multiplier),
            period: Some(period),
            ratio: None,
        }
    }

    #[staticmethod]
    fn risk_reward(ratio: f64) -> Self {
        Self {
            target_type: "risk_reward".to_string(),
            percent: None,
            multiplier: None,
            period: None,
            ratio: Some(ratio),
        }
    }
}

// ============================================================================
// Result Classes
// ============================================================================

/// Python-exposed trade.
#[pyclass(name = "Trade")]
#[derive(Debug, Clone)]
pub struct PyTrade {
    #[pyo3(get)]
    pub id: u64,
    #[pyo3(get)]
    pub symbol: String,
    #[pyo3(get)]
    pub entry_idx: usize,
    #[pyo3(get)]
    pub exit_idx: usize,
    #[pyo3(get)]
    pub entry_price: f64,
    #[pyo3(get)]
    pub exit_price: f64,
    #[pyo3(get)]
    pub size: f64,
    #[pyo3(get)]
    pub direction: i32,
    #[pyo3(get)]
    pub pnl: f64,
    #[pyo3(get)]
    pub return_pct: f64,
    #[pyo3(get)]
    pub entry_time: i64,
    #[pyo3(get)]
    pub exit_time: i64,
    /// Total costs over the round trip, equal to `entry_fees + exit_fees`.
    #[pyo3(get)]
    pub fees: f64,
    /// Costs charged when the position was opened.
    #[pyo3(get)]
    pub entry_fees: f64,
    /// Costs charged when it was closed.
    ///
    /// Zero when the exit was not a trade-out -- an option left to expire is
    /// never sold, so it owes no exit-side brokerage or transaction tax.
    #[pyo3(get)]
    pub exit_fees: f64,
    /// Itemized regulatory costs, when `config.fee_segment` is set.
    ///
    /// Keys: brokerage, stt, exchange_txn, sebi_fee, stamp_duty, gst, total.
    /// `total` equals `fees`, so reported costs and the equity curve agree.
    #[pyo3(get)]
    pub fee_breakdown: Option<HashMap<String, f64>>,
    #[pyo3(get)]
    pub exit_reason: String,
}

#[pymethods]
impl PyTrade {
    fn __repr__(&self) -> String {
        format!(
            "Trade(symbol={}, entry={:.2}, exit={:.2}, pnl={:.2}, return={:.2}%)",
            self.symbol, self.entry_price, self.exit_price, self.pnl, self.return_pct
        )
    }
}

/// Python-exposed backtest metrics.
#[pyclass(name = "BacktestMetrics")]
#[derive(Debug, Clone)]
pub struct PyBacktestMetrics {
    #[pyo3(get)]
    pub total_return_pct: f64,
    #[pyo3(get)]
    pub sharpe_ratio: f64,
    #[pyo3(get)]
    pub sortino_ratio: Option<f64>,
    #[pyo3(get)]
    pub calmar_ratio: Option<f64>,
    #[pyo3(get)]
    pub omega_ratio: Option<f64>,
    #[pyo3(get)]
    pub max_drawdown_pct: f64,
    #[pyo3(get)]
    pub max_drawdown_duration: usize,
    #[pyo3(get)]
    pub win_rate_pct: f64,
    #[pyo3(get)]
    pub profit_factor: Option<f64>,
    #[pyo3(get)]
    pub expectancy: f64,
    #[pyo3(get)]
    pub sqn: f64,
    #[pyo3(get)]
    pub total_trades: usize,
    #[pyo3(get)]
    pub total_closed_trades: usize,
    #[pyo3(get)]
    pub total_open_trades: usize,
    #[pyo3(get)]
    pub open_trade_pnl: f64,
    #[pyo3(get)]
    pub winning_trades: usize,
    #[pyo3(get)]
    pub losing_trades: usize,
    #[pyo3(get)]
    pub start_value: f64,
    #[pyo3(get)]
    pub end_value: f64,
    #[pyo3(get)]
    pub total_fees_paid: f64,
    #[pyo3(get)]
    pub best_trade_pct: f64,
    #[pyo3(get)]
    pub worst_trade_pct: f64,
    #[pyo3(get)]
    pub avg_trade_return_pct: f64,
    #[pyo3(get)]
    pub avg_win_pct: f64,
    #[pyo3(get)]
    pub avg_loss_pct: f64,
    #[pyo3(get)]
    pub avg_winning_duration: f64,
    #[pyo3(get)]
    pub avg_losing_duration: f64,
    #[pyo3(get)]
    pub max_consecutive_wins: usize,
    #[pyo3(get)]
    pub max_consecutive_losses: usize,
    #[pyo3(get)]
    pub avg_holding_period: f64,
    #[pyo3(get)]
    pub exposure_pct: f64,
    #[pyo3(get)]
    pub payoff_ratio: Option<f64>,
    #[pyo3(get)]
    pub recovery_factor: Option<f64>,
}

#[pymethods]
impl PyBacktestMetrics {
    fn __repr__(&self) -> String {
        format!(
            "BacktestMetrics(return={:.2}%, sharpe={:.2}, max_dd={:.2}%, trades={})",
            self.total_return_pct, self.sharpe_ratio, self.max_drawdown_pct, self.total_trades
        )
    }

    /// Convert to dictionary of all metrics.
    fn to_dict(&self, py: Python) -> PyResult<PyObject> {
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("Start Value", self.start_value)?;
        dict.set_item("End Value", self.end_value)?;
        dict.set_item("Total Return [%]", self.total_return_pct)?;
        dict.set_item("Total Fees Paid", self.total_fees_paid)?;
        dict.set_item("Max Drawdown [%]", self.max_drawdown_pct)?;
        dict.set_item("Max Drawdown Duration", self.max_drawdown_duration)?;
        dict.set_item("Total Trades", self.total_trades)?;
        dict.set_item("Total Closed Trades", self.total_closed_trades)?;
        dict.set_item("Total Open Trades", self.total_open_trades)?;
        dict.set_item("Open Trade PnL", self.open_trade_pnl)?;
        dict.set_item("Win Rate [%]", self.win_rate_pct)?;
        dict.set_item("Best Trade [%]", self.best_trade_pct)?;
        dict.set_item("Worst Trade [%]", self.worst_trade_pct)?;
        dict.set_item("Avg Winning Trade [%]", self.avg_win_pct)?;
        dict.set_item("Avg Losing Trade [%]", self.avg_loss_pct)?;
        dict.set_item("Avg Winning Trade Duration", self.avg_winning_duration)?;
        dict.set_item("Avg Losing Trade Duration", self.avg_losing_duration)?;
        dict.set_item("Profit Factor", self.profit_factor)?;
        dict.set_item("Expectancy", self.expectancy)?;
        dict.set_item("SQN", self.sqn)?;
        dict.set_item("Sharpe Ratio", self.sharpe_ratio)?;
        dict.set_item("Sortino Ratio", self.sortino_ratio)?;
        dict.set_item("Calmar Ratio", self.calmar_ratio)?;
        dict.set_item("Omega Ratio", self.omega_ratio)?;
        Ok(dict.into())
    }
}

/// Python-exposed backtest result.
#[pyclass(name = "BacktestResult")]
#[derive(Debug, Clone)]
pub struct PyBacktestResult {
    #[pyo3(get)]
    pub metrics: PyBacktestMetrics,
    equity_curve: Vec<f64>,
    drawdown_curve: Vec<f64>,
    trades: Vec<PyTrade>,
    returns: Vec<f64>,
}

#[pymethods]
impl PyBacktestResult {
    /// Get equity curve as numpy array.
    fn equity_curve<'py>(&self, py: Python<'py>) -> &'py PyArray1<f64> {
        vec_to_numpy_f64(py, self.equity_curve.clone())
    }

    /// Get drawdown curve as numpy array.
    fn drawdown_curve<'py>(&self, py: Python<'py>) -> &'py PyArray1<f64> {
        vec_to_numpy_f64(py, self.drawdown_curve.clone())
    }

    /// Get returns as numpy array.
    fn returns<'py>(&self, py: Python<'py>) -> &'py PyArray1<f64> {
        vec_to_numpy_f64(py, self.returns.clone())
    }

    /// Get list of trades.
    fn trades(&self) -> Vec<PyTrade> {
        self.trades.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "BacktestResult(return={:.2}%, trades={}, max_dd={:.2}%)",
            self.metrics.total_return_pct, self.metrics.total_trades, self.metrics.max_drawdown_pct
        )
    }
}

// ============================================================================
// Backtest Functions
// ============================================================================

// The argument list IS the Python signature; collapsing it into a
// struct would change the public API for no reader benefit.
#[allow(clippy::too_many_arguments)]
/// Run single instrument backtest.
///
/// Note: the array-based runners (precomputed boolean entry/exit signals)
/// are the legacy strategy path. For new strategies prefer the class-based
/// strategy contract (`raptorbt.Strategy` + `run_strategy_backtest`), which
/// shares the same execution core and result types. Array runners remain
/// supported and will only be deprecated in a future major release.
#[pyfunction]
#[pyo3(signature = (timestamps, open, high, low, close, volume, entries, exits, direction=1, weight=1.0, symbol="UNKNOWN", config=None, position_sizes=None, instrument_config=None))]
pub fn run_single_backtest<'py>(
    _py: Python<'py>,
    timestamps: PyReadonlyArray1<i64>,
    open: PyReadonlyArray1<f64>,
    high: PyReadonlyArray1<f64>,
    low: PyReadonlyArray1<f64>,
    close: PyReadonlyArray1<f64>,
    volume: PyReadonlyArray1<f64>,
    entries: PyReadonlyArray1<bool>,
    exits: PyReadonlyArray1<bool>,
    direction: i32,
    weight: f64,
    symbol: &str,
    config: Option<&PyBacktestConfig>,
    position_sizes: Option<PyReadonlyArray1<f64>>,
    instrument_config: Option<&PyInstrumentConfig>,
) -> PyResult<PyBacktestResult> {
    let ohlcv = OhlcvData {
        timestamps: numpy_to_vec_i64(timestamps),
        open: numpy_to_vec_f64(open),
        high: numpy_to_vec_f64(high),
        low: numpy_to_vec_f64(low),
        close: numpy_to_vec_f64(close),
        volume: numpy_to_vec_f64(volume),
    };

    let dir = parse_direction(direction)?;

    let signals = CompiledSignals {
        symbol: symbol.to_string(),
        entries: numpy_to_vec_bool(entries),
        exits: numpy_to_vec_bool(exits),
        position_sizes: position_sizes.map(numpy_to_vec_f64),
        direction: dir,
        weight,
    };

    let rust_config = config.map(BacktestConfig::from).unwrap_or_default();
    let inst_config = instrument_config.map(InstrumentConfig::from);

    let backtest = SingleBacktest::new(rust_config);
    let result = backtest.run_with_instrument_config(&ohlcv, &signals, inst_config.as_ref());

    Ok(convert_result(result))
}

/// Run basket/collective backtest.
#[pyfunction]
#[pyo3(signature = (instruments, config=None, sync_mode="all", instrument_configs=None))]
pub fn run_basket_backtest<'py>(
    _py: Python<'py>,
    instruments: Vec<PyInstrumentArrays<'py>>,
    config: Option<&PyBacktestConfig>,
    sync_mode: &str,
    instrument_configs: Option<HashMap<String, PyInstrumentConfig>>,
) -> PyResult<PyBacktestResult> {
    let rust_instruments: Vec<(OhlcvData, CompiledSignals)> = instruments
        .into_iter()
        .map(|(ts, o, h, l, c, v, entries, exits, dir, weight, sym)| {
            let ohlcv = OhlcvData {
                timestamps: numpy_to_vec_i64(ts),
                open: numpy_to_vec_f64(o),
                high: numpy_to_vec_f64(h),
                low: numpy_to_vec_f64(l),
                close: numpy_to_vec_f64(c),
                volume: numpy_to_vec_f64(v),
            };
            let signals = CompiledSignals {
                symbol: sym,
                entries: numpy_to_vec_bool(entries),
                exits: numpy_to_vec_bool(exits),
                position_sizes: None,
                direction: parse_direction(dir)?,
                weight,
            };
            Ok((ohlcv, signals))
        })
        .collect::<PyResult<_>>()?;

    let mode = match sync_mode {
        "any" => SyncMode::Any,
        "majority" => SyncMode::Majority,
        "master" => SyncMode::Master,
        _ => SyncMode::All,
    };

    let basket_config = BasketConfig {
        base: config.map(BacktestConfig::from).unwrap_or_default(),
        sync_mode: mode,
        ..Default::default()
    };

    // Convert PyInstrumentConfig map to InstrumentConfig map
    let rust_inst_configs: Option<HashMap<String, InstrumentConfig>> =
        instrument_configs.map(|configs| {
            configs.iter().map(|(k, v)| (k.clone(), InstrumentConfig::from(v))).collect()
        });

    let backtest = BasketBacktest::new(basket_config);
    let result =
        backtest.run_with_instrument_configs(&rust_instruments, rust_inst_configs.as_ref());

    Ok(convert_result(result))
}

/// Per-instrument attribution from a portfolio backtest.
#[pyclass(name = "InstrumentSummary")]
#[derive(Debug, Clone)]
pub struct PyInstrumentSummary {
    #[pyo3(get)]
    pub symbol: String,
    #[pyo3(get)]
    pub trades: usize,
    #[pyo3(get)]
    pub pnl: f64,
    /// Entries refused because the portfolio was at its limit or halted.
    #[pyo3(get)]
    pub rejected_entries: usize,
}

#[pymethods]
impl PyInstrumentSummary {
    fn __repr__(&self) -> String {
        format!(
            "InstrumentSummary(symbol='{}', trades={}, pnl={:.2}, rejected_entries={})",
            self.symbol, self.trades, self.pnl, self.rejected_entries
        )
    }
}

/// Result of a shared-capital portfolio backtest.
#[pyclass(name = "PortfolioResult")]
#[derive(Debug, Clone)]
pub struct PyPortfolioResult {
    #[pyo3(get)]
    pub result: PyBacktestResult,
    #[pyo3(get)]
    pub per_instrument: Vec<PyInstrumentSummary>,
    /// Entries refused by the risk gate, across all instruments.
    #[pyo3(get)]
    pub rejected_entries: usize,
    /// Whether the drawdown kill-switch tripped.
    #[pyo3(get)]
    pub halted: bool,
    /// Bar index at which the kill-switch tripped.
    #[pyo3(get)]
    pub halted_at: Option<usize>,
}

#[pymethods]
impl PyPortfolioResult {
    /// Portfolio metrics, computed on the constrained run.
    #[getter]
    fn metrics(&self) -> PyBacktestMetrics {
        self.result.metrics.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "PortfolioResult(instruments={}, rejected_entries={}, halted={})",
            self.per_instrument.len(),
            self.rejected_entries,
            self.halted
        )
    }
}

/// Run a shared-capital portfolio backtest.
///
/// Simulates every instrument against one cash pool, with `max_positions` and
/// the drawdown kill-switch (set on `config`) gating each entry *before* it
/// opens -- so the reported metrics describe the constrained run.
///
/// This is not the same as running one backtest per symbol and summing the
/// equity curves: there each symbol gets its own private capital, so N symbols
/// deploy N times the account and no cross-symbol limit can be applied.
///
/// `allocation`: "equal_weight" (default) divides the pool across the slots
/// that could still open; "full" lets each entry take the whole remaining pool.
#[pyfunction]
#[pyo3(signature = (instruments, config=None, allocation="equal_weight", instrument_configs=None))]
#[allow(clippy::type_complexity)]
pub fn run_portfolio_backtest<'py>(
    _py: Python<'py>,
    instruments: Vec<PyInstrumentArrays<'py>>,
    config: Option<&PyBacktestConfig>,
    allocation: &str,
    instrument_configs: Option<HashMap<String, PyInstrumentConfig>>,
) -> PyResult<PyPortfolioResult> {
    use crate::strategies::portfolio::{
        CapitalAllocation, PortfolioBacktest, PortfolioBacktestConfig,
    };

    if instruments.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "run_portfolio_backtest requires at least one instrument",
        ));
    }

    let rust_instruments: Vec<(OhlcvData, CompiledSignals)> = instruments
        .into_iter()
        .map(|(ts, o, h, l, c, v, entries, exits, dir, weight, sym)| {
            let ohlcv = OhlcvData {
                timestamps: numpy_to_vec_i64(ts),
                open: numpy_to_vec_f64(o),
                high: numpy_to_vec_f64(h),
                low: numpy_to_vec_f64(l),
                close: numpy_to_vec_f64(c),
                volume: numpy_to_vec_f64(v),
            };
            let signals = CompiledSignals {
                symbol: sym,
                entries: numpy_to_vec_bool(entries),
                exits: numpy_to_vec_bool(exits),
                position_sizes: None,
                direction: parse_direction(dir)?,
                weight,
            };
            Ok((ohlcv, signals))
        })
        .collect::<PyResult<_>>()?;

    let n_bars = rust_instruments[0].0.len();
    if let Some((idx, _)) =
        rust_instruments.iter().enumerate().find(|(_, (o, _))| o.len() != n_bars)
    {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "all instruments must have the same number of bars; instrument {idx} has {}, expected {n_bars}",
            rust_instruments[idx].0.len()
        )));
    }

    let allocation = match allocation {
        "full" => CapitalAllocation::Full,
        "equal_weight" => CapitalAllocation::EqualWeight,
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unknown allocation '{other}'; expected 'equal_weight' or 'full'"
            )))
        }
    };

    let portfolio_config = PortfolioBacktestConfig {
        base: config.map(BacktestConfig::from).unwrap_or_default(),
        allocation,
    };

    let rust_inst_configs: Option<HashMap<String, InstrumentConfig>> =
        instrument_configs.map(|configs| {
            configs.iter().map(|(k, v)| (k.clone(), InstrumentConfig::from(v))).collect()
        });

    let backtest = PortfolioBacktest::new(portfolio_config);
    let out = backtest.run(&rust_instruments, rust_inst_configs.as_ref());

    Ok(PyPortfolioResult {
        result: convert_result(out.result),
        per_instrument: out
            .per_instrument
            .into_iter()
            .map(|s| PyInstrumentSummary {
                symbol: s.symbol,
                trades: s.trades,
                pnl: s.pnl,
                rejected_entries: s.rejected_entries,
            })
            .collect(),
        rejected_entries: out.rejected_entries,
        halted: out.halted,
        halted_at: out.halted_at,
    })
}

// The argument list IS the Python signature; collapsing it into a
// struct would change the public API for no reader benefit.
#[allow(clippy::too_many_arguments)]
/// Run options backtest.
#[pyfunction]
#[pyo3(signature = (timestamps, open, high, low, close, volume, option_prices, entries, exits, direction=1, symbol="OPTION", config=None, option_type="call", strike_selection="atm", size_type="percent", size_value=1.0, lot_size=1, strike_interval=50.0))]
pub fn run_options_backtest<'py>(
    _py: Python<'py>,
    timestamps: PyReadonlyArray1<i64>,
    open: PyReadonlyArray1<f64>,
    high: PyReadonlyArray1<f64>,
    low: PyReadonlyArray1<f64>,
    close: PyReadonlyArray1<f64>,
    volume: PyReadonlyArray1<f64>,
    option_prices: PyReadonlyArray1<f64>,
    entries: PyReadonlyArray1<bool>,
    exits: PyReadonlyArray1<bool>,
    direction: i32,
    symbol: &str,
    config: Option<&PyBacktestConfig>,
    option_type: &str,
    strike_selection: &str,
    size_type: &str,
    size_value: f64,
    lot_size: usize,
    strike_interval: f64,
) -> PyResult<PyBacktestResult> {
    let ohlcv = OhlcvData {
        timestamps: numpy_to_vec_i64(timestamps),
        open: numpy_to_vec_f64(open),
        high: numpy_to_vec_f64(high),
        low: numpy_to_vec_f64(low),
        close: numpy_to_vec_f64(close),
        volume: numpy_to_vec_f64(volume),
    };

    let opt_prices = numpy_to_vec_f64(option_prices);

    let dir = parse_direction(direction)?;

    let signals = CompiledSignals {
        symbol: symbol.to_string(),
        entries: numpy_to_vec_bool(entries),
        exits: numpy_to_vec_bool(exits),
        position_sizes: None,
        direction: dir,
        weight: 1.0,
    };

    // These three parsed with a catch-all `_` arm through 0.6.4, so any string
    // the match did not recognise silently selected the first variant. The
    // sharpest case: `option_type="PUT"` (or "PE", or "Put") backtested a long
    // CALL -- roughly the mirror image of the intended payoff -- while the same
    // string is accepted by `run_spread_backtest`. Unknown input is refused now.
    let opt_type = match option_type.to_lowercase().as_str() {
        "call" | "ce" | "c" => OptionType::Call,
        "put" | "pe" | "p" => OptionType::Put,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown option_type {other:?}; expected call/ce/c or put/pe/p"
            )))
        }
    };

    let strike_sel = match strike_selection.to_lowercase().as_str() {
        "atm" => StrikeSelection::Atm,
        "otm1" => StrikeSelection::Otm(1),
        "otm2" => StrikeSelection::Otm(2),
        "itm1" => StrikeSelection::Itm(1),
        "itm2" => StrikeSelection::Itm(2),
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown strike_selection {other:?}; expected atm, otm1, otm2, itm1 or itm2"
            )))
        }
    };

    let size = match size_type.to_lowercase().as_str() {
        "percent" => SizeType::Percent(size_value),
        "contracts" => SizeType::Contracts(size_value as usize),
        "notional" => SizeType::Notional(size_value),
        "risk" => SizeType::RiskPercent(size_value),
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown size_type {other:?}; expected percent, contracts, notional or risk"
            )))
        }
    };

    let options_config = OptionsConfig {
        base: config.map(BacktestConfig::from).unwrap_or_default(),
        option_type: opt_type,
        strike_selection: strike_sel,
        size_type: size,
        lot_size,
        strike_interval,
        target_dte: None,
    };

    let backtest = OptionsBacktest::new(options_config);
    let result = backtest.run(&ohlcv, &opt_prices, &signals);

    Ok(convert_result(result))
}

// The argument list IS the Python signature; collapsing it into a
// struct would change the public API for no reader benefit.
#[allow(clippy::too_many_arguments)]
/// Run pairs trading backtest.
#[pyfunction]
#[pyo3(signature = (leg1_timestamps, leg1_open, leg1_high, leg1_low, leg1_close, leg1_volume, leg2_timestamps, leg2_open, leg2_high, leg2_low, leg2_close, leg2_volume, entries, exits, direction=1, symbol="PAIR", config=None, hedge_ratio=1.0, dynamic_hedge=false))]
pub fn run_pairs_backtest<'py>(
    _py: Python<'py>,
    leg1_timestamps: PyReadonlyArray1<i64>,
    leg1_open: PyReadonlyArray1<f64>,
    leg1_high: PyReadonlyArray1<f64>,
    leg1_low: PyReadonlyArray1<f64>,
    leg1_close: PyReadonlyArray1<f64>,
    leg1_volume: PyReadonlyArray1<f64>,
    leg2_timestamps: PyReadonlyArray1<i64>,
    leg2_open: PyReadonlyArray1<f64>,
    leg2_high: PyReadonlyArray1<f64>,
    leg2_low: PyReadonlyArray1<f64>,
    leg2_close: PyReadonlyArray1<f64>,
    leg2_volume: PyReadonlyArray1<f64>,
    entries: PyReadonlyArray1<bool>,
    exits: PyReadonlyArray1<bool>,
    direction: i32,
    symbol: &str,
    config: Option<&PyBacktestConfig>,
    hedge_ratio: f64,
    dynamic_hedge: bool,
) -> PyResult<PyBacktestResult> {
    let leg1_ohlcv = OhlcvData {
        timestamps: numpy_to_vec_i64(leg1_timestamps),
        open: numpy_to_vec_f64(leg1_open),
        high: numpy_to_vec_f64(leg1_high),
        low: numpy_to_vec_f64(leg1_low),
        close: numpy_to_vec_f64(leg1_close),
        volume: numpy_to_vec_f64(leg1_volume),
    };

    let leg2_ohlcv = OhlcvData {
        timestamps: numpy_to_vec_i64(leg2_timestamps),
        open: numpy_to_vec_f64(leg2_open),
        high: numpy_to_vec_f64(leg2_high),
        low: numpy_to_vec_f64(leg2_low),
        close: numpy_to_vec_f64(leg2_close),
        volume: numpy_to_vec_f64(leg2_volume),
    };

    let dir = parse_direction(direction)?;

    let signals = CompiledSignals {
        symbol: symbol.to_string(),
        entries: numpy_to_vec_bool(entries),
        exits: numpy_to_vec_bool(exits),
        position_sizes: None,
        direction: dir,
        weight: 1.0,
    };

    let pairs_config = PairsConfig {
        base: config.map(BacktestConfig::from).unwrap_or_default(),
        hedge_ratio,
        dynamic_hedge,
        ..Default::default()
    };

    let backtest = PairsBacktest::new(pairs_config);
    let result = backtest.run(&leg1_ohlcv, &leg2_ohlcv, &signals);

    Ok(convert_result(result))
}

// The argument list IS the Python signature; collapsing it into a
// struct would change the public API for no reader benefit.
#[allow(clippy::too_many_arguments)]
/// Run spread backtest (multi-leg options).
#[pyfunction]
#[pyo3(signature = (timestamps, underlying_close, legs_premiums, leg_configs, entries, exits, config=None, spread_type="custom", max_loss=None, target_profit=None, leg_expiry_timestamps=None))]
pub fn run_spread_backtest<'py>(
    _py: Python<'py>,
    timestamps: PyReadonlyArray1<i64>,
    underlying_close: PyReadonlyArray1<f64>,
    legs_premiums: Vec<PyReadonlyArray1<f64>>,
    leg_configs: Vec<(String, f64, i32, usize)>, // (option_type, strike, quantity, lot_size)
    entries: PyReadonlyArray1<bool>,
    exits: PyReadonlyArray1<bool>,
    config: Option<&PyBacktestConfig>,
    spread_type: &str,
    max_loss: Option<f64>,
    target_profit: Option<f64>,
    leg_expiry_timestamps: Option<Vec<i64>>,
) -> PyResult<PyBacktestResult> {
    let ts = numpy_to_vec_i64(timestamps);
    let underlying = numpy_to_vec_f64(underlying_close);
    let premiums: Vec<Vec<f64>> = legs_premiums.into_iter().map(numpy_to_vec_f64).collect();
    let entry_signals = numpy_to_vec_bool(entries);
    let exit_signals = numpy_to_vec_bool(exits);

    // Convert leg configs
    let rust_leg_configs: Vec<LegConfig> = leg_configs
        .into_iter()
        .map(|(opt_type, strike, quantity, lot_size)| {
            let option_type = SpreadOptionType::from_code(&opt_type).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "unknown option type {opt_type:?} for the leg at strike {strike}; \
                     expected one of CE/CALL/C or PE/PUT/P (case-insensitive). \
                     Defaulting would price a put as a call."
                ))
            })?;
            Ok(LegConfig::new(option_type, strike, quantity, lot_size))
        })
        .collect::<PyResult<_>>()?;

    // Parse spread type
    let spread_type_enum = match spread_type.to_lowercase().as_str() {
        "straddle" => SpreadType::Straddle,
        "strangle" => SpreadType::Strangle,
        "vertical_call" | "verticalcall" => SpreadType::VerticalCall,
        "vertical_put" | "verticalput" => SpreadType::VerticalPut,
        "iron_condor" | "ironcondor" => SpreadType::IronCondor,
        "iron_butterfly" | "ironbutterfly" => SpreadType::IronButterfly,
        "butterfly_call" | "butterflycall" => SpreadType::ButterflyCall,
        "butterfly_put" | "butterflyput" => SpreadType::ButterflyPut,
        "calendar" => SpreadType::Calendar,
        "diagonal" => SpreadType::Diagonal,
        "long_call" | "longcall" => SpreadType::LongCall,
        "long_put" | "longput" => SpreadType::LongPut,
        "naked_call" | "nakedcall" => SpreadType::NakedCall,
        "naked_put" | "nakedput" => SpreadType::NakedPut,
        _ => SpreadType::Custom,
    };

    let spread_config = SpreadConfig {
        base: config.map(BacktestConfig::from).unwrap_or_default(),
        spread_type: spread_type_enum,
        leg_configs: rust_leg_configs,
        max_loss,
        target_profit,
        leg_expiry_timestamps,
    };

    let backtest = SpreadBacktest::new(spread_config);
    let result = backtest.run(&ts, &underlying, &premiums, &entry_signals, &exit_signals);

    Ok(convert_result(result))
}

/// A single spread backtest item for batch execution.
#[pyclass(name = "BatchSpreadItem")]
#[derive(Clone)]
pub struct PyBatchSpreadItem {
    #[pyo3(get, set)]
    pub strategy_id: String,
    pub legs_premiums: Vec<Vec<f64>>,
    pub leg_configs: Vec<(String, f64, i32, usize)>,
    pub entries: Vec<bool>,
    pub exits: Vec<bool>,
    #[pyo3(get, set)]
    pub spread_type: String,
    #[pyo3(get, set)]
    pub max_loss: Option<f64>,
    #[pyo3(get, set)]
    pub target_profit: Option<f64>,
}

#[pymethods]
impl PyBatchSpreadItem {
    #[new]
    #[pyo3(signature = (strategy_id, legs_premiums, leg_configs, entries, exits,
        spread_type="custom", max_loss=None, target_profit=None))]
    // The argument list IS the Python signature; collapsing it into a
    // struct would change the public API for no reader benefit.
    #[allow(clippy::too_many_arguments)]
    fn new(
        strategy_id: String,
        legs_premiums: Vec<PyReadonlyArray1<f64>>,
        leg_configs: Vec<(String, f64, i32, usize)>,
        entries: PyReadonlyArray1<bool>,
        exits: PyReadonlyArray1<bool>,
        spread_type: &str,
        max_loss: Option<f64>,
        target_profit: Option<f64>,
    ) -> Self {
        Self {
            strategy_id,
            legs_premiums: legs_premiums.into_iter().map(numpy_to_vec_f64).collect(),
            leg_configs,
            entries: numpy_to_vec_bool(entries),
            exits: numpy_to_vec_bool(exits),
            spread_type: spread_type.to_string(),
            max_loss,
            target_profit,
        }
    }
}

/// Run multiple spread backtests in parallel via Rayon.
///
/// Shared data (timestamps, underlying_close) is converted once, then each
/// item is backtested on its own Rayon thread with the GIL released.
///
/// Returns a Vec of (strategy_id, PyBacktestResult) tuples.
#[pyfunction]
#[pyo3(signature = (timestamps, underlying_close, items, config=None))]
pub fn batch_spread_backtest(
    py: Python<'_>,
    timestamps: PyReadonlyArray1<i64>,
    underlying_close: PyReadonlyArray1<f64>,
    items: Vec<PyBatchSpreadItem>,
    config: Option<&PyBacktestConfig>,
) -> PyResult<Vec<(String, PyBacktestResult)>> {
    use rayon::prelude::*;

    // Convert shared data while holding GIL
    let ts = numpy_to_vec_i64(timestamps);
    let underlying = numpy_to_vec_f64(underlying_close);
    let base_config = config.map(BacktestConfig::from).unwrap_or_default();

    // Prepare each item into a self-contained struct for parallel execution
    struct PreparedItem {
        strategy_id: String,
        premiums: Vec<Vec<f64>>,
        entries: Vec<bool>,
        exits: Vec<bool>,
        spread_config: SpreadConfig,
    }

    let prepared: Vec<PreparedItem> = items
        .into_iter()
        .map(|item| {
            let rust_leg_configs: Vec<LegConfig> = item
                .leg_configs
                .into_iter()
                .map(|(opt_type, strike, quantity, lot_size)| {
                    let option_type = SpreadOptionType::from_code(&opt_type).ok_or_else(|| {
                        PyValueError::new_err(format!(
                            "unknown option type {opt_type:?} for the leg at strike \
                                 {strike}; expected CE/CALL/C or PE/PUT/P \
                                 (case-insensitive)."
                        ))
                    })?;
                    Ok(LegConfig::new(option_type, strike, quantity, lot_size))
                })
                .collect::<PyResult<_>>()?;

            let spread_type_enum = match item.spread_type.to_lowercase().as_str() {
                "straddle" => SpreadType::Straddle,
                "strangle" => SpreadType::Strangle,
                "vertical_call" | "verticalcall" => SpreadType::VerticalCall,
                "vertical_put" | "verticalput" => SpreadType::VerticalPut,
                "iron_condor" | "ironcondor" => SpreadType::IronCondor,
                "iron_butterfly" | "ironbutterfly" => SpreadType::IronButterfly,
                "butterfly_call" | "butterflycall" => SpreadType::ButterflyCall,
                "butterfly_put" | "butterflyput" => SpreadType::ButterflyPut,
                "calendar" => SpreadType::Calendar,
                "diagonal" => SpreadType::Diagonal,
                "long_call" | "longcall" => SpreadType::LongCall,
                "long_put" | "longput" => SpreadType::LongPut,
                "naked_call" | "nakedcall" => SpreadType::NakedCall,
                "naked_put" | "nakedput" => SpreadType::NakedPut,
                _ => SpreadType::Custom,
            };

            let spread_config = SpreadConfig {
                base: base_config.clone(),
                spread_type: spread_type_enum,
                leg_configs: rust_leg_configs.clone(),
                max_loss: item.max_loss,
                target_profit: item.target_profit,
                leg_expiry_timestamps: None,
            };

            Ok(PreparedItem {
                strategy_id: item.strategy_id,
                premiums: item.legs_premiums,
                entries: item.entries,
                exits: item.exits,
                spread_config,
            })
        })
        .collect::<PyResult<_>>()?;

    // Release GIL and run all backtests in parallel via Rayon
    let results: Vec<(String, crate::core::types::BacktestResult)> = py.allow_threads(|| {
        prepared
            .into_par_iter()
            .map(|item| {
                let backtest = SpreadBacktest::new(item.spread_config);
                let result =
                    backtest.run(&ts, &underlying, &item.premiums, &item.entries, &item.exits);
                (item.strategy_id, result)
            })
            .collect()
    });

    // Re-acquire GIL and convert results to Python objects
    Ok(results.into_iter().map(|(id, result)| (id, convert_result(result))).collect())
}

// The argument list IS the Python signature; collapsing it into a
// struct would change the public API for no reader benefit.
#[allow(clippy::too_many_arguments)]
/// Run multi-strategy backtest.
#[pyfunction]
#[pyo3(signature = (timestamps, open, high, low, close, volume, strategies, config=None, combine_mode="any"))]
pub fn run_multi_backtest<'py>(
    _py: Python<'py>,
    timestamps: PyReadonlyArray1<i64>,
    open: PyReadonlyArray1<f64>,
    high: PyReadonlyArray1<f64>,
    low: PyReadonlyArray1<f64>,
    close: PyReadonlyArray1<f64>,
    volume: PyReadonlyArray1<f64>,
    strategies: Vec<PyStrategySignals<'_>>,
    config: Option<&PyBacktestConfig>,
    combine_mode: &str,
) -> PyResult<PyBacktestResult> {
    let ohlcv = OhlcvData {
        timestamps: numpy_to_vec_i64(timestamps),
        open: numpy_to_vec_f64(open),
        high: numpy_to_vec_f64(high),
        low: numpy_to_vec_f64(low),
        close: numpy_to_vec_f64(close),
        volume: numpy_to_vec_f64(volume),
    };

    let rust_strategies: Vec<CompiledSignals> = strategies
        .into_iter()
        .map(|(entries, exits, dir, weight, symbol)| {
            Ok(CompiledSignals {
                symbol,
                entries: numpy_to_vec_bool(entries),
                exits: numpy_to_vec_bool(exits),
                position_sizes: None,
                direction: parse_direction(dir)?,
                weight,
            })
        })
        .collect::<PyResult<_>>()?;

    let mode = match combine_mode {
        "all" => CombineMode::All,
        "majority" => CombineMode::Majority,
        "independent" => CombineMode::Independent,
        "weighted" => CombineMode::Weighted,
        _ => CombineMode::Any,
    };

    let multi_config = MultiStrategyConfig {
        base: config.map(BacktestConfig::from).unwrap_or_default(),
        combine_mode: mode,
        ..Default::default()
    };

    let backtest = MultiStrategyBacktest::new(multi_config);
    let result = backtest.run(&ohlcv, &rust_strategies);

    Ok(convert_result(result))
}

/// Run tick-level backtest on a single instrument.
///
/// All arrays must be the same length N (one element per tick).
/// `buy_qty_delta` and `sell_qty_delta` must already be per-tick deltas —
/// pass the difference from the previous tick, not Zerodha's cumulative totals.
/// `entries` / `exits` are caller-computed boolean signal arrays.
///
/// Returns a `PyBacktestResult` with the same fields as `run_single_backtest`.
#[pyfunction]
#[pyo3(signature = (
    timestamps,
    ltp,
    bid,
    ask,
    buy_qty_delta,
    sell_qty_delta,
    oi,
    entries,
    exits,
    symbol = "TICK",
    initial_capital = 100_000.0,
    fees = 0.001,
    slippage = 0.0,
    stop_loss_pct = 5.0,
    take_profit_pct = 10.0,
    max_hold_seconds = 1800_u64,
    entry_cooldown_ticks = 10_usize,
    max_trades = usize::MAX,
))]
// The argument list IS the Python signature; collapsing it into a
// struct would change the public API for no reader benefit.
#[allow(clippy::too_many_arguments)]
pub fn run_tick_backtest<'py>(
    _py: Python<'py>,
    timestamps: PyReadonlyArray1<i64>,
    ltp: PyReadonlyArray1<f64>,
    bid: PyReadonlyArray1<f64>,
    ask: PyReadonlyArray1<f64>,
    buy_qty_delta: PyReadonlyArray1<f64>,
    sell_qty_delta: PyReadonlyArray1<f64>,
    oi: PyReadonlyArray1<f64>,
    entries: PyReadonlyArray1<bool>,
    exits: PyReadonlyArray1<bool>,
    symbol: &str,
    initial_capital: f64,
    fees: f64,
    slippage: f64,
    stop_loss_pct: f64,
    take_profit_pct: f64,
    max_hold_seconds: u64,
    entry_cooldown_ticks: usize,
    max_trades: usize,
) -> PyResult<PyBacktestResult> {
    let tick_data = crate::core::types::TickData {
        timestamps: numpy_to_vec_i64(timestamps),
        ltp: numpy_to_vec_f64(ltp),
        bid: numpy_to_vec_f64(bid),
        ask: numpy_to_vec_f64(ask),
        buy_qty_delta: numpy_to_vec_f64(buy_qty_delta),
        sell_qty_delta: numpy_to_vec_f64(sell_qty_delta),
        oi: numpy_to_vec_f64(oi),
    };

    let entry_signals = numpy_to_vec_bool(entries);
    let exit_signals = numpy_to_vec_bool(exits);

    let config = TickBacktestConfig {
        base: crate::core::types::BacktestConfig {
            initial_capital,
            fees,
            slippage,
            stop: crate::core::types::StopConfig::None,
            target: crate::core::types::TargetConfig::None,
            upon_bar_close: false,
            // The tick engine reads `slippage` directly (see TickBacktest::run)
            // and always honored it, unlike the bar engine.
            ..Default::default()
        },
        stop_loss_pct,
        take_profit_pct,
        max_hold_seconds,
        entry_cooldown_ticks,
        max_trades,
    };

    let backtest = TickBacktest::new(config);
    let result = backtest.run(&tick_data, &entry_signals, &exit_signals, symbol);

    Ok(convert_result(result))
}

// ============================================================================
// Tick Signal Functions
// ============================================================================

/// Compute tick momentum entry signals from per-tick feature arrays.
///
/// All input arrays must have the same length N. Returns a bool array of length N
/// where True indicates a valid entry tick (all gates passed, not in cooldown).
///
/// Gates (each can be disabled by setting threshold to 0.0):
///   - spread_pct[i] <= spread_pct_max
///   - bsi_delta[i] >= bsi_min  (0.0 = disabled)
///   - |return_1m[i]| >= return_1m_min_abs  (0.0 = disabled; NaN always fails)
///   - cooldown_ticks between consecutive entries
///
/// return_direction: +1 for long (needs positive return_1m), -1 for short.
#[pyfunction]
#[pyo3(signature = (
    spread_pct,
    bsi_delta,
    return_1m,
    spread_pct_max = 5.0,
    bsi_min = 0.0,
    return_1m_min_abs = 0.0,
    return_direction = 1_i8,
    cooldown_ticks = 10_usize,
))]
// The argument list IS the Python signature; collapsing it into a
// struct would change the public API for no reader benefit.
#[allow(clippy::too_many_arguments)]
pub fn compute_tick_entry_signals<'py>(
    py: Python<'py>,
    spread_pct: PyReadonlyArray1<f64>,
    bsi_delta: PyReadonlyArray1<f64>,
    return_1m: PyReadonlyArray1<f64>,
    spread_pct_max: f64,
    bsi_min: f64,
    return_1m_min_abs: f64,
    return_direction: i8,
    cooldown_ticks: usize,
) -> PyResult<&'py PyArray1<bool>> {
    let result = crate::signals::tick_signals::tick_momentum_entry(
        &numpy_to_vec_f64(spread_pct),
        &numpy_to_vec_f64(bsi_delta),
        &numpy_to_vec_f64(return_1m),
        spread_pct_max,
        bsi_min,
        return_1m_min_abs,
        return_direction,
        cooldown_ticks,
    );
    Ok(vec_to_numpy_bool(py, result))
}

/// Compute time-based exit signals (EOD / session-end).
///
/// Sets exit[i] = True for every tick with timestamp >= eod_exit_time_ns.
/// Set eod_exit_time_ns = 0 to disable (returns all False).
///
/// timestamps_ns: nanoseconds-since-epoch for each tick (int64 array).
#[pyfunction]
#[pyo3(signature = (timestamps_ns, eod_exit_time_ns = 0_i64))]
pub fn compute_tick_exit_signals<'py>(
    py: Python<'py>,
    timestamps_ns: PyReadonlyArray1<i64>,
    eod_exit_time_ns: i64,
) -> PyResult<&'py PyArray1<bool>> {
    let result = crate::signals::tick_signals::tick_momentum_exit(
        &numpy_to_vec_i64(timestamps_ns),
        eod_exit_time_ns,
    );
    Ok(vec_to_numpy_bool(py, result))
}

// ============================================================================
// Tick Feature Functions
// ============================================================================

/// Per-tick bid/ask spread as percentage of mid price.
/// Returns 0.0 where both bid and ask are zero.
#[pyfunction]
pub fn tick_spread_pct<'py>(
    py: Python<'py>,
    bid: PyReadonlyArray1<f64>,
    ask: PyReadonlyArray1<f64>,
) -> PyResult<&'py PyArray1<f64>> {
    Ok(vec_to_numpy_f64(
        py,
        crate::indicators::tick_features::spread_pct(
            &numpy_to_vec_f64(bid),
            &numpy_to_vec_f64(ask),
        ),
    ))
}

/// Per-tick delta BSI from Zerodha cumulative session totals.
///
/// buy_qty_cumulative / sell_qty_cumulative must be the raw cumulative running sums
/// from Zerodha (NOT already-converted deltas). Returns [0, 1] per tick; 0.5 = neutral.
#[pyfunction]
pub fn buy_sell_imbalance_delta<'py>(
    py: Python<'py>,
    buy_qty_cumulative: PyReadonlyArray1<f64>,
    sell_qty_cumulative: PyReadonlyArray1<f64>,
) -> PyResult<&'py PyArray1<f64>> {
    Ok(vec_to_numpy_f64(
        py,
        crate::indicators::tick_features::buy_sell_imbalance_delta(
            &numpy_to_vec_f64(buy_qty_cumulative),
            &numpy_to_vec_f64(sell_qty_cumulative),
        ),
    ))
}

/// Per-tick lookback return over a time window.
///
/// timestamps_ns: nanoseconds-since-epoch for each tick.
/// Returns NaN for ticks without sufficient history.
#[pyfunction]
#[pyo3(signature = (timestamps_ns, ltp, window_seconds = 60.0))]
pub fn return_window<'py>(
    py: Python<'py>,
    timestamps_ns: PyReadonlyArray1<i64>,
    ltp: PyReadonlyArray1<f64>,
    window_seconds: f64,
) -> PyResult<&'py PyArray1<f64>> {
    Ok(vec_to_numpy_f64(
        py,
        crate::indicators::tick_features::return_window(
            &numpy_to_vec_i64(timestamps_ns),
            &numpy_to_vec_f64(ltp),
            window_seconds,
        ),
    ))
}

/// Rolling realized volatility proxy: stddev of log-returns over a time window (as %).
/// Returns NaN for ticks without at least 2 data points in the window.
#[pyfunction]
#[pyo3(signature = (timestamps_ns, ltp, window_seconds = 300.0))]
pub fn realized_vol_rolling<'py>(
    py: Python<'py>,
    timestamps_ns: PyReadonlyArray1<i64>,
    ltp: PyReadonlyArray1<f64>,
    window_seconds: f64,
) -> PyResult<&'py PyArray1<f64>> {
    Ok(vec_to_numpy_f64(
        py,
        crate::indicators::tick_features::realized_vol_rolling(
            &numpy_to_vec_i64(timestamps_ns),
            &numpy_to_vec_f64(ltp),
            window_seconds,
        ),
    ))
}

/// Per-tick OI position within the day's high/low range: [0, 100].
/// Returns NaN where oi_day_high <= oi_day_low.
#[pyfunction]
pub fn oi_position_pct<'py>(
    py: Python<'py>,
    oi: PyReadonlyArray1<f64>,
    oi_day_high: f64,
    oi_day_low: f64,
) -> PyResult<&'py PyArray1<f64>> {
    Ok(vec_to_numpy_f64(
        py,
        crate::indicators::tick_features::oi_position_pct(
            &numpy_to_vec_f64(oi),
            oi_day_high,
            oi_day_low,
        ),
    ))
}

/// Rolling tick velocity: ticks per minute over the preceding window_seconds.
#[pyfunction]
#[pyo3(signature = (timestamps_ns, window_seconds = 60.0))]
pub fn tick_velocity<'py>(
    py: Python<'py>,
    timestamps_ns: PyReadonlyArray1<i64>,
    window_seconds: f64,
) -> PyResult<&'py PyArray1<f64>> {
    Ok(vec_to_numpy_f64(
        py,
        crate::indicators::tick_features::tick_velocity(
            &numpy_to_vec_i64(timestamps_ns),
            window_seconds,
        ),
    ))
}

// ============================================================================
// Indicator Functions
// ============================================================================

/// Simple Moving Average.
#[pyfunction]
pub fn sma<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<f64>,
    period: usize,
) -> PyResult<&'py PyArray1<f64>> {
    let vec = numpy_to_vec_f64(data);
    let result = indicators::trend::sma(&vec, period)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(vec_to_numpy_f64(py, result))
}

/// Exponential Moving Average.
#[pyfunction]
pub fn ema<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<f64>,
    period: usize,
) -> PyResult<&'py PyArray1<f64>> {
    let vec = numpy_to_vec_f64(data);
    let result = indicators::trend::ema(&vec, period)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(vec_to_numpy_f64(py, result))
}

/// Relative Strength Index.
#[pyfunction]
pub fn rsi<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<f64>,
    period: usize,
) -> PyResult<&'py PyArray1<f64>> {
    let vec = numpy_to_vec_f64(data);
    let result = indicators::momentum::rsi(&vec, period)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(vec_to_numpy_f64(py, result))
}

/// MACD indicator.
#[pyfunction]
#[pyo3(signature = (data, fast_period=12, slow_period=26, signal_period=9))]
pub fn macd<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<f64>,
    fast_period: usize,
    slow_period: usize,
    signal_period: usize,
) -> PyResult<(&'py PyArray1<f64>, &'py PyArray1<f64>, &'py PyArray1<f64>)> {
    let vec = numpy_to_vec_f64(data);
    let result = indicators::momentum::macd(&vec, fast_period, slow_period, signal_period)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok((
        vec_to_numpy_f64(py, result.macd_line),
        vec_to_numpy_f64(py, result.signal_line),
        vec_to_numpy_f64(py, result.histogram),
    ))
}

/// Stochastic oscillator.
#[pyfunction]
#[pyo3(signature = (high, low, close, k_period=14, d_period=3))]
pub fn stochastic<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<f64>,
    low: PyReadonlyArray1<f64>,
    close: PyReadonlyArray1<f64>,
    k_period: usize,
    d_period: usize,
) -> PyResult<(&'py PyArray1<f64>, &'py PyArray1<f64>)> {
    let h = numpy_to_vec_f64(high);
    let l = numpy_to_vec_f64(low);
    let c = numpy_to_vec_f64(close);
    let result = indicators::momentum::stochastic(&h, &l, &c, k_period, d_period)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok((vec_to_numpy_f64(py, result.k), vec_to_numpy_f64(py, result.d)))
}

/// Average True Range.
#[pyfunction]
pub fn atr<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<f64>,
    low: PyReadonlyArray1<f64>,
    close: PyReadonlyArray1<f64>,
    period: usize,
) -> PyResult<&'py PyArray1<f64>> {
    let h = numpy_to_vec_f64(high);
    let l = numpy_to_vec_f64(low);
    let c = numpy_to_vec_f64(close);
    let result = indicators::volatility::atr(&h, &l, &c, period)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(vec_to_numpy_f64(py, result))
}

/// Bollinger Bands.
#[pyfunction]
#[pyo3(signature = (data, period=20, std_dev=2.0))]
pub fn bollinger_bands<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<f64>,
    period: usize,
    std_dev: f64,
) -> PyResult<(&'py PyArray1<f64>, &'py PyArray1<f64>, &'py PyArray1<f64>)> {
    let vec = numpy_to_vec_f64(data);
    let result = indicators::volatility::bollinger_bands(&vec, period, std_dev)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok((
        vec_to_numpy_f64(py, result.upper),
        vec_to_numpy_f64(py, result.middle),
        vec_to_numpy_f64(py, result.lower),
    ))
}

/// Average Directional Index.
#[pyfunction]
pub fn adx<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<f64>,
    low: PyReadonlyArray1<f64>,
    close: PyReadonlyArray1<f64>,
    period: usize,
) -> PyResult<&'py PyArray1<f64>> {
    let h = numpy_to_vec_f64(high);
    let l = numpy_to_vec_f64(low);
    let c = numpy_to_vec_f64(close);
    let result = indicators::strength::adx(&h, &l, &c, period)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(vec_to_numpy_f64(py, result))
}

/// Volume Weighted Average Price.
#[pyfunction]
pub fn vwap<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<f64>,
    low: PyReadonlyArray1<f64>,
    close: PyReadonlyArray1<f64>,
    volume: PyReadonlyArray1<f64>,
) -> PyResult<&'py PyArray1<f64>> {
    let h = numpy_to_vec_f64(high);
    let l = numpy_to_vec_f64(low);
    let c = numpy_to_vec_f64(close);
    let v = numpy_to_vec_f64(volume);
    let result = indicators::volume::vwap(&h, &l, &c, &v)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(vec_to_numpy_f64(py, result))
}

/// Supertrend indicator.
#[pyfunction]
#[pyo3(signature = (high, low, close, period=10, multiplier=3.0))]
pub fn supertrend<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<f64>,
    low: PyReadonlyArray1<f64>,
    close: PyReadonlyArray1<f64>,
    period: usize,
    multiplier: f64,
) -> PyResult<(&'py PyArray1<f64>, &'py PyArray1<i8>)> {
    let h = numpy_to_vec_f64(high);
    let l = numpy_to_vec_f64(low);
    let c = numpy_to_vec_f64(close);
    let result = indicators::trend::supertrend(&h, &l, &c, period, multiplier)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    let direction_array = PyArray1::from_vec(py, result.direction);
    Ok((vec_to_numpy_f64(py, result.supertrend), direction_array))
}

/// Rolling minimum (Lowest Low Value).
#[pyfunction]
pub fn rolling_min<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<f64>,
    period: usize,
) -> PyResult<&'py PyArray1<f64>> {
    let vec = numpy_to_vec_f64(data);
    let result = indicators::rolling::rolling_min(&vec, period)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(vec_to_numpy_f64(py, result))
}

/// Rolling maximum (Highest High Value).
#[pyfunction]
pub fn rolling_max<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<f64>,
    period: usize,
) -> PyResult<&'py PyArray1<f64>> {
    let vec = numpy_to_vec_f64(data);
    let result = indicators::rolling::rolling_max(&vec, period)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(vec_to_numpy_f64(py, result))
}

// ============================================================================
// Helper Functions
// ============================================================================

// ============================================================================
// Monte Carlo Forward Simulation
// ============================================================================

// The argument list IS the Python signature; collapsing it into a
// struct would change the public API for no reader benefit.
#[allow(clippy::too_many_arguments)]
/// Run Monte Carlo forward simulation for a portfolio.
///
/// Uses Geometric Brownian Motion with Cholesky-decomposed correlated random
/// draws, parallelized via Rayon.
///
/// # Arguments
/// * `returns` - List of per-strategy return arrays (N strategies)
/// * `weights` - Portfolio weight vector (length N, sums to 1)
/// * `correlation_matrix` - N x N correlation matrix (flattened row-major as 2D list)
/// * `initial_value` - Starting portfolio value
/// * `n_simulations` - Number of simulation paths (default: 10000)
/// * `horizon_days` - Forward simulation horizon in trading days (default: 252)
/// * `seed` - Random seed for reproducibility (default: 42)
#[pyfunction]
#[pyo3(signature = (returns, weights, correlation_matrix, initial_value, n_simulations=10000, horizon_days=252, seed=42))]
pub fn simulate_portfolio_mc(
    py: Python<'_>,
    returns: Vec<PyReadonlyArray1<'_, f64>>,
    weights: PyReadonlyArray1<'_, f64>,
    correlation_matrix: Vec<PyReadonlyArray1<'_, f64>>,
    initial_value: f64,
    n_simulations: usize,
    horizon_days: usize,
    seed: u64,
) -> PyResult<PyObject> {
    use crate::portfolio::monte_carlo::{simulate_portfolio_forward, MonteCarloConfig};

    // Convert numpy arrays to Rust vecs
    let rust_returns: Vec<Vec<f64>> =
        returns.iter().map(|arr| arr.as_slice().unwrap().to_vec()).collect();

    let rust_weights: Vec<f64> = weights.as_slice().unwrap().to_vec();

    let rust_corr: Vec<Vec<f64>> =
        correlation_matrix.iter().map(|arr| arr.as_slice().unwrap().to_vec()).collect();

    // Refuse mismatched shapes here rather than indexing past the end inside a
    // Rayon worker. A panic on a worker thread crosses PyO3 as PanicException,
    // which is neither catchable as ValueError nor traceable to the argument
    // that was wrong -- and the caller most likely passed an (n_obs, n_assets)
    // matrix where a per-asset list of series was expected.
    let n_assets = rust_returns.len();
    if n_assets == 0 {
        return Err(PyValueError::new_err("returns must contain at least one asset series"));
    }
    if rust_weights.len() != n_assets {
        return Err(PyValueError::new_err(format!(
            "weights has {} entries but returns has {n_assets} asset series; \
             returns is a list of per-asset series, not an (n_obs, n_assets) matrix",
            rust_weights.len()
        )));
    }
    if rust_corr.len() != n_assets || rust_corr.iter().any(|row| row.len() != n_assets) {
        return Err(PyValueError::new_err(format!(
            "correlation_matrix must be {n_assets}x{n_assets} to match the {n_assets} asset series"
        )));
    }

    let config = MonteCarloConfig { n_simulations, horizon_days, seed };

    // Run simulation (releases GIL for Rayon parallelism)
    let result = py
        .allow_threads(|| {
            simulate_portfolio_forward(
                &rust_returns,
                &rust_weights,
                &rust_corr,
                initial_value,
                &config,
            )
        })
        .map_err(PyValueError::new_err)?;

    // Build Python dict result
    let dict = pyo3::types::PyDict::new(py);

    // percentile_paths: list of (percentile, list[float])
    let paths_list = pyo3::types::PyList::empty(py);
    for (pct, path) in &result.percentile_paths {
        let path_list = pyo3::types::PyList::new(py, path);
        let tuple = pyo3::types::PyTuple::new(py, &[pct.to_object(py), path_list.to_object(py)]);
        paths_list.append(tuple)?;
    }
    dict.set_item("percentile_paths", paths_list)?;

    // final_values as numpy array for efficiency
    let final_arr = PyArray1::from_vec(py, result.final_values);
    dict.set_item("final_values", final_arr)?;

    dict.set_item("expected_return", result.expected_return)?;
    dict.set_item("probability_of_loss", result.probability_of_loss)?;
    dict.set_item("var_95", result.var_95)?;
    dict.set_item("cvar_95", result.cvar_95)?;

    Ok(dict.into())
}

/// Map a non-finite metric to `None`.
///
/// Ratios divide by a denominator that can legitimately be zero -- a strategy
/// with no losing trades has an undefined profit factor, not an infinite one.
/// `f64::INFINITY` crosses to Python as `float('inf')`, which `json.dumps`
/// serializes as a bare `Infinity` token that is not valid JSON.
fn finite(value: f64) -> Option<f64> {
    if value.is_finite() {
        Some(value)
    } else {
        None
    }
}

/// Convert a Rust trade to its Python counterpart.
pub(crate) fn convert_trade(t: crate::core::types::Trade) -> PyTrade {
    PyTrade {
        id: t.id,
        symbol: t.symbol,
        entry_idx: t.entry_idx,
        exit_idx: t.exit_idx,
        entry_price: t.entry_price,
        exit_price: t.exit_price,
        size: t.size,
        direction: t.direction as i32,
        pnl: t.pnl,
        return_pct: t.return_pct,
        entry_time: t.entry_time,
        exit_time: t.exit_time,
        fees: t.fees,
        entry_fees: t.entry_fees,
        exit_fees: t.exit_fees,
        fee_breakdown: t.fee_breakdown.map(|b| {
            HashMap::from([
                ("brokerage".to_string(), b.brokerage),
                ("stt".to_string(), b.stt),
                ("exchange_txn".to_string(), b.exchange_txn),
                ("sebi_fee".to_string(), b.sebi_fee),
                ("stamp_duty".to_string(), b.stamp_duty),
                ("gst".to_string(), b.gst),
                ("total".to_string(), b.total()),
            ])
        }),
        exit_reason: format!("{:?}", t.exit_reason),
    }
}

/// Convert Rust BacktestResult to Python PyBacktestResult.
pub(crate) fn convert_result(result: crate::core::types::BacktestResult) -> PyBacktestResult {
    let metrics = PyBacktestMetrics {
        total_return_pct: result.metrics.total_return_pct,
        sharpe_ratio: result.metrics.sharpe_ratio,
        sortino_ratio: finite(result.metrics.sortino_ratio),
        calmar_ratio: finite(result.metrics.calmar_ratio),
        omega_ratio: finite(result.metrics.omega_ratio),
        max_drawdown_pct: result.metrics.max_drawdown_pct,
        max_drawdown_duration: result.metrics.max_drawdown_duration,
        win_rate_pct: result.metrics.win_rate_pct,
        profit_factor: finite(result.metrics.profit_factor),
        expectancy: result.metrics.expectancy,
        sqn: result.metrics.sqn,
        total_trades: result.metrics.total_trades,
        total_closed_trades: result.metrics.total_closed_trades,
        total_open_trades: result.metrics.total_open_trades,
        open_trade_pnl: result.metrics.open_trade_pnl,
        winning_trades: result.metrics.winning_trades,
        losing_trades: result.metrics.losing_trades,
        start_value: result.metrics.start_value,
        end_value: result.metrics.end_value,
        total_fees_paid: result.metrics.total_fees_paid,
        best_trade_pct: result.metrics.best_trade_pct,
        worst_trade_pct: result.metrics.worst_trade_pct,
        avg_trade_return_pct: result.metrics.avg_trade_return_pct,
        avg_win_pct: result.metrics.avg_win_pct,
        avg_loss_pct: result.metrics.avg_loss_pct,
        avg_winning_duration: result.metrics.avg_winning_duration,
        avg_losing_duration: result.metrics.avg_losing_duration,
        max_consecutive_wins: result.metrics.max_consecutive_wins,
        max_consecutive_losses: result.metrics.max_consecutive_losses,
        avg_holding_period: result.metrics.avg_holding_period,
        exposure_pct: result.metrics.exposure_pct,
        payoff_ratio: finite(result.metrics.payoff_ratio),
        recovery_factor: finite(result.metrics.recovery_factor),
    };

    let trades: Vec<PyTrade> = result.trades.into_iter().map(convert_trade).collect();

    PyBacktestResult {
        metrics,
        equity_curve: result.equity_curve,
        drawdown_curve: result.drawdown_curve,
        trades,
        returns: result.returns,
    }
}
