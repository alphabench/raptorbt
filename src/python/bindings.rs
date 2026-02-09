//! PyO3 function bindings for RaptorBT.

use numpy::{PyArray1, PyReadonlyArray1};
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
use crate::strategies::spreads::{
    LegConfig, OptionType as SpreadOptionType, SpreadBacktest, SpreadConfig, SpreadType,
};

use super::numpy_bridge::*;

// ============================================================================
// Configuration Classes
// ============================================================================

/// Python-exposed backtest configuration.
#[pyclass]
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
    stop_config: StopConfig,
    target_config: TargetConfig,
}

#[pymethods]
impl PyBacktestConfig {
    #[new]
    #[pyo3(signature = (initial_capital=100000.0, fees=0.001, slippage=0.0, upon_bar_close=true))]
    fn new(initial_capital: f64, fees: f64, slippage: f64, upon_bar_close: bool) -> Self {
        Self {
            initial_capital,
            fees,
            slippage,
            upon_bar_close,
            stop_config: StopConfig::None,
            target_config: TargetConfig::None,
        }
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

impl From<&PyBacktestConfig> for BacktestConfig {
    fn from(py_config: &PyBacktestConfig) -> Self {
        BacktestConfig {
            initial_capital: py_config.initial_capital,
            fees: py_config.fees,
            slippage: py_config.slippage,
            stop: py_config.stop_config,
            target: py_config.target_config,
            upon_bar_close: py_config.upon_bar_close,
        }
    }
}

/// Python-exposed per-instrument configuration.
#[pyclass]
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
#[pyclass]
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
#[pyclass]
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
#[pyclass]
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
    #[pyo3(get)]
    pub fees: f64,
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
#[pyclass]
#[derive(Debug, Clone)]
pub struct PyBacktestMetrics {
    #[pyo3(get)]
    pub total_return_pct: f64,
    #[pyo3(get)]
    pub sharpe_ratio: f64,
    #[pyo3(get)]
    pub sortino_ratio: f64,
    #[pyo3(get)]
    pub calmar_ratio: f64,
    #[pyo3(get)]
    pub omega_ratio: f64,
    #[pyo3(get)]
    pub max_drawdown_pct: f64,
    #[pyo3(get)]
    pub max_drawdown_duration: usize,
    #[pyo3(get)]
    pub win_rate_pct: f64,
    #[pyo3(get)]
    pub profit_factor: f64,
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
}

#[pymethods]
impl PyBacktestMetrics {
    fn __repr__(&self) -> String {
        format!(
            "BacktestMetrics(return={:.2}%, sharpe={:.2}, max_dd={:.2}%, trades={})",
            self.total_return_pct, self.sharpe_ratio, self.max_drawdown_pct, self.total_trades
        )
    }

    /// Convert to dictionary matching VectorBT stats() format.
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
#[pyclass]
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

/// Run single instrument backtest.
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

    let dir = Direction::from_int(direction).unwrap_or(Direction::Long);

    let signals = CompiledSignals {
        symbol: symbol.to_string(),
        entries: numpy_to_vec_bool(entries),
        exits: numpy_to_vec_bool(exits),
        position_sizes: position_sizes.map(numpy_to_vec_f64),
        direction: dir,
        weight,
    };

    let rust_config = config.map(|c| BacktestConfig::from(c)).unwrap_or_default();
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
    instruments: Vec<(
        PyReadonlyArray1<i64>,
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<bool>,
        PyReadonlyArray1<bool>,
        i32,
        f64,
        String,
    )>,
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
                direction: Direction::from_int(dir).unwrap_or(Direction::Long),
                weight,
            };
            (ohlcv, signals)
        })
        .collect();

    let mode = match sync_mode {
        "any" => SyncMode::Any,
        "majority" => SyncMode::Majority,
        "master" => SyncMode::Master,
        _ => SyncMode::All,
    };

    let basket_config = BasketConfig {
        base: config.map(|c| BacktestConfig::from(c)).unwrap_or_default(),
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

    let dir = Direction::from_int(direction).unwrap_or(Direction::Long);

    let signals = CompiledSignals {
        symbol: symbol.to_string(),
        entries: numpy_to_vec_bool(entries),
        exits: numpy_to_vec_bool(exits),
        position_sizes: None,
        direction: dir,
        weight: 1.0,
    };

    let opt_type = match option_type {
        "put" => OptionType::Put,
        _ => OptionType::Call,
    };

    let strike_sel = match strike_selection {
        "otm1" => StrikeSelection::Otm(1),
        "otm2" => StrikeSelection::Otm(2),
        "itm1" => StrikeSelection::Itm(1),
        "itm2" => StrikeSelection::Itm(2),
        _ => StrikeSelection::Atm,
    };

    let size = match size_type {
        "contracts" => SizeType::Contracts(size_value as usize),
        "notional" => SizeType::Notional(size_value),
        "risk" => SizeType::RiskPercent(size_value),
        _ => SizeType::Percent(size_value),
    };

    let options_config = OptionsConfig {
        base: config.map(|c| BacktestConfig::from(c)).unwrap_or_default(),
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

    let dir = Direction::from_int(direction).unwrap_or(Direction::Long);

    let signals = CompiledSignals {
        symbol: symbol.to_string(),
        entries: numpy_to_vec_bool(entries),
        exits: numpy_to_vec_bool(exits),
        position_sizes: None,
        direction: dir,
        weight: 1.0,
    };

    let pairs_config = PairsConfig {
        base: config.map(|c| BacktestConfig::from(c)).unwrap_or_default(),
        hedge_ratio,
        dynamic_hedge,
        ..Default::default()
    };

    let backtest = PairsBacktest::new(pairs_config);
    let result = backtest.run(&leg1_ohlcv, &leg2_ohlcv, &signals);

    Ok(convert_result(result))
}

/// Run spread backtest (multi-leg options).
#[pyfunction]
#[pyo3(signature = (timestamps, underlying_close, legs_premiums, leg_configs, entries, exits, config=None, spread_type="custom", max_loss=None, target_profit=None))]
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
            let option_type =
                SpreadOptionType::from_str(&opt_type).unwrap_or(SpreadOptionType::Call);
            LegConfig::new(option_type, strike, quantity, lot_size)
        })
        .collect();

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
        _ => SpreadType::Custom,
    };

    let spread_config = SpreadConfig {
        base: config.map(|c| BacktestConfig::from(c)).unwrap_or_default(),
        spread_type: spread_type_enum,
        leg_configs: rust_leg_configs,
        max_loss,
        target_profit,
        close_at_eod: false,
    };

    let backtest = SpreadBacktest::new(spread_config);
    let result = backtest.run(&ts, &underlying, &premiums, &entry_signals, &exit_signals);

    Ok(convert_result(result))
}

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
    strategies: Vec<(PyReadonlyArray1<bool>, PyReadonlyArray1<bool>, i32, f64, String)>,
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
        .map(|(entries, exits, dir, weight, symbol)| CompiledSignals {
            symbol,
            entries: numpy_to_vec_bool(entries),
            exits: numpy_to_vec_bool(exits),
            position_sizes: None,
            direction: Direction::from_int(dir).unwrap_or(Direction::Long),
            weight,
        })
        .collect();

    let mode = match combine_mode {
        "all" => CombineMode::All,
        "majority" => CombineMode::Majority,
        "independent" => CombineMode::Independent,
        "weighted" => CombineMode::Weighted,
        _ => CombineMode::Any,
    };

    let multi_config = MultiStrategyConfig {
        base: config.map(|c| BacktestConfig::from(c)).unwrap_or_default(),
        combine_mode: mode,
        ..Default::default()
    };

    let backtest = MultiStrategyBacktest::new(multi_config);
    let result = backtest.run(&ohlcv, &rust_strategies);

    Ok(convert_result(result))
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

/// Convert Rust BacktestResult to Python PyBacktestResult.
fn convert_result(result: crate::core::types::BacktestResult) -> PyBacktestResult {
    let metrics = PyBacktestMetrics {
        total_return_pct: result.metrics.total_return_pct,
        sharpe_ratio: result.metrics.sharpe_ratio,
        sortino_ratio: result.metrics.sortino_ratio,
        calmar_ratio: result.metrics.calmar_ratio,
        omega_ratio: result.metrics.omega_ratio,
        max_drawdown_pct: result.metrics.max_drawdown_pct,
        max_drawdown_duration: result.metrics.max_drawdown_duration,
        win_rate_pct: result.metrics.win_rate_pct,
        profit_factor: result.metrics.profit_factor,
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
    };

    let trades: Vec<PyTrade> = result
        .trades
        .into_iter()
        .map(|t| PyTrade {
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
            exit_reason: format!("{:?}", t.exit_reason),
        })
        .collect();

    PyBacktestResult {
        metrics,
        equity_curve: result.equity_curve,
        drawdown_curve: result.drawdown_curve,
        trades,
        returns: result.returns,
    }
}
