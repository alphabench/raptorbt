//! Python bindings for the per-bar strategy session.
//!
//! Exposes [`SingleRunner`] to Python as [`PyKernelSession`], the execution
//! core behind the class-based strategy contract: a Python driver loop feeds
//! bars and per-bar order inputs, and receives engine events to dispatch to
//! strategy hooks. Result accounting is shared with the array-based runners,
//! so both paths produce identical metrics for identical decisions.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::core::types::{BacktestConfig, Direction, InstrumentConfig, StopConfig, TargetConfig};
use crate::portfolio::kernel::{EngineEvent, KernelBar, StepInput};
use crate::portfolio::runner::SingleRunner;

use super::bindings::{
    convert_result, convert_trade, PyBacktestConfig, PyBacktestResult, PyInstrumentConfig, PyTrade,
};

/// One observable outcome of a session step.
///
/// `kind` is `"entered"`, `"exited"`, or `"entry_rejected"`; the optional
/// fields are populated according to the kind.
#[pyclass]
#[derive(Debug, Clone)]
pub struct PyEngineEvent {
    #[pyo3(get)]
    pub kind: String,
    #[pyo3(get)]
    pub idx: usize,
    /// Fill price, for `entered` events.
    #[pyo3(get)]
    pub price: Option<f64>,
    /// Position size, for `entered` events.
    #[pyo3(get)]
    pub size: Option<f64>,
    /// Trade direction (1 long, -1 short), for `entered` events.
    #[pyo3(get)]
    pub direction: Option<i32>,
    /// Completed trade, for `exited` events.
    #[pyo3(get)]
    pub trade: Option<PyTrade>,
    /// Refusal reason, for `entry_rejected` events.
    #[pyo3(get)]
    pub reject_reason: Option<String>,
}

#[pymethods]
impl PyEngineEvent {
    fn __repr__(&self) -> String {
        format!("EngineEvent(kind={}, idx={})", self.kind, self.idx)
    }
}

impl From<EngineEvent> for PyEngineEvent {
    fn from(event: EngineEvent) -> Self {
        match event {
            EngineEvent::Entered { idx, price, size, direction } => Self {
                kind: "entered".to_string(),
                idx,
                price: Some(price),
                size: Some(size),
                direction: Some(direction as i32),
                trade: None,
                reject_reason: None,
            },
            EngineEvent::Exited { idx, trade } => Self {
                kind: "exited".to_string(),
                idx,
                price: Some(trade.exit_price),
                size: Some(trade.size),
                direction: Some(trade.direction as i32),
                trade: Some(convert_trade(trade)),
                reject_reason: None,
            },
            EngineEvent::EntryRejected { idx, reason } => Self {
                kind: "entry_rejected".to_string(),
                idx,
                price: None,
                size: None,
                direction: None,
                trade: None,
                reject_reason: Some(format!("{reason:?}")),
            },
        }
    }
}

/// Read-only view of the session's open position.
#[pyclass]
#[derive(Debug, Clone)]
pub struct PyPositionSnapshot {
    #[pyo3(get)]
    pub entry_idx: usize,
    #[pyo3(get)]
    pub entry_price: f64,
    #[pyo3(get)]
    pub size: f64,
    /// 1 for long, -1 for short.
    #[pyo3(get)]
    pub direction: i32,
    #[pyo3(get)]
    pub stop_price: Option<f64>,
    #[pyo3(get)]
    pub target_price: Option<f64>,
}

#[pymethods]
impl PyPositionSnapshot {
    fn __repr__(&self) -> String {
        format!(
            "PositionSnapshot(entry_idx={}, entry_price={:.2}, size={:.2}, direction={})",
            self.entry_idx, self.entry_price, self.size, self.direction
        )
    }
}

/// Per-bar simulation session for one instrument.
///
/// Drive it by calling [`PyKernelSession::step`] once per bar in ascending
/// order, then [`PyKernelSession::finish`] to obtain the standard backtest
/// result. Scalars cross the boundary per bar, so a Python driver loop pays
/// one FFI call per bar with no array allocation.
#[pyclass]
pub struct PyKernelSession {
    runner: Option<SingleRunner>,
}

#[pymethods]
impl PyKernelSession {
    #[new]
    #[pyo3(signature = (symbol="ASSET", direction=1, config=None, instrument_config=None))]
    fn new(
        symbol: &str,
        direction: i32,
        config: Option<&PyBacktestConfig>,
        instrument_config: Option<&PyInstrumentConfig>,
    ) -> PyResult<Self> {
        let direction = match direction {
            1 => Direction::Long,
            -1 => Direction::Short,
            other => {
                return Err(PyValueError::new_err(format!(
                    "direction must be 1 (long) or -1 (short), got {other}"
                )))
            }
        };

        let rust_config: BacktestConfig = config.map(BacktestConfig::from).unwrap_or_default();
        let inst: Option<InstrumentConfig> = instrument_config.map(InstrumentConfig::from);

        let runner =
            SingleRunner::from_config(rust_config, symbol.to_string(), direction, inst.as_ref());

        Ok(Self { runner: Some(runner) })
    }

    /// Advance the session by one bar.
    ///
    /// `entry`/`exit` carry the strategy's order intents for this bar;
    /// `stop_price`/`target_price` optionally pin explicit exit levels for an
    /// entry opened on this bar, overriding the configured stop/target models.
    #[pyo3(signature = (
        idx, timestamp, open, high, low, close, volume,
        entry=false, exit=false, atr=0.0, size_mult=None,
        stop_price=None, target_price=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn step(
        &mut self,
        idx: usize,
        timestamp: i64,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
        entry: bool,
        exit: bool,
        atr: f64,
        size_mult: Option<f64>,
        stop_price: Option<f64>,
        target_price: Option<f64>,
    ) -> PyResult<Vec<PyEngineEvent>> {
        let runner = self
            .runner
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("session is finished; create a new one"))?;

        let bar = KernelBar { timestamp, open, high, low, close, volume };
        let input = StepInput {
            entry,
            exit,
            atr,
            size_mult,
            stop_price_override: stop_price,
            target_price_override: target_price,
        };

        Ok(runner.step(idx, &bar, input).into_iter().map(PyEngineEvent::from).collect())
    }

    /// Overwrite the open position's stop price; no-op when flat.
    fn set_stop_price(&mut self, price: Option<f64>) -> PyResult<()> {
        self.runner_mut()?.kernel_mut().set_stop_price(price);
        Ok(())
    }

    /// Overwrite the open position's target price; no-op when flat.
    fn set_target_price(&mut self, price: Option<f64>) -> PyResult<()> {
        self.runner_mut()?.kernel_mut().set_target_price(price);
        Ok(())
    }

    /// Mark-to-market equity after the most recent step.
    fn equity(&self) -> PyResult<f64> {
        Ok(self.runner_ref()?.equity())
    }

    /// Current uninvested cash.
    fn cash(&self) -> PyResult<f64> {
        Ok(self.runner_ref()?.cash())
    }

    /// Whether a position is currently open.
    fn is_in_position(&self) -> PyResult<bool> {
        Ok(self.runner_ref()?.is_in_position())
    }

    /// Read-only view of the open position, or `None` when flat.
    fn position(&self) -> PyResult<Option<PyPositionSnapshot>> {
        Ok(self.runner_ref()?.kernel().position_snapshot().map(|p| PyPositionSnapshot {
            entry_idx: p.entry_idx,
            entry_price: p.entry_price,
            size: p.size,
            direction: p.direction as i32,
            stop_price: p.stop_price,
            target_price: p.target_price,
        }))
    }

    /// Force-close any open position and compute final metrics.
    ///
    /// Consumes the session; further calls raise.
    fn finish(&mut self) -> PyResult<PyBacktestResult> {
        let runner = self
            .runner
            .take()
            .ok_or_else(|| PyValueError::new_err("session is already finished"))?;
        Ok(convert_result(runner.finish()))
    }
}

impl PyKernelSession {
    fn runner_ref(&self) -> PyResult<&SingleRunner> {
        self.runner
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("session is finished; create a new one"))
    }

    fn runner_mut(&mut self) -> PyResult<&mut SingleRunner> {
        self.runner
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("session is finished; create a new one"))
    }
}

/// Resolve the ATR period a backtest would use for stop/target computation.
///
/// Mirrors the batch engine's resolution: per-instrument stop/target override
/// the global config; the stop's period wins over the target's; `None` when
/// neither is ATR-based. The Python strategy runner uses this to precompute
/// the same ATR series the array path would.
#[pyfunction]
#[pyo3(signature = (config=None, instrument_config=None))]
pub fn resolve_atr_period(
    config: Option<&PyBacktestConfig>,
    instrument_config: Option<&PyInstrumentConfig>,
) -> Option<usize> {
    let rust_config: BacktestConfig = config.map(BacktestConfig::from).unwrap_or_default();
    let inst: Option<InstrumentConfig> = instrument_config.map(InstrumentConfig::from);

    let effective_stop =
        inst.as_ref().and_then(|ic| ic.stop.as_ref()).copied().unwrap_or(rust_config.stop);
    let effective_target =
        inst.as_ref().and_then(|ic| ic.target.as_ref()).copied().unwrap_or(rust_config.target);

    match (effective_stop, effective_target) {
        (StopConfig::Atr { period, .. }, _) => Some(period),
        (_, TargetConfig::Atr { period, .. }) => Some(period),
        _ => None,
    }
}
