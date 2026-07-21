//! Python bindings for the multi-instrument event session.

use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::core::types::{BacktestConfig, Direction, InstrumentConfig};
use crate::portfolio::kernel::{KernelBar, StepInput};
use crate::portfolio::ledger::PositionPolicy;
use crate::portfolio::session::EventSession;

use super::bindings::{
    convert_result, PyBacktestConfig, PyInstrumentConfig, PyInstrumentSummary, PyPortfolioResult,
};
use super::instrument_bindings::PyInstrumentSpec;
use super::numpy_bridge::{numpy_to_vec_f64, numpy_to_vec_i64};
use super::strategy_bindings::{submit_order_on, PyEngineEvent, PyPositionSnapshot};

/// Multi-instrument session over deterministically merged bar streams.
///
/// One shared cash pool across instruments (cash accounts only). Protocol:
/// `add_instrument` + `set_bars` per symbol, `seal()`, then loop
/// `current()` / `apply_current(...)` to completion and `finish()`.
#[pyclass]
pub struct PyPortfolioSession {
    session: Option<EventSession>,
}

impl PyPortfolioSession {
    fn session_ref(&self) -> PyResult<&EventSession> {
        self.session
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("session is finished; create a new one"))
    }

    fn session_mut(&mut self) -> PyResult<&mut EventSession> {
        self.session
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("session is finished; create a new one"))
    }
}

#[pymethods]
impl PyPortfolioSession {
    #[new]
    #[pyo3(signature = (config=None))]
    fn new(config: Option<&PyBacktestConfig>) -> Self {
        let rust_config: BacktestConfig = config.map(BacktestConfig::from).unwrap_or_default();
        Self { session: Some(EventSession::new(rust_config)) }
    }

    /// Register an instrument; returns its index for routing.
    #[pyo3(signature = (symbol, direction=1, instrument_config=None, instrument=None, oms_type="netting"))]
    fn add_instrument(
        &mut self,
        symbol: &str,
        direction: i32,
        instrument_config: Option<&PyInstrumentConfig>,
        instrument: Option<&PyInstrumentSpec>,
        oms_type: &str,
    ) -> PyResult<usize> {
        let direction = Direction::from_int(direction)
            .ok_or_else(|| PyValueError::new_err("direction must be 1 or -1"))?;
        let policy = match oms_type {
            "netting" => PositionPolicy::Net,
            "hedging" => PositionPolicy::Independent,
            other => {
                return Err(PyValueError::new_err(format!(
                    "oms_type must be 'netting' or 'hedging', got {other:?}"
                )))
            }
        };
        let spec = match instrument {
            Some(py_spec) => {
                let spec = crate::instruments::InstrumentSpec::from(py_spec);
                if !spec.kind.tradable() {
                    return Err(PyValueError::new_err(format!(
                        "instrument {:?} is not tradable",
                        spec.symbol
                    )));
                }
                Some(spec)
            }
            None => None,
        };
        let inst: Option<InstrumentConfig> = instrument_config.map(InstrumentConfig::from);
        Ok(self.session_mut()?.add_instrument(
            symbol.to_string(),
            direction,
            spec,
            inst.as_ref(),
            policy,
        ))
    }

    /// Attach an instrument's bar arrays (ascending timestamps).
    #[allow(clippy::too_many_arguments)]
    fn set_bars(
        &mut self,
        instrument: usize,
        timestamps: PyReadonlyArray1<i64>,
        open: PyReadonlyArray1<f64>,
        high: PyReadonlyArray1<f64>,
        low: PyReadonlyArray1<f64>,
        close: PyReadonlyArray1<f64>,
        volume: PyReadonlyArray1<f64>,
    ) -> PyResult<()> {
        let ts = numpy_to_vec_i64(timestamps);
        let o = numpy_to_vec_f64(open);
        let h = numpy_to_vec_f64(high);
        let l = numpy_to_vec_f64(low);
        let c = numpy_to_vec_f64(close);
        let v = numpy_to_vec_f64(volume);
        let n = ts.len();
        if [o.len(), h.len(), l.len(), c.len(), v.len()].iter().any(|&len| len != n) {
            return Err(PyValueError::new_err("all bar arrays must share one length"));
        }
        let bars: Vec<KernelBar> = (0..n)
            .map(|i| KernelBar {
                timestamp: ts[i],
                open: o[i],
                high: h[i],
                low: l[i],
                close: c[i],
                volume: v[i],
            })
            .collect();
        self.session_mut()?.set_bars(instrument, bars);
        Ok(())
    }

    /// Merge all streams into the deterministic schedule.
    fn seal(&mut self) -> PyResult<()> {
        self.session_mut()?.seal();
        Ok(())
    }

    /// Number of scheduled events.
    fn __len__(&self) -> PyResult<usize> {
        Ok(self.session_ref()?.len())
    }

    /// The pending event: `(instrument, local_idx, ts, o, h, l, c, v)`.
    #[allow(clippy::type_complexity)]
    fn current(&self) -> PyResult<Option<(usize, usize, i64, f64, f64, f64, f64, f64)>> {
        Ok(self.session_ref()?.current().map(|e| {
            (
                e.instrument,
                e.local_idx,
                e.bar.timestamp,
                e.bar.open,
                e.bar.high,
                e.bar.low,
                e.bar.close,
                e.bar.volume,
            )
        }))
    }

    /// Step the pending event through its instrument's kernel and advance.
    #[pyo3(signature = (entry=false, exit=false, atr=0.0, size_mult=None, stop_price=None, target_price=None))]
    fn apply_current(
        &mut self,
        entry: bool,
        exit: bool,
        atr: f64,
        size_mult: Option<f64>,
        stop_price: Option<f64>,
        target_price: Option<f64>,
    ) -> PyResult<Vec<PyEngineEvent>> {
        let input = StepInput {
            entry,
            exit,
            atr,
            size_mult,
            stop_price_override: stop_price,
            target_price_override: target_price,
        };
        Ok(self
            .session_mut()?
            .apply_current(input)
            .into_iter()
            .map(PyEngineEvent::from)
            .collect())
    }

    /// Submit a typed order routed to one instrument's kernel.
    #[pyo3(signature = (
        instrument, side, kind, submitted_idx, submitted_ts, client_id,
        units=None, size_frac=None, limit_price=None, trigger_price=None,
        tif="gtc", expire_ns=None, stop_price=None, target_price=None,
        offset=None, offset_kind="price", limit_offset=0.0,
        post_only=false, reduce_only=false, parent_id=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn submit_order(
        &mut self,
        instrument: usize,
        side: &str,
        kind: &str,
        submitted_idx: usize,
        submitted_ts: i64,
        client_id: &str,
        units: Option<f64>,
        size_frac: Option<f64>,
        limit_price: Option<f64>,
        trigger_price: Option<f64>,
        tif: &str,
        expire_ns: Option<i64>,
        stop_price: Option<f64>,
        target_price: Option<f64>,
        offset: Option<f64>,
        offset_kind: &str,
        limit_offset: f64,
        post_only: bool,
        reduce_only: bool,
        parent_id: Option<u64>,
    ) -> PyResult<u64> {
        submit_order_on(
            self.session_mut()?.kernel_mut(instrument),
            side,
            kind,
            submitted_idx,
            submitted_ts,
            client_id,
            units,
            size_frac,
            limit_price,
            trigger_price,
            tif,
            expire_ns,
            stop_price,
            target_price,
            offset,
            offset_kind,
            limit_offset,
            post_only,
            reduce_only,
            parent_id,
        )
    }

    fn cancel_order(&mut self, instrument: usize, idx: usize, order_id: u64) -> PyResult<bool> {
        Ok(self.session_mut()?.kernel_mut(instrument).cancel_order(idx, order_id))
    }

    fn cancel_all_orders(&mut self, instrument: usize, idx: usize) -> PyResult<Vec<u64>> {
        Ok(self.session_mut()?.kernel_mut(instrument).cancel_all_orders(idx))
    }

    fn link_oco(&mut self, instrument: usize, order_ids: Vec<u64>) -> PyResult<()> {
        self.session_mut()?.kernel_mut(instrument).link_oco(&order_ids);
        Ok(())
    }

    fn request_close(&mut self, instrument: usize, position_id: u64) -> PyResult<()> {
        self.session_mut()?.kernel_mut(instrument).request_close(position_id);
        Ok(())
    }

    /// Open positions of one instrument, in opening order.
    fn positions(&self, instrument: usize) -> PyResult<Vec<PyPositionSnapshot>> {
        Ok(self
            .session_ref()?
            .kernel(instrument)
            .position_snapshots()
            .into_iter()
            .map(super::strategy_bindings::convert_snapshot)
            .collect())
    }

    /// Earliest open position of one instrument, or None.
    fn position(&self, instrument: usize) -> PyResult<Option<PyPositionSnapshot>> {
        Ok(self
            .session_ref()?
            .kernel(instrument)
            .position_snapshot()
            .map(super::strategy_bindings::convert_snapshot))
    }

    /// Portfolio equity: pool plus every instrument's last-known mark.
    fn equity(&self) -> PyResult<f64> {
        Ok(self.session_ref()?.equity())
    }

    /// Uncommitted shared cash.
    fn cash(&self) -> PyResult<f64> {
        Ok(self.session_ref()?.cash())
    }

    /// Force-close all instruments and compute portfolio metrics.
    fn finish(&mut self) -> PyResult<PyPortfolioResult> {
        let session = self
            .session
            .take()
            .ok_or_else(|| PyValueError::new_err("session is already finished"))?;
        let (result, outcomes) = session.finish();
        Ok(PyPortfolioResult {
            result: convert_result(result),
            per_instrument: outcomes
                .into_iter()
                .map(|o| PyInstrumentSummary {
                    symbol: o.symbol,
                    trades: o.trades,
                    pnl: o.pnl,
                    rejected_entries: o.rejected_entries,
                })
                .collect(),
            rejected_entries: 0,
            halted: false,
            halted_at: None,
        })
    }
}
