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
use crate::execution::orders::{OrderKind, OrderSide, QtySpec, TimeInForce, TrailOffset};
use crate::portfolio::kernel::{EngineEvent, KernelBar, StepInput};
use crate::portfolio::runner::SingleRunner;

use super::bindings::{
    convert_result, convert_trade, PyBacktestConfig, PyBacktestResult, PyInstrumentConfig, PyTrade,
};
use super::instrument_bindings::PyInstrumentSpec;

/// One observable outcome of a session step.
///
/// `kind` is `"entered"`, `"exited"`, `"entry_rejected"`, or one of the
/// order-lifecycle kinds (`"order_accepted"`, `"order_triggered"`,
/// `"order_filled"`, `"order_canceled"`, `"order_expired"`,
/// `"order_rejected"`); the optional fields are populated according to the
/// kind.
#[pyclass]
#[derive(Debug, Clone)]
pub struct PyEngineEvent {
    #[pyo3(get)]
    pub kind: String,
    #[pyo3(get)]
    pub idx: usize,
    /// Fill price, for `entered`/`order_filled` events.
    #[pyo3(get)]
    pub price: Option<f64>,
    /// Position size, for `entered`/`order_filled` events.
    #[pyo3(get)]
    pub size: Option<f64>,
    /// Trade direction (1 long, -1 short), for `entered` events.
    #[pyo3(get)]
    pub direction: Option<i32>,
    /// Completed trade, for `exited` events.
    #[pyo3(get)]
    pub trade: Option<PyTrade>,
    /// Refusal reason, for `entry_rejected`/`order_rejected` events.
    #[pyo3(get)]
    pub reject_reason: Option<String>,
    /// Engine order id, for `order_*` events.
    #[pyo3(get)]
    pub order_id: Option<u64>,
    /// Caller-supplied order identifier, for `order_*` events.
    #[pyo3(get)]
    pub client_order_id: Option<String>,
}

#[pymethods]
impl PyEngineEvent {
    fn __repr__(&self) -> String {
        format!("EngineEvent(kind={}, idx={})", self.kind, self.idx)
    }
}

impl From<EngineEvent> for PyEngineEvent {
    fn from(event: EngineEvent) -> Self {
        let empty = Self {
            kind: String::new(),
            idx: 0,
            price: None,
            size: None,
            direction: None,
            trade: None,
            reject_reason: None,
            order_id: None,
            client_order_id: None,
        };
        match event {
            EngineEvent::Entered { idx, price, size, direction } => Self {
                kind: "entered".to_string(),
                idx,
                price: Some(price),
                size: Some(size),
                direction: Some(direction as i32),
                ..empty
            },
            EngineEvent::Exited { idx, trade } => Self {
                kind: "exited".to_string(),
                idx,
                price: Some(trade.exit_price),
                size: Some(trade.size),
                direction: Some(trade.direction as i32),
                trade: Some(convert_trade(trade)),
                ..empty
            },
            EngineEvent::EntryRejected { idx, reason } => Self {
                kind: "entry_rejected".to_string(),
                idx,
                reject_reason: Some(format!("{reason:?}")),
                ..empty
            },
            EngineEvent::OrderAccepted { idx, order_id, client_id } => Self {
                kind: "order_accepted".to_string(),
                idx,
                order_id: Some(order_id),
                client_order_id: Some(client_id),
                ..empty
            },
            EngineEvent::OrderTriggered { idx, order_id, client_id } => Self {
                kind: "order_triggered".to_string(),
                idx,
                order_id: Some(order_id),
                client_order_id: Some(client_id),
                ..empty
            },
            EngineEvent::OrderFilled { idx, order_id, client_id, price, size } => Self {
                kind: "order_filled".to_string(),
                idx,
                price: Some(price),
                size: Some(size),
                order_id: Some(order_id),
                client_order_id: Some(client_id),
                ..empty
            },
            EngineEvent::OrderCanceled { idx, order_id, client_id } => Self {
                kind: "order_canceled".to_string(),
                idx,
                order_id: Some(order_id),
                client_order_id: Some(client_id),
                ..empty
            },
            EngineEvent::OrderExpired { idx, order_id, client_id } => Self {
                kind: "order_expired".to_string(),
                idx,
                order_id: Some(order_id),
                client_order_id: Some(client_id),
                ..empty
            },
            EngineEvent::OrderRejected { idx, order_id, client_id, reason } => Self {
                kind: "order_rejected".to_string(),
                idx,
                reject_reason: Some(reason.to_string()),
                order_id: Some(order_id),
                client_order_id: Some(client_id),
                ..empty
            },
            EngineEvent::MarginCall { idx, equity, required } => Self {
                kind: "margin_call".to_string(),
                idx,
                // Reuse the price/size slots: current equity and the
                // maintenance requirement it fell below.
                price: Some(equity),
                size: Some(required),
                ..empty
            },
        }
    }
}

/// Read-only view of the session's open position.
#[pyclass]
#[derive(Debug, Clone)]
pub struct PyPositionSnapshot {
    /// Ledger position id, unique within a session.
    #[pyo3(get)]
    pub position_id: u64,
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
    #[pyo3(signature = (symbol="ASSET", direction=1, config=None, instrument_config=None, instrument=None,
                        oms_type="netting", account_type="cash", leverage=1.0))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        symbol: &str,
        direction: i32,
        config: Option<&PyBacktestConfig>,
        instrument_config: Option<&PyInstrumentConfig>,
        instrument: Option<&PyInstrumentSpec>,
        oms_type: &str,
        account_type: &str,
        leverage: f64,
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

        let policy = match oms_type {
            "netting" => crate::portfolio::ledger::PositionPolicy::Net,
            "hedging" => crate::portfolio::ledger::PositionPolicy::Independent,
            other => {
                return Err(PyValueError::new_err(format!(
                    "oms_type must be 'netting' or 'hedging', got {other:?}"
                )))
            }
        };
        let account = match account_type {
            "cash" => crate::accounts::AccountMode::Cash,
            "margin" => {
                if leverage <= 0.0 {
                    return Err(PyValueError::new_err("leverage must be > 0"));
                }
                crate::accounts::AccountMode::Margin { leverage }
            }
            other => {
                return Err(PyValueError::new_err(format!(
                    "account_type must be 'cash' or 'margin', got {other:?}"
                )))
            }
        };

        let mut runner =
            SingleRunner::from_config(rust_config, symbol.to_string(), direction, inst.as_ref());
        if let Some(spec) = instrument {
            let spec = crate::instruments::InstrumentSpec::from(spec);
            if !spec.kind.tradable() {
                return Err(PyValueError::new_err(format!(
                    "instrument {:?} is not tradable (kind={:?})",
                    spec.symbol, spec.kind
                )));
            }
            runner = runner.with_instrument(spec);
        }
        runner = runner.with_position_policy(policy).with_account_mode(account);

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

    /// Submit an order.
    ///
    /// `side` is `"buy"`/`"sell"`; `kind` is `"market"`, `"limit"`,
    /// `"stop_market"`, or `"stop_limit"`; `tif` is `"gtc"`, `"day"`,
    /// `"gtd"` (requires `expire_ns`), `"ioc"`, or `"fok"`. Exactly one of
    /// `units`/`size_frac` sizes an opening order; omit both to close the
    /// full position with a closing-side order. Returns the engine order id.
    #[pyo3(signature = (
        side, kind, submitted_idx, submitted_ts, client_id,
        units=None, size_frac=None, limit_price=None, trigger_price=None,
        tif="gtc", expire_ns=None, stop_price=None, target_price=None,
        offset=None, offset_kind="price", limit_offset=0.0,
        post_only=false, reduce_only=false, parent_id=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn submit_order(
        &mut self,
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
            self.runner_mut()?.kernel_mut(),
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

    /// Put working orders in one one-cancels-other group.
    fn link_oco(&mut self, order_ids: Vec<u64>) -> PyResult<()> {
        self.runner_mut()?.kernel_mut().link_oco(&order_ids);
        Ok(())
    }

    /// Cancel a working order; `false` for unknown/finished ids.
    fn cancel_order(&mut self, idx: usize, order_id: u64) -> PyResult<bool> {
        Ok(self.runner_mut()?.kernel_mut().cancel_order(idx, order_id))
    }

    /// Cancel every working order, returning canceled ids.
    fn cancel_all_orders(&mut self, idx: usize) -> PyResult<Vec<u64>> {
        Ok(self.runner_mut()?.kernel_mut().cancel_all_orders(idx))
    }

    /// Replace a working order's prices and/or quantity.
    #[pyo3(signature = (order_id, units=None, size_frac=None, limit_price=None, trigger_price=None))]
    fn modify_order(
        &mut self,
        order_id: u64,
        units: Option<f64>,
        size_frac: Option<f64>,
        limit_price: Option<f64>,
        trigger_price: Option<f64>,
    ) -> PyResult<bool> {
        let qty = match (units, size_frac) {
            (Some(_), Some(_)) => {
                return Err(PyValueError::new_err("pass units or size_frac, not both"))
            }
            (Some(u), None) => Some(QtySpec::Units(u)),
            (None, Some(f)) => Some(QtySpec::CapitalFrac(f)),
            (None, None) => None,
        };
        Ok(self.runner_mut()?.kernel_mut().modify_order(order_id, qty, limit_price, trigger_price))
    }

    /// Ids of all non-terminal orders, in submission order.
    fn open_order_ids(&self) -> PyResult<Vec<u64>> {
        Ok(self.runner_ref()?.kernel().open_orders().iter().map(|o| o.id).collect())
    }

    /// Overwrite a position's stop price; no-op when flat.
    ///
    /// Without `position_id` targets the earliest open position.
    #[pyo3(signature = (price, position_id=None))]
    fn set_stop_price(&mut self, price: Option<f64>, position_id: Option<u64>) -> PyResult<()> {
        let kernel = self.runner_mut()?.kernel_mut();
        match position_id {
            Some(id) => {
                if !kernel.set_stop_price_for(id, price) {
                    return Err(PyValueError::new_err(format!("unknown position id {id}")));
                }
            }
            None => kernel.set_stop_price(price),
        }
        Ok(())
    }

    /// Overwrite a position's target price; no-op when flat.
    #[pyo3(signature = (price, position_id=None))]
    fn set_target_price(&mut self, price: Option<f64>, position_id: Option<u64>) -> PyResult<()> {
        let kernel = self.runner_mut()?.kernel_mut();
        match position_id {
            Some(id) => {
                if !kernel.set_target_price_for(id, price) {
                    return Err(PyValueError::new_err(format!("unknown position id {id}")));
                }
            }
            None => kernel.set_target_price(price),
        }
        Ok(())
    }

    /// Request a close of a specific position on the next step.
    fn request_close(&mut self, position_id: u64) -> PyResult<()> {
        self.runner_mut()?.kernel_mut().request_close(position_id);
        Ok(())
    }

    /// Read-only views of every open position, in opening order.
    fn positions(&self) -> PyResult<Vec<PyPositionSnapshot>> {
        Ok(self
            .runner_ref()?
            .kernel()
            .position_snapshots()
            .into_iter()
            .map(convert_snapshot)
            .collect())
    }

    /// Cash not locked as margin (margin mode); all cash otherwise.
    fn free_capital(&self) -> PyResult<f64> {
        Ok(self.runner_ref()?.kernel().free_capital())
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

    /// Read-only view of the earliest open position, or `None` when flat.
    fn position(&self) -> PyResult<Option<PyPositionSnapshot>> {
        Ok(self.runner_ref()?.kernel().position_snapshot().map(convert_snapshot))
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

pub(crate) fn convert_snapshot(p: crate::portfolio::kernel::PositionSnapshot) -> PyPositionSnapshot {
    PyPositionSnapshot {
        position_id: p.position_id,
        entry_idx: p.entry_idx,
        entry_price: p.entry_price,
        size: p.size,
        direction: p.direction as i32,
        stop_price: p.stop_price,
        target_price: p.target_price,
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


/// Parse Python order arguments and submit onto a kernel — shared by the
/// single-instrument session and the portfolio session.
#[allow(clippy::too_many_arguments)]
pub(crate) fn submit_order_on(
    kernel: &mut crate::portfolio::kernel::EngineKernel,
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
        let side = match side {
            "buy" => OrderSide::Buy,
            "sell" => OrderSide::Sell,
            other => return Err(PyValueError::new_err(format!("unknown side {other:?}"))),
        };
        let qty = match (units, size_frac) {
            (Some(_), Some(_)) => {
                return Err(PyValueError::new_err("pass units or size_frac, not both"))
            }
            (Some(u), None) if u <= 0.0 => {
                return Err(PyValueError::new_err("units must be > 0"))
            }
            (Some(u), None) => QtySpec::Units(u),
            (None, Some(f)) if f <= 0.0 || f > 1.0 => {
                return Err(PyValueError::new_err("size_frac must be in (0, 1]"))
            }
            (None, Some(f)) => QtySpec::CapitalFrac(f),
            (None, None) => QtySpec::FullPosition,
        };
        let need_limit = || {
            limit_price.ok_or_else(|| PyValueError::new_err(format!("{kind} needs limit_price")))
        };
        let need_trigger = || {
            trigger_price
                .ok_or_else(|| PyValueError::new_err(format!("{kind} needs trigger_price")))
        };
        let trail_offset = |kernel: &crate::portfolio::kernel::EngineKernel| -> PyResult<TrailOffset> {
            let raw = offset
                .ok_or_else(|| PyValueError::new_err(format!("{kind} needs offset")))?;
            if raw <= 0.0 {
                return Err(PyValueError::new_err("offset must be > 0"));
            }
            match offset_kind {
                "price" => Ok(TrailOffset::Price(raw)),
                "bps" => Ok(TrailOffset::Bps(raw)),
                "ticks" => {
                    let tick = kernel.price_increment();
                    if tick <= 0.0 {
                        return Err(PyValueError::new_err(
                            "offset_kind='ticks' needs an instrument with price_increment",
                        ));
                    }
                    Ok(TrailOffset::Price(raw * tick))
                }
                other => Err(PyValueError::new_err(format!(
                    "offset_kind must be 'price', 'bps', or 'ticks', got {other:?}"
                ))),
            }
        };

        let kernel_ref = &*kernel;
        let kind_parsed = match kind {
            "market" => OrderKind::Market,
            "limit" => OrderKind::Limit { price: need_limit()? },
            "stop_market" => OrderKind::StopMarket { trigger: need_trigger()? },
            "stop_limit" => {
                OrderKind::StopLimit { trigger: need_trigger()?, price: need_limit()? }
            }
            "market_if_touched" => OrderKind::MarketIfTouched { trigger: need_trigger()? },
            "limit_if_touched" => {
                OrderKind::LimitIfTouched { trigger: need_trigger()?, price: need_limit()? }
            }
            "market_to_limit" => OrderKind::MarketToLimit,
            "trailing_stop_market" => {
                OrderKind::TrailingStopMarket { offset: trail_offset(kernel_ref)? }
            }
            "trailing_stop_limit" => OrderKind::TrailingStopLimit {
                offset: trail_offset(kernel_ref)?,
                limit_offset,
            },
            other => return Err(PyValueError::new_err(format!("unknown order kind {other:?}"))),
        };
        let tif = match tif {
            "gtc" => TimeInForce::Gtc,
            "day" => TimeInForce::Day,
            "gtd" => TimeInForce::Gtd {
                expire_ns: expire_ns
                    .ok_or_else(|| PyValueError::new_err("gtd orders need expire_ns"))?,
            },
            "ioc" => TimeInForce::Ioc,
            "fok" => TimeInForce::Fok,
            "at_open" => TimeInForce::AtOpen,
            "at_close" => TimeInForce::AtClose,
            other => return Err(PyValueError::new_err(format!("unknown tif {other:?}"))),
        };
        if matches!(tif, TimeInForce::AtOpen | TimeInForce::AtClose)
            && !matches!(kind_parsed, OrderKind::Market)
        {
            return Err(PyValueError::new_err("at_open/at_close apply to market orders"));
        }
        if post_only && !matches!(kind_parsed, OrderKind::Limit { .. }) {
            return Err(PyValueError::new_err("post_only applies to limit orders"));
        }

        Ok(kernel.submit_order_full(
            side,
            qty,
            kind_parsed,
            tif,
            submitted_idx,
            submitted_ts,
            client_id.to_string(),
            stop_price,
            target_price,
            post_only,
            reduce_only,
            parent_id,
        ))
}