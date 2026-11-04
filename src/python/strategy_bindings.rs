//! Python bindings for the per-bar strategy session.
//!
//! Exposes [`SingleRunner`] to Python as [`PyKernelSession`], the execution
//! core behind the class-based strategy contract: a Python driver loop feeds
//! bars and per-bar order inputs, and receives engine events to dispatch to
//! strategy hooks. Result accounting is shared with the array-based runners,
//! so both paths produce identical metrics for identical decisions.

use pyo3::prelude::*;

use super::errors::Fail;

use crate::core::types::{BacktestConfig, Direction, InstrumentConfig, StopConfig, TargetConfig};
use crate::execution::orders::{OrderKind, OrderSide, QtySpec, TimeInForce, TrailOffset};
use crate::portfolio::kernel::{EngineEvent, KernelBar, StepInput};
use crate::portfolio::runner::SingleRunner;

use super::bindings::{
    convert_result, convert_trade, PyBacktestConfig, PyBacktestResult, PyInstrumentConfig, PyTrade,
};
use super::instrument_bindings::PyInstrumentSpec;

/// Fold the `units` / `size_frac` kwargs into a quantity spec.
///
/// Shared by the single-symbol and portfolio `modify_order` bindings so the
/// two report the same error for the same mistake.
pub(crate) fn parse_qty_spec(
    units: Option<f64>,
    size_frac: Option<f64>,
) -> PyResult<Option<QtySpec>> {
    match (units, size_frac) {
        (Some(_), Some(_)) => {
            Err(Fail::new("conflicting_order_args", "pass units or size_frac, not both")
                .param("units, size_frac")
                .into())
        }
        (Some(u), None) => Ok(Some(QtySpec::Units(u))),
        (None, Some(f)) => Ok(Some(QtySpec::CapitalFrac(f))),
        (None, None) => Ok(None),
    }
}

/// A `margin_model` on the config funds entries on a margin account. A cash
/// account would ignore it, so the pair is refused rather than run on a rule
/// the caller did not choose.
pub(crate) fn check_margin_model(
    config: &BacktestConfig,
    mode: crate::accounts::AccountMode,
) -> PyResult<()> {
    if config.margin_model.is_some() && matches!(mode, crate::accounts::AccountMode::Cash) {
        return Err(Fail::new(
            "invalid_margin_model",
            "config.margin_model funds entries on a margin account; pass account_type='margin' \
             (and a leverage) or drop the model",
        )
        .param("margin_model")
        .into());
    }
    Ok(())
}

/// Parse the `account_type` / `leverage` kwargs shared by the single-symbol
/// and portfolio session constructors.
pub(crate) fn parse_account_mode(
    account_type: &str,
    leverage: f64,
) -> PyResult<crate::accounts::AccountMode> {
    match account_type {
        "cash" => Ok(crate::accounts::AccountMode::Cash),
        "margin" => {
            if leverage <= 0.0 {
                return Err(Fail::new("invalid_leverage", "leverage must be > 0")
                    .param("leverage")
                    .value(leverage)
                    .into());
            }
            Ok(crate::accounts::AccountMode::Margin { leverage })
        }
        other => Err(Fail::new(
            "invalid_account_type",
            format!("account_type must be 'cash' or 'margin', got {other:?}"),
        )
        .param("account_type")
        .value(other)
        .into()),
    }
}

/// One observable outcome of a session step.
///
/// `kind` is `"entered"`, `"exited"`, `"entry_rejected"`, `"risk_rejected"`,
/// `"risk_halted"`, or one of the order-lifecycle kinds (`"order_accepted"`, `"order_triggered"`,
/// `"order_filled"`, `"order_canceled"`, `"order_expired"`,
/// `"order_rejected"`); the optional fields are populated according to the
/// kind.
#[pyclass(name = "EngineEvent")]
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
    /// Refusal reason, for `entry_rejected`/`order_rejected`/`risk_rejected`
    /// events; the halt reason for `risk_halted`.
    #[pyo3(get)]
    pub reject_reason: Option<String>,
    /// Engine order id, for `order_*` events.
    #[pyo3(get)]
    pub order_id: Option<u64>,
    /// Caller-supplied order identifier, for `order_*` events.
    #[pyo3(get)]
    pub client_order_id: Option<String>,
    /// For `contract_settled`: `"settled"`, `"exercised"`, `"assigned"`,
    /// `"expired"` or `"delivered"`. For `corporate_action`: `"split"`,
    /// `"bonus"` or `"dividend"`. For `lifecycle_refused`: what was refused
    /// (`"settlement"` or `"corporate_action"`).
    #[pyo3(get)]
    pub outcome: Option<String>,
    /// For `contract_settled`: `"cash"` or `"physical"`.
    #[pyo3(get)]
    pub style: Option<String>,
    /// For `contract_settled`: where the settlement price came from,
    /// `"declared"`, `"underlying"` or `"last_price"`.
    #[pyo3(get)]
    pub source: Option<String>,
    /// For `contract_settled`: the underlying units taken in (positive) or
    /// given away (negative); `0.0` when nothing is delivered. For
    /// `corporate_action`: the split or bonus ratio (`0.0` for a dividend).
    #[pyo3(get)]
    pub quantity: Option<f64>,
    /// For `contract_settled`: the price the underlying was delivered at.
    /// For `corporate_action`: the dividend per unit (`0.0` otherwise).
    #[pyo3(get)]
    pub amount: Option<f64>,
    /// For `corporate_action`: the cash the account received (negative when
    /// it paid, as a short pays a dividend).
    #[pyo3(get)]
    pub cash: Option<f64>,
    /// For `order_resized`: the order's units before the cut (`size` holds
    /// the units it now offers).
    #[pyo3(get)]
    pub previous_size: Option<f64>,
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
            outcome: None,
            style: None,
            source: None,
            quantity: None,
            amount: None,
            cash: None,
            previous_size: None,
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
                trade: Some(convert_trade(*trade)),
                ..empty
            },
            EngineEvent::EntryRejected { idx, reason } => Self {
                kind: "entry_rejected".to_string(),
                idx,
                // `as_str()`, not the Debug form. Both this event and
                // `OrderRejected` reach the same Python handler and are
                // counted in one dict keyed by this string, so a Debug
                // `"MaxPositions"` here and an `as_str()` `"max_positions"`
                // there split ONE cause into two buckets that each
                // undercount. `as_str()` is documented as the stable
                // identifier for reporting; this was the one site not using it.
                reject_reason: Some(reason.as_str().to_string()),
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
            EngineEvent::OrderResized { idx, order_id, client_id, from, to } => Self {
                kind: "order_resized".to_string(),
                idx,
                size: Some(to),
                previous_size: Some(from),
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
            EngineEvent::AlgoStarted { idx, algo_id, client_id } => Self {
                kind: "algo_started".to_string(),
                idx,
                order_id: Some(algo_id),
                client_order_id: Some(client_id),
                ..empty
            },
            EngineEvent::AlgoCompleted { idx, algo_id, client_id } => Self {
                kind: "algo_completed".to_string(),
                idx,
                order_id: Some(algo_id),
                client_order_id: Some(client_id),
                ..empty
            },
            EngineEvent::RiskRejected { idx, reason, order_id, client_id } => Self {
                kind: "risk_rejected".to_string(),
                idx,
                reject_reason: Some(reason.to_string()),
                order_id,
                client_order_id: order_id.map(|_| client_id),
                ..empty
            },
            EngineEvent::FillBreach { idx, reason, order_id, client_id } => Self {
                kind: "fill_breach".to_string(),
                idx,
                reject_reason: Some(reason.to_string()),
                order_id: Some(order_id),
                client_order_id: Some(client_id),
                ..empty
            },
            EngineEvent::FillDeferred { idx, order_id, client_id, price, size, reason, .. } => {
                Self {
                    kind: "fill_deferred".to_string(),
                    idx,
                    price: Some(price),
                    size: Some(size),
                    reject_reason: Some(reason),
                    order_id: Some(order_id),
                    client_order_id: Some(client_id),
                    ..empty
                }
            }
            EngineEvent::RiskHalted { idx, reason } => {
                Self { kind: "risk_halted".to_string(), idx, reject_reason: Some(reason), ..empty }
            }
            EngineEvent::Settled {
                idx,
                outcome,
                style,
                source,
                price,
                size,
                direction,
                deliver,
                delivery_price,
            } => Self {
                kind: "contract_settled".to_string(),
                idx,
                price: Some(price),
                size: Some(size),
                direction: Some(direction as i32),
                outcome: Some(outcome.to_string()),
                style: Some(style.as_str().to_string()),
                source: Some(source.as_str().to_string()),
                quantity: Some(deliver),
                amount: Some(delivery_price),
                ..empty
            },
            EngineEvent::CorporateAction { idx, kind, ratio, amount, cash } => Self {
                kind: "corporate_action".to_string(),
                idx,
                outcome: Some(kind.as_str().to_string()),
                quantity: Some(ratio),
                amount: Some(amount),
                cash: Some(cash),
                ..empty
            },
            EngineEvent::LifecycleRefused { idx, what, reason } => Self {
                kind: "lifecycle_refused".to_string(),
                idx,
                outcome: Some(what.to_string()),
                reject_reason: Some(reason),
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
#[pyclass(name = "PositionSnapshot")]
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
/// Refuse a setting of `config` that cannot be carried out on the instrument
/// being registered (a fee schedule with no row for it, a `squareoff_time`
/// with no time zone), as the structured error.
pub(crate) fn check_market(
    config: &BacktestConfig,
    symbol: &str,
    spec: Option<&crate::instruments::InstrumentSpec>,
) -> PyResult<()> {
    config.check_instrument(symbol, spec).map_err(|e| {
        let fail = match e.code {
            "squareoff_needs_timezone" => Fail::new("squareoff_needs_timezone", e.message),
            _ => Fail::new("fee_schedule_gap", e.message),
        };
        fail.param(e.param).value(e.symbol.clone()).symbol(&e.symbol).valid(e.valid).into()
    })
}

/// Drive it by calling [`PyKernelSession::step`] once per bar in ascending
/// order, then [`PyKernelSession::finish`] to obtain the standard backtest
/// result. Scalars cross the boundary per bar, so a Python driver loop pays
/// one FFI call per bar with no array allocation.
#[pyclass(name = "KernelSession")]
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
        (|| -> PyResult<Self> {
            let direction = match direction {
                1 => Direction::Long,
                -1 => Direction::Short,
                other => {
                    return Err(Fail::new(
                        "invalid_direction",
                        format!("direction must be 1 (long) or -1 (short), got {other}"),
                    )
                    .param("direction")
                    .value(other)
                    .into())
                }
            };

            let rust_config: BacktestConfig = config.map(BacktestConfig::from).unwrap_or_default();
            let inst: Option<InstrumentConfig> = instrument_config.map(InstrumentConfig::from);

            let policy = match oms_type {
                "netting" => crate::portfolio::ledger::PositionPolicy::Net,
                "hedging" => crate::portfolio::ledger::PositionPolicy::Independent,
                other => {
                    return Err(Fail::new(
                        "invalid_oms_type",
                        format!("oms_type must be 'netting' or 'hedging', got {other:?}"),
                    )
                    .param("oms_type")
                    .value(other)
                    .into())
                }
            };
            let account = parse_account_mode(account_type, leverage)?;
            check_margin_model(&rust_config, account)?;

            let spec = match instrument {
                Some(spec) => {
                    let spec = crate::instruments::InstrumentSpec::from(spec);
                    if !spec.kind.tradable() {
                        return Err(Fail::new(
                            "untradable_instrument",
                            format!(
                                "instrument {:?} is not tradable (kind={:?})",
                                spec.symbol, spec.kind
                            ),
                        )
                        .param("instrument")
                        .value(spec.symbol.to_string())
                        .into());
                    }
                    Some(spec)
                }
                None => None,
            };
            // The runner books the initial capital: one that cannot be booked is
            // the structured `money_overflow`, not a panic out of the constructor.
            let mut rust_config = rust_config;
            if rust_config.session_minutes.is_none() {
                rust_config.session_minutes =
                    spec.as_ref().and_then(|s| s.session.as_ref()).map(|s| s.trading_minutes());
            }
            check_market(&rust_config, symbol, spec.as_ref())?;
            let runner = super::errors::guard_money("KernelSession", || {
                let mut runner = SingleRunner::from_config(
                    rust_config,
                    symbol.to_string(),
                    direction,
                    inst.as_ref(),
                );
                if let Some(spec) = spec {
                    runner = runner.with_instrument(spec);
                }
                runner.with_position_policy(policy).with_account_mode(account)
            })?;

            Ok(Self { runner: Some(runner) })
        })()
        .map_err(|e| super::errors::with_op(e, "KernelSession"))
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
        let runner = self.runner.as_mut().ok_or_else(|| {
            PyErr::from(Fail::new("session_finished", "session is finished; create a new one"))
        })?;

        let bar = KernelBar { timestamp, open, high, low, close, volume };
        let input = StepInput {
            entry,
            exit,
            atr,
            size_mult,
            stop_price_override: stop_price,
            target_price_override: target_price,
            ..StepInput::default()
        };

        Ok(runner.step(idx, &bar, input).into_iter().map(PyEngineEvent::from).collect())
    }

    /// Meet what rests for bar `idx` before the strategy decides on it: the stops, targets and
    /// resting orders of earlier bars trade the bar's range. Call it ahead of `on_bar` and send the
    /// events to the hooks; `step` then carries on with the bar's decisions.
    #[allow(clippy::too_many_arguments)]
    fn rest(
        &mut self,
        idx: usize,
        timestamp: i64,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
    ) -> PyResult<Vec<PyEngineEvent>> {
        let runner = self.runner.as_mut().ok_or_else(|| {
            PyErr::from(Fail::new("session_finished", "session is finished; create a new one"))
        })?;
        let bar = KernelBar { timestamp, open, high, low, close, volume };
        Ok(runner.rest(idx, &bar).into_iter().map(PyEngineEvent::from).collect())
    }

    /// Submit an order.
    ///
    /// `side` is `"buy"`/`"sell"`; `kind` is `"market"`, `"limit"`,
    /// `"stop_market"`, or `"stop_limit"`; `tif` is `"auto"` (DAY on an instrument with a session, else GTC), `"gtc"`, `"day"`,
    /// `"gtd"` (requires `expire_ns`), `"ioc"`, or `"fok"`. Exactly one of
    /// `units`/`size_frac` sizes an opening order; omit both to close the
    /// full position with a closing-side order. Returns the engine order id.
    #[pyo3(signature = (
        side, kind, submitted_idx, submitted_ts, client_id,
        units=None, size_frac=None, limit_price=None, trigger_price=None,
        tif="auto", expire_ns=None, stop_price=None, target_price=None,
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

    /// Set the underlying price used to settle options at expiry.
    ///
    /// An option's own bars carry the option's price, so intrinsic value
    /// needs the underlying from somewhere else. Without it, contracts
    /// settle at their own close.
    #[pyo3(signature = (price=None))]
    fn set_underlying_price(&mut self, price: Option<f64>) -> PyResult<()> {
        self.runner_mut()?.kernel_mut().set_underlying_price(price);
        Ok(())
    }

    /// Register a TWAP schedule; returns its id.
    #[pyo3(signature = (units, side, slices, interval_ns, submitted_idx, submitted_ts, client_id, reduce_only=false))]
    #[allow(clippy::too_many_arguments)]
    fn submit_twap(
        &mut self,
        units: f64,
        side: &str,
        slices: u32,
        interval_ns: i64,
        submitted_idx: usize,
        submitted_ts: i64,
        client_id: &str,
        reduce_only: bool,
    ) -> PyResult<u64> {
        submit_twap_on(
            self.runner_mut()?.kernel_mut(),
            units,
            side,
            slices,
            interval_ns,
            submitted_idx,
            submitted_ts,
            client_id,
            reduce_only,
        )
    }

    /// Stop a schedule and cancel the slices it has working.
    fn cancel_twap(&mut self, algo_id: u64, idx: usize) -> PyResult<bool> {
        Ok(self.runner_mut()?.kernel_mut().cancel_algo(algo_id, idx))
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
        let qty = parse_qty_spec(units, size_frac)?;
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
                    return Err(Fail::new(
                        "invalid_order_reference",
                        format!("unknown position id {id}"),
                    )
                    .param("position_id")
                    .value(id as i64)
                    .into());
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
                    return Err(Fail::new(
                        "invalid_order_reference",
                        format!("unknown position id {id}"),
                    )
                    .param("position_id")
                    .value(id as i64)
                    .into());
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
        Ok(self.runner_ref()?.free_capital())
    }

    /// Halt the session: new entries are refused as `killed`; working orders
    /// are cancelled and, with `flatten=True`, the position is closed at the
    /// last bar's close when the next bar is stepped.
    #[pyo3(signature = (reason="manual", flatten=false))]
    fn halt(&mut self, reason: &str, flatten: bool) -> PyResult<()> {
        self.runner
            .as_mut()
            .ok_or_else(|| PyErr::from(Fail::new("run_finished", "session is already finished")))?
            .halt(reason, flatten);
        Ok(())
    }

    /// Declare the corporate actions of the run and the adjustment policy its
    /// bars are on (`"raw"` or `"adjusted"`). Call before the first step.
    /// Every action must name this session's symbol.
    #[pyo3(signature = (actions, policy))]
    fn set_corporate_actions(
        &mut self,
        actions: Vec<super::lifecycle_bindings::PyCorporateAction>,
        policy: &str,
    ) -> PyResult<()> {
        let (actions, policy) =
            super::lifecycle_bindings::extract_actions(Some(actions), Some(policy))?;
        let runner = self.runner_mut()?;
        let symbol = runner.kernel().symbol().to_string();
        for a in &actions {
            if a.instrument != symbol {
                return Err(Fail::new(
                    "unknown_symbol",
                    format!(
                        "corporate action on {:?}: this session trades {symbol:?}",
                        a.instrument
                    ),
                )
                .param("corporate_actions")
                .value(a.instrument.as_str())
                .into());
            }
        }
        runner
            .set_corporate_actions(&actions, policy)
            .map_err(super::lifecycle_bindings::action_err)
    }

    /// Dividend cash credited so far (a short's debit is negative).
    #[getter]
    fn dividends_received(&self) -> PyResult<f64> {
        Ok(self.runner_ref()?.dividends_received())
    }

    /// Whether the drawdown limit or a halt has latched.
    fn is_halted(&self) -> PyResult<bool> {
        Ok(self.runner_ref()?.is_halted())
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
            .ok_or_else(|| Fail::new("session_finished", "session is already finished"))?;
        Ok(convert_result(runner.finish()))
    }
}

pub(crate) fn convert_snapshot(
    p: crate::portfolio::kernel::PositionSnapshot,
) -> PyPositionSnapshot {
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
    pub(crate) fn runner_ref(&self) -> PyResult<&SingleRunner> {
        self.runner.as_ref().ok_or_else(|| {
            PyErr::from(Fail::new("session_finished", "session is finished; create a new one"))
        })
    }

    pub(crate) fn runner_mut(&mut self) -> PyResult<&mut SingleRunner> {
        self.runner.as_mut().ok_or_else(|| {
            PyErr::from(Fail::new("session_finished", "session is finished; create a new one"))
        })
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
    atr_period(&rust_config, inst.as_ref())
}

/// The ATR period behind a run's stop/target levels; see [`resolve_atr_period`].
pub(crate) fn atr_period(
    config: &BacktestConfig,
    instrument_config: Option<&InstrumentConfig>,
) -> Option<usize> {
    let effective_stop =
        instrument_config.and_then(|ic| ic.stop.as_ref()).copied().unwrap_or(config.stop);
    let effective_target =
        instrument_config.and_then(|ic| ic.target.as_ref()).copied().unwrap_or(config.target);

    match (effective_stop, effective_target) {
        (StopConfig::Atr { period, .. }, _) => Some(period),
        (_, TargetConfig::Atr { period, .. }) => Some(period),
        _ => None,
    }
}

/// Register a TWAP schedule on a kernel — shared by the single-instrument
/// session, the portfolio session and the runtime's strategy view.
#[allow(clippy::too_many_arguments)]
pub(crate) fn submit_twap_on(
    kernel: &mut crate::portfolio::kernel::EngineKernel,
    units: f64,
    side: &str,
    slices: u32,
    interval_ns: i64,
    submitted_idx: usize,
    submitted_ts: i64,
    client_id: &str,
    reduce_only: bool,
) -> PyResult<u64> {
    let side = match side {
        "buy" => OrderSide::Buy,
        "sell" => OrderSide::Sell,
        other => {
            return Err(Fail::new(
                "invalid_order_side",
                format!("side must be 'buy' or 'sell', got {other:?}"),
            )
            .param("side")
            .value(other)
            .into())
        }
    };
    kernel
        .submit_algo(
            side,
            QtySpec::Units(units),
            OrderKind::Market,
            TimeInForce::Gtc,
            client_id.to_string(),
            crate::execution::algos::ExecAlgorithm::Twap { slices, interval_ns },
            reduce_only,
            submitted_ts,
            submitted_idx,
        )
        .map_err(|e| {
            PyErr::from(Fail::new("invalid_order_value", e.to_string()).param("algo").value("twap"))
        })
}

/// The `tif` of an order that names none: the instrument decides (DAY on one
/// with an exchange session, GTC otherwise).
pub(crate) const TIF_AUTO: &str = "auto";

/// The typed form of Python order arguments.
pub(crate) struct ParsedOrder {
    pub side: OrderSide,
    pub qty: QtySpec,
    pub kind: OrderKind,
    pub tif: TimeInForce,
}

/// Refuse an order whose quantity is not a whole number of lots (or size
/// steps) or whose prices are not a whole number of ticks. One check for every
/// mode: the simulation used to floor and snap while a venue was sent the raw
/// terms. The error names the nearest valid values in `valid`.
pub(crate) fn check_grid(
    grid: &crate::execution::orders::OrderGrid,
    units: Option<f64>,
    prices: &[(&'static str, Option<f64>)],
) -> PyResult<()> {
    grid.check(units, prices).map_err(|off| {
        let what = if off.off_lot {
            "a whole number of lots (or size steps)"
        } else {
            "a whole number of ticks"
        };
        let message = format!(
            "{} {} is not {what}: the step is {}; use one of the valid values, or round it \
             yourself before submitting",
            off.parameter, off.value, off.step
        );
        let fail = if off.off_lot {
            Fail::new("off_lot_quantity", message)
        } else {
            Fail::new("off_tick_price", message)
        };
        fail.param(off.parameter).value(off.value).valid(off.valid).into()
    })
}

/// An order price or level must be a finite number above zero: a NaN compares false against every
/// bound, and a level of zero or less would exit the position at once.
fn finite_positive(name: &'static str, value: f64) -> PyResult<()> {
    if value.is_finite() && value > 0.0 {
        return Ok(());
    }
    Err(Fail::new("invalid_order_value", format!("{name} must be a finite number > 0"))
        .param(name)
        .value(value)
        .into())
}

/// Parse Python order arguments. `price_increment` is the instrument's tick
/// size (`0.0` without one), which a trailing offset in ticks needs.
#[allow(clippy::too_many_arguments)]
pub(crate) fn parse_order(
    price_increment: f64,
    side: &str,
    kind: &str,
    units: Option<f64>,
    size_frac: Option<f64>,
    limit_price: Option<f64>,
    trigger_price: Option<f64>,
    tif: &str,
    expire_ns: Option<i64>,
    offset: Option<f64>,
    offset_kind: &str,
    limit_offset: f64,
    post_only: bool,
) -> PyResult<ParsedOrder> {
    let side = match side {
        "buy" => OrderSide::Buy,
        "sell" => OrderSide::Sell,
        other => {
            return Err(Fail::new("invalid_order_side", format!("unknown side {other:?}"))
                .param("side")
                .value(other)
                .into())
        }
    };
    let qty = match (units, size_frac) {
        (Some(_), Some(_)) => {
            return Err(Fail::new("conflicting_order_args", "pass units or size_frac, not both")
                .param("units, size_frac")
                .into())
        }
        (Some(u), None) if !(u.is_finite() && u > 0.0) => {
            return Err(Fail::new("invalid_order_value", "units must be a finite number > 0")
                .param("units")
                .value(u)
                .into())
        }
        (Some(u), None) => QtySpec::Units(u),
        (None, Some(f)) if !(f > 0.0 && f <= 1.0) => {
            return Err(Fail::new("invalid_order_value", "size_frac must be in (0, 1]")
                .param("size_frac")
                .value(f)
                .valid(["size_frac in (0, 1]"])
                .into())
        }
        (None, Some(f)) => QtySpec::CapitalFrac(f),
        (None, None) => QtySpec::FullPosition,
    };
    let need_limit = || {
        let price = limit_price.ok_or_else(|| {
            PyErr::from(
                Fail::new("missing_order_field", format!("{kind} needs limit_price"))
                    .param("limit_price"),
            )
        })?;
        finite_positive("limit_price", price)?;
        Ok::<f64, PyErr>(price)
    };
    let need_trigger = || {
        let price = trigger_price.ok_or_else(|| {
            PyErr::from(
                Fail::new("missing_order_field", format!("{kind} needs trigger_price"))
                    .param("trigger_price"),
            )
        })?;
        finite_positive("trigger_price", price)?;
        Ok::<f64, PyErr>(price)
    };
    let trail_offset = || -> PyResult<TrailOffset> {
        let raw = offset.ok_or_else(|| {
            Fail::new("missing_order_field", format!("{kind} needs offset")).param("offset")
        })?;
        if !(raw.is_finite() && raw > 0.0) {
            return Err(Fail::new("invalid_order_value", "offset must be a finite number > 0")
                .param("offset")
                .value(raw)
                .into());
        }
        match offset_kind {
            "price" => Ok(TrailOffset::Price(raw)),
            "bps" => Ok(TrailOffset::Bps(raw)),
            "ticks" => {
                let tick = price_increment;
                if tick <= 0.0 {
                    return Err(Fail::new(
                        "incompatible_order_options",
                        "offset_kind='ticks' needs an instrument with price_increment",
                    )
                    .param("offset_kind")
                    .value("ticks")
                    .into());
                }
                Ok(TrailOffset::Price(raw * tick))
            }
            other => Err(Fail::new(
                "invalid_offset_kind",
                format!("offset_kind must be 'price', 'bps', or 'ticks', got {other:?}"),
            )
            .param("offset_kind")
            .value(other)
            .into()),
        }
    };

    let kind_parsed = match kind {
        "market" => OrderKind::Market,
        "limit" => OrderKind::Limit { price: need_limit()? },
        "stop_market" => OrderKind::StopMarket { trigger: need_trigger()? },
        "stop_limit" => OrderKind::StopLimit { trigger: need_trigger()?, price: need_limit()? },
        "market_if_touched" => OrderKind::MarketIfTouched { trigger: need_trigger()? },
        "limit_if_touched" => {
            OrderKind::LimitIfTouched { trigger: need_trigger()?, price: need_limit()? }
        }
        "market_to_limit" => OrderKind::MarketToLimit,
        "trailing_stop_market" => OrderKind::TrailingStopMarket { offset: trail_offset()? },
        "trailing_stop_limit" => {
            if !(limit_offset.is_finite() && limit_offset >= 0.0) {
                return Err(Fail::new(
                    "invalid_order_value",
                    "limit_offset must be a finite number >= 0",
                )
                .param("limit_offset")
                .value(limit_offset)
                .into());
            }
            OrderKind::TrailingStopLimit { offset: trail_offset()?, limit_offset }
        }
        other => {
            return Err(Fail::new("invalid_order_kind", format!("unknown order kind {other:?}"))
                .param("kind")
                .value(other)
                .into())
        }
    };
    let tif = match tif {
        "gtc" => TimeInForce::Gtc,
        "day" => TimeInForce::Day,
        "gtd" => TimeInForce::Gtd {
            expire_ns: expire_ns.ok_or_else(|| {
                Fail::new("missing_order_field", "gtd orders need expire_ns").param("expire_ns")
            })?,
        },
        "ioc" => TimeInForce::Ioc,
        "fok" => TimeInForce::Fok,
        "at_open" => TimeInForce::AtOpen,
        "at_close" => TimeInForce::AtClose,
        other => {
            return Err(Fail::new("invalid_time_in_force", format!("unknown tif {other:?}"))
                .param("tif")
                .value(other)
                .into())
        }
    };
    if matches!(tif, TimeInForce::AtOpen | TimeInForce::AtClose)
        && !matches!(kind_parsed, OrderKind::Market)
    {
        return Err(Fail::new(
            "incompatible_order_options",
            "at_open/at_close apply to market orders",
        )
        .param("tif")
        .into());
    }
    if post_only && !matches!(kind_parsed, OrderKind::Limit { .. }) {
        return Err(Fail::new("incompatible_order_options", "post_only applies to limit orders")
            .param("post_only")
            .into());
    }

    Ok(ParsedOrder { side, qty, kind: kind_parsed, tif })
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
    submit_order_checked(
        kernel,
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
        None,
    )
}

/// [`submit_order_on`], or, with a `refusal`, the same order recorded as
/// rejected with that code and never working (see
/// [`EngineKernel::submit_refused_order`]).
#[allow(clippy::too_many_arguments)]
pub(crate) fn submit_order_checked(
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
    refusal: Option<&'static str>,
) -> PyResult<u64> {
    for (name, level) in [("stop_price", stop_price), ("target_price", target_price)] {
        if let Some(level) = level {
            finite_positive(name, level).map_err(|e| super::errors::with_op(e, "submit_order"))?;
        }
    }
    check_grid(
        &kernel.order_grid(),
        units,
        &[
            ("limit_price", limit_price),
            ("trigger_price", trigger_price),
            ("stop_price", stop_price),
            ("target_price", target_price),
        ],
    )?;
    // An order that names no time in force takes its instrument's default.
    let tif = if tif == TIF_AUTO { kernel.default_tif() } else { tif };
    kernel.check_validity(tif, expire_ns).map_err(|why| {
        PyErr::from(
            Fail::new("order_validity_unsupported", why)
                .param("tif")
                .value(tif)
                .valid(["day", "ioc", "fok"]),
        )
    })?;
    let ParsedOrder { side, qty, kind: kind_parsed, tif } = parse_order(
        kernel.price_increment(),
        side,
        kind,
        units,
        size_frac,
        limit_price,
        trigger_price,
        tif,
        expire_ns,
        offset,
        offset_kind,
        limit_offset,
        post_only,
    )
    .map_err(|e| super::errors::with_op(e, "submit_order"))?;
    // A protective child (a bracket's stop or target) reduces the position
    // its parent opened. Under hedging every order opens its own position and
    // a reduce-only order is always refused at its fill, so the bracket would
    // leave its position open without a word. Refuse it here instead.
    if reduce_only && parent_id.is_some() && kernel.is_hedging() {
        return Err(Fail::new(
            "bracket_under_hedging",
            "a reduce-only child order (a bracket's stop or target) cannot close a position \
             under hedging, where every order opens its own position; attach \
             stop_price/target_price to the entry, or use oms_type='netting'",
        )
        .op("submit_order")
        .param("client_id")
        .value(client_id)
        .into());
    }
    if let Some(reason) = refusal {
        return Ok(kernel.submit_refused_order(
            side,
            qty,
            kind_parsed,
            tif,
            submitted_idx,
            submitted_ts,
            client_id.to_string(),
            reduce_only,
            reason,
        ));
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
