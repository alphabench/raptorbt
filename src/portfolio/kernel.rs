//! Steppable simulation kernel.
//!
//! Holds the per-bar simulation state that [`PortfolioEngine`] previously kept
//! as loop locals. Batch backtests drive this by looping [`EngineKernel::step`]
//! over historical bars; a future live engine drives the same code with bars
//! arriving in real time, which is the point of the extraction — one set of
//! execution semantics rather than a separate live reimplementation.
//!
//! [`PortfolioEngine`]: crate::portfolio::engine::PortfolioEngine

use crate::core::types::{
    BacktestConfig, Direction, ExitReason, InstrumentConfig, OhlcvBar, Price, StopConfig,
    TargetConfig, Trade,
};
use crate::execution::orders::{
    MatchOutcome, Order, OrderEngine, OrderKind, OrderSide, OrderStatus, QtySpec, TimeInForce,
};
use crate::accounts::{AccountMode, MarginBook};
use crate::execution::fill::FillRng;
use crate::execution::{FeeModel, FillModel, FillPrice, SlippageModel};
use crate::instruments::InstrumentSpec;
use crate::portfolio::ledger::{PositionLedger, PositionPolicy};
use crate::portfolio::position::ExitDetails;
use crate::portfolio::risk::{RejectReason, RiskGate};

/// A single bar handed to the kernel.
///
/// Deliberately owns its values rather than borrowing an `OhlcvData` index:
/// a live feed produces one bar at a time with no backing array.
#[derive(Debug, Clone, Copy)]
pub struct KernelBar {
    pub timestamp: i64,
    pub open: Price,
    pub high: Price,
    pub low: Price,
    pub close: Price,
    pub volume: f64,
}

impl KernelBar {
    /// Borrow as an [`OhlcvBar`] for the execution models.
    fn to_ohlcv_bar(self) -> OhlcvBar {
        OhlcvBar {
            timestamp: self.timestamp,
            open: self.open,
            high: self.high,
            low: self.low,
            close: self.close,
            volume: self.volume,
        }
    }
}

/// Observable outcomes of a single [`EngineKernel::step`] call.
///
/// Batch callers can ignore these and read the accumulated trades; live callers
/// need them to drive order placement and alerting.
#[derive(Debug, Clone)]
pub enum EngineEvent {
    /// A position was opened.
    Entered { idx: usize, price: Price, size: f64, direction: Direction },
    /// A position was closed, producing a completed trade.
    Exited { idx: usize, trade: Trade },
    /// An entry signal was refused by the risk gate.
    EntryRejected { idx: usize, reason: RejectReason },
    /// An order started working (resting kinds) or was acknowledged
    /// (market kinds, immediately before their fill).
    OrderAccepted { idx: usize, order_id: u64, client_id: String },
    /// A stop-limit's trigger fired; its limit leg now rests.
    OrderTriggered { idx: usize, order_id: u64, client_id: String },
    /// An order filled. The position consequence follows as a separate
    /// [`EngineEvent::Entered`] or [`EngineEvent::Exited`] event.
    OrderFilled { idx: usize, order_id: u64, client_id: String, price: Price, size: f64 },
    /// An order was canceled (explicitly, or by IOC/FOK exhaustion).
    OrderCanceled { idx: usize, order_id: u64, client_id: String },
    /// An order's time-in-force lapsed.
    OrderExpired { idx: usize, order_id: u64, client_id: String },
    /// An order was refused: position state or sizing made it unfillable.
    OrderRejected { idx: usize, order_id: u64, client_id: String, reason: &'static str },
    /// Equity fell below the maintenance requirement (margin mode). New
    /// entries halt; open positions are not force-liquidated.
    MarginCall { idx: usize, equity: f64, required: f64 },
}

/// Per-bar inputs that vary independently of the bar itself.
#[derive(Debug, Clone, Copy, Default)]
pub struct StepInput {
    /// Entry signal for this bar (post signal-cleaning).
    ///
    /// Note: boolean entry/exit signals will be superseded by order intents
    /// from the class-based strategy contract; they remain supported for the
    /// array-based runners.
    pub entry: bool,
    /// Exit signal for this bar (post signal-cleaning).
    pub exit: bool,
    /// ATR value at this bar; `0.0` when no ATR-based stop/target is configured.
    pub atr: f64,
    /// Optional position-size multiplier from `CompiledSignals::position_sizes`.
    pub size_mult: Option<f64>,
    /// Explicit stop price for an entry opened on this bar.
    ///
    /// Takes precedence over the configured stop model. Ignored when no entry
    /// opens on this bar.
    pub stop_price_override: Option<Price>,
    /// Explicit target price for an entry opened on this bar.
    ///
    /// Takes precedence over the configured target model. Ignored when no
    /// entry opens on this bar.
    pub target_price_override: Option<Price>,
}

/// Read-only view of the currently open position.
#[derive(Debug, Clone, Copy)]
pub struct PositionSnapshot {
    /// Ledger position id (0-based, unique within a session).
    pub position_id: u64,
    /// Entry bar index.
    pub entry_idx: usize,
    /// Entry fill price (slippage-adjusted).
    pub entry_price: Price,
    /// Position size in units.
    pub size: f64,
    /// Trading direction.
    pub direction: Direction,
    /// Active stop price, if any.
    pub stop_price: Option<Price>,
    /// Active target price, if any.
    pub target_price: Option<Price>,
}

/// Stateful simulation core.
///
/// One instance simulates one instrument. All mutable simulation state that the
/// original loop kept as locals lives here.
#[derive(Debug)]
pub struct EngineKernel {
    config: BacktestConfig,
    fee_model: FeeModel,
    slippage_model: SlippageModel,
    fill_price: FillPrice,
    /// Limit/stop fill semantics, including gap-through handling.
    fill_model: FillModel,

    /// Open positions. Net policy holds at most one, reproducing the
    /// original single-position behavior; Independent allows hedging.
    ledger: PositionLedger,
    cash: f64,
    /// Trading direction for new signal-path entries.
    direction: Direction,
    /// Position ids the strategy asked to close, applied on the next step.
    pending_closes: Vec<u64>,
    /// Cash (default, historical) vs leveraged margin funding.
    account: AccountMode,
    /// Per-position locked margin, used only in margin mode.
    margin: MarginBook,
    /// Seeded stream for stochastic fills (prob < 1.0 configs only).
    fill_rng: FillRng,

    /// Pre-trade constraints, checked before an entry opens.
    risk: RiskGate,
    /// Open-position count the risk gate should see, when a portfolio owns
    /// it. `None` (the default) means count this kernel's own ledger.
    external_open_count: Option<usize>,

    effective_stop: StopConfig,
    effective_target: TargetConfig,
    /// Per-instrument capital cap and lot rounding, if any.
    alloted_capital: Option<f64>,
    lot_size: Option<f64>,
    /// Market definition: quantization, contract multiplier, expiry.
    ///
    /// `None` reproduces pre-spec behavior exactly (multiplier 1.0, no
    /// quantization, no expiry).
    spec: Option<InstrumentSpec>,

    /// Resting-order book for the class-based order API. Empty (and
    /// costless) for the signal-array path.
    orders: OrderEngine,
    /// Events produced between steps (order accepted/canceled), delivered
    /// at the front of the next step's event list.
    pending_events: Vec<EngineEvent>,
}

impl EngineKernel {
    /// Build a kernel from engine-level models and optional per-instrument config.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        config: BacktestConfig,
        fee_model: FeeModel,
        slippage_model: SlippageModel,
        fill_price: FillPrice,
        symbol: String,
        direction: Direction,
        inst_config: Option<&InstrumentConfig>,
    ) -> Self {
        // Per-instrument stop/target override the global config.
        let effective_stop =
            inst_config.and_then(|ic| ic.stop.as_ref()).copied().unwrap_or(config.stop);
        let effective_target =
            inst_config.and_then(|ic| ic.target.as_ref()).copied().unwrap_or(config.target);

        let cash = config.initial_capital;
        let config_seed = config.fill_seed;

        Self {
            config,
            fee_model,
            slippage_model,
            fill_price,
            fill_model: FillModel { fill_price, ..FillModel::default() },
            ledger: PositionLedger::new(symbol, PositionPolicy::Net),
            cash,
            direction,
            pending_closes: Vec::new(),
            account: AccountMode::Cash,
            margin: MarginBook::default(),
            fill_rng: FillRng::new(config_seed),
            risk: RiskGate::unconstrained(),
            external_open_count: None,
            effective_stop,
            effective_target,
            alloted_capital: inst_config.and_then(|ic| ic.alloted_capital),
            lot_size: inst_config.and_then(|ic| ic.lot_size),
            spec: None,
            orders: OrderEngine::new(),
            pending_events: Vec::new(),
        }
    }

    /// Attach pre-trade risk constraints.
    pub fn with_risk_gate(mut self, risk: RiskGate) -> Self {
        self.risk = risk;
        self
    }

    /// Set the position policy (netting vs independent/hedging).
    ///
    /// Must be set before any position opens; the default `Net` reproduces
    /// the historical single-position behavior.
    pub fn with_position_policy(mut self, policy: PositionPolicy) -> Self {
        self.set_position_policy(policy);
        self
    }

    /// In-place form of [`EngineKernel::with_position_policy`].
    pub fn set_position_policy(&mut self, policy: PositionPolicy) {
        debug_assert!(!self.ledger.is_in_position());
        let mut ledger = PositionLedger::new(self.ledger.symbol().to_string(), policy);
        ledger.set_contract_multiplier(self.ledger.contract_multiplier());
        self.ledger = ledger;
    }

    /// In-place form of [`EngineKernel::with_account_mode`].
    pub fn set_account_mode(&mut self, account: AccountMode) {
        self.account = account;
    }

    /// Set the account funding mode.
    ///
    /// The default `Cash` reproduces historical behavior exactly. `Margin`
    /// locks initial margin per position (instrument `margin_init`, else
    /// `1 / leverage`), marks equity with direction-aware unrealized PnL,
    /// and emits a `MarginCall` event that halts entries when equity falls
    /// below the maintenance requirement.
    pub fn with_account_mode(mut self, account: AccountMode) -> Self {
        if let AccountMode::Margin { leverage } = account {
            debug_assert!(leverage > 0.0, "leverage must be positive");
        }
        self.account = account;
        self
    }

    /// Per-position initial margin rate; `None` in cash mode.
    fn margin_rate(&self) -> Option<f64> {
        match self.account {
            AccountMode::Cash => None,
            AccountMode::Margin { leverage } => {
                let from_spec =
                    self.spec.as_ref().map(|s| s.margin_init).filter(|&m| m > 0.0);
                Some(from_spec.unwrap_or(1.0 / leverage.max(1.0)))
            }
        }
    }

    /// Maintenance margin rate in margin mode; half the initial rate when
    /// the instrument does not declare one.
    fn maint_rate(&self) -> Option<f64> {
        let init = self.margin_rate()?;
        let from_spec = self.spec.as_ref().map(|s| s.margin_maint).filter(|&m| m > 0.0);
        Some(from_spec.unwrap_or(init * 0.5))
    }

    /// Symbol this kernel simulates.
    pub fn symbol(&self) -> &str {
        // The ledger owns the symbol string; expose it for policy swaps and
        // event labeling.
        self.ledger.symbol()
    }

    /// Attach an instrument market definition.
    ///
    /// Enables price/size quantization, contract-multiplier notional scaling,
    /// and expiry settlement. An explicit `InstrumentConfig` lot size keeps
    /// precedence over the spec's, since it is the user's per-run override.
    pub fn with_instrument(mut self, spec: InstrumentSpec) -> Self {
        self.set_instrument(spec);
        self
    }

    /// In-place form of [`EngineKernel::with_instrument`], for owners that
    /// hold the kernel behind a field.
    pub fn set_instrument(&mut self, spec: InstrumentSpec) {
        if self.lot_size.is_none() && spec.lot_size > 0.0 && spec.lot_size != 1.0 {
            self.lot_size = Some(spec.lot_size);
        }
        self.ledger.set_contract_multiplier(spec.multiplier);
        self.spec = Some(spec);
    }

    /// Contract point value; `1.0` without a spec.
    #[inline]
    fn multiplier(&self) -> f64 {
        match &self.spec {
            Some(spec) if spec.multiplier > 0.0 => spec.multiplier,
            _ => 1.0,
        }
    }

    /// Current uninvested cash.
    #[inline]
    pub fn cash(&self) -> f64 {
        self.cash
    }

    /// Entries refused by the risk gate.
    #[inline]
    pub fn rejected_entries(&self) -> usize {
        self.risk.rejected_entries()
    }

    /// Overwrite available cash.
    ///
    /// Used by the shared-capital portfolio runner, which owns one pool across
    /// several kernels and re-points each one at the pool before stepping it.
    #[inline]
    pub fn set_cash(&mut self, cash: f64) {
        self.cash = cash;
    }

    /// Market value of open positions at the given price, or 0.0 when flat.
    #[inline]
    pub fn position_value(&self, close: Price) -> f64 {
        self.ledger.position_value(close)
    }

    /// Feed current equity to the drawdown kill-switch.
    #[inline]
    pub fn observe_equity(&mut self, equity: f64, peak_equity: f64) {
        self.risk.on_equity(equity, peak_equity);
    }

    /// Whether any position is currently open.
    #[inline]
    pub fn is_in_position(&self) -> bool {
        self.ledger.is_in_position()
    }

    /// Overwrite the earliest open position's stop price; no-op when flat.
    ///
    /// `None` removes the stop. The new price is checked on the next
    /// [`EngineKernel::step`] call. For a specific position under the
    /// Independent policy, use [`EngineKernel::set_stop_price_for`].
    pub fn set_stop_price(&mut self, price: Option<Price>) {
        if let Some(managed) = self.ledger.first_mut() {
            managed.position.stop_price = price;
        }
    }

    /// Overwrite the earliest open position's target price; no-op when flat.
    pub fn set_target_price(&mut self, price: Option<Price>) {
        if let Some(managed) = self.ledger.first_mut() {
            managed.position.target_price = price;
        }
    }

    /// Overwrite a specific position's stop price; `false` for unknown ids.
    pub fn set_stop_price_for(&mut self, position_id: u64, price: Option<Price>) -> bool {
        match self.ledger.get_mut(position_id) {
            Some(managed) => {
                managed.position.stop_price = price;
                true
            }
            None => false,
        }
    }

    /// Overwrite a specific position's target price; `false` for unknown ids.
    pub fn set_target_price_for(&mut self, position_id: u64, price: Option<Price>) -> bool {
        match self.ledger.get_mut(position_id) {
            Some(managed) => {
                managed.position.target_price = price;
                true
            }
            None => false,
        }
    }

    /// Request a close of a specific position; applied on the next step at
    /// the configured fill-price model, like a signal exit.
    pub fn request_close(&mut self, position_id: u64) {
        self.pending_closes.push(position_id);
    }

    /// Submit an order from the class-based order API.
    ///
    /// `submitted_idx` is the bar the strategy was observing when it placed
    /// the order: market orders fill on that bar's step (matching the
    /// signal-entry contract), resting orders begin matching on the next
    /// bar. Returns the engine order id; acknowledgment events are
    /// delivered at the front of the next step's event list.
    #[allow(clippy::too_many_arguments)]
    pub fn submit_order(
        &mut self,
        side: OrderSide,
        qty: QtySpec,
        kind: OrderKind,
        tif: TimeInForce,
        submitted_idx: usize,
        submitted_ts: i64,
        client_id: String,
        stop_price: Option<Price>,
        target_price: Option<Price>,
    ) -> u64 {
        self.submit_order_full(
            side,
            qty,
            kind,
            tif,
            submitted_idx,
            submitted_ts,
            client_id,
            stop_price,
            target_price,
            false,
            false,
            None,
        )
    }

    /// [`EngineKernel::submit_order`] with flags and one-triggers-other
    /// linkage. `parent_id` holds the order (unmatched, no expiry clock)
    /// until the parent fills; a dead parent cancels it.
    #[allow(clippy::too_many_arguments)]
    pub fn submit_order_full(
        &mut self,
        side: OrderSide,
        qty: QtySpec,
        kind: OrderKind,
        tif: TimeInForce,
        submitted_idx: usize,
        submitted_ts: i64,
        client_id: String,
        stop_price: Option<Price>,
        target_price: Option<Price>,
        post_only: bool,
        reduce_only: bool,
        parent_id: Option<u64>,
    ) -> u64 {
        let mut order = Order::plain(side, qty, kind, tif);
        order.client_id = client_id.clone();
        order.submitted_idx = submitted_idx;
        order.submitted_ts = submitted_ts;
        order.stop_price = stop_price;
        order.target_price = target_price;
        order.post_only = post_only;
        order.reduce_only = reduce_only;
        order.parent_id = parent_id;
        let id = self.orders.submit(order);

        // Resting kinds start working immediately (held children stay
        // Submitted until their parent fills); plain market kinds are
        // acknowledged when their fill is processed in the step.
        let rests = !matches!(kind, OrderKind::Market)
            || matches!(tif, TimeInForce::AtOpen | TimeInForce::AtClose);
        if rests && parent_id.is_none() {
            if let Some(order) = self.orders.get_mut(id) {
                let _ = order.transition(OrderStatus::Accepted);
            }
            self.pending_events.push(EngineEvent::OrderAccepted {
                idx: submitted_idx,
                order_id: id,
                client_id,
            });
        }
        id
    }

    /// Put a set of working orders in one one-cancels-other group: the first
    /// fill among them cancels the rest. One-updates-other reduces to this
    /// while fills are all-or-nothing.
    pub fn link_oco(&mut self, ids: &[u64]) {
        let group = ids.iter().copied().min().unwrap_or(0);
        for id in ids {
            if let Some(order) = self.orders.get_mut(*id) {
                order.oco_group = Some(group);
            }
        }
    }

    /// Cancel a working order. Returns `false` for unknown/finished ids.
    pub fn cancel_order(&mut self, idx: usize, id: u64) -> bool {
        let client_id = match self.orders.get(id) {
            Some(order) => order.client_id.clone(),
            None => return false,
        };
        if self.orders.cancel(id) {
            self.pending_events.push(EngineEvent::OrderCanceled { idx, order_id: id, client_id });
            true
        } else {
            false
        }
    }

    /// Cancel every working order.
    pub fn cancel_all_orders(&mut self, idx: usize) -> Vec<u64> {
        let ids = self.orders.cancel_all();
        for id in &ids {
            let client_id =
                self.orders.get(*id).map(|o| o.client_id.clone()).unwrap_or_default();
            self.pending_events.push(EngineEvent::OrderCanceled {
                idx,
                order_id: *id,
                client_id,
            });
        }
        ids
    }

    /// Replace a working order's prices/quantity. Returns `false` when the
    /// order is unknown, finished, or the modification is not applicable.
    pub fn modify_order(
        &mut self,
        id: u64,
        qty: Option<QtySpec>,
        limit_price: Option<Price>,
        trigger_price: Option<Price>,
    ) -> bool {
        self.orders.modify(id, qty, limit_price, trigger_price)
    }

    /// Shared view of an order by engine id.
    pub fn order(&self, id: u64) -> Option<&Order> {
        self.orders.get(id)
    }

    /// Instrument tick size; `0.0` without a spec.
    pub fn price_increment(&self) -> f64 {
        self.spec.as_ref().map(|s| s.price_increment).unwrap_or(0.0)
    }

    /// All non-terminal orders, in submission order.
    pub fn open_orders(&self) -> Vec<&Order> {
        self.orders.working().collect()
    }

    /// Read-only view of the earliest open position, or `None` when flat.
    pub fn position_snapshot(&self) -> Option<PositionSnapshot> {
        self.ledger.first().map(Self::snapshot_of)
    }

    /// Read-only views of every open position, in opening order.
    pub fn position_snapshots(&self) -> Vec<PositionSnapshot> {
        self.ledger.positions().iter().map(Self::snapshot_of).collect()
    }

    fn snapshot_of(managed: &crate::portfolio::ledger::ManagedPosition) -> PositionSnapshot {
        let p = &managed.position;
        PositionSnapshot {
            position_id: managed.id,
            entry_idx: p.entry_idx,
            entry_price: p.entry_price,
            size: p.size,
            direction: p.direction,
            stop_price: p.stop_price,
            target_price: p.target_price,
        }
    }

    /// Mark-to-market equity at the given price.
    ///
    /// Cash mode marks positions at full value (historical model); margin
    /// mode marks balance plus direction-aware unrealized PnL, which prices
    /// shorts correctly.
    #[inline]
    pub fn equity(&self, close: Price) -> f64 {
        match self.account {
            AccountMode::Cash => self.cash + self.position_value(close),
            AccountMode::Margin { .. } => self.cash + self.ledger.unrealized_total(close),
        }
    }

    /// Cash not locked as initial margin (margin mode); all cash otherwise.
    #[inline]
    pub fn free_capital(&self) -> f64 {
        match self.account {
            AccountMode::Cash => self.cash,
            AccountMode::Margin { .. } => self.cash - self.margin.total_locked(),
        }
    }

    /// Initial margin locked by this kernel's open positions.
    ///
    /// The portfolio session sums these across kernels to keep its shared
    /// account's aggregate in step.
    #[inline]
    pub fn locked_margin(&self) -> f64 {
        self.margin.total_locked()
    }

    /// Open positions in this kernel's own ledger.
    #[inline]
    pub fn open_count(&self) -> usize {
        self.ledger.open_count()
    }

    /// Override the open-position count the risk gate checks against.
    ///
    /// A portfolio session sets this to the count across *all* its
    /// instruments before stepping a kernel, so `max_positions` means
    /// concurrent positions portfolio-wide rather than per instrument, and
    /// clears it afterward. `None` restores ledger-derived counting.
    #[inline]
    pub fn set_external_open_count(&mut self, count: Option<usize>) {
        self.external_open_count = count;
    }

    /// The open-position count the risk gate is checked against.
    #[inline]
    fn gating_open_count(&self) -> usize {
        self.external_open_count.unwrap_or_else(|| self.ledger.open_count())
    }

    /// Direction-aware unrealized PnL of open positions, or 0.0 when flat.
    ///
    /// The margin-mode marking counterpart of [`EngineKernel::position_value`].
    #[inline]
    pub fn unrealized_value(&self, close: Price) -> f64 {
        self.ledger.unrealized_total(close)
    }

    /// Maintenance margin required by this kernel's open positions; 0.0 in
    /// cash mode or when flat.
    ///
    /// Lives here rather than in the session so each instrument's own
    /// `margin_maint` applies: a portfolio requirement is the sum of these,
    /// which a single blended rate would get wrong.
    #[inline]
    pub fn maintenance_requirement(&self, close: Price) -> f64 {
        match self.maint_rate() {
            Some(rate) => self.ledger.notional_total(close) * rate,
            None => 0.0,
        }
    }

    /// Trip this kernel's margin-call kill-switch, blocking further entries.
    ///
    /// The portfolio session calls this on every kernel so one shared
    /// account's margin call halts all of its instruments.
    #[inline]
    pub fn halt_margin(&mut self) {
        self.margin.halt();
    }

    /// Whether the drawdown kill-switch on the risk gate has tripped.
    #[inline]
    pub fn risk_halted(&self) -> bool {
        self.risk.is_halted()
    }

    /// Whether this kernel's margin-call kill-switch has tripped.
    #[inline]
    pub fn is_margin_halted(&self) -> bool {
        self.margin.is_halted()
    }

    /// Advance the simulation by one bar.
    ///
    /// Order of operations is load-bearing and mirrors the original loop:
    /// update extremes, then exits (stop > target > signal), then entries.
    /// An exit and a re-entry may both occur on the same bar.
    pub fn step(&mut self, idx: usize, bar: &KernelBar, input: StepInput) -> Vec<EngineEvent> {
        // Acknowledgments queued between steps (order accepted/canceled)
        // lead the event list, preserving submission-time ordering.
        let mut events = std::mem::take(&mut self.pending_events);

        // Expiry settlement pre-empts everything: the contract no longer
        // trades at this bar, so neither exits-by-signal nor entries apply.
        // Working orders die with the contract.
        if self.spec.as_ref().is_some_and(|s| s.is_expired_at(bar.timestamp)) {
            events.extend(self.settle_expiry(idx, bar));
            self.cancel_all_orders(idx);
            events.append(&mut self.pending_events);
            if input.entry {
                self.risk.record_rejection();
                events.push(EngineEvent::EntryRejected { idx, reason: RejectReason::Expired });
            }
            return events;
        }

        // Track running extremes for trailing stops.
        self.ledger.update_price(bar.high, bar.low);

        // Protective/signal exits, per position in opening order. Net policy
        // holds one position, reproducing the original sequence exactly.
        let open_ids: Vec<u64> = self.ledger.positions().iter().map(|p| p.id).collect();
        for position_id in open_ids {
            if let Some(event) = self.try_exit_position(idx, bar, position_id, input.exit) {
                events.push(event);
            }

            // Trail only if the position survived this bar.
            if let StopConfig::Trailing { percent } = self.effective_stop {
                if let Some(managed) = self.ledger.get_mut(position_id) {
                    managed.update_trailing_stop(percent);
                }
            }
        }

        // Strategy-requested closes of specific positions (hedging API),
        // filled like signal exits at the configured fill-price model.
        let requested: Vec<u64> = std::mem::take(&mut self.pending_closes);
        for position_id in requested {
            if self.ledger.get(position_id).is_none() {
                continue; // already closed by a stop/target this bar
            }
            let direction = match self.ledger.get(position_id) {
                Some(managed) => managed.position.direction,
                None => continue,
            };
            let price = self.fill_price_for(bar, direction, false);
            if let Some(event) = self.close_at(idx, bar, position_id, price, ExitReason::Signal) {
                events.push(event);
            }
        }

        // Margin maintenance: mark against this bar's close; a breach halts
        // new entries (latching) but does not force-liquidate.
        if self.ledger.is_in_position() && !self.margin.is_halted() {
            let required = self.maintenance_requirement(bar.close);
            if required > 0.0 {
                let equity = self.equity(bar.close);
                if equity < required {
                    self.margin.halt();
                    events.push(EngineEvent::MarginCall { idx, equity, required });
                }
            }
        }

        // Resting orders committed on earlier bars match against this bar,
        // after the position's own protective exits (stop > target > signal
        // keeps its priority) and before this bar's new signals.
        let outcomes = self.orders.match_bar(idx, &bar.to_ohlcv_bar(), &self.fill_model);
        for outcome in outcomes {
            self.apply_match_outcome(idx, bar, outcome, &mut events);
        }

        if !self.ledger.is_in_position() && input.entry {
            // Not-yet-active instruments refuse entries the same way expired
            // ones do, before the risk gate sees them.
            let active = self
                .spec
                .as_ref()
                .and_then(|s| s.activation_ns)
                .is_none_or(|act| bar.timestamp >= act);
            if !active {
                self.risk.record_rejection();
                events.push(EngineEvent::EntryRejected { idx, reason: RejectReason::Inactive });
                return events;
            }

            if self.margin.is_halted() {
                self.risk.record_rejection();
                events.push(EngineEvent::EntryRejected { idx, reason: RejectReason::MarginCall });
                return events;
            }

            // Gate before opening, so a refused entry never reaches the equity
            // curve and the metrics describe the constrained run.
            let open_positions = self.gating_open_count();
            match self.risk.check_entry(open_positions) {
                Ok(()) => {
                    if let Some(event) = self.try_enter(idx, bar, input) {
                        events.push(event);
                    }
                }
                Err(reason) => {
                    self.risk.record_rejection();
                    events.push(EngineEvent::EntryRejected { idx, reason });
                }
            }
        }

        // Market orders placed while this bar was observed fill last, at the
        // configured fill-price model — the same contract as signal entries.
        let market_ids: Vec<u64> = self
            .orders
            .working()
            .filter(|o| {
                matches!(o.kind, OrderKind::Market)
                    && o.submitted_idx == idx
                    && o.parent_id.is_none()
                    && !matches!(o.tif, TimeInForce::AtOpen | TimeInForce::AtClose)
            })
            .map(|o| o.id)
            .collect();
        for id in market_ids {
            if let Some(order) = self.orders.get_mut(id) {
                let _ = order.transition(OrderStatus::Accepted);
                let client_id = order.client_id.clone();
                events.push(EngineEvent::OrderAccepted { idx, order_id: id, client_id });
            }
            self.apply_match_outcome(
                idx,
                bar,
                MatchOutcome::Fill { order_id: id, price: f64::NAN },
                &mut events,
            );
        }

        events
    }

    /// Apply one matching outcome to position/cash state.
    ///
    /// A `Fill` outcome is a *marketable* order, not a done deal: position
    /// state may still refuse it (opening while a position is open, closing
    /// while flat), in which case the order rejects. A NaN fill price marks
    /// a market order, priced here by the fill-price model.
    fn apply_match_outcome(
        &mut self,
        idx: usize,
        bar: &KernelBar,
        outcome: MatchOutcome,
        events: &mut Vec<EngineEvent>,
    ) {
        let (id, matched_price) = match outcome {
            MatchOutcome::Trigger { order_id } => {
                let client_id =
                    self.orders.get(order_id).map(|o| o.client_id.clone()).unwrap_or_default();
                events.push(EngineEvent::OrderTriggered { idx, order_id, client_id });
                return;
            }
            MatchOutcome::Expire { order_id } => {
                let client_id =
                    self.orders.get(order_id).map(|o| o.client_id.clone()).unwrap_or_default();
                events.push(EngineEvent::OrderExpired { idx, order_id, client_id });
                return;
            }
            MatchOutcome::Cancel { order_id } => {
                let client_id =
                    self.orders.get(order_id).map(|o| o.client_id.clone()).unwrap_or_default();
                events.push(EngineEvent::OrderCanceled { idx, order_id, client_id });
                return;
            }
            MatchOutcome::Reject { order_id, reason } => {
                let client_id =
                    self.orders.get(order_id).map(|o| o.client_id.clone()).unwrap_or_default();
                events.push(EngineEvent::OrderRejected { idx, order_id, client_id, reason });
                return;
            }
            MatchOutcome::Fill { order_id, price } => (order_id, price),
        };

        let Some(order) = self.orders.get(id) else { return };
        let side = order.side;
        let qty = order.qty;
        let kind = order.kind;
        let status = order.status;
        let client_id = order.client_id.clone();
        let stop_attach = order.stop_price;
        let target_attach = order.target_price;
        let reduce_only = order.reduce_only;

        // Stochastic fills: a marketable resting limit may be passed over
        // (queue position, exhausted liquidity); it stays working. Stop and
        // market fills may instead slip one tick against the trader.
        let is_limit_fill = matches!(kind, OrderKind::Limit { .. })
            || (matches!(kind, OrderKind::StopLimit { .. }) && status == OrderStatus::Triggered);
        if is_limit_fill && self.config.fill_prob_limit < 1.0 {
            if self.fill_rng.next_f64() >= self.config.fill_prob_limit {
                return; // untouched: still Accepted/Triggered, retries next bar
            }
        }
        let matched_price = if !is_limit_fill
            && !matched_price.is_nan()
            && self.config.fill_prob_slippage > 0.0
            && self.fill_rng.next_f64() < self.config.fill_prob_slippage
        {
            match &self.spec {
                Some(spec) if spec.price_increment > 0.0 => match side {
                    OrderSide::Buy => matched_price + spec.price_increment,
                    OrderSide::Sell => matched_price - spec.price_increment,
                },
                _ => matched_price,
            }
        } else {
            matched_price
        };

        let mut reject = |orders: &mut OrderEngine, events: &mut Vec<EngineEvent>, reason| {
            if let Some(order) = orders.get_mut(id) {
                let _ = order.transition(OrderStatus::Rejected);
            }
            events.push(EngineEvent::OrderRejected {
                idx,
                order_id: id,
                client_id: client_id.clone(),
                reason,
            });
        };

        // Net policy: buy opens for a long-direction kernel and closes a
        // short one; sell is the mirror. Independent (hedging) policy:
        // every order opens in its own side's direction — closes are
        // explicit (`request_close`) or via protective levels.
        let hedging = self.ledger.policy() == PositionPolicy::Independent;
        let (opens, open_direction) = if hedging {
            let dir = match side {
                OrderSide::Buy => Direction::Long,
                OrderSide::Sell => Direction::Short,
            };
            (true, dir)
        } else {
            let opens = match (self.direction, side) {
                (Direction::Long, OrderSide::Buy) | (Direction::Short, OrderSide::Sell) => true,
                (Direction::Long, OrderSide::Sell) | (Direction::Short, OrderSide::Buy) => false,
            };
            (opens, self.direction)
        };

        if opens {
            if reduce_only {
                // A reduce-only order must never increase exposure.
                reject(&mut self.orders, events, "reduce_only");
                return;
            }
            if !hedging && self.ledger.is_in_position() {
                reject(&mut self.orders, events, "position_open");
                return;
            }
            if self.margin.is_halted() {
                self.risk.record_rejection();
                reject(&mut self.orders, events, "margin_call");
                return;
            }
            if let Err(reason) = self.risk.check_entry(self.gating_open_count()) {
                self.risk.record_rejection();
                reject(&mut self.orders, events, reason.as_str());
                return;
            }
            let raw_price = if matched_price.is_nan() {
                self.fill_price_for(bar, open_direction, true)
            } else {
                matched_price
            };
            let (size_mult, explicit_units) = match qty {
                QtySpec::Units(u) => (None, Some(u)),
                QtySpec::CapitalFrac(f) => (Some(f), None),
                QtySpec::FullPosition => {
                    reject(&mut self.orders, events, "invalid_qty");
                    return;
                }
            };
            match self.open_at(
                idx,
                bar,
                open_direction,
                raw_price,
                size_mult,
                explicit_units,
                0.0,
                stop_attach,
                target_attach,
            ) {
                Some(EngineEvent::Entered { price, size, direction, .. }) => {
                    if let Some(order) = self.orders.get_mut(id) {
                        let _ = order.transition(OrderStatus::Filled);
                    }
                    events.push(EngineEvent::OrderFilled {
                        idx,
                        order_id: id,
                        client_id,
                        price,
                        size,
                    });
                    events.push(EngineEvent::Entered { idx, price, size, direction });
                    self.after_fill(idx, id, events);
                }
                Some(EngineEvent::EntryRejected { reason, .. }) => {
                    reject(&mut self.orders, events, reason.as_str());
                }
                _ => reject(&mut self.orders, events, "unfillable"),
            }
        } else {
            let Some(first) = self.ledger.first() else {
                reject(&mut self.orders, events, "no_position");
                return;
            };
            let position_id = first.id;
            let direction = first.position.direction;
            let raw_price = if matched_price.is_nan() {
                self.fill_price_for(bar, direction, false)
            } else {
                matched_price
            };
            match self.close_at(idx, bar, position_id, raw_price, ExitReason::Order) {
                Some(EngineEvent::Exited { trade, .. }) => {
                    if let Some(order) = self.orders.get_mut(id) {
                        let _ = order.transition(OrderStatus::Filled);
                    }
                    events.push(EngineEvent::OrderFilled {
                        idx,
                        order_id: id,
                        client_id,
                        price: trade.exit_price,
                        size: trade.size,
                    });
                    events.push(EngineEvent::Exited { idx, trade });
                    self.after_fill(idx, id, events);
                }
                _ => reject(&mut self.orders, events, "unfillable"),
            }
        }
    }

    /// Contingency consequences of a fill: activate held one-triggers-other
    /// children, then cancel one-cancels-other siblings.
    fn after_fill(&mut self, idx: usize, filled_id: u64, events: &mut Vec<EngineEvent>) {
        let children: Vec<(u64, String)> = self
            .orders
            .all()
            .iter()
            .filter(|o| o.parent_id == Some(filled_id) && o.status == OrderStatus::Submitted)
            .map(|o| (o.id, o.client_id.clone()))
            .collect();
        for (child_id, client_id) in children {
            if let Some(order) = self.orders.get_mut(child_id) {
                let _ = order.transition(OrderStatus::Accepted);
            }
            events.push(EngineEvent::OrderAccepted { idx, order_id: child_id, client_id });
        }

        let group = self.orders.get(filled_id).and_then(|o| o.oco_group);
        if let Some(group) = group {
            let siblings: Vec<(u64, String)> = self
                .orders
                .all()
                .iter()
                .filter(|o| {
                    o.oco_group == Some(group) && o.id != filled_id && !o.status.is_terminal()
                })
                .map(|o| (o.id, o.client_id.clone()))
                .collect();
            for (sibling_id, client_id) in siblings {
                if self.orders.cancel(sibling_id) {
                    events.push(EngineEvent::OrderCanceled {
                        idx,
                        order_id: sibling_id,
                        client_id,
                    });
                }
            }
        }
    }

    /// Exit path for one position: stop-loss, then take-profit, then signal.
    fn try_exit_position(
        &mut self,
        idx: usize,
        bar: &KernelBar,
        position_id: u64,
        exit_signal: bool,
    ) -> Option<EngineEvent> {
        let managed = self.ledger.get(position_id)?;
        let mut exit_reason: Option<ExitReason> = None;
        let mut exit_price = bar.close;

        let direction = managed.position.direction;
        let ohlcv_bar = bar.to_ohlcv_bar();

        let stop_hit = managed.is_stop_hit(bar.low, bar.high);
        let target_hit = managed.is_target_hit(bar.low, bar.high);

        // When both protective levels are touched in one bar, the legacy
        // assumption is stop-first (conservative). The adaptive path model
        // infers the traversal from candle geometry instead: an up-candle
        // is assumed open→low→high→close, a down-candle open→high→low→
        // close, so the level on the first-visited side fills.
        let target_first = stop_hit
            && target_hit
            && self.config.bar_path_adaptive
            && match direction {
                // Long: target above. Down-candle visits the high first.
                Direction::Long => bar.close < bar.open,
                // Short: target below. Up-candle visits the low first.
                Direction::Short => bar.close >= bar.open,
            };

        if target_first {
            let target_price = managed.position.target_price?;
            exit_reason = Some(ExitReason::TakeProfit);
            exit_price = self
                .fill_model
                .get_limit_fill_price(target_price, &ohlcv_bar, direction, false)
                .unwrap_or(target_price);
        }

        // Stop-loss, with gap-through adjustment against the bar open.
        //
        // Delegates to FillModel, which covers all four (direction, is_entry)
        // cases; the engine previously inlined a long/short-only copy of this.
        if exit_reason.is_none() && stop_hit {
            let stop_price = managed.position.stop_price?;
            exit_reason = Some(ExitReason::StopLoss);
            exit_price = self
                .fill_model
                .get_stop_fill_price(stop_price, &ohlcv_bar, direction, false)
                .unwrap_or(stop_price);
        }

        // Take-profit, filled at the limit price.
        if exit_reason.is_none() && target_hit {
            let target_price = managed.position.target_price?;
            exit_reason = Some(ExitReason::TakeProfit);
            exit_price = self
                .fill_model
                .get_limit_fill_price(target_price, &ohlcv_bar, direction, false)
                .unwrap_or(target_price);
        }

        // Exit signal.
        if exit_reason.is_none() && exit_signal {
            exit_reason = Some(ExitReason::Signal);
            exit_price = self.fill_price_for(bar, direction, false);
        }

        let reason = exit_reason?;
        self.close_at(idx, bar, position_id, exit_price, reason)
    }

    /// Apply a close at a determined raw price: slippage, fees, position
    /// close, cash credit. Shared by the signal path ([`Self::try_exit`])
    /// and order-driven closes, so both produce identical arithmetic.
    fn close_at(
        &mut self,
        idx: usize,
        bar: &KernelBar,
        position_id: u64,
        exit_price: Price,
        reason: ExitReason,
    ) -> Option<EngineEvent> {
        let managed = self.ledger.get(position_id)?;
        let direction = managed.position.direction;
        let size = managed.position.size;
        let entry_ts = managed.entry_timestamp;
        let entry_breakdown = managed.entry_breakdown;

        let exit_price = self.slippage_model.apply(exit_price, direction, false, Some(bar.volume));

        // calculate_side, not calculate: STT lands on the sell leg and stamp
        // duty on the buy leg, so entry and exit are not symmetric.
        //
        // Fee models see the per-contract currency price (price * contract
        // multiplier) and the raw contract count: value-based schedules
        // (percentage, tiered, itemized) then charge on true notional while
        // per-contract schedules charge per contract, not per notional unit.
        let fee_price = exit_price * self.multiplier();
        let exit_breakdown = self.fee_model.breakdown(fee_price, size, direction, false);
        let fees = match exit_breakdown {
            Some(b) => b.total(),
            None => self.fee_model.calculate(fee_price, size, direction),
        };

        // Round-trip breakdown: entry components plus exit components, so the
        // itemized total equals the fees actually deducted from the equity curve.
        let combined = match (entry_breakdown, exit_breakdown) {
            (Some(entry), Some(exit)) => {
                let mut total = entry;
                total.add(&exit);
                Some(total)
            }
            (entry, exit) => entry.or(exit),
        };

        let trade = self.ledger.close_position(
            position_id,
            ExitDetails {
                idx,
                timestamp: bar.timestamp,
                price: exit_price,
                entry_timestamp: entry_ts,
                reason,
                fees,
                fee_breakdown: combined,
            },
        )?;

        self.credit_close(position_id, &trade, fees, exit_price);

        Some(EngineEvent::Exited { idx, trade })
    }

    /// Credit a closed position back to the account.
    ///
    /// Cash mode returns the marked value minus exit fees (historical
    /// arithmetic, golden-pinned). Margin mode releases the locked margin
    /// and books realized PnL: `trade.pnl + entry_fees` equals gross PnL
    /// minus exit fees, and entry fees were already debited at entry.
    fn credit_close(&mut self, position_id: u64, trade: &Trade, exit_fees: f64, exit_price: Price) {
        match self.account {
            AccountMode::Cash => {
                self.cash += exit_price * trade.size * self.multiplier() - exit_fees;
            }
            AccountMode::Margin { .. } => {
                self.margin.release(position_id);
                let entry_fees = trade.fees - exit_fees;
                self.cash += trade.pnl + entry_fees;
            }
        }
    }

    /// Force-close the open position at contract expiry.
    ///
    /// Options settle to intrinsic value when the spec can compute one (the
    /// settlement bar's close is the fallback price); linear contracts settle
    /// at the close. Settlement is an expiry event, not a trade-out, so no
    /// exit fees are charged — consistent with [`EngineKernel::finalize`].
    fn settle_expiry(&mut self, idx: usize, bar: &KernelBar) -> Vec<EngineEvent> {
        let settle_price = match &self.spec {
            Some(spec) => spec.settlement_value(bar.close, None),
            None => bar.close,
        };

        let ids: Vec<u64> = self.ledger.positions().iter().map(|p| p.id).collect();
        let mut events = Vec::new();
        for position_id in ids {
            let Some(managed) = self.ledger.get(position_id) else { continue };
            let entry_ts = managed.entry_timestamp;
            let entry_breakdown = managed.entry_breakdown;
            if let Some(trade) = self.ledger.close_position(
                position_id,
                ExitDetails {
                    idx,
                    timestamp: bar.timestamp,
                    price: settle_price,
                    entry_timestamp: entry_ts,
                    reason: ExitReason::Settlement,
                    fees: 0.0,
                    fee_breakdown: entry_breakdown,
                },
            ) {
                self.credit_close(position_id, &trade, 0.0, settle_price);
                events.push(EngineEvent::Exited { idx, trade });
            }
        }
        events
    }

    /// Entry path: size against available capital, round to lot, open.
    fn try_enter(&mut self, idx: usize, bar: &KernelBar, input: StepInput) -> Option<EngineEvent> {
        let entry_price = self.fill_price_for(bar, self.direction, true);
        self.open_at(
            idx,
            bar,
            self.direction,
            entry_price,
            input.size_mult,
            None,
            input.atr,
            input.stop_price_override,
            input.target_price_override,
        )
    }

    /// Apply an open at a determined raw price: slippage, sizing, fees,
    /// position open, cash debit. Shared by the signal path
    /// ([`Self::try_enter`]) and order-driven opens, so both produce
    /// identical arithmetic. `explicit_units` bypasses capital-fraction
    /// sizing (order API); lot/size-increment rounding still applies.
    #[allow(clippy::too_many_arguments)]
    fn open_at(
        &mut self,
        idx: usize,
        bar: &KernelBar,
        direction: Direction,
        entry_price: Price,
        size_mult: Option<f64>,
        explicit_units: Option<f64>,
        atr: f64,
        stop_override: Option<Price>,
        target_override: Option<Price>,
    ) -> Option<EngineEvent> {
        let adjusted_price =
            self.slippage_model.apply(entry_price, direction, true, Some(bar.volume));

        // Per-instrument capital cap, never exceeding free capital on hand
        // (all cash in cash mode; cash minus locked margin in margin mode).
        let free = self.free_capital();
        let available = self.alloted_capital.map(|cap| cap.min(free)).unwrap_or(free);

        // Cash mode: size = capital / (price * multiplier * (1 + fee_rate))
        // so notional value + entry fee fits. Margin mode: only the initial
        // margin plus the fee must fit.
        let margin_rate = self.margin_rate();
        let contract_value = adjusted_price * self.multiplier();
        let fee_rate = self.config.fees;
        let sizing_denominator = match margin_rate {
            None => contract_value * (1.0 + fee_rate),
            Some(rate) => contract_value * (rate + fee_rate),
        };
        let raw_size = match explicit_units {
            Some(units) => units,
            None => match size_mult {
                Some(mult) => mult * available / sizing_denominator,
                None => available / sizing_denominator,
            },
        };

        let size = match self.lot_size {
            Some(lot) if lot > 0.0 => (raw_size / lot).floor() * lot,
            _ => raw_size,
        };
        let size = match &self.spec {
            Some(spec) => spec.quantize_size(size),
            None => size,
        };

        if size <= 0.0 {
            // Surface the discarded entry instead of silently skipping it —
            // strategies (and their authors) need to learn that the sizing
            // produced zero units, e.g. a size fraction too small for the
            // instrument's lot size. Deliberately does not touch the risk
            // gate's rejection counter: that metric describes constraint
            // refusals, not sizing arithmetic.
            return Some(EngineEvent::EntryRejected { idx, reason: RejectReason::ZeroSize });
        }

        // Same per-contract price convention as the exit path: notional
        // scaling rides on the price, contract count stays raw.
        let entry_breakdown = self.fee_model.breakdown(contract_value, size, direction, true);
        let entry_fees = match entry_breakdown {
            Some(b) => b.total(),
            None => self.fee_model.calculate(contract_value, size, direction),
        };

        // Capital-fraction sizing fits by construction; explicit unit counts
        // (order API) can exceed the account and are refused instead of
        // silently driving cash negative.
        let funding_cost = match margin_rate {
            None => contract_value * size,
            Some(rate) => contract_value * size * rate,
        };
        if explicit_units.is_some() && funding_cost + entry_fees > available {
            return Some(EngineEvent::EntryRejected {
                idx,
                reason: RejectReason::InsufficientCapital,
            });
        }
        let (config_stop, config_target) = self.stop_and_target(adjusted_price, direction, atr);
        // Derived protective prices land on the instrument's tick grid,
        // rounded conservatively; explicit overrides are the caller's exact
        // prices and pass through untouched.
        let quantize = |price: Price| match &self.spec {
            Some(spec) => spec.quantize_protective(price, direction),
            None => price,
        };
        let stop_price = stop_override.or(config_stop.map(quantize));
        let target_price = target_override.or(config_target.map(quantize));

        let position_id = self.ledger.open_position(
            idx,
            bar.timestamp,
            adjusted_price,
            size,
            direction,
            stop_price,
            target_price,
            entry_fees,
            entry_breakdown,
        )?;
        match margin_rate {
            None => self.cash -= contract_value * size + entry_fees,
            Some(rate) => {
                self.margin.lock(position_id, contract_value * size * rate);
                self.cash -= entry_fees;
            }
        }

        Some(EngineEvent::Entered { idx, price: adjusted_price, size, direction })
    }

    /// Force-close any open position at end of data.
    ///
    /// Marked-to-market with zero exit fees: the position is not actually
    /// traded out, so charging exit costs would understate the result.
    /// Returns the earliest position's trade for signature compatibility;
    /// multi-position callers use [`EngineKernel::finalize_all`].
    pub fn finalize(&mut self, idx: usize, bar: &KernelBar) -> Option<Trade> {
        self.finalize_all(idx, bar).into_iter().next()
    }

    /// Force-close every open position at end of data, in opening order.
    pub fn finalize_all(&mut self, idx: usize, bar: &KernelBar) -> Vec<Trade> {
        let ids: Vec<u64> = self.ledger.positions().iter().map(|p| p.id).collect();
        let mut trades = Vec::new();
        for position_id in ids {
            let Some(managed) = self.ledger.get(position_id) else { continue };
            let entry_ts = managed.entry_timestamp;
            let entry_breakdown = managed.entry_breakdown;
            if let Some(trade) = self.ledger.close_position(
                position_id,
                ExitDetails {
                    idx,
                    timestamp: bar.timestamp,
                    price: bar.close,
                    entry_timestamp: entry_ts,
                    reason: ExitReason::EndOfData,
                    fees: 0.0,
                    fee_breakdown: entry_breakdown,
                },
            ) {
                self.credit_close(position_id, &trade, 0.0, bar.close);
                trades.push(trade);
            }
        }
        trades
    }

    /// Resolve fill price from the configured price model.
    ///
    /// Delegates to [`FillPrice::get_price_from_arrays`] rather than matching
    /// inline: the `Worst`/`Best` variants are direction- and entry-dependent,
    /// and duplicating that table invites drift.
    fn fill_price_for(&self, bar: &KernelBar, direction: Direction, is_entry: bool) -> Price {
        self.fill_price
            .get_price_from_arrays(bar.open, bar.high, bar.low, bar.close, direction, is_entry)
    }

    /// Compute stop and target prices for a new position from configuration.
    fn stop_and_target(
        &self,
        entry_price: Price,
        direction: Direction,
        atr_value: f64,
    ) -> (Option<Price>, Option<Price>) {
        let multiplier = direction.multiplier();

        // ATR of 0.0 means warmup has not completed; no stop/target is set
        // rather than one pinned at the entry price.
        let stop_price = match self.effective_stop {
            StopConfig::None => None,
            StopConfig::Fixed { percent } => Some(entry_price * (1.0 - multiplier * percent)),
            StopConfig::Atr { multiplier: atr_mult, .. } => {
                if atr_value > 0.0 {
                    Some(entry_price - multiplier * atr_mult * atr_value)
                } else {
                    None
                }
            }
            StopConfig::Trailing { percent } => Some(entry_price * (1.0 - multiplier * percent)),
        };

        let target_price = match self.effective_target {
            TargetConfig::None => None,
            TargetConfig::Fixed { percent } => Some(entry_price * (1.0 + multiplier * percent)),
            TargetConfig::Atr { multiplier: atr_mult, .. } => {
                if atr_value > 0.0 {
                    Some(entry_price + multiplier * atr_mult * atr_value)
                } else {
                    None
                }
            }
            TargetConfig::RiskReward { ratio } => stop_price.map(|sp| {
                let risk = (entry_price - sp).abs();
                entry_price + multiplier * risk * ratio
            }),
        };

        (stop_price, target_price)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_kernel() -> EngineKernel {
        let config = BacktestConfig::default();
        let fee_model = config.fee_model();
        EngineKernel::new(
            config,
            fee_model,
            SlippageModel::None,
            FillPrice::Close,
            "TEST".to_string(),
            Direction::Long,
            None,
        )
    }

    fn bar(idx: i64, price: Price) -> KernelBar {
        KernelBar {
            timestamp: idx,
            open: price,
            high: price + 1.0,
            low: price - 1.0,
            close: price,
            volume: 1000.0,
        }
    }

    fn enter(kernel: &mut EngineKernel, idx: usize, price: Price) {
        let events = kernel.step(
            idx,
            &bar(idx as i64, price),
            StepInput { entry: true, ..StepInput::default() },
        );
        assert!(
            matches!(events.as_slice(), [EngineEvent::Entered { .. }]),
            "expected entry, got {events:?}"
        );
    }

    #[test]
    fn set_stop_price_is_noop_when_flat() {
        let mut kernel = make_kernel();
        kernel.set_stop_price(Some(90.0));
        assert!(kernel.position_snapshot().is_none());
    }

    #[test]
    fn external_open_count_overrides_the_ledger() {
        // A flat kernel would pass a max_positions=1 gate on its own ledger;
        // a portfolio that already holds a position elsewhere says otherwise.
        let mut kernel = make_kernel().with_risk_gate(RiskGate::new(Some(1), None));
        assert_eq!(kernel.open_count(), 0);

        kernel.set_external_open_count(Some(1));
        let events = kernel.step(
            0,
            &bar(0, 100.0),
            StepInput { entry: true, ..StepInput::default() },
        );
        assert!(
            matches!(
                events.as_slice(),
                [EngineEvent::EntryRejected { reason: RejectReason::MaxPositions, .. }]
            ),
            "expected a portfolio-wide rejection, got {events:?}"
        );
        assert!(!kernel.is_in_position());

        // Clearing it restores ledger-derived counting: the slot is free.
        kernel.set_external_open_count(None);
        enter(&mut kernel, 1, 100.0);
        assert!(kernel.is_in_position());
    }

    #[test]
    fn set_stop_and_target_update_open_position() {
        let mut kernel = make_kernel();
        enter(&mut kernel, 0, 100.0);

        kernel.set_stop_price(Some(95.0));
        kernel.set_target_price(Some(110.0));

        let snap = kernel.position_snapshot().unwrap();
        assert_eq!(snap.stop_price, Some(95.0));
        assert_eq!(snap.target_price, Some(110.0));

        kernel.set_stop_price(None);
        assert_eq!(kernel.position_snapshot().unwrap().stop_price, None);
    }

    #[test]
    fn programmatic_stop_triggers_exit() {
        let mut kernel = make_kernel();
        enter(&mut kernel, 0, 100.0);
        kernel.set_stop_price(Some(98.5));

        // Bar trades down through the stop.
        let events = kernel.step(1, &bar(1, 98.0), StepInput::default());
        match events.as_slice() {
            [EngineEvent::Exited { trade, .. }] => {
                assert_eq!(trade.exit_reason, ExitReason::StopLoss);
            }
            other => panic!("expected stop exit, got {other:?}"),
        }
        assert!(!kernel.is_in_position());
    }

    #[test]
    fn entry_stop_override_beats_config() {
        let config = BacktestConfig {
            stop: StopConfig::Fixed { percent: 0.05 },
            target: TargetConfig::Fixed { percent: 0.10 },
            ..BacktestConfig::default()
        };
        let fee_model = config.fee_model();
        let mut kernel = EngineKernel::new(
            config,
            fee_model,
            SlippageModel::None,
            FillPrice::Close,
            "TEST".to_string(),
            Direction::Long,
            None,
        );

        let events = kernel.step(
            0,
            &bar(0, 100.0),
            StepInput {
                entry: true,
                stop_price_override: Some(97.0),
                target_price_override: Some(104.0),
                ..StepInput::default()
            },
        );
        assert!(matches!(events.as_slice(), [EngineEvent::Entered { .. }]));

        let snap = kernel.position_snapshot().unwrap();
        assert_eq!(snap.stop_price, Some(97.0));
        assert_eq!(snap.target_price, Some(104.0));
    }

    #[test]
    fn zero_size_entry_emits_rejection() {
        let config = BacktestConfig::default();
        let fee_model = config.fee_model();
        // Lot of 10,000 units at price 100 with 100k capital -> raw size
        // ~999 units floors to zero lots.
        let inst = InstrumentConfig {
            lot_size: Some(10_000.0),
            alloted_capital: None,
            stop: None,
            target: None,
            existing_qty: None,
            avg_price: None,
        };
        let mut kernel = EngineKernel::new(
            config,
            fee_model,
            SlippageModel::None,
            FillPrice::Close,
            "TEST".to_string(),
            Direction::Long,
            Some(&inst),
        );

        let events =
            kernel.step(0, &bar(0, 100.0), StepInput { entry: true, ..StepInput::default() });
        match events.as_slice() {
            [EngineEvent::EntryRejected { reason, .. }] => {
                assert_eq!(reason.as_str(), "zero_size");
            }
            other => panic!("expected zero-size rejection, got {other:?}"),
        }
        assert!(!kernel.is_in_position());
    }

    #[test]
    fn multiplier_scales_notional_and_pnl() {
        use crate::instruments::{InstrumentKind, InstrumentSpec};

        let config = BacktestConfig { fees: 0.0, ..BacktestConfig::default() };
        let fee_model = config.fee_model();
        let spec = InstrumentSpec {
            multiplier: 50.0,
            lot_size: 1.0,
            ..InstrumentSpec::new("FUT", InstrumentKind::Contract { underlying: None })
        };
        let mut kernel = EngineKernel::new(
            config,
            fee_model,
            SlippageModel::None,
            FillPrice::Close,
            "FUT".to_string(),
            Direction::Long,
            None,
        )
        .with_instrument(spec);

        // 100k capital at price 100 with multiplier 50: 20 contracts.
        enter(&mut kernel, 0, 100.0);
        let snap = kernel.position_snapshot().unwrap();
        assert!((snap.size - 20.0).abs() < 1e-9, "size {}", snap.size);
        assert!(kernel.cash().abs() < 1e-6, "cash {}", kernel.cash());

        // Price to 102: equity = 102 * 20 * 50 = 102_000.
        assert!((kernel.equity(102.0) - 102_000.0).abs() < 1e-6);

        // Exit at 102: pnl = 2 * 20 * 50 = 2_000.
        let events = kernel.step(1, &bar(1, 102.0), StepInput { exit: true, ..StepInput::default() });
        match events.as_slice() {
            [EngineEvent::Exited { trade, .. }] => {
                assert!((trade.pnl - 2_000.0).abs() < 1e-6, "pnl {}", trade.pnl);
            }
            other => panic!("expected exit, got {other:?}"),
        }
        assert!((kernel.cash() - 102_000.0).abs() < 1e-6);
    }

    fn zero_fee_kernel() -> EngineKernel {
        let config = BacktestConfig { fees: 0.0, ..BacktestConfig::default() };
        let fee_model = config.fee_model();
        EngineKernel::new(
            config,
            fee_model,
            SlippageModel::None,
            FillPrice::Close,
            "TEST".to_string(),
            Direction::Long,
            None,
        )
    }

    #[test]
    fn resting_limit_buy_fills_next_bar_and_opens() {
        let mut kernel = zero_fee_kernel();
        let id = kernel.submit_order(
            OrderSide::Buy,
            QtySpec::Units(10.0),
            OrderKind::Limit { price: 99.0 },
            TimeInForce::Gtc,
            0,
            0,
            "ord-1".into(),
            Some(95.0),
            None,
        );

        // Bar 0 (submission bar): only the acknowledgment, no fill.
        let events = kernel.step(0, &bar(0, 100.0), StepInput::default());
        assert!(matches!(
            events.as_slice(),
            [EngineEvent::OrderAccepted { order_id, .. }] if *order_id == id
        ));
        assert!(!kernel.is_in_position());

        // Bar 1 trades down through the limit: fill at 99, position opens
        // with the attached protective stop.
        let events = kernel.step(1, &bar(1, 99.5), StepInput::default());
        match events.as_slice() {
            [EngineEvent::OrderFilled { order_id, price, size, .. }, EngineEvent::Entered { .. }] => {
                assert_eq!(*order_id, id);
                assert_eq!(*price, 99.0);
                assert_eq!(*size, 10.0);
            }
            other => panic!("expected fill + entered, got {other:?}"),
        }
        let snap = kernel.position_snapshot().unwrap();
        assert_eq!(snap.stop_price, Some(95.0));
        assert_eq!(kernel.order(id).unwrap().status, OrderStatus::Filled);
    }

    #[test]
    fn market_order_fills_on_submission_bar() {
        let mut kernel = zero_fee_kernel();
        let id = kernel.submit_order(
            OrderSide::Buy,
            QtySpec::CapitalFrac(0.5),
            OrderKind::Market,
            TimeInForce::Gtc,
            3,
            3,
            "mkt-1".into(),
            None,
            None,
        );

        let events = kernel.step(3, &bar(3, 100.0), StepInput::default());
        match events.as_slice() {
            [EngineEvent::OrderAccepted { .. }, EngineEvent::OrderFilled { order_id, price, .. }, EngineEvent::Entered { .. }] =>
            {
                assert_eq!(*order_id, id);
                // FillPrice::Close on the submission bar — same contract as
                // the signal-entry path.
                assert_eq!(*price, 100.0);
            }
            other => panic!("expected accept + fill + entered, got {other:?}"),
        }
        // Half the capital: 50k / 100 = 500 units.
        assert!((kernel.position_snapshot().unwrap().size - 500.0).abs() < 1e-9);
    }

    #[test]
    fn sell_limit_closes_position_as_order_exit() {
        let mut kernel = zero_fee_kernel();
        enter(&mut kernel, 0, 100.0);

        kernel.submit_order(
            OrderSide::Sell,
            QtySpec::FullPosition,
            OrderKind::Limit { price: 105.0 },
            TimeInForce::Gtc,
            0,
            0,
            "tp-1".into(),
            None,
            None,
        );

        // Bar 1 stays below the limit.
        let events = kernel.step(1, &bar(1, 103.0), StepInput::default());
        assert!(matches!(events.as_slice(), [EngineEvent::OrderAccepted { .. }]));
        assert!(kernel.is_in_position());

        // Bar 2 trades through it.
        let events = kernel.step(2, &bar(2, 105.5), StepInput::default());
        match events.as_slice() {
            [EngineEvent::OrderFilled { price, .. }, EngineEvent::Exited { trade, .. }] => {
                assert_eq!(*price, 105.0);
                assert_eq!(trade.exit_reason, ExitReason::Order);
            }
            other => panic!("expected fill + exit, got {other:?}"),
        }
        assert!(!kernel.is_in_position());
    }

    #[test]
    fn opening_order_rejected_while_in_position() {
        let mut kernel = zero_fee_kernel();
        enter(&mut kernel, 0, 100.0);

        kernel.submit_order(
            OrderSide::Buy,
            QtySpec::Units(1.0),
            OrderKind::Limit { price: 99.0 },
            TimeInForce::Gtc,
            0,
            0,
            "dup-1".into(),
            None,
            None,
        );

        let events = kernel.step(1, &bar(1, 98.5), StepInput::default());
        assert!(events.iter().any(|e| matches!(
            e,
            EngineEvent::OrderRejected { reason: "position_open", .. }
        )));
        assert_eq!(kernel.position_snapshot().unwrap().entry_idx, 0);
    }

    #[test]
    fn oversized_unit_order_rejects_for_capital() {
        let mut kernel = zero_fee_kernel();
        kernel.submit_order(
            OrderSide::Buy,
            QtySpec::Units(10_000.0), // 10k * 100 = 1M >> 100k capital
            OrderKind::Limit { price: 100.0 },
            TimeInForce::Gtc,
            0,
            0,
            "big-1".into(),
            None,
            None,
        );
        let _ = kernel.step(0, &bar(0, 100.0), StepInput::default());
        let events = kernel.step(1, &bar(1, 99.0), StepInput::default());
        assert!(events.iter().any(|e| matches!(
            e,
            EngineEvent::OrderRejected { reason: "insufficient_capital", .. }
        )));
        assert!(!kernel.is_in_position());
    }

    #[test]
    fn per_contract_fees_charge_on_contracts_not_notional() {
        use crate::instruments::{InstrumentKind, InstrumentSpec};

        // 2.5 currency units per contract per side, IB-style.
        let config = BacktestConfig { fees: 0.0, ..BacktestConfig::default() };
        let spec = InstrumentSpec {
            multiplier: 50.0,
            ..InstrumentSpec::new("FUT", InstrumentKind::Contract { underlying: None })
        };
        let mut kernel = EngineKernel::new(
            config,
            FeeModel::per_share(2.5),
            SlippageModel::None,
            FillPrice::Close,
            "FUT".to_string(),
            Direction::Long,
            None,
        )
        .with_instrument(spec);

        enter(&mut kernel, 0, 100.0);
        let size = kernel.position_snapshot().unwrap().size;

        let events =
            kernel.step(1, &bar(1, 100.0), StepInput { exit: true, ..StepInput::default() });
        match events.as_slice() {
            [EngineEvent::Exited { trade, .. }] => {
                // Round trip: 2.5 per contract per side, NOT 2.5 * 50.
                let expected = 2.0 * 2.5 * size;
                assert!(
                    (trade.fees - expected).abs() < 1e-9,
                    "fees {} != {expected} (size {size})",
                    trade.fees
                );
            }
            other => panic!("expected exit, got {other:?}"),
        }
    }

    #[test]
    fn percentage_fees_charge_on_notional() {
        use crate::instruments::{InstrumentKind, InstrumentSpec};

        let config = BacktestConfig { fees: 0.001, ..BacktestConfig::default() };
        let fee_model = config.fee_model();
        let spec = InstrumentSpec {
            multiplier: 50.0,
            ..InstrumentSpec::new("FUT", InstrumentKind::Contract { underlying: None })
        };
        let mut kernel = EngineKernel::new(
            config,
            fee_model,
            SlippageModel::None,
            FillPrice::Close,
            "FUT".to_string(),
            Direction::Long,
            None,
        )
        .with_instrument(spec);

        enter(&mut kernel, 0, 100.0);
        let size = kernel.position_snapshot().unwrap().size;

        let events =
            kernel.step(1, &bar(1, 100.0), StepInput { exit: true, ..StepInput::default() });
        match events.as_slice() {
            [EngineEvent::Exited { trade, .. }] => {
                // 0.1% of true notional (price * size * multiplier), each side.
                let expected = 2.0 * 0.001 * 100.0 * size * 50.0;
                assert!(
                    (trade.fees - expected).abs() < 1e-6,
                    "fees {} != {expected} (size {size})",
                    trade.fees
                );
            }
            other => panic!("expected exit, got {other:?}"),
        }
    }

    #[test]
    fn expiry_settles_position_and_rejects_entries() {
        use crate::instruments::{InstrumentKind, InstrumentSpec};

        let config = BacktestConfig { fees: 0.0, ..BacktestConfig::default() };
        let fee_model = config.fee_model();
        let spec = InstrumentSpec {
            expiration_ns: Some(5),
            ..InstrumentSpec::new("FUT", InstrumentKind::Contract { underlying: None })
        };
        let mut kernel = EngineKernel::new(
            config,
            fee_model,
            SlippageModel::None,
            FillPrice::Close,
            "FUT".to_string(),
            Direction::Long,
            None,
        )
        .with_instrument(spec);

        enter(&mut kernel, 0, 100.0);

        // Bar at the expiry timestamp: settle at close, no signal needed.
        let events = kernel.step(5, &bar(5, 103.0), StepInput::default());
        match events.as_slice() {
            [EngineEvent::Exited { trade, .. }] => {
                assert_eq!(trade.exit_reason, ExitReason::Settlement);
                assert!((trade.exit_price - 103.0).abs() < 1e-9);
            }
            other => panic!("expected settlement, got {other:?}"),
        }
        assert!(!kernel.is_in_position());

        // Post-expiry entry is refused.
        let events =
            kernel.step(6, &bar(6, 103.0), StepInput { entry: true, ..StepInput::default() });
        match events.as_slice() {
            [EngineEvent::EntryRejected { reason, .. }] => {
                assert_eq!(reason.as_str(), "expired");
            }
            other => panic!("expected expired rejection, got {other:?}"),
        }
    }

    #[test]
    fn pre_activation_entry_is_rejected() {
        use crate::instruments::{InstrumentKind, InstrumentSpec};

        let config = BacktestConfig::default();
        let fee_model = config.fee_model();
        let spec = InstrumentSpec {
            activation_ns: Some(10),
            ..InstrumentSpec::new("FUT", InstrumentKind::Contract { underlying: None })
        };
        let mut kernel = EngineKernel::new(
            config,
            fee_model,
            SlippageModel::None,
            FillPrice::Close,
            "FUT".to_string(),
            Direction::Long,
            None,
        )
        .with_instrument(spec);

        let events =
            kernel.step(0, &bar(5, 100.0), StepInput { entry: true, ..StepInput::default() });
        match events.as_slice() {
            [EngineEvent::EntryRejected { reason, .. }] => {
                assert_eq!(reason.as_str(), "inactive");
            }
            other => panic!("expected inactive rejection, got {other:?}"),
        }

        let events =
            kernel.step(1, &bar(10, 100.0), StepInput { entry: true, ..StepInput::default() });
        assert!(matches!(events.as_slice(), [EngineEvent::Entered { .. }]));
    }

    #[test]
    fn config_stop_quantizes_to_tick_grid() {
        use crate::instruments::{InstrumentKind, InstrumentSpec};

        let config = BacktestConfig {
            stop: StopConfig::Fixed { percent: 0.033 },
            ..BacktestConfig::default()
        };
        let fee_model = config.fee_model();
        let spec = InstrumentSpec {
            price_increment: 0.05,
            ..InstrumentSpec::new("EQ", InstrumentKind::Cash)
        };
        let mut kernel = EngineKernel::new(
            config,
            fee_model,
            SlippageModel::None,
            FillPrice::Close,
            "EQ".to_string(),
            Direction::Long,
            None,
        )
        .with_instrument(spec);

        enter(&mut kernel, 0, 100.0);
        // Raw stop = 96.7; on the 0.05 grid floored for a long -> 96.70
        // exactly (already on grid); use a messier percent to prove rounding:
        let stop = kernel.position_snapshot().unwrap().stop_price.unwrap();
        assert!((stop / 0.05 - (stop / 0.05).round()).abs() < 1e-9, "stop {stop} not on grid");
    }

    #[test]
    fn spec_lot_size_defers_to_instrument_config() {
        use crate::instruments::{InstrumentKind, InstrumentSpec};

        let config = BacktestConfig { fees: 0.0, ..BacktestConfig::default() };
        let fee_model = config.fee_model();
        let inst = InstrumentConfig { lot_size: Some(25.0), ..InstrumentConfig::default() };
        let spec = InstrumentSpec {
            lot_size: 50.0,
            ..InstrumentSpec::new("FUT", InstrumentKind::Contract { underlying: None })
        };
        let mut kernel = EngineKernel::new(
            config,
            fee_model,
            SlippageModel::None,
            FillPrice::Close,
            "FUT".to_string(),
            Direction::Long,
            Some(&inst),
        )
        .with_instrument(spec);

        // 100k at price 100 -> 1000 raw; explicit config lot 25 wins over
        // the spec's 50, and 1000 is already a multiple of 25.
        enter(&mut kernel, 0, 100.0);
        let size = kernel.position_snapshot().unwrap().size;
        assert!((size - 1000.0).abs() < 1e-9, "size {size}");
    }

    #[test]
    fn entry_without_override_uses_config_stop() {
        let config = BacktestConfig {
            stop: StopConfig::Fixed { percent: 0.05 },
            ..BacktestConfig::default()
        };
        let fee_model = config.fee_model();
        let mut kernel = EngineKernel::new(
            config,
            fee_model,
            SlippageModel::None,
            FillPrice::Close,
            "TEST".to_string(),
            Direction::Long,
            None,
        );

        let events = kernel.step(0, &bar(0, 100.0), StepInput { entry: true, ..StepInput::default() });
        assert!(matches!(events.as_slice(), [EngineEvent::Entered { .. }]));

        let snap = kernel.position_snapshot().unwrap();
        assert_eq!(snap.stop_price, Some(95.0));
        assert_eq!(snap.direction, Direction::Long);
        assert_eq!(snap.entry_idx, 0);
    }
}
