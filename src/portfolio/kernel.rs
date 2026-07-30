//! Steppable simulation kernel.
//!
//! Holds the per-bar simulation state that [`PortfolioEngine`] previously kept
//! as loop locals. Batch backtests drive this by looping [`EngineKernel::step`]
//! over historical bars; a future live engine drives the same code with bars
//! arriving in real time, which is the point of the extraction — one set of
//! execution semantics rather than a separate live reimplementation.
//!
//! [`PortfolioEngine`]: crate::portfolio::engine::PortfolioEngine

use crate::accounts::{AccountMode, MarginBook};
use crate::core::types::{
    BacktestConfig, Direction, ExitReason, InstrumentConfig, OhlcvBar, Price, StopConfig,
    TargetConfig, Trade,
};
use crate::data::{DepthTick, OrderBook, QuoteTick, TradeTick};
use crate::execution::algos::AlgoEngine;
use crate::execution::fill::FillRng;
use crate::execution::orders::{
    MatchOutcome, Order, OrderEngine, OrderKind, OrderSide, OrderStatus, QtySpec, TimeInForce,
};
use crate::execution::queue::QueueTracker;
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
    /// An execution schedule was registered. Slices follow as ordinary
    /// order events.
    AlgoStarted { idx: usize, algo_id: u64, client_id: String },
    /// A schedule released its last slice, or was canceled. "Completed"
    /// means fully released, not necessarily fully filled.
    AlgoCompleted { idx: usize, algo_id: u64, client_id: String },
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

/// Which market event is driving a step.
///
/// Selects the matching path only: every other phase of the step is shared.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StepMode {
    /// A completed OHLC bar.
    Bar,
    /// A single trade print, carried as a degenerate bar.
    Trade,
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
    pub(crate) config: BacktestConfig,
    pub(crate) fee_model: FeeModel,
    slippage_model: SlippageModel,
    fill_price: FillPrice,
    /// Limit/stop fill semantics, including gap-through handling.
    pub(crate) fill_model: FillModel,

    /// Open positions. Net policy holds at most one, reproducing the
    /// original single-position behavior; Independent allows hedging.
    pub(crate) ledger: PositionLedger,
    cash: f64,
    /// Default direction for signal-path entries (`enter()` and the signal
    /// arrays). The order path does NOT consult this: an order's own side
    /// decides the direction it opens in, so one kernel can hold a long and
    /// later a short.
    pub(crate) direction: Direction,
    /// Most recent bar's ATR, carried from the step input so an order-path
    /// open can honor an ATR stop/target config. Signal entries read
    /// `input.atr` directly.
    pub(crate) last_atr: f64,
    /// Position ids the strategy asked to close, applied on the next step.
    pub(crate) pending_closes: Vec<u64>,
    /// Cash (default, historical) vs leveraged margin funding.
    account: AccountMode,
    /// Per-position locked margin, used only in margin mode.
    pub(crate) margin: MarginBook,
    /// Seeded stream for stochastic fills (prob < 1.0 configs only).
    pub(crate) fill_rng: FillRng,

    /// Pre-trade constraints, checked before an entry opens.
    pub(crate) risk: RiskGate,
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
    pub(crate) spec: Option<InstrumentSpec>,

    /// Resting-order book for the class-based order API. Empty (and
    /// costless) for the signal-array path.
    pub(crate) orders: OrderEngine,
    /// Events produced between steps (order accepted/canceled), delivered
    /// at the front of the next step's event list.
    pub(crate) pending_events: Vec<EngineEvent>,
    /// Latest observed book, from quotes (L1) or depth (L2). Fills still
    /// price off trade prints; the book informs queue position.
    pub(crate) book: OrderBook,
    /// Per-order queue estimates, used only when `queue_fill_model` is on.
    pub(crate) queue: QueueTracker,
    /// Working execution schedules (TWAP). Empty unless one is submitted.
    pub(crate) algos: AlgoEngine,
    /// Latest underlying price, for settling options to intrinsic value.
    /// `None` settles at the contract's own close.
    underlying_price: Option<Price>,
    /// Whether the current step is driven by a trade print. Only a print
    /// carries volume at a price, which is what the queue model consumes.
    pub(crate) stepping_trade: bool,
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
        let tz_offset_ns = config.session_tz_offset_ns;
        let limit_slippage = config.limit_slippage;

        Self {
            config,
            fee_model,
            slippage_model,
            fill_price,
            fill_model: FillModel { fill_price, limit_slippage, ..FillModel::default() },
            ledger: PositionLedger::new(symbol, PositionPolicy::Net),
            cash,
            direction,
            last_atr: 0.0,
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
            orders: OrderEngine::with_tz_offset(tz_offset_ns),
            pending_events: Vec::new(),
            book: OrderBook::new(),
            queue: QueueTracker::new(),
            algos: AlgoEngine::new(),
            underlying_price: None,
            stepping_trade: false,
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
                let from_spec = self.spec.as_ref().map(|s| s.margin_init).filter(|&m| m > 0.0);
                Some(from_spec.unwrap_or(1.0 / leverage.max(1.0)))
            }
        }
    }

    /// Maintenance margin rate in margin mode; half the initial rate when
    /// the instrument does not declare one, and `None` when the position is
    /// fully funded.
    fn maint_rate(&self) -> Option<f64> {
        let init = self.margin_rate()?;
        if let Some(rate) = self.spec.as_ref().map(|s| s.margin_maint).filter(|&m| m > 0.0) {
            return Some(rate);
        }
        // Fully funded (initial rate >= 1.0): the whole notional is locked, so
        // the position cannot impair the account and nothing needs maintaining.
        // Without this a leverage-1.0 book margin-calls against *gross*
        // notional — hedged legs never net out, so a market-neutral portfolio
        // trips at once — and the halt latches, blocking every later entry.
        if init >= 1.0 {
            return None;
        }
        Some(init * 0.5)
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
    pub(crate) fn multiplier(&self) -> f64 {
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
    pub(crate) fn gating_open_count(&self) -> usize {
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
        self.step_inner(idx, bar, input, StepMode::Bar)
    }

    /// Advance the simulation by one trade print.
    ///
    /// The print drives the same phase order as a bar, carried as a
    /// degenerate bar (`open == high == low == close == price`): expiry,
    /// extremes, exits, pending closes, margin maintenance, order matching,
    /// entries. Only the matching path differs — see
    /// [`OrderEngine::match_trade`].
    ///
    /// Position trailing stops ratchet off every print, so they resolve at
    /// tick rather than bar resolution. That is deliberately *not* identical
    /// to a bar run over the same data: a bar can trigger a stop against a
    /// low that preceded the high which set the watermark, and prints cannot.
    pub fn step_trade(
        &mut self,
        idx: usize,
        tick: &TradeTick,
        input: StepInput,
    ) -> Vec<EngineEvent> {
        let bar = KernelBar {
            timestamp: tick.timestamp,
            open: tick.price,
            high: tick.price,
            low: tick.price,
            close: tick.price,
            volume: tick.size,
        };
        self.step_inner(idx, &bar, input, StepMode::Trade)
    }

    /// Observe a quote, recording the book without simulating anything.
    ///
    /// A quote is not a trade: it does not move the trailing-stop watermark,
    /// mark margin, match orders, or open positions. Ratcheting a stop off a
    /// bid that never traded would manufacture exits, and filling against a
    /// quote asserts a counterparty the engine has no evidence for — the
    /// print that follows is that evidence.
    ///
    /// Returns any acknowledgments queued since the last step, so orders
    /// submitted from a quote handler surface in order.
    pub fn step_quote(&mut self, quote: &QuoteTick) -> Vec<EngineEvent> {
        self.book.apply_quote(quote.timestamp, quote.bid, quote.ask);
        std::mem::take(&mut self.pending_events)
    }

    /// Observe a depth snapshot, recording the book without simulating.
    ///
    /// Same reasoning as [`EngineKernel::step_quote`], and with more force:
    /// a depth update is a quote with four extra levels — pure intent,
    /// wholly cancellable. Touching a price with a displayed order is not a
    /// trade, so nothing fills, marks, or ratchets here. It does change
    /// *future* fills by sizing the queue a resting limit joins, which is
    /// the difference between observing state and transacting.
    pub fn step_depth(&mut self, depth: &DepthTick) -> Vec<EngineEvent> {
        self.book.apply_depth(depth);
        std::mem::take(&mut self.pending_events)
    }

    /// Set the underlying price used to settle options at expiry.
    ///
    /// An option's own bars carry the option's price, so intrinsic value
    /// needs the underlying from somewhere else — the strategy supplies it
    /// rather than the engine guessing.
    pub fn set_underlying_price(&mut self, price: Option<Price>) {
        self.underlying_price = price;
    }

    /// Best bid from the most recent quote, if any.
    #[inline]
    pub fn best_bid(&self) -> Option<Price> {
        self.book.best_bid()
    }

    /// Best ask from the most recent quote, if any.
    #[inline]
    pub fn best_ask(&self) -> Option<Price> {
        self.book.best_ask()
    }

    /// Shared body of the bar and tick step paths.
    ///
    /// The phase order is load-bearing and identical for both; `mode` only
    /// selects how resting orders are matched, since a trade print is not a
    /// bar and cannot honor bar-phase time-in-force.
    pub(crate) fn step_inner(
        &mut self,
        idx: usize,
        bar: &KernelBar,
        input: StepInput,
        mode: StepMode,
    ) -> Vec<EngineEvent> {
        self.stepping_trade = mode == StepMode::Trade;
        // Carried for order-path opens, which have no StepInput of their own.
        self.last_atr = input.atr;
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
                    if self.config.liquidate_on_margin_call {
                        events.extend(self.liquidate_all(idx, bar));
                    }
                }
            }
        }

        // Resting orders committed on earlier bars match against this bar,
        // after the position's own protective exits (stop > target > signal
        // keeps its priority) and before this bar's new signals.
        let ohlcv = bar.to_ohlcv_bar();
        let outcomes = match mode {
            StepMode::Bar => self.orders.match_bar(idx, &ohlcv, &self.fill_model),
            StepMode::Trade => self.orders.match_trade(idx, &ohlcv, &self.fill_model),
        };
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

        // Execution schedules release here, just before the market sweep
        // below, so a slice fills on the step it was released on rather
        // than trailing a step behind its schedule.
        self.release_algo_slices(idx, bar.timestamp, &mut events);

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
    pub(crate) fn close_at(
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
    /// Options settle to intrinsic value against the underlying when one has
    /// been supplied via [`EngineKernel::set_underlying_price`]; without it
    /// they settle at the contract's own close, since an option's bars carry
    /// the option's price and the engine has no second series to read.
    /// Linear contracts always settle at the close.
    ///
    /// A settlement fee is charged when the spec declares one: exercise and
    /// assignment are commonly priced differently from a trade-out.
    fn settle_expiry(&mut self, idx: usize, bar: &KernelBar) -> Vec<EngineEvent> {
        let settle_price = match &self.spec {
            Some(spec) => spec.settlement_value(bar.close, self.underlying_price),
            None => bar.close,
        };
        let fee_rate = self.spec.as_ref().map(|s| s.settlement_fee).unwrap_or(0.0);

        let ids: Vec<u64> = self.ledger.positions().iter().map(|p| p.id).collect();
        let mut events = Vec::new();
        for position_id in ids {
            let Some(managed) = self.ledger.get(position_id) else { continue };
            let entry_ts = managed.entry_timestamp;
            let entry_breakdown = managed.entry_breakdown;
            let settle_fee = if fee_rate > 0.0 {
                settle_price * managed.position.size * self.multiplier() * fee_rate
            } else {
                0.0
            };
            if let Some(trade) = self.ledger.close_position(
                position_id,
                ExitDetails {
                    idx,
                    timestamp: bar.timestamp,
                    price: settle_price,
                    entry_timestamp: entry_ts,
                    reason: ExitReason::Settlement,
                    fees: settle_fee,
                    fee_breakdown: entry_breakdown,
                },
            ) {
                self.credit_close(position_id, &trade, settle_fee, settle_price);
                events.push(EngineEvent::Exited { idx, trade });
            }
        }
        events
    }

    /// Entry path: size against available capital, round to lot, open.
    pub(crate) fn try_enter(
        &mut self,
        idx: usize,
        bar: &KernelBar,
        input: StepInput,
    ) -> Option<EngineEvent> {
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
    pub(crate) fn open_at(
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

    /// Force-close every position on a margin call.
    ///
    /// Unlike end-of-data finalization or expiry settlement, this is a real
    /// trade-out: it prices through the fill model and pays exit costs,
    /// because a broker liquidating a position actually crosses the spread.
    pub fn liquidate_all(&mut self, idx: usize, bar: &KernelBar) -> Vec<EngineEvent> {
        let ids: Vec<u64> = self.ledger.positions().iter().map(|p| p.id).collect();
        let mut events = Vec::new();
        for position_id in ids {
            let Some(managed) = self.ledger.get(position_id) else { continue };
            let direction = managed.position.direction;
            let price = self.fill_price_for(bar, direction, false);
            if let Some(event) =
                self.close_at(idx, bar, position_id, price, ExitReason::Liquidation)
            {
                events.push(event);
            }
        }
        events
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
    pub(crate) fn fill_price_for(
        &self,
        bar: &KernelBar,
        direction: Direction,
        is_entry: bool,
    ) -> Price {
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
#[path = "kernel_tests.rs"]
mod tests;
