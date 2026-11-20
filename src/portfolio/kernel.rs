//! Steppable simulation kernel.
//!
//! Holds the per-bar simulation state that [`PortfolioEngine`] previously kept
//! as loop locals. Batch backtests drive this by looping [`EngineKernel::step`]
//! over historical bars; a future live engine drives the same code with bars
//! arriving in real time, which is the point of the extraction — one set of
//! execution semantics rather than a separate live reimplementation.
//!
//! [`PortfolioEngine`]: crate::portfolio::engine::PortfolioEngine

use crate::accounts::{AccountMode, MarginBook, Money, MoneyScale};
use crate::core::types::{
    BacktestConfig, Direction, ExitReason, FillTiming, InstrumentConfig, OhlcvBar, Price,
    StopConfig, TargetConfig, Timestamp, Trade,
};
use crate::data::{DepthTick, OrderBook, QuoteTick, TradeTick};
use crate::execution::algos::AlgoEngine;
use crate::execution::fill::FillRng;
use crate::execution::orders::QtySpec;
use crate::execution::orders::{
    MatchOutcome, OrderEngine, OrderKind, OrderSide, OrderStatus, TimeInForce,
};
use crate::execution::queue::QueueTracker;
use crate::execution::{FeeModel, FillModel, FillPrice, SlippageModel};
use crate::instruments::{InstrumentKind, InstrumentSpec};
use crate::portfolio::kernel_session::{SessionGate, SessionState};
use crate::portfolio::ledger::{PositionLedger, PositionPolicy};
use crate::portfolio::option_groups::OptionLeg;
use crate::portfolio::position::ExitDetails;
use crate::portfolio::risk::{RejectReason, RiskGate, SizeLimits};

/// Follow each risk-limit refusal with a `RiskRejected` event carrying the
/// same reason. Only the five risk-limit reasons qualify, so a run without the
/// limits sees the event list unchanged.
fn with_risk_events(events: Vec<EngineEvent>) -> Vec<EngineEvent> {
    if !events.iter().any(|e| match e {
        EngineEvent::EntryRejected { reason, .. } => reason.is_risk_limit(),
        EngineEvent::OrderRejected { reason, .. } => RejectReason::is_risk_limit_str(reason),
        _ => false,
    }) {
        return events;
    }
    let mut out = Vec::with_capacity(events.len() + 1);
    for event in events {
        let extra = match &event {
            EngineEvent::EntryRejected { idx, reason } if reason.is_risk_limit() => {
                Some(EngineEvent::RiskRejected {
                    idx: *idx,
                    reason: reason.as_str(),
                    order_id: None,
                    client_id: String::new(),
                })
            }
            EngineEvent::OrderRejected { idx, order_id, client_id, reason }
                if RejectReason::is_risk_limit_str(reason) =>
            {
                Some(EngineEvent::RiskRejected {
                    idx: *idx,
                    reason,
                    order_id: Some(*order_id),
                    client_id: client_id.clone(),
                })
            }
            _ => None,
        };
        out.push(event);
        out.extend(extra);
    }
    out
}

/// How an entry is funded under the margin account.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum EntryFunding {
    /// The account's rate on notional — `contract_value × size × rate`,
    /// evaluated in that order so existing runs reproduce bit-for-bit.
    /// `modelled` marks a rate the instrument spec set (`margin_init`).
    Rate { rate: f64, modelled: bool },
    /// A sold option's SPAN-style deposit per contract, from the spec, or the
    /// per-unit requirement of the margin model the account was given.
    Deposit { per_contract: f64 },
    /// The margin model requires nothing (`no_margin`). Nothing is locked.
    /// Capital-fraction sizing still prices the lot at its full notional, so
    /// a fraction of capital never buys more than the same fraction in cash.
    Unmargined,
}

/// The option lines other instruments hold in one instrument's margin group:
/// each as its instrument, signed quantity and price, with the margin they
/// lock together and the time the model prices them at.
#[derive(Debug, Clone)]
pub(crate) struct GroupContext {
    pub lines: Vec<(InstrumentSpec, f64, f64)>,
    pub locked: f64,
    pub now: Timestamp,
}

/// An option instrument and its open positions as `(id, signed quantity,
/// entry price)`.
pub type ModelOptionLines<'a> = (&'a InstrumentSpec, Vec<(u64, f64, f64)>);

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
    pub(crate) fn to_ohlcv_bar(self) -> OhlcvBar {
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
    ///
    /// The trade is boxed to keep `EngineEvent` small: a `Trade` is by far the
    /// largest thing any variant carries, and an unboxed one made every event
    /// in every queue pay its width.
    Exited { idx: usize, trade: Box<Trade> },
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
    /// A working order's quantity was cut by the engine: a sibling of a
    /// one-cancels-other group whose partner filled part of the shared
    /// quantity may not offer more than what is left. `from` and `to` are
    /// units.
    OrderResized { idx: usize, order_id: u64, client_id: String, from: f64, to: f64 },
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
    /// An entry or order was refused by one of the risk limits
    /// (`order_size`, `max_position`, `max_exposure`, `daily_loss`,
    /// `killed`). Follows the `EntryRejected` / `OrderRejected` event that
    /// reports the same refusal, which keeps firing as before.
    RiskRejected { idx: usize, reason: &'static str, order_id: Option<u64>, client_id: String },
    /// A venue fill was booked although it went past one of the book's
    /// limits (`order_size`, `max_position`, `max_exposure`,
    /// `insufficient_capital`, `reduce_only`): the venue
    /// executed it, so the position follows the venue and this reports the
    /// breach.
    FillBreach { idx: usize, reason: &'static str, order_id: u64, client_id: String },
    /// A venue fill the order took but the book cannot book yet (a
    /// reduce-only fill that arrived before the fill that opens its
    /// position). It is kept and booked as soon as the book can; until then
    /// the book and the venue disagree about the position.
    FillDeferred {
        idx: usize,
        order_id: u64,
        exec_id: String,
        client_id: String,
        price: Price,
        size: f64,
        reason: String,
    },
    /// A risk limit stopped trading: the day's loss limit was reached
    /// (`reason` is `"daily_loss"`) or the strategy was halted (`reason` is
    /// the halt reason, `"manual"` by default).
    RiskHalted { idx: usize, reason: String },
    /// A contract expired and a position in it settled. Follows the
    /// position's [`EngineEvent::Exited`] (reason `Settlement`).
    ///
    /// `outcome` is `"settled"` (cash), or for a physically settled
    /// contract `"exercised"` (a long option in the money), `"assigned"` (a
    /// short one), `"expired"` (out of the money, nothing delivered) or
    /// `"delivered"` (a future). `price` is what the position closed at and
    /// `source` says where it came from. `deliver` is the underlying units
    /// the book takes in (positive) or gives away (negative) at
    /// `delivery_price`; `0.0` when nothing is delivered.
    Settled {
        idx: usize,
        outcome: &'static str,
        style: crate::instruments::SettlementStyle,
        source: crate::instruments::SettlementSource,
        price: Price,
        size: f64,
        direction: Direction,
        deliver: f64,
        delivery_price: Price,
    },
    /// A corporate action was applied to the instrument. `ratio` is the
    /// split or bonus ratio (`0.0` for a dividend), `amount` the dividend per
    /// unit (`0.0` otherwise), `cash` the money the account received for it
    /// (a short pays a dividend, so it is negative; for a split, the cash in
    /// lieu of a fractional unit).
    CorporateAction {
        idx: usize,
        kind: crate::instruments::CorporateKind,
        ratio: f64,
        amount: f64,
        cash: f64,
    },
    /// Something the lifecycle needed was not there, so it did not happen: a
    /// settlement price the contract's terms require, a mark to pay a
    /// fractional unit at. `what` is `"settlement"` or `"corporate_action"`;
    /// `reason` a stable identifier. The position or action is left as it
    /// was.
    LifecycleRefused { idx: usize, what: &'static str, reason: String },
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
    /// Explicit unit count for an entry opened on this bar, instead of a
    /// fraction of free capital. Lot and size-increment rounding still apply,
    /// and an entry that the account cannot fund is refused rather than
    /// resized. Takes precedence over `size_mult`.
    pub units: Option<f64>,
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
    /// A ceiling on the capital an entry opened on this step may draw, below
    /// what the account has free. How a book gives each entry its share of a
    /// pool (see `strategies::portfolio::CapitalAllocation`); `None` is all of it.
    pub budget: Option<f64>,
    /// The close an entry decided on this step is sized against under `NextBarOpen`; `None` is
    /// this bar's close. A decision made before the bar was seen (a hook that heard a fill at its
    /// open) names the previous bar's.
    pub size_ref: Option<Price>,
}

/// Signal intents carried from one bar to the next under
/// [`FillTiming::NextBarOpen`].
///
/// A decision made while observing bar i may only trade at bar i+1's open.
/// The intent is stashed when bar i's signals are processed and consumed at
/// the very top of the next bar's step — before any code that can create a
/// new intent runs — so a bar's own fill logic can never see an intent that
/// bar created. That ordering, not convention, is what rules the look-ahead
/// out.
#[derive(Debug, Clone, Default)]
struct DeferredIntent {
    /// A signal entry, with the per-bar payload it was decided with.
    entry: Option<DeferredEntry>,
    /// A signal exit for every open position.
    exit: bool,
    /// Strategy-requested closes of specific positions.
    closes: Vec<u64>,
    /// A forced flatten of every open position, with the reason it is booked under.
    flatten: Option<ExitReason>,
}

/// The decision-time payload of a deferred signal entry.
///
/// Everything here is information from the decision bar: the sizing
/// multiplier and ATR are that bar's values, and the overrides are the
/// strategy's own prices. Only the fill price comes from the fill bar.
#[derive(Debug, Clone, Copy)]
struct DeferredEntry {
    size_mult: Option<f64>,
    units: Option<f64>,
    atr: f64,
    stop_price_override: Option<Price>,
    target_price_override: Option<Price>,
    /// The signal bar's close: what a capital-fraction entry is sized at.
    size_ref: Option<Price>,
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

/// The protection a newly opened position would have had in-process.
#[derive(Debug, Clone, PartialEq)]
pub struct ProtectRequest {
    pub direction: Direction,
    pub size: f64,
    pub entry_price: Price,
    pub stop: Option<Price>,
    pub target: Option<Price>,
    /// The stop trails: its distance from the entry, in price.
    pub trail: Option<f64>,
    /// The position the protection is for (what a hedging venue's protective order names).
    pub position_id: u64,
    /// The request amends the protection of a position that already has some (an add named a
    /// level): a venue moves the legs it holds before placing any.
    pub amend: bool,
}

/// What the hedge rule says about one exit.
enum HedgeGuard {
    Go,
    Refuse(EngineEvent),
    /// Not carried out now: the engine closes the group's sold legs first (a [`HeldTrigger`]).
    Held,
}

/// An exit of a bought option the engine holds back instead of closing: the stop or target of a
/// hedge leg that fired, or a forced exit (square-off, liquidation, a time exit) of one. The book
/// closes the group's sold legs with reduce-only market orders, then this position the same way.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct HeldTrigger {
    pub position_id: u64,
    pub reason: ExitReason,
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
    /// When a bar-i decision may execute. Under `NextBarOpen`, signal
    /// entries/exits stash a [`DeferredIntent`] instead of filling.
    pub(crate) fill_timing: FillTiming,
    /// Intent stashed by the previous bar's signals, consumed at the top of
    /// the next bar step. Always `None` outside `NextBarOpen`.
    deferred: Option<DeferredIntent>,
    /// Limit/stop fill semantics, including gap-through handling.
    pub(crate) fill_model: FillModel,

    /// Open positions. Net policy holds at most one, reproducing the
    /// original single-position behavior; Independent allows hedging.
    pub(crate) ledger: PositionLedger,
    /// Uninvested cash in minor units; see `accounts::money`.
    cash: Money,
    /// `cash` as an `f64`, kept in step so a bar's equity reads it without a
    /// conversion.
    cash_view: f64,
    /// Decimals cash, fees and realised P&L are booked in.
    money_scale: MoneyScale,
    /// Default direction for signal-path entries (`enter()` and the signal
    /// arrays). The order path does NOT consult this: an order's own side
    /// decides the direction it opens in, so one kernel can hold a long and
    /// later a short.
    pub(crate) direction: Direction,
    /// Most recent bar's ATR, carried from the step input so an order-path
    /// open can honor an ATR stop/target config. Signal entries read
    /// `input.atr` directly.
    pub(crate) last_atr: f64,
    /// Close of the last bar this kernel stepped: the last price known when a
    /// decision is made between steps.
    pub(crate) last_close: Option<Price>,
    /// The last bar stepped and its ordinal, for an order that arrives with it.
    pub(crate) last_step: Option<(usize, KernelBar)>,
    /// The bar whose resting part `step_rest` met while `last_step` still reads the bar before
    /// it: what answers a fill heard there (`fill_follow_up`) meets this bar.
    pub(crate) rest_step: Option<(usize, KernelBar)>,
    /// The timestamp of the bar stepped before `last_step`.
    pub(crate) prev_bar_ts: Option<i64>,
    /// Price the next capital-fraction open is sized at, set just before an
    /// open and consumed by it (see `Order::size_ref`).
    pub(crate) sizing_ref: Option<Price>,
    /// A capital-fraction order is waiting for the close of the bar it was
    /// decided on (stamped by the next step).
    pub(crate) size_ref_pending: bool,
    /// Positions here move only on orders sent to a venue: stops, targets,
    /// execution schedules and the session square-off do not act in-process
    /// but leave requests for the runtime to send (see `kernel_venue.rs`).
    pub(crate) venue_managed: bool,
    /// What each execution booked on a venue-driven order did to the ledger,
    /// the cash and the margin: what a correction or bust takes back.
    pub(crate) venue_bookings: crate::portfolio::kernel_venue::VenueBookings,
    /// A venue fill is being booked: it is a fact, so its price is booked as
    /// reported (no slippage, no tick snap), its size is not quantised, and
    /// the book's limits and funding checks record a breach instead of
    /// refusing it. Set only for the span of `book_venue_fill`.
    pub(crate) booking_fact: bool,
    /// Limits a booked venue fill went past, drained into events by the
    /// booking that raised them.
    pub(crate) fact_breaches: Vec<&'static str>,
    /// Venue fills the order took but the book could not yet book, oldest
    /// first; see `kernel_venue.rs`.
    pub(crate) unbooked_fills: Vec<crate::portfolio::kernel_venue::UnbookedFill>,
    /// Protective levels of the positions opened since the runtime last asked.
    pub(crate) protect_outbox: Vec<ProtectRequest>,
    /// Schedule slices released since the runtime last asked.
    pub(crate) slice_outbox: Vec<crate::execution::algos::PendingSlice>,
    /// The session square-off came due: the book must be flattened by orders.
    pub(crate) squareoff_outbox: bool,
    /// Position ids the strategy asked to close, applied on the next step.
    pub(crate) pending_closes: Vec<u64>,
    /// Scratch for the ids of the open positions a step walks while it may close them, kept
    /// so a step with a position open does not allocate. Empty between uses.
    open_ids: Vec<u64>,
    /// A forced flatten (halt, margin call, daily loss) the strategy's book
    /// asked for: the exit reason, and the first bar stamp that may fill it. It is
    /// an ordinary reduce-only market order for every position, so it fills
    /// where one would under the run's fill timing.
    pub(crate) flatten_request: Option<(ExitReason, Timestamp)>,
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
    /// Order-size, position-size and exposure limits from the config.
    pub(crate) limits: SizeLimits,
    /// Gross notional held by the *other* instruments of a portfolio, when a
    /// portfolio owns this kernel and `max_gross_exposure` is set.
    pub(crate) external_exposure: Option<f64>,

    effective_stop: StopConfig,
    effective_target: TargetConfig,
    /// Per-instrument capital cap and lot rounding, if any.
    alloted_capital: Option<f64>,
    pub(crate) lot_size: Option<f64>,
    /// The option lines the account's other instruments hold in this
    /// instrument's margin group, handed over by the book before each step.
    /// `None` unless the account's model groups option lines and there are
    /// some.
    group_context: Option<GroupContext>,
    /// Whether another instrument in this instrument's margin group has a
    /// buying order working that is yet to fill, handed over by the book
    /// before each step. A sold leg refused for want of funding then waits
    /// for that hedge instead of being rejected.
    pub(crate) hedge_pending: bool,
    /// The bar timestamp of the latest order event this kernel reported that happened after the
    /// bar's open (a fill away from the open price, an expiry, a cancel). A follow-up answering
    /// such an event must not be priced at that bar's open (see [`Order::follow_up`]).
    pub(crate) off_open_at: Option<i64>,
    /// The bar whose open [`EngineKernel::step_open`] has met and [`EngineKernel::step`] has not
    /// yet finished, with what the open swept for the step to carry on from.
    opened_at: Option<i64>,
    open_swept: Vec<u64>,
    open_held_before: Vec<u64>,
    /// The bar whose resting part [`EngineKernel::step_rest`] has run ahead of the strategy's
    /// decision and [`EngineKernel::step`] has not yet finished, with the session gate it read.
    rested_at: Option<i64>,
    rest_gate: Option<SessionGate>,
    /// Exits of a hedge leg the engine holds back, for the book to carry out as orders, sold legs
    /// first (see [`EngineKernel::take_held_triggers`]).
    pub(crate) held_triggers: Vec<HeldTrigger>,
    /// The step's `StepInput::budget`, read by the entries it opens.
    pub(crate) entry_budget: Option<f64>,
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
    /// Physically settled lines the last expiry ended, for the owning book
    /// to deliver (see `kernel_lifecycle`).
    pub(crate) contract_ends: Vec<ContractEnd>,
    /// Whether a refused settlement has been reported already.
    settlement_refused: bool,
    /// Whether the owner holds this contract's settlement until its
    /// underlying has printed at the instant the expiry was noticed.
    settlement_held: bool,
    /// Whether an owner (a portfolio book) delivers physically settled
    /// lines; without one a physical settlement is refused.
    delivers: bool,
    /// Whether the current step is driven by a trade print. Only a print
    /// carries volume at a price, which is what the queue model consumes.
    pub(crate) stepping_trade: bool,
    /// Square-off bookkeeping for an instrument with a trading calendar.
    pub(crate) session_state: SessionState,
}

/// What a caller states about a position it adopts that the engine cannot
/// derive. Amounts are minor units of the account's money scale.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Adoption {
    /// Margin the account has posted for this position (margin accounts).
    pub margin_minor: Option<i128>,
    /// Cash paid for this position (cash accounts), in place of price times size.
    pub cost_minor: Option<i128>,
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

        let money_scale = MoneyScale::new(config.money_decimals)
            .unwrap_or_else(|e| panic!("invalid BacktestConfig.money_decimals: {e}"));
        let cash = money_scale.money(config.initial_capital).unwrap_or_else(|e| {
            crate::accounts::money::money_fault(format!("initial_capital: {e}"))
        });
        let mut ledger = PositionLedger::new(symbol, PositionPolicy::Net);
        ledger.set_money_scale(money_scale);
        let config_seed = config.fill_seed;
        let tz_offset_ns = config.session_tz_offset_ns;
        let order_latency_ns = config.order_latency_ns;
        let limit_slippage = config.limit_slippage;
        let fill_timing = config.resolved_fill_timing();
        let limits = config.size_limits();

        Self {
            config,
            fee_model,
            slippage_model,
            fill_price,
            fill_timing,
            deferred: None,
            fill_model: FillModel { fill_price, limit_slippage },
            ledger,
            cash_view: money_scale.to_f64(cash),
            cash,
            money_scale,
            direction,
            last_atr: 0.0,
            last_close: None,
            last_step: None,
            rest_step: None,
            prev_bar_ts: None,
            sizing_ref: None,
            size_ref_pending: false,
            venue_managed: false,
            venue_bookings: Default::default(),
            booking_fact: false,
            fact_breaches: Vec::new(),
            unbooked_fills: Vec::new(),
            protect_outbox: Vec::new(),
            slice_outbox: Vec::new(),
            squareoff_outbox: false,
            pending_closes: Vec::new(),
            open_ids: Vec::new(),
            flatten_request: None,
            account: AccountMode::Cash,
            margin: MarginBook::default(),
            fill_rng: FillRng::new(config_seed),
            risk: RiskGate::unconstrained(),
            external_open_count: None,
            limits,
            external_exposure: None,
            effective_stop,
            effective_target,
            alloted_capital: inst_config.and_then(|ic| ic.alloted_capital),
            lot_size: inst_config.and_then(|ic| ic.lot_size),
            group_context: None,
            hedge_pending: false,
            off_open_at: None,
            opened_at: None,
            open_swept: Vec::new(),
            open_held_before: Vec::new(),
            rested_at: None,
            rest_gate: None,
            held_triggers: Vec::new(),
            entry_budget: None,
            spec: None,
            orders: OrderEngine::with_tz_offset(tz_offset_ns).with_latency(order_latency_ns),
            pending_events: Vec::new(),
            book: OrderBook::new(),
            queue: QueueTracker::new(),
            algos: AlgoEngine::new(),
            underlying_price: None,
            contract_ends: Vec::new(),
            settlement_refused: false,
            settlement_held: false,
            delivers: false,
            stepping_trade: false,
            session_state: SessionState::default(),
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
        ledger.set_money_scale(self.money_scale);
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
                Some(crate::accounts::margin_model::initial_rate(self.spec.as_ref(), leverage))
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

    /// How an entry in `direction` is funded under the margin account, or
    /// `None` in cash mode.
    ///
    /// `Rate(rate, modelled)` keeps the account's arithmetic exactly as it
    /// was — `contract_value × size × rate`, evaluated in that order, so
    /// every existing margin run reproduces bit-for-bit. `modelled` says
    /// an instrument-level `margin_init` set the rate (a future), the
    /// distinction between "the lot was unaffordable" and "the lot was
    /// affordable but its margin was not" when sizing lands on zero.
    ///
    /// `Deposit(per_contract)` is a SOLD option under the spec's
    /// `span_pct`/`exposure_pct` model: `(span_pct + exposure_pct) × strike
    /// × multiplier` per contract, with the premium it collects left in the
    /// balance rather than the requirement. A BOUGHT option can lose only
    /// its premium and keeps the rate path (at leverage 1.0, the premium).
    pub(crate) fn entry_funding(
        &self,
        direction: Direction,
        price: Price,
    ) -> std::result::Result<Option<EntryFunding>, RejectReason> {
        if let Some(model) = self.chosen_margin_model() {
            return self.modelled_funding(model, direction, price).map(Some);
        }
        Ok(self.built_in_funding(direction))
    }

    /// The margin model the config chose, when the account is on margin.
    /// A cash account never reads it.
    fn chosen_margin_model(&self) -> Option<&crate::accounts::MarginModel> {
        match self.account {
            AccountMode::Margin { .. } => self.config.margin_model.as_deref(),
            AccountMode::Cash => None,
        }
    }

    /// One funding path for a chosen model: the model's requirement for one
    /// unit at `price`, locked per unit. A requirement of zero locks nothing
    /// ([`EntryFunding::Unmargined`]); one that is infinite (delivery cash
    /// refuses a short) or cannot be priced refuses the entry, with a stable
    /// reason each, never a requirement of zero.
    fn modelled_funding(
        &self,
        model: &crate::accounts::MarginModel,
        direction: Direction,
        price: Price,
    ) -> std::result::Result<EntryFunding, RejectReason> {
        use crate::accounts::LineView;
        let Some(spec) = &self.spec else { return Err(RejectReason::MarginUnpriced) };
        let sign = if direction == Direction::Long { 1.0 } else { -1.0 };
        match model.required(&LineView::new(spec, sign, price)) {
            Err(_) => Err(RejectReason::MarginUnpriced),
            Ok(per_unit) if per_unit.is_nan() || per_unit.is_infinite() => {
                Err(RejectReason::InsufficientMargin)
            }
            Ok(per_unit) if per_unit <= 0.0 => Ok(EntryFunding::Unmargined),
            Ok(per_unit) => Ok(EntryFunding::Deposit { per_contract: per_unit }),
        }
    }

    fn built_in_funding(&self, direction: Direction) -> Option<EntryFunding> {
        let rate = self.margin_rate()?;
        let Some(spec) = &self.spec else {
            return Some(EntryFunding::Rate { rate, modelled: false });
        };
        if direction == Direction::Short {
            if let Some(deposit) = spec.short_option_margin_per_contract() {
                return Some(EntryFunding::Deposit { per_contract: deposit });
            }
        }
        Some(EntryFunding::Rate { rate, modelled: spec.margin_init > 0.0 })
    }

    /// Whether a SHORT option position in this kernel is margined by the
    /// spec's SPAN-style model rather than by the account's leverage rate.
    pub(crate) fn short_option_margin_modelled(&self) -> bool {
        self.chosen_margin_model().is_none()
            && self.margin_rate().is_some()
            && self.spec.as_ref().is_some_and(|s| s.short_option_margin_per_contract().is_some())
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
        self.orders.set_session(spec.session.clone());
        // Squared off the same day when the run or the venue says so: cash
        // equity is then charged the intraday row. An instrument no fee row
        // fits stays unbound; the bindings refuse it when it is registered.
        let intraday = self.config.intraday_for(&spec);
        let _ = self.fee_model.bind_instrument(&spec, intraday);
        self.spec = Some(spec);
    }

    /// The unverified fee rows this kernel's fills were priced on.
    pub fn unverified_fee_rows(&self) -> Vec<String> {
        self.fee_model.unverified_rows()
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
        self.cash_view
    }

    /// Current uninvested cash, in booked minor units.
    #[inline]
    pub fn cash_money(&self) -> Money {
        self.cash
    }

    /// The scale cash, fees and realised P&L are booked at.
    #[inline]
    pub fn money_scale(&self) -> MoneyScale {
        self.money_scale
    }

    /// The booked cost basis and entry fees of the open positions, in minor
    /// units: what a reconciliation sets against cash and realised P&L.
    pub fn open_booked(&self) -> (Money, Money) {
        self.ledger.positions().iter().fold((Money::ZERO, Money::ZERO), |(cost, fees), p| {
            (cost + p.cost, fees + p.entry_fees_money)
        })
    }

    /// Entries refused by the risk gate.
    #[inline]
    /// Every order this run has seen, in submission order.
    pub fn order_book(&self) -> &[crate::execution::orders::Order] {
        self.orders.book()
    }

    pub fn rejected_entries(&self) -> usize {
        self.risk.rejected_entries()
    }

    /// Overwrite available cash.
    ///
    /// Used by the shared-capital portfolio runner, which owns one pool across
    /// several kernels and re-points each one at the pool before stepping it.
    #[inline]
    pub fn set_cash(&mut self, cash: f64) {
        let cash = self
            .money_scale
            .money(cash)
            .unwrap_or_else(|e| crate::accounts::money::money_fault(format!("lending cash: {e}")));
        self.set_cash_money(cash);
    }

    /// Lend `cash` to the kernel and return the booked amount it holds, so
    /// the caller can take the kernel's movement back as an exact difference.
    pub fn lend_cash(&mut self, cash: f64) -> Money {
        self.set_cash(cash);
        self.cash
    }

    /// Take back the lent cash: zero, with no conversion.
    #[inline]
    pub fn clear_cash(&mut self) {
        self.cash = Money::ZERO;
        self.cash_view = 0.0;
    }

    /// [`EngineKernel::set_cash`] in booked minor units, which loses nothing.
    #[inline]
    pub fn set_cash_money(&mut self, cash: Money) {
        self.cash = cash;
        self.cash_view = self.money_scale.to_f64(cash);
    }

    /// Move cash by a booked amount.
    #[inline]
    fn move_cash(&mut self, delta: Money) {
        self.set_cash_money(self.cash + delta);
    }

    /// Market value of open positions at the given price, or 0.0 when flat.
    #[inline]
    pub fn position_value(&self, close: Price) -> f64 {
        self.ledger.position_value(close)
    }

    /// Adopt a position the account already holds (broker-truth seeding).
    ///
    /// Holding-coverage deployments mirror shares a user already owns: the
    /// strategy must start KNOWING it holds them, at the user's real average
    /// cost, without fabricating an order, a fill, fees, or a trade record.
    /// Cash is reduced by the cost basis so equity stays
    /// `initial_cash + unrealized`, exactly like an account that bought
    /// earlier. No `Entered` event is emitted and the trade counter is
    /// untouched — nothing about the adoption may read as a trade.
    ///
    /// Cash and fully funded margin accounts only. Under leverage the margin
    /// a broker has already posted against a position it holds cannot be
    /// derived from quantity and average price, so it is refused rather than
    /// guessed — inventing a figure would misstate free capital, which gates
    /// every later entry. At an initial margin rate of 1.0 the whole notional
    /// is locked and the posted margin simply IS the cost basis, so the
    /// objection lapses and adoption locks that amount instead of debiting
    /// cash. Adoption remains long-only.
    ///
    /// Call before the first stepped event, so the adopted position is in
    /// every "before" snapshot and a position-diff signal translation never
    /// reads it as a fresh entry. The kernel holds no equity curve and so
    /// cannot check this itself: [`EventSession::adopt_position`] enforces it
    /// for every caller that reaches adoption through a session, which is
    /// every caller reachable from Python. A consumer driving this kernel
    /// directly owns the ordering — adopt after stepping and the equity curve
    /// you build will understate max drawdown, because the flat pre-adoption
    /// stretch holds the running peak below where it belongs.
    pub fn adopt_position(
        &mut self,
        timestamp: Timestamp,
        price: Price,
        size: f64,
    ) -> Result<u64, String> {
        // The long-only form: a short is a different statement, made with
        // `adopt_position_with`.
        #[allow(clippy::neg_cmp_op_on_partial_ord)]
        if !(size > 0.0) {
            return Err(format!(
                "adopt_position needs positive price and size (got size {size} @ {price})"
            ));
        }
        self.adopt_position_with(timestamp, price, size, &Adoption::default())
    }

    /// [`EngineKernel::adopt_position`] for a position that may be short and
    /// for books whose posted margin the caller states.
    ///
    /// `signed_size` is positive for a long and negative for a short. A short
    /// needs a margin account: a cash account cannot hold one.
    ///
    /// On a margin account the margin already posted against a position is a
    /// fact of the account, not a function of quantity and average price. The
    /// caller gives it as [`Adoption::margin_minor`] (the amount the account
    /// has locked for this position, in minor units of the account's money
    /// scale) and the position is adopted with exactly that locked, whatever
    /// the leverage, margin model or instrument (options included). Without
    /// it the old rule holds: only a fully funded account (or a model that
    /// demands the whole cost basis) is adopted, locking the notional, and
    /// anything else is refused rather than guessed. Cash accounts adopt a long
    /// only, debiting its cost basis: the exact amount paid in minor units
    /// when [`Adoption::cost_minor`] gives it, else price times size.
    ///
    /// Nothing else changes with the sign: no order, fill, fee, trade record
    /// or `Entered` event. The position's entry is `price`, so a short's
    /// unrealized profit is what the market has fallen since.
    pub fn adopt_position_with(
        &mut self,
        timestamp: Timestamp,
        price: Price,
        signed_size: f64,
        extras: &Adoption,
    ) -> Result<u64, String> {
        // `!(x > 0.0)` rather than `x <= 0.0`, deliberately: the negated form
        // is also false for NaN, so NaN is refused here. Clippy's
        // `neg_cmp_op_on_partial_ord` suggestion would let a NaN price through
        // and adopt a position priced at NaN, which then poisons cash, equity
        // and every drawdown figure downstream.
        #[allow(clippy::neg_cmp_op_on_partial_ord)]
        if !(price > 0.0) || !(signed_size.abs() > 0.0) || !signed_size.is_finite() {
            return Err(format!(
                "adopt_position needs positive price and a non-zero size (got size {signed_size} @ {price})"
            ));
        }
        let direction = if signed_size > 0.0 { Direction::Long } else { Direction::Short };
        let size = signed_size.abs();
        let on_margin = matches!(self.account, AccountMode::Margin { .. });
        if direction == Direction::Short && !on_margin {
            return Err("adopt_position cannot seed a short on a cash account".to_string());
        }
        if extras.margin_minor.is_some() && !on_margin {
            return Err("a posted margin applies to margin accounts only".to_string());
        }
        if extras.cost_minor.is_some() && on_margin {
            return Err(
                "a cost basis applies to cash accounts: a margin account debits nothing for a \
                 position it adopts"
                    .to_string(),
            );
        }
        if extras.margin_minor.is_some_and(|m| m < 0) || extras.cost_minor.is_some_and(|c| c < 0) {
            return Err("a posted margin or cost basis cannot be negative".to_string());
        }
        // A leveraged book is still refused without the posted margin: the
        // margin a broker has already posted against a position it holds
        // cannot be derived from quantity and average price, and inventing a
        // figure would misstate free capital, the number that gates every
        // later entry. Fully funded (rate >= 1.0) is the one case where it IS
        // derivable, because the whole notional is locked and the posted
        // margin is the cost basis.
        let mut margin_rate = self.margin_rate();
        if extras.margin_minor.is_none() {
            if let Some(model) = self.chosen_margin_model() {
                // A chosen model must demand the whole cost basis, for the
                // same reason: a smaller posted margin is not derivable.
                use crate::accounts::LineView;
                let required = self
                    .spec
                    .as_ref()
                    .ok_or_else(|| {
                        "adopt_position needs an instrument for the margin model".to_string()
                    })
                    .and_then(|spec| {
                        model
                            .required(&LineView::new(
                                spec,
                                if direction == Direction::Long { size } else { -size },
                                price,
                            ))
                            .map_err(|e| e.to_string())
                    })?;
                if required < price * size * self.multiplier() {
                    return Err(format!(
                        "adopt_position requires a cash or fully funded account, or the posted \
                         margin (the margin model requires {required} against a cost basis of \
                         {}); posted margin on a broker-held position is not derivable under \
                         leverage",
                        price * size * self.multiplier()
                    ));
                }
                margin_rate = Some(1.0);
            }
            if let Some(rate) = margin_rate {
                if rate < 1.0 {
                    return Err(format!(
                        "adopt_position requires a cash or fully funded account, or the posted \
                         margin (initial margin rate {rate} < 1.0); posted margin on a \
                         broker-held position is not derivable under leverage"
                    ));
                }
            }
        }
        let cost = match extras.cost_minor {
            Some(minor) => Money::from_minor(minor),
            None => self
                .ledger
                .try_notional(price, size)
                .map_err(|e| format!("adopt_position cannot book size {size} @ {price}: {e}"))?,
        };
        let id = self
            .ledger
            .open_position(0, timestamp, price, size, direction, None, None, 0.0, None)
            .ok_or_else(|| "ledger refused adoption: a position is already open".to_string())?;
        // Fund it the way the mode funds an ordinary open (see `open_at`):
        // cash mode debits the balance, margin mode locks the margin and
        // leaves the balance alone. Getting this wrong is not cosmetic:
        // margin equity is `balance + unrealized`, with no position-value
        // term, so a cash-style debit here would never be offset and would
        // understate equity by the cost basis for the whole run. No fee term
        // in either arm: an adoption is not a trade and charges nothing.
        match (extras.margin_minor, margin_rate) {
            (Some(minor), _) => {
                let posted = self.ledger.money_scale().to_f64(Money::from_minor(minor));
                self.margin.lock(id, posted);
            }
            (None, None) => self.move_cash(-cost),
            (None, Some(rate)) => self.margin.lock(id, price * size * self.multiplier() * rate),
        }
        Ok(id)
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

    /// Whether a signal entry decided on an earlier bar is waiting to fill
    /// at the next step's open (NextBarOpen).
    #[inline]
    pub fn has_pending_entry(&self) -> bool {
        self.deferred.as_ref().is_some_and(|d| d.entry.is_some())
    }

    /// Drop a waiting deferred entry that the caller's shared gates refused,
    /// so it cannot fill on some later bar.
    pub fn clear_pending_entry(&mut self) {
        if let Some(d) = self.deferred.as_mut() {
            d.entry = None;
        }
    }

    /// Overwrite the earliest open position's stop price; no-op when flat.
    ///
    /// `None` removes the stop. The new price is checked on the next
    /// [`EngineKernel::step`] call. For a specific position under the
    /// Independent policy, use [`EngineKernel::set_stop_price_for`].
    pub fn set_stop_price(&mut self, price: Option<Price>) {
        if let Some(managed) = self.ledger.first_mut() {
            managed.position.stop_price = price;
            managed.stop_explicit = price.is_some();
        }
    }

    /// Overwrite the earliest open position's target price; no-op when flat.
    pub fn set_target_price(&mut self, price: Option<Price>) {
        if let Some(managed) = self.ledger.first_mut() {
            managed.position.target_price = price;
            managed.target_fresh = price.is_some();
            managed.target_explicit = price.is_some();
        }
    }

    /// Overwrite a specific position's stop price; `false` for unknown ids.
    pub fn set_stop_price_for(&mut self, position_id: u64, price: Option<Price>) -> bool {
        match self.ledger.get_mut(position_id) {
            Some(managed) => {
                managed.position.stop_price = price;
                managed.stop_explicit = price.is_some();
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
                managed.target_fresh = price.is_some();
                managed.target_explicit = price.is_some();
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
    /// Cash mode marks longs at market value and shorts at collateral plus
    /// unrealized gain; margin mode marks balance plus unrealized PnL.
    #[inline]
    pub fn equity(&self, close: Price) -> f64 {
        match self.account {
            AccountMode::Cash => self.cash() + self.position_value(close),
            AccountMode::Margin { .. } => self.cash() + self.ledger.unrealized_total(close),
        }
    }

    /// Cash not locked as initial margin (margin mode); all cash otherwise.
    #[inline]
    pub fn free_capital(&self) -> f64 {
        match self.account {
            AccountMode::Cash => self.cash(),
            AccountMode::Margin { .. } => self.cash() - self.margin.total_locked(),
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

    /// Set the gross notional held by the portfolio's other instruments, for
    /// `max_gross_exposure`. `None` counts only this kernel's own positions.
    #[inline]
    pub fn set_external_exposure(&mut self, exposure: Option<f64>) {
        self.external_exposure = exposure;
    }

    /// Whether `max_gross_exposure` is set, so a portfolio knows to supply
    /// [`EngineKernel::set_external_exposure`].
    #[inline]
    pub fn tracks_exposure(&self) -> bool {
        self.limits.tracks_exposure()
    }

    /// Gross notional of this kernel's open positions marked at `mark`.
    pub fn gross_notional(&self, mark: Price) -> f64 {
        let multiplier = self.multiplier();
        self.ledger.positions().iter().map(|m| m.position.size * mark * multiplier).sum()
    }

    /// Signed net size of this kernel's open positions.
    pub(crate) fn net_size(&self) -> f64 {
        self.ledger
            .positions()
            .iter()
            .map(|m| match m.position.direction {
                Direction::Long => m.position.size,
                Direction::Short => -m.position.size,
            })
            .sum()
    }

    /// Check a fill of `size` units at `contract_value` per unit against the
    /// order-size, position-size and exposure limits. `check_order` is false
    /// for the later slices of an order already checked whole.
    pub(crate) fn check_size_limits(
        &self,
        bar: &KernelBar,
        direction: Direction,
        size: f64,
        contract_value: f64,
        check_order: bool,
    ) -> Result<(), RejectReason> {
        let external = self.external_exposure.unwrap_or(0.0);
        self.check_size_limits_against(bar, direction, size, contract_value, check_order, external)
    }

    /// [`EngineKernel::check_size_limits`] with the other instruments' gross exposure given.
    pub(crate) fn check_size_limits_against(
        &self,
        bar: &KernelBar,
        direction: Direction,
        size: f64,
        contract_value: f64,
        check_order: bool,
        external_exposure: f64,
    ) -> Result<(), RejectReason> {
        if !self.limits.is_active() {
            return Ok(());
        }
        if check_order {
            self.limits.check_order(size, size * contract_value)?;
        }
        let before = self.net_size();
        let after = match direction {
            Direction::Long => before + size,
            Direction::Short => before - size,
        };
        self.limits.check_position(before, after)?;
        if self.limits.tracks_exposure() {
            let existing = self.gross_notional(bar.close) + external_exposure;
            self.limits.check_exposure(existing, size * contract_value)?;
        }
        Ok(())
    }

    /// Check an explicit-units order as a whole, before its first slice
    /// fills: a partly filled order must not be rejected midway for being
    /// too large.
    pub(crate) fn check_whole_order(
        &self,
        direction: Direction,
        units: f64,
        price: Price,
    ) -> Result<(), RejectReason> {
        if !self.limits.is_active() {
            return Ok(());
        }
        self.limits.check_order(units, units * price * self.multiplier())?;
        let before = self.net_size();
        let after = match direction {
            Direction::Long => before + units,
            Direction::Short => before - units,
        };
        self.limits.check_position(before, after)
    }

    /// Whether orders may be placed on the instrument: not a reference such as an index.
    pub fn is_tradable(&self) -> bool {
        self.spec.as_ref().is_none_or(|s| s.kind.tradable())
    }

    /// Report an entry the host refused before the book saw it, counted like any other
    /// refused entry.
    pub fn refuse_entry(&mut self, idx: usize, reason: RejectReason) {
        self.risk.record_rejection();
        self.pending_events.push(EngineEvent::EntryRejected { idx, reason });
    }

    /// Refuse a would-be fill, counting it like any other constraint refusal.
    fn refuse(&mut self, idx: usize, reason: RejectReason) -> Option<EngineEvent> {
        self.risk.record_rejection();
        Some(EngineEvent::EntryRejected { idx, reason })
    }

    /// Latch an explicit halt on this kernel: entries are refused as `Killed`.
    #[inline]
    pub fn kill(&mut self) {
        self.risk.kill();
    }

    /// The stamp of the bar stepped before the latest one: the instant the latest bar opened at,
    /// when bars follow one another.
    pub fn previous_bar_ts(&self) -> Option<i64> {
        self.prev_bar_ts
    }

    /// Whether an explicit halt has been latched.
    #[inline]
    pub fn is_killed(&self) -> bool {
        self.risk.is_killed()
    }

    /// Whether new entries are refused for the day's loss.
    #[inline]
    pub fn is_day_blocked(&self) -> bool {
        self.risk.is_day_blocked()
    }

    /// Whether a daily-loss breach should also close open positions.
    #[inline]
    pub fn flatten_on_daily_loss(&self) -> bool {
        self.risk.flatten_on_daily_loss()
    }

    /// Feed the end-of-event equity to the daily-loss limit. Returns true on
    /// the edge where the limit is first reached for the day.
    #[inline]
    pub fn observe_day(&mut self, timestamp: i64, equity: f64) -> bool {
        self.observe_day_in(self.risk_day(timestamp), equity)
    }

    /// [`EngineKernel::observe_day`] for a day the caller already decided,
    /// as a portfolio book does for all its kernels.
    #[inline]
    pub fn observe_day_in(&mut self, day: Option<i64>, equity: f64) -> bool {
        self.risk.daily_active() && self.risk.on_day_equity(day, equity)
    }

    /// The trading day of `timestamp` for the daily-loss limit: the session
    /// day when the instrument has a session schedule (`None` on a weekend
    /// or holiday, so the day does not roll), else the local calendar date
    /// from `session_tz_offset_ns`.
    pub fn risk_day(&self, timestamp: i64) -> Option<i64> {
        match self.spec.as_ref().and_then(|s| s.session.as_ref()) {
            Some(schedule) => schedule.session_day(timestamp),
            None => Some(self.config.day_of(timestamp)),
        }
    }

    /// Cancel every working order and drop any waiting signal entry, then
    /// optionally close every position at `bar`'s close. The shared tail of a
    /// halt and of a daily-loss flatten. Position cash comes back through the
    /// kernel's own cash, so a portfolio must lend and drain around the call.
    pub fn halt_cleanup(&mut self, idx: usize, cancel_orders: bool) -> Vec<EngineEvent> {
        if cancel_orders {
            // A flatten already on its way is not an order the halt cancels: a
            // strategy that halts again on the next bar must not call off its exit.
            self.deferred = self
                .deferred
                .take()
                .and_then(|d| d.flatten)
                .map(|reason| DeferredIntent { flatten: Some(reason), ..Default::default() });
            self.pending_closes.clear();
            self.cancel_all_orders(idx);
        }
        std::mem::take(&mut self.pending_events)
    }

    /// Ask for every position to be closed by an ordinary reduce-only market
    /// order decided in the event being decided:
    /// under `same_bar_close` it fills at the close of the first bar stamped at
    /// or after `first_bar_ts`, under `next_bar_open` at the open of the bar
    /// after that. The first request's reason and stamp are kept.
    pub fn request_flatten(&mut self, reason: ExitReason, first_bar_ts: Timestamp) {
        self.flatten_request.get_or_insert((reason, first_bar_ts));
    }

    /// Close every position after an event that was detected at `bar`'s close
    /// (daily loss, margin call): at that close under `same_bar_close`, and under
    /// `next_bar_open` at the next bar's open, as an order decided at that
    /// close would be. Returns the closes made now.
    pub fn flatten_after(
        &mut self,
        idx: usize,
        bar: &KernelBar,
        reason: ExitReason,
    ) -> Vec<EngineEvent> {
        if self.fill_timing == FillTiming::NextBarOpen {
            self.deferred.get_or_insert_with(DeferredIntent::default).flatten.get_or_insert(reason);
            return Vec::new();
        }
        self.liquidate_all_as(idx, bar, reason)
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
        if let Some(model) = self.chosen_margin_model() {
            // A chosen model gives an initial requirement only. Maintenance
            // is half of that requirement re-read at the mark, and nothing
            // when the requirement covers the whole notional (fully funded),
            // as for the built-in rule. A line the model cannot price keeps
            // half of what it locked at entry.
            use crate::accounts::LineView;
            let multiplier = self.multiplier();
            let grouped_options = self.model_option_lines().is_some();
            return self
                .ledger
                .positions()
                .iter()
                .map(|managed| {
                    let size = managed.position.size;
                    let notional = close * size * multiplier;
                    if grouped_options {
                        // An option leg's lock is its share of its group's
                        // requirement, which a leg priced alone would
                        // overstate (a hedged leg is not naked).
                        let locked = self.margin.locked_for(managed.id);
                        return if locked >= notional { 0.0 } else { locked * 0.5 };
                    }
                    let signed =
                        if managed.position.direction == Direction::Long { size } else { -size };
                    let required = self
                        .spec
                        .as_ref()
                        .and_then(|spec| model.required(&LineView::new(spec, signed, close)).ok())
                        .filter(|r| r.is_finite())
                        .unwrap_or_else(|| self.margin.locked_for(managed.id));
                    if required >= notional {
                        0.0
                    } else {
                        required * 0.5
                    }
                })
                .sum();
        }
        let rate = self.maint_rate();
        let modelled_shorts = self.short_option_margin_modelled();
        if rate.is_none() && !modelled_shorts {
            return 0.0;
        }
        let multiplier = self.multiplier();
        self.ledger
            .positions()
            .iter()
            .map(|managed| {
                // A sold option's SPAN-style deposit stays blocked for the
                // life of the position; the broker maintains the deposit,
                // not a fraction of the premium's notional.
                if modelled_shorts && managed.position.direction == Direction::Short {
                    return self.margin.locked_for(managed.id);
                }
                match rate {
                    Some(rate) => close * managed.position.size * multiplier * rate,
                    None => 0.0,
                }
            })
            .sum()
    }

    /// This kernel's open option positions as position-group legs, when the
    /// instrument is an option whose sold side is deposit-modelled. Empty
    /// otherwise, so a share or a plain future never joins a group.
    pub fn open_option_legs(&self, kernel_index: usize) -> Vec<OptionLeg> {
        // A chosen margin model locks each line by its own requirement; the
        // built-in grouping of sold legs does not apply.
        if self.chosen_margin_model().is_some() {
            return Vec::new();
        }
        let Some(spec) = &self.spec else { return Vec::new() };
        let InstrumentKind::Option { strike, right, .. } = &spec.kind else {
            return Vec::new();
        };
        if spec.short_option_margin_per_contract().is_none() {
            return Vec::new();
        }
        self.ledger
            .positions()
            .iter()
            .map(|managed| OptionLeg {
                position_id: managed.id,
                kernel: kernel_index,
                strike: *strike,
                right: *right,
                direction: managed.position.direction,
                size: managed.position.size,
                entry_price: managed.position.entry_price,
                multiplier: self.multiplier(),
                span_pct: spec.span_pct,
                exposure_pct: spec.exposure_pct,
            })
            .collect()
    }

    /// The margin model of the account when it margins option lines together
    /// (`defined_risk`, `spread_credit`, `from_spec`, directly or as the
    /// `OPTION` entry of a `by_type`).
    pub fn grouping_margin_model(&self) -> Option<&crate::accounts::MarginModel> {
        self.chosen_margin_model().filter(|m| m.groups_options())
    }

    /// This kernel's open option positions as lines of a chosen margin model:
    /// the instrument, and per position its id, signed quantity and entry
    /// price. `None` unless the account's model groups option lines and the
    /// instrument is an option with positions open.
    pub fn model_option_lines(&self) -> Option<ModelOptionLines<'_>> {
        self.grouping_margin_model()?;
        let spec = self.spec.as_ref()?;
        if !matches!(spec.kind, InstrumentKind::Option { .. }) || !self.ledger.is_in_position() {
            return None;
        }
        let lines = self
            .ledger
            .positions()
            .iter()
            .map(|m| {
                let sign = if m.position.direction == Direction::Long { 1.0 } else { -1.0 };
                (m.id, sign * m.position.size, m.position.entry_price)
            })
            .collect();
        Some((spec, lines))
    }

    /// [`Self::model_option_lines`] as the account stands once the market orders working here
    /// that reduce a line have filled: what another instrument's order of the same step is judged
    /// against. A hedge closed in the same decision as the sold option it covers leaves the
    /// account marginable, whichever of the two the step reaches first. A line a pending order
    /// closes whole is dropped.
    pub(crate) fn model_option_lines_after_pending(&self) -> Option<ModelOptionLines<'_>> {
        let (spec, mut lines) = self.model_option_lines()?;
        for order in self.orders.working() {
            if !matches!(order.kind, OrderKind::Market)
                || order.parent_id.is_some()
                || order.is_venue_driven()
            {
                continue;
            }
            let mut left = match order.qty {
                QtySpec::Units(units) => (units - order.filled_qty).max(0.0),
                QtySpec::FullPosition => f64::INFINITY,
                QtySpec::CapitalFrac(_) => continue,
            };
            // A buy closes sold lines, a sell closes bought ones.
            let closes = if order.side == OrderSide::Buy { -1.0 } else { 1.0 };
            for line in lines.iter_mut().filter(|l| l.1 * closes > 0.0) {
                let take = left.min(line.1.abs());
                line.1 -= closes * take;
                left -= take;
            }
        }
        lines.retain(|l| l.1.abs() > 1e-12);
        Some((spec, lines))
    }

    /// The key that says which legs hedge each other: the underlying (or
    /// the symbol itself when none is declared) and the expiry.
    pub fn option_group_key(&self) -> Option<(String, Option<Timestamp>)> {
        let spec = self.spec.as_ref()?;
        let InstrumentKind::Option { underlying, .. } = &spec.kind else { return None };
        Some((underlying.clone().unwrap_or_else(|| spec.symbol.clone()), spec.expiration_ns))
    }

    /// Total open size across this kernel's positions (0.0 when flat).
    pub fn open_size(&self) -> f64 {
        self.ledger.positions().iter().map(|p| p.position.size).sum()
    }

    /// Rewrite one open position's locked margin (the group share the
    /// session computed), returning the change for the shared account.
    pub fn set_locked_margin(&mut self, position_id: u64, amount: f64) -> f64 {
        self.margin.set_locked(position_id, amount)
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

    /// Whether the drawdown limit, rather than an explicit kill, has tripped.
    #[inline]
    pub fn drawdown_halted(&self) -> bool {
        self.risk.is_drawdown_halted()
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
        self.book.apply_quote(
            quote.timestamp,
            quote.bid,
            quote.ask,
            quote.bid_size,
            quote.ask_size,
        );
        if self.config.queue_fill_model {
            // Observation still, not a fill: a quote can only shorten what
            // a resting order believes is queued ahead of it.
            self.queue.on_quote(quote.bid, quote.bid_size, quote.ask, quote.ask_size);
        }
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
        self.entry_budget = input.budget;
        // The first event of a new trading day lifts the previous day's
        // loss block before any entry on it is judged.
        if self.risk.daily_active() {
            if let Some(day) = self.risk_day(bar.timestamp) {
                self.risk.roll_day(day);
            }
        }
        let events = self.step_events(idx, bar, input, mode);
        if mode == StepMode::Bar {
            self.opened_at = None;
            self.rested_at = None;
            self.rest_gate = None;
        }
        if self.limits.is_active() || self.risk.is_killed() || self.risk.daily_active() {
            return with_risk_events(events);
        }
        events
    }

    fn step_events(
        &mut self,
        idx: usize,
        bar: &KernelBar,
        input: StepInput,
        mode: StepMode,
    ) -> Vec<EngineEvent> {
        self.stepping_trade = mode == StepMode::Trade;
        // The open of this bar was already met (`step_open`): the fills at its open are done and
        // the step goes on from there.
        let opened = mode == StepMode::Bar && self.opened_at == Some(bar.timestamp);
        // The resting part of this bar was already met (`step_rest`), ahead of the decision.
        let rested = mode == StepMode::Bar && self.rested_at == Some(bar.timestamp);
        // A fill on this step is priced on the fee row in force on its date.
        self.fee_model.set_time(bar.timestamp);
        // Carried for order-path opens, which have no StepInput of their own.
        self.last_atr = input.atr;
        self.last_close = Some(bar.close);
        if mode == StepMode::Bar && !opened {
            self.prev_bar_ts = self.last_step.map(|(_, b)| b.timestamp);
            self.last_step = Some((idx, *bar));
        }
        // Acknowledgments queued between steps (order accepted/canceled)
        // lead the event list, preserving submission-time ordering.
        let mut events = std::mem::take(&mut self.pending_events);
        let gate = match self.rest_gate.take().filter(|_| rested) {
            Some(gate) => gate,
            None => self.session_gate(bar.timestamp, mode),
        };

        // Expiry settlement pre-empts everything: the contract no longer
        // trades at this bar, so neither exits-by-signal nor entries apply.
        // Working orders die with the contract.
        if self.spec.as_ref().is_some_and(|s| s.is_expired_at(bar.timestamp)) {
            // A deferred intent dies with the contract: it can no longer trade.
            self.deferred = None;
            events.extend(self.settle_expiry(idx, bar));
            self.cancel_all_orders(idx);
            events.append(&mut self.pending_events);
            if input.entry {
                self.risk.record_rejection();
                events.push(EngineEvent::EntryRejected { idx, reason: RejectReason::Expired });
            }
            return events;
        }

        // Intents stashed by the previous bar's signals (NextBarOpen) fill
        // FIRST, at this bar's open, before anything on this bar can create
        // a new intent — so a bar's own fill logic can never reach a signal
        // that bar generated. Exits before the entry, mirroring the phase
        // order below; the entry then participates in this bar's own
        // stop/target checks, since a position opened at the open lives
        // through the rest of the bar. With nothing stashed the call is skipped: taking `None`
        // leaves `None`, and the call alone (a large frame that reads the whole intent) was a
        // measurable part of every step.
        if mode == StepMode::Bar && !opened && self.deferred.is_some() {
            self.fill_deferred(idx, bar, gate.data_open, &mut events);
        }

        // The bar's resting part: the position's own protective exits, the square-off, the margin
        // check and the resting orders of earlier bars. Nothing in it depends on this bar's
        // decision, so a run that meets it ahead of the decision (`step_rest`) has done it.
        if !rested {
            self.rest_part(idx, bar, mode, &mut events);
        }

        // Under NextBarOpen this bar's signals only stash intents; they fill at the top of the
        // next bar step. Protective stop/target exits are NOT deferred: their triggers are
        // intra-bar prices, already causal.
        let defer_signals = self.fill_timing == FillTiming::NextBarOpen && mode == StepMode::Bar;
        if self.size_ref_pending && defer_signals {
            self.size_ref_pending = false;
            let waiting: Vec<u64> = self
                .orders
                .working()
                .filter(|o| o.size_ref.is_none() && o.submitted_idx == idx)
                .filter(|o| matches!(o.qty, QtySpec::CapitalFrac(_)))
                .map(|o| o.id)
                .collect();
            for id in waiting {
                if let Some(order) = self.orders.get_mut(id) {
                    order.size_ref = Some(bar.close);
                }
            }
        }
        if defer_signals && input.exit {
            self.deferred.get_or_insert_with(DeferredIntent::default).exit = true;
        }
        // An exit signal of this bar closes what the bar left, at the configured fill price.
        if !defer_signals && input.exit {
            let open_ids: Vec<u64> = self.ledger.positions().iter().map(|p| p.id).collect();
            for position_id in open_ids {
                let Some(managed) = self.ledger.get(position_id) else { continue };
                let price = self.fill_price_for(bar, managed.position.direction, false);
                if let Some(event) =
                    self.close_at(idx, bar, position_id, price, ExitReason::Signal, false)
                {
                    events.push(event);
                }
            }
        }

        // Strategy-requested closes of specific positions, filled like
        // signal exits at the configured fill-price model. Under NextBarOpen
        // they are decisions of this bar and fill at the next bar's open,
        // exactly like the untargeted exit.
        let mut requested: Vec<u64> = std::mem::take(&mut self.pending_closes);
        if defer_signals && !requested.is_empty() {
            self.deferred.get_or_insert_with(DeferredIntent::default).closes.append(&mut requested);
        }
        for position_id in requested {
            if self.ledger.get(position_id).is_none() {
                continue; // already closed by a stop/target this bar
            }
            let direction = match self.ledger.get(position_id) {
                Some(managed) => managed.position.direction,
                None => continue,
            };
            let price = self.fill_price_for(bar, direction, false);
            if let Some(event) =
                self.close_at(idx, bar, position_id, price, ExitReason::Signal, false)
            {
                events.push(event);
            }
        }

        // A forced flatten (halt): an ordinary reduce-only market order for every
        // position, decided in this event. It fills at this bar's close, or under
        // deferral at the next bar's open; a bar stamped before the decision
        // does not take it.
        if self.flatten_request.is_some_and(|(_, first)| bar.timestamp >= first) {
            if let Some((reason, _)) = self.flatten_request.take() {
                if defer_signals {
                    self.deferred
                        .get_or_insert_with(DeferredIntent::default)
                        .flatten
                        .get_or_insert(reason);
                } else {
                    events.extend(self.liquidate_all_as(idx, bar, reason));
                }
            }
        }

        // An entry decided while the market is closed (or on the square-off
        // event) is refused on the spot, not parked for the next open.
        let refused = if input.entry && (defer_signals || !self.ledger.is_in_position()) {
            self.session_refusal(gate.decision_open)
        } else {
            None
        };
        if let Some(reason) = refused {
            self.risk.record_rejection();
            events.push(EngineEvent::EntryRejected { idx, reason });
        } else if defer_signals && input.entry {
            // Stash the decision-time payload; the position-state and risk
            // gates run at fill time, at the top of the next bar step,
            // against the state that bar actually sees.
            self.deferred.get_or_insert_with(DeferredIntent::default).entry = Some(DeferredEntry {
                size_mult: input.size_mult,
                units: input.units,
                atr: input.atr,
                stop_price_override: input.stop_price_override,
                target_price_override: input.target_price_override,
                size_ref: input.size_ref.or(Some(bar.close)),
            });
        } else if !self.ledger.is_in_position() && input.entry {
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

        // Market orders fill last, at the configured fill-price model — the
        // same contract as signal entries. SameBarClose (and the legacy
        // look-ahead mode) sweeps the orders this bar's observation placed;
        // NextBarOpen sweeps the PREVIOUS bar's — a bar-i submission is
        // unreachable by bar i's sweep and fills at bar i+1's open, exactly
        // like a deferred signal.
        //
        // A print sweeps every market order submitted at or before its
        // ordinal. `submitted_idx` is a per-instrument EVENT ordinal, and a
        // quote consumes one without stepping the kernel: an order placed
        // from a quote handler, or from another instrument's event (whose
        // ordinal is unrelated to this one's), never lands on a print with
        // exactly its ordinal. Requiring equality left such orders working
        // forever, unfilled and unacknowledged.
        let is_plain_market = is_plain_market_order;
        if defer_signals {
            // Acknowledge this bar's new market orders now; their fill
            // arrives on the next bar step.
            let ack_ids: Vec<u64> = self
                .orders
                .working()
                .filter(is_plain_market)
                .filter(|o| o.submitted_idx == idx)
                .map(|o| o.id)
                .collect();
            for id in ack_ids {
                if let Some(order) = self.orders.get_mut(id) {
                    let _ = order.transition(OrderStatus::Accepted);
                    let client_id = order.client_id.clone();
                    events.push(EngineEvent::OrderAccepted { idx, order_id: id, client_id });
                }
            }
        }
        // A limit that is marketable at the price its decision was made at
        // trades there, like a market order placed on the same bar: it does
        // not wait for a later bar to come back to it (which may never happen).
        // Under `next_bar_open` the arrival rule already trades it at the next
        // open.
        if !defer_signals && mode == StepMode::Bar && gate.data_open {
            let marketable: Vec<(u64, Price)> = self
                .orders
                .working()
                .filter(|o| {
                    matches!(o.kind, OrderKind::Limit { .. })
                        && !o.post_only
                        && o.parent_id.is_none()
                        && o.algo_id.is_none()
                        && !o.is_venue_driven()
                        && o.submitted_idx == idx
                        && o.status == OrderStatus::Accepted
                        && !matches!(o.tif, TimeInForce::AtOpen | TimeInForce::AtClose)
                })
                .filter_map(|o| {
                    let OrderKind::Limit { price } = o.kind else { return None };
                    let (direction, is_entry) = match o.side {
                        OrderSide::Buy => (Direction::Long, true),
                        OrderSide::Sell => (Direction::Long, false),
                    };
                    let decision = self.fill_price_for(bar, direction, is_entry);
                    let through = match o.side {
                        // Strictly through: a limit at the price itself may
                        // rest behind the spread, as at a venue.
                        OrderSide::Buy => price > decision,
                        OrderSide::Sell => price < decision,
                    };
                    through.then_some((o.id, decision))
                })
                .collect();
            for (id, price) in marketable {
                self.apply_match_outcome(
                    idx,
                    bar,
                    MatchOutcome::Fill { order_id: id, price },
                    &mut events,
                );
            }
        }
        let (swept, held_before) = if opened {
            (std::mem::take(&mut self.open_swept), std::mem::take(&mut self.open_held_before))
        } else if !self.orders.has_working() {
            // Nothing is working, so the sweep would find nothing to fill and return two empty
            // lists; its call is skipped.
            (Vec::new(), Vec::new())
        } else {
            self.sweep_market(idx, bar, gate.data_open, mode, defer_signals, &mut events)
        };

        if defer_signals && mode == StepMode::Bar && !swept.is_empty() {
            self.entry_bar_protection(idx, bar, &swept, &held_before, &mut events);
        }

        events
    }

    /// Entry-bar protection (`next_bar_open`): an entry filled at this
    /// bar's open lives through the rest of the bar, so its protective
    /// levels trade against this bar too, as the working legs of a real
    /// venue would, whichever way they were attached (the position's own
    /// stop and target, a typed order's `stop_price`/`target_price`, or a
    /// bracket's child orders). Without this, a stop set on the entry bar
    /// was honoured or skipped depending on the API that set it.
    fn entry_bar_protection(
        &mut self,
        idx: usize,
        bar: &KernelBar,
        swept: &[u64],
        held_before: &[u64],
        events: &mut Vec<EngineEvent>,
    ) {
        let ohlcv = bar.to_ohlcv_bar();
        let opened: Vec<u64> = self
            .ledger
            .positions()
            .iter()
            .map(|p| p.id)
            .filter(|id| !held_before.contains(id))
            .collect();
        // The entry was at this bar's open, so the whole bar is part of the
        // position's life: its range counts toward the excursions, as it
        // does for a signal entry filled at the top of the step.
        if !opened.is_empty() {
            self.ledger.update_price(bar.high, bar.low);
        }
        for position_id in opened {
            if let Some(event) = self.try_exit_position(idx, bar, position_id, false, true) {
                events.push(event);
            }
        }
        let children: Vec<u64> = self
            .orders
            .working()
            .filter(|o| o.parent_id.is_some_and(|p| swept.contains(&p)))
            .filter(|o| o.status == OrderStatus::Accepted)
            .map(|o| o.id)
            .collect();
        if !children.is_empty() {
            let outcomes = self.orders.match_bar_among(idx, &ohlcv, &self.fill_model, &children);
            for outcome in outcomes {
                self.apply_match_outcome(idx, bar, outcome, events);
            }
        }
    }

    /// The part of a bar's step that no decision of the bar can change: the extremes, the
    /// position's own stop and target (and its trailing), the session square-off, the margin
    /// check, and the resting orders committed on earlier bars matching the bar's range. An order
    /// decided on this bar is not among them (it was submitted on this bar's ordinal) and never
    /// meets the bar's range. A venue fills all of these while the bar forms, so a strategy that
    /// decides on the bar's close must see what they left: the runtime meets this part ahead of
    /// the decision ([`EngineKernel::step_rest`]); a caller that does not gets it at the top of
    /// [`EngineKernel::step`], in the same order.
    fn rest_part(
        &mut self,
        idx: usize,
        bar: &KernelBar,
        mode: StepMode,
        events: &mut Vec<EngineEvent>,
    ) {
        // Track running extremes for trailing stops.
        self.ledger.update_price(bar.high, bar.low);

        // Protective exits, per position in opening order: stop > target. Net policy holds one
        // position, reproducing the original sequence exactly.
        let mut open_ids = std::mem::take(&mut self.open_ids);
        debug_assert!(open_ids.is_empty());
        open_ids.extend(self.ledger.positions().iter().map(|p| p.id));
        for &position_id in &open_ids {
            if let Some(event) = self.try_exit_position(idx, bar, position_id, false, false) {
                events.push(event);
            }

            // Trail only if the position survived this bar. A venue-managed kernel holds no stop
            // of its own: the venue (or the engine's watcher of a hedge) trails it.
            if let (StopConfig::Trailing { percent }, false) =
                (self.effective_stop, self.venue_managed)
            {
                if let Some(managed) = self.ledger.get_mut(position_id) {
                    managed.update_trailing_stop(percent);
                }
            }
        }
        open_ids.clear();
        self.open_ids = open_ids;

        // Session square-off: the first event at or after the cutoff flattens
        // the book, after the position's own protective exits and before
        // resting orders can open anything new.
        if self.session_state.squareoff_now {
            events.extend(self.squareoff_all(idx, bar));
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
                        events.extend(self.flatten_after(idx, bar, ExitReason::Liquidation));
                    }
                }
            }
        }

        // Resting orders committed on earlier bars match against this bar, after the position's
        // own protective exits (stop > target keeps its priority).
        let ohlcv = bar.to_ohlcv_bar();
        let outcomes = match mode {
            StepMode::Bar => self.orders.match_bar(idx, &ohlcv, &self.fill_model),
            StepMode::Trade => self.orders.match_trade(idx, &ohlcv, &self.fill_model),
        };
        for outcome in outcomes {
            self.apply_match_outcome(idx, bar, outcome, events);
        }
    }

    /// Take the range of a bar into the running extremes of the open positions. A venue-managed
    /// book is closed by the venue's report ahead of its own step, which would otherwise find the
    /// exit bar's range missing from the excursions.
    pub(crate) fn observe_range(&mut self, high: Price, low: Price) {
        self.ledger.update_price(high, low);
    }

    /// Whether there is anything for [`EngineKernel::step_rest`] to meet: a position that holds
    /// a level or can be squared off or margin-called, or an order resting for the bar's range.
    pub(crate) fn has_rest_work(&self) -> bool {
        self.ledger.is_in_position() || self.orders.working().any(|o| !is_plain_market_order(&o))
    }

    /// Meet the resting part of a bar (`rest_part`) before the strategy has decided on it, as a
    /// venue does while the bar forms. The events are returned for the strategy to hear before
    /// `on_bar`, and [`EngineKernel::step`] carries on from here with this bar's decisions.
    /// Under `next_bar_open` the open was met first ([`EngineKernel::step_open`]) and the entries
    /// it filled are protected for the rest of the bar here. Returns nothing, and the bar is left
    /// whole for `step`, when there is nothing resting or the bar ends the contract.
    pub(crate) fn step_rest(&mut self, idx: usize, bar: &KernelBar) -> Vec<EngineEvent> {
        if self.spec.as_ref().is_some_and(|s| s.is_expired_at(bar.timestamp))
            || !self.has_rest_work()
        {
            return Vec::new();
        }
        self.entry_budget = None;
        let opened = self.opened_at == Some(bar.timestamp);
        if !opened {
            if self.risk.daily_active() {
                if let Some(day) = self.risk_day(bar.timestamp) {
                    self.risk.roll_day(day);
                }
            }
            self.stepping_trade = false;
            self.fee_model.set_time(bar.timestamp);
            // `last_step` is not moved: this bar is stepped by `step`, and what asks whether a bar
            // was stepped (an order routed from another instrument's event, a follow-up, the
            // sizing reference of a capital-fraction order) must not find it so before the
            // decision.
        }
        // `last_close` stays the previous bar's: what the hooks that hear this part queue was
        // decided before the bar closed.
        self.rested_at = Some(bar.timestamp);
        self.rest_step = Some((idx, *bar));
        let mut events = std::mem::take(&mut self.pending_events);
        let gate = self.session_gate(bar.timestamp, StepMode::Bar);
        self.rest_gate = Some(gate);
        self.rest_part(idx, bar, StepMode::Bar, &mut events);
        if opened && !self.open_swept.is_empty() {
            let swept = std::mem::take(&mut self.open_swept);
            let held_before = std::mem::take(&mut self.open_held_before);
            self.entry_bar_protection(idx, bar, &swept, &held_before, &mut events);
        }
        events
    }

    /// Fill the intents the previous bar's signals stashed (`NextBarOpen`) at this bar's open:
    /// exits, then the entry, which then lives through the rest of the bar.
    fn fill_deferred(
        &mut self,
        idx: usize,
        bar: &KernelBar,
        data_open: bool,
        events: &mut Vec<EngineEvent>,
    ) {
        if let Some(intent) = self.deferred.take() {
            if intent.exit {
                let open_ids: Vec<u64> = self.ledger.positions().iter().map(|p| p.id).collect();
                for position_id in open_ids {
                    let direction = match self.ledger.get(position_id) {
                        Some(managed) => managed.position.direction,
                        None => continue,
                    };
                    let price = self.fill_price_for(bar, direction, false);
                    if let Some(event) =
                        self.close_at(idx, bar, position_id, price, ExitReason::Signal, false)
                    {
                        events.push(event);
                    }
                }
            }
            for position_id in intent.closes {
                let Some(managed) = self.ledger.get(position_id) else {
                    continue; // already closed by a stop, target or exit
                };
                let price = self.fill_price_for(bar, managed.position.direction, false);
                if let Some(event) =
                    self.close_at(idx, bar, position_id, price, ExitReason::Signal, false)
                {
                    events.push(event);
                }
            }
            if let Some(reason) = intent.flatten {
                let open_ids: Vec<u64> = self.ledger.positions().iter().map(|p| p.id).collect();
                for position_id in open_ids {
                    let Some(managed) = self.ledger.get(position_id) else { continue };
                    let price = self.fill_price_for(bar, managed.position.direction, false);
                    if let Some(event) = self.close_at(idx, bar, position_id, price, reason, false)
                    {
                        events.push(event);
                    }
                }
            }
            if let Some(entry) = intent.entry {
                // Gates run at fill time, against the state the fill bar
                // actually sees; a refusal surfaces on this bar. No early
                // return: a refused deferred entry must not skip the rest
                // of the bar.
                if !self.ledger.is_in_position() {
                    let active = self
                        .spec
                        .as_ref()
                        .and_then(|s| s.activation_ns)
                        .is_none_or(|act| bar.timestamp >= act);
                    if !active {
                        self.risk.record_rejection();
                        events.push(EngineEvent::EntryRejected {
                            idx,
                            reason: RejectReason::Inactive,
                        });
                    } else if let Some(reason) = self.session_refusal(data_open) {
                        self.risk.record_rejection();
                        events.push(EngineEvent::EntryRejected { idx, reason });
                    } else if self.margin.is_halted() {
                        self.risk.record_rejection();
                        events.push(EngineEvent::EntryRejected {
                            idx,
                            reason: RejectReason::MarginCall,
                        });
                    } else {
                        match self.risk.check_entry(self.gating_open_count()) {
                            Ok(()) => {
                                let entry_input = StepInput {
                                    entry: true,
                                    atr: entry.atr,
                                    size_mult: entry.size_mult,
                                    units: entry.units,
                                    stop_price_override: entry.stop_price_override,
                                    target_price_override: entry.target_price_override,
                                    ..StepInput::default()
                                };
                                self.sizing_ref = entry.size_ref;
                                if let Some(event) = self.try_enter(idx, bar, entry_input) {
                                    events.push(event);
                                }
                            }
                            Err(reason) => {
                                self.risk.record_rejection();
                                events.push(EngineEvent::EntryRejected { idx, reason });
                            }
                        }
                    }
                }
            }
        }
    }

    /// The market sweep of a bar: every plain market order decided on an earlier bar fills at the
    /// configured fill-price model (the next open, under `NextBarOpen`). Returns the orders it
    /// swept and the positions held before it, for the entry-bar protection that follows.
    fn sweep_market(
        &mut self,
        idx: usize,
        bar: &KernelBar,
        data_open: bool,
        mode: StepMode,
        defer_signals: bool,
        events: &mut Vec<EngineEvent>,
    ) -> (Vec<u64>, Vec<u64>) {
        let is_plain_market = is_plain_market_order;
        let market_ids: Vec<u64> = self
            .orders
            .working()
            .filter(is_plain_market)
            // Nothing sweeps while the session is closed; the order waits.
            .filter(|_| data_open)
            .filter(|o| match (defer_signals, mode) {
                // The timestamp test is what keeps an order routed here from
                // another instrument's event off a bar that closed at or before
                // its decision: that order carries this instrument's last
                // stepped ordinal, which says nothing about which side of the
                // decision the next bar falls on.
                (true, _) => {
                    (o.submitted_idx < idx && bar.timestamp > o.submitted_ts)
                        // A follow-up meets the bar of the step its fill came in.
                        || (o.follow_up && bar.timestamp >= o.submitted_ts)
                }
                // On the tick path an order in flight (entry latency) is
                // invisible to prints before it reaches the venue.
                (false, StepMode::Trade) => {
                    o.submitted_idx <= idx
                        && bar.timestamp
                            >= o.submitted_ts.saturating_add(self.config.order_latency_ns)
                }
                // `<=`: an order routed here from another instrument's event is stamped
                // with this instrument's last stepped bar and fills on its next one.
                (false, StepMode::Bar) => o.submitted_idx <= idx,
            })
            .map(|o| o.id)
            .collect();
        // Read only by the entry-bar protection, which runs only when something was swept.
        let held_before: Vec<u64> =
            if defer_signals && mode == StepMode::Bar && !market_ids.is_empty() {
                self.ledger.positions().iter().map(|p| p.id).collect()
            } else {
                Vec::new()
            };
        let swept = market_ids.clone();
        for id in market_ids {
            // Under deferral an order of this instrument was acknowledged on
            // its submission bar; one routed here from another instrument's
            // event was not, and is acknowledged now, just before it fills.
            // A partially filled market order is already acknowledged and
            // simply keeps sweeping; only a fresh submission is acked here.
            if let Some(order) =
                self.orders.get_mut(id).filter(|o| o.status == OrderStatus::Submitted)
            {
                let _ = order.transition(OrderStatus::Accepted);
                let client_id = order.client_id.clone();
                events.push(EngineEvent::OrderAccepted { idx, order_id: id, client_id });
            }
            self.apply_match_outcome(
                idx,
                bar,
                MatchOutcome::Fill { order_id: id, price: f64::NAN },
                events,
            );
        }
        (swept, held_before)
    }

    /// Whether there is anything for [`EngineKernel::step_open`] to meet: an intent the previous
    /// bar stashed or a market order waiting for its open.
    pub(crate) fn has_open_work(&self) -> bool {
        self.deferred.is_some() || self.orders.working().any(|o| is_plain_market_order(&o))
    }

    /// Meet the open of a bar before the strategy has seen the bar (`NextBarOpen`): the intents
    /// the previous bar stashed and the market orders decided on earlier bars fill at this
    /// bar's open, and what they produced is returned for the strategy to hear while it knows
    /// only what was known at that open. The rest of the bar (stops, targets, resting orders,
    /// this bar's decisions) is [`EngineKernel::step`], which carries on from here. Returns
    /// nothing, and the bar is left whole for `step`, when the fill timing is another one or
    /// the bar ends the contract.
    pub(crate) fn step_open(&mut self, idx: usize, bar: &KernelBar) -> Vec<EngineEvent> {
        if self.fill_timing != FillTiming::NextBarOpen
            || self.spec.as_ref().is_some_and(|s| s.is_expired_at(bar.timestamp))
        {
            return Vec::new();
        }
        self.entry_budget = None;
        if self.risk.daily_active() {
            if let Some(day) = self.risk_day(bar.timestamp) {
                self.risk.roll_day(day);
            }
        }
        self.stepping_trade = false;
        self.fee_model.set_time(bar.timestamp);
        self.prev_bar_ts = self.last_step.map(|(_, b)| b.timestamp);
        self.last_step = Some((idx, *bar));
        self.opened_at = Some(bar.timestamp);
        let mut events = std::mem::take(&mut self.pending_events);
        // The session gate keeps state (the square-off is due once): it is read by `step`.
        let data_open = self.schedule().is_none_or(|s| s.is_open(bar.timestamp.saturating_sub(1)));
        self.fill_deferred(idx, bar, data_open, &mut events);
        let (swept, held_before) =
            self.sweep_market(idx, bar, data_open, StepMode::Bar, true, &mut events);
        self.open_swept = swept;
        self.open_held_before = held_before;
        events
    }

    /// Exit path for one position: stop-loss, then take-profit, then signal.
    fn try_exit_position(
        &mut self,
        idx: usize,
        bar: &KernelBar,
        position_id: u64,
        exit_signal: bool,
        arrived: bool,
    ) -> Option<EngineEvent> {
        // The first bar to examine a target is where it arrives: marketable if that bar opens
        // through it. Every later bar finds it resting.
        let target_fresh = std::mem::take(&mut self.ledger.get_mut(position_id)?.target_fresh);
        let managed = self.ledger.get(position_id)?;
        let mut exit_reason: Option<ExitReason> = None;
        let mut exit_price = bar.close;

        let direction = managed.position.direction;
        let ohlcv_bar = bar.to_ohlcv_bar();
        // A target is a resting limit and meets the market as one: on the first bar that
        // examines it, and on the bar its position was entered at the open of, it is marketable
        // if the open is through it, and trades there. After that it fills at its level, except
        // across a session open that gaps past it (a print has no range to gap).
        let arrived = arrived || managed.entry_timestamp == bar.timestamp || target_fresh;
        let session_gap = arrived
            || (!self.stepping_trade && self.orders.opens_session(bar.timestamp.saturating_sub(1)));

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
                .get_limit_fill_price(target_price, &ohlcv_bar, direction, false, session_gap)
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
                .get_limit_fill_price(target_price, &ohlcv_bar, direction, false, session_gap)
                .unwrap_or(target_price);
        }

        // Exit signal.
        if exit_reason.is_none() && exit_signal {
            exit_reason = Some(ExitReason::Signal);
            exit_price = self.fill_price_for(bar, direction, false);
        }

        let reason = exit_reason?;
        if matches!(reason, ExitReason::StopLoss | ExitReason::TakeProfit)
            && self.engine_holds_levels(direction)
            && self
                .close_leaves_unmarginable(position_id, self.ledger.get(position_id)?.position.size)
        {
            // The level of a hedge leg whose close would leave a sold leg of the group without
            // its cover is the engine's, in every mode: it closes the sold legs left bare first
            // and then this one with ordinary orders, never this leg alone. Any other level is an
            // ordinary resting order and fills at its price.
            self.held_triggers.push(HeldTrigger { position_id, reason });
            self.set_stop_price_for(position_id, None);
            self.set_target_price_for(position_id, None);
            if exit_price != bar.open {
                // Touched after the open: an order answering it is not priced at that open.
                self.off_open_at = Some(bar.timestamp);
            }
            return None;
        }
        self.close_at(idx, bar, position_id, exit_price, reason, reason == ExitReason::TakeProfit)
    }

    /// Whether a position held `direction` here is a leg that may hedge a sold one: a bought
    /// option of a margin group. Its stop or target is the engine's to carry out only when
    /// closing it would leave a sold leg without its cover
    /// ([`EngineKernel::close_leaves_unmarginable`]); otherwise it is an ordinary resting order.
    pub fn engine_holds_levels(&self, direction: Direction) -> bool {
        direction == Direction::Long
            && self.option_group_key().is_some()
            && self.grouping_margin_model().is_some()
    }

    /// Round a unit count down to the lot size, then the size increment.
    pub(crate) fn quantize_units(&self, raw_size: f64) -> f64 {
        let size = match self.lot_size {
            Some(lot) if lot > 0.0 => (raw_size / lot).floor() * lot,
            _ => raw_size,
        };
        match &self.spec {
            Some(spec) => spec.quantize_size(size),
            None => size,
        }
    }

    /// Per-unit funding plus proportional fee, the divisor of capital-fraction sizing.
    fn sizing_denominator(
        funding: Option<EntryFunding>,
        contract_value: f64,
        fee_rate: f64,
    ) -> f64 {
        match funding {
            None => contract_value * (1.0 + fee_rate),
            Some(EntryFunding::Rate { rate, .. }) => contract_value * (rate + fee_rate),
            Some(EntryFunding::Deposit { per_contract }) => {
                per_contract + contract_value * fee_rate
            }
            Some(EntryFunding::Unmargined) => contract_value * (1.0 + fee_rate),
        }
    }

    /// Entry cost of `size` contracts: funding plus the active model's fee.
    fn entry_cost(
        &self,
        size: f64,
        contract_value: f64,
        direction: Direction,
        funding: Option<EntryFunding>,
    ) -> f64 {
        let funding_cost = match funding {
            None => contract_value * size,
            Some(EntryFunding::Rate { rate, .. }) => contract_value * size * rate,
            Some(EntryFunding::Deposit { per_contract }) => per_contract * size,
            Some(EntryFunding::Unmargined) => 0.0,
        };
        funding_cost + self.fee_model.calculate_side(contract_value, size, direction, true)
    }

    /// Largest lot- and step-aligned size whose entry cost (funding plus the
    /// active fee model's actual fee) fits in `budget`.
    ///
    /// For a model whose fee is not proportional to notional, start from the
    /// flat-rate size, refine with the fee rate the model charges at that
    /// size, then step down until the real cost fits. The result never
    /// exceeds `budget` after fees; it may fall a step short of the optimum.
    fn size_within_fees(
        &self,
        budget: f64,
        contract_value: f64,
        direction: Direction,
        funding: Option<EntryFunding>,
    ) -> f64 {
        let mut size = self.quantize_units(
            budget / Self::sizing_denominator(funding, contract_value, self.config.fees),
        );
        if contract_value > 0.0 {
            for _ in 0..4 {
                if size <= 0.0 {
                    break;
                }
                let fee = self.fee_model.calculate_side(contract_value, size, direction, true);
                let effective = fee / (contract_value * size);
                size = self.quantize_units(
                    budget / Self::sizing_denominator(funding, contract_value, effective),
                );
            }
        }
        for _ in 0..64 {
            if size <= 0.0 {
                return 0.0;
            }
            let cost = self.entry_cost(size, contract_value, direction, funding);
            if cost <= budget {
                return size;
            }
            let proportional = self.quantize_units(size * budget / cost);
            size = if proportional < size {
                proportional
            } else {
                let step = match (self.lot_size, &self.spec) {
                    (Some(lot), _) if lot > 0.0 => lot,
                    (_, Some(spec)) if spec.size_increment > 0.0 => spec.size_increment,
                    _ => size * 1e-12,
                };
                self.quantize_units(size - step)
            };
        }
        0.0
    }

    /// The price a fill actually executes at: slippage (skipped for
    /// limit-priced fills), then the instrument's price grid. Market-style
    /// fills snap against the trader, limit-priced ones in their favour; see
    /// [`InstrumentSpec::snap_fill`]. Without a price increment the result is
    /// the slipped price untouched.
    pub(crate) fn executed_price(
        &self,
        raw: Price,
        direction: Direction,
        is_entry: bool,
        limit_priced: bool,
        bar: &KernelBar,
    ) -> Price {
        // A venue's price is booked as it reported it.
        if self.booking_fact {
            return raw;
        }
        let price = if limit_priced {
            raw
        } else {
            self.slippage_model.apply(raw, direction, is_entry, Some(bar.volume))
        };
        match &self.spec {
            Some(spec) => {
                spec.snap_fill(price, (direction == Direction::Long) == is_entry, limit_priced)
            }
            None => price,
        }
    }

    /// A fill's charge, booked: the itemised breakdown (when the model has
    /// one) with its total set to the booked amount, and that amount's `f64`
    /// view. The parts stay as computed; the total is what moved cash, so a
    /// trade's `fee_breakdown.total()` equals its `fees` to the last digit.
    fn booked_charge(
        &self,
        breakdown: Option<crate::execution::costs::FeeBreakdown>,
        flat: impl FnOnce() -> f64,
    ) -> (Option<crate::execution::costs::FeeBreakdown>, f64) {
        let fees = self.booked_fee(breakdown.map_or_else(flat, |b| b.total()));
        self.fee_model.note_booked(breakdown.as_ref());
        (
            breakdown
                .map(|b| crate::execution::costs::FeeBreakdown { exact_total: Some(fees), ..b }),
            fees,
        )
    }

    /// A computed charge as the booked amount's `f64` view: what the account
    /// is debited, so the fee a trade reports is the fee that moved cash.
    #[inline]
    fn booked_fee(&self, raw: f64) -> f64 {
        self.money_scale.to_f64(self.ledger.book(raw))
    }

    /// Units a `budget` buys: closed form for a flat percentage or no fees,
    /// stepped by [`Self::size_within_fees`] for any other model.
    fn sized_units(
        &self,
        budget: f64,
        contract_value: f64,
        direction: Direction,
        funding: Option<EntryFunding>,
    ) -> f64 {
        match self.fee_model {
            FeeModel::Percentage(rate) => self
                .quantize_units(budget / Self::sizing_denominator(funding, contract_value, rate)),
            FeeModel::None => {
                self.quantize_units(budget / Self::sizing_denominator(funding, contract_value, 0.0))
            }
            _ => self.size_within_fees(budget, contract_value, direction, funding),
        }
    }

    /// Sizing fits the budget in `f64`, but the account is debited the
    /// notional and the fee each rounded to the booked scale, which can come
    /// to one minor unit more. When the booked debit of a cash-funded entry
    /// would exceed what is available, size again on a budget one unit
    /// tighter (up to four times) rather than overdraw; `0.0` if none fits.
    fn fit_booked_debit(
        &self,
        mut size: f64,
        budget: f64,
        available: f64,
        contract_value: f64,
        direction: Direction,
        funding: Option<EntryFunding>,
    ) -> f64 {
        let unit = 10f64.powi(-(self.money_scale.places() as i32));
        for tighten in 1..=4 {
            if size <= 0.0 {
                return size;
            }
            let fee = match self.fee_model.breakdown(contract_value, size, direction, true) {
                Some(b) => b.total(),
                None => self.fee_model.calculate(contract_value, size, direction),
            };
            let debit = self.ledger.book(contract_value * size) + self.ledger.book(fee);
            if self.money_scale.to_f64(debit) <= available {
                return size;
            }
            size = self.sized_units(
                budget - f64::from(tighten) * unit,
                contract_value,
                direction,
                funding,
            );
        }
        0.0
    }

    /// Apply a close at a determined raw price: slippage, fees, position
    /// close, cash credit. Shared by the signal path ([`Self::try_exit`])
    /// and order-driven closes, so both produce identical arithmetic.
    ///
    /// `limit_priced` marks a resting-limit or fixed-target fill: its price is
    /// already the limit (adjusted only by `limit_slippage`), so the global
    /// slippage model must not make it worse than the limit.
    pub(crate) fn close_at(
        &mut self,
        idx: usize,
        bar: &KernelBar,
        position_id: u64,
        exit_price: Price,
        reason: ExitReason,
        limit_priced: bool,
    ) -> Option<EngineEvent> {
        let managed = self.ledger.get(position_id)?;
        let direction = managed.position.direction;
        let size = managed.position.size;
        let entry_ts = managed.entry_timestamp;
        let entry_breakdown = managed.entry_breakdown;
        let squared = managed.same_day_squared;
        match self.guard_hedge_close(idx, position_id, size, reason) {
            HedgeGuard::Go => {}
            HedgeGuard::Refuse(refusal) => return Some(refusal),
            HedgeGuard::Held => return None,
        }

        let exit_price = self.executed_price(exit_price, direction, false, limit_priced, bar);

        // calculate_side, not calculate: STT lands on the sell leg and stamp
        // duty on the buy leg, so entry and exit are not symmetric.
        //
        // Fee models see the per-contract currency price (price * contract
        // multiplier) and the raw contract count: value-based schedules
        // (percentage, tiered, itemized) then charge on true notional while
        // per-contract schedules charge per contract, not per notional unit.
        let fee_price = exit_price * self.multiplier();
        let (exit_breakdown, fees) = self
            .booked_charge(self.fee_model.breakdown(fee_price, size, direction, false), || {
                self.fee_model.calculate(fee_price, size, direction)
            });

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

        let mut trade = self.ledger.close_position(
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

        let refund =
            self.restate_same_day(&mut trade, entry_breakdown, exit_breakdown, squared, None);
        self.credit_close(position_id, &trade);
        self.move_cash(refund);

        Some(EngineEvent::Exited { idx, trade: Box::new(trade) })
    }

    /// A sale that closes cash equity opened on the same trading date was an intraday trade,
    /// though the entry was charged as delivery (the exit was not known then). Reprice the
    /// trade's entry and exit on the intraday row: its fees, breakdown, booked amounts and
    /// P&L are restated, and the entry charge already debited above the restated one is
    /// returned (the amount this gives back, to be moved to cash after the close is
    /// credited). A trade the model does not classify by day is left as it was.
    ///
    /// `squared` is what earlier slices of the same entry squared that day, and `open` the
    /// position still open after this slice (a partial close). The entry is one order, charged
    /// rounded once on the sum of its parts: this slice takes that charge less what the earlier
    /// slices were booked at and what the open rest keeps (its own delivery charge), so the
    /// slices of the entry add up to the order's charge to the paisa.
    fn restate_same_day(
        &mut self,
        trade: &mut Trade,
        entry_breakdown: Option<crate::execution::costs::FeeBreakdown>,
        exit_breakdown: Option<crate::execution::costs::FeeBreakdown>,
        squared: Option<(crate::execution::costs::FeeBreakdown, Money)>,
        open: Option<u64>,
    ) -> Money {
        let multiplier = self.multiplier();
        let Some(m) = trade.money.as_ref() else { return Money::from_minor(0) };
        let scale = self.money_scale;
        let brokerage = crate::execution::fees::SameDayBrokerage {
            entry: entry_breakdown.map(|b| b.brokerage),
            entry_total: scale.to_f64(Money::from_minor(m.entry_fees)),
            exit: exit_breakdown.map_or(0.0, |b| b.brokerage),
        };
        let rest = open.and_then(|id| self.ledger.get(id)).map(|r| {
            let value = r.position.entry_price * r.position.size * multiplier;
            (value, r.entry_fees_money)
        });
        let order = crate::execution::fees::SameDayOrder {
            squared: squared.map(|(parts, _)| parts),
            open: rest.map(|(value, booked)| (value, scale.to_f64(booked))),
        };
        let Some(c) = self.fee_model.same_day_charges(
            trade.entry_price * trade.size * multiplier,
            trade.exit_price * trade.size * multiplier,
            trade.direction,
            trade.entry_time,
            trade.exit_time,
            brokerage,
            order,
        ) else {
            return Money::from_minor(0);
        };
        let Some(m) = trade.money.as_mut() else { return Money::from_minor(0) };
        let squared_booked = squared.map_or(Money::from_minor(0), |(_, booked)| booked);
        let (old_open, new_open) = match rest {
            Some((_, booked)) => (booked, self.ledger.book(c.open_entry)),
            None => (Money::from_minor(0), Money::from_minor(0)),
        };
        let new_entry = self.ledger.book(c.order_entry) - squared_booked - new_open;
        let new_exit = self.ledger.book(c.exit.total());
        let (old_entry, old_exit) =
            (Money::from_minor(m.entry_fees), Money::from_minor(m.exit_fees));
        let pnl = Money::from_minor(m.pnl) + old_entry + old_exit - new_entry - new_exit;
        (m.entry_fees, m.exit_fees, m.pnl) = (new_entry.minor(), new_exit.minor(), pnl.minor());
        trade.entry_fees = scale.to_f64(new_entry);
        trade.exit_fees = scale.to_f64(new_exit);
        trade.fees = scale.to_f64(new_entry + new_exit);
        trade.pnl = scale.to_f64(pnl);
        let cost_basis = trade.entry_price * trade.size * multiplier;
        trade.return_pct = if cost_basis > 0.0 { trade.pnl / cost_basis * 100.0 } else { 0.0 };
        let mut combined = c.entry;
        combined.add(&c.exit);
        combined.exact_total = Some(trade.fees);
        trade.fee_breakdown = Some(combined);
        if let Some(r) = open.and_then(|id| self.ledger.get_mut(id)) {
            let mut parts = c.entry_parts;
            if let Some((earlier, _)) = squared {
                parts.add(&earlier);
            }
            r.same_day_squared = Some((parts, squared_booked + new_entry));
            r.entry_fees_money = new_open;
            r.position.entry_fees = scale.to_f64(new_open);
        }
        old_entry + old_open - new_entry - new_open
    }

    /// Credit a closed position back to the account, in booked minor units.
    ///
    /// Cash mode returns the marked value (the booked entry notional plus the
    /// gain for a short, so `2 * cost - exit` there) minus exit fees. Margin
    /// mode releases the locked margin and books realised P&L: `pnl +
    /// entry_fees` is gross P&L minus exit fees, and entry fees were already
    /// debited at entry. Both are integer sums of the trade's booked
    /// amounts, so cash, cost basis and realised P&L reconcile exactly.
    fn credit_close(&mut self, position_id: u64, trade: &Trade) {
        if matches!(self.account, AccountMode::Margin { .. }) {
            self.margin.release(position_id);
        }
        self.credit_slice(trade);
    }

    /// The cash movement of a closed trade or slice (see
    /// [`Self::credit_close`]), without touching the lock.
    fn credit_slice(&mut self, trade: &Trade) {
        let m = trade.money.expect("a trade closed by the kernel carries its booked amounts");
        match self.account {
            AccountMode::Cash => {
                let cost = Money::from_minor(m.entry_notional);
                let exit = Money::from_minor(m.exit_notional);
                let marked = match trade.direction {
                    Direction::Long => exit,
                    Direction::Short => cost + cost - exit,
                };
                self.move_cash(marked - Money::from_minor(m.exit_fees));
            }
            AccountMode::Margin { .. } => {
                self.move_cash(Money::from_minor(m.pnl) + Money::from_minor(m.entry_fees));
            }
        }
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
            input.units,
            input.atr,
            input.stop_price_override,
            input.target_price_override,
            false,
        )
    }

    /// The units an entry sized by `size_mult` of the capital on hand would
    /// open at `entry_price` on `bar`, as `open_at` would size it, without
    /// opening anything. `None` when the sizing yields nothing.
    pub(crate) fn planned_entry_units(
        &self,
        bar: &KernelBar,
        direction: Direction,
        entry_price: Price,
        size_mult: Option<f64>,
        limit_priced: bool,
    ) -> Result<f64, &'static str> {
        let adjusted_price = self.executed_price(entry_price, direction, true, limit_priced, bar);
        let free = self.entry_budget.map_or(self.free_capital(), |b| b.min(self.free_capital()));
        let available = self.alloted_capital.map(|cap| cap.min(free)).unwrap_or(free);
        let contract_value = adjusted_price * self.multiplier();
        let funding = self.entry_funding(direction, adjusted_price).map_err(|r| r.as_str())?;
        let budget = size_mult.map_or(available, |mult| mult * available);
        let size = self.sized_units(budget, contract_value, direction, funding);
        let size = if funding.is_none() {
            self.fit_booked_debit(size, budget, available, contract_value, direction, funding)
        } else {
            size
        };
        if size > 0.0 {
            return Ok(size);
        }
        Err(match funding {
            Some(EntryFunding::Deposit { .. })
            | Some(EntryFunding::Rate { modelled: true, .. }) => RejectReason::InsufficientMargin,
            _ => RejectReason::ZeroSize,
        }
        .as_str())
    }

    /// Hand the kernel the option lines the account's other instruments hold
    /// in its margin group (see [`GroupContext`]), or clear them.
    pub(crate) fn set_group_context(&mut self, context: Option<GroupContext>) {
        self.group_context = context;
    }

    /// Say whether a hedge in this instrument's margin group is still to
    /// fill; see `hedge_pending`.
    pub(crate) fn set_hedge_pending(&mut self, pending: bool) {
        self.hedge_pending = pending;
    }

    /// Whether this is an option with a buying order working that would open
    /// a position: the hedge a sold leg of its margin group is waiting for.
    pub(crate) fn has_working_option_buy(&self) -> bool {
        self.option_group_key().is_some()
            && self
                .orders
                .working()
                .any(|o| !o.reduce_only && o.side == crate::execution::orders::OrderSide::Buy)
    }

    /// The one place every exit cause meets the rule on taking a hedge off a sold option. An exit
    /// that would leave the group's model unable to state a requirement (a loss without bound) is
    /// refused when the strategy asked for it (`Signal`), as a broker's risk system refuses an exit
    /// beyond the margin available; an order is judged where it fills (`kernel_orders`), and a
    /// forced exit (target, square-off, liquidation, a time exit) is not carried out here but
    /// held, for the book to close the group's sold legs and then this one with orders
    /// ([`EngineKernel::take_held_triggers`]), as a venue would be sent them. Settlement,
    /// delivery and end of data end every line together and need no judging.
    fn guard_hedge_close(
        &mut self,
        idx: usize,
        position_id: u64,
        size: f64,
        reason: ExitReason,
    ) -> HedgeGuard {
        match reason {
            ExitReason::Order
            | ExitReason::EndOfData
            | ExitReason::Settlement
            | ExitReason::Delivery
            | ExitReason::CorporateAction => HedgeGuard::Go,
            _ if !self.close_leaves_unmarginable(position_id, size) => HedgeGuard::Go,
            ExitReason::Signal => HedgeGuard::Refuse(EngineEvent::RiskRejected {
                idx,
                reason: "insufficient_margin",
                order_id: None,
                client_id: String::new(),
            }),
            _ => {
                if !self.held_triggers.iter().any(|h| h.position_id == position_id) {
                    self.held_triggers.push(HeldTrigger { position_id, reason });
                }
                HedgeGuard::Held
            }
        }
    }

    /// The exits of hedge legs held back since the last call.
    pub(crate) fn take_held_triggers(&mut self) -> Vec<HeldTrigger> {
        std::mem::take(&mut self.held_triggers)
    }

    /// Whether closing `size` of position `position_id`, a bought option, would leave the margin
    /// group's option lines with a requirement the account's model cannot state because the
    /// loss is unbounded (a sold option with no hedge left). Judged on the group context lent
    /// for the step; `false` without one.
    pub(crate) fn close_leaves_unmarginable(&self, position_id: u64, size: f64) -> bool {
        use crate::accounts::LineView;
        let (Some(context), Some(model), Some((own_spec, lines))) =
            (&self.group_context, self.grouping_margin_model(), self.model_option_lines())
        else {
            return false;
        };
        let closes_a_hedge =
            lines.iter().any(|(id, quantity, _)| *id == position_id && *quantity > 0.0);
        if !closes_a_hedge {
            return false;
        }
        let mut views: Vec<LineView<'_>> = context
            .lines
            .iter()
            .map(|(spec, quantity, price)| LineView::new(spec, *quantity, *price))
            .collect();
        for (id, quantity, price) in &lines {
            let left = if *id == position_id { quantity - size } else { *quantity };
            if left.abs() > 1e-12 {
                views.push(LineView::new(own_spec, left, *price));
            }
        }
        matches!(model.group_required(&views, context.now), Ok(required) if !required.is_finite())
    }

    /// What this entry adds to the margin its group locks, given the hedging
    /// legs already open.
    ///
    /// A sold option is funded alone at its fill by its own deposit, and a
    /// model that cannot price a naked sold leg refuses it. When the
    /// account's model margins option lines together and the legs that hedge
    /// it are open, the group needs less: the entry adds the group's
    /// requirement with it, less what the group already locks. `None` when
    /// there is nothing to lean on, so a leg never borrows against a hedge
    /// that is not there.
    fn grouped_deposit(&self, price: Price, direction: Direction, size: f64) -> Option<f64> {
        use crate::accounts::LineView;
        let (Some(context), Some(model), Some(spec)) =
            (&self.group_context, self.grouping_margin_model(), &self.spec)
        else {
            return None;
        };
        let sign = if direction == Direction::Long { 1.0 } else { -1.0 };
        let mut views: Vec<LineView<'_>> = context
            .lines
            .iter()
            .map(|(spec, quantity, price)| LineView::new(spec, *quantity, *price))
            .collect();
        if let Some((own_spec, lines)) = self.model_option_lines() {
            views.extend(
                lines.iter().map(|(_, quantity, price)| LineView::new(own_spec, *quantity, *price)),
            );
        }
        views.push(LineView::new(spec, sign * size, price));
        let required = model.group_required(&views, context.now).ok()?;
        let locked = context.locked + self.locked_margin();
        Some((required - locked).max(0.0))
    }

    /// What opening `units` on `bar` would take out of free capital, as
    /// `(funding, costs)`: the funding is the full notional in a cash account
    /// and the deposit in a margin one, the costs are the entry charges, both
    /// as `open_at` would take them. `None` when the funding itself is refused.
    ///
    /// Nothing is opened or booked. A caller that opens several legs as one
    /// structure sums this over them to know whether the whole fits before it
    /// opens the first.
    pub(crate) fn entry_outlay(&self, bar: &KernelBar, units: f64) -> Option<(f64, f64)> {
        let direction = self.direction;
        let entry_price = self.fill_price_for(bar, direction, true);
        let adjusted_price = self.executed_price(entry_price, direction, true, false, bar);
        let contract_value = adjusted_price * self.multiplier();
        let funding = self.entry_funding(direction, adjusted_price).ok()?;
        let size = self.quantize_units(units);
        if size <= 0.0 {
            return None;
        }
        let fees = self.fee_model.breakdown(contract_value, size, direction, true).map_or_else(
            || self.fee_model.calculate(contract_value, size, direction),
            |breakdown| breakdown.total(),
        );
        let funded = match funding {
            None => contract_value * size,
            Some(EntryFunding::Rate { rate, .. }) => contract_value * size * rate,
            Some(EntryFunding::Deposit { per_contract }) => per_contract * size,
            Some(EntryFunding::Unmargined) => 0.0,
        };
        Some((funded, fees))
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
        limit_priced: bool,
    ) -> Option<EngineEvent> {
        // A capital-fraction open under `next_bar_open` is sized at the price
        // known when it was decided, not at the open it fills at.
        let sizing_ref = self.sizing_ref.take().filter(|_| explicit_units.is_none());
        let adjusted_price = self.executed_price(entry_price, direction, true, limit_priced, bar);

        // Per-instrument capital cap, never exceeding free capital on hand
        // (all cash in cash mode; cash minus locked margin in margin mode).
        let free = self.entry_budget.map_or(self.free_capital(), |b| b.min(self.free_capital()));
        let available = self.alloted_capital.map(|cap| cap.min(free)).unwrap_or(free);

        // Cash mode: size = capital / (price * multiplier * (1 + fee_rate))
        // so notional value + entry fee fits. Margin mode: only the initial
        // margin plus the fee must fit. The fee comes from the ACTIVE fee
        // model: a flat percentage solves in closed form; any other model
        // (itemised, per-share, fixed, tiered) is sized by `size_within_fees`.
        let contract_value = adjusted_price * self.multiplier();
        let funding = match self.entry_funding(direction, adjusted_price) {
            Ok(funding) => funding,
            // A sold leg the model cannot price alone may be carried by the
            // hedge that is already open: priced with its group below.
            Err(RejectReason::InsufficientMargin)
                if explicit_units.is_some() && self.group_context.is_some() =>
            {
                Some(EntryFunding::Deposit { per_contract: f64::INFINITY })
            }
            Err(reason) => return Some(EngineEvent::EntryRejected { idx, reason }),
        };
        let (sizing_value, sizing_funding) = match sizing_ref {
            Some(reference) => {
                let sized_at = self.executed_price(reference, direction, true, limit_priced, bar);
                (
                    sized_at * self.multiplier(),
                    self.entry_funding(direction, sized_at).unwrap_or(funding),
                )
            }
            None => (contract_value, funding),
        };
        let size = match explicit_units {
            Some(units) if self.booking_fact => units,
            Some(units) => self.quantize_units(units),
            None => {
                let budget = size_mult.map_or(available, |mult| mult * available);
                let fit = |budget: f64, value: f64, funding: Option<EntryFunding>| {
                    let size = self.sized_units(budget, value, direction, funding);
                    if funding.is_none() {
                        self.fit_booked_debit(size, budget, available, value, direction, funding)
                    } else {
                        size
                    }
                };
                let decided = fit(budget, sizing_value, sizing_funding);
                if sizing_ref.is_some() {
                    // Sized at the decision price, but never more than the
                    // capital on hand buys at the price it fills at: a gap up
                    // cannot overdraw the account.
                    decided.min(fit(available, contract_value, funding))
                } else {
                    decided
                }
            }
        };

        if size <= 0.0 {
            // Surface the discarded entry instead of silently skipping it —
            // strategies (and their authors) need to learn that the sizing
            // produced zero units, e.g. a size fraction too small for the
            // instrument's lot size. Deliberately does not touch the risk
            // gate's rejection counter: that metric describes constraint
            // refusals, not sizing arithmetic. When an instrument-level
            // margin model set the requirement, say so: the lot's notional
            // may well have fit, and "loosen the entry" is the wrong advice.
            let reason = match funding {
                Some(EntryFunding::Deposit { .. })
                | Some(EntryFunding::Rate { modelled: true, .. }) => {
                    RejectReason::InsufficientMargin
                }
                _ => RejectReason::ZeroSize,
            };
            return Some(EngineEvent::EntryRejected { idx, reason });
        }

        if let Err(reason) = self.check_size_limits(bar, direction, size, contract_value, true) {
            if !self.booking_fact {
                return self.refuse(idx, reason);
            }
            self.fact_breaches.push(reason.as_str());
        }

        let funding = match (funding, explicit_units) {
            (Some(EntryFunding::Deposit { per_contract }), Some(_)) => {
                match self.grouped_deposit(adjusted_price, direction, size) {
                    Some(grouped) => Some(EntryFunding::Deposit {
                        per_contract: (per_contract * size).min(grouped) / size,
                    }),
                    None if per_contract.is_infinite() => {
                        return Some(EngineEvent::EntryRejected {
                            idx,
                            reason: RejectReason::InsufficientMargin,
                        })
                    }
                    None => funding,
                }
            }
            _ => funding,
        };

        // Same per-contract price convention as the exit path: notional
        // scaling rides on the price, contract count stays raw.
        let (entry_breakdown, entry_fees) = self
            .booked_charge(self.fee_model.breakdown(contract_value, size, direction, true), || {
                self.fee_model.calculate(contract_value, size, direction)
            });

        // Capital-fraction sizing fits by construction; explicit unit counts
        // (order API) can exceed the account and are refused instead of
        // silently driving cash negative.
        let funding_cost = match funding {
            None => contract_value * size,
            Some(EntryFunding::Rate { rate, .. }) => contract_value * size * rate,
            Some(EntryFunding::Deposit { per_contract }) => per_contract * size,
            Some(EntryFunding::Unmargined) => 0.0,
        };
        if explicit_units.is_some() && funding_cost + entry_fees > available {
            if !self.booking_fact {
                return Some(EngineEvent::EntryRejected {
                    idx,
                    reason: RejectReason::InsufficientCapital,
                });
            }
            self.fact_breaches.push(RejectReason::InsufficientCapital.as_str());
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
        let (stop_price, target_price) = if self.venue_managed {
            let trail = match (self.effective_stop, stop_override) {
                (StopConfig::Trailing { percent }, None) => Some(adjusted_price * percent),
                _ => None,
            };
            self.protect_outbox.push(ProtectRequest {
                direction,
                size,
                entry_price: adjusted_price,
                stop: stop_price,
                target: target_price,
                trail,
                position_id: 0, // set once the position exists, below
                amend: false,
            });
            (None, None)
        } else {
            (stop_price, target_price)
        };

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
        if self.venue_managed {
            if let Some(request) = self.protect_outbox.last_mut() {
                request.position_id = position_id;
            }
        }
        if let Some(managed) = self.ledger.get_mut(position_id) {
            managed.stop_explicit = stop_override.is_some();
            managed.target_explicit = target_override.is_some();
        }
        let fees_money = self.ledger.book(entry_fees);
        match funding {
            None => self.move_cash(-(self.ledger.notional(adjusted_price, size) + fees_money)),
            Some(EntryFunding::Rate { rate, .. }) => {
                self.margin.lock(position_id, contract_value * size * rate);
                self.move_cash(-fees_money);
            }
            Some(EntryFunding::Deposit { per_contract }) => {
                self.margin.lock(position_id, per_contract * size);
                self.move_cash(-fees_money);
            }
            Some(EntryFunding::Unmargined) => {
                self.margin.lock(position_id, 0.0);
                self.move_cash(-fees_money);
            }
        }

        Some(EngineEvent::Entered { idx, price: adjusted_price, size, direction })
    }

    /// Close `qty` of a position at `exit_price`: the closed slice books a
    /// trade and releases its share of cash or margin; the rest stays open.
    /// A `qty` at or above the position's size closes it whole.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn reduce_at(
        &mut self,
        idx: usize,
        bar: &KernelBar,
        position_id: u64,
        qty: f64,
        exit_price: Price,
        reason: ExitReason,
        limit_priced: bool,
    ) -> Option<EngineEvent> {
        let managed = self.ledger.get(position_id)?;
        let size = managed.position.size;
        if qty >= size {
            return self.close_at(idx, bar, position_id, exit_price, reason, limit_priced);
        }
        let direction = managed.position.direction;
        let entry_ts = managed.entry_timestamp;
        match self.guard_hedge_close(idx, position_id, qty, reason) {
            HedgeGuard::Go => {}
            HedgeGuard::Refuse(refusal) => return Some(refusal),
            HedgeGuard::Held => return None,
        }
        let exit_price = self.executed_price(exit_price, direction, false, limit_priced, bar);
        let fee_price = exit_price * self.multiplier();
        let (exit_breakdown, fees) = self
            .booked_charge(self.fee_model.breakdown(fee_price, qty, direction, false), || {
                self.fee_model.calculate(fee_price, qty, direction)
            });
        let fraction = qty / size;
        let mut trade = self.ledger.reduce_position(
            position_id,
            qty,
            ExitDetails {
                idx,
                timestamp: bar.timestamp,
                price: exit_price,
                entry_timestamp: entry_ts,
                reason,
                fees,
                fee_breakdown: exit_breakdown,
            },
        )?;
        if matches!(self.account, AccountMode::Margin { .. }) {
            // Release the closed slice's share of the lock; the rest
            // stays locked against the remaining size.
            let locked = self.margin.locked_for(position_id);
            let _ = self.margin.set_locked(position_id, locked * (1.0 - fraction));
        }
        // The slice's cash movement is the whole-close arithmetic on the
        // slice's booked amounts; the lock was released above, not here.
        let squared = self.ledger.get(position_id).and_then(|m| m.same_day_squared);
        let refund =
            self.restate_same_day(&mut trade, None, exit_breakdown, squared, Some(position_id));
        self.credit_slice(&trade);
        self.move_cash(refund);
        Some(EngineEvent::Exited { idx, trade: Box::new(trade) })
    }

    /// The units a capital-fraction order adds to a held position: sized by the rule
    /// [`Self::open_at`] sizes an opening with (the fraction of the capital on hand, at the
    /// decision price under `next_bar_open` but never more than the fill price buys). `0.0`
    /// when the fraction buys nothing.
    pub(crate) fn capital_units(
        &self,
        bar: &KernelBar,
        direction: Direction,
        entry_price: Price,
        fraction: f64,
        size_ref: Option<Price>,
        limit_priced: bool,
    ) -> f64 {
        let sizing_ref = size_ref.filter(|_| !limit_priced);
        let price = self.executed_price(entry_price, direction, true, limit_priced, bar);
        let free = self.entry_budget.map_or(self.free_capital(), |b| b.min(self.free_capital()));
        let available = self.alloted_capital.map(|cap| cap.min(free)).unwrap_or(free);
        let value = price * self.multiplier();
        let Ok(funding) = self.entry_funding(direction, price) else { return 0.0 };
        let (sizing_value, sizing_funding) = match sizing_ref {
            Some(reference) => {
                let at = self.executed_price(reference, direction, true, limit_priced, bar);
                (at * self.multiplier(), self.entry_funding(direction, at).unwrap_or(funding))
            }
            None => (value, funding),
        };
        let fit = |budget: f64, value: f64, funding: Option<EntryFunding>| {
            let size = self.sized_units(budget, value, direction, funding);
            if funding.is_none() {
                self.fit_booked_debit(size, budget, available, value, direction, funding)
            } else {
                size
            }
        };
        let decided = fit(fraction * available, sizing_value, sizing_funding);
        if sizing_ref.is_some() {
            decided.min(fit(available, value, funding))
        } else {
            decided
        }
    }

    /// Add a fill slice to a held position (a later slice of the order that opened it, or,
    /// under netting, any other order on its side): charge entry fees and cash or margin for
    /// the slice, average the entry. A level derived from the entry (percent, ATR) follows the
    /// new average; a level the order names (`stop_attach`, `target_attach`) becomes the
    /// position's level, so the net position is protected at it, and the other is kept.
    /// Returns `(price, size)` of the slice, or `None` when the slice cannot
    /// be funded.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn add_at(
        &mut self,
        idx: usize,
        bar: &KernelBar,
        position_id: u64,
        entry_price: Price,
        units: f64,
        stop_attach: Option<Price>,
        target_attach: Option<Price>,
        limit_priced: bool,
    ) -> Option<(Price, f64)> {
        let direction = self.ledger.get(position_id)?.position.direction;
        let price = self.executed_price(entry_price, direction, true, limit_priced, bar);
        let size = match self.lot_size {
            _ if self.booking_fact => units,
            Some(lot) if lot > 0.0 => (units / lot).floor() * lot,
            _ => units,
        };
        let size = match &self.spec {
            Some(spec) if !self.booking_fact => spec.quantize_size(size),
            _ => size,
        };
        if size <= 0.0 {
            return None;
        }
        let contract_value = price * self.multiplier();
        // The order was checked whole when its first slice opened; the
        // position and exposure caps still bind each added slice.
        if let Err(reason) = self.check_size_limits(bar, direction, size, contract_value, false) {
            if !self.booking_fact {
                self.risk.record_rejection();
                return None;
            }
            self.fact_breaches.push(reason.as_str());
        }
        let entry_fees = self.booked_fee(
            match self.fee_model.breakdown(contract_value, size, direction, true) {
                Some(b) => b.total(),
                None => self.fee_model.calculate(contract_value, size, direction),
            },
        );
        let funding = match self.entry_funding(direction, price) {
            Ok(funding) => funding,
            // A sold leg the model cannot price alone may be carried by the hedge that is open:
            // priced with its group below, as the order that opened the line was.
            Err(RejectReason::InsufficientMargin) if self.group_context.is_some() => {
                Some(EntryFunding::Deposit { per_contract: f64::INFINITY })
            }
            Err(_) => return None,
        };
        let funding = match funding {
            Some(EntryFunding::Deposit { per_contract }) => {
                match self.grouped_deposit(price, direction, size) {
                    Some(grouped) => Some(EntryFunding::Deposit {
                        per_contract: (per_contract * size).min(grouped) / size,
                    }),
                    None if per_contract.is_infinite() => return None,
                    None => funding,
                }
            }
            _ => funding,
        };
        let funding_cost = match funding {
            None => contract_value * size,
            Some(EntryFunding::Rate { rate, .. }) => contract_value * size * rate,
            Some(EntryFunding::Deposit { per_contract }) => per_contract * size,
            Some(EntryFunding::Unmargined) => 0.0,
        };
        let free = self.entry_budget.map_or(self.free_capital(), |b| b.min(self.free_capital()));
        let available = self.alloted_capital.map(|cap| cap.min(free)).unwrap_or(free);
        if funding_cost + entry_fees > available {
            if !self.booking_fact {
                return None;
            }
            self.fact_breaches.push(RejectReason::InsufficientCapital.as_str());
        }
        // A level derived from the entry (a percent or ATR setting) shifts with the average; one
        // the caller named rests where it was, as a broker's would. A level this add names
        // replaces the position's.
        self.ledger.add_to_position(position_id, price, size, entry_fees, true)?;
        if !self.venue_managed {
            if let Some(managed) = self.ledger.get_mut(position_id) {
                if stop_attach.is_some() {
                    managed.position.stop_price = stop_attach;
                    managed.stop_explicit = true;
                }
                if target_attach.is_some() {
                    managed.position.target_price = target_attach;
                    managed.target_fresh = true;
                    managed.target_explicit = true;
                }
            }
        } else if stop_attach.is_some() || target_attach.is_some() {
            // The venue holds the legs: tell it the levels the add names, for the whole position.
            let total = self.ledger.get(position_id).map_or(size, |m| m.position.size);
            self.protect_outbox.push(ProtectRequest {
                direction,
                size: total,
                entry_price: price,
                stop: stop_attach,
                target: target_attach,
                trail: None,
                position_id,
                amend: true,
            });
        }
        let fees_money = self.ledger.book(entry_fees);
        match funding {
            None => self.move_cash(-(self.ledger.notional(price, size) + fees_money)),
            Some(EntryFunding::Rate { rate, .. }) => {
                let locked = self.margin.locked_for(position_id);
                let _ = self.margin.set_locked(position_id, locked + contract_value * size * rate);
                self.move_cash(-fees_money);
            }
            Some(EntryFunding::Deposit { per_contract }) => {
                let locked = self.margin.locked_for(position_id);
                let _ = self.margin.set_locked(position_id, locked + per_contract * size);
                self.move_cash(-fees_money);
            }
            Some(EntryFunding::Unmargined) => self.move_cash(-fees_money),
        }
        let _ = idx;
        Some((price, size))
    }

    /// Force-close every position on a margin call.
    ///
    /// Unlike end-of-data finalization or expiry settlement, this is a real
    /// trade-out: it pays exit costs, because a broker liquidating a
    /// position actually crosses the spread.
    ///
    /// Fills at the bar's close unconditionally rather than through the
    /// fill-price model: the breach is *detected* marking equity at the
    /// close, and a broker liquidates on detection. The bar's open predates
    /// the detection, so pricing this through an open-based model would
    /// liquidate at a price from before the information existed.
    pub fn liquidate_all(&mut self, idx: usize, bar: &KernelBar) -> Vec<EngineEvent> {
        self.liquidate_all_as(idx, bar, ExitReason::Liquidation)
    }

    fn liquidate_all_as(
        &mut self,
        idx: usize,
        bar: &KernelBar,
        reason: ExitReason,
    ) -> Vec<EngineEvent> {
        let ids: Vec<u64> = self.ledger.positions().iter().map(|p| p.id).collect();
        let mut events = Vec::new();
        for position_id in ids {
            if self.ledger.get(position_id).is_none() {
                continue;
            }
            if let Some(event) = self.close_at(idx, bar, position_id, bar.close, reason, false) {
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
                self.credit_close(position_id, &trade);
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

#[path = "kernel_lifecycle.rs"]
mod lifecycle;
pub(crate) use lifecycle::ContractEnd;
pub use lifecycle::SplitOutcome;

#[cfg(test)]
#[path = "kernel_partial_tests.rs"]
mod partial_tests;

#[cfg(test)]
#[path = "kernel_venue_tests.rs"]
mod venue_tests;

/// A market order the matcher sweeps on its own: no parent, no venue, no auction time-in-force.
fn is_plain_market_order(o: &&crate::execution::orders::Order) -> bool {
    matches!(o.kind, OrderKind::Market)
        && o.parent_id.is_none()
        && !o.is_venue_driven()
        && !matches!(o.tif, TimeInForce::AtOpen | TimeInForce::AtClose)
}
