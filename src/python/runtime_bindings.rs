//! Python bindings for the multi-strategy runtime.
//!
//! [`PyRuntime`] owns a [`RuntimeCore`] and runs the per-event loop in Rust
//! while holding the interpreter: for each event it walks the subscribed
//! strategies in registration order and does what the single-strategy runner
//! does for one (clock, composite bars, indicators, `on_bar`, the command
//! path, engine events), calling the strategy's Python hooks.
//!
//! Nothing here holds a `RefCell` borrow or a `PyRef` across a call into
//! Python: strategy code re-enters through [`PyRuntimeContext`] and
//! [`PyStrategyView`], which borrow the core for the length of one query.
//!
//! The Python side (`raptorbt.strategy.multi_runner`) does the setup that
//! reuses the legacy helpers, then calls [`PyRuntime::run`].

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;

use crate::portfolio::RejectReason;

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyAttributeError, PyKeyError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyDict, PyInt, PyList, PySlice, PyString, PyTuple};

use crate::accounts::AccountMode;
use crate::core::error::RaptorError;
use crate::core::types::{BacktestConfig, InstrumentConfig, TickData};
use crate::data::book::{BookLevel, DepthTick};
use crate::data::events::{QuoteTick, TradeTick};
use crate::data::{AggregationUnit, BarSpec, BarStamp};
use crate::portfolio::book::PortfolioBook;
use crate::portfolio::kernel::{EngineEvent, KernelBar, StepInput};
use crate::portfolio::session::ScheduleData;
use crate::runtime::commands::{
    CmdError, Command, CommandState, ErrVal, Intent, Order, OrderSpec, Queue, Target, TwapSpec,
    MAX_TAGS, MAX_TAG_KEY, MAX_TAG_VALUE,
};
use crate::runtime::groups::GroupSpec;
use crate::runtime::store::{BarColumns, DepthColumns, TickColumns, TickSlices};
use crate::runtime::{
    AggId, AggKey, ColumnSource, Current, EventKind, EventRef, HeapSize, IndInput, IndKey,
    IndicatorBank, InstrumentId, InstrumentParams, LiveFeed, LocalStreamId, MarketStore,
    MemoryReport, MergeCursor, Ohlc, Pushed, Reveal, RingError, RingKind, RuntimeBuilder,
    RuntimeCore, SlotId, StrategyId, TickRow,
};

mod venue;

use crate::journal::{
    kind as jk, source as jsrc, Control, Journal, JournalingAdapter, Value as JV,
};

use self::venue::VenueRoute;
use super::account_bindings::{convert_group, PyAccountResult};
use super::bindings::{PyBacktestConfig, PyInstrumentConfig, PyPortfolioResult, PyTrade};
use super::errors::{Fail, Val};
use super::execution_bindings::{report_ts, PyAdapter, PyExecutionReport};
use super::indicator_bindings::PyIndicator;
use super::instrument_bindings::PyInstrumentSpec;
use super::journal_bindings::{journal_err, PyJournal};
use super::session_bindings::{
    convert_outcome, parse_instrument_args, working_order_rows, WorkingOrderRow,
};
use super::strategy_bindings::{
    atr_period, convert_snapshot, parse_account_mode, parse_qty_spec, submit_order_checked,
    submit_order_on, submit_twap_on, PyEngineEvent, PyKernelSession, PyPositionSnapshot,
};

/// Events between interpreter signal checks.
const SIGNAL_EVERY: u64 = 4096;

// Bits of the hook mask: the hooks a strategy overrides. Must match
// `CHANNEL_BITS` and `PLAIN_EVENT_BITS` in `raptorbt/strategy/api.py`.
const HOOK_TIME_EVENT: u32 = 1;
const HOOK_COMPOSITE: u32 = 2;
const HOOK_BAR: u32 = 4;
const HOOK_TRADE: u32 = 8;
const HOOK_QUOTE: u32 = 16;
const HOOK_BOOK: u32 = 32;
const HOOK_ORDER_FILLED: u32 = 1 << 6;
const HOOK_RISK_EVENT: u32 = 1 << 7;
const HOOK_POSITION_OPENED: u32 = 1 << 8;
const HOOK_POSITION_CLOSED: u32 = 1 << 9;
const HOOK_ORDER_REJECTED: u32 = 1 << 10;
const HOOK_ORDER_ACCEPTED: u32 = 1 << 11;
const HOOK_ORDER_TRIGGERED: u32 = 1 << 12;
const HOOK_ORDER_CANCELED: u32 = 1 << 13;
const HOOK_ORDER_EXPIRED: u32 = 1 << 14;
const HOOK_ORDER_EVENT: u32 = 1 << 15;
const HOOK_ALGO_STARTED: u32 = 1 << 16;
const HOOK_ALGO_COMPLETED: u32 = 1 << 17;
const HOOK_MARGIN_CALL: u32 = 1 << 18;
const HOOK_CONTRACT_EVENT: u32 = 1 << 19;
const HOOK_CORPORATE_ACTION: u32 = 1 << 20;

const SERIES_KEYS: [&str; 6] = ["timestamps", "open", "high", "low", "close", "volume"];

fn runtime_err(e: RaptorError) -> PyErr {
    PyErr::from(e)
}

fn finished() -> PyErr {
    Fail::new("run_finished", "this strategy's run is finished").into()
}

/// Call `method` on the context host attached to a context
/// (`ctx._raptorbt_host`), with the context and `arg`.
fn ctx_host<'py, A: IntoPyObject<'py>>(
    slf: &Bound<'py, PyRuntimeContext>,
    method: &str,
    arg: A,
) -> PyResult<Py<PyAny>> {
    let host = slf
        .as_any()
        .getattr_opt("_raptorbt_host")?
        .ok_or_else(|| PyAttributeError::new_err("this context has no instrument host"))?;
    Ok(host.call_method1(method, (slf, arg))?.unbind())
}

fn bad_refusal(verdict: &Bound<'_, PyAny>) -> PyErr {
    Fail::new(
        "invalid_order_refusal",
        "order_check must return None to accept an order, or a refusal code (a str, or an \
         object with a str `code`)",
    )
    .param("order_check")
    .value(verdict.get_type().to_string())
    .into()
}

fn busy(message: &str) -> PyErr {
    Fail::new("runtime_busy", message).into()
}

fn poisoned() -> PyErr {
    Fail::new(
        "node_poisoned",
        "a call stopped part way through a booking, so this node's books cannot be trusted; \
         discard it and rebuild it from the venue's positions",
    )
    .into()
}

/// A symbol the run has no data for.
fn unknown_symbol(symbol: &str) -> PyErr {
    Fail::new("unknown_symbol", format!("unknown symbol {symbol:?}: the run has no data for it"))
        .param("symbol")
        .value(symbol)
        .into()
}

/// The runtime's own call order was broken. `Runtime` is not public: `LiveNode` and the runners
/// finalize it before anything else can reach it, so this is an internal fault, not a code.
fn not_ready(message: &str) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(format!(
        "internal error (runtime not ready): {message}"
    ))
}

/// The loop state held for one push. If the push unwinds, the node is marked
/// poisoned: a panic can leave cash and positions half moved.
struct LiveGuard<'a> {
    state: std::cell::RefMut<'a, Option<LoopState>>,
    poisoned: &'a Cell<bool>,
}

impl Drop for LiveGuard<'_> {
    fn drop(&mut self) {
        if std::thread::panicking() {
            self.poisoned.set(true);
        }
    }
}

impl std::ops::Deref for LiveGuard<'_> {
    type Target = Option<LoopState>;
    fn deref(&self) -> &Self::Target {
        &self.state
    }
}

impl std::ops::DerefMut for LiveGuard<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.state
    }
}

enum Inner {
    Building(Box<RuntimeBuilder>),
    Running(Box<RuntimeCore>),
    /// Between states, or after `close`.
    Gone,
}

/// Python callables and classes the loop needs, handed in by the runner.
struct Classes {
    bar: Py<PyAny>,
    composite: Py<PyAny>,
    working_order: Py<PyAny>,
    dispatch: Py<PyAny>,
    trade: Py<PyAny>,
    quote: Py<PyAny>,
    book: Py<PyAny>,
}

/// A bar as pushed: timestamp, open, high, low, close, volume.
type PushedBar = (i64, f64, f64, f64, f64, f64);

/// A clock and the dicts it keeps its schedule in.
struct ClockSlot {
    clock: Py<PyAny>,
    alerts: Py<PyDict>,
    timers: Py<PyDict>,
    /// Exactly `Clock`: with an empty schedule `_advance` only stores the
    /// time, which the loop then does itself.
    plain: bool,
}

/// One composite stream of a strategy: a subscription on one instrument.
struct StreamInfo {
    sub: usize,
    step: u32,
    unit: Py<PyString>,
    kwargs: Py<PyDict>,
    /// Indicators this stream updates itself (not linked to the bank) that
    /// are not `Indicator`s: fed when the stream's bar is handled.
    private: Vec<Py<PyAny>>,
    /// The `Indicator`s it feeds itself, when the event is applied.
    native: Vec<Py<PyIndicator>>,
    reveal: Option<Reveal>,
}

/// How a strategy's queued work reaches the engine.
enum Route {
    /// The Python `CommandPath` (`RAPTORBT_PY_COMMANDS=1`), and the dict it
    /// holds intents in.
    Python {
        path: Py<PyAny>,
        held: Py<PyDict>,
    },
    Native(RefCell<CommandState>),
}

/// Everything the loop needs about one strategy.
/// A strategy and the commands and intents it queued before the run began.
type BootQueued = (Rc<Strat>, Vec<Command>, Vec<Intent>);

struct Strat {
    sid: StrategyId,
    strategy: Py<PyAny>,
    ctx: Py<PyRuntimeContext>,
    mask: u32,
    queue: Py<PyCommandQueue>,
    route: Route,
    active_clock: Cell<usize>,
    /// What grows with the instruments; one borrow per delivery.
    tables: RefCell<StratTables>,
    /// What a node needs to give this strategy an instrument added after it started: the
    /// subscriptions it declared `(step, unit, brick)`, a pristine clock to copy per instrument,
    /// and the private indicators every instrument feeds.
    late: Late,
}

/// A strategy's per-instrument state. It grows when an instrument is added to a running node
/// (a node holds the live-loop guard then, so no event holds a borrow).
struct StratTables {
    clocks: Vec<ClockSlot>,
    /// Per local instrument: indicators fed by its primary bars, and the reveal for the linked ones.
    primary_private: Vec<Vec<Py<PyAny>>>,
    /// The same for `Indicator`s, fed when the event is applied.
    primary_native: Vec<Vec<Py<PyIndicator>>>,
    primary_reveal: Vec<Option<Reveal>>,
    /// By local stream id.
    streams: Vec<StreamInfo>,
}

struct Late {
    subscriptions: Vec<(u32, String, f64)>,
    clock: Py<PyAny>,
    /// `(subscription it follows or None for the primary bars, the indicator)`.
    indicators: Vec<(Option<usize>, Py<PyAny>)>,
}

struct StreamPlan {
    sub: usize,
    step: u32,
    unit: String,
    local: usize,
    agg: AggId,
}

struct IndPlan {
    obj: Py<PyAny>,
    stream_id: Option<usize>,
    /// Local instrument for a routed registration.
    local: Option<usize>,
}

/// A strategy between `register_streams` and `finalize`.
struct Plan {
    sid: StrategyId,
    strategy: Py<PyAny>,
    ctx: Py<PyRuntimeContext>,
    mask: u32,
    commands: Option<Py<PyAny>>,
    queue: Py<PyCommandQueue>,
    clocks: Vec<ClockSlot>,
    subs: usize,
    subscriptions: Vec<(u32, String, f64)>,
    /// A pristine copy of the strategy's clock schedule, for instruments added later.
    clock: Py<PyAny>,
    /// By local stream id.
    streams: Vec<StreamPlan>,
    /// `[local instrument][subscription]` -> local stream id.
    stream_of: Vec<Vec<u16>>,
    inds: Vec<IndPlan>,
}

/// A truncated-series dict, the first row it holds and the watermark it was
/// cut at.
type SeriesCache = (u64, u64, Py<PyDict>);

/// A per-instrument table that is read on every event and changes only when an instrument is
/// added to or retired from a running node. Changing it takes the node's [`LiveGuard`] (no push
/// is running); a read is a shared borrow.
struct Table<T>(RefCell<Vec<T>>);

impl<T> Table<T> {
    fn new(items: Vec<T>) -> Self {
        Self(RefCell::new(items))
    }

    #[inline(always)]
    fn get(&self) -> std::cell::Ref<'_, [T]> {
        std::cell::Ref::map(self.0.borrow(), Vec::as_slice)
    }

    fn push(&self, _quiet: &LiveGuard<'_>, item: T) {
        self.0.borrow_mut().push(item);
    }

    fn set(&self, _quiet: &LiveGuard<'_>, at: usize, item: T) {
        self.0.borrow_mut()[at] = item;
    }
}

struct Shared {
    inner: RefCell<Inner>,
    /// The run's config, for the checks an instrument added later must pass.
    config: BacktestConfig,
    /// Interned symbols by instrument id.
    symbols: Table<Py<PyString>>,
    sym_ids: RefCell<HashMap<String, InstrumentId>>,
    store: Arc<MarketStore>,
    /// Per instrument, whether it has ticks or depth.
    has_ticks: Table<bool>,
    /// Per instrument, the caller's bar arrays (`None` without bars).
    arrays: RefCell<Vec<Option<Py<PyDict>>>>,
    /// Per instrument, the caller's tick arrays (`None` without ticks).
    tick_arrays: RefCell<Vec<Option<Py<PyDict>>>>,
    offsets: Table<i64>,
    /// Truncated series dicts by instrument, valid for one watermark.
    series: RefCell<Vec<Option<SeriesCache>>>,
    /// Rows each live ring holds; `None` for a batch run over a store.
    live_history: Option<usize>,
    /// The live rings, built when the runtime is finalized.
    feed: RefCell<Option<LiveFeed>>,
    /// `keep_history` requests made before the rings exist.
    pending_keeps: RefCell<Vec<(StrategyId, InstrumentId, u64)>>,
    /// Rows a book keeps of its sampled curves, or all of them.
    curve_window: Cell<Option<usize>>,
    classes: Classes,
    tz_offset_ns: i64,
    stamp: BarStamp,
    share: bool,
    keep_curves: bool,
    plans: RefCell<Vec<Plan>>,
    strats: RefCell<Vec<Rc<Strat>>>,
    /// Declared corporate actions and the adjustment policy the history is
    /// on; given to every book and every indicator key when finalized.
    lifecycle:
        RefCell<(Vec<crate::instruments::CorporateAction>, crate::instruments::AdjustmentPolicy)>,
    /// Where orders go instead of the in-process matcher, once chosen.
    venue: RefCell<Option<Rc<VenueRoute>>>,
    /// `venue` is set: one flag read per event, no borrow.
    venue_on: Cell<bool>,
    /// The venue chosen with `set_execution`, built when the runtime is.
    exec_choice: RefCell<Option<ExecChoice>>,
    /// The three times of the event or report in flight, when they are not
    /// all the event's own: see [`Times`].
    times: Cell<Option<Times>>,
    /// Instruments whose bar of the instant in flight a hook may not read: while a fill at the
    /// open of that bar is delivered under `next_bar_open`, the bar's close is not yet known.
    /// Their market data shows the bars before it.
    blind: RefCell<Vec<InstrumentId>>,
    /// The hooks that hear what a step matched are running (not the lifecycle ahead of a decision).
    matching: Cell<bool>,
    /// The hooks that hear the fills at the open of a bar are running, before the strategy has
    /// seen the bar (`next_bar_open`, see [`open_phase`]): what they answer fills at that open.
    opening: Cell<bool>,
    /// The hooks that hear the fills of what rested through a bar's range are running, before the
    /// strategy has decided on the bar (see [`rest_phase`]): they know the bars before it only.
    resting: Cell<bool>,
    /// The open of the bar whose resting fills are being heard (`resting`): a fill at that price
    /// happened at the open (a stop or limit the bar gapped through), and what a hook answers it
    /// with fills at that open, as for any fill at an open.
    rest_open: Cell<Option<f64>>,
    /// Intents a hook queued while hearing a fill at an open, before the strategy saw the bar:
    /// `(strategy, local instrument)` -> the close they were decided at (the previous bar's),
    /// which sizes them when the bar's step stashes them.
    open_intents: RefCell<std::collections::HashMap<(usize, usize), f64>>,
    /// The latest bar the feed took for each instrument, as pushed: a bar taken is delivered
    /// whatever a hook then raised, and a retry of it is a duplicate.
    delivered: RefCell<std::collections::HashMap<usize, PushedBar>>,
    /// The first error a hook raised while its event was being delivered. The event is still
    /// applied (fills, protective levels, marks), what the hook queued is dropped, and the error
    /// is returned once the delivery is over (see [`hook_failed`]).
    hook_error: RefCell<Option<PyErr>>,
    /// Declared processing latency of a live node (ns).
    processing_latency_ns: Cell<i64>,
    /// A push stopped on a panic (a booked amount that could not be
    /// represented) with the books half moved. The node refuses every call
    /// from then on: its state cannot be trusted.
    poisoned: Cell<bool>,
    /// The bars being pushed are history, replayed to prime the strategies.
    warmup: Cell<bool>,
    /// Orders decided at an event time before this (ns) are refused, `warmup`. The bars
    /// before it still feed data and indicators.
    trade_from: Cell<i64>,
    /// The journal of a live node, when it keeps one (see `crate::journal`).
    journal: RefCell<Option<Rc<RefCell<Journal>>>>,
    /// `journal` is set: one flag read per event, no borrow.
    journal_on: Cell<bool>,
    /// What the journal's adapter wrapper shares with this runtime.
    control: Rc<Control>,
    /// Warmup is muted: the strategies' hooks run and their state updates,
    /// but the orders and intents they queue are discarded.
    warmup_muted: Cell<bool>,
    /// Set while the commands the handlers of the current event queued are routed.
    deciding: Cell<bool>,
    /// Orders a strategy sent in answer to a fill it just heard, `(strategy, local instrument,
    /// engine order id)`, still to meet the bar of the step the fill came in.
    follow_ups: RefCell<Vec<(StrategyId, usize, u64)>>,
    /// The host's pre-submit order check and the class of the order it is
    /// shown, once given.
    order_check: RefCell<Option<(Py<PyAny>, Py<PyAny>)>>,
    /// The check judges typed orders only (`order_check_scope="typed"`): `enter()` and
    /// `close_position()` intents are not refused for it and are not shown to it.
    order_check_typed_only: Cell<bool>,
    /// What the strategies queued in `on_start`, set aside while muted
    /// warm-up history runs (it would discard them) and given back after.
    boot_queue: RefCell<Vec<BootQueued>>,
    /// Strategies halted by hand (`ctx.halt`, `LiveNode.halt`, `halt_account`): their
    /// latch is already on the record, so it is not recorded again as the engine's.
    manual_halts: RefCell<std::collections::HashSet<usize>>,
    /// The run's `oms_type`, for the books of instruments added later.
    oms_type: String,
    /// The primary bar spec every tick instrument is cut into, if one was set.
    primary: Cell<Option<crate::data::BarSpec>>,
    /// The longest lookback of any indicator: a ring of an instrument added later holds at
    /// least this many rows.
    lookback: Cell<usize>,
    /// Instruments retired from the node, by id.
    retired: Table<bool>,
    /// The node's own execution adapter, to tell about instruments added and retired.
    adapter_obj: RefCell<Option<Py<PyAny>>>,
    /// A group is open (`begin_group`): pushes are held, and delivered as one
    /// group per timestamp and kind once it is complete.
    bracket: Cell<bool>,
    /// The pushes an open group holds.
    held: RefCell<Vec<GroupEvent>>,
    /// Events pushed since the group opened, for its closing record.
    bracket_events: Cell<u64>,
}

impl Shared {
    /// The interned symbol text of an instrument id, bound to `py`.
    #[inline]
    fn symbol<'py>(&self, py: Python<'py>, id: usize) -> Bound<'py, PyString> {
        self.symbols.get()[id].bind(py).clone()
    }

    fn symbol_text(&self, id: usize) -> String {
        self.symbols.get()[id].to_string()
    }

    fn offset_of(&self, id: usize) -> i64 {
        self.offsets.get()[id]
    }

    fn jcount(&self) -> u64 {
        if !self.journal_on.get() {
            return 0;
        }
        self.journal.borrow().as_ref().map_or(0, |j| j.borrow().count())
    }

    /// Write one record, built only when a journal is kept. The record is
    /// written before the engine acts on what it describes.
    fn jlog(
        &self,
        kind: &str,
        event_ns: i64,
        arrival_ns: i64,
        source: &str,
        client_order_id: Option<&str>,
        payload: impl FnOnce() -> JV,
    ) -> PyResult<()> {
        if !self.journal_on.get() {
            return Ok(());
        }
        self.control.now.set((event_ns, arrival_ns));
        let journal = self.journal.borrow().clone().expect("journal_on implies a journal");
        let appended = journal.borrow_mut().append(
            kind,
            event_ns,
            arrival_ns,
            source,
            client_order_id,
            payload(),
        );
        appended.map(|_| ()).map_err(journal_err)
    }

    /// A record about something that happened inside the current event: it
    /// carries that event's times.
    fn jlog_now(
        &self,
        kind: &str,
        source: &str,
        client_order_id: Option<&str>,
        payload: impl FnOnce() -> JV,
    ) -> PyResult<()> {
        let (ev, arr) = self.control.now.get();
        self.jlog(kind, ev, arr, source, client_order_id, payload)
    }

    /// Whether the strategy's book has latched a halt (margin call, drawdown
    /// limit, halt by hand). Read only when a journal is kept.
    fn halted(&self, sid: StrategyId) -> bool {
        self.journal_on.get() && self.with_book(sid, |b, _| Ok(b.is_halted())).unwrap_or(false)
    }

    /// The engine latched a halt of its own (the drawdown limit, a margin
    /// call) while applying an event: it is on the record before the strategy
    /// hears of it and before anything reaches a venue. `before` is
    /// [`Shared::halted`] ahead of the apply.
    fn journal_halt_edge(
        &self,
        sid: StrategyId,
        before: bool,
        events: &[EngineEvent],
    ) -> PyResult<()> {
        if before || !self.halted(sid) || self.manual_halts.borrow().contains(&sid.index()) {
            return Ok(());
        }
        let margin = events.iter().any(|e| matches!(e, EngineEvent::MarginCall { .. }));
        self.jlog_now(jk::HALT, jsrc::ENGINE, None, || {
            JV::map([
                ("strategy", JV::Int(sid.index() as i64)),
                ("reason", JV::str(if margin { "margin_call" } else { "drawdown" })),
                ("flatten", JV::Bool(false)),
            ])
        })
    }

    /// While a replay rebuilds the node, the failure of a push is the failure
    /// the original run recorded, not the one the replay raised (the adapter
    /// was not called: it answered with what the journal holds, so the
    /// exception differs). The payload the journal holds for the next record,
    /// when that record is of `kind`.
    fn recorded(&self, kind: &str) -> Option<JV> {
        let journal = self.journal.borrow().clone()?;
        let j = journal.borrow();
        if !j.is_replaying() {
            return None;
        }
        let next = j.peek_expected()?;
        (next.record.kind == kind).then(|| next.record.payload.clone())
    }

    /// A push failed after it was journaled: say so, so a replay expects the
    /// same failure.
    fn jnote<T>(&self, py: Python<'_>, op: &str, before: u64, result: &PyResult<T>) {
        if !self.journal_on.get() {
            return;
        }
        let Err(err) = result else { return };
        let Some(journal) = self.journal.borrow().clone() else { return };
        if journal.borrow().count() <= before {
            return;
        }
        let value = err.value(py);
        let code =
            value.getattr("code").ok().and_then(|c| c.extract::<String>().ok()).unwrap_or_else(
                || err.get_type(py).name().map(|n| n.to_string()).unwrap_or_default(),
            );
        let message = value.to_string();
        let (ev, arr) = self.control.now.get();
        let mut note = JV::map([
            ("op", JV::str(op)),
            ("code", JV::str(code)),
            ("message", JV::str(message.lines().next().unwrap_or(""))),
        ]);
        if let Some(recorded) = self.recorded(jk::ERROR) {
            note = recorded;
        }
        let mut j = journal.borrow_mut();
        // A journal that cannot take the note is already broken.
        let _ = j.append(jk::ERROR, ev, arr, jsrc::ENGINE, None, note);
    }
}

/// `(client_order_id, strategy, symbol, side, quantity, filled_qty, status)`.
type VenueOrderRow = (String, usize, String, &'static str, f64, f64, String);
type WorkingRequestRow =
    (crate::python::execution_bindings::PyOrderRequest, String, f64, Option<f64>);

/// The venue a node was told to use.
struct ExecChoice {
    /// `"paper"`, or `"adapter"` for a Python object.
    kind: String,
    adapter: Option<Py<PyAny>>,
    account_id: Option<String>,
    /// Protective orders are watched by the engine, not held by the venue.
    protective_local: bool,
}

/// The three times of an event: when the exchange says it happened, when it
/// reached the engine, and when the engine acted on it. In a batch run all
/// three are the event's own time; a live node may declare an arrival time
/// per push and a processing latency.
#[derive(Debug, Clone, Copy)]
struct Times {
    event: i64,
    arrival: i64,
    processing: i64,
}

impl Shared {
    /// Ask the host's order check about an order about to be submitted.
    /// `None` accepts it; `Some(code)` refuses it with the host's code.
    fn refusal_for(
        &self,
        sid: StrategyId,
        local: usize,
        spec: &OrderSpec,
        client_id: &str,
        ts: i64,
    ) -> PyResult<Option<&'static str>> {
        if let Some(reason) = self.host_refusal(sid, local, ts)? {
            self.jlog_now(jk::REFUSAL, jsrc::ENGINE, Some(client_id), || {
                JV::map([
                    ("strategy", JV::Int(sid.index() as i64)),
                    ("code", JV::str(reason.as_str())),
                    ("message", JV::str(reason.as_str())),
                ])
            })?;
            return Ok(Some(reason.as_str()));
        }
        let Some((check, proposal)) = self
            .order_check
            .borrow()
            .as_ref()
            .map(|(c, p)| Python::attach(|py| (c.clone_ref(py), p.clone_ref(py))))
        else {
            return Ok(None);
        };
        Python::attach(|py| {
            let global = self.global_of(sid, local)?;
            let tags = PyDict::new(py);
            for (k, v) in &spec.tags {
                tags.set_item(k, v)?;
            }
            let kw = PyDict::new(py);
            kw.set_item("symbol", self.symbol(py, global.index()))?;
            kw.set_item("strategy_index", sid.index())?;
            kw.set_item("client_id", client_id)?;
            kw.set_item("side", &spec.side)?;
            kw.set_item("kind", &spec.kind)?;
            kw.set_item("units", spec.units)?;
            kw.set_item("size_frac", spec.size_frac)?;
            kw.set_item("limit_price", spec.limit_price)?;
            kw.set_item("trigger_price", spec.trigger_price)?;
            kw.set_item("tif", &spec.tif)?;
            kw.set_item("reduce_only", spec.reduce_only)?;
            kw.set_item("post_only", spec.post_only)?;
            kw.set_item("tags", tags)?;
            kw.set_item("ts_ns", ts)?;
            let order = proposal.bind(py).call((), Some(&kw))?;
            let ctx = {
                let strats = self.strats.borrow();
                let st = strats.iter().find(|s| s.sid == sid).ok_or_else(finished)?;
                st.ctx.clone_ref(py)
            };
            let verdict = check.bind(py).call1((order, ctx))?;
            if verdict.is_none() {
                return Ok(None);
            }
            let code: String = if let Ok(code) = verdict.extract::<String>() {
                code
            } else if let Some(code) = verdict.getattr_opt("code")? {
                code.extract::<String>().map_err(|_| bad_refusal(&verdict))?
            } else {
                return Err(bad_refusal(&verdict));
            };
            let code = crate::portfolio::intern_host_code(&code).map_err(|msg| {
                PyErr::from(Fail::new("invalid_order_refusal", msg).param("order_check"))
            })?;
            // The refusal is a fact of the run: it is journaled like any other
            // refusal the engine makes, with the order it was about.
            self.jlog_now(jk::REFUSAL, jsrc::ENGINE, Some(client_id), || {
                JV::map([
                    ("strategy", JV::Int(sid.index() as i64)),
                    ("code", JV::str(code)),
                    ("message", JV::str("order_check")),
                ])
            })?;
            Ok(Some(code))
        })
    }

    /// Why an order decided at `ts` on a strategy's `local` instrument may not trade at all:
    /// the instrument is a reference (an index), or the run has not reached its `trade_from`.
    fn host_refusal(
        &self,
        sid: StrategyId,
        local: usize,
        ts: i64,
    ) -> PyResult<Option<RejectReason>> {
        if !self.with_book(sid, |b, _| Ok(b.kernel(local).is_tradable()))? {
            return Ok(Some(RejectReason::NotTradable));
        }
        Ok((ts < self.trade_from.get()).then_some(RejectReason::Warmup))
    }

    fn with_book<R>(
        &self,
        sid: StrategyId,
        f: impl FnOnce(&PortfolioBook, Option<usize>) -> PyResult<R>,
    ) -> PyResult<R> {
        if self.poisoned.get() {
            return Err(poisoned());
        }
        let inner = self.inner.try_borrow().map_err(|_| busy("the runtime is busy"))?;
        match &*inner {
            Inner::Building(b) => f(b.book(sid), None),
            Inner::Running(c) => {
                let cur = c.current().and_then(|cur| c.local_of(sid, cur.event.instrument));
                f(c.book(sid).ok_or_else(finished)?, cur)
            }
            Inner::Gone => Err(finished()),
        }
    }

    fn with_book_mut<R>(
        &self,
        sid: StrategyId,
        f: impl FnOnce(&mut PortfolioBook, Option<usize>) -> PyResult<R>,
    ) -> PyResult<R> {
        if self.poisoned.get() {
            return Err(poisoned());
        }
        let mut inner = self.inner.try_borrow_mut().map_err(|_| busy("the runtime is busy"))?;
        match &mut *inner {
            Inner::Building(b) => f(b.book_mut(sid), None),
            Inner::Running(c) => {
                let cur = c.current().and_then(|cur| c.local_of(sid, cur.event.instrument));
                f(c.book_mut(sid).ok_or_else(finished)?, cur)
            }
            Inner::Gone => Err(finished()),
        }
    }

    /// The bar of the event in flight when it is `local`'s instrument's, else
    /// the last print the strategy's book stepped on it.
    fn current_bar(&self, sid: StrategyId, local: usize) -> Option<KernelBar> {
        let now = match &*self.inner.borrow() {
            Inner::Running(c) => c
                .current()
                .filter(|cur| c.local_of(sid, cur.event.instrument) == Some(local))
                .map(|cur| cur.bar)
                .filter(|bar| bar.close > 0.0)
                // Inside a timestamp group every instrument's bar of the group
                // is the market a decision sees.
                .or_else(|| c.instrument_of(sid, local).and_then(|global| c.group_bar(global))),
            _ => None,
        };
        now.or_else(|| self.with_book(sid, |b, _| Ok(b.last_bar(local))).ok().flatten())
    }

    fn watermark(&self, instrument: InstrumentId) -> u64 {
        let blind = self.blind.borrow().contains(&instrument) as u64;
        match &*self.inner.borrow() {
            Inner::Running(c) => c.watermark(instrument).saturating_sub(blind),
            _ => 0,
        }
    }

    fn tick_watermark(&self, instrument: InstrumentId) -> u64 {
        match &*self.inner.borrow() {
            Inner::Running(c) => c.tick_watermark(instrument),
            _ => 0,
        }
    }

    /// The close of the bar being decided, for a change a decision makes to an order of the
    /// event's own instrument: that bar is not stepped yet, so the kernel's last close is the
    /// bar before. A hook that hears a rest-step fill decides before the bar closed and reads
    /// the bar before, as the kernel does: `None`.
    fn decided_close(&self, sid: StrategyId, local: usize) -> Option<f64> {
        if self.resting.get() {
            return None;
        }
        let global = self.global_of(sid, local).ok()?;
        match &*self.inner.borrow() {
            Inner::Running(c) => c.current().and_then(|cur| match cur.data {
                ScheduleData::Bar(b) if cur.event.instrument == global => Some(b.close),
                _ => None,
            }),
            _ => None,
        }
    }

    fn global_of(&self, sid: StrategyId, local: usize) -> PyResult<InstrumentId> {
        match &*self.inner.borrow() {
            Inner::Building(b) => b.instrument_of(sid, local),
            Inner::Running(c) => c.instrument_of(sid, local),
            Inner::Gone => return Err(finished()),
        }
        .ok_or_else(|| PyKeyError::new_err(local))
    }

    fn local_of(&self, sid: StrategyId, global: InstrumentId) -> Option<usize> {
        match &*self.inner.borrow() {
            Inner::Building(b) => b.local_of(sid, global),
            Inner::Running(c) => c.local_of(sid, global),
            Inner::Gone => None,
        }
    }

    fn locals(&self, sid: StrategyId) -> usize {
        match &*self.inner.borrow() {
            Inner::Building(b) => b.instruments_of(sid),
            Inner::Running(c) => c.instruments_of(sid),
            Inner::Gone => 0,
        }
    }

    /// A depth snapshot by the row an event carried.
    fn depth_at(&self, instrument: InstrumentId, row: u64) -> Option<DepthTick> {
        match &*self.feed.borrow() {
            Some(feed) => feed.depth_at(instrument, row),
            None => self.store.depth_at(instrument, row as u32),
        }
    }

    /// The truncated, read-only arrays of an instrument: its bars up to the
    /// latest bar reached, or, for an instrument without bars, its tick
    /// arrays up to the latest print or quote reached. A live node serves
    /// the rows its rings still hold.
    fn series_of(&self, py: Python<'_>, global: InstrumentId) -> PyResult<Py<PyDict>> {
        if self.live_history.is_some() {
            return self.live_series_of(py, global);
        }
        let g = global.index();
        let (full, n, keys): (Bound<'_, PyDict>, u64, Vec<Bound<'_, PyAny>>) = {
            let arrays = self.arrays.borrow();
            let tick_arrays = self.tick_arrays.borrow();
            match (&arrays[g], &tick_arrays[g]) {
                (Some(bars), _) => (
                    bars.bind(py).clone(),
                    self.watermark(global),
                    SERIES_KEYS.iter().map(|k| PyString::new(py, k).into_any()).collect(),
                ),
                (None, Some(ticks)) => {
                    let d = ticks.bind(py).clone();
                    let keys = d.keys().iter().collect();
                    (d, self.tick_watermark(global), keys)
                }
                (None, None) => return Err(PyKeyError::new_err(self.symbol_text(g))),
            }
        };
        let mut cache = self.series.borrow_mut();
        let slot = &mut cache[g];
        if let Some((_, w, dict)) = slot {
            if *w == n {
                return Ok(dict.clone_ref(py));
            }
        }
        let out = PyDict::new(py);
        let cut = PySlice::new(py, 0, n as isize, 1);
        let kw = PyDict::new(py);
        kw.set_item("write", false)?;
        for key in keys {
            let arr = full.get_item(&key)?.ok_or_else(|| PyKeyError::new_err(key.to_string()))?;
            let view = arr.get_item(&cut)?;
            view.call_method(intern!(py, "setflags"), (), Some(&kw))?;
            out.set_item(key, view)?;
        }
        let out = out.unbind();
        *slot = Some((0, n, out.clone_ref(py)));
        Ok(out)
    }

    /// The rows a live node still holds for an instrument, up to its
    /// watermark: bars once it has had any, else ticks. Copies out of the
    /// ring, once per watermark however many strategies ask.
    fn live_series_of(&self, py: Python<'_>, global: InstrumentId) -> PyResult<Py<PyDict>> {
        let g = global.index();
        // Before the rings exist (in `on_start`) there is nothing to read.
        let feed = self.feed.borrow();
        let bars = feed.as_ref().is_some_and(|f| f.has_bars(global));
        let ticks = feed.as_ref().and_then(|f| f.ticks(global));
        let (first, end) = if bars {
            let ring = feed.as_ref().and_then(|f| f.bars(global)).expect("has bars");
            (ring.first_row(), self.watermark(global).min(ring.end_row()))
        } else if let Some(ring) = ticks {
            (ring.first_row(), self.tick_watermark(global).min(ring.end_row()))
        } else {
            (0, 0)
        };
        let end = end.max(first);
        if let Some((f, w, dict)) = &self.series.borrow()[g] {
            if (*f, *w) == (first, end) {
                return Ok(dict.clone_ref(py));
            }
        }
        let out = PyDict::new(py);
        let kw = PyDict::new(py);
        kw.set_item("write", false)?;
        let put = |key: &str, array: Bound<'_, PyAny>| -> PyResult<()> {
            array.call_method(intern!(py, "setflags"), (), Some(&kw))?;
            out.set_item(key, array)
        };
        match ticks.filter(|_| !bars) {
            None => {
                let rows: Vec<KernelBar> = match feed.as_ref().and_then(|f| f.bars(global)) {
                    Some(ring) => (first..end).filter_map(|r| ring.get(r).ok()).collect(),
                    None => Vec::new(),
                };
                let ts = rows.iter().map(|b| b.timestamp);
                put("timestamps", PyArray1::from_iter(py, ts).into_any())?;
                let columns: [BarColumn; 5] = [
                    ("open", |b| b.open),
                    ("high", |b| b.high),
                    ("low", |b| b.low),
                    ("close", |b| b.close),
                    ("volume", |b| b.volume),
                ];
                for (key, f) in columns {
                    put(key, PyArray1::from_iter(py, rows.iter().map(f)).into_any())?;
                }
            }
            Some(ring) => {
                let rows: Vec<TickRow> = (first..end).filter_map(|r| ring.get(r).ok()).collect();
                let ts = rows.iter().map(|t| t.timestamp);
                put("timestamps", PyArray1::from_iter(py, ts).into_any())?;
                for (key, f) in TICK_COLUMNS {
                    put(key, PyArray1::from_iter(py, rows.iter().map(f)).into_any())?;
                }
            }
        }
        let out = out.unbind();
        drop(feed);
        self.series.borrow_mut()[g] = Some((first, end, out.clone_ref(py)));
        Ok(out)
    }
}

/// The columns of a live node's tick series, after `timestamps`.
type TickColumn = (&'static str, fn(&TickRow) -> f64);
type BarColumn = (&'static str, fn(&KernelBar) -> f64);

const TICK_COLUMNS: [TickColumn; 9] = [
    ("ltp", |t| t.ltp),
    ("bid", |t| t.bid),
    ("ask", |t| t.ask),
    ("buy_qty_delta", |t| t.buy_qty_delta),
    ("sell_qty_delta", |t| t.sell_qty_delta),
    ("ltq", |t| t.ltq),
    ("bid_qty", |t| t.bid_qty),
    ("ask_qty", |t| t.ask_qty),
    ("oi", |t| t.oi),
];

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

/// The latest prices a context shows for one local instrument.
#[derive(Default)]
struct Px {
    bid: Cell<Option<f64>>,
    ask: Cell<Option<f64>>,
    last: Cell<Option<f64>>,
}

/// The `PortfolioContext` surface over one strategy's book in the runtime.
#[pyclass(name = "RuntimeContext", dict, unsendable)]
pub struct PyRuntimeContext {
    shared: Rc<Shared>,
    sid: StrategyId,
    /// Local index of the instrument whose event is being processed.
    local: Cell<usize>,
    idx: Cell<usize>,
    bar: RefCell<Option<Py<PyAny>>>,
    close_ts: Cell<Option<i64>>,
    /// The print or quote being handled, only inside its hook.
    tick: RefCell<Option<Py<PyAny>>>,
    quote: RefCell<Option<Py<PyAny>>>,
    /// Per local instrument.
    prices: Table<Px>,
    /// Per local instrument, the last book: its depth row and, once built,
    /// the Python object. Strategies without `on_order_book` never pay for
    /// the object unless they read `ctx.book`.
    books: RefCell<Vec<Option<BookSlot>>>,
}

/// The last book of a local instrument: its depth row and, once built, the
/// Python object.
type BookSlot = (u64, Option<Py<PyAny>>);

/// A `BookSnapshot` of one depth row.
fn book_object(
    py: Python<'_>,
    sh: &Shared,
    instrument: InstrumentId,
    tick: &crate::data::book::DepthTick,
) -> PyResult<Py<PyAny>> {
    let levels = |levels: &[BookLevel]| PyTuple::new(py, levels.iter().map(|l| (l.price, l.size)));
    let bids = levels(&tick.bids[..tick.bid_len as usize])?;
    let asks = levels(&tick.asks[..tick.ask_len as usize])?;
    let symbol = sh.symbol(py, instrument.index());
    Ok(sh.classes.book.bind(py).call1((tick.timestamp, bids, asks, symbol))?.unbind())
}

impl PyRuntimeContext {
    /// The three times of the event or report in flight.
    fn times(&self) -> Option<Times> {
        self.shared.times.get().or_else(|| match &*self.shared.inner.borrow() {
            Inner::Running(c) => c.current().map(|cur| Times {
                event: cur.event.ts,
                arrival: cur.event.ts,
                processing: cur.event.ts,
            }),
            _ => None,
        })
    }

    fn index_of(&self, symbol: Option<&str>) -> PyResult<usize> {
        match symbol {
            None | Some("") => Ok(self.local.get()),
            Some(name) => self
                .shared
                .sym_ids
                .borrow()
                .get(name)
                .and_then(|&g| self.shared.local_of(self.sid, g))
                .ok_or_else(|| unknown_symbol(name)),
        }
    }

    fn snapshots(&self, local: usize) -> PyResult<Vec<PyPositionSnapshot>> {
        self.shared.with_book(self.sid, |book, _| {
            Ok(book.kernel(local).position_snapshots().into_iter().map(convert_snapshot).collect())
        })
    }
}

#[pymethods]
impl PyRuntimeContext {
    /// The spec of a symbol, of a role bound to one, or of the symbol being
    /// handled; answered by the run's context host (see `ctx_host`).
    #[pyo3(signature = (key=None))]
    fn instrument(slf: &Bound<'_, Self>, key: Option<&str>) -> PyResult<Py<PyAny>> {
        ctx_host(slf, "instrument", key)
    }

    /// An `OptionChain` of the option contracts listed and unexpired now.
    #[pyo3(signature = (underlying=None))]
    fn option_chain(slf: &Bound<'_, Self>, underlying: Option<&str>) -> PyResult<Py<PyAny>> {
        ctx_host(slf, "option_chain", underlying)
    }

    /// The symbol (or tuple of symbols) a requirements role means now.
    fn role(slf: &Bound<'_, Self>, name: &str) -> PyResult<Py<PyAny>> {
        ctx_host(slf, "role", name)
    }

    /// Whether the event being handled is history replayed to prime the
    /// strategy (see `LiveNode`'s `warmup`), not the live feed.
    #[getter]
    fn is_warmup(&self) -> bool {
        self.shared.warmup.get()
    }

    /// The symbol whose bar is being processed.
    #[getter]
    fn symbol(&self, py: Python<'_>) -> PyResult<Py<PyString>> {
        let g = self.shared.global_of(self.sid, self.local.get())?;
        Ok(self.shared.symbols.get()[g.index()].clone_ref(py))
    }

    /// Bar ordinal within the current symbol.
    /// When the exchange says the event in flight happened (ns).
    #[getter]
    fn event_ns(&self) -> Option<i64> {
        self.times().map(|t| t.event)
    }

    /// When the event reached the engine (ns). Equal to `event_ns` unless
    /// the live node was given an arrival time for the push.
    #[getter]
    fn arrival_ns(&self) -> Option<i64> {
        self.times().map(|t| t.arrival)
    }

    /// When the engine acted on the event (ns): its arrival plus the node's
    /// declared processing latency. Deterministic: a replay reads the same.
    #[getter]
    fn processing_ns(&self) -> Option<i64> {
        self.times().map(|t| t.processing)
    }

    #[getter]
    fn idx(&self) -> usize {
        self.idx.get()
    }

    /// The instant of the fill a hook is hearing, when it is not the event's own: under
    /// `next_bar_open` a fill is at the open of the bar, the instant the previous bar closed at.
    /// `None` outside such a hook.
    #[getter]
    fn fill_ns(&self) -> PyResult<Option<i64>> {
        if self.shared.blind.borrow().is_empty() {
            return Ok(None);
        }
        let local = self.local.get();
        let g = self.shared.global_of(self.sid, local)?;
        if !self.shared.blind.borrow().contains(&g) {
            return Ok(None);
        }
        self.shared.with_book(self.sid, |b, _| Ok(b.kernel(local).previous_bar_ts()))
    }

    #[getter]
    fn bar(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        if !self.shared.blind.borrow().is_empty() {
            let g = self.shared.global_of(self.sid, self.local.get())?;
            if self.shared.blind.borrow().contains(&g) {
                // The bar of this instant is not known yet: the last one that is.
                let data = self.shared.series_of(py, g)?;
                let data = data.bind(py);
                let timestamps = data.get_item("timestamps")?.ok_or_else(finished)?;
                // No earlier bar: nothing could have been decided before this one, and no fill
                // can be at its open.
                if timestamps.len()? == 0 {
                    return Ok(self.bar.borrow().as_ref().map(|b| b.clone_ref(py)));
                }
                let row = |k: &str| -> PyResult<Bound<'_, PyAny>> {
                    data.get_item(k)?.ok_or_else(finished)?.get_item(-1)?.call_method0("item")
                };
                let bar = self.shared.classes.bar.bind(py).call1((
                    row("timestamps")?,
                    row("open")?,
                    row("high")?,
                    row("low")?,
                    row("close")?,
                    row("volume")?,
                ))?;
                return Ok(Some(bar.unbind()));
            }
        }
        Ok(self.bar.borrow().as_ref().map(|b| b.clone_ref(py)))
    }

    /// The current bar's timestamp as the caller stamped it.
    #[getter]
    fn timestamp(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &*self.bar.borrow() {
            Some(bar) => bar.bind(py).getattr("timestamp").map(Bound::unbind),
            None => {
                Err(PyAttributeError::new_err("'NoneType' object has no attribute 'timestamp'"))
            }
        }
    }

    /// Epoch ns at which the current bar closed.
    #[getter]
    fn bar_close_ts(&self) -> Option<i64> {
        self.close_ts.get()
    }

    /// The print being handled, or ``None`` outside ``on_trade_tick``.
    #[getter]
    fn tick(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.tick.borrow().as_ref().map(|t| t.clone_ref(py))
    }

    /// The quote being handled, or ``None`` outside ``on_quote``.
    #[getter]
    fn quote(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.quote.borrow().as_ref().map(|q| q.clone_ref(py))
    }

    /// Last seen book for the current symbol, or ``None``. Persists outside
    /// ``on_order_book``.
    #[getter]
    fn book(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        let local = self.local.get();
        let Some((row, built)) = self.books.borrow()[local]
            .as_ref()
            .map(|(r, b)| (*r, b.as_ref().map(|b| b.clone_ref(py))))
        else {
            return Ok(None);
        };
        if built.is_some() {
            return Ok(built);
        }
        let g = self.shared.global_of(self.sid, local)?;
        let tick = self.shared.depth_at(g, row).ok_or_else(|| {
            PyErr::from(Fail::new(
                "history_exhausted",
                "the book's depth row is gone: it was overwritten in the live depth history",
            ))
        })?;
        let object = book_object(py, &self.shared, g, &tick)?;
        if let Some(slot) = self.books.borrow_mut()[local].as_mut() {
            slot.1 = Some(object.clone_ref(py));
        }
        Ok(Some(object))
    }

    /// Last observed bid for the current symbol.
    #[getter]
    fn best_bid(&self) -> Option<f64> {
        self.prices.get()[self.local.get()].bid.get()
    }

    /// Last observed ask for the current symbol.
    #[getter]
    fn best_ask(&self) -> Option<f64> {
        self.prices.get()[self.local.get()].ask.get()
    }

    /// Last trade price (or bar close) for the current symbol.
    #[getter]
    fn last_price(&self) -> Option<f64> {
        self.prices.get()[self.local.get()].last.get()
    }

    #[getter]
    fn equity(&self) -> PyResult<f64> {
        self.shared.with_book(self.sid, |b, _| Ok(b.equity()))
    }

    #[getter]
    fn cash(&self) -> PyResult<f64> {
        self.shared.with_book(self.sid, |b, _| Ok(b.cash()))
    }

    #[getter]
    fn free_capital(&self) -> PyResult<f64> {
        self.shared.with_book(self.sid, |b, _| Ok(b.free_capital()))
    }

    /// Whether a market order decided at a bar fills at the next bar's open (`fill_timing=
    /// "next_bar_open"`), not at the close of the bar it was decided on.
    #[getter]
    fn fills_at_next_open(&self) -> PyResult<bool> {
        self.shared.with_book(self.sid, |b, _| Ok(b.kernel(0).fills_at_next_open()))
    }

    /// Halt this strategy: new entries are refused as `killed`, working
    /// orders are cancelled and, with `flatten=True`, open positions are
    /// closed at the last seen prices when the next event is applied.
    #[pyo3(signature = (reason="manual", flatten=false))]
    fn halt(&self, reason: &str, flatten: bool) -> PyResult<()> {
        let source = jsrc::STRATEGY;
        self.shared.manual_halts.borrow_mut().insert(self.sid.index());
        self.shared.jlog_now(jk::HALT, source, None, || {
            JV::map([
                ("strategy", JV::Int(self.sid.index() as i64)),
                ("reason", JV::str(reason)),
                ("flatten", JV::Bool(flatten)),
            ])
        })?;
        if self.shared.venue_on.get() {
            let venue = self.shared.venue.borrow().clone();
            if let Some(venue) = venue {
                return venue.halt(&self.shared, self.sid, reason, flatten);
            }
        }
        self.shared.with_book_mut(self.sid, |b, _| {
            b.halt(reason, flatten);
            Ok(())
        })
    }

    /// Whether a margin call, the drawdown limit or a halt has latched.
    fn is_halted(&self) -> PyResult<bool> {
        self.shared.with_book(self.sid, |b, _| Ok(b.is_halted()))
    }

    /// The strategy's symbols, in the run's data order.
    #[getter]
    fn symbols<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let mut out = Vec::new();
        let retired = self.shared.retired.get();
        for local in 0..self.shared.locals(self.sid) {
            let g = self.shared.global_of(self.sid, local)?;
            if !retired[g.index()] {
                out.push(self.shared.symbol(py, g.index()));
            }
        }
        PyList::new(py, out)
    }

    /// OHLCV arrays of a symbol (default: the current one), up to and
    /// including the latest bar of that symbol the run has reached. Views are
    /// read-only; bars that have not happened yet are not there.
    #[pyo3(signature = (symbol=None))]
    fn series(&self, py: Python<'_>, symbol: Option<&str>) -> PyResult<Py<PyDict>> {
        let local = self.index_of(symbol)?;
        let g = self.shared.global_of(self.sid, local)?;
        self.shared.series_of(py, g)
    }

    /// Ask a live node to keep at least the last `rows` bars and ticks of a
    /// symbol (default: the current one) readable through ``series()``.
    ///
    /// A node holds a fixed number of rows per symbol (its ``history``) and
    /// overwrites the oldest. Rows a strategy asked to keep are never
    /// overwritten: if keeping them would need more than the node can hold,
    /// the push that would drop one raises ``RuntimeError`` instead. A batch
    /// run holds all its data, so there the call changes nothing.
    #[pyo3(signature = (rows, symbol=None))]
    fn keep_history(&self, rows: u64, symbol: Option<&str>) -> PyResult<()> {
        if rows == 0 {
            return Err(Fail::new("invalid_history", "keep_history needs at least one row")
                .op("keep_history")
                .param("rows")
                .value(0_i64)
                .into());
        }
        let g = self.shared.global_of(self.sid, self.index_of(symbol)?)?;
        if self.shared.live_history.is_none() {
            return Ok(());
        }
        match &mut *self.shared.feed.borrow_mut() {
            Some(feed) => feed.keep(self.sid, g, rows),
            None => self.shared.pending_keeps.borrow_mut().push((self.sid, g, rows)),
        }
        Ok(())
    }

    /// Earliest open position of the current symbol, or ``None``.
    #[getter]
    fn position(&self) -> PyResult<Option<PyPositionSnapshot>> {
        let local = self.local.get();
        self.shared.with_book(self.sid, |b, _| {
            Ok(b.kernel(local).position_snapshot().map(convert_snapshot))
        })
    }

    /// Earliest open position of a symbol, or ``None``.
    #[pyo3(signature = (symbol=None))]
    fn position_for(&self, symbol: Option<&str>) -> PyResult<Option<PyPositionSnapshot>> {
        let local = self.index_of(symbol)?;
        self.shared.with_book(self.sid, |b, _| {
            Ok(b.kernel(local).position_snapshot().map(convert_snapshot))
        })
    }

    /// All open positions of a symbol, in opening order.
    #[pyo3(signature = (symbol=None))]
    fn positions(&self, symbol: Option<&str>) -> PyResult<Vec<PyPositionSnapshot>> {
        self.snapshots(self.index_of(symbol)?)
    }

    #[pyo3(signature = (price, symbol=None))]
    fn set_underlying_price(&self, price: Option<f64>, symbol: Option<&str>) -> PyResult<()> {
        let local = self.index_of(symbol)?;
        self.shared.with_book_mut(self.sid, |b, _| {
            b.kernel_mut(local).set_underlying_price(price);
            Ok(())
        })
    }

    /// Signed unit total across the current symbol's open positions.
    #[getter]
    fn net_position(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.net(py, self.local.get())
    }

    #[pyo3(signature = (symbol=None))]
    fn net_position_for(&self, py: Python<'_>, symbol: Option<&str>) -> PyResult<Py<PyAny>> {
        self.net(py, self.index_of(symbol)?)
    }

    #[getter]
    fn is_net_long(&self) -> PyResult<bool> {
        Ok(self.net_value(self.local.get())? > 0.0)
    }

    #[getter]
    fn is_net_short(&self) -> PyResult<bool> {
        Ok(self.net_value(self.local.get())? < 0.0)
    }

    #[getter]
    fn is_flat(&self) -> PyResult<bool> {
        Ok(self.snapshots(self.local.get())?.is_empty())
    }

    /// Orders still working on a symbol (default: the current one).
    #[pyo3(signature = (symbol=None))]
    fn open_orders<'py>(
        &self,
        py: Python<'py>,
        symbol: Option<&str>,
    ) -> PyResult<Bound<'py, PyList>> {
        let local = self.index_of(symbol)?;
        let g = self.shared.global_of(self.sid, local)?;
        let name = self.shared.symbol(py, g.index()).clone();
        let rows: Vec<WorkingOrderRow> =
            self.shared.with_book(self.sid, |b, _| Ok(working_order_rows(b.kernel(local))))?;
        let kw = PyDict::new(py);
        kw.set_item("symbol", name)?;
        let cls = self.shared.classes.working_order.bind(py);
        let out = rows
            .into_iter()
            .map(|r| {
                cls.call(
                    PyTuple::new(
                        py,
                        [
                            r.0.into_pyobject(py)?.into_any(),
                            r.1.into_pyobject(py)?.into_any(),
                            r.2.into_pyobject(py)?.into_any(),
                            r.3.into_pyobject(py)?.into_any(),
                            r.4.into_pyobject(py)?.into_any(),
                            r.5.into_pyobject(py)?.into_any(),
                            r.6.into_pyobject(py)?.into_any(),
                        ],
                    )?,
                    Some(&kw),
                )
            })
            .collect::<PyResult<Vec<_>>>()?;
        PyList::new(py, out)
    }

    /// ``(locked_margin, realized_pnl, unrealized_pnl, gross_exposure,
    /// net_exposure)`` of this strategy's book; backs ``ctx.portfolio``.
    fn _portfolio_totals(&self) -> PyResult<(f64, f64, f64, f64, f64)> {
        self.shared.with_book(self.sid, |b, _| {
            let t = b.totals();
            Ok((
                t.locked_margin,
                t.realized_pnl,
                t.unrealized_pnl,
                t.gross_exposure,
                t.net_exposure,
            ))
        })
    }

    /// The local index of a symbol in this strategy.
    fn _instrument_index(&self, symbol: Option<&str>) -> PyResult<usize> {
        self.index_of(symbol)
    }
}

impl PyRuntimeContext {
    fn net_value(&self, local: usize) -> PyResult<f64> {
        Python::attach(|py| self.net(py, local)?.extract::<f64>(py))
    }

    /// The legacy `sum(p.size * p.direction for p in positions)`, computed by
    /// Python's own `sum`: since 3.12 it compensates float rounding, so a
    /// Rust fold would differ from the legacy runner in the last bit.
    fn net(&self, py: Python<'_>, local: usize) -> PyResult<Py<PyAny>> {
        let terms: Vec<f64> =
            self.snapshots(local)?.iter().map(|p| p.size * f64::from(p.direction)).collect();
        let sum = py.import("builtins")?.getattr("sum")?;
        Ok(sum.call1((PyList::new(py, terms)?,))?.unbind())
    }
}

// ---------------------------------------------------------------------------
// Session-shaped view of one strategy's book
// ---------------------------------------------------------------------------

/// One strategy's book with the method surface of `PortfolioSession`, so the
/// legacy command path drives it unchanged.
#[pyclass(name = "StrategyView", unsendable)]
pub struct PyStrategyView {
    shared: Rc<Shared>,
    sid: StrategyId,
}

#[pymethods]
impl PyStrategyView {
    #[pyo3(signature = (
        instrument, side, kind, submitted_idx, submitted_ts, client_id,
        units=None, size_frac=None, limit_price=None, trigger_price=None,
        tif="auto", expire_ns=None, stop_price=None, target_price=None,
        offset=None, offset_kind="price", limit_offset=0.0,
        post_only=false, reduce_only=false, parent_id=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn submit_order(
        &self,
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
        self.shared.with_book_mut(self.sid, |book, current| {
            let submitted_idx = book.submission_idx(instrument, submitted_idx, current);
            submit_order_on(
                book.kernel_mut(instrument),
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
        })
    }

    /// Register a TWAP schedule on an instrument; returns its id.
    #[pyo3(signature = (instrument, units, side, slices, interval_ns, submitted_idx, submitted_ts, client_id, reduce_only=false))]
    #[allow(clippy::too_many_arguments)]
    fn submit_twap(
        &self,
        instrument: usize,
        units: f64,
        side: &str,
        slices: u32,
        interval_ns: i64,
        submitted_idx: usize,
        submitted_ts: i64,
        client_id: &str,
        reduce_only: bool,
    ) -> PyResult<u64> {
        self.shared.with_book_mut(self.sid, |book, current| {
            let submitted_idx = book.submission_idx(instrument, submitted_idx, current);
            submit_twap_on(
                book.kernel_mut(instrument),
                units,
                side,
                slices,
                interval_ns,
                submitted_idx,
                submitted_ts,
                client_id,
                reduce_only,
            )
        })
    }

    /// Stop a schedule and cancel the slices it has working.
    fn cancel_twap(&self, instrument: usize, algo_id: u64, idx: usize) -> PyResult<bool> {
        self.shared
            .with_book_mut(self.sid, |b, _| Ok(b.kernel_mut(instrument).cancel_algo(algo_id, idx)))
    }

    fn cancel_order(&self, instrument: usize, idx: usize, order_id: u64) -> PyResult<bool> {
        self.shared.with_book_mut(self.sid, |b, _| {
            Ok(b.kernel_mut(instrument).cancel_order(idx, order_id))
        })
    }

    #[pyo3(signature = (instrument, order_id, units=None, size_frac=None, limit_price=None, trigger_price=None))]
    fn modify_order(
        &self,
        instrument: usize,
        order_id: u64,
        units: Option<f64>,
        size_frac: Option<f64>,
        limit_price: Option<f64>,
        trigger_price: Option<f64>,
    ) -> PyResult<bool> {
        let qty = parse_qty_spec(units, size_frac)?;
        self.shared.with_book_mut(self.sid, |b, _| {
            Ok(b.kernel_mut(instrument).modify_order(order_id, qty, limit_price, trigger_price))
        })
    }

    fn cancel_all_orders(&self, instrument: usize, idx: usize) -> PyResult<Vec<u64>> {
        self.shared
            .with_book_mut(self.sid, |b, _| Ok(b.kernel_mut(instrument).cancel_all_orders(idx)))
    }

    fn link_oco(&self, instrument: usize, order_ids: Vec<u64>) -> PyResult<()> {
        self.shared.with_book_mut(self.sid, |b, _| {
            b.kernel_mut(instrument).link_oco(&order_ids);
            Ok(())
        })
    }

    fn request_close(&self, instrument: usize, position_id: u64) -> PyResult<()> {
        self.shared.with_book_mut(self.sid, |b, _| {
            b.kernel_mut(instrument).request_close(position_id);
            Ok(())
        })
    }

    #[pyo3(signature = (instrument, price=None))]
    fn set_underlying_price(&self, instrument: usize, price: Option<f64>) -> PyResult<()> {
        self.shared.with_book_mut(self.sid, |b, _| {
            b.kernel_mut(instrument).set_underlying_price(price);
            Ok(())
        })
    }

    fn positions(&self, instrument: usize) -> PyResult<Vec<PyPositionSnapshot>> {
        self.shared.with_book(self.sid, |b, _| {
            Ok(b.kernel(instrument)
                .position_snapshots()
                .into_iter()
                .map(convert_snapshot)
                .collect())
        })
    }

    fn position(&self, instrument: usize) -> PyResult<Option<PyPositionSnapshot>> {
        self.shared.with_book(self.sid, |b, _| {
            Ok(b.kernel(instrument).position_snapshot().map(convert_snapshot))
        })
    }

    fn working_orders(&self, instrument: usize) -> PyResult<Vec<WorkingOrderRow>> {
        self.shared.with_book(self.sid, |b, _| Ok(working_order_rows(b.kernel(instrument))))
    }

    fn equity(&self) -> PyResult<f64> {
        self.shared.with_book(self.sid, |b, _| Ok(b.equity()))
    }

    /// Whether a margin call or drawdown kill-switch has latched.
    fn is_halted(&self) -> PyResult<bool> {
        self.shared.with_book(self.sid, |b, _| Ok(b.is_halted()))
    }

    /// Halt the strategy; see `RuntimeContext.halt`.
    #[pyo3(signature = (reason="manual", flatten=false))]
    fn halt(&self, reason: &str, flatten: bool) -> PyResult<()> {
        let source = jsrc::HOST;
        self.shared.manual_halts.borrow_mut().insert(self.sid.index());
        self.shared.jlog_now(jk::HALT, source, None, || {
            JV::map([
                ("strategy", JV::Int(self.sid.index() as i64)),
                ("reason", JV::str(reason)),
                ("flatten", JV::Bool(flatten)),
            ])
        })?;
        if self.shared.venue_on.get() {
            let venue = self.shared.venue.borrow().clone();
            if let Some(venue) = venue {
                return venue.halt(&self.shared, self.sid, reason, flatten);
            }
        }
        self.shared.with_book_mut(self.sid, |b, _| {
            b.halt(reason, flatten);
            Ok(())
        })
    }

    fn cash(&self) -> PyResult<f64> {
        self.shared.with_book(self.sid, |b, _| Ok(b.cash()))
    }

    fn free_capital(&self) -> PyResult<f64> {
        self.shared.with_book(self.sid, |b, _| Ok(b.free_capital()))
    }
}

// ---------------------------------------------------------------------------
// Command queue
// ---------------------------------------------------------------------------

/// `order.name`, or `None` when the order has no such attribute.
fn order_attr<'py, T>(order: &Bound<'py, PyAny>, name: &str) -> PyResult<Option<T>>
where
    T: for<'a> FromPyObject<'a, 'py>,
    for<'a> <T as FromPyObject<'a, 'py>>::Error: Into<PyErr>,
{
    match order.getattr_opt(name)? {
        Some(v) if !v.is_none() => Ok(Some(v.extract::<T>().map_err(Into::into)?)),
        _ => Ok(None),
    }
}

fn order_spec(order: &Bound<'_, PyAny>) -> PyResult<OrderSpec> {
    let spec = OrderSpec {
        side: order.getattr("side")?.extract()?,
        kind: order.getattr("kind")?.extract()?,
        units: order_attr(order, "units")?,
        size_frac: order_attr(order, "size_frac")?,
        limit_price: order_attr(order, "price")?,
        trigger_price: order_attr(order, "trigger")?,
        tif: order.getattr("tif")?.extract()?,
        expire_ns: order_attr(order, "expire_ns")?,
        stop_price: order_attr(order, "stop_price")?,
        target_price: order_attr(order, "target_price")?,
        offset: order_attr(order, "offset")?,
        offset_kind: order_attr(order, "offset_kind")?.unwrap_or_else(|| "price".to_string()),
        limit_offset: order_attr(order, "limit_offset")?.unwrap_or(0.0),
        post_only: order_attr(order, "post_only")?.unwrap_or(false),
        reduce_only: order.getattr("reduce_only")?.extract()?,
        position_id: None,
        tags: order_tags(order)?,
    };
    Ok(spec)
}

/// The order's `tags`: at most [`MAX_TAGS`] string pairs of bounded size. The
/// engine's own `client_id` tag is reserved.
fn order_tags(order: &Bound<'_, PyAny>) -> PyResult<std::collections::BTreeMap<String, String>> {
    let bad = |msg: String, value: String| -> PyErr {
        Fail::new("invalid_order_tags", msg).param("tags").value(value).into()
    };
    let Some(raw) = order.getattr_opt("tags")? else { return Ok(Default::default()) };
    if raw.is_none() {
        return Ok(Default::default());
    }
    let dict = raw.cast::<PyDict>().map_err(|_| {
        bad("order tags must be a dict of str to str".to_string(), raw.get_type().to_string())
    })?;
    if dict.len() > MAX_TAGS {
        return Err(bad(
            format!("an order carries at most {MAX_TAGS} tags"),
            dict.len().to_string(),
        ));
    }
    let mut out = std::collections::BTreeMap::new();
    for (k, v) in dict.iter() {
        let (Ok(k), Ok(v)) = (k.extract::<String>(), v.extract::<String>()) else {
            return Err(bad("order tag keys and values must be str".to_string(), String::new()));
        };
        if k.is_empty() || k.len() > MAX_TAG_KEY || v.len() > MAX_TAG_VALUE {
            return Err(bad(
                format!("an order tag key is 1..={MAX_TAG_KEY} bytes and a value at most {MAX_TAG_VALUE}"),
                k,
            ));
        }
        if k == "client_id" {
            return Err(bad(
                "the tag 'client_id' is reserved for the order's own id".to_string(),
                k,
            ));
        }
        if k == FOLLOW_UP_TAG {
            return Err(bad(
                format!(
                    "the tag '{FOLLOW_UP_TAG}' is reserved for the engine: an order a strategy \
                     sends meets the next bar under next_bar_open, never the bar it decided on"
                ),
                k,
            ));
        }
        out.insert(k, v);
    }
    Ok(out)
}

fn twap_spec(order: &Bound<'_, PyAny>) -> PyResult<TwapSpec> {
    Ok(TwapSpec {
        units: order.getattr("units")?.extract()?,
        side: order.getattr("side")?.extract()?,
        slices: order.getattr("slices")?.extract()?,
        interval_ns: order.getattr("interval_ns")?.extract()?,
        reduce_only: order.getattr("reduce_only")?.extract()?,
    })
}

/// A strategy's queued commands and intents. The strategy's order methods
/// push into it; the runner takes the work out. Orders are converted to
/// plain data when they are submitted.
#[pyclass(name = "CommandQueue")]
#[derive(Default)]
pub struct PyCommandQueue {
    pub(crate) queue: Queue,
    /// Hook bits the strategy switched on after it was bound (see `enable_order_events`).
    hooks: std::sync::atomic::AtomicU32,
    /// Set by the runtime while it delivers one fill event to the strategy: the instant of that
    /// fill. A market order queued meanwhile is the strategy's answer to it, and only such an
    /// order is (no attribute of the order can say so).
    answering: Option<i64>,
}

/// The hook mask a delivery uses: the one the strategy was bound with, and what it has switched on since.
fn live_mask(py: Python<'_>, queue: &Py<PyCommandQueue>, mask: u32) -> u32 {
    mask | queue.bind(py).borrow().hooks.load(std::sync::atomic::Ordering::Relaxed)
}

#[pymethods]
impl PyCommandQueue {
    #[new]
    fn new() -> Self {
        Self::default()
    }

    /// Have the strategy's `on_order_event` called for every order event from now on. A strategy
    /// that places structure groups turns it on before it queues its first leg, so no event it
    /// needs is missed; one that never does pays no call per order event.
    fn enable_order_events(&self) {
        self.hooks.fetch_or(HOOK_ORDER_EVENT, std::sync::atomic::Ordering::Relaxed);
    }

    /// Queue a market entry intent.
    #[pyo3(signature = (size_frac=None, stop_price=None, target_price=None))]
    fn enter(
        &mut self,
        size_frac: Option<f64>,
        stop_price: Option<f64>,
        target_price: Option<f64>,
    ) {
        self.queue.push_intent(Intent::Enter { size_frac, stop_price, target_price });
    }

    /// Queue a whole-position close intent.
    fn close_intent(&mut self) {
        self.queue.push_intent(Intent::Close);
    }

    /// Queue a typed order, or a `Twap` schedule when `twap`.
    #[pyo3(signature = (order, client_id, parent=None, symbol=None, twap=false))]
    fn submit(
        &mut self,
        order: &Bound<'_, PyAny>,
        client_id: String,
        parent: Option<String>,
        symbol: Option<String>,
        twap: bool,
    ) -> PyResult<()> {
        let mut order =
            if twap { Order::Twap(twap_spec(order)?) } else { Order::Plain(order_spec(order)?) };
        if let (Some(_), Order::Plain(spec)) = (self.answering, &mut order) {
            if spec.kind == "market" {
                spec.tags.insert(FOLLOW_UP_TAG.to_string(), "1".to_string());
            }
        }
        self.queue.push_command(Command::Submit {
            client_id,
            order: Box::new(order),
            parent,
            symbol,
        });
        Ok(())
    }

    fn cancel(&mut self, client_id: String) {
        self.queue.push_command(Command::Cancel(client_id));
    }

    fn cancel_all(&mut self) {
        self.queue.push_command(Command::CancelAll);
    }

    fn link_oco(&mut self, client_ids: Vec<String>) {
        self.queue.push_command(Command::LinkOco(client_ids));
    }

    #[pyo3(signature = (position_id, symbol=None))]
    fn close(&mut self, position_id: u64, symbol: Option<String>) {
        self.queue.push_command(Command::Close { position_id, symbol });
    }

    fn close_all_for(&mut self, symbol: String) {
        self.queue.push_command(Command::CloseAllFor(symbol));
    }

    #[pyo3(signature = (client_id, units=None, size_frac=None, limit_price=None, trigger_price=None))]
    fn modify(
        &mut self,
        client_id: String,
        units: Option<f64>,
        size_frac: Option<f64>,
        limit_price: Option<f64>,
        trigger_price: Option<f64>,
    ) {
        self.queue.push_command(Command::Modify {
            client_id,
            units,
            size_frac,
            limit_price,
            trigger_price,
        });
    }

    fn has_commands(&self) -> bool {
        self.queue.has_commands()
    }

    fn has_intents(&self) -> bool {
        self.queue.has_intents()
    }

    /// Take the queued commands as tuples, leaving the queue without any:
    /// `("submit", client_id, order, parent, symbol)` with `order` a dict of
    /// the engine's submit arguments (a schedule carries `"twap": True`),
    /// `("cancel", id)`, `("cancel_all",)`, `("link_oco", ids)`,
    /// `("close", position_id, symbol)`, `("close_all_for", symbol)` and
    /// `("modify", id, kwargs)`.
    fn drain_commands<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let out = PyList::empty(py);
        for command in self.queue.take_commands() {
            out.append(command_tuple(py, command)?)?;
        }
        Ok(out)
    }

    /// Take the queued intents as tuples: `("enter", size_frac, stop, target)`
    /// and `("close",)`.
    fn drain_intents<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let out = PyList::empty(py);
        for intent in self.queue.take_intents() {
            out.append(intent_tuple(py, intent)?)?;
        }
        Ok(out)
    }

    /// Put intents (as `drain_intents` returns them) ahead of the queue.
    fn restore_intents(&mut self, intents: Vec<Bound<'_, PyTuple>>) -> PyResult<()> {
        let held = intents.iter().map(intent_of).collect::<PyResult<Vec<_>>>()?;
        self.queue.restore_intents(held);
        Ok(())
    }
}

fn intent_tuple(py: Python<'_>, intent: Intent) -> PyResult<Bound<'_, PyTuple>> {
    match intent {
        Intent::Enter { size_frac, stop_price, target_price } => {
            ("enter", size_frac, stop_price, target_price).into_pyobject(py)
        }
        Intent::Close => ("close",).into_pyobject(py),
    }
}

fn intent_of(t: &Bound<'_, PyTuple>) -> PyResult<Intent> {
    match t.get_item(0)?.extract::<String>()?.as_str() {
        "enter" => Ok(Intent::Enter {
            size_frac: t.get_item(1)?.extract()?,
            stop_price: t.get_item(2)?.extract()?,
            target_price: t.get_item(3)?.extract()?,
        }),
        _ => Ok(Intent::Close),
    }
}

fn command_tuple(py: Python<'_>, command: Command) -> PyResult<Bound<'_, PyTuple>> {
    match command {
        Command::Submit { client_id, order, parent, symbol } => {
            let d = PyDict::new(py);
            match *order {
                Order::Plain(o) => {
                    d.set_item("side", o.side)?;
                    d.set_item("kind", o.kind)?;
                    d.set_item("units", o.units)?;
                    d.set_item("size_frac", o.size_frac)?;
                    d.set_item("limit_price", o.limit_price)?;
                    d.set_item("trigger_price", o.trigger_price)?;
                    d.set_item("tif", o.tif)?;
                    d.set_item("expire_ns", o.expire_ns)?;
                    d.set_item("stop_price", o.stop_price)?;
                    d.set_item("target_price", o.target_price)?;
                    d.set_item("offset", o.offset)?;
                    d.set_item("offset_kind", o.offset_kind)?;
                    d.set_item("limit_offset", o.limit_offset)?;
                    d.set_item("post_only", o.post_only)?;
                    d.set_item("reduce_only", o.reduce_only)?;
                }
                Order::Twap(t) => {
                    d.set_item("twap", true)?;
                    d.set_item("units", t.units)?;
                    d.set_item("side", t.side)?;
                    d.set_item("slices", t.slices)?;
                    d.set_item("interval_ns", t.interval_ns)?;
                    d.set_item("reduce_only", t.reduce_only)?;
                }
            }
            ("submit", client_id, d, parent, symbol).into_pyobject(py)
        }
        Command::Cancel(id) => ("cancel", id).into_pyobject(py),
        Command::CancelAll => ("cancel_all",).into_pyobject(py),
        Command::LinkOco(ids) => ("link_oco", ids).into_pyobject(py),
        Command::Close { position_id, symbol } => ("close", position_id, symbol).into_pyobject(py),
        Command::CloseAllFor(symbol) => ("close_all_for", symbol).into_pyobject(py),
        Command::Modify { client_id, units, size_frac, limit_price, trigger_price } => {
            let d = PyDict::new(py);
            d.set_item("units", units)?;
            d.set_item("size_frac", size_frac)?;
            d.set_item("limit_price", limit_price)?;
            d.set_item("trigger_price", trigger_price)?;
            ("modify", client_id, d).into_pyobject(py)
        }
    }
}

// ---------------------------------------------------------------------------
// Runtime
// ---------------------------------------------------------------------------

fn parse_spec(step: u32, unit: &str) -> PyResult<BarSpec> {
    let unit = AggregationUnit::parse(unit)
        .map_err(|e| Fail::new("invalid_bar_spec", e.to_string()).param("unit").value(unit))?;
    BarSpec::new(step, unit)
        .map_err(|e| Fail::new("invalid_bar_spec", e.to_string()).param("step").value(step as i64))
        .map_err(PyErr::from)
}

fn column<'py, T: numpy::Element>(
    dict: &Bound<'py, PyDict>,
    key: &str,
) -> PyResult<PyReadonlyArray1<'py, T>> {
    dict.get_item(key)?.ok_or_else(|| PyKeyError::new_err(key.to_string()))?.extract()
}

/// One of the tick or depth dicts of [`PyMarketData`], or `None`.
type MaybeDict<'py> = Option<Bound<'py, PyDict>>;

/// A required column of `kind['SYMBOL']`, or a `ValueError` naming both.
fn field<'py, T: numpy::Element>(
    kind: &str,
    symbol: &str,
    dict: &Bound<'py, PyDict>,
    key: &str,
) -> PyResult<PyReadonlyArray1<'py, T>> {
    optional_field(kind, symbol, dict, key)?.ok_or_else(|| {
        PyErr::from(
            Fail::new("missing_field", format!("{kind}['{symbol}']: missing '{key}'")).param(key),
        )
    })
}

fn optional_field<'py, T: numpy::Element>(
    kind: &str,
    symbol: &str,
    dict: &Bound<'py, PyDict>,
    key: &str,
) -> PyResult<Option<PyReadonlyArray1<'py, T>>> {
    match dict.get_item(key)? {
        None => Ok(None),
        Some(v) if v.is_none() => Ok(None),
        Some(v) => v.extract().map(Some).map_err(|e| {
            PyErr::from(
                Fail::new(
                    "invalid_array",
                    format!("{kind}['{symbol}']: '{key}' is not a usable array: {e}"),
                )
                .param(key),
            )
        }),
    }
}

fn slice_of<'a>(a: &'a Option<PyReadonlyArray1<'_, f64>>) -> PyResult<Option<&'a [f64]>> {
    a.as_ref().map(|a| a.as_slice()).transpose().map_err(PyErr::from)
}

fn tick_columns(symbol: &str, dict: &Bound<'_, PyDict>) -> PyResult<TickColumns> {
    let get = |key: &str| optional_field::<f64>("ticks", symbol, dict, key);
    let ts = field::<i64>("ticks", symbol, dict, "timestamps")?;
    let ltp = field::<f64>("ticks", symbol, dict, "ltp")?;
    let (bid, ask) = (get("bid")?, get("ask")?);
    let (buy, sell, oi) = (get("buy_qty_delta")?, get("sell_qty_delta")?, get("oi")?);
    let (ltq, bid_qty, ask_qty) = (get("ltq")?, get("bid_qty")?, get("ask_qty")?);
    TickColumns::from_slices(
        &format!("ticks['{symbol}']"),
        &TickSlices {
            ts: ts.as_slice()?,
            ltp: ltp.as_slice()?,
            bid: slice_of(&bid)?,
            ask: slice_of(&ask)?,
            buy_qty_delta: slice_of(&buy)?,
            sell_qty_delta: slice_of(&sell)?,
            oi: slice_of(&oi)?,
            ltq: slice_of(&ltq)?,
            bid_qty: slice_of(&bid_qty)?,
            ask_qty: slice_of(&ask_qty)?,
        },
    )
    .map_err(runtime_err)
}

fn depth_columns(symbol: &str, dict: &Bound<'_, PyDict>) -> PyResult<DepthColumns> {
    let label = format!("depth['{symbol}']");
    let ts = field::<i64>("depth", symbol, dict, "timestamps")?;
    let matrix = |key: &str| -> PyResult<PyReadonlyArray2<'_, f64>> {
        dict.get_item(key)?
            .ok_or_else(|| {
                Fail::new("missing_field", format!("{label}: missing '{key}'")).param(key)
            })?
            .extract()
            .map_err(|e| {
                PyErr::from(
                    Fail::new(
                        "invalid_array",
                        format!("{label}: '{key}' must be a 2-D float array: {e}"),
                    )
                    .param(key),
                )
            })
    };
    let (bp, bs) = (matrix("bid_prices")?, matrix("bid_sizes")?);
    let (ap, asz) = (matrix("ask_prices")?, matrix("ask_sizes")?);
    let ts = ts.as_slice()?;
    let n = ts.len();
    for (key, m) in
        [("bid_prices", &bp), ("bid_sizes", &bs), ("ask_prices", &ap), ("ask_sizes", &asz)]
    {
        if m.shape()[0] != n {
            return Err(Fail::new(
                "length_mismatch",
                format!(
                    "{label}: '{key}' has {} rows, expected {n} (one per timestamp)",
                    m.shape()[0]
                ),
            )
            .param(key)
            .value(m.shape()[0])
            .into());
        }
    }
    for (a, b, side) in [("bid_prices", "bid_sizes", 0), ("ask_prices", "ask_sizes", 1)] {
        let (pa, pb) = if side == 0 { (&bp, &bs) } else { (&ap, &asz) };
        if pa.shape()[1] != pb.shape()[1] {
            return Err(Fail::new(
                "length_mismatch",
                format!(
                    "{label}: '{a}' has {} levels but '{b}' has {}",
                    pa.shape()[1],
                    pb.shape()[1]
                ),
            )
            .param(a)
            .into());
        }
    }
    let (bp, bs, ap, asz) = (bp.as_array(), bs.as_array(), ap.as_array(), asz.as_array());
    // A level without a positive price is empty, as in `PortfolioSession.set_depth`.
    let snapshots: Vec<DepthTick> = (0..n)
        .map(|row| {
            let bids: Vec<BookLevel> = (0..bp.shape()[1])
                .map(|l| BookLevel { price: bp[[row, l]], size: bs[[row, l]] })
                .filter(|level| level.price > 0.0)
                .collect();
            let asks: Vec<BookLevel> = (0..ap.shape()[1])
                .map(|l| BookLevel { price: ap[[row, l]], size: asz[[row, l]] })
                .filter(|level| level.price > 0.0)
                .collect();
            DepthTick::from_levels(ts[row], &bids, &asks)
        })
        .collect();
    DepthColumns::from_slice(&label, &snapshots).map_err(runtime_err)
}

/// The market data of a run, built once and shared by every runtime over it:
/// the columns in a [`MarketStore`] behind an `Arc`, and the caller's arrays
/// for `ctx.series()`.
#[pyclass(name = "MarketData")]
pub struct PyMarketData {
    /// Rows per ring when this is the shell of a live node.
    live: Option<usize>,
    store: Arc<MarketStore>,
    symbols: Vec<String>,
    arrays: Vec<Option<Py<PyDict>>>,
    ticks: Vec<Option<Py<PyDict>>>,
    offsets: Vec<i64>,
}

#[pymethods]
impl PyMarketData {
    /// `arrays[i]` holds instrument `i`'s coerced OHLCV arrays as the caller
    /// stamped them (`None` for an instrument without bars); `offsets[i]`
    /// moves a stamp to the bar's close time. `ticks[i]` and `depth[i]` hold
    /// its tick and depth arrays in the `run_tick_strategy` formats, or
    /// `None`. Every instrument needs bars or ticks. Each column is copied
    /// once.
    #[new]
    #[pyo3(signature = (symbols, arrays, offsets, ticks=None, depth=None))]
    fn new(
        symbols: Vec<String>,
        arrays: Vec<MaybeDict<'_>>,
        offsets: Vec<i64>,
        ticks: Option<Vec<MaybeDict<'_>>>,
        depth: Option<Vec<MaybeDict<'_>>>,
    ) -> PyResult<Self> {
        let n = symbols.len();
        let none = || (0..n).map(|_| None).collect::<Vec<MaybeDict<'_>>>();
        let ticks = ticks.unwrap_or_else(none);
        let depth = depth.unwrap_or_else(none);
        if [arrays.len(), offsets.len(), ticks.len(), depth.len()].iter().any(|&len| len != n) {
            return Err(
                Fail::new("length_mismatch", "per-instrument lists must share one length").into()
            );
        }
        let mut store = MarketStore::new();
        for i in 0..n {
            let id = store.add_instrument(&symbols[i]).map_err(runtime_err)?;
            if arrays[i].is_none() && ticks[i].is_none() {
                return Err(Fail::new(
                    "empty_data",
                    format!("'{}' has neither bars nor ticks", symbols[i]),
                )
                .param("symbols")
                .value(symbols[i].as_str())
                .into());
            }
            if let Some(a) = &arrays[i] {
                let ts: Vec<i64> = column::<i64>(a, "timestamps")?
                    .as_slice()?
                    .iter()
                    .map(|t| t + offsets[i])
                    .collect();
                let cols = (
                    column::<f64>(a, "open")?,
                    column::<f64>(a, "high")?,
                    column::<f64>(a, "low")?,
                    column::<f64>(a, "close")?,
                    column::<f64>(a, "volume")?,
                );
                let bars = BarColumns::from_slices(
                    &format!("{} bars", symbols[i]),
                    &ts,
                    cols.0.as_slice()?,
                    cols.1.as_slice()?,
                    cols.2.as_slice()?,
                    cols.3.as_slice()?,
                    cols.4.as_slice()?,
                )
                .map_err(runtime_err)?;
                store.set_bars(id, bars).map_err(runtime_err)?;
            }
            if let Some(t) = &ticks[i] {
                store.set_ticks(id, tick_columns(&symbols[i], t)?).map_err(runtime_err)?;
            }
            if let Some(d) = &depth[i] {
                store.set_depth(id, depth_columns(&symbols[i], d)?).map_err(runtime_err)?;
            }
        }
        Ok(Self {
            live: None,
            store: Arc::new(store),
            symbols,
            arrays: arrays.into_iter().map(|a| a.map(Bound::unbind)).collect(),
            ticks: ticks.into_iter().map(|a| a.map(Bound::unbind)).collect(),
            offsets,
        })
    }

    /// The shell of a live node: `symbols` with no data, a bar stamp offset
    /// per symbol, and rings of `history` rows to be filled by pushes.
    #[staticmethod]
    fn live(symbols: Vec<String>, offsets: Vec<i64>, history: usize) -> PyResult<Self> {
        if offsets.len() != symbols.len() {
            return Err(
                Fail::new("length_mismatch", "per-instrument lists must share one length").into()
            );
        }
        if history == 0 {
            return Err(Fail::new("invalid_history", "history must be at least 1 row")
                .param("history")
                .value(0_i64)
                .into());
        }
        let mut store = MarketStore::new();
        for symbol in &symbols {
            store.add_instrument(symbol).map_err(runtime_err)?;
        }
        let n = symbols.len();
        Ok(Self {
            live: Some(history),
            store: Arc::new(store),
            symbols,
            arrays: (0..n).map(|_| None).collect(),
            ticks: (0..n).map(|_| None).collect(),
            offsets,
        })
    }
}

/// The multi-strategy runtime: shared data, isolated books, a Rust loop.
#[pyclass(name = "Runtime", unsendable)]
pub struct PyRuntime {
    shared: Rc<Shared>,
    /// What a live node's pushes reuse from event to event.
    live_state: RefCell<Option<LoopState>>,
}

#[pymethods]
impl PyRuntime {
    /// `directions`, `instrument_configs` and `instruments` hold one entry per
    /// instrument of `market`, in its order.
    #[new]
    #[pyo3(signature = (
        market, directions, instrument_configs, instruments,
        config, oms_type, account_type, leverage, tz_offset_ns, stamp, share,
        keep_curves, bar_cls, composite_cls, working_order_cls, dispatch,
        trade_cls, quote_cls, book_cls,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        market: PyRef<'_, PyMarketData>,
        directions: Vec<i32>,
        instrument_configs: Vec<Option<PyRef<'_, PyInstrumentConfig>>>,
        instruments: Vec<Option<PyRef<'_, PyInstrumentSpec>>>,
        config: Option<PyRef<'_, PyBacktestConfig>>,
        oms_type: &str,
        account_type: &str,
        leverage: f64,
        tz_offset_ns: i64,
        stamp: &str,
        share: bool,
        keep_curves: bool,
        bar_cls: Py<PyAny>,
        composite_cls: Py<PyAny>,
        working_order_cls: Py<PyAny>,
        dispatch: Py<PyAny>,
        trade_cls: Py<PyAny>,
        quote_cls: Py<PyAny>,
        book_cls: Py<PyAny>,
    ) -> PyResult<Self> {
        let symbols = &market.symbols;
        let n = symbols.len();
        if [directions.len(), instrument_configs.len(), instruments.len()]
            .iter()
            .any(|&len| len != n)
        {
            return Err(
                Fail::new("length_mismatch", "per-instrument lists must share one length").into()
            );
        }
        let mut rust_config: BacktestConfig =
            config.as_deref().map(BacktestConfig::from).unwrap_or_default();
        let mode: AccountMode = parse_account_mode(account_type, leverage)?;
        super::strategy_bindings::check_margin_model(&rust_config, mode)?;
        let stamp = BarStamp::parse(stamp).map_err(|e| {
            Fail::new("invalid_bar_stamp", e.to_string())
                .param("stamp")
                .value(stamp)
                .valid(["open", "close"])
        })?;

        let mut params = Vec::with_capacity(n);
        let mut sym_ids = HashMap::new();
        let mut policy = None;
        for i in 0..n {
            sym_ids.insert(symbols[i].clone(), InstrumentId::new(i as u32));
            let (direction, p, spec) =
                parse_instrument_args(directions[i], instruments[i].as_deref(), oms_type, true)?;
            policy = Some(p);
            super::strategy_bindings::check_market(&rust_config, &symbols[i], spec.as_ref())?;
            let inst_cfg: Option<InstrumentConfig> =
                instrument_configs[i].as_deref().map(InstrumentConfig::from);
            params.push(InstrumentParams {
                direction,
                spec,
                atr_period: atr_period(&rust_config, inst_cfg.as_ref()).unwrap_or(0),
                config: inst_cfg,
            });
        }
        // The session length annualisation counts per day: the instruments' own schedule when the
        // config names none (the first instrument that has one), else the data's trading day.
        if rust_config.session_minutes.is_none() {
            rust_config.session_minutes = params
                .iter()
                .find_map(|p| p.spec.as_ref()?.session.as_ref().map(|s| s.trading_minutes()));
        }
        let builder = RuntimeBuilder::new(
            market.store.clone(),
            params,
            rust_config.clone(),
            mode,
            policy.unwrap_or(crate::portfolio::ledger::PositionPolicy::Net),
        )
        .map_err(runtime_err)?;

        let shared = Shared {
            inner: RefCell::new(Inner::Building(Box::new(builder))),
            config: rust_config,
            symbols: Table::new(symbols.iter().map(|s| PyString::intern(py, s).unbind()).collect()),
            sym_ids: RefCell::new(sym_ids),
            store: market.store.clone(),
            // A live node takes ticks, depth and bars on any symbol, and
            // runs every symbol as a tick session.
            has_ticks: Table::new(
                (0..n)
                    .map(|i| {
                        let id = InstrumentId::new(i as u32);
                        market.live.is_some()
                            || market.store.ticks(id).is_some()
                            || market.store.depth(id).is_some()
                    })
                    .collect(),
            ),
            arrays: RefCell::new(
                market.arrays.iter().map(|a| a.as_ref().map(|a| a.clone_ref(py))).collect(),
            ),
            tick_arrays: RefCell::new(
                market.ticks.iter().map(|a| a.as_ref().map(|a| a.clone_ref(py))).collect(),
            ),
            offsets: Table::new(market.offsets.clone()),
            series: RefCell::new((0..n).map(|_| None).collect()),
            live_history: market.live,
            feed: RefCell::new(None),
            pending_keeps: RefCell::new(Vec::new()),
            curve_window: Cell::new(None),
            classes: Classes {
                bar: bar_cls,
                composite: composite_cls,
                working_order: working_order_cls,
                dispatch,
                trade: trade_cls,
                quote: quote_cls,
                book: book_cls,
            },
            tz_offset_ns,
            stamp,
            share,
            keep_curves,
            plans: RefCell::new(Vec::new()),
            lifecycle: RefCell::new(Default::default()),
            strats: RefCell::new(Vec::new()),
            venue: RefCell::new(None),
            venue_on: Cell::new(false),
            exec_choice: RefCell::new(None),
            times: Cell::new(None),
            blind: RefCell::new(Vec::new()),
            matching: Cell::new(false),
            opening: Cell::new(false),
            resting: Cell::new(false),
            rest_open: Cell::new(None),
            open_intents: RefCell::new(Default::default()),
            delivered: RefCell::new(Default::default()),
            hook_error: RefCell::new(None),
            processing_latency_ns: Cell::new(0),
            poisoned: Cell::new(false),
            warmup: Cell::new(false),
            trade_from: Cell::new(i64::MIN),
            journal: RefCell::new(None),
            journal_on: Cell::new(false),
            control: Rc::new(Control::default()),
            warmup_muted: Cell::new(false),
            deciding: Cell::new(false),
            follow_ups: RefCell::new(Vec::new()),
            order_check: RefCell::new(None),
            order_check_typed_only: Cell::new(false),
            boot_queue: RefCell::new(Vec::new()),
            manual_halts: RefCell::new(Default::default()),
            oms_type: oms_type.to_string(),
            primary: Cell::new(None),
            lookback: Cell::new(0),
            retired: Table::new(vec![false; n]),
            adapter_obj: RefCell::new(None),
            bracket: Cell::new(false),
            held: RefCell::new(Vec::new()),
            bracket_events: Cell::new(0),
        };
        Ok(Self { shared: Rc::new(shared), live_state: RefCell::new(None) })
    }

    /// Add a strategy over `symbols` (in the run's data order). Returns its
    /// id, its context and its session-shaped view.
    #[pyo3(signature = (symbols, account=None))]
    fn add_strategy(
        &self,
        py: Python<'_>,
        symbols: Vec<String>,
        account: Option<u32>,
    ) -> PyResult<(usize, Py<PyRuntimeContext>, Py<PyStrategyView>)> {
        let ids = symbols
            .iter()
            .map(|s| self.shared.sym_ids.borrow().get(s).copied().ok_or_else(|| unknown_symbol(s)))
            .collect::<PyResult<Vec<_>>>()?;
        let sid = match &mut *self.shared.inner.borrow_mut() {
            Inner::Building(b) => {
                let sid = b.add_strategy_in(&ids, account).map_err(runtime_err)?;
                b.book_mut(sid).set_curve_window(self.shared.curve_window.get());
                sid
            }
            _ => return Err(not_ready("strategies are added before the run starts")),
        };
        let ctx = Py::new(
            py,
            PyRuntimeContext {
                shared: self.shared.clone(),
                sid,
                local: Cell::new(0),
                idx: Cell::new(0),
                bar: RefCell::new(None),
                close_ts: Cell::new(None),
                tick: RefCell::new(None),
                quote: RefCell::new(None),
                prices: Table::new((0..ids.len()).map(|_| Px::default()).collect()),
                books: RefCell::new((0..ids.len()).map(|_| None).collect()),
            },
        )?;
        let view = Py::new(py, PyStrategyView { shared: self.shared.clone(), sid })?;
        Ok((sid.index(), ctx, view))
    }

    /// Add an account for `members` strategies to share; returns its id for
    /// `add_strategy(account=...)`. Call before adding those strategies.
    #[pyo3(signature = (
        initial_capital, account_type, leverage, members,
        max_drawdown_pct=None, max_daily_loss=None, max_daily_loss_pct=None,
        max_gross_exposure=None, max_positions=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn add_account(
        &self,
        initial_capital: f64,
        account_type: &str,
        leverage: f64,
        members: usize,
        max_drawdown_pct: Option<f64>,
        max_daily_loss: Option<f64>,
        max_daily_loss_pct: Option<f64>,
        max_gross_exposure: Option<f64>,
        max_positions: Option<usize>,
    ) -> PyResult<u32> {
        if !initial_capital.is_finite() || initial_capital <= 0.0 {
            return Err(Fail::new(
                "invalid_account_spec",
                "an account's initial_capital must be a positive number",
            )
            .param("initial_capital")
            .value(initial_capital)
            .into());
        }
        let mode = parse_account_mode(account_type, leverage)?;
        if let Inner::Building(b) = &*self.shared.inner.borrow() {
            // A margin model funds entries on a margin account; a cash
            // account would ignore it, so the pair is refused here as it is
            // for the plain account.
            super::strategy_bindings::check_margin_model(b.config(), mode)?;
        }
        let spec = GroupSpec {
            initial_capital,
            mode,
            max_drawdown_pct,
            max_daily_loss,
            max_daily_loss_pct,
            max_gross_exposure,
            max_positions,
        };
        match &mut *self.shared.inner.borrow_mut() {
            Inner::Building(b) => {
                let gid = b.add_group(spec, members).map_err(runtime_err)?;
                b.group_mut(gid).set_curve_window(self.shared.curve_window.get());
                Ok(gid)
            }
            _ => Err(not_ready("accounts are added before the run starts")),
        }
    }

    /// Halt every strategy on an account; see `LiveNode.halt_account`.
    fn halt_account(&self, account: u32, reason: &str, flatten: bool) -> PyResult<()> {
        if self.shared.venue_on.get() {
            return Err(Fail::new(
                "unsupported_in_venue_run",
                "halting an account is not supported when orders go to an execution adapter",
            )
            .param("account")
            .into());
        }
        let all: Vec<usize> = self.shared.strats.borrow().iter().map(|st| st.sid.index()).collect();
        self.shared.manual_halts.borrow_mut().extend(all);
        match &mut *self.shared.inner.borrow_mut() {
            Inner::Running(core) => core.halt_group(account, reason, flatten).map_err(runtime_err),
            _ => Err(not_ready("finalize() before halting an account")),
        }
    }

    /// An account's result, once every strategy on it is finished.
    fn finish_account(&self, account: u32, name: String) -> PyResult<PyAccountResult> {
        match &mut *self.shared.inner.borrow_mut() {
            Inner::Running(core) => core
                .finish_group(account)
                .map(|outcome| convert_group(name, outcome))
                .map_err(runtime_err),
            _ => Err(not_ready("finalize() before finish_account()")),
        }
    }

    /// Record what a strategy declared in `on_start`: its composite
    /// subscriptions `(step, unit, brick_size)`, its indicator registrations
    /// `(indicator, stream_id, symbol)`, its per-symbol clocks (in symbol
    /// order), its command path and which loop hooks it overrides.
    ///
    /// Returns whether any indicator was registered without a symbol.
    #[allow(clippy::too_many_arguments)]
    fn register_streams(
        &self,
        py: Python<'_>,
        sid: usize,
        strategy: Py<PyAny>,
        ctx: Py<PyRuntimeContext>,
        commands: Option<Py<PyAny>>,
        clocks: Vec<Py<PyAny>>,
        hooks: u32,
        subscriptions: Vec<(u32, String, f64)>,
        indicators: Vec<(Py<PyAny>, Option<usize>, Option<String>)>,
    ) -> PyResult<bool> {
        let sid = StrategyId::new(sid as u32);
        let locals = self.shared.locals(sid);
        if clocks.len() != locals {
            return Err(Fail::new("length_mismatch", "one clock per symbol is required")
                .param("clocks")
                .value(clocks.len())
                .into());
        }
        let clock_cls = py.import("raptorbt.strategy.clock")?.getattr("Clock")?;
        let pristine = match clocks.first() {
            Some(first) => first.bind(py).call_method0("clone_schedule")?.unbind(),
            None => {
                return Err(Fail::new("length_mismatch", "one clock per symbol is required")
                    .param("clocks")
                    .value(0)
                    .into())
            }
        };
        let mut clock_slots = Vec::with_capacity(locals);
        for clock in clocks {
            let b = clock.bind(py);
            let plain = b.get_type().is(&clock_cls);
            clock_slots.push(ClockSlot {
                alerts: b.getattr("_alerts")?.cast_into::<PyDict>()?.unbind(),
                timers: b.getattr("_timers")?.cast_into::<PyDict>()?.unbind(),
                clock,
                plain,
            });
        }

        // Subscriptions: one aggregator per (symbol, subscription), numbered
        // instrument-major so a strategy's streams on one instrument keep
        // their declaration order.
        let mut streams = Vec::new();
        let mut stream_of = vec![vec![0u16; subscriptions.len()]; locals];
        {
            let mut inner = self.shared.inner.borrow_mut();
            let Inner::Building(builder) = &mut *inner else {
                return Err(not_ready("streams are registered before the run starts"));
            };
            for (local, row) in stream_of.iter_mut().enumerate() {
                let g = builder.instrument_of(sid, local).ok_or_else(finished)?;
                for (sub, (step, unit, brick)) in subscriptions.iter().enumerate() {
                    let key = AggKey::new(
                        g,
                        parse_spec(*step, unit)?,
                        *brick,
                        self.shared.tz_offset_ns,
                        self.shared.stamp,
                    )
                    .map_err(runtime_err)?;
                    let (agg, stream) =
                        builder.register_aggregation(sid, key).map_err(runtime_err)?;
                    row[sub] = stream;
                    debug_assert_eq!(stream as usize, streams.len());
                    streams.push(StreamPlan { sub, step: *step, unit: unit.clone(), local, agg });
                }
            }
        }

        let mut unrouted = false;
        let mut inds = Vec::with_capacity(indicators.len());
        for (obj, stream_id, symbol) in indicators {
            let local = match &symbol {
                None => {
                    unrouted = true;
                    None
                }
                Some(name) => {
                    let found = self
                        .shared
                        .sym_ids
                        .borrow()
                        .get(name)
                        .and_then(|&g| self.shared.local_of(sid, g));
                    match found {
                        Some(l) => Some(l),
                        None => {
                            let known = (0..locals)
                                .map(|l| {
                                    self.shared
                                        .global_of(sid, l)
                                        .map(|g| self.shared.symbol_text(g.index()))
                                })
                                .collect::<PyResult<Vec<_>>>()?
                                .join(", ");
                            let shown = PyString::new(py, name).repr()?;
                            return Err(Fail::new(
                                "unknown_symbol",
                                format!(
                                    "register_indicator(symbol={shown}) names a symbol that is not \
                                     in this run; known symbols: {known}"
                                ),
                            )
                            .op("register_indicator")
                            .param("symbol")
                            .value(name.as_str())
                            .valid(self.shared.symbols.get().iter().map(|s| s.to_string()))
                            .into());
                        }
                    }
                }
            };
            inds.push(IndPlan { obj, stream_id, local });
        }

        let strategy_queue =
            strategy.bind(py).getattr("_queue")?.cast_into::<PyCommandQueue>()?.unbind();
        self.shared.plans.borrow_mut().push(Plan {
            sid,
            strategy,
            ctx,
            mask: hooks,
            commands,
            queue: strategy_queue,
            clocks: clock_slots,
            subs: subscriptions.len(),
            subscriptions: subscriptions.clone(),
            clock: pristine,
            streams,
            stream_of,
            inds,
        });
        Ok(unrouted)
    }

    /// Aggregate every instrument's trade prints into `step` `unit` bars: the
    /// primary bars of a tick run. Call before `finalize`.
    fn set_primary_bars(&self, step: u32, unit: &str) -> PyResult<()> {
        let spec = parse_spec(step, unit)?;
        let mut inner = self.shared.inner.borrow_mut();
        let Inner::Building(builder) = &mut *inner else {
            return Err(not_ready("primary bars are set before the run starts"));
        };
        self.shared.primary.set(Some(spec));
        for (i, &has) in self.shared.has_ticks.get().iter().enumerate() {
            if has {
                builder
                    .set_primary_bars(InstrumentId::new(i as u32), spec, self.shared.tz_offset_ns)
                    .map_err(runtime_err)?;
            }
        }
        Ok(())
    }

    /// Freeze the registrations: link what can be linked and build the core.
    /// Declare the corporate actions of the run and the adjustment policy its
    /// history is on (`"raw"` or `"adjusted"`). Call before `finalize`.
    ///
    /// Every action must name a symbol of the run. The policy is part of
    /// every indicator's identity, so strategies on different bases never
    /// share one.
    #[pyo3(signature = (actions, policy))]
    fn set_corporate_actions(
        &self,
        actions: Vec<super::lifecycle_bindings::PyCorporateAction>,
        policy: &str,
    ) -> PyResult<()> {
        let (actions, policy) =
            super::lifecycle_bindings::extract_actions(Some(actions), Some(policy))?;
        if let Some(a) =
            actions.iter().find(|a| !self.shared.sym_ids.borrow().contains_key(&a.instrument))
        {
            let ids = self.shared.sym_ids.borrow();
            let mut valid: Vec<&String> = ids.keys().collect();
            valid.sort();
            return Err(Fail::new(
                "unknown_symbol",
                format!("corporate action on {:?}: no data for that symbol", a.instrument),
            )
            .param("corporate_actions")
            .value(a.instrument.as_str())
            .valid(valid)
            .into());
        }
        if !matches!(&*self.shared.inner.borrow(), Inner::Building(_)) {
            return Err(not_ready("corporate actions must be declared before finalize"));
        }
        *self.shared.lifecycle.borrow_mut() = (actions, policy);
        Ok(())
    }

    fn finalize(&self, py: Python<'_>) -> PyResult<()> {
        let plans = std::mem::take(&mut *self.shared.plans.borrow_mut());
        let (actions, policy) = self.shared.lifecycle.borrow().clone();
        let mut builder = match std::mem::replace(&mut *self.shared.inner.borrow_mut(), Inner::Gone)
        {
            Inner::Building(b) => b,
            other => {
                *self.shared.inner.borrow_mut() = other;
                return Err(not_ready("the runtime is already built"));
            }
        };

        // A live node's rings hold at least what its indicators read.
        let live_rows = match self.shared.live_history {
            Some(history) => {
                let mut lookback = 0;
                for plan in &plans {
                    for ind in &plan.inds {
                        if let Ok(obj) = ind.obj.bind(py).cast::<PyIndicator>() {
                            lookback = lookback.max(obj.borrow().spec().lookback());
                        }
                    }
                }
                self.shared.lookback.set(lookback);
                Some(history.max(lookback))
            }
            None => None,
        };

        // Registrations per indicator object, across strategies.
        let mut counts: HashMap<usize, usize> = HashMap::new();
        for plan in &plans {
            for ind in &plan.inds {
                *counts.entry(ind.obj.as_ptr() as usize).or_default() += 1;
            }
        }

        // (plan, indicator) -> reader slot, for the ones that get linked.
        let mut linked: HashMap<(usize, usize), SlotId> = HashMap::new();

        if self.shared.share {
            for (pi, plan) in plans.iter().enumerate() {
                for (ii, ind) in plan.inds.iter().enumerate() {
                    let Ok(obj) = ind.obj.bind(py).cast::<PyIndicator>() else { continue };
                    let spec = {
                        let r = obj.borrow();
                        if r.updates() != 0 || r.is_linked() {
                            continue;
                        }
                        r.spec()
                    };
                    if counts[&(ind.obj.as_ptr() as usize)] != 1 {
                        continue;
                    }
                    let (input, stream) = match (ind.stream_id, ind.local) {
                        (None, Some(local)) => {
                            let g = builder.instrument_of(plan.sid, local).ok_or_else(finished)?;
                            (IndInput::Primary(g), None)
                        }
                        // Fed by every instrument of the strategy. A live node's set
                        // grows, and the bank's input set is frozen: the indicator
                        // stays private and follows the instruments (see `Late`).
                        (None, None) if self.shared.live_history.is_some() => continue,
                        (None, None) => {
                            let all: Vec<InstrumentId> = (0..builder_locals(&builder, plan.sid))
                                .filter_map(|l| builder.instrument_of(plan.sid, l))
                                .collect();
                            (
                                IndInput::AllPrimary(
                                    builder.primary_order(&all).map_err(runtime_err)?,
                                ),
                                None,
                            )
                        }
                        (Some(sub), Some(local)) if sub < plan.subs => {
                            let stream = plan.stream_of[local][sub];
                            let agg = plan.streams[stream as usize].agg;
                            (IndInput::Composite(agg), Some(LocalStreamId(u32::from(stream))))
                        }
                        // Fed by every symbol's stream, or by none: private.
                        _ => continue,
                    };
                    let mut key = IndKey::new(input, spec);
                    key.adjustment = policy;
                    let reg =
                        builder.register_indicator(plan.sid, key, stream).map_err(runtime_err)?;
                    linked.insert((pi, ii), reg.slot);
                }
            }
        }

        for plan in &plans {
            builder
                .book_mut(plan.sid)
                .set_corporate_actions(&actions, policy)
                .map_err(super::lifecycle_bindings::action_err)?;
        }
        let route = match self.shared.exec_choice.borrow_mut().take() {
            None => None,
            Some(choice) => {
                if self.shared.live_history.is_none() {
                    return Err(Fail::new(
                        "invalid_execution",
                        "an execution adapter belongs to a live node; a batch run is simulated in-process",
                    )
                    .param("execution")
                    .into());
                }
                if plans.iter().any(|p| p.commands.is_some()) {
                    return Err(Fail::new(
                        "invalid_execution",
                        "an execution adapter needs the native command path (RAPTORBT_PY_COMMANDS is set)",
                    )
                    .param("execution")
                    .into());
                }
                *self.shared.adapter_obj.borrow_mut() = choice
                    .adapter
                    .as_ref()
                    .filter(|_| choice.kind == "adapter")
                    .map(|o| o.clone_ref(py));
                builder.manage_books_by_venue();
                builder.check_venue_safe().map_err(|e| {
                    PyErr::from(Fail::new("invalid_execution", e.to_string()).param("execution"))
                })?;
                let raised = Rc::new(RefCell::new(None));
                let adapter: Box<dyn crate::execution::adapter::ExecutionAdapter> = match (
                    choice.kind.as_str(),
                    choice.adapter.as_ref().map(|o| o.clone_ref(py)),
                ) {
                    ("paper", _) => Box::new(crate::execution::adapter::PaperVenue::new(
                        builder.shadow_books().map_err(runtime_err)?,
                    )),
                    (_, Some(obj)) => Box::new(PyAdapter::new(obj, raised.clone())),
                    _ => unreachable!("set_execution checks the adapter"),
                };
                // A node with an adapter of its own, a journal or an execution
                // epoch sends every request through the journal's wrapper, which
                // tracks the requests whose outcome is unknown whether or not
                // there is a journal; a paper node with neither is not wrapped.
                let journal = self.shared.journal.borrow().clone();
                let adapter: Box<dyn crate::execution::adapter::ExecutionAdapter> = if journal
                    .is_some()
                    || self.shared.control.epoch.get().is_some()
                    || choice.kind == "adapter"
                {
                    Box::new(JournalingAdapter::new(adapter, journal, self.shared.control.clone()))
                } else {
                    adapter
                };
                Some(Rc::new(VenueRoute::new(
                    adapter,
                    choice.account_id,
                    raised,
                    plans.len(),
                    choice.protective_local,
                )))
            }
        };

        let core = builder.build().map_err(runtime_err)?;
        let bank: std::sync::Arc<IndicatorBank> = core.bank().clone();
        for (&(pi, ii), &slot) in &linked {
            let obj = plans[pi].inds[ii].obj.bind(py).cast::<PyIndicator>()?.clone();
            obj.borrow_mut().link(bank.clone(), slot);
        }

        let mut strats = Vec::with_capacity(plans.len());
        for (pi, plan) in plans.into_iter().enumerate() {
            let locals = core.instruments_of(plan.sid);
            let mut primary_private: Vec<Vec<Py<PyAny>>> =
                (0..locals).map(|_| Vec::new()).collect();
            let mut primary_native: Vec<Vec<Py<PyIndicator>>> =
                (0..locals).map(|_| Vec::new()).collect();
            let mut stream_private: Vec<Vec<Py<PyAny>>> =
                (0..plan.streams.len()).map(|_| Vec::new()).collect();
            let mut stream_native: Vec<Vec<Py<PyIndicator>>> =
                (0..plan.streams.len()).map(|_| Vec::new()).collect();
            for (ii, ind) in plan.inds.iter().enumerate() {
                if linked.contains_key(&(pi, ii)) {
                    continue;
                }
                let native =
                    ind.obj.bind(py).cast::<PyIndicator>().ok().map(|o| o.clone().unbind());
                if let Some(native) = &native {
                    let mut o = native.bind(py).borrow_mut();
                    o.attach(bank.clone());
                    // Fed by every instrument's stream: no event's stream is its own.
                    if ind.local.is_none() {
                        o.set_fleet();
                    }
                }
                let targets: Vec<usize> = match ind.local {
                    Some(l) => vec![l],
                    None => (0..locals).collect(),
                };
                for l in targets {
                    match ind.stream_id {
                        None => match &native {
                            Some(n) => primary_native[l].push(n.clone_ref(py)),
                            None => primary_private[l].push(ind.obj.clone_ref(py)),
                        },
                        Some(sub) if sub < plan.subs => {
                            let stream = plan.stream_of[l][sub];
                            match &native {
                                Some(n) => stream_native[stream as usize].push(n.clone_ref(py)),
                                None => stream_private[stream as usize].push(ind.obj.clone_ref(py)),
                            }
                        }
                        // A handle that names no subscription is never fed.
                        Some(_) => {}
                    }
                }
            }
            let late = Late {
                subscriptions: plan.subscriptions.clone(),
                clock: plan.clock.clone_ref(py),
                indicators: plan
                    .inds
                    .iter()
                    .enumerate()
                    .filter(|&(ii, ind)| {
                        !linked.contains_key(&(pi, ii))
                            && ind.local.is_none()
                            && ind.stream_id.is_none_or(|sub| sub < plan.subs)
                    })
                    .map(|(_, ind)| (ind.stream_id, ind.obj.clone_ref(py)))
                    .collect(),
            };
            let route = match plan.commands {
                Some(path) => {
                    let held = path.bind(py).getattr("_held")?.cast_into::<PyDict>()?.unbind();
                    Route::Python { path, held }
                }
                None => Route::Native(RefCell::new(CommandState::default())),
            };
            let primary_reveal = (0..locals)
                .map(|l| {
                    core.instrument_of(plan.sid, l).and_then(|g| bank.primary_handle(plan.sid, g))
                })
                .collect();
            let stream_infos = plan
                .streams
                .iter()
                .zip(stream_private.into_iter().zip(stream_native))
                .enumerate()
                .map(|(i, (sp, (private, native)))| -> PyResult<StreamInfo> {
                    let g = core.instrument_of(plan.sid, sp.local).ok_or_else(finished)?;
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("symbol", self.shared.symbol(py, g.index()))?;
                    Ok(StreamInfo {
                        sub: sp.sub,
                        step: sp.step,
                        unit: PyString::new(py, &sp.unit).unbind(),
                        kwargs: kwargs.unbind(),
                        private,
                        native,
                        reveal: bank.stream_handle(plan.sid, LocalStreamId(i as u32)),
                    })
                })
                .collect::<PyResult<Vec<_>>>()?;
            strats.push(Rc::new(Strat {
                sid: plan.sid,
                strategy: plan.strategy,
                ctx: plan.ctx,
                mask: plan.mask,
                queue: plan.queue,
                route,
                active_clock: Cell::new(usize::MAX),
                tables: RefCell::new(StratTables {
                    clocks: plan.clocks,
                    primary_private,
                    primary_native,
                    primary_reveal,
                    streams: stream_infos,
                }),
                late,
            }));
        }
        *self.shared.strats.borrow_mut() = strats;
        self.shared.venue_on.set(route.is_some());
        *self.shared.venue.borrow_mut() = route;
        *self.shared.inner.borrow_mut() = Inner::Running(Box::new(core));
        if let Some(rows) = live_rows {
            let mut feed = LiveFeed::new(
                self.shared.symbols.get().iter().map(|s| s.to_string()).collect(),
                rows,
            );
            for (owner, instrument, keep) in self.shared.pending_keeps.take() {
                feed.keep(owner, instrument, keep);
            }
            *self.shared.feed.borrow_mut() = Some(feed);
            *self.live_state.borrow_mut() = Some(LoopState::new(&self.shared)?);
        }
        Ok(())
    }

    /// Send this node's orders to a venue instead of matching them
    /// in-process. `kind` is `"paper"` (a simulated exchange) or `"adapter"`
    /// (a Python object with `submit`, `cancel` and `modify`). Call before
    /// `finalize()`.
    #[pyo3(signature = (kind, adapter=None, account_id=None, protective="venue"))]
    fn set_execution(
        &self,
        py: Python<'_>,
        kind: &str,
        adapter: Option<Py<PyAny>>,
        account_id: Option<String>,
        protective: &str,
    ) -> PyResult<()> {
        if !matches!(protective, "venue" | "local") {
            return Err(Fail::new("invalid_execution", "protective must be 'venue' or 'local'")
                .param("protective")
                .value(protective)
                .valid(["venue", "local"])
                .into());
        }
        match kind {
            "paper" => {}
            "adapter" => {
                let obj = adapter.as_ref().ok_or_else(|| {
                    PyErr::from(Fail::new("invalid_execution", "kind='adapter' needs an adapter"))
                })?;
                for method in ["submit", "cancel", "modify"] {
                    if !obj.bind(py).getattr_opt(method)?.is_some_and(|m| m.is_callable()) {
                        return Err(Fail::new(
                            "invalid_execution",
                            format!("an execution adapter needs a callable {method}()"),
                        )
                        .param("execution")
                        .value(obj.bind(py).get_type().name()?.to_string())
                        .into());
                    }
                }
            }
            other => {
                return Err(Fail::new(
                    "invalid_execution",
                    format!("unknown execution kind {other:?}"),
                )
                .param("execution")
                .value(other)
                .valid(["paper", "adapter"])
                .into())
            }
        }
        if !matches!(&*self.shared.inner.borrow(), Inner::Building(_)) {
            return Err(not_ready("set_execution() before finalize()"));
        }
        *self.shared.exec_choice.borrow_mut() = Some(ExecChoice {
            kind: kind.to_string(),
            adapter,
            account_id,
            protective_local: protective == "local",
        });
        Ok(())
    }

    /// Keep a journal of this node, and/or give it an execution epoch. Call
    /// before `finalize()`.
    #[pyo3(signature = (journal=None, epoch=None))]
    fn set_journal(
        &self,
        journal: Option<PyRef<'_, PyJournal>>,
        epoch: Option<u64>,
    ) -> PyResult<()> {
        if !matches!(&*self.shared.inner.borrow(), Inner::Building(_)) {
            return Err(not_ready("set_journal() before finalize()"));
        }
        if let Some(j) = journal {
            *self.shared.journal.borrow_mut() = Some(j.inner.clone());
            self.shared.journal_on.set(true);
        }
        self.shared.control.epoch.set(epoch);
        Ok(())
    }

    /// Take a new execution epoch after the journal was replayed. The venue
    /// learns of it at the next `heartbeat()` or request.
    fn set_epoch(&self, epoch: u64) {
        self.shared.control.epoch.set(Some(epoch));
    }

    /// While set, requests are not sent to a real venue: the node is being
    /// rebuilt from its journal (see `raptorbt.journal`).
    fn set_replay(&self, replay: bool) {
        self.shared.control.suppress.set(replay);
    }

    /// The event time and arrival time of the latest input the node took.
    fn clock(&self) -> (i64, i64) {
        self.shared.control.now.get()
    }

    /// The latest event time the node's instruments have reached (`0` before the first): the
    /// node's clock, kept with or without a journal.
    fn last_event_ns(&self) -> i64 {
        let fed = self.shared.feed.borrow().as_ref().and_then(LiveFeed::last_event_ns);
        fed.unwrap_or(0).max(self.shared.control.now.get().0)
    }

    /// The venue lists an order as open, so its request arrived.
    fn confirm(&self, client_order_id: &str) {
        self.shared.control.unconfirmed.borrow_mut().remove(client_order_id);
    }

    /// Client order ids whose request may or may not have reached the venue.
    fn unconfirmed(&self) -> Vec<String> {
        self.shared.control.unconfirmed.borrow().iter().cloned().collect()
    }

    /// Whether a call stopped part way through a booking: the books cannot be trusted and every
    /// push is refused as `node_poisoned`.
    /// The latest bar the node took for `instrument`, as pushed, whether or not the hooks that
    /// ran on it raised: `(timestamp, open, high, low, close, volume)`.
    fn delivered_bar(&self, instrument: usize) -> Option<PushedBar> {
        self.shared.delivered.borrow().get(&instrument).copied()
    }

    fn poisoned(&self) -> bool {
        self.shared.poisoned.get()
    }

    /// The venue fills the books have not booked, oldest first per instrument: `(client order id,
    /// quantity, price, reason)`. Empty when every book agrees with its venue.
    fn unbooked_fills(&self) -> PyResult<Vec<(String, f64, f64, String)>> {
        let sh = &*self.shared;
        let mut out = Vec::new();
        let sids: Vec<StrategyId> = sh.strats.borrow().iter().map(|s| s.sid).collect();
        for sid in sids {
            for local in 0..sh.locals(sid) {
                sh.with_book(sid, |b, _| {
                    out.extend(
                        b.kernel(local)
                            .unbooked_fills()
                            .iter()
                            .map(|u| (u.client_id.clone(), u.qty, u.price, u.reason.clone())),
                    );
                    Ok(())
                })?;
            }
        }
        Ok(out)
    }

    /// Why each strategy was stopped for holding a sold option its margin model cannot bound (a
    /// hedge taken off with none on its way): `(strategy index, reason)`. Empty when none was.
    fn margin_faults(&self) -> PyResult<Vec<(usize, String)>> {
        let sh = &*self.shared;
        let mut out = Vec::new();
        let sids: Vec<StrategyId> = sh.strats.borrow().iter().map(|s| s.sid).collect();
        for sid in sids {
            sh.with_book(sid, |b, _| {
                if let Some(why) = b.unmarginable() {
                    out.push((sid.index(), why.to_string()));
                }
                Ok(())
            })?;
        }
        Ok(out)
    }

    /// The exits of hedge legs each strategy's book is still carrying out:
    /// `(strategy index, hedge instrument, hedge position id, sold legs sent, hedge close sent)`.
    fn pending_covers(&self) -> PyResult<Vec<PendingCover>> {
        let sh = &*self.shared;
        let mut out = Vec::new();
        let sids: Vec<StrategyId> = sh.strats.borrow().iter().map(|s| s.sid).collect();
        for sid in sids {
            sh.with_book(sid, |b, _| {
                out.extend(
                    b.pending_covers().into_iter().map(|(i, p, n, h)| (sid.index(), i, p, n, h)),
                );
                Ok(())
            })?;
        }
        Ok(out)
    }

    /// Adapter calls a replay answered "unavailable" for, since last asked.
    fn take_replay_errors(&self) -> u32 {
        self.shared.control.replay_errors.replace(0)
    }

    /// Check the execution epoch with the venue now; raises `fenced` when a
    /// newer epoch owns the account.
    fn heartbeat(&self) -> PyResult<()> {
        match self.shared.venue.borrow().clone() {
            Some(venue) => venue.heartbeat(),
            None => Ok(()),
        }
    }

    /// Send an order again under its first request (same client order id). For
    /// an order a reconciliation found missing at the venue.
    fn resend(&self, client_order_id: &str) -> PyResult<()> {
        let venue = self.shared.venue.borrow().clone().ok_or_else(|| {
            PyErr::from(Fail::new("invalid_execution", "this node has no venue to resend to"))
        })?;
        self.shared.jlog_now(jk::ANNOTATION, jsrc::HOST, Some(client_order_id), || {
            JV::map([("what", JV::str("resend"))])
        })?;
        venue.resend(client_order_id)
    }

    /// Register the orders a venue makes on its own, with the ids it was told
    /// (see `ExitRuleRequest.exit_client_ids`), so their reports are booked.
    /// `orders` are `(client_order_id, symbol_index, side, quantity)` of strategy
    /// `strategy`.
    fn register_venue_orders(
        &self,
        strategy: usize,
        orders: Vec<(String, usize, String, f64)>,
    ) -> PyResult<()> {
        let sh = &*self.shared;
        let venue = sh.venue.borrow().clone().ok_or_else(|| {
            PyErr::from(Fail::new(
                "invalid_execution",
                "this node has no venue to register orders at",
            ))
        })?;
        let sid = sh
            .strats
            .borrow()
            .iter()
            .find(|s| s.sid.index() == strategy)
            .map(|s| s.sid)
            .ok_or_else(|| PyKeyError::new_err(strategy))?;
        let ts = sh.control.now.get().0;
        for (id, symbol, side, quantity) in orders {
            let global = InstrumentId::new(symbol as u32);
            let local = sh.local_of(sid, global).ok_or_else(|| {
                PyErr::from(Fail::new(
                    "unknown_symbol",
                    "the strategy does not trade the symbol of an order the venue holds",
                ))
            })?;
            let idx = with_feed(sh, |feed| Ok(feed.last_ordinal(global)))?;
            let side = match side.as_str() {
                "buy" => crate::execution::orders::OrderSide::Buy,
                "sell" => crate::execution::orders::OrderSide::Sell,
                other => {
                    return Err(Fail::new(
                        "invalid_execution",
                        format!("side {other:?} is not buy or sell"),
                    )
                    .into())
                }
            };
            venue.register_external(sh, sid, local, &id, side, quantity, idx, ts)?;
        }
        Ok(())
    }

    /// Mark orders as having an outcome nobody knows: a reconciliation reports them until the
    /// venue is asked (`confirm`).
    fn mark_unconfirmed(&self, ids: Vec<String>) {
        self.shared.control.unconfirmed.borrow_mut().extend(ids);
    }

    /// End registered exit orders: the venue's exit rule was cancelled or replaced, so the
    /// orders it would have fired are cancelled in the book. See `register_venue_orders`.
    fn end_venue_orders(&self, strategy: usize, ids: Vec<String>) -> PyResult<()> {
        let sh = &*self.shared;
        let venue = sh.venue.borrow().clone().ok_or_else(|| {
            PyErr::from(Fail::new("invalid_execution", "this node has no venue to end orders at"))
        })?;
        let sid = sh
            .strats
            .borrow()
            .iter()
            .find(|s| s.sid.index() == strategy)
            .map(|s| s.sid)
            .ok_or_else(|| PyKeyError::new_err(strategy))?;
        let ts = sh.control.now.get().0;
        venue.end_external(sh, sid, &ids, ts)
    }

    /// Ids of the orders a venue-held exit rule would fire that are still waiting: the
    /// venue lists them only once the rule fires, so a reconciliation does not look for them.
    fn venue_held_orders(&self) -> PyResult<Vec<String>> {
        let sh = &*self.shared;
        match sh.venue.borrow().clone() {
            Some(venue) => venue.held_ids(sh),
            None => Ok(Vec::new()),
        }
    }

    /// The orders the node holds as working at the venue, as requests with their current
    /// terms: `(request, status, filled_qty, average_fill_price)`. A restarted adapter that
    /// has no record of them is taught from this (see `BrokerAdapter.restore_orders`).
    /// Orders the venue makes itself are not listed.
    fn working_requests(&self) -> PyResult<Vec<WorkingRequestRow>> {
        let sh = &*self.shared;
        let Some(venue) = sh.venue.borrow().clone() else { return Ok(Vec::new()) };
        let mut out = Vec::new();
        for (_, sid, local, order_id, request, external) in venue.sent_with_origin() {
            if external {
                continue;
            }
            let state = sh.with_book(StrategyId::new(sid as u32), |b, _| {
                Ok(b.kernel(local)
                    .order(order_id)
                    .map(|o| (o.status, o.qty, o.kind, o.filled_qty, o.avg_fill_price())))
            })?;
            let Some((status, qty, kind, filled, avg)) = state else { continue };
            if status.is_terminal() {
                continue;
            }
            let mut request = request;
            if let crate::execution::orders::QtySpec::Units(units) = qty {
                request.quantity = units;
            }
            request.kind = kind;
            out.push((
                crate::python::execution_bindings::PyOrderRequest(request),
                format!("{status:?}").to_lowercase(),
                filled,
                avg,
            ));
        }
        Ok(out)
    }

    /// Every order sent to the venue: `(client_order_id, strategy, symbol, side,
    /// quantity, filled_qty, status)`.
    fn venue_orders(&self) -> PyResult<Vec<VenueOrderRow>> {
        let sh = &*self.shared;
        let Some(venue) = sh.venue.borrow().clone() else { return Ok(Vec::new()) };
        let mut out = Vec::new();
        for (id, sid, local, order_id, request) in venue.sent() {
            let state = sh.with_book(StrategyId::new(sid as u32), |b, _| {
                Ok(b.kernel(local).order(order_id).map(|o| (o.status, o.filled_qty)))
            })?;
            let Some((status, filled)) = state else { continue };
            out.push((
                id,
                sid,
                request.symbol.clone(),
                request.side.as_str(),
                request.quantity,
                filled,
                format!("{status:?}").to_lowercase(),
            ));
        }
        Ok(out)
    }

    /// A newer execution epoch owns the account.
    #[getter]
    fn fenced(&self) -> bool {
        self.shared.control.fenced.get()
    }

    /// Mark the pushes that follow as history replayed to prime the
    /// strategies. A real venue is never sent an order made on history.
    #[pyo3(signature = (warmup, muted=false))]
    fn set_warmup(&self, warmup: bool, muted: bool) {
        self.shared.warmup.set(warmup);
        self.shared.warmup_muted.set(warmup && muted);
    }

    /// Refuse every order decided before `ns` (an event time, ns since the epoch) with the
    /// reason `warmup`; `None` lifts it.
    #[pyo3(signature = (ns=None))]
    fn set_trade_from(&self, ns: Option<i64>) {
        self.shared.trade_from.set(ns.unwrap_or(i64::MIN));
    }

    /// Set aside what every strategy has queued so far (its `on_start`), so
    /// muted warm-up history, which discards queued work, does not take an
    /// order the strategy made at start. `release_boot` queues it again.
    fn hold_boot(&self, py: Python<'_>) {
        let strats: Vec<Rc<Strat>> = self.shared.strats.borrow().clone();
        let mut boot = self.shared.boot_queue.borrow_mut();
        for st in strats {
            let (commands, intents) = {
                let mut q = st.queue.bind(py).borrow_mut();
                (q.queue.take_commands(), q.queue.take_intents())
            };
            boot.push((st, commands, intents));
        }
    }

    /// Queue again what `hold_boot` set aside, ahead of anything queued since.
    fn release_boot(&self, py: Python<'_>) {
        for (st, commands, intents) in self.shared.boot_queue.borrow_mut().drain(..) {
            let mut q = st.queue.bind(py).borrow_mut();
            q.queue.restore_commands(commands);
            q.queue.restore_intents(intents);
        }
    }

    /// Install the host's pre-submit order check: `check(order, ctx)` is
    /// called with an `OrderProposal` (built by `proposal`) for every typed
    /// order a strategy submits, before the engine or a venue sees it. `typed_only` says the
    /// check judges typed orders only, so intents are left alone instead of refused.
    #[pyo3(signature = (check, proposal, typed_only=false))]
    fn set_order_check(&self, check: Py<PyAny>, proposal: Py<PyAny>, typed_only: bool) {
        *self.shared.order_check.borrow_mut() = Some((check, proposal));
        self.shared.order_check_typed_only.set(typed_only);
    }

    /// Declare how long the engine takes to act on an event after it
    /// arrives (ns): the gap between a push's arrival and processing times.
    fn set_processing_latency(&self, latency_ns: i64) {
        self.shared.processing_latency_ns.set(latency_ns);
    }

    /// Hand the node what the venue said about an order. The report is
    /// applied at once, in the order pushes arrive, like market data;
    /// `client_order_id` is the id the order was sent under (the request's
    /// `client_order_id`). Returns `"applied"`, `"duplicate"`, `"late"` (nothing
    /// it says can matter), `"deferred"` (the order took the execution and the book could not
    /// book it yet, an early reduce-only fill for one: it is kept, journaled as `unbooked_fill`,
    /// and books itself when the book can) or `"unbooked"` (an execution on an order the node
    /// no longer holds: journaled, refused, never booked, reconcile the position).
    fn push_report(
        &self,
        py: Python<'_>,
        client_order_id: &str,
        report: PyRef<'_, PyExecutionReport>,
        arrival_ns: i64,
    ) -> PyResult<&'static str> {
        let before = self.shared.jcount();
        let result = (|| -> PyResult<&'static str> {
            let sh = &*self.shared;
            let mut guard = self.live_loop()?;
            self.drain_held(py, &mut guard)?;
            self.settle_push();
            let venue = sh.venue.borrow().clone().ok_or_else(|| {
                PyErr::from(Fail::new(
                    "invalid_execution",
                    "this node sends its orders in-process; there is no venue to report",
                ))
            })?;
            let report = &report.0;
            let ts = report_ts(report);
            // Every inbound report is written down before anything is decided
            // about it, so a report that is dropped, refused or unknown is
            // still on the record.
            sh.jlog(jk::REPORT, ts, arrival_ns, jsrc::VENUE, Some(client_order_id), || {
                report_value(report)
            })?;
            if venue.is_dropped(client_order_id) {
                // The order ended long ago and its entries are gone. What the
                // venue says about it can no longer change the order or the
                // book; an execution (a fill, a correction, a bust) is money,
                // so it is put on the record as one the book did not take.
                return Ok(match report {
                    crate::execution::orders::ExecutionReport::Fill { .. }
                    | crate::execution::orders::ExecutionReport::Correction { .. }
                    | crate::execution::orders::ExecutionReport::Bust { .. } => {
                        sh.jlog_now(
                            jk::UNBOOKED_FILL,
                            jsrc::ENGINE,
                            Some(client_order_id),
                            || {
                                let mut v = report_value(report);
                                if let JV::Map(m) = &mut v {
                                    m.insert("reason".to_string(), JV::str("order_evicted"));
                                }
                                v
                            },
                        )?;
                        "unbooked"
                    }
                    _ => "late",
                });
            }
            let (sid, local) = venue.lookup(client_order_id).ok_or_else(|| {
                PyErr::from(
                    Fail::new(
                        "unknown_client_order_id",
                        format!("no order was sent under {client_order_id:?}"),
                    )
                    .param("client_order_id")
                    .value(client_order_id),
                )
            })?;
            let st = sh
                .strats
                .borrow()
                .iter()
                .find(|s| s.sid.index() == sid)
                .cloned()
                .expect("an order's strategy is registered");
            let g = sh.global_of(st.sid, local)?;
            // The venue spoke about the order: its request did arrive.
            sh.control.unconfirmed.borrow_mut().remove(client_order_id);
            let idx = with_feed(sh, |feed| Ok(feed.last_ordinal(g)))?;
            sh.times.set(Some(Times {
                event: ts,
                arrival: arrival_ns,
                processing: arrival_ns.saturating_add(sh.processing_latency_ns.get()),
            }));
            let kept_before =
                sh.with_book(st.sid, |b, _| Ok(b.kernel(local).unbooked_fills().len()))?;
            let outcome = venue.deliver(py, sh, &st, client_order_id, report, idx, ts);
            let settled = outcome.and_then(|outcome| {
                if has_commands(py, &st) || has_intents(py, &st) {
                    settle(py, sh, &st, g, local, idx, ts)?;
                }
                Ok(outcome)
            });
            sh.times.set(None);
            if let (Ok(outcome), true) = (
                &settled,
                matches!(
                    report,
                    crate::execution::orders::ExecutionReport::Correction { .. }
                        | crate::execution::orders::ExecutionReport::Bust { .. }
                ),
            ) {
                // The book took an execution back (and, for a correction, booked
                // it again as restated). Say so, with what the order now counts.
                let outcome = format!("{outcome:?}").to_lowercase();
                sh.jlog_now(jk::RESTATEMENT, jsrc::ENGINE, Some(client_order_id), || {
                    let mut v = report_value(report);
                    if let JV::Map(m) = &mut v {
                        m.insert("outcome".to_string(), JV::str(outcome));
                    }
                    v
                })?;
            }
            let settled = settled?;
            // An execution the order took and the book could not book yet is kept (journaled as
            // `unbooked_fill` with its reason; `unbooked_fills()` lists them) and tried again
            // after every booking: it is deferred, not applied, and the position at the venue and
            // the book disagree until it books or is reconciled.
            let kept_after =
                sh.with_book(st.sid, |b, _| Ok(b.kernel(local).unbooked_fills().len()))?;
            if kept_after > kept_before {
                return Ok("deferred");
            }
            Ok(match settled {
                crate::execution::orders::ReportOutcome::Applied => "applied",
                crate::execution::orders::ReportOutcome::Duplicate => "duplicate",
                crate::execution::orders::ReportOutcome::Late => "late",
            })
        })();
        self.shared.jnote(py, "report", before, &result);
        result
    }

    /// Run the whole tape.
    fn run(&self, py: Python<'_>) -> PyResult<()> {
        let sh = &*self.shared;
        let store = match &*sh.inner.borrow() {
            Inner::Running(c) => c.store().clone(),
            _ => return Err(not_ready("finalize() before run()")),
        };
        let mut ls = LoopState::new(sh)?;
        // Only instruments someone subscribes to: the others have no readers,
        // so their events would change nothing.
        let mut cursor = match &*sh.inner.borrow() {
            Inner::Running(c) => MergeCursor::filtered(&store, |id| !c.subscribers(id).is_empty()),
            _ => return Err(finished()),
        };
        let mut n: u64 = 0;
        let mut group: Vec<GroupEvent> = Vec::new();
        while let Some(key) = cursor.peek_group() {
            let Some((event, payload)) = RuntimeCore::pull(&mut cursor).map_err(runtime_err)?
            else {
                break;
            };
            if cursor.peek_group() == Some(key) {
                // Events that share a timestamp and a kind are one group: all
                // are applied before any is decided or matched.
                group.clear();
                group.push(GroupEvent { event, payload, times: None });
                while cursor.peek_group() == Some(key) {
                    let Some((event, payload)) =
                        RuntimeCore::pull(&mut cursor).map_err(runtime_err)?
                    else {
                        break;
                    };
                    group.push(GroupEvent { event, payload, times: None });
                }
                n += group.len() as u64;
                run_group(py, sh, &mut ls, &group)?;
            } else {
                // A group of one: the event is applied, decided and matched.
                run_one(py, sh, &mut ls, GroupEvent { event, payload, times: None })?;
                n += 1;
            }
            if n >= SIGNAL_EVERY {
                n = 0;
                py.check_signals()?;
            }
        }
        Ok(())
    }

    /// Keep only the most recent `window` samples of every book's curves.
    /// Call before adding strategies. See `PortfolioBook::set_curve_window`.
    fn set_curve_window(&self, window: Option<usize>) {
        self.shared.curve_window.set(window);
    }

    /// Adopt a position the account already holds, on a strategy's local
    /// instrument, before the first event.
    #[pyo3(signature = (sid, local, timestamp_ns, price, quantity, margin_minor=None, cost_minor=None))]
    #[allow(clippy::too_many_arguments)]
    fn adopt_position(
        &self,
        sid: usize,
        local: usize,
        timestamp_ns: i64,
        price: f64,
        quantity: f64,
        margin_minor: Option<i128>,
        cost_minor: Option<i128>,
    ) -> PyResult<u64> {
        let sid = StrategyId::new(sid as u32);
        // An adopted position is the account's, not the engine's own doing:
        // it is on the record before the book takes it.
        let (ev, arr) = self.shared.control.now.get();
        self.shared.jlog(jk::ADOPTION, ev, arr, jsrc::HOST, None, || {
            let minor = |m: Option<i128>| match m {
                None => JV::Null,
                Some(m) => i64::try_from(m).map_or_else(|_| JV::str(m.to_string()), JV::Int),
            };
            JV::map([
                ("strategy", JV::Int(sid.index() as i64)),
                ("local", JV::Int(local as i64)),
                ("ts_ns", JV::Int(timestamp_ns)),
                ("price", JV::Float(price)),
                ("quantity", JV::Float(quantity)),
                ("margin_minor", minor(margin_minor)),
                ("cost_minor", minor(cost_minor)),
            ])
        })?;
        let extras = crate::portfolio::Adoption { margin_minor, cost_minor };
        let adopted = match &mut *self.shared.inner.borrow_mut() {
            Inner::Building(b) => {
                b.adopt_position_with(sid, local, timestamp_ns, price, quantity, extras)
            }
            Inner::Running(c) => {
                c.adopt_position_with(sid, local, timestamp_ns, price, quantity, extras)
            }
            Inner::Gone => return Err(finished()),
        };
        adopted.map_err(|e| {
            PyErr::from(
                Fail::new("adoption_refused", e)
                    .op("adopt_position")
                    .param("quantity")
                    .value(quantity),
            )
        })
    }

    /// Feed one tick row to a live node. Returns the events it appended
    /// (0..=2). See `LiveNode.push_tick`.
    #[pyo3(signature = (
        instrument, timestamp, ltp, bid=0.0, ask=0.0, buy_qty_delta=0.0, sell_qty_delta=0.0,
        ltq=0.0, bid_qty=0.0, ask_qty=0.0, oi=0.0, arrival_ns=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn push_tick(
        &self,
        py: Python<'_>,
        instrument: usize,
        timestamp: i64,
        ltp: f64,
        bid: f64,
        ask: f64,
        buy_qty_delta: f64,
        sell_qty_delta: f64,
        ltq: f64,
        bid_qty: f64,
        ask_qty: f64,
        oi: f64,
        arrival_ns: Option<i64>,
    ) -> PyResult<usize> {
        let before = self.shared.jcount();
        let result = (|| -> PyResult<usize> {
            let sh = &*self.shared;
            let mut guard = self.live_loop()?;
            let g = self.instrument(instrument)?;
            let ts = timestamp;
            self.stamp(ts, arrival_ns);
            with_feed(sh, |feed| feed.check_timestamp(g, ts).map_err(runtime_err))?;
            sh.jlog(jk::TICK, ts, arrival_ns.unwrap_or(ts), jsrc::FEED, None, || {
                JV::map([
                    ("symbol", JV::str(sh.symbol_text(g.index()))),
                    ("ltp", JV::Float(ltp)),
                    ("bid", JV::Float(bid)),
                    ("ask", JV::Float(ask)),
                    ("buy_qty_delta", JV::Float(buy_qty_delta)),
                    ("sell_qty_delta", JV::Float(sell_qty_delta)),
                    ("ltq", JV::Float(ltq)),
                    ("bid_qty", JV::Float(bid_qty)),
                    ("ask_qty", JV::Float(ask_qty)),
                    ("oi", JV::Float(oi)),
                ])
            })?;
            let trade = ltp > 0.0;
            let quote = bid > 0.0 && ask > 0.0;
            if !trade && !quote {
                // A row the book refuses still proves the market reached its
                // instant: what is due on this symbol's clock fires now.
                drop(guard);
                self.mark_time(py, g, ts)?;
                return Ok(0);
            }
            let row = TickRow {
                timestamp: ts,
                ltp,
                bid,
                ask,
                buy_qty_delta,
                sell_qty_delta,
                ltq,
                bid_qty,
                ask_qty,
                oi,
            };
            let row = store_row(sh, g, RingKind::Ticks, |feed| feed.push_tick(g, row))?;
            let mut appended = 0;
            if trade {
                // The exchange's last traded quantity when supplied, else the
                // flow-delta proxy: the rule `tick_data_to_events` applies.
                let size = if ltq > 0.0 { ltq } else { buy_qty_delta.abs() + sell_qty_delta.abs() };
                let payload = Pushed::Trade(TradeTick {
                    timestamp: ts,
                    price: ltp,
                    size,
                    signed_size: buy_qty_delta.abs() - sell_qty_delta.abs(),
                    oi,
                });
                self.deliver(py, &mut guard, g, EventKind::Trade, ts, row, payload)?;
                appended += 1;
            }
            if quote {
                let payload = Pushed::Quote(QuoteTick {
                    timestamp: ts,
                    bid,
                    ask,
                    bid_size: TickData::displayed(bid_qty),
                    ask_size: TickData::displayed(ask_qty),
                });
                self.deliver(py, &mut guard, g, EventKind::Quote, ts, row, payload)?;
                appended += 1;
            }
            self.settle_push();
            Ok(appended)
        })();
        self.shared.jnote(py, "tick", before, &result);
        result
    }

    /// Feed one closed bar to a live node; `timestamp` is as the caller
    /// stamps bars, and the symbol's offset moves it to the close.
    #[pyo3(signature = (instrument, timestamp, open, high, low, close, volume, arrival_ns=None))]
    #[allow(clippy::too_many_arguments)]
    fn push_bar(
        &self,
        py: Python<'_>,
        instrument: usize,
        timestamp: i64,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
        arrival_ns: Option<i64>,
    ) -> PyResult<()> {
        let before = self.shared.jcount();
        let result = (|| -> PyResult<()> {
            let sh = &*self.shared;
            let mut guard = self.live_loop()?;
            let g = self.instrument(instrument)?;
            let close_ts = timestamp + sh.offset_of(g.index());
            self.stamp(close_ts, arrival_ns);
            with_feed(sh, |feed| feed.check_timestamp(g, close_ts).map_err(runtime_err))?;
            sh.jlog(jk::BAR, close_ts, arrival_ns.unwrap_or(close_ts), jsrc::FEED, None, || {
                JV::map([
                    ("symbol", JV::str(sh.symbol_text(g.index()))),
                    ("timestamp", JV::Int(timestamp)),
                    ("open", JV::Float(open)),
                    ("high", JV::Float(high)),
                    ("low", JV::Float(low)),
                    ("close", JV::Float(close)),
                    ("volume", JV::Float(volume)),
                    ("warmup", JV::Bool(sh.warmup.get())),
                ])
            })?;
            let stamped = KernelBar { timestamp, open, high, low, close, volume };
            let event =
                store_row(sh, g, RingKind::Bars, |feed| feed.push_bar(g, close_ts, stamped))?;
            sh.delivered
                .borrow_mut()
                .insert(g.index(), (timestamp, open, high, low, close, volume));
            let bar = KernelBar { timestamp: close_ts, ..stamped };
            self.deliver_event(py, &mut guard, event, Pushed::Bar(bar))?;
            self.settle_push();
            Ok(())
        })();
        self.shared.jnote(py, "bar", before, &result);
        result
    }

    /// Feed one depth snapshot to a live node: `(price, size)` lists, best
    /// level first.
    #[pyo3(signature = (instrument, timestamp, bids, asks, arrival_ns=None))]
    fn push_depth(
        &self,
        py: Python<'_>,
        instrument: usize,
        timestamp: i64,
        bids: Vec<(f64, f64)>,
        asks: Vec<(f64, f64)>,
        arrival_ns: Option<i64>,
    ) -> PyResult<()> {
        let before = self.shared.jcount();
        let result = (|| -> PyResult<()> {
            let sh = &*self.shared;
            let mut guard = self.live_loop()?;
            let g = self.instrument(instrument)?;
            self.stamp(timestamp, arrival_ns);
            with_feed(sh, |feed| feed.check_timestamp(g, timestamp).map_err(runtime_err))?;
            sh.jlog(
                jk::DEPTH,
                timestamp,
                arrival_ns.unwrap_or(timestamp),
                jsrc::FEED,
                None,
                || {
                    let side = |levels: &[(f64, f64)]| {
                        JV::List(
                            levels
                                .iter()
                                .map(|&(p, q)| JV::List(vec![JV::Float(p), JV::Float(q)]))
                                .collect(),
                        )
                    };
                    JV::map([
                        ("symbol", JV::str(sh.symbol_text(g.index()))),
                        ("bids", side(&bids)),
                        ("asks", side(&asks)),
                    ])
                },
            )?;
            let levels = |levels: &[(f64, f64)]| -> Vec<BookLevel> {
                levels
                    .iter()
                    .filter(|(price, _)| *price > 0.0)
                    .map(|&(price, size)| BookLevel { price, size })
                    .collect()
            };
            let snapshot = DepthTick::from_levels(timestamp, &levels(&bids), &levels(&asks));
            let event = store_row(sh, g, RingKind::Depth, |feed| feed.push_depth(g, snapshot))?;
            self.deliver_event(py, &mut guard, event, Pushed::Depth(snapshot))?;
            self.settle_push();
            Ok(())
        })();
        self.shared.jnote(py, "depth", before, &result);
        result
    }

    /// Open a timestamp group. The pushes that follow are held; those that
    /// share a timestamp and a kind are delivered together by `flush()` (or
    /// when a push stamped later, `advance_time`, a report or `finish`
    /// completes them): every event is applied before any is decided, and
    /// every decision is taken before any is matched. `timestamp` is the
    /// group's, if the caller knows it; it is only recorded.
    ///
    /// `record=False` keeps the boundary off the journal: the node's own
    /// warm-up replay, which a restore re-runs from the same bars.
    #[pyo3(signature = (timestamp=None, record=true))]
    fn begin_group(&self, py: Python<'_>, timestamp: Option<i64>, record: bool) -> PyResult<()> {
        let before = self.shared.jcount();
        let result = (|| -> PyResult<()> {
            let sh = &*self.shared;
            let _guard = self.live_loop()?;
            if sh.bracket.get() {
                return Err(Fail::new(
                    "group_open",
                    "a timestamp group is already open: flush() it before opening another",
                )
                .into());
            }
            let (ev, arr) = sh.control.now.get();
            if record {
                sh.jlog(jk::GROUP_BEGIN, ev, arr, jsrc::FEED, None, || {
                    JV::map([("timestamp", timestamp.map_or(JV::Null, JV::Int))])
                })?;
            }
            sh.bracket.set(true);
            sh.bracket_events.set(0);
            Ok(())
        })();
        self.shared.jnote(py, "group_begin", before, &result);
        result
    }

    /// Whether a timestamp group is open (`begin_group` without `flush`).
    #[getter]
    fn group_open(&self) -> bool {
        self.shared.bracket.get()
    }

    /// What an open group holds, undelivered: `(instrument index, event time)` for each push.
    /// A node restored from a journal that ends inside a group reports the pushes it replayed,
    /// so the host sends the rest and flushes. Empty when no group is open.
    fn held_pushes(&self) -> Vec<(usize, i64)> {
        self.shared.held.borrow().iter().map(|h| (h.event.instrument.index(), h.event.ts)).collect()
    }

    /// Refuse a bar `push_bar` would refuse for its instrument or time (a retired instrument,
    /// a timestamp older than the instrument's latest), changing nothing. A host calls it for
    /// every bar of a group before opening it.
    fn check_bar(&self, instrument: usize, timestamp: i64) -> PyResult<()> {
        let sh = &*self.shared;
        let g = self.instrument(instrument)?;
        let close_ts = timestamp + sh.offset_of(g.index());
        with_feed(sh, |feed| feed.check_timestamp(g, close_ts).map_err(runtime_err))
    }

    /// Close the open timestamp group and deliver what it holds. Nothing is
    /// open: nothing happens.
    #[pyo3(signature = (record=true))]
    fn flush(&self, py: Python<'_>, record: bool) -> PyResult<()> {
        let before = self.shared.jcount();
        let result = (|| -> PyResult<()> {
            let sh = &*self.shared;
            let mut guard = self.live_loop()?;
            if !sh.bracket.get() {
                return Ok(());
            }
            let (ev, arr) = sh.control.now.get();
            let events = sh.bracket_events.get();
            // Written before the engine acts on the group, like any input.
            if record {
                sh.jlog(jk::GROUP_END, ev, arr, jsrc::FEED, None, || {
                    JV::map([("events", JV::Int(events as i64))])
                })?;
            }
            sh.bracket.set(false);
            self.drain_held(py, &mut guard)
        })();
        self.shared.jnote(py, "group_end", before, &result);
        result
    }

    /// Move every strategy's clocks to `timestamp`, symbol by symbol, firing
    /// the alerts and timers that are due. No data is involved.
    fn advance_time(&self, py: Python<'_>, timestamp: i64) -> PyResult<()> {
        let before = self.shared.jcount();
        let result = (|| -> PyResult<()> {
            let sh = &*self.shared;
            let mut guard = self.live_loop()?;
            self.drain_held(py, &mut guard)?;
            self.settle_push();
            sh.jlog(jk::TIME, timestamp, timestamp, jsrc::FEED, None, || {
                JV::map([("timestamp", JV::Int(timestamp))])
            })?;
            let strats: Vec<Rc<Strat>> = sh.strats.borrow().clone();
            for st in &strats {
                let locals = match &*sh.inner.borrow() {
                    Inner::Running(c) => c.instruments_of(st.sid),
                    _ => return Err(finished()),
                };
                for local in 0..locals {
                    let g = sh.global_of(st.sid, local)?;
                    time_only(py, sh, st, g, timestamp)?;
                }
            }
            Ok(())
        })();
        self.shared.jnote(py, "time", before, &result);
        result
    }

    /// The result of a strategy's run: liquidates its book.
    fn finish(&self, py: Python<'_>, sid: usize) -> PyResult<PyPortfolioResult> {
        self.flush_held(py)?;
        let sid = StrategyId::new(sid as u32);
        let mut outcome = match &mut *self.shared.inner.borrow_mut() {
            Inner::Running(core) => core.finish(sid).map_err(runtime_err)?,
            _ => return Err(not_ready("finalize() before finish()")),
        };
        if !self.shared.keep_curves {
            let last = outcome.result.equity_curve.last().copied();
            outcome.result.equity_curve = last.into_iter().collect();
            outcome.result.drawdown_curve = Vec::new();
            outcome.result.returns = Vec::new();
        }
        Ok(convert_outcome(outcome))
    }

    /// Drop the Python objects the runtime holds (strategies, contexts).
    /// Call when the run is over; the runtime is unusable afterwards.
    fn close(&self) {
        self.shared.plans.borrow_mut().clear();
        self.shared.strats.borrow_mut().clear();
        *self.shared.inner.borrow_mut() = Inner::Gone;
    }

    /// Heap bytes by category; see `MemoryReport`. Python objects are not
    /// counted.
    /// Add an instrument to a running live node; see `LiveNode.add_instrument`. `strategies` are
    /// strategy ids in ascending order; `history` sizes the instrument's rings (raised to the
    /// longest indicator lookback); `offset` moves a pushed bar's stamp to its close. Returns the
    /// instrument's index, which is what the push methods take.
    #[pyo3(signature = (symbol, spec, direction, strategies, history=None, offset=0))]
    #[allow(clippy::too_many_arguments)]
    fn add_instrument(
        &self,
        py: Python<'_>,
        symbol: String,
        spec: Option<PyRef<'_, PyInstrumentSpec>>,
        direction: i32,
        strategies: Vec<usize>,
        history: Option<usize>,
        offset: i64,
    ) -> PyResult<usize> {
        let before = self.shared.jcount();
        let result =
            self.add_instrument_now(py, symbol, spec, direction, strategies, history, offset);
        self.shared.jnote(py, "instrument_add", before, &result);
        result
    }

    /// Retire an instrument of a running live node; see `LiveNode.retire_instrument`.
    fn retire_instrument(&self, py: Python<'_>, symbol: &str) -> PyResult<()> {
        let before = self.shared.jcount();
        let result = self.retire_instrument_now(py, symbol);
        self.shared.jnote(py, "instrument_retire", before, &result);
        result
    }

    fn memory_report<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let inner = self.shared.inner.borrow();
        let Inner::Running(core) = &*inner else {
            return Err(not_ready("the runtime is not built"));
        };
        let r: MemoryReport = core.memory_report();
        let d = PyDict::new(py);
        d.set_item("strategies", r.strategies)?;
        d.set_item("shared_store", r.shared_store)?;
        d.set_item("aggregators", r.aggregators)?;
        d.set_item("indicators", r.indicators)?;
        d.set_item("index", r.index)?;
        d.set_item("atr", r.atr)?;
        d.set_item("engine_per_strategy", r.engine_per_strategy)?;
        d.set_item("portfolio_per_strategy", r.portfolio_per_strategy)?;
        d.set_item("curves", r.curves)?;
        d.set_item("total", r.total())?;
        d.set_item("engine_bytes_per_strategy", r.engine_bytes_per_strategy())?;
        d.set_item("portfolio_bytes_per_strategy", r.portfolio_bytes_per_strategy())?;
        d.set_item("curve_bytes_per_strategy", r.curve_bytes_per_strategy())?;
        d.set_item("aggregator_count", core.hub().aggregators())?;
        d.set_item("indicator_cells", core.bank().cells())?;
        d.set_item("indicator_registrations", core.bank().registrations())?;
        d.set_item("events", core.events())?;
        // A live node's rings: fixed once allocated, whatever is pushed.
        let rings = self.shared.feed.borrow().as_ref().map_or(0, HeapSize::heap_bytes);
        d.set_item("live_rings", rings)?;
        d.set_item("total", r.total() + rings)?;
        Ok(d)
    }
}

impl PyRuntime {
    /// The loop state of a live node, borrowed for one push. A push from
    /// inside a hook would need it again, and is refused.
    fn live_loop(&self) -> PyResult<LiveGuard<'_>> {
        if self.shared.poisoned.get() {
            return Err(poisoned());
        }
        if self.shared.live_history.is_none() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "internal error: a live-node call reached a runtime that is not a live node",
            ));
        }
        let guard = self
            .live_state
            .try_borrow_mut()
            .map_err(|_| busy("the runtime is busy: a push was made from a hook"))?;
        if guard.is_none() {
            return Err(not_ready("finalize() before pushing"));
        }
        Ok(LiveGuard { state: guard, poisoned: &self.shared.poisoned })
    }

    /// Call the optional capability `method` of the node's own execution adapter, if it has it.
    fn tell_adapter(&self, py: Python<'_>, method: &str, args: Bound<'_, PyTuple>) -> PyResult<()> {
        if let Some(adapter) = self.shared.adapter_obj.borrow().as_ref() {
            if let Some(m) = adapter.bind(py).getattr_opt(method)?.filter(|m| m.is_callable()) {
                m.call1(args)?;
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn add_instrument_now(
        &self,
        py: Python<'_>,
        symbol: String,
        spec: Option<PyRef<'_, PyInstrumentSpec>>,
        direction: i32,
        strategies: Vec<usize>,
        history: Option<usize>,
        offset: i64,
    ) -> PyResult<usize> {
        let sh = &*self.shared;
        let mut guard = self.live_loop()?;
        self.drain_held(py, &mut guard)?;
        let known = sh.sym_ids.borrow().get(&symbol).copied();
        let revive = known.filter(|g| sh.retired.get()[g.index()]);
        if symbol.is_empty() || (known.is_some() && revive.is_none()) {
            return Err(Fail::new(
                "duplicate_symbols",
                format!("{symbol:?} is empty or already an instrument of this node"),
            )
            .param("symbol")
            .value(symbol.as_str())
            .into());
        }
        if history == Some(0) {
            return Err(Fail::new("invalid_history", "history must be a positive number of rows")
                .param("history")
                .value(0_i64)
                .into());
        }
        let spec_py = spec.as_ref().map(|s| Py::new(py, PyInstrumentSpec::clone(s))).transpose()?;
        let (direction, policy, spec) =
            parse_instrument_args(direction, spec.as_deref(), &sh.oms_type, true)?;
        let strats: Vec<Rc<Strat>> = sh.strats.borrow().clone();
        if strategies.is_empty() || strategies.windows(2).any(|w| w[0] >= w[1]) {
            return Err(Fail::new(
                "invalid_strategies",
                "strategies must name at least one strategy, each once, in ascending order",
            )
            .param("strategies")
            .into());
        }
        let mut chosen: Vec<Rc<Strat>> = Vec::with_capacity(strategies.len());
        for &k in &strategies {
            let Some(st) = strats.get(k) else {
                return Err(Fail::new(
                    "invalid_strategies",
                    format!("strategy {k} does not exist; the node has {}", strats.len()),
                )
                .param("strategies")
                .value(k as i64)
                .into());
            };
            chosen.push(st.clone());
        }
        let sids: Vec<StrategyId> = chosen.iter().map(|s| s.sid).collect();
        if let Some(g) = revive {
            let holders = match &*sh.inner.borrow() {
                Inner::Running(core) => core.holders(g),
                _ => return Err(finished()),
            };
            if let Some(bad) = sids.iter().find(|s| !holders.contains(s)) {
                return Err(Fail::new(
                    "invalid_strategies",
                    format!(
                        "strategy {} never held {symbol:?}; a retired symbol comes back only for \
                         strategies that had it",
                        bad.index()
                    ),
                )
                .param("strategies")
                .value(bad.index() as i64)
                .into());
            }
        }

        // The record is written before the engine acts on it.
        // A setting the run was given (a fee schedule that picks its row by venue, a
        // squareoff_time that needs a zone) must be carriable out on the new instrument too.
        super::strategy_bindings::check_market(&sh.config, &symbol, spec.as_ref())?;
        let ts = sh.control.now.get().0;
        let spec_json = match &spec {
            Some(s) => Some(serde_json::to_string(s).map_err(|e| {
                Fail::new("invalid_parameter", e.to_string()).param("spec").value("spec")
            })?),
            None => None,
        };
        sh.jlog(jk::INSTRUMENT_ADD, ts, ts, jsrc::HOST, None, || {
            JV::map([
                ("symbol", JV::str(symbol.as_str())),
                ("spec", JV::opt_str(spec_json.as_deref())),
                ("direction", JV::Int(direction as i8 as i64)),
                ("strategies", JV::List(strategies.iter().map(|&k| JV::Int(k as i64)).collect())),
                ("history", history.map_or(JV::Null, |h| JV::Int(h as i64))),
                ("offset", JV::Int(offset)),
                ("revive", JV::Bool(revive.is_some())),
            ])
        })?;
        // The venue learns the contract before any order for it can reach it. A refusal leaves
        // the engine as it was.
        self.tell_adapter(py, "add_instrument", (symbol.as_str(), spec_py).into_pyobject(py)?)?;

        let primary = sh.primary.get().map(|bars| (bars, sh.tz_offset_ns));
        let capacity = history.map(|h| h.max(sh.lookback.get()));
        let session = spec.as_ref().and_then(|s| s.session.clone());
        let mut mask = 0u32;
        for st in &chosen {
            mask |= st.mask;
        }
        if let Some(g) = revive {
            match &mut *sh.inner.borrow_mut() {
                Inner::Running(core) => core
                    .revive_instrument(g, &sids, primary, session.as_ref())
                    .map_err(runtime_err)?,
                _ => return Err(finished()),
            }
            with_feed(sh, |feed| {
                feed.revive_instrument(g, capacity);
                Ok(())
            })?;
            sh.retired.set(&guard, g.index(), false);
            guard.as_mut().expect("checked by live_loop").wanted[g.index()] = mask;
            return Ok(g.index());
        }

        let params =
            InstrumentParams { direction, spec: spec.clone(), config: None, atr_period: 0 };
        let g = match &mut *sh.inner.borrow_mut() {
            Inner::Running(core) => core
                .add_instrument(&symbol, &params, &sids, primary, sh.venue_on.get())
                .map_err(runtime_err)?,
            _ => return Err(finished()),
        };
        let gi = g.index();
        debug_assert_eq!(gi, sh.symbols.get().len());
        sh.symbols.push(&guard, PyString::intern(py, &symbol).unbind());
        sh.sym_ids.borrow_mut().insert(symbol.clone(), g);
        sh.has_ticks.push(&guard, true);
        sh.arrays.borrow_mut().push(None);
        sh.tick_arrays.borrow_mut().push(None);
        sh.offsets.push(&guard, offset);
        sh.series.borrow_mut().push(None);
        sh.retired.push(&guard, false);
        with_feed(sh, |feed| {
            feed.add_instrument(symbol.clone(), capacity);
            Ok(())
        })?;
        let symbol_obj = sh.symbol(py, gi);
        let clock_cls = py.import("raptorbt.strategy.clock")?.getattr("Clock")?;
        let mut natives = false;
        for st in &chosen {
            st.ctx.bind(py).borrow().prices.push(&guard, Px::default());
            st.ctx.bind(py).borrow().books.borrow_mut().push(None);
            let clock = st.late.clock.bind(py).call_method0("clone_schedule")?;
            let mut t = st.tables.borrow_mut();
            t.clocks.push(ClockSlot {
                alerts: clock.getattr("_alerts")?.cast_into::<PyDict>()?.unbind(),
                timers: clock.getattr("_timers")?.cast_into::<PyDict>()?.unbind(),
                plain: clock.get_type().is(&clock_cls),
                clock: clock.unbind(),
            });
            let (privates, primary_natives) = split_indicators(
                py,
                st.late.indicators.iter().filter(|(sub, _)| sub.is_none()).map(|(_, obj)| obj),
            );
            t.primary_private.push(privates);
            natives |= !primary_natives.is_empty();
            t.primary_native.push(primary_natives);
            t.primary_reveal.push(None);
            for (sub, (step, unit, brick)) in st.late.subscriptions.iter().enumerate() {
                let key =
                    AggKey::new(g, parse_spec(*step, unit)?, *brick, sh.tz_offset_ns, sh.stamp)
                        .map_err(runtime_err)?;
                let (_, stream) = match &mut *sh.inner.borrow_mut() {
                    Inner::Running(core) => core
                        .subscribe_aggregation(st.sid, key, session.as_ref())
                        .map_err(runtime_err)?,
                    _ => return Err(finished()),
                };
                let kwargs = PyDict::new(py);
                kwargs.set_item("symbol", &symbol_obj)?;
                let (private, native) = split_indicators(
                    py,
                    st.late.indicators.iter().filter(|(s, _)| *s == Some(sub)).map(|(_, obj)| obj),
                );
                natives |= !native.is_empty();
                debug_assert_eq!(stream as usize, t.streams.len());
                t.streams.push(StreamInfo {
                    sub,
                    step: *step,
                    unit: PyString::new(py, unit).unbind(),
                    kwargs: kwargs.unbind(),
                    private,
                    native,
                    reveal: None,
                });
            }
        }
        let ls = guard.as_mut().expect("checked by live_loop");
        ls.wanted.push(mask);
        // The group loop indexes `rank` and `seen` by instrument: they grow with the node. A
        // new name can fall between old ones, so the ranks are taken again (their order is kept).
        ls.seen.push(0);
        ls.rank = symbol_ranks(sh);
        ls.natives |= natives;
        if let Some(route) = sh.venue.borrow().clone() {
            for sid in &sids {
                route.add_instrument(sid.index(), &symbol, direction, spec.as_ref(), policy)?;
            }
        }
        Ok(gi)
    }

    fn retire_instrument_now(&self, py: Python<'_>, symbol: &str) -> PyResult<()> {
        let sh = &*self.shared;
        let mut guard = self.live_loop()?;
        self.drain_held(py, &mut guard)?;
        let Some(g) = sh.sym_ids.borrow().get(symbol).copied() else {
            return Err(Fail::new(
                "unknown_symbol",
                format!("{symbol:?} is not an instrument of this node"),
            )
            .param("symbol")
            .value(symbol)
            .valid(sh.symbols.get().iter().map(|s| s.to_string()))
            .into());
        };
        if sh.retired.get()[g.index()] {
            // `LiveNode.retire_instrument` refuses a retired symbol as an unknown one first.
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "internal error: {symbol:?} is already retired"
            )));
        }
        let strats: Vec<Rc<Strat>> = sh.strats.borrow().clone();
        let mut holding = Vec::new();
        let subscribers = match &*sh.inner.borrow() {
            Inner::Running(core) => {
                for (sid, positions, orders) in core.instrument_in_use(g) {
                    holding.push(format!(
                        "strategy {}: {positions} open positions, {orders} working orders",
                        sid.index()
                    ));
                }
                core.subscribers(g).to_vec()
            }
            _ => return Err(finished()),
        };
        let venue = sh.venue.borrow().clone();
        for &sid in &subscribers {
            let local = match &*sh.inner.borrow() {
                Inner::Running(core) => core.local_of(sid, g).expect("subscribed"),
                _ => return Err(finished()),
            };
            let st = &strats[sid.index()];
            if venue.as_ref().is_some_and(|v| v.watches(sid.index(), local)) {
                holding.push(format!(
                    "strategy {}: a protective level the engine watches",
                    sid.index()
                ));
            }
            if let Route::Native(state) = &st.route {
                if state.borrow().has_held_for(local) {
                    holding
                        .push(format!("strategy {}: intents held for its next step", sid.index()));
                }
            }
        }
        if !holding.is_empty() {
            return Err(Fail::new(
                "instrument_in_use",
                format!(
                    "{symbol:?} cannot be retired while it is in use ({}); close the positions and \
                     cancel the orders first",
                    holding.join("; ")
                ),
            )
            .param("symbol")
            .value(symbol)
            .into());
        }
        let ts = sh.control.now.get().0;
        sh.jlog(jk::INSTRUMENT_RETIRE, ts, ts, jsrc::HOST, None, || {
            JV::map([("symbol", JV::str(symbol))])
        })?;
        self.tell_adapter(py, "retire_instrument", (symbol,).into_pyobject(py)?)?;
        if let Inner::Running(core) = &mut *sh.inner.borrow_mut() {
            core.retire_instrument(g).map_err(runtime_err)?;
        }
        with_feed(sh, |feed| {
            feed.retire_instrument(g);
            Ok(())
        })?;
        sh.series.borrow_mut()[g.index()] = None;
        sh.retired.set(&guard, g.index(), true);
        guard.as_mut().expect("checked by live_loop").wanted[g.index()] = 0;
        Ok(())
    }

    fn instrument(&self, index: usize) -> PyResult<InstrumentId> {
        if index >= self.shared.symbols.get().len() {
            return Err(PyKeyError::new_err(index));
        }
        Ok(InstrumentId::new(index as u32))
    }

    /// Make the next event of `g` current and run every subscriber over it.
    #[allow(clippy::too_many_arguments)]
    fn deliver(
        &self,
        py: Python<'_>,
        guard: &mut LiveGuard<'_>,
        g: InstrumentId,
        kind: EventKind,
        ts: i64,
        row: u64,
        payload: Pushed,
    ) -> PyResult<()> {
        let event = with_feed(&self.shared, |feed| Ok(feed.event(g, kind, ts, row)))?;
        self.deliver_event(py, guard, event, payload)
    }

    fn deliver_event(
        &self,
        py: Python<'_>,
        guard: &mut LiveGuard<'_>,
        event: EventRef,
        payload: Pushed,
    ) -> PyResult<()> {
        let sh = &*self.shared;
        let ls = guard.as_mut().expect("checked by live_loop");
        let pushed = GroupEvent { event, payload, times: sh.times.get() };
        if sh.bracket.get() {
            sh.bracket_events.set(sh.bracket_events.get() + 1);
        } else {
            // Each push is a group of one, delivered on arrival.
            return run_one(py, sh, ls, pushed);
        }
        // An open group holds its pushes. Whatever is stamped earlier than this
        // one is complete: no event of it can still arrive.
        let ready: Vec<GroupEvent> = {
            let mut held = sh.held.borrow_mut();
            let (ready, keep): (Vec<_>, Vec<_>) =
                held.drain(..).partition(|h| h.event.ts < event.ts);
            *held = keep;
            held.push(pushed);
            ready
        };
        run_ready(py, sh, ls, ready)
    }

    /// Deliver what an open group holds, if the call is not made from a hook.
    fn flush_held(&self, py: Python<'_>) -> PyResult<()> {
        if self.shared.held.borrow().is_empty() {
            return Ok(());
        }
        let mut guard = self.live_loop()?;
        self.drain_held(py, &mut guard)
    }

    /// Deliver what an open group holds. The group stays open: a later push
    /// still joins it.
    fn drain_held(&self, py: Python<'_>, guard: &mut LiveGuard<'_>) -> PyResult<()> {
        let sh = &*self.shared;
        let ready: Vec<GroupEvent> = std::mem::take(&mut *sh.held.borrow_mut());
        if ready.is_empty() {
            return Ok(());
        }
        let ls = guard.as_mut().expect("checked by live_loop");
        let result = run_ready(py, sh, ls, ready);
        self.settle_push();
        result
    }

    /// Nothing is in flight between pushes.
    fn settle_push(&self) {
        self.shared.times.set(None);
        if let Inner::Running(core) = &mut *self.shared.inner.borrow_mut() {
            core.clear_current();
        }
    }

    /// Record the three times of the event being pushed. With no declared
    /// arrival and no processing latency they are all the event's own time,
    /// which is what an unset value reads as.
    fn stamp(&self, event_ns: i64, arrival_ns: Option<i64>) {
        let sh = &*self.shared;
        let latency = sh.processing_latency_ns.get();
        if arrival_ns.is_none() && latency == 0 {
            return;
        }
        let arrival = arrival_ns.unwrap_or(event_ns);
        sh.times.set(Some(Times {
            event: event_ns,
            arrival,
            processing: arrival.saturating_add(latency),
        }));
    }

    /// A refused row: advance the clock of its instrument for every strategy
    /// subscribed to it.
    fn mark_time(&self, py: Python<'_>, g: InstrumentId, ts: i64) -> PyResult<()> {
        let sh = &*self.shared;
        let _guard = self.live_loop()?;
        self.settle_push();
        let subs: Vec<StrategyId> = match &*sh.inner.borrow() {
            Inner::Running(c) => c.subscribers(g).to_vec(),
            _ => return Err(finished()),
        };
        let strats: Vec<Rc<Strat>> = sh.strats.borrow().clone();
        for sid in subs {
            let st = strats.iter().find(|s| s.sid == sid).expect("subscribed");
            time_only(py, sh, st, g, ts)?;
        }
        Ok(())
    }
}

/// One event on its way to being delivered, possibly with the others of its
/// timestamp group.
#[derive(Clone, Copy)]
struct GroupEvent {
    event: EventRef,
    payload: Pushed,
    /// The three times a live push declared, set while the event is handled.
    times: Option<Times>,
}

/// Deliver one event alone: a group of one. This is the path of every
/// single-instrument run, and of a live push outside an open group.
fn run_one(py: Python<'_>, sh: &Shared, ls: &mut LoopState, e: GroupEvent) -> PyResult<()> {
    sh.times.set(e.times);
    let cur = {
        let mut inner = sh.inner.borrow_mut();
        let Inner::Running(core) = &mut *inner else { return Err(finished()) };
        core.advance_pushed(e.event, e.payload).map_err(runtime_err)?;
        *core.current().expect("advance sets the current event")
    };
    open_phase(py, sh, ls, std::slice::from_ref(&e), false)?;
    rest_phase(py, sh, ls, std::slice::from_ref(&e), false)?;
    let delivered = dispatch(py, sh, ls, &e.event, &cur, None);
    hook_error(sh, delivered)
}

/// Deliver events that may span several timestamps and kinds: in time order,
/// each `(timestamp, kind)` as one group.
fn run_ready(
    py: Python<'_>,
    sh: &Shared,
    ls: &mut LoopState,
    mut events: Vec<GroupEvent>,
) -> PyResult<()> {
    events.sort_by_key(|e| (e.event.ts, e.event.kind.phase()));
    let mut at = 0;
    while at < events.len() {
        let key = (events[at].event.ts, events[at].event.kind.phase());
        let end = events[at..]
            .iter()
            .position(|e| (e.event.ts, e.event.kind.phase()) != key)
            .map_or(events.len(), |n| at + n);
        run_group(py, sh, ls, &events[at..end])?;
        at = end;
    }
    Ok(())
}

/// Deliver the events of one timestamp group, whatever order they came in
/// (docs/architecture/event-semantics.md, the two-phase rule).
///
/// Every event is applied before any is decided, and every decision is taken
/// before any is matched, so what a decision sees of another instrument and
/// the bar a cross-instrument order meets do not depend on the order the
/// instruments were listed in. The events go in ascending symbol: the order
/// in which simultaneous decisions that compete for cash are resolved, which
/// is a stated convention and cannot change what a handler sees. An
/// instrument that appears twice in a group is split across rounds, each its
/// own group.
fn run_group(
    py: Python<'_>,
    sh: &Shared,
    ls: &mut LoopState,
    events: &[GroupEvent],
) -> PyResult<()> {
    if events.len() == 1 {
        return run_one(py, sh, ls, events[0]);
    }
    let mut order = std::mem::take(&mut ls.order);
    order.clear();
    order.extend(0..events.len());
    // `rank` orders instruments by symbol, ties by place in the group.
    order.sort_unstable_by_key(|&i| (ls.rank[events[i].event.instrument.index()], i));
    // The usual group has each instrument once: it is one round, in order.
    ls.epoch += 1;
    let epoch = ls.epoch;
    let mut repeated = false;
    for e in events {
        let seen = &mut ls.seen[e.event.instrument.index()];
        repeated |= *seen == epoch;
        *seen = epoch;
    }
    let mut round = std::mem::take(&mut ls.round);
    let result = if repeated {
        let mut rounds: Vec<Vec<GroupEvent>> = Vec::new();
        // How many times each instrument has appeared so far.
        let mut seen: Vec<(usize, usize)> = Vec::new();
        for &i in &order {
            let id = events[i].event.instrument.index();
            let at = match seen.iter_mut().find(|(g, _)| *g == id) {
                Some((_, count)) => {
                    *count += 1;
                    *count - 1
                }
                None => {
                    seen.push((id, 1));
                    0
                }
            };
            if rounds.len() <= at {
                rounds.push(Vec::new());
            }
            rounds[at].push(events[i]);
        }
        rounds.iter().try_for_each(|r| {
            if r.len() == 1 {
                run_one(py, sh, ls, r[0])
            } else {
                run_round(py, sh, ls, r)
            }
        })
    } else {
        round.clear();
        round.extend(order.iter().map(|&i| events[i]));
        run_round(py, sh, ls, &round)
    };
    ls.round = round;
    ls.order = order;
    result
}

/// Select event `i` of the open group and return it.
fn select_event(sh: &Shared, i: usize, times: Option<Times>) -> PyResult<(Current, usize)> {
    sh.times.set(times);
    let mut inner = sh.inner.borrow_mut();
    let Inner::Running(core) = &mut *inner else { return Err(finished()) };
    core.select(i);
    Ok((*core.group_event(i), i))
}

/// Under `next_bar_open`, meet the open of the bars of a round before any strategy has seen them.
///
/// A venue reports the fills of what was decided on the previous bar at the open of this one,
/// before the bar has traded. So for each bar of the round, first every book matches its open (the
/// intents and market orders of earlier bars fill at it), and only then do the hooks hear those
/// fills, while the strategy, its indicators and everything it owns know only the bars up to the
/// previous one. What a hook answers fills at that same open. The strategy sees the bar, and
/// decides, afterwards; a fill the rest of the bar brings is heard after that decision, and what
/// it is answered with goes to the next bar.
///
/// Every book of the round is matched before any hook hears a fill, so an answer for another
/// instrument meets that instrument's open whichever symbol sorts first. `grouped` says the
/// events are the round's, selected one by one; a lone event is the one in flight.
fn open_phase(
    py: Python<'_>,
    sh: &Shared,
    ls: &mut LoopState,
    events: &[GroupEvent],
    grouped: bool,
) -> PyResult<()> {
    if !ls.opens || sh.warmup_muted.get() {
        return Ok(());
    }
    let mut met: Vec<(usize, StrategyId, usize, Matched)> = Vec::new();
    for (i, e) in events.iter().enumerate() {
        if e.event.kind != EventKind::Bar {
            continue;
        }
        if grouped {
            select_event(sh, i, e.times)?;
        }
        let subs: Vec<StrategyId> = match &*sh.inner.borrow() {
            Inner::Running(c) => c.subscribers(e.event.instrument).to_vec(),
            _ => return Err(finished()),
        };
        let ts_obj = e.event.ts.into_pyobject(py)?;
        let symbols = sh.symbols.get();
        let symbol = symbols[e.event.instrument.index()].bind(py);
        let with_ticks = sh.has_ticks.get()[e.event.instrument.index()];
        let ev = Ev { event: &e.event, ts_obj: &ts_obj, symbol, with_ticks, pos: Cell::new(0) };
        for (pos, &sid) in subs.iter().enumerate() {
            ev.pos.set(pos);
            let st = ls.by_sid[sid.index()].as_ref().expect("subscribed");
            let local = match &*sh.inner.borrow() {
                Inner::Running(c) => c.local_of(sid, e.event.instrument).expect("subscribed"),
                _ => return Err(finished()),
            };
            if let Some(matched) = open_match(py, sh, st, &ev, local)? {
                met.push((i, sid, local, matched));
            }
        }
    }
    for (i, sid, local, matched) in met {
        let e = &events[i];
        if grouped {
            select_event(sh, i, e.times)?;
        }
        let st = ls.by_sid[sid.index()].as_ref().expect("subscribed");
        let ts_obj = e.event.ts.into_pyobject(py)?;
        let symbols = sh.symbols.get();
        let symbol = symbols[e.event.instrument.index()].bind(py);
        let with_ticks = sh.has_ticks.get()[e.event.instrument.index()];
        let pos = match &*sh.inner.borrow() {
            Inner::Running(c) => c.subscribers(e.event.instrument).iter().position(|s| *s == sid),
            _ => return Err(finished()),
        }
        .expect("subscribed");
        let ev = Ev { event: &e.event, ts_obj: &ts_obj, symbol, with_ticks, pos: Cell::new(pos) };
        {
            let ctx = st.ctx.bind(py).borrow();
            ctx.local.set(local);
            ctx.idx.set(e.event.local_idx as usize);
        }
        if let Inner::Running(c) = &mut *sh.inner.borrow_mut() {
            c.decide_view(sid, pos, true);
        }
        sh.opening.set(true);
        ls.bank.set_opening(true);
        let heard = step_deliver(py, sh, st, &ev, local, false, false, &mut None, matched);
        ls.bank.set_opening(false);
        sh.opening.set(false);
        heard?;
        if st.has_held(py) {
            let close = sh.with_book(sid, |b, _| Ok(b.kernel(local).last_close))?;
            if let Some(close) = close {
                sh.open_intents.borrow_mut().insert((sid.index(), local), close);
            }
        }
    }
    Ok(())
}

/// Meet what rests through each bar of a round before any strategy has decided on the bars.
///
/// A venue fills a stop, a target or a resting limit while the bar forms, so a strategy that
/// decides on the bar's close must see the position the bar left. For each bar of the round every
/// book matches the orders and levels committed on earlier bars against the bar's range
/// (`RuntimeCore::rest_event`; on a simulated venue the shadow book does the same), and only then
/// do the hooks hear the fills, in the order of the round's events (ascending symbol) and, within
/// one, of the strategies. Every book of the round is matched before any hook hears, as in the
/// match phase, so an answer for another instrument meets that instrument's bar whichever symbol
/// sorts first.
///
/// The hooks hear a fill that happened inside the bar as the strategy would live: they know the
/// bars before it (`blind`, and the indicators show their value before the group), and `on_bar`
/// has not run. What they answer is an ordinary order, decided before the bar closed: it meets the
/// run's fill timing (this bar's close under `same_bar_close`, the next open under
/// `next_bar_open`). An order placed at the decision of this bar is not among the orders matched.
fn rest_phase(
    py: Python<'_>,
    sh: &Shared,
    ls: &mut LoopState,
    events: &[GroupEvent],
    grouped: bool,
) -> PyResult<()> {
    if sh.warmup_muted.get() {
        return Ok(());
    }
    let mut met: Vec<(usize, StrategyId, usize, Matched)> = Vec::new();
    for (i, e) in events.iter().enumerate() {
        if e.event.kind != EventKind::Bar {
            continue;
        }
        if grouped {
            select_event(sh, i, e.times)?;
        }
        let n = match &*sh.inner.borrow() {
            Inner::Running(c) => c.subscribers(e.event.instrument).len(),
            _ => return Err(finished()),
        };
        for pos in 0..n {
            let (sid, local) = match &*sh.inner.borrow() {
                Inner::Running(c) => {
                    let sid = c.subscribers(e.event.instrument)[pos];
                    (sid, c.local_of(sid, e.event.instrument).expect("subscribed"))
                }
                _ => return Err(finished()),
            };
            let st = ls.by_sid[sid.index()].as_ref().expect("subscribed");
            let matched = if sh.venue_on.get() {
                let venue = sh.venue.borrow().clone();
                let Some(venue) = venue else { continue };
                let ts_obj = e.event.ts.into_pyobject(py)?;
                let symbols = sh.symbols.get();
                let symbol = symbols[e.event.instrument.index()].bind(py);
                let with_ticks = sh.has_ticks.get()[e.event.instrument.index()];
                let ev = Ev {
                    event: &e.event,
                    ts_obj: &ts_obj,
                    symbol,
                    with_ticks,
                    pos: Cell::new(pos),
                };
                // The venue's reports close the strategy's book ahead of its own step: the bar's
                // range is taken into the excursions first.
                let range = match &*sh.inner.borrow() {
                    Inner::Running(c) => c.current().and_then(|cur| match cur.data {
                        ScheduleData::Bar(b) => Some((b.high, b.low)),
                        _ => None,
                    }),
                    _ => None,
                };
                if let Some((high, low)) = range {
                    sh.with_book_mut(sid, |b, _| {
                        b.kernel_mut(local).observe_range(high, low);
                        Ok(())
                    })?;
                }
                let Some(out) = venue.poll_rest(sh, st, &ev, local)? else { continue };
                let (venue_heard, venue_held) = venue.poll_book(py, sh, st, &ev, out)?;
                Matched { events: Vec::new(), venue_held, venue_heard }
            } else {
                let events = match &mut *sh.inner.borrow_mut() {
                    Inner::Running(c) => c.rest_event(sid).map_err(runtime_err)?,
                    _ => return Err(finished()),
                };
                if events.is_empty() {
                    continue;
                }
                Matched { events, venue_held: Vec::new(), venue_heard: false }
            };
            met.push((i, sid, local, matched));
        }
    }
    for (i, sid, local, matched) in met {
        let e = &events[i];
        if grouped {
            select_event(sh, i, e.times)?;
        }
        let st = ls.by_sid[sid.index()].as_ref().expect("subscribed");
        let ts_obj = e.event.ts.into_pyobject(py)?;
        let symbols = sh.symbols.get();
        let symbol = symbols[e.event.instrument.index()].bind(py);
        let with_ticks = sh.has_ticks.get()[e.event.instrument.index()];
        let pos = match &*sh.inner.borrow() {
            Inner::Running(c) => c.subscribers(e.event.instrument).iter().position(|s| *s == sid),
            _ => return Err(finished()),
        }
        .expect("subscribed");
        let ev = Ev { event: &e.event, ts_obj: &ts_obj, symbol, with_ticks, pos: Cell::new(pos) };
        {
            let ctx = st.ctx.bind(py).borrow();
            ctx.local.set(local);
            ctx.idx.set(e.event.local_idx as usize);
        }
        if let Inner::Running(c) = &mut *sh.inner.borrow_mut() {
            c.decide_view(sid, pos, true);
        }
        let open = match &*sh.inner.borrow() {
            Inner::Running(c) => c.current().and_then(|cur| match cur.data {
                ScheduleData::Bar(b) => Some(b.open),
                _ => None,
            }),
            _ => None,
        };
        sh.rest_open.set(open);
        sh.resting.set(true);
        ls.bank.set_opening(true);
        let heard = step_deliver(py, sh, st, &ev, local, false, false, &mut None, matched);
        ls.bank.set_opening(false);
        sh.resting.set(false);
        sh.rest_open.set(None);
        heard?;
        if st.has_held(py) {
            let close = sh.with_book(sid, |b, _| Ok(b.kernel(local).last_close))?;
            if let Some(close) = close {
                sh.open_intents.borrow_mut().insert((sid.index(), local), close);
            }
        }
    }
    Ok(())
}

/// The match half of the open of one bar for one strategy: its book (or, on a simulated venue,
/// the venue's shadow of it) fills what was decided earlier at the open. `None` when there is
/// nothing to hear or the bar cannot be opened ahead of the strategy.
fn open_match(
    py: Python<'_>,
    sh: &Shared,
    st: &Strat,
    ev: &Ev<'_, '_>,
    local: usize,
) -> PyResult<Option<Matched>> {
    if sh.venue_on.get() {
        let venue = sh.venue.borrow().clone();
        let Some(venue) = venue else { return Ok(None) };
        let Some(out) = venue.poll_open(sh, st, ev, local)? else { return Ok(None) };
        let (venue_heard, venue_held) = venue.poll_book(py, sh, st, ev, out)?;
        return Ok(Some(Matched { events: Vec::new(), venue_held, venue_heard }));
    }
    let events = match &mut *sh.inner.borrow_mut() {
        Inner::Running(c) => c.open_event(st.sid).map_err(runtime_err)?,
        _ => return Err(finished()),
    };
    if events.is_empty() {
        return Ok(None);
    }
    Ok(Some(Matched { events, venue_held: Vec::new(), venue_heard: false }))
}

/// The three phases over events of one timestamp and kind, each instrument
/// once, in the order they are to be decided and matched.
fn run_round(
    py: Python<'_>,
    sh: &Shared,
    ls: &mut LoopState,
    events: &[GroupEvent],
) -> PyResult<()> {
    let payloads: Vec<(EventRef, Pushed)> = events.iter().map(|e| (e.event, e.payload)).collect();
    // 1. Apply: every event is ingested, the books' lifecycle and marks follow.
    match &mut *sh.inner.borrow_mut() {
        Inner::Running(core) => core.begin_group(&payloads).map_err(runtime_err)?,
        _ => return Err(finished()),
    }
    let mut decided = std::mem::take(&mut ls.decided);
    decided.resize_with(events.len(), Vec::new);
    let phases = (|| -> PyResult<()> {
        // 1b. Under `next_bar_open` the open of every bar is met before any strategy has seen
        // the bars: the fills at the open are heard by hooks that know only what was known then.
        open_phase(py, sh, ls, events, true)?;
        // 1c. What rests (stops, targets, orders of earlier bars) trades every bar's range
        // before any strategy decides on the bars, as at a venue while they form.
        rest_phase(py, sh, ls, events, true)?;
        // The indicators strategies own are fed with every event of the group
        // too, so a decision reads each instrument's indicators as of the group.
        for (i, e) in events.iter().enumerate() {
            let (cur, _) = select_event(sh, i, e.times)?;
            feed_owned(py, sh, ls, &e.event, &cur)?;
        }
        // 2. Decide: handlers run on each event, what they queue is kept.
        for (i, e) in events.iter().enumerate() {
            let (cur, _) = select_event(sh, i, e.times)?;
            dispatch(py, sh, ls, &e.event, &cur, Some(&mut decided[i]))?;
        }
        // 3. Match: the queued work reaches every instrument's kernel, then
        // each kernel steps. With no command queued there is nothing to route.
        for d in decided.iter().flatten() {
            ls.remaining[d.sid.index()] += 1;
        }
        if decided.iter().flatten().any(|d| !d.no_commands()) {
            for pass in 0..3 {
                for (i, e) in events.iter().enumerate() {
                    select_event(sh, i, e.times)?;
                    dispatch_route(py, sh, ls, &e.event, &mut decided[i], pass)?;
                }
            }
        }
        // Every instrument of the group is matched before any hook hears a fill: a hook's
        // command applies from the next bar and cannot reach a bar of this timestamp that has
        // not been matched, whatever the symbols are called.
        let mut heard: Vec<Vec<Pending>> = Vec::with_capacity(events.len());
        for (i, e) in events.iter().enumerate() {
            select_event(sh, i, e.times)?;
            let mut out = Vec::new();
            dispatch_match(py, sh, ls, &e.event, &mut decided[i], &mut out)?;
            heard.push(out);
        }
        for (i, (e, pending)) in events.iter().zip(heard).enumerate() {
            select_event(sh, i, e.times)?;
            dispatch_deliver(py, sh, ls, &e.event, pending)?;
        }
        Ok(())
    })();
    if let Inner::Running(core) = &mut *sh.inner.borrow_mut() {
        core.end_group();
    }
    decided.iter_mut().for_each(Vec::clear);
    ls.decided = decided;
    ls.remaining.fill(0);
    hook_error(sh, phases)
}

/// Run `f` on the live rings.
fn with_feed<R>(sh: &Shared, f: impl FnOnce(&mut LiveFeed) -> PyResult<R>) -> PyResult<R> {
    match &mut *sh.feed.borrow_mut() {
        Some(feed) => f(feed),
        None => Err(not_ready("finalize() before pushing")),
    }
}

/// Store a pushed row, turning a full ring into a clear `RuntimeError`.
fn store_row<R>(
    sh: &Shared,
    g: InstrumentId,
    kind: RingKind,
    f: impl FnOnce(&mut LiveFeed) -> Result<R, RingError>,
) -> PyResult<R> {
    with_feed(sh, |feed| {
        f(feed).map_err(|e| PyErr::from(Fail::new("history_exhausted", feed.explain(g, kind, &e))))
    })
}

fn builder_locals(builder: &RuntimeBuilder, sid: StrategyId) -> usize {
    builder.instruments_of(sid)
}

/// What the per-event loop carries from one event to the next.
struct LoopState {
    by_sid: Vec<Option<Rc<Strat>>>,
    /// Hooks any subscriber of an instrument overrides: a print, quote or
    /// book object is built once per event, and only if one needs it.
    wanted: Vec<u32>,
    bank: Arc<IndicatorBank>,
    subs: Vec<StrategyId>,
    scratch: Scratch,
    /// Each instrument's place in ascending symbol order: the order a group is
    /// decided in (equal symbols share a rank).
    rank: Vec<u32>,
    /// The last group each instrument appeared in, by `epoch`: tells a group
    /// that has an instrument twice from the usual one.
    seen: Vec<u32>,
    epoch: u32,
    /// Buffers a group reuses.
    order: Vec<usize>,
    round: Vec<GroupEvent>,
    decided: Vec<Vec<Decided>>,
    /// Decisions of the open group still to be matched, by strategy.
    remaining: Vec<u32>,
    /// Some strategy owns an `Indicator` the loop feeds when an event is applied.
    natives: bool,
    /// Some strategy's book fills at the next bar's open: its bars are opened ahead of the
    /// strategy (see [`open_phase`]).
    opens: bool,
}

/// Each instrument's place in ascending symbol order (equal symbols share a rank).
fn symbol_ranks(sh: &Shared) -> Vec<u32> {
    let symbols = sh.symbols.get();
    let names: Vec<String> = Python::attach(|py| {
        symbols.iter().map(|s| s.bind(py).to_str().unwrap_or("").to_owned()).collect()
    });
    let mut sorted: Vec<&str> = names.iter().map(String::as_str).collect();
    sorted.sort_unstable();
    sorted.dedup();
    names
        .iter()
        .map(|n| sorted.binary_search(&n.as_str()).expect("a name is in its own list") as u32)
        .collect()
}

impl LoopState {
    fn new(sh: &Shared) -> PyResult<Self> {
        let inner = sh.inner.borrow();
        let Inner::Running(core) = &*inner else { return Err(finished()) };
        let strats: Vec<Rc<Strat>> = sh.strats.borrow().clone();
        let mut by_sid: Vec<Option<Rc<Strat>>> = vec![None; strats.len()];
        let mut wanted = vec![0u32; sh.symbols.get().len()];
        for s in &strats {
            by_sid[s.sid.index()] = Some(s.clone());
            for local in 0..core.instruments_of(s.sid) {
                if let Some(g) = core.instrument_of(s.sid, local) {
                    wanted[g.index()] |= s.mask;
                }
            }
        }
        let rank = symbol_ranks(sh);
        let n_ranked = rank.len();
        Ok(Self {
            by_sid,
            wanted,
            bank: core.bank().clone(),
            subs: Vec::new(),
            scratch: Scratch::default(),
            rank,
            seen: vec![0; n_ranked],
            epoch: 0,
            order: Vec::new(),
            round: Vec::new(),
            decided: Vec::new(),
            remaining: vec![0; strats.len()],
            natives: strats.iter().any(|s| {
                let t = s.tables.borrow();
                t.primary_native.iter().any(|v| !v.is_empty())
                    || t.primary_private.iter().any(|v| !v.is_empty())
                    || t.streams.iter().any(|i| !i.native.is_empty())
            }),
            opens: strats
                .iter()
                .any(|s| core.book(s.sid).is_some_and(|b| b.kernel(0).fills_at_next_open())),
        })
    }
}

/// Feed the `Indicator`s the subscribers of `event` own with what the event
/// brought: the bars its aggregations completed and its primary bar. Done when
/// the event is applied, before any event of its timestamp group is decided,
/// so a decision on another instrument reads the value after this event.
/// The selected event must be `event` (the aggregations' deliveries are read
/// from it). What a strategy's own hooks see of these is settled by their
/// reveals (`composites`, `primary_bar`).
fn feed_owned(
    py: Python<'_>,
    sh: &Shared,
    ls: &mut LoopState,
    event: &EventRef,
    cur: &Current,
) -> PyResult<()> {
    if !ls.natives {
        return Ok(());
    }
    let LoopState { by_sid, subs, scratch, .. } = ls;
    let primary = match cur.data {
        ScheduleData::Bar(b) => {
            Some(Ohlc { open: b.open, high: b.high, low: b.low, close: b.close })
        }
        ScheduleData::Trade(_) => {
            cur.primary.map(|b| Ohlc { open: b.open, high: b.high, low: b.low, close: b.close })
        }
        _ => None,
    };
    subs.clear();
    match &*sh.inner.borrow() {
        Inner::Running(c) => subs.extend_from_slice(c.subscribers(event.instrument)),
        _ => return Err(finished()),
    }
    for &sid in subs.iter() {
        let st = by_sid[sid.index()].as_ref().expect("subscribed");
        let t = st.tables.borrow();
        let local = match &*sh.inner.borrow() {
            Inner::Running(c) => {
                c.deliveries_into(sid, &mut scratch.deliveries);
                c.local_of(sid, event.instrument).expect("subscribed")
            }
            _ => return Err(finished()),
        };
        for &(stream, ref b) in &scratch.deliveries {
            let bar = Ohlc { open: b.open, high: b.high, low: b.low, close: b.close };
            for ind in &t.streams[stream as usize].native {
                ind.bind(py).borrow_mut().feed(bar);
            }
        }
        if let Some(bar) = primary {
            for ind in &t.primary_native[local] {
                ind.bind(py).borrow_mut().feed(bar);
            }
            // A strategy's own objects (anything with `update_bar`) are current like an
            // `Indicator`, whatever its symbol is called: every instrument's of a timestamp is
            // up to date before any handler of it runs.
            for ind in &t.primary_private[local] {
                ind.bind(py).call_method1(
                    intern!(py, "update_bar"),
                    (bar.open, bar.high, bar.low, bar.close),
                )?;
            }
        }
    }
    Ok(())
}

/// Run every strategy subscribed to the current event over it, in
/// registration order. The one body behind a batch run's tape and a live
/// node's pushes.
///
/// With `defer` given, the handlers run and what each strategy decided is
/// pushed on it, unmatched: see [`dispatch_match`].
fn dispatch(
    py: Python<'_>,
    sh: &Shared,
    ls: &mut LoopState,
    event: &EventRef,
    cur: &Current,
    mut defer: Option<&mut Vec<Decided>>,
) -> PyResult<()> {
    if defer.is_none() {
        // A lone event is applied here; a group's events were all applied
        // before the first was decided (`run_round`).
        feed_owned(py, sh, ls, event, cur)?;
    }
    let LoopState { by_sid, wanted, bank, subs, scratch, .. } = ls;
    subs.clear();
    match &*sh.inner.borrow() {
        Inner::Running(c) => subs.extend_from_slice(c.subscribers(event.instrument)),
        _ => return Err(finished()),
    }
    if let Some(out) = defer.as_deref_mut() {
        out.reserve(subs.len());
    }
    let mask = wanted[event.instrument.index()];
    // One Python int per event, not per strategy.
    let ts_obj = event.ts.into_pyobject(py)?;
    let symbols = sh.symbols.get();
    let symbol = symbols[event.instrument.index()].bind(py);
    let with_ticks = sh.has_ticks.get()[event.instrument.index()];
    let ev = Ev { event, ts_obj: &ts_obj, symbol, with_ticks, pos: Cell::new(0) };
    match cur.data {
        ScheduleData::Bar(bar) => {
            let offset = sh.offset_of(event.instrument.index());
            let bar_obj = sh.classes.bar.bind(py).call1((
                bar.timestamp - offset,
                bar.open,
                bar.high,
                bar.low,
                bar.close,
                bar.volume,
            ))?;
            for (pos, &sid) in subs.iter().enumerate() {
                ev.pos.set(pos);
                let st = by_sid[sid.index()].as_ref().expect("subscribed");
                process_bar(py, sh, st, bank, &ev, &bar, &bar_obj, scratch, defer.as_deref_mut())?;
            }
        }
        ScheduleData::Trade(t) => {
            let tick = (mask & HOOK_TRADE != 0)
                .then(|| {
                    sh.classes.trade.bind(py).call1((t.timestamp, t.price, t.size, symbol, t.oi))
                })
                .transpose()?;
            let primary = cur
                .primary
                .map(|b| {
                    sh.classes
                        .bar
                        .bind(py)
                        .call1((b.timestamp, b.open, b.high, b.low, b.close, b.volume))
                        .map(|obj| (b, obj))
                })
                .transpose()?;
            for (pos, &sid) in subs.iter().enumerate() {
                ev.pos.set(pos);
                let st = by_sid[sid.index()].as_ref().expect("subscribed");
                process_trade(
                    py,
                    sh,
                    st,
                    bank,
                    &ev,
                    &t,
                    primary.as_ref().map(|(b, o)| (b, o)),
                    tick.as_ref(),
                    scratch,
                    defer.as_deref_mut(),
                )?;
            }
        }
        ScheduleData::Quote(q) => {
            let quote = (mask & HOOK_QUOTE != 0)
                .then(|| {
                    sh.classes.quote.bind(py).call1((
                        q.timestamp,
                        q.bid,
                        q.ask,
                        symbol,
                        q.bid_size,
                        q.ask_size,
                    ))
                })
                .transpose()?;
            for (pos, &sid) in subs.iter().enumerate() {
                ev.pos.set(pos);
                let st = by_sid[sid.index()].as_ref().expect("subscribed");
                process_quote(py, sh, st, &ev, &q, quote.as_ref(), defer.as_deref_mut())?;
            }
        }
        ScheduleData::Depth(_) => {
            let tick = match &*sh.inner.borrow() {
                Inner::Running(c) => c.depth_of(event),
                _ => return Err(finished()),
            }
            .ok_or_else(|| {
                PyErr::from(Fail::new("history_exhausted", "a book event has no depth row"))
            })?;
            let book = (mask & HOOK_BOOK != 0)
                .then(|| book_object(py, sh, event.instrument, &tick).map(|o| o.into_bound(py)))
                .transpose()?;
            for (pos, &sid) in subs.iter().enumerate() {
                ev.pos.set(pos);
                let st = by_sid[sid.index()].as_ref().expect("subscribed");
                process_book(py, sh, st, &ev, &tick, book.as_ref(), defer.as_deref_mut())?;
            }
        }
    }
    Ok(())
}

/// Advance one instrument's clock for one strategy to `ts` and fire what is
/// due, with no data event: a refused row, or `advance_time`. Work the hooks
/// queue is settled at once, and no event is in flight, so an order for a
/// target is stamped on the target's last print.
fn time_only(
    py: Python<'_>,
    sh: &Shared,
    st: &Strat,
    global: InstrumentId,
    ts: i64,
) -> PyResult<()> {
    let local = match &*sh.inner.borrow() {
        Inner::Running(c) => c.local_of(st.sid, global).expect("subscribed"),
        _ => return Err(finished()),
    };
    let ctx = st.ctx.bind(py);
    ctx.borrow().local.set(local);
    let strategy = st.strategy.bind(py);
    let tables = st.tables.borrow();
    let cs = &tables.clocks[local];
    let clock = cs.clock.bind(py);
    if st.active_clock.get() != local {
        strategy.setattr(intern!(py, "clock"), clock)?;
        st.active_clock.set(local);
    }
    let due = clock.call_method1(intern!(py, "_advance"), (ts,))?;
    for time_event in due.try_iter()? {
        strategy.call_method1(intern!(py, "on_time_event"), (ctx, time_event?))?;
    }
    if has_commands(py, st) || has_intents(py, st) {
        settle(py, sh, st, global, local, 0, ts)?;
    }
    Ok(())
}

/// Split indicator objects into the ones that are not `Indicator`s, fed when
/// their bar is handled, and the `Indicator`s, fed when the event is applied.
fn split_indicators<'a>(
    py: Python<'_>,
    objs: impl Iterator<Item = &'a Py<PyAny>>,
) -> (Vec<Py<PyAny>>, Vec<Py<PyIndicator>>) {
    let (mut privates, mut natives) = (Vec::new(), Vec::new());
    for obj in objs {
        match obj.bind(py).cast::<PyIndicator>() {
            Ok(native) => natives.push(native.clone().unbind()),
            Err(_) => privates.push(obj.clone_ref(py)),
        }
    }
    (privates, natives)
}

/// Buffers reused across events.
#[derive(Default)]
struct Scratch {
    deliveries: Vec<(u16, crate::core::types::OhlcvBar)>,
}

/// The event every subscriber of it is shown.
struct Ev<'a, 'py> {
    event: &'a EventRef,
    ts_obj: &'a Bound<'py, PyInt>,
    /// The event's symbol and whether its instrument takes ticks, read once per event.
    symbol: &'a Bound<'py, PyString>,
    with_ticks: bool,
    /// The place, among the subscribers of the event's instrument, of the
    /// strategy being shown the event.
    pos: Cell<usize>,
}

fn has_commands(py: Python<'_>, st: &Strat) -> bool {
    st.queue.bind(py).borrow().queue.has_commands()
}

fn has_intents(py: Python<'_>, st: &Strat) -> bool {
    st.queue.bind(py).borrow().queue.has_intents()
}

impl Strat {
    /// Intents are held for the next step of some instrument.
    fn has_held(&self, py: Python<'_>) -> bool {
        match &self.route {
            Route::Python { held, .. } => !held.bind(py).is_empty(),
            Route::Native(state) => state.borrow().has_held(),
        }
    }
}

/// What every event starts with: point the context at the event's symbol and
/// ordinal, advance that symbol's clock and fire the timers it crossed.
/// Returns the local instrument and whether a time-event hook ran.
///
/// `bar` is the bar to show the time-event hooks: a bar-only instrument has
/// its bar current before its clock advances (as `run_portfolio_strategy`),
/// an instrument with ticks has its previous one (as `run_tick_strategy`).
fn enter(
    py: Python<'_>,
    sh: &Shared,
    st: &Strat,
    t: &StratTables,
    ev: &Ev<'_, '_>,
    bar: Option<(&Bound<'_, PyAny>, i64)>,
) -> PyResult<(usize, bool)> {
    let local = match &*sh.inner.borrow() {
        Inner::Running(c) => c.local_of(st.sid, ev.event.instrument).expect("subscribed"),
        _ => return Err(finished()),
    };
    if let Inner::Running(c) = &mut *sh.inner.borrow_mut() {
        c.decide_view(st.sid, ev.pos.get(), true);
    }
    let ctx = st.ctx.bind(py);
    {
        let c = ctx.borrow();
        c.local.set(local);
        c.idx.set(ev.event.local_idx as usize);
        if let Some((bar, close_ts)) = bar {
            release(py, c.bar.replace(Some(bar.clone().unbind())));
            c.close_ts.set(Some(close_ts));
        }
    }
    let strategy = st.strategy.bind(py);
    // Corporate actions and expiries effective by this event land before the
    // strategy sees it.
    let lifecycle = match &mut *sh.inner.borrow_mut() {
        Inner::Running(c) => c.begin_event(st.sid).map_err(runtime_err)?,
        _ => return Err(finished()),
    };
    if !lifecycle.is_empty() {
        deliver(py, sh, st, lifecycle, ev.symbol)?;
    }
    let cs = &t.clocks[local];
    let clock = cs.clock.bind(py);
    if st.active_clock.get() != local {
        strategy.setattr(intern!(py, "clock"), clock)?;
        st.active_clock.set(local);
    }
    let due = if cs.plain && cs.alerts.bind(py).is_empty() && cs.timers.bind(py).is_empty() {
        clock.setattr(intern!(py, "_now"), ev.ts_obj)?;
        None
    } else {
        Some(clock.call_method1(intern!(py, "_advance"), (ev.event.ts,))?)
    };
    let mut fired = false;
    if let Some(due) = due {
        if st.mask & HOOK_TIME_EVENT != 0 {
            for time_event in due.try_iter()? {
                fired = true;
                strategy.call_method1(intern!(py, "on_time_event"), (ctx, time_event?))?;
            }
        }
    }
    Ok((local, fired))
}

/// Completed composite bars of the event, in subscription order.
fn composites(
    py: Python<'_>,
    sh: &Shared,
    st: &Strat,
    t: &StratTables,
    bank: &IndicatorBank,
    scratch: &mut Scratch,
) -> PyResult<()> {
    scratch.deliveries.clear();
    if let Inner::Running(core) = &*sh.inner.borrow() {
        core.deliveries_into(st.sid, &mut scratch.deliveries);
    }
    let strategy = st.strategy.bind(py);
    let ctx = st.ctx.bind(py);
    let streams = &t.streams;
    let mut last: Option<u16> = None;
    let mut emission = 0u32;
    for &(stream, ref b) in &scratch.deliveries {
        emission = if last == Some(stream) { emission + 1 } else { 1 };
        last = Some(stream);
        let info = &streams[stream as usize];
        if let Some(handle) = info.reveal {
            bank.reveal(handle, emission);
        }
        for ind in &info.native {
            ind.bind(py).borrow_mut().reveal(emission);
        }
        for ind in &info.private {
            ind.bind(py)
                .call_method1(intern!(py, "update_bar"), (b.open, b.high, b.low, b.close))?;
        }
        if st.mask & HOOK_COMPOSITE != 0 {
            let composite = sh.classes.composite.bind(py).call(
                (
                    info.sub,
                    info.step,
                    info.unit.bind(py),
                    b.timestamp,
                    b.open,
                    b.high,
                    b.low,
                    b.close,
                    b.volume,
                ),
                Some(info.kwargs.bind(py)),
            )?;
            strategy.call_method1(intern!(py, "on_composite_bar"), (ctx, composite))?;
        }
    }
    Ok(())
}

/// The primary bar: reveal its indicators, then `on_bar`. The strategy's own objects were fed
/// when the event was applied (`feed_owned`).
fn primary_bar(
    py: Python<'_>,
    st: &Strat,
    t: &StratTables,
    bank: &IndicatorBank,
    local: usize,
) -> PyResult<()> {
    if let Some(handle) = t.primary_reveal[local] {
        bank.reveal(handle, 1);
    }
    for ind in &t.primary_native[local] {
        ind.bind(py).borrow_mut().reveal(u32::MAX);
    }
    if st.mask & HOOK_BAR != 0 {
        st.strategy.bind(py).call_method1(intern!(py, "on_bar"), (st.ctx.bind(py),))?;
    }
    Ok(())
}

/// Route the work the hooks queued so far to the engine: the commands.
fn drain_commands(
    py: Python<'_>,
    sh: &Shared,
    st: &Strat,
    local: usize,
    ev: &Ev<'_, '_>,
) -> PyResult<()> {
    if has_commands(py, st) {
        // What the handlers of this event decided: a venue is told, so an order for another
        // instrument whose bar of this instant is already stepped meets that bar, as in a batch
        // run (`met_bar`, see `PaperVenue::submit`).
        sh.deciding.set(true);
        let routed = route_commands(py, sh, st, local, ev.event.local_idx as usize, ev.event.ts);
        sh.deciding.set(false);
        routed?;
    }
    Ok(())
}

/// Route the queued commands to the instruments they name; `current` is the
/// event's local instrument.
fn route_commands(
    py: Python<'_>,
    sh: &Shared,
    st: &Strat,
    current: usize,
    idx: usize,
    ts: i64,
) -> PyResult<()> {
    match &st.route {
        Route::Python { path, .. } => {
            path.bind(py).call1((current, idx, ts))?;
        }
        Route::Native(state) => {
            let commands = st.queue.bind(py).borrow_mut().queue.take_commands();
            sh.jlog_now(jk::COMMAND, jsrc::STRATEGY, None, || commands_value(st.sid, &commands))?;
            let mut target = RuntimeTarget { sh, sid: st.sid, idx, ts };
            let applied = state.borrow_mut().apply(&mut target, commands, current, idx, ts);
            if let Err(e) = &applied {
                // The failure being reported matters more than a journal that
                // cannot take its note: that surfaces on the next append.
                let recorded = sh.recorded(jk::REFUSAL);
                let _ = sh.jlog_now(jk::REFUSAL, jsrc::ENGINE, None, || {
                    recorded.unwrap_or_else(|| refusal_value(py, st.sid, e))
                });
            }
            applied?;
        }
    }
    Ok(())
}

/// Close out an event whose hooks may have queued more work: commands go to
/// the event's own instrument now; intents cannot step a second time, so
/// they are held for the instrument's next step.
fn settle(
    py: Python<'_>,
    sh: &Shared,
    st: &Strat,
    global: InstrumentId,
    local: usize,
    local_idx: usize,
    ts: i64,
) -> PyResult<()> {
    if sh.warmup_muted.get() {
        discard_queued(py, st);
        return Ok(());
    }
    match &st.route {
        Route::Python { path, .. } => {
            let symbol = sh.symbol(py, global.index());
            path.bind(py).call_method1(intern!(py, "settle"), (local, symbol, local_idx, ts))?;
        }
        Route::Native(state) => {
            if has_commands(py, st) {
                route_commands(py, sh, st, local, local_idx, ts)?;
            }
            if has_intents(py, st) {
                state.borrow_mut().hold(&mut st.queue.bind(py).borrow_mut().queue, local);
            }
        }
    }
    Ok(())
}

/// With a host order check installed, `enter()` / `close_position()` intents
/// are refused: they carry no order to show the check. Not when the check declared it
/// judges typed orders only (`order_check_scope="typed"`).
fn refuse_intents(sh: &Shared, input: &StepInput) -> PyResult<()> {
    if (input.entry || input.exit)
        && sh.order_check.borrow().is_some()
        && !sh.order_check_typed_only.get()
    {
        return Err(Fail::new(
            "order_check_intent",
            "enter() and close_position() carry no order for the host's order_check to see: \
             submit typed orders (submit_order) when an order_check is installed",
        )
        .param("order_check")
        .into());
    }
    Ok(())
}

/// An `enter()` intent on an instrument that takes no orders, or decided before the run's
/// `trade_from`, is refused: it is dropped, and the book reports it as a rejected entry
/// (`not_tradable`, `warmup`) like any other refused entry.
fn refuse_entry_intent(
    sh: &Shared,
    sid: StrategyId,
    local: usize,
    local_idx: usize,
    ts: i64,
    input: &mut StepInput,
) -> PyResult<()> {
    if !input.entry {
        return Ok(());
    }
    let Some(reason) = sh.host_refusal(sid, local, ts)? else { return Ok(()) };
    input.entry = false;
    sh.with_book_mut(sid, |b, _| {
        b.kernel_mut(local).refuse_entry(local_idx, reason);
        Ok(())
    })?;
    sh.jlog_now(jk::REFUSAL, jsrc::ENGINE, None, || {
        JV::map([
            ("strategy", JV::Int(sid.index() as i64)),
            ("code", JV::str(reason.as_str())),
            ("message", JV::str(reason.as_str())),
        ])
    })
}

/// Drop everything the strategy queued, held intents included: muted warmup
/// lets the strategy run but never lets it trade.
fn discard_queued(py: Python<'_>, st: &Strat) {
    {
        let mut q = st.queue.bind(py).borrow_mut();
        drop(q.queue.take_commands());
        drop(q.queue.take_intents());
    }
    match &st.route {
        Route::Python { held, .. } => held.bind(py).clear(),
        Route::Native(state) => state.borrow_mut().clear_held(),
    }
}

/// The exception for a refusal the command router made.
fn refusal(error: CmdError) -> PyErr {
    let fail = Fail::new(error.code, error.message).param(error.parameter);
    let fail = match error.value {
        ErrVal::Str(s) => fail.value(s),
        ErrVal::List(l) => fail.value(Val::List(l)),
    };
    let fail = match &error.valid {
        Some(valid) if !valid.is_empty() => fail.valid(valid),
        _ => fail,
    };
    // An empty list of alternatives is the code's rule from the registry, never an empty answer.
    PyErr::from(fail)
}

/// The order tag that marks a market order as the strategy's answer to a fill it just heard: it
/// meets the bar of the step that fill came in (`Order::follow_up`).
pub(crate) const FOLLOW_UP_TAG: &str = "after_fill";

/// A hedge exit in progress: `(strategy index, hedge instrument, hedge position id, sold legs sent,
/// hedge close sent)`.
type PendingCover = (usize, usize, u64, usize, bool);

/// A strategy's book, as the target its commands are routed to.
struct RuntimeTarget<'a> {
    sh: &'a Shared,
    sid: StrategyId,
    /// The ordinal and time of the event the commands were queued on.
    idx: usize,
    ts: i64,
}

impl RuntimeTarget<'_> {
    /// The venue orders go to, if the node has one.
    fn venue(&self) -> Option<Rc<VenueRoute>> {
        if self.sh.venue_on.get() {
            self.sh.venue.borrow().clone()
        } else {
            None
        }
    }
}

impl Target for RuntimeTarget<'_> {
    type Error = PyErr;

    fn refuse(error: CmdError) -> PyErr {
        refusal(error)
    }

    fn n_instruments(&self) -> usize {
        self.sh.locals(self.sid)
    }

    fn instrument_of(&mut self, symbol: Option<&str>, current: usize) -> PyResult<usize> {
        match symbol {
            None | Some("") => Ok(current),
            Some(name) => self
                .sh
                .sym_ids
                .borrow()
                .get(name)
                .and_then(|&g| self.sh.local_of(self.sid, g))
                .ok_or_else(|| unknown_symbol(name)),
        }
    }

    fn submit_order(
        &mut self,
        instrument: usize,
        spec: &OrderSpec,
        idx: usize,
        ts: i64,
        client_id: &str,
        parent: Option<u64>,
    ) -> PyResult<u64> {
        if let Some(venue) = self.venue() {
            return venue.submit(self.sh, self.sid, instrument, spec, idx, ts, client_id, parent);
        }
        let refusal = self.sh.refusal_for(self.sid, instrument, spec, client_id, ts)?;
        let follow_up = spec.tags.contains_key(FOLLOW_UP_TAG);
        let id = self.sh.with_book_mut(self.sid, |book, current| {
            let idx = book.submission_idx(instrument, idx, current);
            submit_order_checked(
                book.kernel_mut(instrument),
                &spec.side,
                &spec.kind,
                idx,
                ts,
                client_id,
                spec.units,
                spec.size_frac,
                spec.limit_price,
                spec.trigger_price,
                &spec.tif,
                spec.expire_ns,
                spec.stop_price,
                spec.target_price,
                spec.offset,
                &spec.offset_kind,
                spec.limit_offset,
                spec.post_only,
                spec.reduce_only,
                parent,
                refusal,
            )
        })?;
        if follow_up {
            self.sh.with_book_mut(self.sid, |book, _| {
                book.kernel_mut(instrument).set_order_follow_up(id);
                Ok(())
            })?;
            self.sh.follow_ups.borrow_mut().push((self.sid, instrument, id));
        }
        Ok(id)
    }

    fn check_terms(&mut self, instrument: usize, spec: &OrderSpec) -> PyResult<()> {
        let grid = self.sh.with_book(self.sid, |b, _| Ok(b.kernel(instrument).order_grid()))?;
        crate::python::strategy_bindings::check_grid(
            &grid,
            spec.units,
            &[
                ("limit_price", spec.limit_price),
                ("trigger_price", spec.trigger_price),
                ("stop_price", spec.stop_price),
                ("target_price", spec.target_price),
            ],
        )
    }

    fn submit_twap(
        &mut self,
        instrument: usize,
        spec: &TwapSpec,
        idx: usize,
        ts: i64,
        client_id: &str,
    ) -> PyResult<u64> {
        self.sh.with_book_mut(self.sid, |book, current| {
            let idx = book.submission_idx(instrument, idx, current);
            submit_twap_on(
                book.kernel_mut(instrument),
                spec.units,
                &spec.side,
                spec.slices,
                spec.interval_ns,
                idx,
                ts,
                client_id,
                spec.reduce_only,
            )
        })
    }

    fn cancel_twap(&mut self, instrument: usize, algo_id: u64, idx: usize) -> PyResult<()> {
        self.sh.with_book_mut(self.sid, |b, _| {
            b.kernel_mut(instrument).cancel_algo(algo_id, idx);
            Ok(())
        })
    }

    fn cancel_order(&mut self, instrument: usize, idx: usize, order_id: u64) -> PyResult<()> {
        if let Some(venue) = self.venue() {
            return venue.cancel(self.sh, self.sid, instrument, idx, order_id);
        }
        self.sh.with_book_mut(self.sid, |b, _| {
            b.kernel_mut(instrument).cancel_order(idx, order_id);
            Ok(())
        })
    }

    fn cancel_all_orders(&mut self, instrument: usize, idx: usize) -> PyResult<()> {
        if let Some(venue) = self.venue() {
            return venue.cancel_all(self.sh, self.sid, instrument, idx);
        }
        self.sh.with_book_mut(self.sid, |b, _| {
            b.kernel_mut(instrument).cancel_all_orders(idx);
            Ok(())
        })
    }

    fn link_oco(&mut self, instrument: usize, order_ids: &[u64]) -> PyResult<()> {
        if let Some(venue) = self.venue() {
            return venue.link_oco(self.sid, instrument, order_ids);
        }
        self.sh.with_book_mut(self.sid, |b, _| {
            b.kernel_mut(instrument).link_oco(order_ids);
            Ok(())
        })
    }

    fn request_close(&mut self, instrument: usize, position_id: u64) -> PyResult<()> {
        if let Some(venue) = self.venue() {
            return venue.close(
                self.sh,
                self.sid,
                instrument,
                self.idx,
                self.ts,
                position_id,
                false,
            );
        }
        self.sh.with_book_mut(self.sid, |b, _| {
            b.kernel_mut(instrument).request_close(position_id);
            Ok(())
        })
    }

    fn position_ids(&mut self, instrument: usize) -> PyResult<Vec<u64>> {
        self.sh.with_book(self.sid, |b, _| {
            Ok(b.kernel(instrument).position_snapshots().iter().map(|p| p.position_id).collect())
        })
    }

    fn modify_order(
        &mut self,
        instrument: usize,
        order_id: u64,
        units: Option<f64>,
        size_frac: Option<f64>,
        limit_price: Option<f64>,
        trigger_price: Option<f64>,
    ) -> PyResult<()> {
        let grid = self.sh.with_book(self.sid, |b, _| Ok(b.kernel(instrument).order_grid()))?;
        crate::python::strategy_bindings::check_grid(
            &grid,
            units,
            &[("limit_price", limit_price), ("trigger_price", trigger_price)],
        )?;
        if let Some(venue) = self.venue() {
            return venue.modify(
                self.sh,
                self.sid,
                instrument,
                self.idx,
                order_id,
                units,
                size_frac,
                limit_price,
                trigger_price,
            );
        }
        let qty = parse_qty_spec(units, size_frac)?;
        let decided = self.sh.decided_close(self.sid, instrument);
        self.sh.with_book_mut(self.sid, |b, _| {
            b.kernel_mut(instrument).modify_order_at(
                order_id,
                qty,
                limit_price,
                trigger_price,
                decided,
            );
            Ok(())
        })
    }
}

/// The step arguments for the queued intents, those held for the instrument
/// ahead of them.
fn fold_step(
    py: Python<'_>,
    st: &Strat,
    symbol: &Bound<'_, PyString>,
    local: usize,
    local_idx: usize,
    atr: f64,
) -> PyResult<StepInput> {
    match &st.route {
        Route::Python { path, .. } => {
            let kw = PyDict::new(py);
            kw.set_item("symbol", symbol)?;
            kw.set_item("atr", atr)?;
            let folded = path
                .bind(py)
                .call_method(intern!(py, "fold"), (local_idx,), Some(&kw))?
                .cast_into::<PyDict>()?;
            let get = |key: &str| -> PyResult<Bound<'_, PyAny>> {
                folded.get_item(key)?.ok_or_else(|| PyKeyError::new_err(key.to_string()))
            };
            Ok(StepInput {
                entry: get("entry")?.extract()?,
                exit: get("exit")?.extract()?,
                atr: get("atr")?.extract()?,
                size_mult: get("size_mult")?.extract()?,
                stop_price_override: get("stop_price")?.extract()?,
                target_price_override: get("target_price")?.extract()?,
                ..StepInput::default()
            })
        }
        Route::Native(state) => {
            let folded = state
                .borrow_mut()
                .fold(&mut st.queue.bind(py).borrow_mut().queue, local)
                .map_err(|e| refusal(e.refusal(&format!("{symbol} event {local_idx}"))))?;
            Ok(StepInput {
                entry: folded.entry,
                exit: folded.exit,
                atr,
                size_mult: folded.size_mult,
                stop_price_override: folded.stop_price,
                target_price_override: folded.target_price,
                ..StepInput::default()
            })
        }
    }
}

/// The command router's view of a single-instrument session.
struct SessionTarget<'a>(&'a mut PyKernelSession);

impl Target for SessionTarget<'_> {
    type Error = PyErr;

    fn refuse(error: CmdError) -> PyErr {
        refusal(error)
    }

    fn n_instruments(&self) -> usize {
        1
    }

    /// There is one instrument, so every symbol routes to it.
    fn instrument_of(&mut self, _symbol: Option<&str>, _current: usize) -> PyResult<usize> {
        Ok(0)
    }

    fn submit_order(
        &mut self,
        _instrument: usize,
        spec: &OrderSpec,
        idx: usize,
        ts: i64,
        client_id: &str,
        parent: Option<u64>,
    ) -> PyResult<u64> {
        submit_order_on(
            self.0.runner_mut()?.kernel_mut(),
            &spec.side,
            &spec.kind,
            idx,
            ts,
            client_id,
            spec.units,
            spec.size_frac,
            spec.limit_price,
            spec.trigger_price,
            &spec.tif,
            spec.expire_ns,
            spec.stop_price,
            spec.target_price,
            spec.offset,
            &spec.offset_kind,
            spec.limit_offset,
            spec.post_only,
            spec.reduce_only,
            parent,
        )
    }

    fn check_terms(&mut self, _instrument: usize, spec: &OrderSpec) -> PyResult<()> {
        crate::python::strategy_bindings::check_grid(
            &self.0.runner_ref()?.kernel().order_grid(),
            spec.units,
            &[
                ("limit_price", spec.limit_price),
                ("trigger_price", spec.trigger_price),
                ("stop_price", spec.stop_price),
                ("target_price", spec.target_price),
            ],
        )
    }

    fn submit_twap(
        &mut self,
        _instrument: usize,
        spec: &TwapSpec,
        idx: usize,
        ts: i64,
        client_id: &str,
    ) -> PyResult<u64> {
        submit_twap_on(
            self.0.runner_mut()?.kernel_mut(),
            spec.units,
            &spec.side,
            spec.slices,
            spec.interval_ns,
            idx,
            ts,
            client_id,
            spec.reduce_only,
        )
    }

    fn cancel_twap(&mut self, _instrument: usize, algo_id: u64, idx: usize) -> PyResult<()> {
        self.0.runner_mut()?.kernel_mut().cancel_algo(algo_id, idx);
        Ok(())
    }

    fn cancel_order(&mut self, _instrument: usize, idx: usize, order_id: u64) -> PyResult<()> {
        self.0.runner_mut()?.kernel_mut().cancel_order(idx, order_id);
        Ok(())
    }

    fn cancel_all_orders(&mut self, _instrument: usize, idx: usize) -> PyResult<()> {
        self.0.runner_mut()?.kernel_mut().cancel_all_orders(idx);
        Ok(())
    }

    fn link_oco(&mut self, _instrument: usize, order_ids: &[u64]) -> PyResult<()> {
        self.0.runner_mut()?.kernel_mut().link_oco(order_ids);
        Ok(())
    }

    fn request_close(&mut self, _instrument: usize, position_id: u64) -> PyResult<()> {
        self.0.runner_mut()?.kernel_mut().request_close(position_id);
        Ok(())
    }

    fn position_ids(&mut self, _instrument: usize) -> PyResult<Vec<u64>> {
        Ok(self
            .0
            .runner_ref()?
            .kernel()
            .position_snapshots()
            .iter()
            .map(|p| p.position_id)
            .collect())
    }

    fn modify_order(
        &mut self,
        _instrument: usize,
        order_id: u64,
        units: Option<f64>,
        size_frac: Option<f64>,
        limit_price: Option<f64>,
        trigger_price: Option<f64>,
    ) -> PyResult<()> {
        let qty = parse_qty_spec(units, size_frac)?;
        crate::python::strategy_bindings::check_grid(
            &self.0.runner_ref()?.kernel().order_grid(),
            units,
            &[("limit_price", limit_price), ("trigger_price", trigger_price)],
        )?;
        self.0.runner_mut()?.kernel_mut().modify_order(order_id, qty, limit_price, trigger_price);
        Ok(())
    }
}

/// `run_strategy_backtest`'s per-bar work after `on_bar`, in Rust: route the
/// strategy's queued commands, fold its intents into the bar's step, step the
/// session and send the engine events to the hooks.
///
/// Nothing is settled afterwards: work the hooks queue waits for the next
/// bar, as the array engine's parity requires.
#[pyclass(name = "KernelDriver", unsendable)]
pub struct PyKernelDriver {
    session: Py<PyKernelSession>,
    strategy: Py<PyAny>,
    ctx: Py<PyAny>,
    queue: Py<PyCommandQueue>,
    state: RefCell<CommandState>,
    mask: u32,
    /// The ATR per bar; empty when no ATR-based level is configured.
    atr: Vec<f64>,
}

#[pymethods]
impl PyKernelDriver {
    #[new]
    #[pyo3(signature = (session, strategy, ctx, hooks, atr=None))]
    fn new(
        session: Py<PyKernelSession>,
        strategy: Py<PyAny>,
        ctx: Py<PyAny>,
        hooks: u32,
        atr: Option<PyReadonlyArray1<'_, f64>>,
        py: Python<'_>,
    ) -> PyResult<Self> {
        let queue = strategy.bind(py).getattr("_queue")?.cast_into::<PyCommandQueue>()?.unbind();
        Ok(Self {
            session,
            strategy,
            ctx,
            queue,
            state: RefCell::new(CommandState::default()),
            mask: hooks,
            atr: atr.map(|a| a.as_array().to_vec()).unwrap_or_default(),
        })
    }

    /// Bar `idx`, stamped `ts`, before the strategy has seen it: what rests (stops, targets,
    /// orders of earlier bars) trades the bar's range and the hooks hear the fills.
    #[allow(clippy::too_many_arguments)]
    fn rest(
        &self,
        py: Python<'_>,
        idx: usize,
        ts: i64,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
    ) -> PyResult<()> {
        let bar = KernelBar { timestamp: ts, open, high, low, close, volume };
        let events = self.session.bind(py).borrow_mut().runner_mut()?.rest(idx, &bar);
        if events.is_empty() {
            return Ok(());
        }
        let ctx = self.ctx.bind(py);
        let symbol = ctx.getattr_opt(intern!(py, "symbol"))?.and_then(|s| s.cast_into().ok());
        dispatch_events(
            py,
            self.strategy.bind(py),
            ctx,
            live_mask(py, &self.queue, self.mask),
            events,
            symbol.as_ref(),
        )
    }

    /// Bar `idx`, stamped `ts`: everything after the strategy's `on_bar`.
    #[allow(clippy::too_many_arguments)]
    fn step(
        &self,
        py: Python<'_>,
        idx: usize,
        ts: i64,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
    ) -> PyResult<()> {
        let commands = self.queue.bind(py).borrow_mut().queue.take_commands();
        let folded = {
            let mut session = self.session.bind(py).borrow_mut();
            if !commands.is_empty() {
                let mut target = SessionTarget(&mut session);
                self.state.borrow_mut().apply(&mut target, commands, 0, idx, ts)?;
            }
            let folded = self
                .state
                .borrow_mut()
                .fold(&mut self.queue.bind(py).borrow_mut().queue, 0)
                .map_err(|e| refusal(e.refusal(&format!("bar {idx}"))))?;
            // An enter+close pair while in position would exit and
            // immediately re-enter on the same bar; refuse rather than
            // guess the intent.
            if folded.entry && folded.exit && session.runner_ref()?.is_in_position() {
                return Err(Fail::new(
                    "invalid_order_intent",
                    format!(
                        "bar {idx}: enter() and close_position() queued on the same bar \
                         while in position; emit one intent per bar"
                    ),
                )
                .op("run_strategy_backtest")
                .param("intents")
                .value(idx)
                .into());
            }
            folded
        };
        let input = StepInput {
            entry: folded.entry,
            exit: folded.exit,
            atr: self.atr.get(idx).copied().unwrap_or(0.0),
            size_mult: folded.size_mult,
            stop_price_override: folded.stop_price,
            target_price_override: folded.target_price,
            ..StepInput::default()
        };
        let bar = KernelBar { timestamp: ts, open, high, low, close, volume };
        let events = self.session.bind(py).borrow_mut().runner_mut()?.step(idx, &bar, input);
        if events.is_empty() {
            return Ok(());
        }
        let ctx = self.ctx.bind(py);
        // The symbol the context names, if it names one.
        let symbol = ctx.getattr_opt(intern!(py, "symbol"))?.and_then(|s| s.cast_into().ok());
        dispatch_events(
            py,
            self.strategy.bind(py),
            ctx,
            live_mask(py, &self.queue, self.mask),
            events,
            symbol.as_ref(),
        )
    }
}

/// What an engine event is, for routing.
#[derive(Clone, Copy, PartialEq)]
enum Tag {
    Entered,
    Exited,
    EntryRejected,
    AlgoStarted,
    AlgoCompleted,
    MarginCall,
    RiskRejected,
    RiskHalted,
    Accepted,
    Triggered,
    Filled,
    Canceled,
    Resized,
    Expired,
    Rejected,
    Contract,
    Corporate,
}

fn tag_of(event: &EngineEvent) -> Tag {
    match event {
        EngineEvent::Entered { .. } => Tag::Entered,
        EngineEvent::Exited { .. } => Tag::Exited,
        EngineEvent::EntryRejected { .. } => Tag::EntryRejected,
        EngineEvent::AlgoStarted { .. } => Tag::AlgoStarted,
        EngineEvent::AlgoCompleted { .. } => Tag::AlgoCompleted,
        EngineEvent::MarginCall { .. } => Tag::MarginCall,
        EngineEvent::RiskRejected { .. }
        | EngineEvent::FillBreach { .. }
        | EngineEvent::FillDeferred { .. } => Tag::RiskRejected,
        EngineEvent::RiskHalted { .. } => Tag::RiskHalted,
        EngineEvent::OrderAccepted { .. } => Tag::Accepted,
        EngineEvent::OrderTriggered { .. } => Tag::Triggered,
        EngineEvent::OrderFilled { .. } => Tag::Filled,
        EngineEvent::OrderCanceled { .. } => Tag::Canceled,
        EngineEvent::OrderResized { .. } => Tag::Resized,
        EngineEvent::OrderExpired { .. } => Tag::Expired,
        EngineEvent::OrderRejected { .. } => Tag::Rejected,
        EngineEvent::Settled { .. } => Tag::Contract,
        EngineEvent::LifecycleRefused { what: "corporate_action", .. }
        | EngineEvent::CorporateAction { .. } => Tag::Corporate,
        EngineEvent::LifecycleRefused { .. } => Tag::Contract,
    }
}

/// The hooks an event of this kind reaches, given whether an `order_filled`
/// event just fired `on_order_filled` for the position event after it.
fn hooks_of(tag: Tag, preceded: bool) -> u32 {
    let fill = if preceded { 0 } else { HOOK_ORDER_FILLED };
    match tag {
        Tag::Entered => fill | HOOK_POSITION_OPENED,
        Tag::Exited => fill | HOOK_POSITION_CLOSED,
        Tag::EntryRejected => HOOK_ORDER_REJECTED,
        Tag::AlgoStarted => HOOK_ALGO_STARTED | HOOK_ORDER_EVENT,
        Tag::AlgoCompleted => HOOK_ALGO_COMPLETED | HOOK_ORDER_EVENT,
        Tag::MarginCall => HOOK_MARGIN_CALL | HOOK_ORDER_EVENT,
        Tag::RiskRejected | Tag::RiskHalted => HOOK_RISK_EVENT,
        Tag::Accepted => HOOK_ORDER_ACCEPTED | HOOK_ORDER_EVENT,
        Tag::Triggered => HOOK_ORDER_TRIGGERED | HOOK_ORDER_EVENT,
        Tag::Filled => HOOK_ORDER_FILLED | HOOK_ORDER_EVENT,
        Tag::Canceled => HOOK_ORDER_CANCELED | HOOK_ORDER_EVENT,
        Tag::Resized => HOOK_ORDER_EVENT,
        Tag::Expired => HOOK_ORDER_EXPIRED | HOOK_ORDER_EVENT,
        Tag::Rejected => HOOK_ORDER_REJECTED | HOOK_ORDER_EVENT,
        Tag::Contract => HOOK_CONTRACT_EVENT,
        Tag::Corporate => HOOK_CORPORATE_ACTION,
    }
}

/// What the cache records of an order event: status, client id, price, size
/// and refusal reason.
type OrderView<'a> = (&'static str, &'a str, Option<f64>, Option<f64>, Option<&'static str>);

fn order_view(event: &EngineEvent) -> Option<OrderView<'_>> {
    match event {
        EngineEvent::OrderAccepted { client_id, .. } => {
            Some(("accepted", client_id, None, None, None))
        }
        EngineEvent::OrderTriggered { client_id, .. } => {
            Some(("triggered", client_id, None, None, None))
        }
        EngineEvent::OrderFilled { client_id, price, size, .. } => {
            Some(("filled", client_id, Some(*price), Some(*size), None))
        }
        EngineEvent::OrderCanceled { client_id, .. } => {
            Some(("canceled", client_id, None, None, None))
        }
        EngineEvent::OrderExpired { client_id, .. } => {
            Some(("expired", client_id, None, None, None))
        }
        EngineEvent::OrderRejected { client_id, reason, .. } => {
            Some(("rejected", client_id, None, None, Some(*reason)))
        }
        _ => None,
    }
}

static CACHE_TYPES: PyOnceLock<(Py<PyAny>, Py<PyAny>)> = PyOnceLock::new();

/// Fold one engine event into the strategy's `Cache` (what
/// `Cache._observe` does), for the exact `Cache` class. `trade` is the
/// completed trade of an `exited` event.
fn observe(
    py: Python<'_>,
    cache: &Bound<'_, PyAny>,
    event: &EngineEvent,
    trade: Option<&PyTrade>,
    symbol: Option<&Bound<'_, PyString>>,
) -> PyResult<()> {
    if let Some((status, client_id, price, size, reason)) = order_view(event) {
        if client_id.is_empty() {
            return Ok(());
        }
        let orders = cache.getattr(intern!(py, "_orders"))?.cast_into::<PyDict>()?;
        let record = match orders.get_item(client_id)? {
            Some(record) => record,
            None => {
                let (_, record_cls) = CACHE_TYPES.get(py).expect("set when the cache was matched");
                let record = record_cls.bind(py).call1((client_id, status))?;
                orders.set_item(client_id, &record)?;
                record
            }
        };
        record.setattr(intern!(py, "status"), status)?;
        if let Some(symbol) = symbol {
            record.setattr(intern!(py, "symbol"), symbol)?;
        }
        if let Some(price) = price {
            record.setattr(intern!(py, "last_price"), price)?;
        }
        if let Some(size) = size {
            record.setattr(intern!(py, "last_size"), size)?;
        }
        if let Some(reason) = reason {
            record.setattr(intern!(py, "reject_reason"), reason)?;
        }
    } else if let (EngineEvent::Exited { .. }, Some(trade)) = (event, trade) {
        cache
            .getattr(intern!(py, "_closed_trades"))?
            .cast_into::<PyList>()?
            .append(Py::new(py, trade.clone())?)?;
    }
    Ok(())
}

/// Whether `cache` is the exact `Cache` class, whose `_observe` is done here.
fn plain_cache(py: Python<'_>, cache: &Bound<'_, PyAny>) -> PyResult<bool> {
    let (cache_cls, _) = CACHE_TYPES.get_or_try_init(py, || {
        let module = py.import("raptorbt.strategy.cache")?;
        Ok::<_, PyErr>((module.getattr("Cache")?.unbind(), module.getattr("OrderRecord")?.unbind()))
    })?;
    Ok(cache.get_type().is(cache_cls.bind(py)))
}

/// Route engine events into the strategy's hooks (what `dispatch_events` in
/// `_commands.py` does).
///
/// The `entered`/`exited` events double as fill notifications for the legacy
/// intent path; an order-driven fill already fired `on_order_filled` for its
/// own `order_filled` event, so the position event right after it must not
/// fire it again. An event object is built only when a hook that reads it is
/// overridden, or the cache needs it.
fn dispatch_events(
    py: Python<'_>,
    strategy: &Bound<'_, PyAny>,
    ctx: &Bound<'_, PyAny>,
    mask: u32,
    events: Vec<EngineEvent>,
    symbol: Option<&Bound<'_, PyString>>,
) -> PyResult<()> {
    let cache = strategy.getattr(intern!(py, "cache"))?;
    let native_cache = plain_cache(py, &cache)?;
    let mut preceded = false;
    for event in events {
        let tag = tag_of(&event);
        let hooks = hooks_of(tag, preceded) & mask;
        // An `exited` event carries its trade into the cache.
        let object = if hooks != 0 || !native_cache || tag == Tag::Exited {
            Some(Py::new(py, PyEngineEvent::from(event.clone()))?)
        } else {
            None
        };
        if native_cache {
            let trade = object.as_ref().and_then(|o| o.borrow(py).trade.clone());
            observe(py, &cache, &event, trade.as_ref(), symbol)?;
        } else {
            let object = object.as_ref().expect("built for a custom cache");
            cache.call_method1(intern!(py, "_observe"), (object, symbol))?;
        }
        let call = |bit: u32, name: &Bound<'_, PyString>| -> PyResult<()> {
            if hooks & bit != 0 {
                let object = object.as_ref().expect("built for an overridden hook");
                strategy.call_method1(name, (ctx, object))?;
            }
            Ok(())
        };
        match tag {
            Tag::Entered => {
                call(HOOK_ORDER_FILLED, intern!(py, "on_order_filled"))?;
                call(HOOK_POSITION_OPENED, intern!(py, "on_position_opened"))?;
                preceded = false;
            }
            Tag::Exited => {
                call(HOOK_ORDER_FILLED, intern!(py, "on_order_filled"))?;
                call(HOOK_POSITION_CLOSED, intern!(py, "on_position_closed"))?;
                preceded = false;
            }
            Tag::EntryRejected => call(HOOK_ORDER_REJECTED, intern!(py, "on_order_rejected"))?,
            Tag::AlgoStarted => {
                call(HOOK_ALGO_STARTED, intern!(py, "on_algo_started"))?;
                call(HOOK_ORDER_EVENT, intern!(py, "on_order_event"))?;
            }
            Tag::AlgoCompleted => {
                call(HOOK_ALGO_COMPLETED, intern!(py, "on_algo_completed"))?;
                call(HOOK_ORDER_EVENT, intern!(py, "on_order_event"))?;
            }
            Tag::MarginCall => {
                call(HOOK_MARGIN_CALL, intern!(py, "on_margin_call"))?;
                call(HOOK_ORDER_EVENT, intern!(py, "on_order_event"))?;
            }
            // Risk events reach `on_risk_event` alone and leave `preceded` as it was.
            Tag::RiskRejected | Tag::RiskHalted => {
                call(HOOK_RISK_EVENT, intern!(py, "on_risk_event"))?;
            }
            // A resize reaches `on_order_event` alone and leaves `preceded` as it was.
            Tag::Resized => call(HOOK_ORDER_EVENT, intern!(py, "on_order_event"))?,
            // Lifecycle events reach their own hook and leave `preceded` as it was.
            Tag::Contract => call(HOOK_CONTRACT_EVENT, intern!(py, "on_contract_event"))?,
            Tag::Corporate => call(HOOK_CORPORATE_ACTION, intern!(py, "on_corporate_action"))?,
            order => {
                let (bit, name) = match order {
                    Tag::Accepted => (HOOK_ORDER_ACCEPTED, intern!(py, "on_order_accepted")),
                    Tag::Triggered => (HOOK_ORDER_TRIGGERED, intern!(py, "on_order_triggered")),
                    Tag::Filled => (HOOK_ORDER_FILLED, intern!(py, "on_order_filled")),
                    Tag::Canceled => (HOOK_ORDER_CANCELED, intern!(py, "on_order_canceled")),
                    Tag::Expired => (HOOK_ORDER_EXPIRED, intern!(py, "on_order_expired")),
                    _ => (HOOK_ORDER_REJECTED, intern!(py, "on_order_rejected")),
                };
                call(bit, name)?;
                call(HOOK_ORDER_EVENT, intern!(py, "on_order_event"))?;
                preceded = order == Tag::Filled;
            }
        }
    }
    Ok(())
}

fn exit_reason_name(reason: crate::core::types::ExitReason) -> &'static str {
    use crate::core::types::ExitReason as R;
    match reason {
        R::Signal => "signal",
        R::StopLoss => "stop_loss",
        R::TakeProfit => "take_profit",
        R::TrailingStop => "trailing_stop",
        R::EndOfData => "end_of_data",
        R::Order => "order",
        R::Settlement => "settlement",
        R::Delivery => "delivery",
        R::CorporateAction => "corporate_action",
        R::Liquidation => "liquidation",
        R::TimeExit => "time_exit",
        R::Squareoff => "squareoff",
    }
}

/// Put on the record what the engine did to orders, positions or cash on
/// its own, before the strategy hears of it: a cut to a working order, a
/// position closed by an engine rule (stop, target, trail, liquidation,
/// time exit, square-off), an order or entry the book refused, a contract settled, a corporate action applied
/// or refused. What an order or the market did is already on the record as
/// the input that caused it; these have no order of their own.
fn journal_engine_events(
    sh: &Shared,
    st: &Strat,
    events: &[EngineEvent],
    from_report: bool,
) -> PyResult<()> {
    use crate::core::types::ExitReason as R;
    if !sh.journal_on.get() {
        return Ok(());
    }
    let sid = JV::Int(st.sid.index() as i64);
    for event in events {
        match event {
            EngineEvent::OrderResized { order_id, client_id, from, to, .. } => {
                sh.jlog_now(jk::ORDER_RESIZED, jsrc::ENGINE, Some(client_id), || {
                    JV::map([
                        ("strategy", sid.clone()),
                        ("order_id", JV::Int(*order_id as i64)),
                        ("from", JV::Float(*from)),
                        ("to", JV::Float(*to)),
                    ])
                })?;
            }
            EngineEvent::Exited { trade, .. }
                if !matches!(
                    trade.exit_reason,
                    R::Signal | R::Order | R::EndOfData | R::Settlement | R::CorporateAction
                ) =>
            {
                sh.jlog_now(jk::ENGINE_EXIT, jsrc::ENGINE, None, || {
                    JV::map([
                        ("strategy", sid.clone()),
                        ("reason", JV::str(exit_reason_name(trade.exit_reason))),
                        ("price", JV::Float(trade.exit_price)),
                        ("size", JV::Float(trade.size)),
                        ("direction", JV::Int(trade.direction as i64)),
                    ])
                })?;
            }
            EngineEvent::RiskHalted { reason, .. } if reason == "daily_loss" => {
                sh.jlog_now(jk::HALT, jsrc::ENGINE, None, || {
                    JV::map([
                        ("strategy", sid.clone()),
                        ("reason", JV::str("daily_loss")),
                        ("flatten", JV::Bool(false)),
                    ])
                })?;
            }
            EngineEvent::FillBreach { order_id, client_id, reason, .. } => {
                sh.jlog_now(jk::REJECTION, jsrc::ENGINE, Some(client_id), || {
                    JV::map([
                        ("strategy", sid.clone()),
                        ("event", JV::str("fill_breach")),
                        ("order_id", JV::Int(*order_id as i64)),
                        ("reason", JV::str(*reason)),
                    ])
                })?;
            }
            // A venue's own refusal is the `report` record, with its reason.
            EngineEvent::OrderRejected { .. } | EngineEvent::EntryRejected { .. }
                if from_report => {}
            EngineEvent::OrderRejected { order_id, client_id, reason, .. } => {
                sh.jlog_now(jk::REJECTION, jsrc::ENGINE, Some(client_id), || {
                    JV::map([
                        ("strategy", sid.clone()),
                        ("event", JV::str("order_rejected")),
                        ("order_id", JV::Int(*order_id as i64)),
                        ("reason", JV::str(*reason)),
                    ])
                })?;
            }
            EngineEvent::EntryRejected { reason, .. } => {
                sh.jlog_now(jk::REJECTION, jsrc::ENGINE, None, || {
                    JV::map([
                        ("strategy", sid.clone()),
                        ("event", JV::str("entry_rejected")),
                        ("reason", JV::str(reason.as_str())),
                    ])
                })?;
            }
            EngineEvent::Settled {
                outcome,
                style,
                source,
                price,
                size,
                direction,
                deliver,
                delivery_price,
                ..
            } => {
                sh.jlog_now(jk::LIFECYCLE, jsrc::ENGINE, None, || {
                    JV::map([
                        ("strategy", sid.clone()),
                        ("what", JV::str("settlement")),
                        ("outcome", JV::str(*outcome)),
                        ("style", JV::str(style.as_str())),
                        ("source", JV::str(source.as_str())),
                        ("price", JV::Float(*price)),
                        ("size", JV::Float(*size)),
                        ("direction", JV::Int(*direction as i64)),
                        ("deliver", JV::Float(*deliver)),
                        ("delivery_price", JV::Float(*delivery_price)),
                    ])
                })?;
            }
            EngineEvent::CorporateAction { kind, ratio, amount, cash, .. } => {
                sh.jlog_now(jk::LIFECYCLE, jsrc::ENGINE, None, || {
                    JV::map([
                        ("strategy", sid.clone()),
                        ("what", JV::str("corporate_action")),
                        ("outcome", JV::str(kind.as_str())),
                        ("ratio", JV::Float(*ratio)),
                        ("amount", JV::Float(*amount)),
                        ("cash", JV::Float(*cash)),
                    ])
                })?;
            }
            EngineEvent::LifecycleRefused { what, reason, .. } => {
                sh.jlog_now(jk::LIFECYCLE, jsrc::ENGINE, None, || {
                    JV::map([
                        ("strategy", sid.clone()),
                        ("what", JV::str(*what)),
                        ("outcome", JV::str("refused")),
                        ("reason", JV::str(reason)),
                    ])
                })?;
            }
            _ => {}
        }
    }
    Ok(())
}

/// Route engine events to the strategy's hooks, natively or through the
/// Python command path.
fn deliver(
    py: Python<'_>,
    sh: &Shared,
    st: &Strat,
    events: Vec<EngineEvent>,
    symbol: &Bound<'_, PyString>,
) -> PyResult<()> {
    journal_engine_events(sh, st, &events, false)?;
    heard(py, sh, st, events, &mut |events| deliver_to_hooks(py, sh, st, events, symbol))
}

/// Run `call` over `events`, each fill alone, with the strategy's queue told which fill it is
/// hearing: an order queued meanwhile is the answer to that fill, and no other order is. Under
/// `next_bar_open` the hook is also blind to the bars of the fill's instant (see
/// [`blind_to_the_instant`]).
fn heard(
    py: Python<'_>,
    sh: &Shared,
    st: &Strat,
    events: Vec<EngineEvent>,
    call: &mut dyn FnMut(Vec<EngineEvent>) -> PyResult<()>,
) -> PyResult<()> {
    // A fill and the position event that follows it are one chunk (the second does not fire
    // `on_order_filled` again); a new fill starts a new one.
    let fills = |e: &EngineEvent| matches!(e, EngineEvent::OrderFilled { .. });
    if events.iter().filter(|e| fills(e)).count() > 1 {
        let mut chunk: Vec<EngineEvent> = Vec::new();
        for event in events {
            if fills(&event) && chunk.iter().any(fills) {
                heard(py, sh, st, std::mem::take(&mut chunk), call)?;
            }
            chunk.push(event);
        }
        return heard(py, sh, st, chunk, call);
    }
    // A position's own stop that the bar opened through leaves as `Exited` alone (no order fill):
    // at the open it is a fill at the open, as a resting order's is, and its hook is answered there.
    let exited_at_open = sh.resting.get()
        && sh.with_book(st.sid, |b, _| Ok(b.kernel(0).fills_at_next_open())).unwrap_or(false)
        && events.iter().any(|e| {
            matches!(e, EngineEvent::Exited { trade, .. } if Some(trade.exit_price) == sh.rest_open.get())
        });
    let fill_at = match (events.iter().any(fills) || exited_at_open, &*sh.inner.borrow()) {
        (true, Inner::Running(c)) => c.current().map(|cur| cur.event.ts),
        _ => None,
    };
    let blind = fill_at.is_some() || sh.matching.get();
    // Under `next_bar_open` only the fills at the open of a bar, heard before the strategy has
    // seen the bar, can be answered at that open; a fill the bar brings is answered at the next.
    let at_open = fill_at.is_some()
        && sh.with_book(st.sid, |b, _| Ok(b.kernel(0).fills_at_next_open())).unwrap_or(false);
    // A resting order the bar gapped through filled at the bar's open: heard before the bar,
    // answered at that open. One that traded later in the bar is answered at the next open.
    let gapped = at_open
        && sh.resting.get()
        && (exited_at_open
            || events.iter().any(|e| {
                matches!(e, EngineEvent::OrderFilled { price, .. } if Some(*price) == sh.rest_open.get())
            }));
    if fill_at.is_some() && (!at_open || sh.opening.get() || gapped) {
        st.queue.bind(py).borrow_mut().answering = fill_at;
    }
    if blind {
        blind_to_the_instant(sh, st.sid);
    }
    let delivered = call(events);
    if fill_at.is_some() {
        st.queue.bind(py).borrow_mut().answering = None;
    }
    if blind {
        sh.blind.borrow_mut().clear();
    }
    delivered
}

/// Under `next_bar_open` a fill is at the open of its bar, and what the hook that hears it may know
/// is what was known at the open: the bars of the instant in flight (every instrument of the open
/// group) are hidden from its market data until the hook is done.
fn blind_to_the_instant(sh: &Shared, sid: StrategyId) {
    let at_open = sh.with_book(sid, |b, _| Ok(b.kernel(0).fills_at_next_open())).unwrap_or(false);
    if !at_open && !sh.resting.get() {
        return;
    }
    let mut blind = sh.blind.borrow_mut();
    blind.clear();
    if let Inner::Running(c) = &*sh.inner.borrow() {
        let n = c.group_len();
        if n == 0 {
            if let Some(cur) = c.current() {
                if cur.event.kind == EventKind::Bar {
                    blind.push(cur.event.instrument);
                }
            }
        }
        for i in 0..n {
            let e = c.group_event(i).event;
            if e.kind == EventKind::Bar {
                blind.push(e.instrument);
            }
        }
    }
}

fn deliver_to_hooks(
    py: Python<'_>,
    sh: &Shared,
    st: &Strat,
    events: Vec<EngineEvent>,
    symbol: &Bound<'_, PyString>,
) -> PyResult<()> {
    let strategy = st.strategy.bind(py);
    let ctx = st.ctx.bind(py);
    if matches!(st.route, Route::Native(_)) {
        return dispatch_events(
            py,
            strategy,
            ctx.as_any(),
            live_mask(py, &st.queue, st.mask),
            events,
            Some(symbol),
        );
    }
    let list = PyList::new(
        py,
        events
            .into_iter()
            .map(|e| Py::new(py, PyEngineEvent::from(e)))
            .collect::<PyResult<Vec<_>>>()?,
    )?;
    sh.classes.dispatch.bind(py).call1((strategy, ctx, list))?;
    Ok(())
}

/// The step: queued commands, then the book's step, then engine events to
/// hooks and the work those queued.
///
/// `fold` is whether the event can carry an entry: bars and prints fold the
/// intents (and those held for the symbol) into the step; a quote or a book
/// steps with none, and what its hooks queued is held for the symbol's next
/// step. `hooked` says whether a hook that could have queued work ran, so an
/// event nobody handles skips the queue checks.
/// Refuse an entry whose explicit stop or target is off the instrument's tick
/// grid (the same rule as an order's prices). Levels derived from a configured
/// percentage or ATR multiple are rounded conservatively by the engine instead.
fn check_intent_grid(
    sh: &Shared,
    sid: StrategyId,
    local: usize,
    input: &StepInput,
) -> PyResult<()> {
    if input.stop_price_override.is_none() && input.target_price_override.is_none() {
        return Ok(());
    }
    let grid = sh.with_book(sid, |b, _| Ok(b.kernel(local).order_grid()))?;
    crate::python::strategy_bindings::check_grid(
        &grid,
        None,
        &[("stop_price", input.stop_price_override), ("target_price", input.target_price_override)],
    )
}

#[allow(clippy::too_many_arguments)]
fn step(
    py: Python<'_>,
    sh: &Shared,
    st: &Strat,
    ev: &Ev<'_, '_>,
    local: usize,
    fold: bool,
    hooked: bool,
    mut stash: Option<&mut Decided>,
) -> PyResult<()> {
    let matched = step_match(py, sh, st, ev, local, fold, hooked, &mut stash)?;
    step_deliver(py, sh, st, ev, local, fold, hooked, &mut stash, matched)
}

/// What a strategy's step matched, before the hooks that hear it run.
struct Matched {
    events: Vec<EngineEvent>,
    /// What a simulated venue reported for the event, booked, and the events that made, not yet
    /// told to the strategy; whether the strategy heard of anything.
    venue_held: venue::HeldEvents,
    venue_heard: bool,
}

/// The first half of a step: queued commands, then the book's step. Nothing here reaches a hook.
#[allow(clippy::too_many_arguments)]
fn step_match(
    py: Python<'_>,
    sh: &Shared,
    st: &Strat,
    ev: &Ev<'_, '_>,
    local: usize,
    fold: bool,
    hooked: bool,
    stash: &mut Option<&mut Decided>,
) -> PyResult<Matched> {
    // With a `stash`, the context still shows another event of the group and the
    // strategy's queue is empty (see `dispatch_step`): it is put back before
    // anything that could read it.
    let restore = |stash: &mut Option<&mut Decided>| {
        if let Some(d) = stash.take() {
            unstash(py, st, d, true);
        }
    };
    let sid = st.sid;
    let local_idx = ev.event.local_idx as usize;
    let ts = ev.event.ts;
    let symbol = ev.symbol;
    if sh.warmup_muted.get() {
        discard_queued(py, st);
    }

    // 4. Commands, only when work is queued; then the step. Sent to a venue,
    // the entry and close intents go first: in-process they fill ahead of
    // the market orders the step sweeps.
    let mut venue_intents = None;
    // An intent an open hook queued was decided at the open, before this bar traded: its ATR is
    // the previous bar's.
    let atr_idx = if sh.open_intents.borrow().contains_key(&(sid.index(), local)) {
        local_idx.saturating_sub(1)
    } else {
        local_idx
    };
    if sh.venue_on.get() && fold && (has_intents(py, st) || st.has_held(py)) {
        let atr = match &*sh.inner.borrow() {
            Inner::Running(c) => c.atr_at(sid, local, atr_idx),
            _ => return Err(finished()),
        };
        let input = fold_step(py, st, symbol, local, local_idx, atr)?;
        refuse_intents(sh, &input)?;
        check_intent_grid(sh, sid, local, &input)?;
        let venue = sh.venue.borrow().clone();
        if let Some(venue) = venue {
            venue.intents(sh, st, local, &input, local_idx, ts)?;
        }
        venue_intents = Some(());
    }
    if stash.is_none() && (fold || hooked) {
        drain_commands(py, sh, st, local, ev)?;
    }
    let atr = if fold {
        match &*sh.inner.borrow() {
            Inner::Running(c) => c.atr_at(sid, local, atr_idx),
            _ => return Err(finished()),
        }
    } else {
        0.0
    };
    let any_held = match &st.route {
        Route::Python { held, .. } => !held.bind(py).is_empty(),
        Route::Native(state) => state.borrow().has_held(),
    };
    // A venue holds the positions: its intents became orders above, and the
    // book steps with none.
    if any_held {
        restore(stash);
    }
    let input = if venue_intents.is_none()
        && fold
        && (any_held || (stash.is_none() && has_intents(py, st)))
    {
        let mut input = fold_step(py, st, symbol, local, local_idx, atr)?;
        input.size_ref = sh.open_intents.borrow_mut().remove(&(sid.index(), local));
        refuse_intents(sh, &input)?;
        refuse_entry_intent(sh, sid, local, local_idx, ts, &mut input)?;
        check_intent_grid(sh, sid, local, &input)?;
        input
    } else {
        StepInput { atr, ..StepInput::default() }
    };
    let halted_before = sh.halted(sid);
    let events = match &mut *sh.inner.borrow_mut() {
        Inner::Running(c) => c.apply(sid, input).map_err(runtime_err)?,
        _ => return Err(finished()),
    };
    sh.journal_halt_edge(sid, halted_before, &events)?;

    // A simulated venue steps with the event now, with the rest of the match, and what it
    // reported is booked now; what the strategy's hooks hear of it waits until every instrument
    // of the round is matched.
    let (mut venue_held, mut venue_heard) = (Vec::new(), false);
    if sh.venue_on.get() {
        let venue = sh.venue.borrow().clone();
        if let Some(venue) = venue {
            venue.flush(sh, st, local, local_idx, ts)?;
            if let Some(out) = venue.poll_step(sh, st, ev, local, atr)? {
                (venue_heard, venue_held) = venue.poll_book(py, sh, st, ev, out)?;
            }
        }
    }
    Ok(Matched { events, venue_held, venue_heard })
}

/// The second half of a step: the engine events to hooks, and the work those queued.
#[allow(clippy::too_many_arguments)]
fn step_deliver(
    py: Python<'_>,
    sh: &Shared,
    st: &Strat,
    ev: &Ev<'_, '_>,
    local: usize,
    fold: bool,
    hooked: bool,
    stash: &mut Option<&mut Decided>,
    matched: Matched,
) -> PyResult<()> {
    sh.matching.set(true);
    let done = step_deliver_inner(py, sh, st, ev, local, fold, hooked, stash, matched);
    sh.matching.set(false);
    done
}

#[allow(clippy::too_many_arguments)]
fn step_deliver_inner(
    py: Python<'_>,
    sh: &Shared,
    st: &Strat,
    ev: &Ev<'_, '_>,
    local: usize,
    fold: bool,
    hooked: bool,
    stash: &mut Option<&mut Decided>,
    matched: Matched,
) -> PyResult<()> {
    let restore = |stash: &mut Option<&mut Decided>| {
        if let Some(d) = stash.take() {
            unstash(py, st, d, true);
        }
    };
    let Matched { events, venue_held, venue_heard } = matched;
    let symbol = ev.symbol;
    let local_idx = ev.event.local_idx as usize;
    let ts = ev.event.ts;
    // 5. Engine events to hooks; 6. work those hooks queued.
    let mut dispatched = !events.is_empty();
    if dispatched {
        restore(stash);
        deliver(py, sh, st, events, symbol)?;
    }
    // 5b. A simulated venue answers once the strategy has seen the event: the reports it made
    // were booked in the match and are heard like a real venue's.
    if sh.venue_on.get() {
        let venue = sh.venue.borrow().clone();
        if let Some(venue) = venue {
            venue.hear_held(py, sh, st, venue_held)?;
            dispatched |= venue_heard;
        }
    }
    if stash.is_none()
        && (dispatched || (!fold && hooked))
        && (has_commands(py, st) || has_intents(py, st))
    {
        settle(py, sh, st, ev.event.instrument, local, local_idx, ts)?;
        meet_follow_ups(py, sh, st, ev, local, local_idx)?;
    }
    Ok(())
}

/// Drop a reference the thread holds the GIL for. `Py`'s own drop first asks
/// the thread-local interpreter state whether the thread is attached; the
/// `Bound` made here has the token and releases directly, which matters per
/// strategy per event.
fn release(py: Python<'_>, object: Option<Py<PyAny>>) {
    if let Some(object) = object {
        drop(object.into_bound(py));
    }
}

/// Fill the orders the hooks sent in answer to a fill (`FOLLOW_UP_TAG`) on the bar the fill came
/// in, deliver what that did, and settle what those hooks queue in turn. A chain of answers is
/// bounded: an answer that keeps answering is a strategy bug, refused rather than looped.
fn meet_follow_ups(
    py: Python<'_>,
    sh: &Shared,
    st: &Strat,
    ev: &Ev<'_, '_>,
    local: usize,
    local_idx: usize,
) -> PyResult<()> {
    let ts = ev.event.ts;
    for _ in 0..FOLLOW_UP_DEPTH {
        if sh.venue_on.get() {
            let venue = sh.venue.borrow().clone();
            if let Some(venue) = venue {
                if venue.hear_same_event(py, sh, st, ev, local)?
                    && (has_commands(py, st) || has_intents(py, st))
                {
                    settle(py, sh, st, ev.event.instrument, local, local_idx, ts)?;
                    continue;
                }
            }
        }
        let pending: Vec<_> = std::mem::take(&mut *sh.follow_ups.borrow_mut());
        if pending.is_empty() {
            return Ok(());
        }
        for (sid, target, order_id) in pending {
            let events = match &mut *sh.inner.borrow_mut() {
                Inner::Running(core) => core.fill_follow_up(sid, target, order_id, ts),
                _ => Vec::new(),
            };
            if events.is_empty() {
                continue;
            }
            let global = sh.global_of(sid, target)?;
            let symbols = sh.symbols.get();
            let symbol = symbols[global.index()].bind(py);
            deliver(py, sh, st, events, symbol)?;
        }
        if has_commands(py, st) || has_intents(py, st) {
            settle(py, sh, st, ev.event.instrument, local, local_idx, ts)?;
        }
    }
    if sh.follow_ups.borrow().iter().all(|(sid, _, _)| *sid != st.sid) {
        return Ok(());
    }
    // The answer that would be the next link is not booked: it is cancelled with the strategy's
    // other working orders, and the strategy is halted with the reason on the journal.
    halt_follow_up_loop(sh, st.sid)
}

/// Halt a strategy whose answers to fills kept answering fills (`follow_up_loop`): the pending
/// answers are dropped and cancelled before they fill, entries are refused, and the halt is on the
/// journal with that reason. The node keeps running.
fn halt_follow_up_loop(sh: &Shared, sid: StrategyId) -> PyResult<()> {
    const REASON: &str = "follow_up_loop";
    sh.follow_ups.borrow_mut().retain(|(s, _, _)| *s != sid);
    sh.manual_halts.borrow_mut().insert(sid.index());
    sh.jlog_now(jk::HALT, jsrc::ENGINE, None, || {
        JV::map([
            ("strategy", JV::Int(sid.index() as i64)),
            ("reason", JV::str(REASON)),
            ("flatten", JV::Bool(false)),
            ("depth", JV::Int(FOLLOW_UP_DEPTH as i64)),
        ])
    })?;
    if sh.venue_on.get() {
        let venue = sh.venue.borrow().clone();
        if let Some(venue) = venue {
            return venue.halt(sh, sid, REASON, false);
        }
    }
    sh.with_book_mut(sid, |b, _| {
        b.halt(REASON, false);
        Ok(())
    })
}

/// How many answers to fills one step settles before it refuses to go on.
const FOLLOW_UP_DEPTH: usize = 8;

/// What a strategy's handlers left of one event of a timestamp group: where
/// the context pointed and the work queued, kept until the group is matched.
struct Decided {
    sid: StrategyId,
    local: u32,
    idx: u32,
    fold: bool,
    hooked: bool,
    bar: Option<Py<PyAny>>,
    close_ts: Option<i64>,
    /// What the handlers queued; `None` is the common case, nothing.
    work: Option<Box<Work>>,
}

/// The commands and intents a decision queued.
#[derive(Default)]
struct Work {
    commands: Vec<crate::runtime::commands::Command>,
    intents: Vec<crate::runtime::commands::Intent>,
}

impl Decided {
    fn no_commands(&self) -> bool {
        self.work.as_ref().is_none_or(|w| w.commands.is_empty())
    }
}

/// The step of an event, or, in a timestamp group, the strategy's decision
/// kept for the match phase: its handlers have run, the commands and intents
/// they queued are taken off the queue and the context's pointer is noted.
#[allow(clippy::too_many_arguments)]
fn finish_event(
    py: Python<'_>,
    sh: &Shared,
    st: &Strat,
    ev: &Ev<'_, '_>,
    local: usize,
    fold: bool,
    hooked: bool,
    defer: Option<&mut Vec<Decided>>,
) -> PyResult<()> {
    let Some(out) = defer else {
        return step(py, sh, st, ev, local, fold, hooked, None);
    };
    if let Inner::Running(c) = &mut *sh.inner.borrow_mut() {
        c.decide_view(st.sid, ev.pos.get(), false);
    }
    let ctx = st.ctx.bind(py).borrow();
    let mut q = st.queue.bind(py).borrow_mut();
    // Nothing queued is the common case: no allocation either way.
    let (commands, intents) = (q.queue.take_commands(), q.queue.take_intents());
    out.push(Decided {
        sid: st.sid,
        local: local as u32,
        idx: ctx.idx.get() as u32,
        fold,
        hooked,
        bar: ctx.bar.borrow().as_ref().map(|b| b.clone_ref(py)),
        close_ts: ctx.close_ts.get(),
        work: (!commands.is_empty() || !intents.is_empty())
            .then(|| Box::new(Work { commands, intents })),
    });
    Ok(())
}

/// Put a decision back where the handlers left it: the context's pointer and
/// the work the strategy queued.
///
/// With `last`, the decision is not needed again: its bar moves into the
/// context instead of being copied.
fn unstash(py: Python<'_>, st: &Strat, d: &mut Decided, last: bool) {
    let ctx = st.ctx.bind(py).borrow();
    ctx.local.set(d.local as usize);
    ctx.idx.set(d.idx as usize);
    let bar = if last { d.bar.take() } else { d.bar.as_ref().map(|b| b.clone_ref(py)) };
    release(py, ctx.bar.replace(bar));
    ctx.close_ts.set(d.close_ts);
    let Some(work) = d.work.as_mut() else { return };
    if work.commands.is_empty() && work.intents.is_empty() {
        return;
    }
    let mut q = st.queue.bind(py).borrow_mut();
    q.queue.restore_commands(std::mem::take(&mut work.commands));
    q.queue.restore_intents(std::mem::take(&mut work.intents));
}

/// Take a strategy's queued work back into its decision.
fn restash(py: Python<'_>, st: &Strat, d: &mut Decided) {
    let mut q = st.queue.bind(py).borrow_mut();
    let (commands, intents) = (q.queue.take_commands(), q.queue.take_intents());
    d.work =
        (!commands.is_empty() || !intents.is_empty()).then(|| Box::new(Work { commands, intents }));
}

/// First half of the match phase: the commands each strategy queued on this
/// event reach their instruments' kernels. Every instrument of the group gets
/// its commands before any kernel steps, so an order decided on one
/// instrument for another meets the bar the other prints at the same
/// timestamp, whichever of the two is stepped first.
///
/// Sent to a venue, the order of entries, closes and commands is the venue
/// route's own (see `step`), so nothing is sent early.
///
/// Commands are routed in three passes over the whole group (`pass` 0, 1, 2):
/// the orders that name no parent, then the orders that do, then everything
/// else (cancel, modify, close, link). A cancel or parent that names an order
/// another event of the group created therefore finds it, whatever order the
/// events are in.
fn dispatch_route(
    py: Python<'_>,
    sh: &Shared,
    ls: &mut LoopState,
    event: &EventRef,
    decided: &mut [Decided],
    pass: u8,
) -> PyResult<()> {
    if sh.venue_on.get() {
        return Ok(());
    }
    let ts_obj = event.ts.into_pyobject(py)?;
    let symbols = sh.symbols.get();
    let symbol = symbols[event.instrument.index()].bind(py);
    let with_ticks = sh.has_ticks.get()[event.instrument.index()];
    let ev = Ev { event, ts_obj: &ts_obj, symbol, with_ticks, pos: Cell::new(0) };
    for d in decided.iter_mut() {
        if sh.warmup_muted.get() || !(d.fold || d.hooked) || d.no_commands() {
            continue;
        }
        let st = ls.by_sid[d.sid.index()].as_ref().expect("subscribed");
        // `unstash` below hands the intents back with the commands; a pass
        // keeps only its own commands in the queue.
        let class = |c: &crate::runtime::commands::Command| -> u8 {
            use crate::runtime::commands::Command;
            match c {
                Command::Submit { parent, .. } if parent.as_deref().is_none_or(str::is_empty) => 0,
                Command::Submit { .. } => 1,
                _ => 2,
            }
        };
        let work = d.work.as_mut().expect("commands are work");
        let (now, later): (Vec<_>, Vec<_>) =
            std::mem::take(&mut work.commands).into_iter().partition(|c| class(c) == pass);
        if now.is_empty() {
            work.commands = later;
            continue;
        }
        work.commands = now;
        unstash(py, st, d, false);
        let routed = drain_commands(py, sh, st, d.local as usize, &ev);
        restash(py, st, d);
        // `restash` replaced the work: the commands of the later passes go back.
        d.work.get_or_insert_with(Default::default).commands = later;
        routed?;
    }
    Ok(())
}

/// One strategy's step of an event of a round, matched and not yet heard: where the context
/// pointed, so the hooks that hear it can be run once the whole round is matched.
struct Pending {
    sid: StrategyId,
    local: u32,
    idx: u32,
    fold: bool,
    hooked: bool,
    bar: Option<Py<PyAny>>,
    close_ts: Option<i64>,
    matched: Matched,
}

/// The match half of a step for one event of a round: queued commands reach the kernel and it
/// steps, and nothing is told to a hook. The hooks wait until every event of the round is
/// matched ([`dispatch_deliver`]), so what a fill hook does can reach no bar of the round that has
/// not been matched.
fn dispatch_match(
    py: Python<'_>,
    sh: &Shared,
    ls: &mut LoopState,
    event: &EventRef,
    decided: &mut Vec<Decided>,
    out: &mut Vec<Pending>,
) -> PyResult<()> {
    let ts_obj = event.ts.into_pyobject(py)?;
    let symbols = sh.symbols.get();
    let symbol = symbols[event.instrument.index()].bind(py);
    let with_ticks = sh.has_ticks.get()[event.instrument.index()];
    let ev = Ev { event, ts_obj: &ts_obj, symbol, with_ticks, pos: Cell::new(0) };
    for mut d in decided.drain(..) {
        let st = ls.by_sid[d.sid.index()].as_ref().expect("subscribed");
        let left = &mut ls.remaining[d.sid.index()];
        *left -= 1;
        // A decision that queued nothing leaves the context where it is until a step needs it;
        // the strategy's last event of the group always puts it back.
        let bar = d.bar.as_ref().map(|b| b.clone_ref(py));
        let (local, idx, close_ts, fold, hooked, sid) =
            (d.local, d.idx, d.close_ts, d.fold, d.hooked, d.sid);
        let matched =
            if *left > 0 && d.work.is_none() && !sh.venue_on.get() && !sh.warmup_muted.get() {
                let mut stash = Some(&mut d);
                let m = step_match(py, sh, st, &ev, local as usize, fold, hooked, &mut stash)?;
                release(py, d.bar.take());
                m
            } else {
                unstash(py, st, &mut d, true);
                step_match(py, sh, st, &ev, local as usize, fold, hooked, &mut None)?
            };
        out.push(Pending { sid, local, idx, fold, hooked, bar, close_ts, matched });
    }
    Ok(())
}

/// The hearing half: the hooks hear what their strategy's step matched, and the work they queue
/// is settled. Commands from here apply from the next bar of every instrument of the round.
fn dispatch_deliver(
    py: Python<'_>,
    sh: &Shared,
    ls: &mut LoopState,
    event: &EventRef,
    pending: Vec<Pending>,
) -> PyResult<()> {
    let ts_obj = event.ts.into_pyobject(py)?;
    let symbols = sh.symbols.get();
    let symbol = symbols[event.instrument.index()].bind(py);
    let with_ticks = sh.has_ticks.get()[event.instrument.index()];
    let ev = Ev { event, ts_obj: &ts_obj, symbol, with_ticks, pos: Cell::new(0) };
    for p in pending {
        let st = ls.by_sid[p.sid.index()].as_ref().expect("subscribed");
        {
            let ctx = st.ctx.bind(py).borrow();
            ctx.local.set(p.local as usize);
            ctx.idx.set(p.idx as usize);
            release(py, ctx.bar.replace(p.bar));
            ctx.close_ts.set(p.close_ts);
        }
        step_deliver(py, sh, st, &ev, p.local as usize, p.fold, p.hooked, &mut None, p.matched)?;
    }
    Ok(())
}

/// A hook raised. It decided nothing: what it queued is dropped. The event is applied all the
/// same, as the market it describes happened, so a fill, a protective level or a mark that
/// belongs to it is not lost with the strategy's mistake. The first error is kept and returned
/// by the caller once the delivery is over.
fn hook_failed(py: Python<'_>, sh: &Shared, st: &Strat, called: PyResult<()>) {
    let Err(error) = called else { return };
    {
        let mut q = st.queue.bind(py).borrow_mut();
        drop(q.queue.take_commands());
        drop(q.queue.take_intents());
    }
    sh.hook_error.borrow_mut().get_or_insert(error);
}

/// The error a hook raised during the delivery that just ended, if any; also forgets an old one.
fn hook_error(sh: &Shared, delivery: PyResult<()>) -> PyResult<()> {
    let raised = sh.hook_error.borrow_mut().take();
    delivery?;
    raised.map_or(Ok(()), Err)
}

/// A bar: composites, the primary bar's indicators and `on_bar`, the step.
#[allow(clippy::too_many_arguments)]
fn process_bar(
    py: Python<'_>,
    sh: &Shared,
    st: &Strat,
    bank: &IndicatorBank,
    ev: &Ev<'_, '_>,
    bar: &KernelBar,
    bar_obj: &Bound<'_, PyAny>,
    scratch: &mut Scratch,
    defer: Option<&mut Vec<Decided>>,
) -> PyResult<()> {
    let t = st.tables.borrow();
    let with_ticks = ev.with_ticks;
    let first = (!with_ticks).then_some((bar_obj, bar.timestamp));
    let (local, _) = enter(py, sh, st, &t, ev, first)?;
    {
        let ctx = st.ctx.bind(py).borrow();
        if with_ticks {
            *ctx.bar.borrow_mut() = Some(bar_obj.clone().unbind());
            ctx.close_ts.set(Some(bar.timestamp));
        }
        ctx.prices.get()[local].last.set(Some(bar.close));
    }
    hook_failed(py, sh, st, composites(py, sh, st, &t, bank, scratch));
    hook_failed(py, sh, st, primary_bar(py, st, &t, bank, local));
    finish_event(py, sh, st, ev, local, true, true, defer)
}

/// A print. A primary bar it completes dispatches first, since that bar
/// closed strictly before the print, and its intents ride this print's step.
#[allow(clippy::too_many_arguments)]
fn process_trade(
    py: Python<'_>,
    sh: &Shared,
    st: &Strat,
    bank: &IndicatorBank,
    ev: &Ev<'_, '_>,
    trade: &crate::data::events::TradeTick,
    primary: Option<(&crate::core::types::OhlcvBar, &Bound<'_, PyAny>)>,
    tick: Option<&Bound<'_, PyAny>>,
    scratch: &mut Scratch,
    defer: Option<&mut Vec<Decided>>,
) -> PyResult<()> {
    let t = st.tables.borrow();
    let (local, _) = enter(py, sh, st, &t, ev, None)?;
    let ctx = st.ctx.bind(py);
    let deferred = defer.is_some();
    if let Some((bar, bar_obj)) = primary {
        {
            let c = ctx.borrow();
            *c.bar.borrow_mut() = Some(bar_obj.clone().unbind());
            c.close_ts.set(Some(bar.timestamp));
        }
        hook_failed(py, sh, st, primary_bar(py, st, &t, bank, local));
        // In a timestamp group nothing is sent anywhere until every decision
        // of the group is taken.
        if !deferred {
            drain_commands(py, sh, st, local, ev)?;
        }
    }
    hook_failed(py, sh, st, composites(py, sh, st, &t, bank, scratch));
    ctx.borrow().prices.get()[local].last.set(Some(trade.price));
    if st.mask & HOOK_TRADE != 0 {
        let tick = tick.expect("built for a strategy that overrides on_trade_tick");
        *ctx.borrow().tick.borrow_mut() = Some(tick.clone().unbind());
        let called = st.strategy.bind(py).call_method1(intern!(py, "on_trade_tick"), (ctx, tick));
        *ctx.borrow().tick.borrow_mut() = None;
        hook_failed(py, sh, st, called.map(|_| ()));
    }
    finish_event(py, sh, st, ev, local, true, true, defer)
}

/// A quote: observation only. What its hook queues is held for the symbol's
/// next step.
fn process_quote(
    py: Python<'_>,
    sh: &Shared,
    st: &Strat,
    ev: &Ev<'_, '_>,
    quote: &crate::data::events::QuoteTick,
    obj: Option<&Bound<'_, PyAny>>,
    defer: Option<&mut Vec<Decided>>,
) -> PyResult<()> {
    let t = st.tables.borrow();
    let (local, timed) = enter(py, sh, st, &t, ev, None)?;
    let ctx = st.ctx.bind(py);
    {
        let c = ctx.borrow();
        let prices = c.prices.get();
        prices[local].bid.set(Some(quote.bid));
        prices[local].ask.set(Some(quote.ask));
    }
    if st.mask & HOOK_QUOTE != 0 {
        let obj = obj.expect("built for a strategy that overrides on_quote");
        *ctx.borrow().quote.borrow_mut() = Some(obj.clone().unbind());
        let called = st.strategy.bind(py).call_method1(intern!(py, "on_quote"), (ctx, obj));
        *ctx.borrow().quote.borrow_mut() = None;
        hook_failed(py, sh, st, called.map(|_| ()));
    }
    finish_event(py, sh, st, ev, local, false, timed || st.mask & HOOK_QUOTE != 0, defer)
}

/// A book snapshot: observation only, like a quote.
fn process_book(
    py: Python<'_>,
    sh: &Shared,
    st: &Strat,
    ev: &Ev<'_, '_>,
    tick: &crate::data::book::DepthTick,
    obj: Option<&Bound<'_, PyAny>>,
    defer: Option<&mut Vec<Decided>>,
) -> PyResult<()> {
    let t = st.tables.borrow();
    let (local, timed) = enter(py, sh, st, &t, ev, None)?;
    let ctx = st.ctx.bind(py);
    {
        let c = ctx.borrow();
        if tick.bid_len > 0 {
            c.prices.get()[local].bid.set(Some(tick.bids[0].price));
        }
        if tick.ask_len > 0 {
            c.prices.get()[local].ask.set(Some(tick.asks[0].price));
        }
        c.books.borrow_mut()[local] = Some((ev.event.row, obj.map(|o| o.clone().unbind())));
    }
    if st.mask & HOOK_BOOK != 0 {
        let obj = obj.expect("built for a strategy that overrides on_order_book");
        let called = st.strategy.bind(py).call_method1(intern!(py, "on_order_book"), (ctx, obj));
        hook_failed(py, sh, st, called.map(|_| ()));
    }
    finish_event(py, sh, st, ev, local, false, timed || st.mask & HOOK_BOOK != 0, defer)
}

// -- what the journal writes about commands and reports --------------------

/// An execution report as the journal states it. Every report carries the
/// venue's own time; the fields are those of the report's constructor.
pub(crate) fn report_value(report: &crate::execution::orders::ExecutionReport) -> JV {
    use crate::execution::orders::ExecutionReport as R;
    let (name, mut fields): (&str, Vec<(&str, JV)>) = match report {
        R::Accepted { venue_order_id, ts_ns } => (
            "accepted",
            vec![("venue_order_id", JV::str(venue_order_id)), ("ts_ns", JV::Int(*ts_ns))],
        ),
        R::Rejected { reason, ts_ns } => {
            ("rejected", vec![("reason", JV::str(reason)), ("ts_ns", JV::Int(*ts_ns))])
        }
        R::Fill { exec_id, qty, price, fee, ts_ns } => (
            "fill",
            vec![
                ("exec_id", JV::str(exec_id)),
                ("qty", JV::Float(*qty)),
                ("price", JV::Float(*price)),
                ("fee", JV::Float(*fee)),
                ("ts_ns", JV::Int(*ts_ns)),
            ],
        ),
        R::CancelAck { ts_ns } => ("cancel_ack", vec![("ts_ns", JV::Int(*ts_ns))]),
        R::CancelRejected { reason, ts_ns } => {
            ("cancel_rejected", vec![("reason", JV::str(reason)), ("ts_ns", JV::Int(*ts_ns))])
        }
        R::ReplaceAck { new_qty, new_limit, new_trigger, ts_ns } => (
            "replace_ack",
            vec![
                ("new_qty", JV::opt_float(*new_qty)),
                ("new_limit", JV::opt_float(*new_limit)),
                ("new_trigger", JV::opt_float(*new_trigger)),
                ("ts_ns", JV::Int(*ts_ns)),
            ],
        ),
        R::ReplaceRejected { reason, ts_ns } => {
            ("replace_rejected", vec![("reason", JV::str(reason)), ("ts_ns", JV::Int(*ts_ns))])
        }
        R::Expired { ts_ns } => ("expired", vec![("ts_ns", JV::Int(*ts_ns))]),
        R::Correction { exec_id, qty, price, ts_ns } => (
            "correction",
            vec![
                ("exec_id", JV::str(exec_id)),
                ("qty", JV::Float(*qty)),
                ("price", JV::Float(*price)),
                ("ts_ns", JV::Int(*ts_ns)),
            ],
        ),
        R::Bust { exec_id, ts_ns } => {
            ("bust", vec![("exec_id", JV::str(exec_id)), ("ts_ns", JV::Int(*ts_ns))])
        }
    };
    fields.push(("report", JV::str(name)));
    JV::Map(fields.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
}

fn order_value(order: &crate::runtime::commands::Order) -> JV {
    use crate::runtime::commands::Order as O;
    match order {
        O::Plain(o) => JV::map([
            ("type", JV::str("order")),
            ("side", JV::str(&o.side)),
            ("kind", JV::str(&o.kind)),
            ("units", JV::opt_float(o.units)),
            ("size_frac", JV::opt_float(o.size_frac)),
            ("limit_price", JV::opt_float(o.limit_price)),
            ("trigger_price", JV::opt_float(o.trigger_price)),
            ("tif", JV::str(&o.tif)),
            ("expire_ns", o.expire_ns.map_or(JV::Null, JV::Int)),
            ("stop_price", JV::opt_float(o.stop_price)),
            ("target_price", JV::opt_float(o.target_price)),
            ("offset", JV::opt_float(o.offset)),
            ("offset_kind", JV::str(&o.offset_kind)),
            ("limit_offset", JV::Float(o.limit_offset)),
            ("post_only", JV::Bool(o.post_only)),
            ("reduce_only", JV::Bool(o.reduce_only)),
            ("tags", JV::Map(o.tags.iter().map(|(k, v)| (k.clone(), JV::str(v))).collect())),
        ]),
        O::Twap(t) => JV::map([
            ("type", JV::str("twap")),
            ("units", JV::Float(t.units)),
            ("side", JV::str(&t.side)),
            ("slices", JV::Int(i64::from(t.slices))),
            ("interval_ns", JV::Int(t.interval_ns)),
            ("reduce_only", JV::Bool(t.reduce_only)),
        ]),
    }
}

/// The batch of commands a strategy queued, as the journal states it.
pub(crate) fn commands_value(
    sid: StrategyId,
    commands: &[crate::runtime::commands::Command],
) -> JV {
    use crate::runtime::commands::Command as C;
    let one = |c: &C| match c {
        C::Submit { client_id, order, parent, symbol } => JV::map([
            ("command", JV::str("submit")),
            ("client_id", JV::str(client_id)),
            ("order", order_value(order)),
            ("parent", JV::opt_str(parent.as_deref())),
            ("symbol", JV::opt_str(symbol.as_deref())),
        ]),
        C::Cancel(id) => JV::map([("command", JV::str("cancel")), ("client_id", JV::str(id))]),
        C::CancelAll => JV::map([("command", JV::str("cancel_all"))]),
        C::LinkOco(ids) => JV::map([
            ("command", JV::str("link_oco")),
            ("client_ids", JV::List(ids.iter().map(JV::str).collect())),
        ]),
        C::Close { position_id, symbol } => JV::map([
            ("command", JV::str("close")),
            ("position_id", JV::Int(*position_id as i64)),
            ("symbol", JV::opt_str(symbol.as_deref())),
        ]),
        C::CloseAllFor(symbol) => {
            JV::map([("command", JV::str("close_all_for")), ("symbol", JV::str(symbol))])
        }
        C::Modify { client_id, units, size_frac, limit_price, trigger_price } => JV::map([
            ("command", JV::str("modify")),
            ("client_id", JV::str(client_id)),
            ("units", JV::opt_float(*units)),
            ("size_frac", JV::opt_float(*size_frac)),
            ("limit_price", JV::opt_float(*limit_price)),
            ("trigger_price", JV::opt_float(*trigger_price)),
        ]),
    };
    JV::map([
        ("strategy", JV::Int(sid.index() as i64)),
        ("commands", JV::List(commands.iter().map(one).collect())),
    ])
}

/// A refusal of a command batch: the code the engine gave.
fn refusal_value(py: Python<'_>, sid: StrategyId, err: &PyErr) -> JV {
    let value = err.value(py);
    let code = value
        .getattr("code")
        .ok()
        .and_then(|c| c.extract::<String>().ok())
        .unwrap_or_else(|| err.get_type(py).name().map(|n| n.to_string()).unwrap_or_default());
    let message = value.to_string();
    JV::map([
        ("strategy", JV::Int(sid.index() as i64)),
        ("code", JV::str(code)),
        ("message", JV::str(message.lines().next().unwrap_or(""))),
    ])
}
