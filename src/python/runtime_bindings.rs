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

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyAttributeError, PyKeyError, PyRuntimeError, PyValueError};
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
};
use crate::runtime::groups::GroupSpec;
use crate::runtime::store::{BarColumns, DepthColumns, TickColumns, TickSlices};
use crate::runtime::{
    AggId, AggKey, ColumnSource, Current, EventKind, EventRef, HeapSize, IndInput, IndKey,
    IndicatorBank, InstrumentId, InstrumentParams, LiveFeed, LocalStreamId, MarketStore,
    MemoryReport, MergeCursor, Pushed, Reveal, RingError, RingKind, RuntimeBuilder, RuntimeCore,
    SlotId, StrategyId, TickRow,
};

use super::account_bindings::{convert_group, PyAccountResult};
use super::bindings::{PyBacktestConfig, PyInstrumentConfig, PyPortfolioResult, PyTrade};
use super::errors::{Fail, Val};
use super::indicator_bindings::PyIndicator;
use super::instrument_bindings::PyInstrumentSpec;
use super::session_bindings::{
    convert_outcome, parse_instrument_args, working_order_rows, WorkingOrderRow,
};
use super::strategy_bindings::{
    atr_period, convert_snapshot, parse_account_mode, parse_qty_spec, submit_order_on,
    submit_twap_on, PyEngineEvent, PyKernelSession, PyPositionSnapshot,
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

const SERIES_KEYS: [&str; 6] = ["timestamps", "open", "high", "low", "close", "volume"];

fn runtime_err(e: RaptorError) -> PyErr {
    PyErr::from(e)
}

fn finished() -> PyErr {
    Fail::new("run_finished", "this strategy's run is finished").into()
}

fn busy(message: &str) -> PyErr {
    Fail::new("runtime_busy", message).into()
}

fn not_ready(message: &str) -> PyErr {
    Fail::new("runtime_not_ready", message).into()
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
    /// Indicators this stream updates itself (not linked to the bank).
    private: Vec<Py<PyAny>>,
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
struct Strat {
    sid: StrategyId,
    strategy: Py<PyAny>,
    ctx: Py<PyRuntimeContext>,
    mask: u32,
    queue: Py<PyCommandQueue>,
    route: Route,
    clocks: Vec<ClockSlot>,
    active_clock: Cell<usize>,
    /// Per local instrument: indicators fed by its primary bars, and the
    /// reveal for the linked ones.
    primary_private: Vec<Vec<Py<PyAny>>>,
    primary_reveal: Vec<Option<Reveal>>,
    /// By local stream id.
    streams: Vec<StreamInfo>,
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
    /// By local stream id.
    streams: Vec<StreamPlan>,
    /// `[local instrument][subscription]` -> local stream id.
    stream_of: Vec<Vec<u16>>,
    inds: Vec<IndPlan>,
}

/// A truncated-series dict, the first row it holds and the watermark it was
/// cut at.
type SeriesCache = (u64, u64, Py<PyDict>);

struct Shared {
    inner: RefCell<Inner>,
    /// Interned symbols by instrument id.
    symbols: Vec<Py<PyString>>,
    sym_ids: HashMap<String, InstrumentId>,
    store: Arc<MarketStore>,
    /// Per instrument, whether it has ticks or depth.
    has_ticks: Vec<bool>,
    /// Per instrument, the caller's bar arrays (`None` without bars).
    arrays: Vec<Option<Py<PyDict>>>,
    /// Per instrument, the caller's tick arrays (`None` without ticks).
    tick_arrays: Vec<Option<Py<PyDict>>>,
    offsets: Vec<i64>,
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
}

impl Shared {
    fn with_book<R>(
        &self,
        sid: StrategyId,
        f: impl FnOnce(&PortfolioBook, Option<usize>) -> PyResult<R>,
    ) -> PyResult<R> {
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

    fn watermark(&self, instrument: InstrumentId) -> u64 {
        match &*self.inner.borrow() {
            Inner::Running(c) => c.watermark(instrument),
            _ => 0,
        }
    }

    fn tick_watermark(&self, instrument: InstrumentId) -> u64 {
        match &*self.inner.borrow() {
            Inner::Running(c) => c.tick_watermark(instrument),
            _ => 0,
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
        let (full, n, keys): (&Bound<'_, PyDict>, u64, Vec<Bound<'_, PyAny>>) =
            match (&self.arrays[g], &self.tick_arrays[g]) {
                (Some(bars), _) => (
                    bars.bind(py),
                    self.watermark(global),
                    SERIES_KEYS.iter().map(|k| PyString::new(py, k).into_any()).collect(),
                ),
                (None, Some(ticks)) => {
                    let d = ticks.bind(py);
                    (d, self.tick_watermark(global), d.keys().iter().collect())
                }
                (None, None) => return Err(PyKeyError::new_err(self.symbols[g].to_string())),
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
    best_bid: Vec<Cell<Option<f64>>>,
    best_ask: Vec<Cell<Option<f64>>>,
    last_price: Vec<Cell<Option<f64>>>,
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
    let symbol = sh.symbols[instrument.index()].bind(py);
    Ok(sh.classes.book.bind(py).call1((tick.timestamp, bids, asks, symbol))?.unbind())
}

impl PyRuntimeContext {
    fn index_of(&self, symbol: Option<&str>) -> PyResult<usize> {
        match symbol {
            None | Some("") => Ok(self.local.get()),
            Some(name) => self
                .shared
                .sym_ids
                .get(name)
                .and_then(|&g| self.shared.local_of(self.sid, g))
                .ok_or_else(|| PyKeyError::new_err(name.to_string())),
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
    /// The symbol whose bar is being processed.
    #[getter]
    fn symbol(&self, py: Python<'_>) -> PyResult<Py<PyString>> {
        let g = self.shared.global_of(self.sid, self.local.get())?;
        Ok(self.shared.symbols[g.index()].clone_ref(py))
    }

    /// Bar ordinal within the current symbol.
    #[getter]
    fn idx(&self) -> usize {
        self.idx.get()
    }

    #[getter]
    fn bar(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.bar.borrow().as_ref().map(|b| b.clone_ref(py))
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
            PyRuntimeError::new_err(
                "the book's depth row is gone: it was overwritten in the live depth history",
            )
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
        self.best_bid[self.local.get()].get()
    }

    /// Last observed ask for the current symbol.
    #[getter]
    fn best_ask(&self) -> Option<f64> {
        self.best_ask[self.local.get()].get()
    }

    /// Last trade price (or bar close) for the current symbol.
    #[getter]
    fn last_price(&self) -> Option<f64> {
        self.last_price[self.local.get()].get()
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

    /// Halt this strategy: new entries are refused as `killed`, working
    /// orders are cancelled and, with `flatten=True`, open positions are
    /// closed at the last seen prices when the next event is applied.
    #[pyo3(signature = (reason="manual", flatten=false))]
    fn halt(&self, reason: &str, flatten: bool) -> PyResult<()> {
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
        for local in 0..self.shared.locals(self.sid) {
            let g = self.shared.global_of(self.sid, local)?;
            out.push(self.shared.symbols[g.index()].bind(py).clone());
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
            return Err(PyValueError::new_err("keep_history needs at least one row"));
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
        let name = self.shared.symbols[g.index()].bind(py).clone();
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
        tif="gtc", expire_ns=None, stop_price=None, target_price=None,
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
    Ok(OrderSpec {
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
    })
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
}

#[pymethods]
impl PyCommandQueue {
    #[new]
    fn new() -> Self {
        Self::default()
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
        let order =
            if twap { Order::Twap(twap_spec(order)?) } else { Order::Plain(order_spec(order)?) };
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
            PyValueError::new_err(format!("{kind}['{symbol}']: '{key}' is not a usable array: {e}"))
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
                PyValueError::new_err(format!("{label}: '{key}' must be a 2-D float array: {e}"))
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
            return Err(PyValueError::new_err("history must be at least 1 row"));
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
        let rust_config: BacktestConfig =
            config.as_deref().map(BacktestConfig::from).unwrap_or_default();
        let mode: AccountMode = parse_account_mode(account_type, leverage)?;
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
                parse_instrument_args(directions[i], instruments[i].as_deref(), oms_type)?;
            policy = Some(p);
            let inst_cfg: Option<InstrumentConfig> =
                instrument_configs[i].as_deref().map(InstrumentConfig::from);
            params.push(InstrumentParams {
                direction,
                spec,
                atr_period: atr_period(&rust_config, inst_cfg.as_ref()).unwrap_or(0),
                config: inst_cfg,
            });
        }
        let builder = RuntimeBuilder::new(
            market.store.clone(),
            params,
            rust_config,
            mode,
            policy.unwrap_or(crate::portfolio::ledger::PositionPolicy::Net),
        )
        .map_err(runtime_err)?;

        let shared = Shared {
            inner: RefCell::new(Inner::Building(Box::new(builder))),
            symbols: symbols.iter().map(|s| PyString::intern(py, s).unbind()).collect(),
            sym_ids,
            store: market.store.clone(),
            // A live node takes ticks, depth and bars on any symbol, and
            // runs every symbol as a tick session.
            has_ticks: (0..n)
                .map(|i| {
                    let id = InstrumentId::new(i as u32);
                    market.live.is_some()
                        || market.store.ticks(id).is_some()
                        || market.store.depth(id).is_some()
                })
                .collect(),
            arrays: market.arrays.iter().map(|a| a.as_ref().map(|a| a.clone_ref(py))).collect(),
            tick_arrays: market.ticks.iter().map(|a| a.as_ref().map(|a| a.clone_ref(py))).collect(),
            offsets: market.offsets.clone(),
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
            strats: RefCell::new(Vec::new()),
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
            .map(|s| {
                self.shared.sym_ids.get(s).copied().ok_or_else(|| PyKeyError::new_err(s.clone()))
            })
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
                best_bid: (0..ids.len()).map(|_| Cell::new(None)).collect(),
                best_ask: (0..ids.len()).map(|_| Cell::new(None)).collect(),
                last_price: (0..ids.len()).map(|_| Cell::new(None)).collect(),
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
        let spec = GroupSpec {
            initial_capital,
            mode: parse_account_mode(account_type, leverage)?,
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
            return Err(PyValueError::new_err("one clock per symbol is required"));
        }
        let clock_cls = py.import("raptorbt.strategy.clock")?.getattr("Clock")?;
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
                    let found =
                        self.shared.sym_ids.get(name).and_then(|&g| self.shared.local_of(sid, g));
                    match found {
                        Some(l) => Some(l),
                        None => {
                            let known = (0..locals)
                                .map(|l| {
                                    self.shared
                                        .global_of(sid, l)
                                        .map(|g| self.shared.symbols[g.index()].to_string())
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
                            .valid(self.shared.symbols.iter().map(|s| s.to_string()))
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
        for (i, &has) in self.shared.has_ticks.iter().enumerate() {
            if has {
                builder
                    .set_primary_bars(InstrumentId::new(i as u32), spec, self.shared.tz_offset_ns)
                    .map_err(runtime_err)?;
            }
        }
        Ok(())
    }

    /// Freeze the registrations: link what can be linked and build the core.
    fn finalize(&self, py: Python<'_>) -> PyResult<()> {
        let plans = std::mem::take(&mut *self.shared.plans.borrow_mut());
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
                let mut rows = history;
                for plan in &plans {
                    for ind in &plan.inds {
                        if let Ok(obj) = ind.obj.bind(py).cast::<PyIndicator>() {
                            rows = rows.max(obj.borrow().spec().lookback());
                        }
                    }
                }
                Some(rows)
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
                    let reg = builder
                        .register_indicator(plan.sid, IndKey::new(input, spec), stream)
                        .map_err(runtime_err)?;
                    linked.insert((pi, ii), reg.slot);
                }
            }
        }

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
            let mut stream_private: Vec<Vec<Py<PyAny>>> =
                (0..plan.streams.len()).map(|_| Vec::new()).collect();
            for (ii, ind) in plan.inds.iter().enumerate() {
                if linked.contains_key(&(pi, ii)) {
                    continue;
                }
                let targets: Vec<usize> = match ind.local {
                    Some(l) => vec![l],
                    None => (0..locals).collect(),
                };
                for l in targets {
                    match ind.stream_id {
                        None => primary_private[l].push(ind.obj.clone_ref(py)),
                        Some(sub) if sub < plan.subs => {
                            let stream = plan.stream_of[l][sub];
                            stream_private[stream as usize].push(ind.obj.clone_ref(py));
                        }
                        // A handle that names no subscription is never fed.
                        Some(_) => {}
                    }
                }
            }
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
                .zip(stream_private)
                .enumerate()
                .map(|(i, (sp, private))| -> PyResult<StreamInfo> {
                    let g = core.instrument_of(plan.sid, sp.local).ok_or_else(finished)?;
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("symbol", self.shared.symbols[g.index()].bind(py))?;
                    Ok(StreamInfo {
                        sub: sp.sub,
                        step: sp.step,
                        unit: PyString::new(py, &sp.unit).unbind(),
                        kwargs: kwargs.unbind(),
                        private,
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
                clocks: plan.clocks,
                active_clock: Cell::new(usize::MAX),
                primary_private,
                primary_reveal,
                streams: stream_infos,
            }));
        }
        *self.shared.strats.borrow_mut() = strats;
        *self.shared.inner.borrow_mut() = Inner::Running(Box::new(core));
        if let Some(rows) = live_rows {
            let mut feed =
                LiveFeed::new(self.shared.symbols.iter().map(|s| s.to_string()).collect(), rows);
            for (owner, instrument, keep) in self.shared.pending_keeps.take() {
                feed.keep(owner, instrument, keep);
            }
            *self.shared.feed.borrow_mut() = Some(feed);
            *self.live_state.borrow_mut() = Some(LoopState::new(&self.shared)?);
        }
        Ok(())
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
        loop {
            let (event, cur) = {
                let mut inner = sh.inner.borrow_mut();
                let Inner::Running(core) = &mut *inner else { return Err(finished()) };
                let Some(event) = core.advance(&mut cursor).map_err(runtime_err)? else {
                    break;
                };
                (event, *core.current().expect("advance sets the current event"))
            };
            dispatch(py, sh, &mut ls, &event, &cur)?;
            n += 1;
            if n.is_multiple_of(SIGNAL_EVERY) {
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
    fn adopt_position(
        &self,
        sid: usize,
        local: usize,
        timestamp_ns: i64,
        price: f64,
        quantity: f64,
    ) -> PyResult<u64> {
        let sid = StrategyId::new(sid as u32);
        let adopted = match &mut *self.shared.inner.borrow_mut() {
            Inner::Building(b) => b.adopt_position(sid, local, timestamp_ns, price, quantity),
            Inner::Running(c) => c.adopt_position(sid, local, timestamp_ns, price, quantity),
            Inner::Gone => return Err(finished()),
        };
        adopted.map_err(PyValueError::new_err)
    }

    /// Feed one tick row to a live node. Returns the events it appended
    /// (0..=2). See `LiveNode.push_tick`.
    #[pyo3(signature = (
        instrument, timestamp, ltp, bid=0.0, ask=0.0, buy_qty_delta=0.0, sell_qty_delta=0.0,
        ltq=0.0, bid_qty=0.0, ask_qty=0.0, oi=0.0,
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
    ) -> PyResult<usize> {
        let sh = &*self.shared;
        let mut guard = self.live_loop()?;
        let g = self.instrument(instrument)?;
        let ts = timestamp;
        with_feed(sh, |feed| feed.check_timestamp(g, ts).map_err(runtime_err))?;
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
    }

    /// Feed one closed bar to a live node; `timestamp` is as the caller
    /// stamps bars, and the symbol's offset moves it to the close.
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
    ) -> PyResult<()> {
        let sh = &*self.shared;
        let mut guard = self.live_loop()?;
        let g = self.instrument(instrument)?;
        let close_ts = timestamp + sh.offsets[g.index()];
        with_feed(sh, |feed| feed.check_timestamp(g, close_ts).map_err(runtime_err))?;
        let stamped = KernelBar { timestamp, open, high, low, close, volume };
        let event = store_row(sh, g, RingKind::Bars, |feed| feed.push_bar(g, close_ts, stamped))?;
        let bar = KernelBar { timestamp: close_ts, ..stamped };
        self.deliver_event(py, &mut guard, event, Pushed::Bar(bar))?;
        self.settle_push();
        Ok(())
    }

    /// Feed one depth snapshot to a live node: `(price, size)` lists, best
    /// level first.
    fn push_depth(
        &self,
        py: Python<'_>,
        instrument: usize,
        timestamp: i64,
        bids: Vec<(f64, f64)>,
        asks: Vec<(f64, f64)>,
    ) -> PyResult<()> {
        let sh = &*self.shared;
        let mut guard = self.live_loop()?;
        let g = self.instrument(instrument)?;
        with_feed(sh, |feed| feed.check_timestamp(g, timestamp).map_err(runtime_err))?;
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
    }

    /// Move every strategy's clocks to `timestamp`, symbol by symbol, firing
    /// the alerts and timers that are due. No data is involved.
    fn advance_time(&self, py: Python<'_>, timestamp: i64) -> PyResult<()> {
        let sh = &*self.shared;
        let _guard = self.live_loop()?;
        self.settle_push();
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
    }

    /// The result of a strategy's run: liquidates its book.
    fn finish(&self, sid: usize) -> PyResult<PyPortfolioResult> {
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
    fn memory_report<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let inner = self.shared.inner.borrow();
        let Inner::Running(core) = &*inner else {
            return Err(PyRuntimeError::new_err("the runtime is not built"));
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
    fn live_loop(&self) -> PyResult<std::cell::RefMut<'_, Option<LoopState>>> {
        if self.shared.live_history.is_none() {
            return Err(PyRuntimeError::new_err("this runtime is not a live node"));
        }
        let guard = self
            .live_state
            .try_borrow_mut()
            .map_err(|_| busy("the runtime is busy: a push was made from a hook"))?;
        if guard.is_none() {
            return Err(not_ready("finalize() before pushing"));
        }
        Ok(guard)
    }

    fn instrument(&self, index: usize) -> PyResult<InstrumentId> {
        if index >= self.shared.symbols.len() {
            return Err(PyKeyError::new_err(index));
        }
        Ok(InstrumentId::new(index as u32))
    }

    /// Make the next event of `g` current and run every subscriber over it.
    #[allow(clippy::too_many_arguments)]
    fn deliver(
        &self,
        py: Python<'_>,
        guard: &mut std::cell::RefMut<'_, Option<LoopState>>,
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
        guard: &mut std::cell::RefMut<'_, Option<LoopState>>,
        event: EventRef,
        payload: Pushed,
    ) -> PyResult<()> {
        let sh = &*self.shared;
        let cur = {
            let mut inner = sh.inner.borrow_mut();
            let Inner::Running(core) = &mut *inner else { return Err(finished()) };
            core.advance_pushed(event, payload).map_err(runtime_err)?;
            *core.current().expect("advance sets the current event")
        };
        let ls = guard.as_mut().expect("checked by live_loop");
        dispatch(py, sh, ls, &event, &cur)
    }

    /// Nothing is in flight between pushes.
    fn settle_push(&self) {
        if let Inner::Running(core) = &mut *self.shared.inner.borrow_mut() {
            core.clear_current();
        }
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
    with_feed(sh, |feed| f(feed).map_err(|e| PyRuntimeError::new_err(feed.explain(g, kind, &e))))
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
}

impl LoopState {
    fn new(sh: &Shared) -> PyResult<Self> {
        let inner = sh.inner.borrow();
        let Inner::Running(core) = &*inner else { return Err(finished()) };
        let strats: Vec<Rc<Strat>> = sh.strats.borrow().clone();
        let mut by_sid: Vec<Option<Rc<Strat>>> = vec![None; strats.len()];
        let mut wanted = vec![0u32; sh.symbols.len()];
        for s in &strats {
            by_sid[s.sid.index()] = Some(s.clone());
            for local in 0..core.instruments_of(s.sid) {
                if let Some(g) = core.instrument_of(s.sid, local) {
                    wanted[g.index()] |= s.mask;
                }
            }
        }
        Ok(Self {
            by_sid,
            wanted,
            bank: core.bank().clone(),
            subs: Vec::new(),
            scratch: Scratch::default(),
        })
    }
}

/// Run every strategy subscribed to the current event over it, in
/// registration order. The one body behind a batch run's tape and a live
/// node's pushes.
fn dispatch(
    py: Python<'_>,
    sh: &Shared,
    ls: &mut LoopState,
    event: &EventRef,
    cur: &Current,
) -> PyResult<()> {
    let LoopState { by_sid, wanted, bank, subs, scratch } = ls;
    subs.clear();
    match &*sh.inner.borrow() {
        Inner::Running(c) => subs.extend_from_slice(c.subscribers(event.instrument)),
        _ => return Err(finished()),
    }
    let mask = wanted[event.instrument.index()];
    // One Python int per event, not per strategy.
    let ts_obj = event.ts.into_pyobject(py)?;
    let ev = Ev { event, ts_obj: &ts_obj };
    match cur.data {
        ScheduleData::Bar(bar) => {
            let offset = sh.offsets[event.instrument.index()];
            let bar_obj = sh.classes.bar.bind(py).call1((
                bar.timestamp - offset,
                bar.open,
                bar.high,
                bar.low,
                bar.close,
                bar.volume,
            ))?;
            for &sid in subs.iter() {
                let st = by_sid[sid.index()].as_ref().expect("subscribed");
                process_bar(py, sh, st, bank, &ev, &bar, &bar_obj, scratch)?;
            }
        }
        ScheduleData::Trade(t) => {
            let symbol = sh.symbols[event.instrument.index()].bind(py);
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
            for &sid in subs.iter() {
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
                )?;
            }
        }
        ScheduleData::Quote(q) => {
            let symbol = sh.symbols[event.instrument.index()].bind(py);
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
            for &sid in subs.iter() {
                let st = by_sid[sid.index()].as_ref().expect("subscribed");
                process_quote(py, sh, st, &ev, &q, quote.as_ref())?;
            }
        }
        ScheduleData::Depth(_) => {
            let tick = match &*sh.inner.borrow() {
                Inner::Running(c) => c.depth_of(event),
                _ => return Err(finished()),
            }
            .ok_or_else(|| PyRuntimeError::new_err("a book event has no depth row"))?;
            let book = (mask & HOOK_BOOK != 0)
                .then(|| book_object(py, sh, event.instrument, &tick).map(|o| o.into_bound(py)))
                .transpose()?;
            for &sid in subs.iter() {
                let st = by_sid[sid.index()].as_ref().expect("subscribed");
                process_book(py, sh, st, &ev, &tick, book.as_ref())?;
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
    let cs = &st.clocks[local];
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

/// Buffers reused across events.
#[derive(Default)]
struct Scratch {
    deliveries: Vec<(u16, crate::core::types::OhlcvBar)>,
}

/// The event every subscriber of it is shown.
struct Ev<'a, 'py> {
    event: &'a EventRef,
    ts_obj: &'a Bound<'py, PyInt>,
}

fn has_commands(py: Python<'_>, st: &Strat) -> bool {
    st.queue.bind(py).borrow().queue.has_commands()
}

fn has_intents(py: Python<'_>, st: &Strat) -> bool {
    st.queue.bind(py).borrow().queue.has_intents()
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
    ev: &Ev<'_, '_>,
    bar: Option<(&Bound<'_, PyAny>, i64)>,
) -> PyResult<(usize, bool)> {
    let local = match &*sh.inner.borrow() {
        Inner::Running(c) => c.local_of(st.sid, ev.event.instrument).expect("subscribed"),
        _ => return Err(finished()),
    };
    let ctx = st.ctx.bind(py);
    {
        let c = ctx.borrow();
        c.local.set(local);
        c.idx.set(ev.event.local_idx as usize);
        if let Some((bar, close_ts)) = bar {
            *c.bar.borrow_mut() = Some(bar.clone().unbind());
            c.close_ts.set(Some(close_ts));
        }
    }
    let strategy = st.strategy.bind(py);
    let cs = &st.clocks[local];
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
    bank: &IndicatorBank,
    scratch: &mut Scratch,
) -> PyResult<()> {
    scratch.deliveries.clear();
    if let Inner::Running(core) = &*sh.inner.borrow() {
        scratch.deliveries.extend(core.deliveries(st.sid).map(|(l, b)| (l, *b)));
    }
    let strategy = st.strategy.bind(py);
    let ctx = st.ctx.bind(py);
    let mut last: Option<u16> = None;
    let mut emission = 0u32;
    for &(stream, ref b) in &scratch.deliveries {
        emission = if last == Some(stream) { emission + 1 } else { 1 };
        last = Some(stream);
        let info = &st.streams[stream as usize];
        if let Some(handle) = info.reveal {
            bank.reveal(handle, emission);
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

/// The primary bar: its indicators, then `on_bar`.
fn primary_bar(
    py: Python<'_>,
    st: &Strat,
    bank: &IndicatorBank,
    local: usize,
    ohlc: (f64, f64, f64, f64),
) -> PyResult<()> {
    if let Some(handle) = st.primary_reveal[local] {
        bank.reveal(handle, 1);
    }
    for ind in &st.primary_private[local] {
        ind.bind(py).call_method1(intern!(py, "update_bar"), ohlc)?;
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
        route_commands(py, sh, st, local, ev.event.local_idx as usize, ev.event.ts)?;
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
            let mut target = RuntimeTarget { sh, sid: st.sid };
            state.borrow_mut().apply(&mut target, commands, current, idx, ts)?;
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
    match &st.route {
        Route::Python { path, .. } => {
            let symbol = sh.symbols[global.index()].bind(py);
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
    let err = PyErr::from(fail);
    if matches!(&error.valid, Some(valid) if valid.is_empty()) {
        // An explicit empty list, not the registry's default.
        Python::attach(|py| {
            let _ = err.value(py).setattr("valid", Vec::<String>::new());
        });
    }
    err
}

/// A strategy's book, as the target its commands are routed to.
struct RuntimeTarget<'a> {
    sh: &'a Shared,
    sid: StrategyId,
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
                .get(name)
                .and_then(|&g| self.sh.local_of(self.sid, g))
                .ok_or_else(|| PyKeyError::new_err(name.to_string())),
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
        self.sh.with_book_mut(self.sid, |book, current| {
            let idx = book.submission_idx(instrument, idx, current);
            submit_order_on(
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
            )
        })
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
        self.sh.with_book_mut(self.sid, |b, _| {
            b.kernel_mut(instrument).cancel_order(idx, order_id);
            Ok(())
        })
    }

    fn cancel_all_orders(&mut self, instrument: usize, idx: usize) -> PyResult<()> {
        self.sh.with_book_mut(self.sid, |b, _| {
            b.kernel_mut(instrument).cancel_all_orders(idx);
            Ok(())
        })
    }

    fn link_oco(&mut self, instrument: usize, order_ids: &[u64]) -> PyResult<()> {
        self.sh.with_book_mut(self.sid, |b, _| {
            b.kernel_mut(instrument).link_oco(order_ids);
            Ok(())
        })
    }

    fn request_close(&mut self, instrument: usize, position_id: u64) -> PyResult<()> {
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
        let qty = parse_qty_spec(units, size_frac)?;
        self.sh.with_book_mut(self.sid, |b, _| {
            b.kernel_mut(instrument).modify_order(order_id, qty, limit_price, trigger_price);
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
        };
        let bar = KernelBar { timestamp: ts, open, high, low, close, volume };
        let events = self.session.bind(py).borrow_mut().runner_mut()?.step(idx, &bar, input);
        if events.is_empty() {
            return Ok(());
        }
        let ctx = self.ctx.bind(py);
        // The symbol the context names, if it names one.
        let symbol = ctx.getattr_opt(intern!(py, "symbol"))?.and_then(|s| s.cast_into().ok());
        dispatch_events(py, self.strategy.bind(py), ctx, self.mask, events, symbol.as_ref())
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
    Expired,
    Rejected,
}

fn tag_of(event: &EngineEvent) -> Tag {
    match event {
        EngineEvent::Entered { .. } => Tag::Entered,
        EngineEvent::Exited { .. } => Tag::Exited,
        EngineEvent::EntryRejected { .. } => Tag::EntryRejected,
        EngineEvent::AlgoStarted { .. } => Tag::AlgoStarted,
        EngineEvent::AlgoCompleted { .. } => Tag::AlgoCompleted,
        EngineEvent::MarginCall { .. } => Tag::MarginCall,
        EngineEvent::RiskRejected { .. } => Tag::RiskRejected,
        EngineEvent::RiskHalted { .. } => Tag::RiskHalted,
        EngineEvent::OrderAccepted { .. } => Tag::Accepted,
        EngineEvent::OrderTriggered { .. } => Tag::Triggered,
        EngineEvent::OrderFilled { .. } => Tag::Filled,
        EngineEvent::OrderCanceled { .. } => Tag::Canceled,
        EngineEvent::OrderExpired { .. } => Tag::Expired,
        EngineEvent::OrderRejected { .. } => Tag::Rejected,
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
        Tag::Expired => HOOK_ORDER_EXPIRED | HOOK_ORDER_EVENT,
        Tag::Rejected => HOOK_ORDER_REJECTED | HOOK_ORDER_EVENT,
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

/// The step: queued commands, then the book's step, then engine events to
/// hooks and the work those queued.
///
/// `fold` is whether the event can carry an entry: bars and prints fold the
/// intents (and those held for the symbol) into the step; a quote or a book
/// steps with none, and what its hooks queued is held for the symbol's next
/// step. `hooked` says whether a hook that could have queued work ran, so an
/// event nobody handles skips the queue checks.
fn step(
    py: Python<'_>,
    sh: &Shared,
    st: &Strat,
    ev: &Ev<'_, '_>,
    local: usize,
    fold: bool,
    hooked: bool,
) -> PyResult<()> {
    let sid = st.sid;
    let local_idx = ev.event.local_idx as usize;
    let ts = ev.event.ts;
    let strategy = st.strategy.bind(py);
    let ctx = st.ctx.bind(py);
    let symbol = sh.symbols[ev.event.instrument.index()].bind(py);

    // 4. Commands, only when work is queued; then the step.
    if fold || hooked {
        drain_commands(py, sh, st, local, ev)?;
    }
    let atr = if fold {
        match &*sh.inner.borrow() {
            Inner::Running(c) => c.atr_at(sid, local, local_idx),
            _ => return Err(finished()),
        }
    } else {
        0.0
    };
    let any_held = match &st.route {
        Route::Python { held, .. } => !held.bind(py).is_empty(),
        Route::Native(state) => state.borrow().has_held(),
    };
    let input = if fold && (has_intents(py, st) || any_held) {
        fold_step(py, st, symbol, local, local_idx, atr)?
    } else {
        StepInput { atr, ..StepInput::default() }
    };
    let events = match &mut *sh.inner.borrow_mut() {
        Inner::Running(c) => c.apply(sid, input).map_err(runtime_err)?,
        _ => return Err(finished()),
    };

    // 5. Engine events to hooks; 6. work those hooks queued.
    let dispatched = !events.is_empty();
    if dispatched {
        if matches!(st.route, Route::Native(_)) {
            dispatch_events(py, strategy, ctx.as_any(), st.mask, events, Some(symbol))?;
        } else {
            let list = PyList::new(
                py,
                events
                    .into_iter()
                    .map(|e| Py::new(py, PyEngineEvent::from(e)))
                    .collect::<PyResult<Vec<_>>>()?,
            )?;
            sh.classes.dispatch.bind(py).call1((strategy, ctx, list))?;
        }
    }
    if (dispatched || (!fold && hooked)) && (has_commands(py, st) || has_intents(py, st)) {
        settle(py, sh, st, ev.event.instrument, local, local_idx, ts)?;
    }
    Ok(())
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
) -> PyResult<()> {
    let with_ticks = sh.has_ticks[ev.event.instrument.index()];
    let first = (!with_ticks).then_some((bar_obj, bar.timestamp));
    let (local, _) = enter(py, sh, st, ev, first)?;
    {
        let ctx = st.ctx.bind(py).borrow();
        if with_ticks {
            *ctx.bar.borrow_mut() = Some(bar_obj.clone().unbind());
            ctx.close_ts.set(Some(bar.timestamp));
        }
        ctx.last_price[local].set(Some(bar.close));
    }
    composites(py, sh, st, bank, scratch)?;
    primary_bar(py, st, bank, local, (bar.open, bar.high, bar.low, bar.close))?;
    step(py, sh, st, ev, local, true, true)
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
) -> PyResult<()> {
    let (local, _) = enter(py, sh, st, ev, None)?;
    let ctx = st.ctx.bind(py);
    if let Some((bar, bar_obj)) = primary {
        {
            let c = ctx.borrow();
            *c.bar.borrow_mut() = Some(bar_obj.clone().unbind());
            c.close_ts.set(Some(bar.timestamp));
        }
        primary_bar(py, st, bank, local, (bar.open, bar.high, bar.low, bar.close))?;
        drain_commands(py, sh, st, local, ev)?;
    }
    composites(py, sh, st, bank, scratch)?;
    ctx.borrow().last_price[local].set(Some(trade.price));
    if st.mask & HOOK_TRADE != 0 {
        let tick = tick.expect("built for a strategy that overrides on_trade_tick");
        *ctx.borrow().tick.borrow_mut() = Some(tick.clone().unbind());
        let called = st.strategy.bind(py).call_method1(intern!(py, "on_trade_tick"), (ctx, tick));
        *ctx.borrow().tick.borrow_mut() = None;
        called?;
    }
    step(py, sh, st, ev, local, true, true)
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
) -> PyResult<()> {
    let (local, timed) = enter(py, sh, st, ev, None)?;
    let ctx = st.ctx.bind(py);
    {
        let c = ctx.borrow();
        c.best_bid[local].set(Some(quote.bid));
        c.best_ask[local].set(Some(quote.ask));
    }
    if st.mask & HOOK_QUOTE != 0 {
        let obj = obj.expect("built for a strategy that overrides on_quote");
        *ctx.borrow().quote.borrow_mut() = Some(obj.clone().unbind());
        let called = st.strategy.bind(py).call_method1(intern!(py, "on_quote"), (ctx, obj));
        *ctx.borrow().quote.borrow_mut() = None;
        called?;
    }
    step(py, sh, st, ev, local, false, timed || st.mask & HOOK_QUOTE != 0)
}

/// A book snapshot: observation only, like a quote.
fn process_book(
    py: Python<'_>,
    sh: &Shared,
    st: &Strat,
    ev: &Ev<'_, '_>,
    tick: &crate::data::book::DepthTick,
    obj: Option<&Bound<'_, PyAny>>,
) -> PyResult<()> {
    let (local, timed) = enter(py, sh, st, ev, None)?;
    let ctx = st.ctx.bind(py);
    {
        let c = ctx.borrow();
        if tick.bid_len > 0 {
            c.best_bid[local].set(Some(tick.bids[0].price));
        }
        if tick.ask_len > 0 {
            c.best_ask[local].set(Some(tick.asks[0].price));
        }
        c.books.borrow_mut()[local] = Some((ev.event.row, obj.map(|o| o.clone().unbind())));
    }
    if st.mask & HOOK_BOOK != 0 {
        let obj = obj.expect("built for a strategy that overrides on_order_book");
        st.strategy.bind(py).call_method1(intern!(py, "on_order_book"), (ctx, obj))?;
    }
    step(py, sh, st, ev, local, false, timed || st.mask & HOOK_BOOK != 0)
}
