//! Shared indicator state: one streaming core per distinct key, many readers.
//!
//! Strategies used to own a private core per registered indicator, so N
//! strategies on the same input did N identical updates. Here a registration
//! is reduced to an [`IndKey`]; two registrations share a cell if and only if
//! their whole keys are equal.
//!
//! # Key
//!
//! * the input stream: one instrument's primary bars, the interleaved primary
//!   bars of an instrument set (an indicator registered without a symbol in a
//!   multi-symbol run), or a composite aggregation;
//! * the kind and its parameters, canonicalised so equal behaviour gives equal
//!   bits (periods are clamped to at least 1 like the cores do, `-0.0` becomes
//!   `0.0`, every NaN becomes one NaN);
//! * which bar field a scalar kind consumes;
//! * warmup, session and adjustment policies. Each has a single variant today;
//!   they are in the key so that a future second meaning can never silently
//!   share state with the first.
//!
//! [`IndicatorRegistry`] dedups at registration time and numbers cells in
//! first-registration order. Every registration also gets its own [`SlotId`],
//! the reader's handle.
//!
//! # Bank and reveal protocol
//!
//! [`IndicatorBank`] freezes a registry into cells. Each cell owns one
//! streaming core and is updated once per input emission, however many
//! strategies read it. Strategies must still observe the values a private core
//! would show them at each point of the event (see below), so a cell keeps the
//! value before the event (`prev`), the value after the last emission (`cur`)
//! and the values after the earlier emissions of a multi-emission event
//! (`hist`; a composite stream such as Renko can emit several bars at once).
//!
//! Per event the runtime calls, in this order:
//!
//! 1. [`IndicatorBank::begin_event`];
//! 2. every emission of the event: [`emit_composite`](IndicatorBank::emit_composite)
//!    for each bar a composite stream produced, in emission order, and
//!    [`emit_primary`](IndicatorBank::emit_primary) for the primary bar. All
//!    emissions come before the first reveal;
//! 3. for each strategy that receives the event, in its own order, mirroring
//!    what a private-core strategy does:
//!    * for each completed composite stream in subscription order, and for
//!      each of its emissions `k = 1..=m`:
//!      [`reveal_stream(s, stream, k)`](IndicatorBank::reveal_stream), then the
//!      strategy's composite callback;
//!    * [`reveal_primary(s, instrument)`](IndicatorBank::reveal_primary), then
//!      the strategy's bar callback.
//!
//! A read by strategy `s` through slot `x` returns, for the cell behind `x`:
//!
//! * `cur`, if the cell was not fed in this event: a cell fed by another event
//!   of the same timestamp group shows its value after that event, whether the
//!   event is decided before or after this one (every event of a group is
//!   applied before any is decided; see `event-semantics.md`);
//! * for a cell fed by an instrument set (`AllPrimary`) the stream is the
//!   reader's own whichever of its instruments is decided, so the protocol
//!   applies to the group: `prev` is the value before the group, and the
//!   reader's own reveal in an event shows the value after the whole group
//!   (`cur`), for every symbol order. Outside a group this is the single event
//!   rule below;
//! * else (the cell is the selected event's own stream) `prev`, if `s` has not
//!   yet revealed the stream behind `x`;
//! * else the value after the `k`-th emission (`hist[k-1]`, or `cur` for the
//!   last one) once `s` revealed emission `k`; `reveal_primary` reveals all.
//!
//! Every reader of a stream must reveal every emission of every event it is
//! fed; a reader that skips one would see the cell run ahead of it.
//!
//! # Threading
//!
//! Exactly one thread (the runtime's) may call `begin_event`, `emit_*` and
//! `reveal_*`. Values live in atomics and the cores behind one mutex, so the
//! bank is `Send + Sync` and an `Arc<IndicatorBank>` can back a Python-visible
//! reader that is `Sync`; but a read from another thread while an event is in
//! flight sees a state that is in the middle of the protocol. Hand the bank
//! over at event boundaries.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicU8, Ordering};
use std::sync::Mutex;

use crate::core::error::RaptorError;
use crate::core::Result;
use crate::indicators::streaming::{
    StreamingAtr, StreamingBollinger, StreamingDonchian, StreamingEma, StreamingIndicator,
    StreamingMacd, StreamingRoc, StreamingRsi, StreamingSma, StreamingStdDev, StreamingWma,
};
use crate::runtime::ids::{AggId, IndId, InstrumentId, StrategyId};
use crate::runtime::memory::HeapSize;

/// An interned set of instruments whose primary bars are fed, interleaved in
/// merge order, to one indicator.
///
/// Two strategies with the same set see the same interleaved sequence, so an
/// indicator over it can be shared between them. Obtain one from
/// [`IndicatorRegistry::primary_order`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PrimaryOrder(u32);

impl PrimaryOrder {
    #[inline]
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

/// A strategy-local composite stream, as numbered by the strategy.
///
/// Two local streams of one strategy may be backed by the same aggregation;
/// they are revealed independently.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LocalStreamId(pub u32);

/// A reader's handle: one per registration, in registration order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SlotId(u32);

impl SlotId {
    #[inline]
    pub(crate) const fn new(index: u32) -> Self {
        Self(index)
    }

    #[inline]
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

/// What feeds an indicator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IndInput {
    /// One instrument's primary bars.
    Primary(InstrumentId),
    /// Every primary bar of an instrument set, interleaved.
    AllPrimary(PrimaryOrder),
    /// The bars a composite aggregation emits.
    Composite(AggId),
}

/// The streaming cores a key can name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum IndKind {
    Sma,
    Ema,
    WilderMa,
    Wma,
    Roc,
    StdDev,
    Rsi,
    Atr,
    Donchian,
    Bollinger,
    Macd,
}

impl IndKind {
    /// Whether the core consumes a whole bar rather than one scalar.
    pub const fn is_bar(self) -> bool {
        matches!(self, IndKind::Atr | IndKind::Donchian)
    }

    /// How many numbers the value has: 1, 2 (Donchian) or 3.
    pub const fn width(self) -> usize {
        match self {
            IndKind::Donchian => 2,
            IndKind::Bollinger | IndKind::Macd => 3,
            _ => 1,
        }
    }
}

/// The bar field a scalar kind consumes. Bar kinds (ATR, Donchian) use
/// [`InputField::Bar`]; scalar kinds must not.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InputField {
    Close,
    Open,
    High,
    Low,
    Bar,
}

/// Warmup semantics. Only the cores' own seeding exists today.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum WarmupPolicy {
    #[default]
    Standard,
}

/// Session semantics. Only continuous accumulation exists today.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum SessionPolicy {
    #[default]
    Continuous,
}

pub use crate::instruments::lifecycle::AdjustmentPolicy;

/// A kind with its parameters, before canonicalisation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IndSpec {
    Sma(usize),
    Ema(usize),
    WilderMa(usize),
    Wma(usize),
    Roc(usize),
    StdDev(usize),
    Rsi(usize),
    Atr(usize),
    Donchian(usize),
    Bollinger { period: usize, k: f64 },
    Macd { fast: usize, slow: usize, signal: usize },
}

fn canon_f64(x: f64) -> u64 {
    if x == 0.0 {
        0
    } else if x.is_nan() {
        f64::NAN.to_bits()
    } else {
        x.to_bits()
    }
}

fn canon_period(p: usize) -> u64 {
    p.max(1) as u64
}

impl IndSpec {
    /// The kind this spec names.
    pub fn kind(self) -> IndKind {
        self.parts().0
    }

    /// Bars of history the indicator reads to produce one value: the span a
    /// strategy would need to recompute it from raw bars.
    pub fn lookback(self) -> usize {
        match self {
            IndSpec::Sma(p)
            | IndSpec::Ema(p)
            | IndSpec::WilderMa(p)
            | IndSpec::Wma(p)
            | IndSpec::StdDev(p)
            | IndSpec::Rsi(p)
            | IndSpec::Atr(p)
            | IndSpec::Donchian(p) => p,
            // A rate of change compares against the bar `p` back.
            IndSpec::Roc(p) => p + 1,
            IndSpec::Bollinger { period, .. } => period,
            IndSpec::Macd { slow, signal, .. } => slow + signal,
        }
    }

    fn parts(self) -> (IndKind, [u64; 3]) {
        let one = |kind, p| (kind, [canon_period(p), 0, 0]);
        match self {
            IndSpec::Sma(p) => one(IndKind::Sma, p),
            IndSpec::Ema(p) => one(IndKind::Ema, p),
            IndSpec::WilderMa(p) => one(IndKind::WilderMa, p),
            IndSpec::Wma(p) => one(IndKind::Wma, p),
            IndSpec::Roc(p) => one(IndKind::Roc, p),
            IndSpec::StdDev(p) => one(IndKind::StdDev, p),
            IndSpec::Rsi(p) => one(IndKind::Rsi, p),
            IndSpec::Atr(p) => one(IndKind::Atr, p),
            IndSpec::Donchian(p) => one(IndKind::Donchian, p),
            IndSpec::Bollinger { period, k } => {
                (IndKind::Bollinger, [canon_period(period), canon_f64(k), 0])
            }
            IndSpec::Macd { fast, slow, signal } => {
                (IndKind::Macd, [canon_period(fast), canon_period(slow), canon_period(signal)])
            }
        }
    }
}

/// Everything that decides an indicator's state. Two registrations share a
/// cell if and only if their keys are equal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IndKey {
    pub input: IndInput,
    pub kind: IndKind,
    /// Canonicalised parameters; unused entries are zero. Bollinger's `k`
    /// is stored as `f64` bits.
    pub params: [u64; 3],
    pub field: InputField,
    pub warmup: WarmupPolicy,
    pub session: SessionPolicy,
    pub adjustment: AdjustmentPolicy,
}

impl IndKey {
    /// The key for `spec` over `input`, consuming the close (or the whole bar
    /// for ATR and Donchian), with default policies.
    pub fn new(input: IndInput, spec: IndSpec) -> Self {
        let (kind, params) = spec.parts();
        Self {
            input,
            kind,
            params,
            field: if kind.is_bar() { InputField::Bar } else { InputField::Close },
            warmup: WarmupPolicy::default(),
            session: SessionPolicy::default(),
            adjustment: AdjustmentPolicy::default(),
        }
    }

    /// The same key reading another field. Rejected at registration unless
    /// the kind is scalar and `field` is not [`InputField::Bar`].
    pub fn with_field(mut self, field: InputField) -> Self {
        self.field = field;
        self
    }
}

/// What one registration recorded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlotReg {
    pub strategy: StrategyId,
    pub ind: IndId,
    pub stream: Option<LocalStreamId>,
}

/// The outcome of one registration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Registration {
    /// This registration's reader handle.
    pub slot: SlotId,
    /// The cell behind it.
    pub ind: IndId,
    /// Whether the cell already existed.
    pub shared: bool,
}

/// Registration-time dedup.
#[derive(Debug, Default)]
pub struct IndicatorRegistry {
    keys: Vec<IndKey>,
    index: HashMap<IndKey, IndId>,
    orders: Vec<Box<[InstrumentId]>>,
    order_index: HashMap<Box<[InstrumentId]>, PrimaryOrder>,
    slots: Vec<SlotReg>,
}

fn full(what: &str) -> RaptorError {
    RaptorError::invalid_parameter(format!("indicator registry is full: too many {what}"))
}

impl IndicatorRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Intern an instrument set as an [`IndInput::AllPrimary`] order. Order
    /// and duplicates in `instruments` do not matter.
    pub fn primary_order(&mut self, instruments: &[InstrumentId]) -> Result<PrimaryOrder> {
        let mut set: Vec<InstrumentId> = instruments.to_vec();
        set.sort_unstable();
        set.dedup();
        if set.is_empty() {
            return Err(RaptorError::invalid_parameter("a primary order needs an instrument"));
        }
        let set: Box<[InstrumentId]> = set.into();
        if let Some(&order) = self.order_index.get(&set) {
            return Ok(order);
        }
        let order = PrimaryOrder(u32::try_from(self.orders.len()).map_err(|_| full("orders"))?);
        self.orders.push(set.clone());
        self.order_index.insert(set, order);
        Ok(order)
    }

    /// Register an indicator for `strategy`.
    ///
    /// `stream` names the strategy's local composite stream and is required
    /// exactly when the input is [`IndInput::Composite`]. Cells are numbered
    /// in first-registration order.
    pub fn register(
        &mut self,
        strategy: StrategyId,
        key: IndKey,
        stream: Option<LocalStreamId>,
    ) -> Result<Registration> {
        if key.field == InputField::Bar && !key.kind.is_bar() {
            return Err(RaptorError::invalid_parameter(format!(
                "{:?} consumes one scalar; field Bar is for ATR and Donchian",
                key.kind
            )));
        }
        if key.field != InputField::Bar && key.kind.is_bar() {
            return Err(RaptorError::invalid_parameter(format!(
                "{:?} consumes a whole bar; field must be Bar",
                key.kind
            )));
        }
        match (key.input, stream) {
            (IndInput::Composite(_), None) => {
                return Err(RaptorError::invalid_parameter(
                    "a composite indicator needs its local stream",
                ));
            }
            (IndInput::Primary(_) | IndInput::AllPrimary(_), Some(_)) => {
                return Err(RaptorError::invalid_parameter(
                    "only composite indicators take a local stream",
                ));
            }
            (IndInput::AllPrimary(order), _) if order.index() >= self.orders.len() => {
                return Err(RaptorError::invalid_parameter("unknown primary order"));
            }
            _ => {}
        }
        let slot = SlotId::new(u32::try_from(self.slots.len()).map_err(|_| full("slots"))?);
        let (ind, shared) = match self.index.get(&key) {
            Some(&ind) => (ind, true),
            None => {
                let ind = IndId::new(u32::try_from(self.keys.len()).map_err(|_| full("cells"))?);
                self.keys.push(key);
                self.index.insert(key, ind);
                (ind, false)
            }
        };
        self.slots.push(SlotReg { strategy, ind, stream });
        Ok(Registration { slot, ind, shared })
    }

    /// Distinct cells (distinct keys).
    pub fn cells(&self) -> usize {
        self.keys.len()
    }

    /// Registrations made, shared or not.
    pub fn registrations(&self) -> usize {
        self.slots.len()
    }

    pub fn key(&self, ind: IndId) -> Option<&IndKey> {
        self.keys.get(ind.index())
    }

    /// The instruments of an interned order, sorted.
    pub fn order_instruments(&self, order: PrimaryOrder) -> Option<&[InstrumentId]> {
        self.orders.get(order.index()).map(|s| &**s)
    }

    /// Cell keys by [`IndId`].
    pub fn keys(&self) -> &[IndKey] {
        &self.keys
    }

    /// Registrations by [`SlotId`].
    pub fn slots(&self) -> &[SlotReg] {
        &self.slots
    }

    /// Interned instrument sets by [`PrimaryOrder`].
    pub fn orders(&self) -> &[Box<[InstrumentId]>] {
        &self.orders
    }
}

impl HeapSize for IndicatorRegistry {
    /// Counts the vectors and the dedup maps' entries; hash table slack and
    /// control bytes are not included.
    fn heap_bytes(&self) -> usize {
        use std::mem::size_of;
        self.keys.capacity() * size_of::<IndKey>()
            + self.index.capacity() * (size_of::<IndKey>() + size_of::<IndId>())
            + self.slots.capacity() * size_of::<SlotReg>()
            + self.orders.capacity() * size_of::<Box<[InstrumentId]>>()
            + self.order_index.capacity()
                * (size_of::<Box<[InstrumentId]>>() + size_of::<PrimaryOrder>())
            + self.orders.iter().map(|o| 2 * o.len() * size_of::<InstrumentId>()).sum::<usize>()
    }
}

// ---------------------------------------------------------------------------
// Bank
// ---------------------------------------------------------------------------

/// One bar as the cores consume it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ohlc {
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
}

/// An indicator value as a reader sees it: the cores' `Option` flattened.
///
/// `values` holds `kind.width()` numbers (Donchian: upper, lower; Bollinger
/// and MACD: three) and zeros after them; all zeros while uninitialised.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Reading {
    pub initialized: bool,
    pub values: [f64; 3],
}

impl Reading {
    pub(crate) const UNINIT: Self = Self { initialized: false, values: [0.0; 3] };

    fn scalar(v: Option<f64>) -> Self {
        v.map_or(Self::UNINIT, |x| Self { initialized: true, values: [x, 0.0, 0.0] })
    }

    fn pair(v: Option<(f64, f64)>) -> Self {
        v.map_or(Self::UNINIT, |(a, b)| Self { initialized: true, values: [a, b, 0.0] })
    }

    fn triple(v: Option<(f64, f64, f64)>) -> Self {
        v.map_or(Self::UNINIT, |(a, b, c)| Self { initialized: true, values: [a, b, c] })
    }

    /// The flag and the bit patterns, for exact comparison (NaN-safe).
    pub fn bits(&self) -> (bool, [u64; 3]) {
        (self.initialized, self.values.map(f64::to_bits))
    }
}

/// [`Reading`] in atomics: readers need no lock.
#[derive(Debug, Default)]
struct AtomicReading {
    initialized: AtomicBool,
    values: [AtomicU64; 3],
}

impl AtomicReading {
    #[inline]
    fn load(&self) -> Reading {
        Reading {
            initialized: self.initialized.load(Ordering::Relaxed),
            values: [0, 1, 2].map(|i| f64::from_bits(self.values[i].load(Ordering::Relaxed))),
        }
    }

    fn store(&self, r: Reading) {
        self.initialized.store(r.initialized, Ordering::Relaxed);
        for (slot, v) in self.values.iter().zip(r.values) {
            slot.store(v.to_bits(), Ordering::Relaxed);
        }
    }
}

/// The streaming cores, one per [`IndKind`].
#[derive(Debug)]
enum Core {
    Sma(StreamingSma),
    Ema(StreamingEma),
    WilderMa(StreamingEma),
    Wma(StreamingWma),
    Roc(StreamingRoc),
    StdDev(StreamingStdDev),
    Rsi(StreamingRsi),
    Atr(StreamingAtr),
    Donchian(StreamingDonchian),
    Bollinger(StreamingBollinger),
    Macd(StreamingMacd),
}

/// A core with the field it reads.
#[derive(Debug)]
pub struct CellCore {
    core: Core,
    field: InputField,
}

impl CellCore {
    /// The core of an indicator a strategy owns, reading the close (or the
    /// whole bar), as a bank cell of the same spec would.
    pub fn for_spec(spec: IndSpec) -> Self {
        Self::new(&IndKey::new(IndInput::Primary(InstrumentId::new(0)), spec))
    }

    fn new(key: &IndKey) -> Self {
        let p = |i: usize| key.params[i] as usize;
        let core = match key.kind {
            IndKind::Sma => Core::Sma(StreamingSma::new(p(0))),
            IndKind::Ema => Core::Ema(StreamingEma::new(p(0))),
            IndKind::WilderMa => Core::WilderMa(StreamingEma::wilder(p(0))),
            IndKind::Wma => Core::Wma(StreamingWma::new(p(0))),
            IndKind::Roc => Core::Roc(StreamingRoc::new(p(0))),
            IndKind::StdDev => Core::StdDev(StreamingStdDev::new(p(0))),
            IndKind::Rsi => Core::Rsi(StreamingRsi::new(p(0))),
            IndKind::Atr => Core::Atr(StreamingAtr::new(p(0))),
            IndKind::Donchian => Core::Donchian(StreamingDonchian::new(p(0))),
            IndKind::Bollinger => {
                Core::Bollinger(StreamingBollinger::new(p(0), f64::from_bits(key.params[1])))
            }
            IndKind::Macd => Core::Macd(StreamingMacd::new(p(0), p(1), p(2))),
        };
        Self { core, field: key.field }
    }

    /// Feed one bar; the value after it.
    pub fn update(&mut self, bar: Ohlc) -> Reading {
        let x = match self.field {
            InputField::Close => bar.close,
            InputField::Open => bar.open,
            InputField::High => bar.high,
            InputField::Low => bar.low,
            // Bar kinds ignore it; the registry refuses Bar on scalar kinds.
            InputField::Bar => bar.close,
        };
        match &mut self.core {
            Core::Sma(c) => Reading::scalar(c.update(x)),
            Core::Ema(c) | Core::WilderMa(c) => Reading::scalar(c.update(x)),
            Core::Wma(c) => Reading::scalar(c.update(x)),
            Core::Roc(c) => Reading::scalar(c.update(x)),
            Core::StdDev(c) => Reading::scalar(c.update(x)),
            Core::Rsi(c) => Reading::scalar(c.update(x)),
            Core::Atr(c) => Reading::scalar(c.update_bar(bar.high, bar.low, bar.close)),
            Core::Donchian(c) => Reading::pair(c.update_bar(bar.high, bar.low)),
            Core::Bollinger(c) => Reading::triple(c.update(x)),
            Core::Macd(c) => Reading::triple(c.update(x)),
        }
    }
}

/// Heap owned by a freshly built core: its retained windows. Mirrors the
/// `with_capacity` calls in `indicators::streaming`; the cores keep their
/// fields private, so this is derived from the parameters.
fn core_heap_bytes(key: &IndKey) -> usize {
    let f = std::mem::size_of::<f64>();
    let p = key.params[0] as usize;
    match key.kind {
        IndKind::Sma | IndKind::Ema | IndKind::WilderMa | IndKind::Wma | IndKind::StdDev => {
            (p + 1) * f
        }
        IndKind::Roc => (p + 2) * f,
        IndKind::Rsi => 2 * p * f,
        IndKind::Atr => p * f,
        IndKind::Donchian => 2 * (p + 1) * f,
        IndKind::Bollinger => (p + 1) * f,
        IndKind::Macd => key.params.iter().map(|&q| (q as usize + 1) * f).sum(),
    }
}

/// A cell's values. `touched` is the event sequence that last fed the cell,
/// and `emissions` counts its emissions in that event.
#[derive(Debug)]
struct CellVals {
    prev: AtomicReading,
    cur: AtomicReading,
    touched: AtomicU64,
    emissions: AtomicU32,
    /// Emissions over the whole run.
    total: AtomicU64,
}

impl CellVals {
    fn new() -> Self {
        Self {
            prev: AtomicReading::default(),
            cur: AtomicReading::default(),
            touched: AtomicU64::new(0),
            emissions: AtomicU32::new(0),
            total: AtomicU64::new(0),
        }
    }
}

/// State only the runtime thread mutates: the cores and, for events with
/// several emissions, the values after all but the last one.
#[derive(Debug)]
struct Writer {
    cores: Vec<CellCore>,
    hist: Vec<Vec<Reading>>,
}

/// A slot's reveal state packs `event << STATE_BITS | emissions revealed`.
const STATE_BITS: u32 = 24;
const STATE_MASK: u64 = (1 << STATE_BITS) - 1;
/// The "everything" marker written by primary reveals.
const ALL_EMISSIONS: u64 = STATE_MASK;
/// Most emissions of one cell in one event.
pub const MAX_EMISSIONS: u32 = (STATE_MASK - 1) as u32;
const MAX_EVENTS: u64 = 1 << 40;

const KEY_PRIMARY: u64 = 0;
const KEY_STREAM: u64 = 2 << 32;

/// A slot: its reveal state (`event << STATE_BITS | emissions revealed`) next
/// to the cell it reads, so one cache line serves a reveal and a read.
#[derive(Debug)]
struct SlotCell {
    state: AtomicU64,
    cell: u32,
    /// The cell is fed by an instrument set (`AllPrimary`): every event of the
    /// group is the reader's own (see [`IndicatorBank::own`]).
    fleet: bool,
}

/// A strategy's route to reveal, resolved once with
/// [`IndicatorBank::primary_handle`] or [`IndicatorBank::stream_handle`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Reveal(u32);

#[derive(Debug, Clone, Copy)]
struct Route {
    key: u64,
    start: u32,
    end: u32,
}

fn csr(lists: Vec<Vec<u32>>) -> (Box<[u32]>, Box<[u32]>) {
    let mut offsets = Vec::with_capacity(lists.len() + 1);
    let mut items = Vec::new();
    offsets.push(0);
    for list in lists {
        items.extend(list);
        offsets.push(items.len() as u32);
    }
    (offsets.into(), items.into())
}

fn csr_row<'a>(offsets: &[u32], items: &'a [u32], row: usize) -> &'a [u32] {
    match (offsets.get(row), offsets.get(row + 1)) {
        (Some(&a), Some(&b)) => &items[a as usize..b as usize],
        _ => &[],
    }
}

/// The cells of a frozen [`IndicatorRegistry`] and the reveal protocol over
/// them. See the module docs.
#[derive(Debug)]
pub struct IndicatorBank {
    event: AtomicU64,
    /// The first event of the timestamp group in flight (or the event itself
    /// outside a group): a cell touched at or after it was fed by this group.
    floor: AtomicU64,
    /// `0` no group, `1` a group opened and awaiting its first event, `2` in progress.
    group: AtomicU8,
    updates: AtomicU64,
    keys: Box<[IndKey]>,
    cells: Box<[CellVals]>,
    writer: Mutex<Writer>,
    /// Cell behind each slot.
    slots: Box<[SlotCell]>,
    /// Per strategy, its routes sorted by key; the slots of a route.
    strategy_routes: Box<[u32]>,
    routes: Box<[Route]>,
    route_slots: Box<[u32]>,
    /// Cells fed by an instrument's primary bar, and by an aggregation.
    primary_off: Box<[u32]>,
    primary_cells: Box<[u32]>,
    agg_off: Box<[u32]>,
    agg_cells: Box<[u32]>,
}

impl IndicatorBank {
    /// Freeze `registry`'s registrations into a bank. The registry is no
    /// longer needed afterwards.
    pub fn new(registry: &IndicatorRegistry) -> std::sync::Arc<Self> {
        let keys: Box<[IndKey]> = registry.keys().into();

        let mut by_primary: Vec<Vec<u32>> = Vec::new();
        let mut by_agg: Vec<Vec<u32>> = Vec::new();
        let push = |lists: &mut Vec<Vec<u32>>, at: usize, cell: u32| {
            if lists.len() <= at {
                lists.resize_with(at + 1, Vec::new);
            }
            lists[at].push(cell);
        };
        for (i, key) in keys.iter().enumerate() {
            let cell = i as u32;
            match key.input {
                IndInput::Primary(inst) => push(&mut by_primary, inst.index(), cell),
                IndInput::AllPrimary(order) => {
                    for inst in &registry.orders()[order.index()] {
                        push(&mut by_primary, inst.index(), cell);
                    }
                }
                IndInput::Composite(agg) => push(&mut by_agg, agg.index(), cell),
            }
        }
        let (primary_off, primary_cells) = csr(by_primary);
        let (agg_off, agg_cells) = csr(by_agg);

        let slots = registry.slots();
        let slot_cells: Box<[SlotCell]> = slots
            .iter()
            .map(|s| SlotCell {
                state: AtomicU64::new(0),
                cell: s.ind.raw(),
                fleet: matches!(keys[s.ind.index()].input, IndInput::AllPrimary(_)),
            })
            .collect();
        // A slot is revealed by one route per primary bar that feeds it, or by
        // its composite stream's route.
        let mut grouped: Vec<(u32, u64, u32)> = Vec::with_capacity(slots.len());
        for (i, s) in slots.iter().enumerate() {
            let mut add = |key: u64| grouped.push((s.strategy.raw(), key, i as u32));
            match (keys[s.ind.index()].input, s.stream) {
                (IndInput::Primary(inst), _) => add(KEY_PRIMARY | u64::from(inst.raw())),
                (IndInput::AllPrimary(order), _) => {
                    for inst in &registry.orders()[order.index()] {
                        add(KEY_PRIMARY | u64::from(inst.raw()));
                    }
                }
                (IndInput::Composite(_), Some(l)) => add(KEY_STREAM | u64::from(l.0)),
                (IndInput::Composite(_), None) => {
                    unreachable!("the registry requires a stream for composites")
                }
            }
        }
        grouped.sort_unstable();
        let strategies = grouped.last().map_or(0, |g| g.0 as usize + 1);
        let mut strategy_routes = vec![0u32; strategies + 1];
        let mut routes: Vec<Route> = Vec::new();
        let mut route_slots: Vec<u32> = Vec::with_capacity(grouped.len());
        let mut last: Option<(u32, u64)> = None;
        for &(strategy, key, slot) in &grouped {
            if last != Some((strategy, key)) {
                if let Some(open) = routes.last_mut() {
                    open.end = route_slots.len() as u32;
                }
                routes.push(Route { key, start: route_slots.len() as u32, end: 0 });
                strategy_routes[strategy as usize + 1] += 1;
                last = Some((strategy, key));
            }
            route_slots.push(slot);
        }
        if let Some(open) = routes.last_mut() {
            open.end = route_slots.len() as u32;
        }
        for s in 0..strategies {
            strategy_routes[s + 1] += strategy_routes[s];
        }

        let cores: Vec<CellCore> = keys.iter().map(CellCore::new).collect();
        let hist = vec![Vec::new(); keys.len()];
        std::sync::Arc::new(Self {
            event: AtomicU64::new(0),
            floor: AtomicU64::new(1),
            group: AtomicU8::new(0),
            updates: AtomicU64::new(0),
            cells: keys.iter().map(|_| CellVals::new()).collect(),
            keys,
            writer: Mutex::new(Writer { cores, hist }),
            slots: slot_cells,
            strategy_routes: strategy_routes.into(),
            routes: routes.into(),
            route_slots: route_slots.into(),
            primary_off,
            primary_cells,
            agg_off,
            agg_cells,
        })
    }

    /// Distinct cells.
    pub fn cells(&self) -> usize {
        self.cells.len()
    }

    /// Registrations (reader slots).
    pub fn registrations(&self) -> usize {
        self.slots.len()
    }

    /// Core updates performed so far: one per cell per emission, independent
    /// of the number of readers.
    pub fn updates(&self) -> u64 {
        self.updates.load(Ordering::Relaxed)
    }

    /// The key of a cell.
    pub fn key(&self, ind: IndId) -> Option<&IndKey> {
        self.keys.get(ind.index())
    }

    /// The cell a slot reads.
    pub fn cell_of(&self, slot: SlotId) -> IndId {
        IndId::new(self.slots[slot.index()].cell)
    }

    /// Start an event. Call before any emission of it.
    ///
    /// # Panics
    /// After 2^40 events, the limit of the reveal stamps.
    pub fn begin_event(&self) {
        let next = self.event.load(Ordering::Relaxed) + 1;
        assert!(next < MAX_EVENTS, "indicator bank: event counter exhausted");
        self.event.store(next, Ordering::Relaxed);
        match self.group.load(Ordering::Relaxed) {
            0 => self.floor.store(next, Ordering::Relaxed),
            1 => {
                self.floor.store(next, Ordering::Relaxed);
                self.group.store(2, Ordering::Relaxed);
            }
            _ => {}
        }
    }

    /// The events until [`IndicatorBank::close_group`] are one timestamp group:
    /// a cell fed by several of them shows, to a reader before it has revealed
    /// its own event, the value before the group.
    pub fn open_group(&self) {
        self.group.store(1, Ordering::Relaxed);
    }

    /// End the group opened by [`IndicatorBank::open_group`].
    pub fn close_group(&self) {
        self.group.store(0, Ordering::Relaxed);
    }

    /// The first event of the group in flight: a cell touched at or after it
    /// was fed by this group.
    pub fn floor(&self) -> u64 {
        self.floor.load(Ordering::Relaxed)
    }

    /// The number of the event in flight, to select again later with
    /// [`IndicatorBank::set_event`]. A group of events at one timestamp is
    /// ingested event by event, then decided event by event: the selected
    /// event is the one whose own stream the reveal protocol applies to; every
    /// other cell shows its value after the group.
    pub fn event_mark(&self) -> u64 {
        self.event.load(Ordering::Relaxed)
    }

    /// Make `event` (from [`IndicatorBank::event_mark`]) the event the readers
    /// see as the one being decided. After a group, select the last one again
    /// so the next [`IndicatorBank::begin_event`] numbers past every cell's
    /// stamp.
    pub fn set_event(&self, event: u64) {
        self.event.store(event, Ordering::Relaxed);
    }

    /// Feed `instrument`'s primary bar to every cell that reads it: its own
    /// `Primary` cells and the `AllPrimary` cells whose set contains it.
    pub fn emit_primary(&self, instrument: InstrumentId, bar: Ohlc) -> Result<()> {
        self.emit(csr_row(&self.primary_off, &self.primary_cells, instrument.index()), bar)
    }

    /// Feed one bar emitted by a composite aggregation. Call once per bar, in
    /// emission order.
    pub fn emit_composite(&self, agg: AggId, bar: Ohlc) -> Result<()> {
        self.emit(csr_row(&self.agg_off, &self.agg_cells, agg.index()), bar)
    }

    fn emit(&self, cells: &[u32], bar: Ohlc) -> Result<()> {
        let event = self.event.load(Ordering::Relaxed);
        let floor = self.floor.load(Ordering::Relaxed);
        let mut w = self.writer.lock().expect("indicator writer poisoned");
        for &i in cells {
            let c = &self.cells[i as usize];
            if c.touched.load(Ordering::Relaxed) >= floor
                && c.emissions.load(Ordering::Relaxed) >= MAX_EMISSIONS
            {
                return Err(RaptorError::invalid_parameter(format!(
                    "indicator {i}: more than {MAX_EMISSIONS} emissions in one event"
                )));
            }
        }
        for &i in cells {
            let i = i as usize;
            let c = &self.cells[i];
            if c.touched.load(Ordering::Relaxed) >= floor {
                let after_previous = c.cur.load();
                w.hist[i].push(after_previous);
            } else {
                c.prev.store(c.cur.load());
                c.touched.store(event, Ordering::Relaxed);
                c.emissions.store(0, Ordering::Relaxed);
                w.hist[i].clear();
            }
            let value = w.cores[i].update(bar);
            c.cur.store(value);
            c.emissions.fetch_add(1, Ordering::Relaxed);
            c.total.fetch_add(1, Ordering::Relaxed);
        }
        self.updates.fetch_add(cells.len() as u64, Ordering::Relaxed);
        Ok(())
    }

    /// The handle for revealing `strategy`'s composite stream `stream`, or
    /// `None` if it has no indicator on it. Resolve once at setup.
    pub fn stream_handle(&self, strategy: StrategyId, stream: LocalStreamId) -> Option<Reveal> {
        self.route(strategy, KEY_STREAM | u64::from(stream.0))
    }

    /// The handle for revealing `strategy`'s indicators fed by `instrument`'s
    /// primary bar: those on that instrument and its `AllPrimary` ones whose
    /// set contains it. `None` if there are none. Resolve once at setup.
    pub fn primary_handle(&self, strategy: StrategyId, instrument: InstrumentId) -> Option<Reveal> {
        self.route(strategy, KEY_PRIMARY | u64::from(instrument.raw()))
    }

    /// Strategy `strategy` has processed emission `emission` (1-based) of its
    /// local composite stream `stream`: its indicators on that stream now
    /// show the value after it.
    pub fn reveal_stream(&self, strategy: StrategyId, stream: LocalStreamId, emission: u32) {
        if let Some(h) = self.stream_handle(strategy, stream) {
            self.reveal(h, emission);
        }
    }

    /// Strategy `strategy` has processed the primary bar of `instrument`: its
    /// indicators fed by it now show the value after it.
    pub fn reveal_primary(&self, strategy: StrategyId, instrument: InstrumentId) {
        if let Some(h) = self.primary_handle(strategy, instrument) {
            self.reveal(h, 1);
        }
    }

    /// [`reveal_stream`](Self::reveal_stream) or
    /// [`reveal_primary`](Self::reveal_primary) through a handle, without a
    /// lookup. A primary handle reveals everything and ignores `emission`.
    #[inline]
    pub fn reveal(&self, handle: Reveal, emission: u32) {
        let route = &self.routes[handle.0 as usize];
        let k = if route.key >= KEY_STREAM {
            u64::from(emission).min(ALL_EMISSIONS - 1)
        } else {
            ALL_EMISSIONS
        };
        let state = self.event.load(Ordering::Relaxed) << STATE_BITS | k;
        for &s in &self.route_slots[route.start as usize..route.end as usize] {
            self.slots[s as usize].state.store(state, Ordering::Relaxed);
        }
    }

    fn route(&self, strategy: StrategyId, key: u64) -> Option<Reveal> {
        let s = strategy.index();
        let (a, b) = (*self.strategy_routes.get(s)?, *self.strategy_routes.get(s + 1)?);
        let routes = &self.routes[a as usize..b as usize];
        let i = routes.binary_search_by_key(&key, |r| r.key).ok()?;
        Some(Reveal(a + i as u32))
    }

    /// How many bars the slot's strategy has fed its cell so far: the
    /// `update_count` of a private indicator, which counts the same bars.
    pub fn updates_seen(&self, slot: SlotId) -> u64 {
        let sc = &self.slots[slot.index()];
        let c = &self.cells[sc.cell as usize];
        let total = c.total.load(Ordering::Relaxed);
        let event = self.event.load(Ordering::Relaxed);
        if !self.own(sc, c, event) {
            return total;
        }
        let state = sc.state.load(Ordering::Relaxed);
        let k = if state >> STATE_BITS == event { state & STATE_MASK } else { 0 };
        let m = u64::from(c.emissions.load(Ordering::Relaxed));
        total - m + k.min(m)
    }

    /// Whether the reveal protocol applies to a read of `c` at `event`: the
    /// cell was fed by the selected event, or, for a cell fed by an instrument
    /// set, by the group in flight (the reader is a strategy deciding one of
    /// the set's instruments, so the group's events are all its own).
    #[inline]
    fn own(&self, sc: &SlotCell, c: &CellVals, event: u64) -> bool {
        let touched = c.touched.load(Ordering::Relaxed);
        if sc.fleet {
            touched >= self.floor.load(Ordering::Relaxed)
        } else {
            touched == event
        }
    }

    /// The value the slot's strategy sees now.
    #[inline]
    pub fn read(&self, slot: SlotId) -> Reading {
        let sc = &self.slots[slot.index()];
        let cell = sc.cell as usize;
        let c = &self.cells[cell];
        let event = self.event.load(Ordering::Relaxed);
        // Only the event that fed the cell, which is the selected event for a
        // reader of its own instrument's stream, has a reveal protocol. A cell
        // fed by another event of the group, earlier or later, shows the value
        // after it: every event of a group is applied before any is decided.
        if !self.own(sc, c, event) {
            return c.cur.load();
        }
        let state = sc.state.load(Ordering::Relaxed);
        let k = if state >> STATE_BITS == event { state & STATE_MASK } else { 0 };
        if k == 0 {
            return c.prev.load();
        }
        if k >= u64::from(c.emissions.load(Ordering::Relaxed)) {
            return c.cur.load();
        }
        self.writer.lock().expect("indicator writer poisoned").hist[cell][k as usize - 1]
    }
}

impl IndicatorBank {
    /// Heap held by per-registration entries (reader slots and their reveal
    /// routes), as opposed to the shared cells.
    pub fn link_heap_bytes(&self) -> usize {
        use std::mem::size_of;
        self.slots.len() * size_of::<SlotCell>()
            + self.strategy_routes.len() * size_of::<u32>()
            + self.routes.len() * size_of::<Route>()
            + self.route_slots.len() * size_of::<u32>()
    }
}

impl HeapSize for IndicatorBank {
    /// Everything the bank owns, excluding allocator slack: fixed arrays at
    /// their length, `hist` at its capacity, and each core's windows as
    /// derived from its parameters (see `core_heap_bytes`).
    fn heap_bytes(&self) -> usize {
        use std::mem::size_of;
        let w = self.writer.lock().expect("indicator writer poisoned");
        self.keys.len() * size_of::<IndKey>()
            + self.cells.len() * size_of::<CellVals>()
            + w.cores.capacity() * size_of::<CellCore>()
            + self.keys.iter().map(core_heap_bytes).sum::<usize>()
            + w.hist.capacity() * size_of::<Vec<Reading>>()
            + w.hist.iter().map(|h| h.capacity() * size_of::<Reading>()).sum::<usize>()
            + self.slots.len() * size_of::<SlotCell>()
            + self.strategy_routes.len() * size_of::<u32>()
            + self.routes.len() * size_of::<Route>()
            + self.route_slots.len() * size_of::<u32>()
            + (self.primary_off.len() + self.primary_cells.len()) * size_of::<u32>()
            + (self.agg_off.len() + self.agg_cells.len()) * size_of::<u32>()
    }
}
