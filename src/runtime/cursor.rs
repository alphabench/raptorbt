//! Lazy k-way merge over a [`MarketStore`], with no materialised schedule.
//!
//! The cursor yields the same events in the same order as the batch event
//! session's sealed schedule, but holds only one heap entry per stream and
//! hands out small [`EventRef`]s that point back into the columns.
//!
//! # Ordering
//!
//! Events are delivered by `(timestamp, phase, stream, row)` ascending, the
//! key `EventFeed` uses. Streams are numbered exactly as the session numbers
//! them when it seals:
//!
//! 1. one bar stream per instrument, `0..I` (empty if the instrument has no
//!    bars);
//! 2. for each instrument with a tick series attached, in id order, a trade
//!    stream then a quote stream;
//! 3. for each instrument with non-empty depth, in id order, a depth stream.
//!
//! A tick row yields a trade when its last price is positive and a quote when
//! both sides of the book are positive (`tick_data_to_events`'s rules), so a
//! row can produce either, both or neither.
//!
//! `local_idx` is the per-instrument ordinal of the event across all of that
//! instrument's streams, in delivery order.

use std::cmp::Reverse;
use std::collections::binary_heap::PeekMut;
use std::collections::BinaryHeap;

use crate::data::book::DepthTick;
use crate::data::events::{QuoteTick, TradeTick};
use crate::portfolio::kernel::KernelBar;
use crate::runtime::ids::InstrumentId;
use crate::runtime::store::{MarketStore, TickColumns};

/// What an [`EventRef`] points at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventKind {
    Trade,
    Quote,
    Depth,
    Bar,
}

impl EventKind {
    /// Merge priority at equal timestamps; matches `MarketEvent::phase`:
    /// trades, then quotes, then depth, then the bar summarising them.
    #[inline]
    pub const fn phase(self) -> u8 {
        match self {
            EventKind::Trade => 0,
            EventKind::Quote => 1,
            EventKind::Depth => 2,
            EventKind::Bar => 3,
        }
    }
}

/// A pointer to one event in the store.
///
/// `row` indexes the instrument's bars, ticks or depth according to `kind`.
/// Build the payload on demand with [`MergeCursor::bar`] and friends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EventRef {
    pub ts: i64,
    pub instrument: InstrumentId,
    pub row: u64,
    /// Per-instrument event ordinal, monotone across all its streams.
    pub local_idx: u64,
    pub kind: EventKind,
}

/// Heap key. Derived ordering is field order: `(ts, phase, stream, row)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Key {
    ts: i64,
    phase: u8,
    stream: u32,
    row: u32,
}

#[derive(Clone, Copy)]
enum Source<'a> {
    Bars(&'a [i64]),
    Trades(&'a TickColumns),
    Quotes(&'a TickColumns),
    Depth(&'a [DepthTick]),
}

impl Source<'_> {
    fn kind(&self) -> EventKind {
        match self {
            Source::Bars(_) => EventKind::Bar,
            Source::Trades(_) => EventKind::Trade,
            Source::Quotes(_) => EventKind::Quote,
            Source::Depth(_) => EventKind::Depth,
        }
    }

    fn len(&self) -> usize {
        match self {
            Source::Bars(ts) => ts.len(),
            Source::Trades(t) | Source::Quotes(t) => t.len(),
            Source::Depth(d) => d.len(),
        }
    }

    /// Timestamp of `row` if that row produces an event on this stream.
    #[inline]
    fn event_ts(&self, row: u32) -> Option<i64> {
        match self {
            Source::Bars(ts) => Some(ts[row as usize]),
            Source::Trades(t) => t.has_trade(row).then(|| t.ts[row as usize]),
            Source::Quotes(t) => t.has_quote(row).then(|| t.ts[row as usize]),
            Source::Depth(d) => Some(d[row as usize].timestamp),
        }
    }
}

struct Stream<'a> {
    instrument: InstrumentId,
    source: Source<'a>,
    /// Next row to examine.
    next: u32,
    len: u32,
}

impl Stream<'_> {
    /// Advance to this stream's next event row and return its key.
    #[inline]
    fn fill(&mut self, id: u32) -> Option<Key> {
        let phase = self.source.kind().phase();
        while self.next < self.len {
            let row = self.next;
            self.next += 1;
            if let Some(ts) = self.source.event_ts(row) {
                return Some(Key { ts, phase, stream: id, row });
            }
        }
        None
    }
}

/// Lazy merge of every stream in a store, in `EventFeed` order.
pub struct MergeCursor<'a> {
    store: &'a MarketStore,
    streams: Vec<Stream<'a>>,
    heap: BinaryHeap<Reverse<Key>>,
    /// Next `local_idx` per instrument.
    local_next: Vec<u64>,
    /// Last bar row yielded per instrument.
    bar_visible: Vec<Option<u32>>,
    /// Last tick row that yielded a trade or quote, per instrument.
    tick_visible: Vec<Option<u32>>,
}

impl<'a> MergeCursor<'a> {
    pub fn new(store: &'a MarketStore) -> Self {
        Self::filtered(store, |_| true)
    }

    /// A cursor over the instruments for which `keep` holds. The events it
    /// yields are those of `new` minus the dropped instruments', in the same
    /// relative order and with the same per-instrument ordinals.
    pub fn filtered(store: &'a MarketStore, keep: impl Fn(InstrumentId) -> bool) -> Self {
        let n = store.len();
        let mut streams: Vec<Stream<'a>> = Vec::new();
        let mut push = |instrument: InstrumentId, source: Source<'a>| {
            // Row counts were checked against `u32` when the columns were built.
            let len = source.len() as u32;
            streams.push(Stream { instrument, source, next: 0, len });
        };
        let ids = || (0..n as u32).map(InstrumentId::new).filter(|&id| keep(id));
        for id in ids() {
            let ts: &[i64] = store.bars(id).map_or(&[], |b| &b.ts);
            push(id, Source::Bars(ts));
        }
        for id in ids() {
            if let Some(t) = store.ticks(id) {
                push(id, Source::Trades(t));
                push(id, Source::Quotes(t));
            }
        }
        for id in ids() {
            if let Some(d) = store.depth(id).filter(|d| !d.is_empty()) {
                push(id, Source::Depth(&d.ticks));
            }
        }
        let mut heap = Vec::with_capacity(streams.len());
        for (i, stream) in streams.iter_mut().enumerate() {
            if let Some(key) = stream.fill(i as u32) {
                heap.push(Reverse(key));
            }
        }
        Self {
            store,
            streams,
            heap: BinaryHeap::from(heap),
            local_next: vec![0; n],
            bar_visible: vec![None; n],
            tick_visible: vec![None; n],
        }
    }

    /// The `(timestamp, phase)` of the next event, without taking it. Events
    /// that share it are one group (see `docs/architecture/event-semantics.md`).
    #[inline]
    pub fn peek_group(&self) -> Option<(i64, u8)> {
        self.heap.peek().map(|Reverse(k)| (k.ts, k.phase))
    }

    /// Events yielded so far for `instrument`, across all its streams.
    #[inline]
    pub fn delivered(&self, instrument: InstrumentId) -> u64 {
        self.local_next.get(instrument.index()).copied().unwrap_or(0)
    }

    /// Whether every stream has been read to its end.
    #[inline]
    pub fn is_drained(&self) -> bool {
        self.heap.is_empty()
    }

    /// Last row of the instrument's bar stream already yielded.
    #[inline]
    pub fn visible_row(&self, instrument: InstrumentId) -> Option<u32> {
        self.bar_visible.get(instrument.index()).copied().flatten()
    }

    /// Last tick row of the instrument that has yielded a trade or quote.
    #[inline]
    pub fn visible_tick_row(&self, instrument: InstrumentId) -> Option<u32> {
        self.tick_visible.get(instrument.index()).copied().flatten()
    }

    /// The bar behind a [`EventKind::Bar`] event.
    #[inline]
    pub fn bar(&self, e: &EventRef) -> Option<KernelBar> {
        (e.kind == EventKind::Bar).then(|| self.store.bar(e.instrument, e.row as u32)).flatten()
    }

    /// The print behind a [`EventKind::Trade`] event.
    #[inline]
    pub fn trade(&self, e: &EventRef) -> Option<TradeTick> {
        (e.kind == EventKind::Trade).then(|| self.store.trade(e.instrument, e.row as u32)).flatten()
    }

    /// The quote behind a [`EventKind::Quote`] event.
    #[inline]
    pub fn quote(&self, e: &EventRef) -> Option<QuoteTick> {
        (e.kind == EventKind::Quote).then(|| self.store.quote(e.instrument, e.row as u32)).flatten()
    }

    /// The snapshot behind a [`EventKind::Depth`] event.
    #[inline]
    pub fn depth(&self, e: &EventRef) -> Option<DepthTick> {
        (e.kind == EventKind::Depth)
            .then(|| self.store.depth_at(e.instrument, e.row as u32))
            .flatten()
    }
}

impl Iterator for MergeCursor<'_> {
    type Item = EventRef;

    fn next(&mut self) -> Option<EventRef> {
        let mut top = self.heap.peek_mut()?;
        let key = top.0;
        let stream = &mut self.streams[key.stream as usize];
        let instrument = stream.instrument;
        let kind = stream.source.kind();
        // Refill in place: one sift instead of a pop and a push.
        match stream.fill(key.stream) {
            Some(next) => *top = Reverse(next),
            None => {
                PeekMut::pop(top);
            }
        }
        let i = instrument.index();
        let local_idx = self.local_next[i];
        self.local_next[i] += 1;
        let visible = match kind {
            EventKind::Bar => Some(&mut self.bar_visible[i]),
            EventKind::Trade | EventKind::Quote => Some(&mut self.tick_visible[i]),
            EventKind::Depth => None,
        };
        if let Some(slot) = visible {
            *slot = Some(slot.map_or(key.row, |r| r.max(key.row)));
        }
        Some(EventRef { ts: key.ts, instrument, row: u64::from(key.row), local_idx, kind })
    }
}
