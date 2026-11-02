//! Bounded storage and ordering for data that is pushed, not loaded.
//!
//! A [`LiveFeed`] keeps, per instrument, a ring of bars, a ring of accepted
//! tick rows and a ring of depth snapshots, all addressed by absolute row, and
//! hands out the [`EventRef`] of each pushed event. It is the live counterpart
//! of the immutable [`MarketStore`](crate::runtime::MarketStore) plus
//! [`MergeCursor`](crate::runtime::MergeCursor): the same `EventRef`s go to
//! the same [`RuntimeCore`](crate::runtime::RuntimeCore) code.
//!
//! # Memory
//!
//! Each ring is allocated on the instrument's first push and never grows.
//! Rows nobody has asked to keep are overwritten oldest-first. A strategy that
//! has asked to keep the last `n` rows ([`LiveFeed::keep`]) holds them back;
//! if the ring cannot take a push without dropping a row such a strategy still
//! needs, the push fails with [`RingError::Backpressure`] before anything
//! changes. Nothing is ever dropped silently.

use crate::core::error::RaptorError;
use crate::core::Result;
use crate::data::book::DepthTick;
use crate::portfolio::kernel::KernelBar;
use crate::runtime::cursor::{EventKind, EventRef};
use crate::runtime::ids::{InstrumentId, StrategyId};
use crate::runtime::memory::HeapSize;
use crate::runtime::ring::{ColumnSource, ConsumerId, RingColumns, RingError, RowRing, TickRow};

/// A strategy's request to keep the last `rows` rows of one instrument.
#[derive(Debug, Clone, Copy)]
struct Keep {
    owner: StrategyId,
    rows: u64,
    bars: ConsumerId,
    ticks: ConsumerId,
}

#[derive(Debug, Default)]
struct Rings {
    bars: Option<RingColumns>,
    ticks: Option<RowRing<TickRow>>,
    depth: Option<RowRing<DepthTick>>,
    keeps: Vec<Keep>,
}

/// Which ring a [`RingError`] came from, for the message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RingKind {
    Bars,
    Ticks,
    Depth,
}

impl RingKind {
    pub fn name(self) -> &'static str {
        match self {
            RingKind::Bars => "bar",
            RingKind::Ticks => "tick",
            RingKind::Depth => "depth",
        }
    }
}

/// Bounded rings for every instrument of a live node.
#[derive(Debug)]
pub struct LiveFeed {
    symbols: Vec<String>,
    capacity: usize,
    rings: Vec<Rings>,
    /// Latest event timestamp per instrument.
    last_ts: Vec<i64>,
    /// Next per-instrument event ordinal.
    local_next: Vec<u64>,
}

impl LiveFeed {
    /// A feed for `symbols`, every ring holding `capacity` rows.
    pub fn new(symbols: Vec<String>, capacity: usize) -> Self {
        let n = symbols.len();
        Self {
            symbols,
            capacity: capacity.max(1),
            rings: (0..n).map(|_| Rings::default()).collect(),
            last_ts: vec![i64::MIN; n],
            local_next: vec![0; n],
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Refuse an event older than the instrument's latest one. Equal
    /// timestamps are allowed.
    pub fn check_timestamp(&self, instrument: InstrumentId, ts: i64) -> Result<()> {
        let i = instrument.index();
        if ts < self.last_ts[i] {
            return Err(RaptorError::invalid_parameter(format!(
                "{} live feed: timestamps must not decrease, but event {} is {ts} after {}",
                self.symbols[i], self.local_next[i], self.last_ts[i],
            )));
        }
        Ok(())
    }

    /// Ask that the last `rows` bars and ticks of `instrument` stay readable
    /// for `owner`. Repeating the request keeps the larger number.
    pub fn keep(&mut self, owner: StrategyId, instrument: InstrumentId, rows: u64) {
        let capacity = self.capacity;
        let r = &mut self.rings[instrument.index()];
        if let Some(k) = r.keeps.iter_mut().find(|k| k.owner == owner) {
            k.rows = k.rows.max(rows);
            return;
        }
        let bars = r.bars.get_or_insert_with(|| RingColumns::new(capacity, 0)).register_consumer();
        let ticks = r
            .ticks
            .get_or_insert_with(|| RowRing::new(capacity, 0, TickRow::default()))
            .register_consumer();
        r.keeps.push(Keep { owner, rows, bars, ticks });
    }

    fn release(r: &mut Rings, kind: RingKind) {
        // What each keeper needs after the coming push: the last `rows`
        // rows including the new one.
        for k in &r.keeps {
            match kind {
                RingKind::Bars => {
                    if let Some(ring) = r.bars.as_mut() {
                        let to = (ring.end_row() + 1).saturating_sub(k.rows).min(ring.end_row());
                        let to = to.max(ring.watermark(k.bars).unwrap_or(0));
                        let _ = ring.advance(k.bars, to);
                    }
                }
                RingKind::Ticks => {
                    if let Some(ring) = r.ticks.as_mut() {
                        let to = (ring.end_row() + 1).saturating_sub(k.rows).min(ring.end_row());
                        let to = to.max(ring.watermark(k.ticks).unwrap_or(0));
                        let _ = ring.advance(k.ticks, to);
                    }
                }
                RingKind::Depth => {}
            }
        }
    }

    /// Store a bar and return the event for it.
    pub fn push_bar(
        &mut self,
        instrument: InstrumentId,
        event_ts: i64,
        bar: KernelBar,
    ) -> std::result::Result<EventRef, RingError> {
        let capacity = self.capacity;
        let r = &mut self.rings[instrument.index()];
        Self::release(r, RingKind::Bars);
        let ring = r.bars.get_or_insert_with(|| RingColumns::new(capacity, 0));
        let row = ring.end_row();
        ring.push(row, bar)?;
        Ok(self.event(instrument, EventKind::Bar, event_ts, row))
    }

    /// Store an accepted tick row. Returns its absolute row; the events it
    /// yields (a trade, a quote) are made with [`LiveFeed::event`].
    pub fn push_tick(
        &mut self,
        instrument: InstrumentId,
        row: TickRow,
    ) -> std::result::Result<u64, RingError> {
        let capacity = self.capacity;
        let r = &mut self.rings[instrument.index()];
        Self::release(r, RingKind::Ticks);
        let ring = r.ticks.get_or_insert_with(|| RowRing::new(capacity, 0, TickRow::default()));
        let seq = ring.end_row();
        ring.push(seq, row)?;
        Ok(seq)
    }

    /// Store a depth snapshot and return the event for it.
    pub fn push_depth(
        &mut self,
        instrument: InstrumentId,
        snapshot: DepthTick,
    ) -> std::result::Result<EventRef, RingError> {
        let capacity = self.capacity;
        let ts = snapshot.timestamp;
        let r = &mut self.rings[instrument.index()];
        let ring = r
            .depth
            .get_or_insert_with(|| RowRing::new(capacity, 0, DepthTick::from_levels(0, &[], &[])));
        let row = ring.end_row();
        ring.push(row, snapshot)?;
        Ok(self.event(instrument, EventKind::Depth, ts, row))
    }

    /// The next event of `instrument`: takes the next ordinal and moves the
    /// instrument's latest timestamp.
    pub fn event(
        &mut self,
        instrument: InstrumentId,
        kind: EventKind,
        ts: i64,
        row: u64,
    ) -> EventRef {
        let i = instrument.index();
        let local_idx = self.local_next[i];
        self.local_next[i] += 1;
        self.last_ts[i] = self.last_ts[i].max(ts);
        EventRef { ts, instrument, row, local_idx, kind }
    }

    /// Ordinal of the instrument's latest event (`0` before it has had one).
    pub fn last_ordinal(&self, instrument: InstrumentId) -> usize {
        self.local_next[instrument.index()].saturating_sub(1) as usize
    }

    /// Whether the instrument has ever been pushed a bar.
    pub fn has_bars(&self, instrument: InstrumentId) -> bool {
        self.rings[instrument.index()].bars.as_ref().is_some_and(|b| b.end_row() > 0)
    }

    pub fn bars(&self, instrument: InstrumentId) -> Option<&RingColumns> {
        self.rings[instrument.index()].bars.as_ref()
    }

    pub fn ticks(&self, instrument: InstrumentId) -> Option<&RowRing<TickRow>> {
        self.rings[instrument.index()].ticks.as_ref()
    }

    pub fn depth_at(&self, instrument: InstrumentId, row: u64) -> Option<DepthTick> {
        self.rings[instrument.index()].depth.as_ref()?.get(row).ok()
    }

    /// Explain a ring error for the user of `instrument`'s `kind` ring.
    pub fn explain(&self, instrument: InstrumentId, kind: RingKind, err: &RingError) -> String {
        let symbol = &self.symbols[instrument.index()];
        match err {
            RingError::Backpressure { capacity, oldest, slowest } => format!(
                "{symbol}: the {} history is full ({capacity} rows) and row {oldest} is still \
                 needed by a strategy that asked to keep it (it has released rows below \
                 {slowest}). Nothing was dropped and the push was refused. Raise `history`, or \
                 ask the strategy to keep fewer rows.",
                kind.name()
            ),
            other => format!("{symbol}: {} history: {other}", kind.name()),
        }
    }
}

impl HeapSize for LiveFeed {
    fn heap_bytes(&self) -> usize {
        use std::mem::size_of;
        let rings: usize = self
            .rings
            .iter()
            .map(|r| {
                r.bars.as_ref().map_or(0, HeapSize::heap_bytes)
                    + r.ticks.as_ref().map_or(0, HeapSize::heap_bytes)
                    + r.depth.as_ref().map_or(0, HeapSize::heap_bytes)
                    + r.keeps.capacity() * size_of::<Keep>()
            })
            .sum();
        rings
            + self.rings.capacity() * size_of::<Rings>()
            + self.last_ts.capacity() * size_of::<i64>()
            + self.local_next.capacity() * size_of::<u64>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bar(ts: i64) -> KernelBar {
        KernelBar { timestamp: ts, open: 1.0, high: 2.0, low: 0.5, close: 1.5, volume: 3.0 }
    }

    fn row(ts: i64) -> TickRow {
        TickRow { timestamp: ts, ltp: 10.0, ..TickRow::default() }
    }

    fn id(i: u32) -> InstrumentId {
        InstrumentId::new(i)
    }

    #[test]
    fn events_take_per_instrument_ordinals_and_the_latest_timestamp() {
        let mut feed = LiveFeed::new(vec!["A".into(), "B".into()], 4);
        let a0 = feed.push_bar(id(0), 10, bar(10)).unwrap();
        let b0 = feed.push_bar(id(1), 5, bar(5)).unwrap();
        let a1 = feed.push_bar(id(0), 10, bar(10)).unwrap();
        assert_eq!((a0.local_idx, b0.local_idx, a1.local_idx), (0, 0, 1));
        assert_eq!((a0.row, a1.row), (0, 1));
        assert!(feed.check_timestamp(id(0), 10).is_ok(), "equal timestamps are allowed");
        let err = feed.check_timestamp(id(0), 9).unwrap_err().to_string();
        assert!(err.contains("A live feed: timestamps must not decrease"), "{err}");
        assert!(err.contains("event 2 is 9 after 10"), "{err}");
        assert!(feed.check_timestamp(id(1), 6).is_ok(), "instruments are independent");
    }

    #[test]
    fn unkept_rows_are_overwritten_and_memory_is_fixed() {
        let mut feed = LiveFeed::new(vec!["A".into()], 8);
        feed.push_bar(id(0), 0, bar(0)).unwrap();
        feed.push_tick(id(0), row(0)).unwrap();
        let fixed = feed.heap_bytes();
        for t in 1..10_000 {
            feed.push_bar(id(0), t, bar(t)).unwrap();
            feed.push_tick(id(0), row(t)).unwrap();
        }
        assert_eq!(feed.heap_bytes(), fixed);
        let bars = feed.bars(id(0)).unwrap();
        assert_eq!((bars.first_row(), bars.end_row()), (9_992, 10_000));
    }

    #[test]
    fn a_kept_row_refuses_the_push_that_would_evict_it() {
        let mut feed = LiveFeed::new(vec!["A".into()], 4);
        // Four rows are held; the strategy keeps five, which cannot fit.
        feed.keep(StrategyId::new(0), id(0), 5);
        for t in 0..4 {
            feed.push_bar(id(0), t, bar(t)).unwrap();
        }
        let err = feed.push_bar(id(0), 4, bar(4)).unwrap_err();
        assert!(matches!(err, RingError::Backpressure { capacity: 4, oldest: 0, .. }), "{err:?}");
        let bars = feed.bars(id(0)).unwrap();
        assert_eq!((bars.first_row(), bars.end_row()), (0, 4), "the refused push changed nothing");
        let msg = feed.explain(id(0), RingKind::Bars, &err);
        assert!(msg.contains("Nothing was dropped"), "{msg}");
    }

    #[test]
    fn keeping_what_fits_never_blocks() {
        let mut feed = LiveFeed::new(vec!["A".into()], 4);
        feed.keep(StrategyId::new(0), id(0), 4);
        for t in 0..100 {
            feed.push_bar(id(0), t, bar(t)).unwrap();
            feed.push_tick(id(0), row(t)).unwrap();
        }
        let bars = feed.bars(id(0)).unwrap();
        assert_eq!((bars.first_row(), bars.end_row()), (96, 100));
        assert_eq!(feed.ticks(id(0)).unwrap().first_row(), 96);
    }

    #[test]
    fn depth_is_read_back_by_absolute_row() {
        let mut feed = LiveFeed::new(vec!["A".into()], 2);
        let snap = |ts| DepthTick::from_levels(ts, &[], &[]);
        let e0 = feed.push_depth(id(0), snap(1)).unwrap();
        feed.push_depth(id(0), snap(2)).unwrap();
        feed.push_depth(id(0), snap(3)).unwrap();
        assert_eq!(e0.row, 0);
        assert!(feed.depth_at(id(0), 0).is_none(), "overwritten");
        assert_eq!(feed.depth_at(id(0), 2).unwrap().timestamp, 3);
    }
}
