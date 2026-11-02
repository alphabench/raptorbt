//! Bounded bar storage for live data.
//!
//! A live feed has no backing array, so each instrument keeps a fixed-capacity
//! ring addressed by an absolute `u64` row index (the feed's sequence number).
//! Rows are only ever evicted once every registered consumer has moved past
//! them; when that would lose data a reader still needs, `push` fails with
//! [`RingError::Backpressure`] instead of dropping anything silently.

use std::ops::Range;

use thiserror::Error;

use crate::portfolio::kernel::KernelBar;
use crate::runtime::memory::HeapSize;
use crate::runtime::store::BarColumns;

/// Why an absolute row is not readable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowMiss {
    /// The row was pushed and has since been overwritten; `oldest` is the
    /// first row still held.
    Evicted { oldest: u64 },
    /// The row has not been pushed yet; `next` is the next row to arrive.
    NotYetArrived { next: u64 },
}

/// Bar columns addressed by absolute row, whether held in full (backtest)
/// or in a bounded window (live).
pub trait ColumnSource {
    /// The bar at `abs_row`.
    fn bar(&self, abs_row: u64) -> std::result::Result<KernelBar, RowMiss>;

    /// The oldest row still held.
    fn first_row(&self) -> u64;

    /// One past the newest row held.
    fn end_row(&self) -> u64;

    /// Rows readable by a reader whose last visible row is `last_visible`
    /// (`None` before its first): the held rows up to and including it.
    fn visible(&self, last_visible: Option<u64>) -> Range<u64> {
        let first = self.first_row();
        let end = last_visible.map_or(first, |r| r.saturating_add(1).min(self.end_row()));
        first..end.max(first)
    }
}

impl ColumnSource for BarColumns {
    fn bar(&self, abs_row: u64) -> std::result::Result<KernelBar, RowMiss> {
        u32::try_from(abs_row)
            .ok()
            .and_then(|row| BarColumns::bar(self, row))
            .ok_or(RowMiss::NotYetArrived { next: self.len() as u64 })
    }

    fn first_row(&self) -> u64 {
        0
    }

    fn end_row(&self) -> u64 {
        self.len() as u64
    }
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum RingError {
    /// The pushed sequence number is not the next expected one: rows were
    /// skipped (`got > expected`) or repeated (`got < expected`).
    #[error("sequence gap: expected row {expected}, got {got}")]
    Gap { expected: u64, got: u64 },

    /// The ring is full and the oldest row is still needed. Nothing was
    /// pushed or dropped.
    #[error(
        "ring full (capacity {capacity}): row {oldest} is still needed by a consumer at \
         watermark {slowest}"
    )]
    Backpressure { capacity: usize, oldest: u64, slowest: u64 },

    /// A pushed bar's timestamp is older than the previous bar's.
    #[error("timestamps must not decrease: row {row} is {got} after {last}")]
    TimeRegression { row: u64, last: i64, got: i64 },

    #[error("unknown consumer {0}")]
    UnknownConsumer(u32),

    /// A watermark may only move forward, and not past the next row to arrive.
    #[error("invalid watermark {to}: current {current}, next row {next}")]
    BadWatermark { current: u64, to: u64, next: u64 },
}

/// A registered reader of a ring.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConsumerId(u32);

/// The bookkeeping every ring shares: which absolute rows are held, who still
/// needs them, and when a push may evict.
///
/// A consumer's watermark is the next row it has yet to read: every row below
/// it is released. A row may be evicted only when the ring is full and every
/// registered consumer's watermark is past it. With no consumers registered
/// nothing holds rows back.
#[derive(Debug)]
struct Window {
    capacity: usize,
    /// Oldest row held.
    first: u64,
    /// Next row to arrive.
    next: u64,
    /// Watermark per consumer slot; `None` once unregistered.
    consumers: Vec<Option<u64>>,
}

impl Window {
    fn new(capacity: usize, start: u64) -> Self {
        assert!(capacity > 0, "ring capacity must be positive");
        Self { capacity, first: start, next: start, consumers: Vec::new() }
    }

    fn len(&self) -> usize {
        (self.next - self.first) as usize
    }

    fn register(&mut self) -> ConsumerId {
        self.consumers.push(Some(self.first));
        ConsumerId((self.consumers.len() - 1) as u32)
    }

    fn unregister(&mut self, id: ConsumerId) -> Result<(), RingError> {
        self.watermark(id)?;
        self.consumers[id.0 as usize] = None;
        Ok(())
    }

    fn watermark(&self, id: ConsumerId) -> Result<u64, RingError> {
        self.consumers.get(id.0 as usize).copied().flatten().ok_or(RingError::UnknownConsumer(id.0))
    }

    fn advance(&mut self, id: ConsumerId, to: u64) -> Result<(), RingError> {
        let current = self.watermark(id)?;
        if to < current || to > self.next {
            return Err(RingError::BadWatermark { current, to, next: self.next });
        }
        self.consumers[id.0 as usize] = Some(to);
        Ok(())
    }

    fn slowest(&self) -> Option<u64> {
        self.consumers.iter().flatten().copied().min()
    }

    fn check_seq(&self, seq: u64) -> Result<(), RingError> {
        if seq != self.next {
            return Err(RingError::Gap { expected: self.next, got: seq });
        }
        Ok(())
    }

    /// Claim the slot of the next row, evicting the oldest if the ring is
    /// full and nobody needs it. On `Backpressure` nothing changes.
    fn admit(&mut self) -> Result<usize, RingError> {
        if self.len() == self.capacity {
            if let Some(slowest) = self.slowest().filter(|&w| w <= self.first) {
                return Err(RingError::Backpressure {
                    capacity: self.capacity,
                    oldest: self.first,
                    slowest,
                });
            }
            self.first += 1;
        }
        let slot = self.slot(self.next);
        self.next += 1;
        Ok(slot)
    }

    /// The slot of a held row, or why it is not held.
    fn locate(&self, abs_row: u64) -> Result<usize, RowMiss> {
        if abs_row < self.first {
            return Err(RowMiss::Evicted { oldest: self.first });
        }
        if abs_row >= self.next {
            return Err(RowMiss::NotYetArrived { next: self.next });
        }
        Ok(self.slot(abs_row))
    }

    #[inline]
    fn slot(&self, abs_row: u64) -> usize {
        (abs_row % self.capacity as u64) as usize
    }
}

/// Fixed-capacity bar ring with an absolute row index.
///
/// See [`Window`] for when a row may be evicted.
#[derive(Debug)]
pub struct RingColumns {
    ts: Box<[i64]>,
    open: Box<[f64]>,
    high: Box<[f64]>,
    low: Box<[f64]>,
    close: Box<[f64]>,
    volume: Box<[f64]>,
    window: Window,
}

impl RingColumns {
    /// A ring holding `capacity` rows, starting at absolute row `start`
    /// (non-zero when joining a feed mid-stream).
    pub fn new(capacity: usize, start: u64) -> Self {
        let window = Window::new(capacity, start);
        Self {
            ts: vec![0; capacity].into(),
            open: vec![0.0; capacity].into(),
            high: vec![0.0; capacity].into(),
            low: vec![0.0; capacity].into(),
            close: vec![0.0; capacity].into(),
            volume: vec![0.0; capacity].into(),
            window,
        }
    }

    pub fn capacity(&self) -> usize {
        self.window.capacity
    }

    /// Rows currently held.
    pub fn len(&self) -> usize {
        self.window.len()
    }

    pub fn is_empty(&self) -> bool {
        self.window.next == self.window.first
    }

    /// Register a consumer. It starts at the oldest held row, so it has seen
    /// nothing yet and holds back everything currently in the ring.
    pub fn register_consumer(&mut self) -> ConsumerId {
        self.window.register()
    }

    /// Detach a consumer so it no longer holds rows back. A detached consumer
    /// that falls behind finds its rows [`RowMiss::Evicted`] and must register
    /// again to resume from the oldest held row.
    pub fn unregister_consumer(&mut self, id: ConsumerId) -> Result<(), RingError> {
        self.window.unregister(id)
    }

    /// The next row `id` has yet to read.
    pub fn watermark(&self, id: ConsumerId) -> Result<u64, RingError> {
        self.window.watermark(id)
    }

    /// Release every row below `to` for `id`.
    pub fn advance(&mut self, id: ConsumerId, to: u64) -> Result<(), RingError> {
        self.window.advance(id, to)
    }

    /// Append the bar with sequence number `seq`, which must be the next one.
    ///
    /// When the ring is full the oldest row is evicted to make room, but only
    /// if every consumer has released it; otherwise nothing changes and
    /// `Backpressure` is returned.
    pub fn push(&mut self, seq: u64, bar: KernelBar) -> Result<(), RingError> {
        self.window.check_seq(seq)?;
        if !self.is_empty() {
            let last = self.ts[self.window.slot(seq - 1)];
            if bar.timestamp < last {
                return Err(RingError::TimeRegression { row: seq, last, got: bar.timestamp });
            }
        }
        let i = self.window.admit()?;
        self.ts[i] = bar.timestamp;
        self.open[i] = bar.open;
        self.high[i] = bar.high;
        self.low[i] = bar.low;
        self.close[i] = bar.close;
        self.volume[i] = bar.volume;
        Ok(())
    }

    /// The bar at `abs_row`, or why it is not available.
    pub fn get(&self, abs_row: u64) -> Result<KernelBar, RowMiss> {
        let i = self.window.locate(abs_row)?;
        Ok(KernelBar {
            timestamp: self.ts[i],
            open: self.open[i],
            high: self.high[i],
            low: self.low[i],
            close: self.close[i],
            volume: self.volume[i],
        })
    }
}

impl ColumnSource for RingColumns {
    fn bar(&self, abs_row: u64) -> std::result::Result<KernelBar, RowMiss> {
        self.get(abs_row)
    }

    fn first_row(&self) -> u64 {
        self.window.first
    }

    fn end_row(&self) -> u64 {
        self.window.next
    }
}

impl HeapSize for RingColumns {
    fn heap_bytes(&self) -> usize {
        self.capacity() * (std::mem::size_of::<i64>() + 5 * std::mem::size_of::<f64>())
            + self.window.consumers.capacity() * std::mem::size_of::<Option<u64>>()
    }
}

/// One accepted feed row of an instrument's ticks, as pushed.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct TickRow {
    pub timestamp: i64,
    pub ltp: f64,
    pub bid: f64,
    pub ask: f64,
    pub buy_qty_delta: f64,
    pub sell_qty_delta: f64,
    pub ltq: f64,
    pub bid_qty: f64,
    pub ask_qty: f64,
    pub oi: f64,
}

/// Fixed-capacity ring of whole rows (ticks, depth snapshots) with an
/// absolute row index and the same consumer rules as [`RingColumns`].
///
/// It does not look at timestamps; the feed that owns it does.
#[derive(Debug)]
pub struct RowRing<T> {
    rows: Box<[T]>,
    window: Window,
}

impl<T: Copy> RowRing<T> {
    /// A ring of `capacity` rows; `blank` fills slots not yet written.
    pub fn new(capacity: usize, start: u64, blank: T) -> Self {
        let window = Window::new(capacity, start);
        Self { rows: vec![blank; capacity].into(), window }
    }

    pub fn capacity(&self) -> usize {
        self.window.capacity
    }

    pub fn len(&self) -> usize {
        self.window.len()
    }

    pub fn is_empty(&self) -> bool {
        self.window.next == self.window.first
    }

    pub fn first_row(&self) -> u64 {
        self.window.first
    }

    pub fn end_row(&self) -> u64 {
        self.window.next
    }

    pub fn register_consumer(&mut self) -> ConsumerId {
        self.window.register()
    }

    pub fn unregister_consumer(&mut self, id: ConsumerId) -> Result<(), RingError> {
        self.window.unregister(id)
    }

    pub fn watermark(&self, id: ConsumerId) -> Result<u64, RingError> {
        self.window.watermark(id)
    }

    pub fn advance(&mut self, id: ConsumerId, to: u64) -> Result<(), RingError> {
        self.window.advance(id, to)
    }

    /// Append the row with sequence number `seq`, which must be the next one.
    /// As [`RingColumns::push`], without the timestamp check.
    pub fn push(&mut self, seq: u64, row: T) -> Result<(), RingError> {
        self.window.check_seq(seq)?;
        let i = self.window.admit()?;
        self.rows[i] = row;
        Ok(())
    }

    pub fn get(&self, abs_row: u64) -> Result<T, RowMiss> {
        Ok(self.rows[self.window.locate(abs_row)?])
    }
}

impl<T> HeapSize for RowRing<T> {
    fn heap_bytes(&self) -> usize {
        self.rows.len() * std::mem::size_of::<T>()
            + self.window.consumers.capacity() * std::mem::size_of::<Option<u64>>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bar(ts: i64) -> KernelBar {
        KernelBar {
            timestamp: ts,
            open: ts as f64,
            high: ts as f64 + 1.0,
            low: ts as f64 - 1.0,
            close: ts as f64 + 0.5,
            volume: 10.0,
        }
    }

    fn fill(ring: &mut RingColumns, range: Range<u64>) {
        for seq in range {
            ring.push(seq, bar(seq as i64)).unwrap();
        }
    }

    #[test]
    fn unwatched_ring_evicts_oldest_when_full() {
        let mut ring = RingColumns::new(3, 0);
        fill(&mut ring, 0..5);
        assert_eq!((ring.first_row(), ring.end_row(), ring.len()), (2, 5, 3));
        assert_eq!(ring.get(1).unwrap_err(), RowMiss::Evicted { oldest: 2 });
        assert_eq!(ring.get(2).unwrap().timestamp, 2);
        assert_eq!(ring.get(4).unwrap().close, 4.5);
        assert_eq!(ring.get(5).unwrap_err(), RowMiss::NotYetArrived { next: 5 });
    }

    #[test]
    fn slowest_consumer_applies_backpressure_and_nothing_is_lost() {
        let mut ring = RingColumns::new(2, 0);
        let slow = ring.register_consumer();
        let fast = ring.register_consumer();
        fill(&mut ring, 0..2);
        ring.advance(fast, 2).unwrap();
        // Full, and `slow` has not released row 0.
        let err = ring.push(2, bar(2)).unwrap_err();
        assert_eq!(err, RingError::Backpressure { capacity: 2, oldest: 0, slowest: 0 });
        assert_eq!((ring.first_row(), ring.end_row()), (0, 2), "rejected push changed nothing");
        assert_eq!(ring.get(0).unwrap().timestamp, 0);
        // Releasing row 0 lets exactly one row go.
        ring.advance(slow, 1).unwrap();
        ring.push(2, bar(2)).unwrap();
        assert_eq!((ring.first_row(), ring.end_row()), (1, 3));
        assert_eq!(ring.get(0).unwrap_err(), RowMiss::Evicted { oldest: 1 });
        // Row 1 is still unreleased by `slow`.
        assert!(matches!(ring.push(3, bar(3)), Err(RingError::Backpressure { oldest: 1, .. })));
    }

    #[test]
    fn a_skipped_or_repeated_sequence_is_an_error() {
        let mut ring = RingColumns::new(4, 10);
        ring.push(10, bar(1)).unwrap();
        assert_eq!(ring.push(12, bar(2)), Err(RingError::Gap { expected: 11, got: 12 }));
        assert_eq!(ring.push(10, bar(2)), Err(RingError::Gap { expected: 11, got: 10 }));
        assert_eq!(ring.end_row(), 11, "rejected pushes changed nothing");
        assert!(matches!(ring.push(11, bar(0)), Err(RingError::TimeRegression { .. })));
        ring.push(11, bar(1)).unwrap();
    }

    #[test]
    fn detached_consumer_that_fell_behind_sees_evicted_and_recovers() {
        let mut ring = RingColumns::new(2, 0);
        let c = ring.register_consumer();
        fill(&mut ring, 0..2);
        assert!(ring.push(2, bar(2)).is_err());
        ring.unregister_consumer(c).unwrap();
        assert!(matches!(ring.watermark(c), Err(RingError::UnknownConsumer(_))));
        fill(&mut ring, 2..6);
        // The consumer's old position (row 0) is gone: evicted, not "not yet".
        assert_eq!(ring.get(0).unwrap_err(), RowMiss::Evicted { oldest: 4 });
        let c = ring.register_consumer();
        assert_eq!(ring.watermark(c).unwrap(), 4, "resumes from the oldest held row");
        assert_eq!(ring.get(ring.watermark(c).unwrap()).unwrap().timestamp, 4);
    }

    #[test]
    fn watermarks_only_move_forward_within_arrived_rows() {
        let mut ring = RingColumns::new(4, 0);
        let c = ring.register_consumer();
        fill(&mut ring, 0..3);
        ring.advance(c, 2).unwrap();
        assert!(matches!(ring.advance(c, 1), Err(RingError::BadWatermark { .. })));
        assert!(matches!(ring.advance(c, 4), Err(RingError::BadWatermark { .. })));
        assert_eq!(ring.watermark(c).unwrap(), 2);
    }

    #[test]
    fn column_source_is_uniform_over_full_and_bounded_storage() {
        fn visible_closes<S: ColumnSource>(s: &S, last: Option<u64>) -> Vec<f64> {
            s.visible(last).map(|r| s.bar(r).unwrap().close).collect()
        }
        let bars: Vec<KernelBar> = (0..5).map(bar).collect();
        let full = BarColumns::from_kernel_bars("A", &bars).unwrap();
        let mut ring = RingColumns::new(3, 0);
        fill(&mut ring, 0..5);

        assert_eq!(visible_closes(&full, Some(2)), vec![0.5, 1.5, 2.5]);
        assert_eq!(visible_closes(&full, None), Vec::<f64>::new());
        assert_eq!(visible_closes(&full, Some(99)).len(), 5);
        // The ring holds rows 2..5; a reader at row 3 sees what is held up to it.
        assert_eq!(visible_closes(&ring, Some(3)), vec![2.5, 3.5]);
        assert!(ring.visible(Some(0)).is_empty(), "an evicted watermark shows nothing");
        assert_eq!(ColumnSource::bar(&full, 5).unwrap_err(), RowMiss::NotYetArrived { next: 5 });
    }

    #[test]
    fn row_ring_follows_the_same_eviction_rules() {
        let row = |t: i64| TickRow { timestamp: t, ltp: t as f64, ..TickRow::default() };
        let mut ring = RowRing::new(2, 0, TickRow::default());
        let c = ring.register_consumer();
        ring.push(0, row(0)).unwrap();
        ring.push(1, row(1)).unwrap();
        assert!(matches!(ring.push(2, row(2)), Err(RingError::Backpressure { oldest: 0, .. })));
        assert_eq!((ring.first_row(), ring.end_row()), (0, 2));
        ring.advance(c, 1).unwrap();
        ring.push(2, row(2)).unwrap();
        assert_eq!(ring.get(0).unwrap_err(), RowMiss::Evicted { oldest: 1 });
        assert_eq!(ring.get(2).unwrap().ltp, 2.0);
        assert_eq!(ring.get(3).unwrap_err(), RowMiss::NotYetArrived { next: 3 });
        assert_eq!(ring.push(5, row(5)), Err(RingError::Gap { expected: 3, got: 5 }));
    }

    #[test]
    fn row_ring_heap_bytes_are_fixed_by_capacity() {
        let mut ring = RowRing::new(10, 0, TickRow::default());
        let fixed = ring.heap_bytes();
        for seq in 0..1000 {
            ring.push(seq, TickRow::default()).unwrap();
        }
        assert_eq!(ring.heap_bytes(), fixed);
        assert_eq!(fixed, 10 * std::mem::size_of::<TickRow>());
    }

    #[test]
    fn ring_heap_bytes_are_fixed_by_capacity() {
        let mut ring = RingColumns::new(100, 0);
        assert_eq!(ring.heap_bytes(), 100 * 48);
        fill(&mut ring, 0..250);
        assert_eq!(ring.heap_bytes(), 100 * 48);
    }
}
