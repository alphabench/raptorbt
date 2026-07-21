//! Deterministic k-way merge of pre-sorted event streams.
//!
//! Ordering contract, documented and tested: events are delivered by
//! `(timestamp, phase, stream, sequence)` ascending, where phase puts
//! trades before quotes before bars at equal timestamps (a bar closing at
//! `t` summarizes data ≤ `t`, so it must come last), and stream id then
//! per-stream sequence break remaining ties. Same inputs, same order —
//! always.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::data::events::MarketEvent;

/// Heap entry; reversed ordering turns `BinaryHeap` into a min-heap.
struct Entry {
    event: MarketEvent,
    stream_slot: usize,
    seq: usize,
}

impl Entry {
    fn key(&self) -> (i64, u8, u32, usize) {
        (self.event.timestamp(), self.event.phase(), self.event.stream, self.seq)
    }
}

impl PartialEq for Entry {
    fn eq(&self, other: &Self) -> bool {
        self.key() == other.key()
    }
}
impl Eq for Entry {}
impl PartialOrd for Entry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Entry {
    fn cmp(&self, other: &Self) -> Ordering {
        other.key().cmp(&self.key()) // reversed: min-heap
    }
}

/// K-way merge over owned, individually time-sorted event streams.
#[derive(Default)]
pub struct EventFeed {
    streams: Vec<std::vec::IntoIter<MarketEvent>>,
    heap: BinaryHeap<Entry>,
    cursors: Vec<usize>,
    primed: bool,
}

impl EventFeed {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register one time-sorted stream; returns its slot index.
    ///
    /// Debug builds assert monotone timestamps; release builds trust the
    /// caller (streams come from sorted arrays).
    pub fn add_stream(&mut self, events: Vec<MarketEvent>) -> usize {
        debug_assert!(
            events.windows(2).all(|w| w[0].timestamp() <= w[1].timestamp()),
            "stream must be time-sorted"
        );
        let slot = self.streams.len();
        self.streams.push(events.into_iter());
        self.cursors.push(0);
        slot
    }

    fn prime(&mut self) {
        for slot in 0..self.streams.len() {
            self.advance(slot);
        }
        self.primed = true;
    }

    fn advance(&mut self, slot: usize) {
        if let Some(event) = self.streams[slot].next() {
            let seq = self.cursors[slot];
            self.cursors[slot] += 1;
            self.heap.push(Entry { event, stream_slot: slot, seq });
        }
    }
}

impl Iterator for EventFeed {
    type Item = MarketEvent;

    fn next(&mut self) -> Option<MarketEvent> {
        if !self.primed {
            self.prime();
        }
        let entry = self.heap.pop()?;
        self.advance(entry.stream_slot);
        Some(entry.event)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::OhlcvBar;
    use crate::data::events::{EventPayload, QuoteTick, TradeTick};

    fn bar_event(stream: u32, ts: i64) -> MarketEvent {
        MarketEvent {
            instrument: 0,
            stream,
            payload: EventPayload::Bar(OhlcvBar {
                timestamp: ts,
                open: 1.0,
                high: 1.0,
                low: 1.0,
                close: 1.0,
                volume: 1.0,
            }),
        }
    }

    fn trade_event(stream: u32, ts: i64) -> MarketEvent {
        MarketEvent {
            instrument: 0,
            stream,
            payload: EventPayload::Trade(TradeTick { timestamp: ts, price: 1.0, size: 1.0 }),
        }
    }

    fn quote_event(stream: u32, ts: i64) -> MarketEvent {
        MarketEvent {
            instrument: 0,
            stream,
            payload: EventPayload::Quote(QuoteTick { timestamp: ts, bid: 1.0, ask: 1.0 }),
        }
    }

    #[test]
    fn merges_by_time_then_phase_then_stream() {
        let mut feed = EventFeed::new();
        feed.add_stream(vec![bar_event(0, 10), bar_event(0, 20)]);
        feed.add_stream(vec![trade_event(1, 10), trade_event(1, 15)]);
        feed.add_stream(vec![quote_event(2, 10)]);

        let order: Vec<(i64, u8, u32)> =
            feed.map(|e| (e.timestamp(), e.phase(), e.stream)).collect();
        // At ts=10: trade (phase 0) < quote (1) < bar (2).
        assert_eq!(
            order,
            vec![(10, 0, 1), (10, 1, 2), (10, 2, 0), (15, 0, 1), (20, 2, 0)]
        );
    }

    #[test]
    fn equal_everything_breaks_on_stream_then_sequence() {
        let mut feed = EventFeed::new();
        feed.add_stream(vec![trade_event(0, 5), trade_event(0, 5)]);
        feed.add_stream(vec![trade_event(1, 5)]);
        let order: Vec<u32> = feed.map(|e| e.stream).collect();
        assert_eq!(order, vec![0, 0, 1]);
    }

    #[test]
    fn deterministic_across_runs() {
        let build = || {
            let mut feed = EventFeed::new();
            feed.add_stream((0..50).map(|i| trade_event(0, i / 3)).collect());
            feed.add_stream((0..50).map(|i| bar_event(1, i / 2)).collect());
            feed.add_stream((0..50).map(|i| quote_event(2, i / 5)).collect());
            feed.map(|e| (e.timestamp(), e.phase(), e.stream)).collect::<Vec<_>>()
        };
        assert_eq!(build(), build());
    }
}
