//! Market events: the common currency of the multi-stream feed.

use crate::core::types::{OhlcvBar, Price, TickData, Timestamp};

/// Best bid/ask observation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuoteTick {
    pub timestamp: Timestamp,
    pub bid: Price,
    pub ask: Price,
}

/// One trade print.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TradeTick {
    pub timestamp: Timestamp,
    pub price: Price,
    pub size: f64,
}

/// One record of one stream.
///
/// `stream` identifies the source series (assigned by the feed at
/// registration); `instrument` the symbol slot. Both are small integers so
/// events stay `Copy`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MarketEvent {
    pub instrument: u32,
    pub stream: u32,
    pub payload: EventPayload,
}

/// The event body.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EventPayload {
    Bar(OhlcvBar),
    Quote(QuoteTick),
    Trade(TradeTick),
}

impl MarketEvent {
    /// Event timestamp (ns).
    #[inline]
    pub fn timestamp(&self) -> Timestamp {
        match &self.payload {
            EventPayload::Bar(b) => b.timestamp,
            EventPayload::Quote(q) => q.timestamp,
            EventPayload::Trade(t) => t.timestamp,
        }
    }

    /// Merge priority at equal timestamps: intra-bar data precedes the bar
    /// that summarizes it — trades, then quotes, then bars. A bar closing
    /// at `t` therefore "sees" every tick ≤ `t`.
    #[inline]
    pub fn phase(&self) -> u8 {
        match &self.payload {
            EventPayload::Trade(_) => 0,
            EventPayload::Quote(_) => 1,
            EventPayload::Bar(_) => 2,
        }
    }
}

/// Split raw tick arrays into trade and quote event streams.
///
/// Every tick with a last-traded price becomes a [`TradeTick`] (size = buy
/// plus sell quantity delta, `0.0` when unavailable); ticks carrying a
/// two-sided book become [`QuoteTick`]s as well. Zero prices mark missing
/// data and are skipped.
pub fn tick_data_to_events(ticks: &TickData, instrument: u32, trade_stream: u32, quote_stream: u32) -> Vec<MarketEvent> {
    let mut events = Vec::with_capacity(ticks.len() * 2);
    for i in 0..ticks.len() {
        let ts = ticks.timestamps[i];
        let ltp = ticks.ltp[i];
        if ltp > 0.0 {
            let size = ticks.buy_qty_delta[i].abs() + ticks.sell_qty_delta[i].abs();
            events.push(MarketEvent {
                instrument,
                stream: trade_stream,
                payload: EventPayload::Trade(TradeTick { timestamp: ts, price: ltp, size }),
            });
        }
        let (bid, ask) = (ticks.bid[i], ticks.ask[i]);
        if bid > 0.0 && ask > 0.0 {
            events.push(MarketEvent {
                instrument,
                stream: quote_stream,
                payload: EventPayload::Quote(QuoteTick { timestamp: ts, bid, ask }),
            });
        }
    }
    events
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tick_conversion_splits_streams_and_skips_gaps() {
        let ticks = TickData {
            timestamps: vec![1, 2, 3],
            ltp: vec![100.0, 0.0, 101.0],
            bid: vec![99.5, 100.0, 0.0],
            ask: vec![100.5, 100.5, 101.5],
            buy_qty_delta: vec![5.0, 0.0, 3.0],
            sell_qty_delta: vec![2.0, 0.0, 0.0],
            oi: vec![0.0, 0.0, 0.0],
        };
        let events = tick_data_to_events(&ticks, 0, 1, 2);
        let trades: Vec<_> = events
            .iter()
            .filter(|e| matches!(e.payload, EventPayload::Trade(_)))
            .collect();
        let quotes: Vec<_> = events
            .iter()
            .filter(|e| matches!(e.payload, EventPayload::Quote(_)))
            .collect();
        // ltp=0 at ts=2 skipped; ask-only book at ts=3 skipped.
        assert_eq!(trades.len(), 2);
        assert_eq!(quotes.len(), 2);
        match trades[0].payload {
            EventPayload::Trade(t) => {
                assert_eq!(t.size, 7.0);
                assert_eq!(t.price, 100.0);
            }
            _ => unreachable!(),
        }
    }
}
