//! Market events: the common currency of the multi-stream feed.

use crate::core::types::{OhlcvBar, Price, TickData, Timestamp};

/// Best bid/ask observation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuoteTick {
    pub timestamp: Timestamp,
    pub bid: Price,
    pub ask: Price,
    /// Displayed size at the bid, or `NaN` when the feed carried none.
    pub bid_size: f64,
    /// Displayed size at the ask, or `NaN` when the feed carried none.
    pub ask_size: f64,
}

impl QuoteTick {
    /// A quote whose sizes are unknown (the pre-L1-size shape).
    pub fn without_sizes(timestamp: Timestamp, bid: Price, ask: Price) -> Self {
        Self { timestamp, bid, ask, bid_size: f64::NAN, ask_size: f64::NAN }
    }
}

/// One trade print.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TradeTick {
    pub timestamp: Timestamp,
    pub price: Price,
    /// Total traded size, unsigned.
    pub size: f64,
    /// Buy-initiated minus sell-initiated size. `0.0` means the split is
    /// unknown, not that flow was balanced — consumers that need a
    /// direction fall back to the tick rule.
    pub signed_size: f64,
    /// Open interest published with the print (0.0 when the feed carried
    /// none — equities have no open interest).
    pub oi: f64,
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

/// A depth snapshot's slot in the owning session's store.
///
/// Depth rides the feed as a handle rather than inline: a five-level book is
/// ~176 bytes, and putting that in the payload would make every event —
/// bars and quotes included — four times fatter for no benefit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DepthRef {
    pub slot: u32,
    pub timestamp: Timestamp,
}

/// The event body.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EventPayload {
    Bar(OhlcvBar),
    Quote(QuoteTick),
    Trade(TradeTick),
    Depth(DepthRef),
}

impl MarketEvent {
    /// Event timestamp (ns).
    #[inline]
    pub fn timestamp(&self) -> Timestamp {
        match &self.payload {
            EventPayload::Bar(b) => b.timestamp,
            EventPayload::Quote(q) => q.timestamp,
            EventPayload::Trade(t) => t.timestamp,
            EventPayload::Depth(d) => d.timestamp,
        }
    }

    /// Merge priority at equal timestamps: intra-bar data precedes the bar
    /// that summarizes it — trades, then book updates, then bars. A bar
    /// closing at `t` therefore "sees" every tick ≤ `t`.
    ///
    /// A print precedes the book state observed alongside it, so a handler
    /// reading the book during a trade sees what stood *before* that print
    /// rather than the book the print itself moved.
    #[inline]
    pub fn phase(&self) -> u8 {
        match &self.payload {
            EventPayload::Trade(_) => 0,
            EventPayload::Quote(_) => 1,
            EventPayload::Depth(_) => 2,
            EventPayload::Bar(_) => 3,
        }
    }
}

/// The trade print one tick row carries, or `None` without a last-traded
/// price (zero marks missing data).
///
/// `size` is the exchange's last traded quantity when the feed carried one
/// (`ltq > 0`), else the flow-delta proxy (every existing bar's volume
/// depends on it). The signed split rides alongside.
#[inline]
pub fn trade_from_row(
    timestamp: Timestamp,
    ltp: Price,
    buy_qty_delta: f64,
    sell_qty_delta: f64,
    ltq: f64,
    oi: f64,
) -> Option<TradeTick> {
    if ltp > 0.0 {
        let size = if ltq > 0.0 { ltq } else { buy_qty_delta.abs() + sell_qty_delta.abs() };
        let signed_size = buy_qty_delta.abs() - sell_qty_delta.abs();
        Some(TradeTick { timestamp, price: ltp, size, signed_size, oi })
    } else {
        None
    }
}

/// The quote one tick row carries, or `None` unless both sides of the book
/// are present.
#[inline]
pub fn quote_from_row(
    timestamp: Timestamp,
    bid: Price,
    ask: Price,
    bid_qty: f64,
    ask_qty: f64,
) -> Option<QuoteTick> {
    if bid > 0.0 && ask > 0.0 {
        Some(QuoteTick {
            timestamp,
            bid,
            ask,
            bid_size: TickData::displayed(bid_qty),
            ask_size: TickData::displayed(ask_qty),
        })
    } else {
        None
    }
}

/// Split raw tick arrays into trade and quote event streams.
///
/// Every tick with a last-traded price becomes a [`TradeTick`] (size = buy
/// plus sell quantity delta, `0.0` when unavailable); ticks carrying a
/// two-sided book become [`QuoteTick`]s as well. Zero prices mark missing
/// data and are skipped.
pub fn tick_data_to_events(
    ticks: &TickData,
    instrument: u32,
    trade_stream: u32,
    quote_stream: u32,
) -> Vec<MarketEvent> {
    let mut events = Vec::with_capacity(ticks.len() * 2);
    for i in 0..ticks.len() {
        let ts = ticks.timestamps[i];
        let ltq = ticks.ltq.get(i).copied().unwrap_or(0.0);
        if let Some(trade) = trade_from_row(
            ts,
            ticks.ltp[i],
            ticks.buy_qty_delta[i],
            ticks.sell_qty_delta[i],
            ltq,
            ticks.oi[i],
        ) {
            events.push(MarketEvent {
                instrument,
                stream: trade_stream,
                payload: EventPayload::Trade(trade),
            });
        }
        if let Some(quote) =
            quote_from_row(ts, ticks.bid[i], ticks.ask[i], ticks.bid_qty[i], ticks.ask_qty[i])
        {
            events.push(MarketEvent {
                instrument,
                stream: quote_stream,
                payload: EventPayload::Quote(quote),
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
            ltq: vec![0.0, 0.0, 0.0],
            bid_qty: vec![0.0, 0.0, 0.0],
            ask_qty: vec![0.0, 0.0, 0.0],
        };
        let events = tick_data_to_events(&ticks, 0, 1, 2);
        let trades: Vec<_> =
            events.iter().filter(|e| matches!(e.payload, EventPayload::Trade(_))).collect();
        let quotes: Vec<_> =
            events.iter().filter(|e| matches!(e.payload, EventPayload::Quote(_))).collect();
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

#[cfg(test)]
mod size_tests {
    use super::*;

    #[test]
    fn ltq_is_the_print_size_when_present_and_sizes_reach_the_quote() {
        let ticks = TickData {
            timestamps: vec![1, 2],
            ltp: vec![100.0, 100.0],
            bid: vec![99.0, 99.0],
            ask: vec![101.0, 101.0],
            buy_qty_delta: vec![5.0, 5.0],
            sell_qty_delta: vec![2.0, 2.0],
            oi: vec![1_500.0, 0.0],
            ltq: vec![40.0, 0.0],
            bid_qty: vec![300.0, 0.0],
            ask_qty: vec![0.0, 0.0],
        };
        let events = tick_data_to_events(&ticks, 0, 1, 2);
        let trades: Vec<TradeTick> = events
            .iter()
            .filter_map(|e| match e.payload {
                EventPayload::Trade(t) => Some(t),
                _ => None,
            })
            .collect();
        // Row 0 carried ltq: that is the size. Row 1 did not: the proxy.
        assert_eq!(trades[0].size, 40.0);
        assert_eq!(trades[0].oi, 1_500.0);
        assert_eq!(trades[1].size, 7.0);
        let quotes: Vec<QuoteTick> = events
            .iter()
            .filter_map(|e| match e.payload {
                EventPayload::Quote(q) => Some(q),
                _ => None,
            })
            .collect();
        assert_eq!(quotes[0].bid_size, 300.0);
        assert!(quotes[0].ask_size.is_nan(), "an absent size stays unknown, never 0");
        assert!(quotes[1].bid_size.is_nan());
    }
}
