//! Order book state: L1 from quotes, L2 from depth snapshots.
//!
//! [`OrderBook`] is the stateful accumulator a kernel keeps; [`DepthTick`] is
//! the wire form that rides the event feed. Both are `Copy` and fixed-size —
//! the feed, the schedule, and the session's cursor all depend on that, so
//! depth is capped at [`BOOK_DEPTH`] levels rather than heap-allocated.
//!
//! A level's size is deliberately three-valued. A real quantity means that
//! much is displayed; `NaN` means the price is known but its size is not
//! (an L1 quote says nothing about depth); and a level outside the visible
//! window reports `None` rather than zero, because "we cannot see it" and
//! "there is nothing there" lead to opposite fill decisions.

use crate::core::types::{Price, Timestamp};

/// Levels retained per side. Indian venues publish exactly five, which is
/// what the tick feed this engine targets carries.
pub const BOOK_DEPTH: usize = 5;

/// One price level.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct BookLevel {
    pub price: Price,
    /// Displayed size, or `NaN` when the price is known but the size is not.
    pub size: f64,
}

impl BookLevel {
    pub const EMPTY: Self = Self { price: 0.0, size: 0.0 };

    /// A top-of-book level: sized when the feed displayed a size, else
    /// unquantified.
    pub fn l1(price: Price, size: f64) -> Self {
        if size.is_finite() && size > 0.0 {
            Self { price, size }
        } else {
            Self::unquantified(price)
        }
    }

    /// A level whose price is known but whose size is not (an L1 quote).
    pub fn unquantified(price: Price) -> Self {
        Self { price, size: f64::NAN }
    }
}

/// Which side of the book.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BookSide {
    Bid,
    Ask,
}

/// A depth snapshot, as it rides the event feed.
///
/// Snapshot semantics, not deltas: each tick replaces the visible book.
/// Books deeper than [`BOOK_DEPTH`] are truncated, and a full window means
/// "at least this deep", never "exactly this deep".
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DepthTick {
    pub timestamp: Timestamp,
    pub bids: [BookLevel; BOOK_DEPTH],
    pub asks: [BookLevel; BOOK_DEPTH],
    pub bid_len: u8,
    pub ask_len: u8,
}

impl DepthTick {
    /// Build from level slices, keeping the best [`BOOK_DEPTH`] of each.
    ///
    /// Callers supply levels best-first: bids descending, asks ascending.
    pub fn from_levels(timestamp: Timestamp, bids: &[BookLevel], asks: &[BookLevel]) -> Self {
        let mut tick = Self {
            timestamp,
            bids: [BookLevel::EMPTY; BOOK_DEPTH],
            asks: [BookLevel::EMPTY; BOOK_DEPTH],
            bid_len: 0,
            ask_len: 0,
        };
        for (i, level) in bids.iter().take(BOOK_DEPTH).enumerate() {
            tick.bids[i] = *level;
            tick.bid_len = (i + 1) as u8;
        }
        for (i, level) in asks.iter().take(BOOK_DEPTH).enumerate() {
            tick.asks[i] = *level;
            tick.ask_len = (i + 1) as u8;
        }
        tick
    }

    /// Whether either side filled the visible window, so the real book may
    /// run deeper than what is recorded.
    pub fn is_truncated(&self) -> bool {
        self.bid_len as usize == BOOK_DEPTH || self.ask_len as usize == BOOK_DEPTH
    }
}

/// The book a kernel keeps, updated from quotes (L1) or depth (L2).
#[derive(Debug, Clone, Copy, Default)]
pub struct OrderBook {
    bids: [BookLevel; BOOK_DEPTH],
    asks: [BookLevel; BOOK_DEPTH],
    bid_len: u8,
    ask_len: u8,
    last_update_ns: Timestamp,
}

impl OrderBook {
    pub fn new() -> Self {
        Self {
            bids: [BookLevel::EMPTY; BOOK_DEPTH],
            asks: [BookLevel::EMPTY; BOOK_DEPTH],
            bid_len: 0,
            ask_len: 0,
            last_update_ns: 0,
        }
    }

    /// Apply a top-of-book quote.
    ///
    /// Sets each side's best price with its displayed size when the feed
    /// carried one (a finite positive `bid_size`/`ask_size`), else marks
    /// it unquantified. Levels below the touch are dropped, since a new L1
    /// observation says nothing about whether they still stand.
    pub fn apply_quote(
        &mut self,
        timestamp: Timestamp,
        bid: Price,
        ask: Price,
        bid_size: f64,
        ask_size: f64,
    ) {
        self.bids = [BookLevel::EMPTY; BOOK_DEPTH];
        self.asks = [BookLevel::EMPTY; BOOK_DEPTH];
        self.bid_len = 0;
        self.ask_len = 0;
        if bid > 0.0 {
            self.bids[0] = BookLevel::l1(bid, bid_size);
            self.bid_len = 1;
        }
        if ask > 0.0 {
            self.asks[0] = BookLevel::l1(ask, ask_size);
            self.ask_len = 1;
        }
        self.last_update_ns = timestamp;
    }

    /// Replace the visible book from a depth snapshot.
    pub fn apply_depth(&mut self, depth: &DepthTick) {
        self.bids = depth.bids;
        self.asks = depth.asks;
        self.bid_len = depth.bid_len;
        self.ask_len = depth.ask_len;
        self.last_update_ns = depth.timestamp;
    }

    /// Best bid price, if the book has one.
    pub fn best_bid(&self) -> Option<Price> {
        (self.bid_len > 0).then(|| self.bids[0].price)
    }

    /// Best ask price, if the book has one.
    pub fn best_ask(&self) -> Option<Price> {
        (self.ask_len > 0).then(|| self.asks[0].price)
    }

    /// Levels on one side, best first.
    pub fn levels(&self, side: BookSide) -> &[BookLevel] {
        match side {
            BookSide::Bid => &self.bids[..self.bid_len as usize],
            BookSide::Ask => &self.asks[..self.ask_len as usize],
        }
    }

    /// Ask minus bid, when both sides are known.
    pub fn spread(&self) -> Option<Price> {
        Some(self.best_ask()? - self.best_bid()?)
    }

    /// Midpoint of the touch, when both sides are known.
    pub fn mid(&self) -> Option<Price> {
        Some((self.best_ask()? + self.best_bid()?) / 2.0)
    }

    /// Size-weighted touch price, when both sides carry sizes.
    ///
    /// `None` on a quote-only book: sizes are unquantified there, and a
    /// midpoint dressed up as a microprice would be a false precision.
    pub fn microprice(&self) -> Option<Price> {
        let bid = *self.levels(BookSide::Bid).first()?;
        let ask = *self.levels(BookSide::Ask).first()?;
        if bid.size.is_nan() || ask.size.is_nan() {
            return None;
        }
        let total = bid.size + ask.size;
        if total <= 0.0 {
            return None;
        }
        Some((bid.price * ask.size + ask.price * bid.size) / total)
    }

    /// Displayed size at an exact price.
    ///
    /// `None` when the price is outside the visible window — unknown, which
    /// a caller must not read as "no liquidity".
    pub fn size_at(&self, side: BookSide, price: Price) -> Option<f64> {
        self.levels(side).iter().find(|level| level.price == price).map(|level| level.size)
    }

    /// Whether any level has been recorded.
    pub fn is_empty(&self) -> bool {
        self.bid_len == 0 && self.ask_len == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn level(price: Price, size: f64) -> BookLevel {
        BookLevel { price, size }
    }

    #[test]
    fn quote_sets_the_touch_and_leaves_size_unquantified() {
        let mut book = OrderBook::new();
        book.apply_quote(10, 99.0, 101.0, f64::NAN, f64::NAN);
        assert_eq!(book.best_bid(), Some(99.0));
        assert_eq!(book.best_ask(), Some(101.0));
        assert_eq!(book.spread(), Some(2.0));
        assert_eq!(book.mid(), Some(100.0));
        // A quote carries no depth: the size is unknown, not zero.
        assert!(book.size_at(BookSide::Bid, 99.0).expect("visible").is_nan());
        assert_eq!(book.microprice(), None, "no sizes means no microprice");
    }

    #[test]
    fn one_sided_quote_has_no_spread_or_mid() {
        let mut book = OrderBook::new();
        book.apply_quote(10, 99.0, 0.0, f64::NAN, f64::NAN);
        assert_eq!(book.best_bid(), Some(99.0));
        assert_eq!(book.best_ask(), None);
        assert_eq!(book.spread(), None);
        assert_eq!(book.mid(), None);
    }

    #[test]
    fn depth_replaces_the_visible_book() {
        let mut book = OrderBook::new();
        book.apply_quote(10, 99.0, 101.0, f64::NAN, f64::NAN);
        let depth = DepthTick::from_levels(
            20,
            &[level(99.0, 500.0), level(98.0, 300.0)],
            &[level(101.0, 400.0), level(102.0, 200.0)],
        );
        book.apply_depth(&depth);

        assert_eq!(book.levels(BookSide::Bid).len(), 2);
        assert_eq!(book.size_at(BookSide::Bid, 99.0), Some(500.0));
        assert_eq!(book.size_at(BookSide::Ask, 102.0), Some(200.0));
        // Now that sizes are known, the microprice resolves.
        // Weighted by the opposite side's size: the heavier bid (500 vs
        // 400) pulls the fair price up toward the ask.
        let micro = book.microprice().expect("two-sided with sizes");
        assert!((micro - 100.111_111_1).abs() < 1e-6, "got {micro}");
    }

    #[test]
    fn size_outside_the_window_is_unknown_not_zero() {
        let mut book = OrderBook::new();
        book.apply_depth(&DepthTick::from_levels(
            10,
            &[level(99.0, 500.0)],
            &[level(101.0, 400.0)],
        ));
        // 97.0 is not visible. Reporting 0.0 would claim there is no
        // liquidity there, which the snapshot does not establish.
        assert_eq!(book.size_at(BookSide::Bid, 97.0), None);
    }

    #[test]
    fn books_deeper_than_the_window_truncate_and_flag() {
        let deep: Vec<BookLevel> =
            (0..8).map(|i| level(100.0 - i as f64, 10.0 * (i + 1) as f64)).collect();
        let tick = DepthTick::from_levels(10, &deep, &deep);
        assert_eq!(tick.bid_len as usize, BOOK_DEPTH);
        assert!(tick.is_truncated());

        let mut book = OrderBook::new();
        book.apply_depth(&tick);
        assert_eq!(book.levels(BookSide::Bid).len(), BOOK_DEPTH);
        // The sixth level was dropped, so it reads as unknown.
        assert_eq!(book.size_at(BookSide::Bid, 95.0), None);
    }

    #[test]
    fn a_quote_after_depth_drops_the_lower_levels() {
        let mut book = OrderBook::new();
        book.apply_depth(&DepthTick::from_levels(
            10,
            &[level(99.0, 500.0), level(98.0, 300.0)],
            &[level(101.0, 400.0)],
        ));
        // A fresh L1 observation says nothing about whether 98.0 still stands.
        book.apply_quote(20, 99.5, 100.5, f64::NAN, f64::NAN);
        assert_eq!(book.levels(BookSide::Bid).len(), 1);
        assert_eq!(book.size_at(BookSide::Bid, 98.0), None);
    }
}
