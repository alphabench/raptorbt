//! Queue-position model for resting limit fills.
//!
//! The default fill model draws a Bernoulli per marketable limit
//! (`fill_prob_limit`), which has two defects: it has no memory, so an order
//! passed over four times is no likelier to fill on the fifth; and it
//! ignores the tape, so a one-lot print and a ten-thousand-lot print at the
//! same price count the same.
//!
//! This model fixes both with the only evidence a market-by-price feed
//! actually provides. It estimates the size queued ahead **once**, when the
//! order first rests, and then consumes that estimate with observed print
//! volume at the order's price. Progress is monotone: volume that trades
//! ahead of you never un-trades.
//!
//! What it deliberately does **not** claim: a real queue rank. Without an
//! order-by-order feed there is no way to know you are third of eleven, nor
//! to tell size that executed ahead of you from size that was cancelled.
//! Both are well-known limits of market-by-price data, and pretending
//! otherwise would be fiction dressed as precision.

use std::collections::HashMap;

use crate::core::types::{Direction, Price};
use crate::data::{BookSide, OrderBook};

/// One resting order's queue estimate.
#[derive(Debug, Clone, Copy)]
struct QueueState {
    /// Size estimated to sit ahead at this price, fixed when the order rested.
    ahead: f64,
    /// Print volume observed at this price since.
    traded: f64,
    /// The price the estimate belongs to; a modified order re-queues.
    price: Price,
}

/// Whether a print fills a resting limit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QueueVerdict {
    /// The level traded through: everything at this price and better was
    /// swept, so the order fills regardless of queue position.
    FilledThrough,
    /// Enough volume printed at this price to exhaust the queue ahead.
    FilledInQueue,
    /// Still queued; the order stays working.
    Resting,
    /// No book to reason from — the caller should fall back to its
    /// probabilistic model.
    Unknown,
}

/// Per-order queue estimates for one kernel.
#[derive(Debug, Default)]
pub struct QueueTracker {
    orders: HashMap<u64, QueueState>,
}

impl QueueTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Drop an order's estimate once it is no longer working.
    pub fn forget(&mut self, order_id: u64) {
        self.orders.remove(&order_id);
    }

    /// Estimated size still ahead of an order, for diagnostics.
    pub fn ahead_of(&self, order_id: u64) -> Option<f64> {
        self.orders.get(&order_id).map(|s| (s.ahead - s.traded).max(0.0))
    }

    /// Decide whether a print fills a resting limit, updating the estimate.
    ///
    /// `direction` is the order's side: a buy rests on the bid, a sell on
    /// the ask. `print_price`/`print_size` describe the trade that just
    /// happened.
    pub fn observe_print(
        &mut self,
        order_id: u64,
        limit_price: Price,
        direction: Direction,
        book: &OrderBook,
        print_price: Price,
        print_size: f64,
    ) -> QueueVerdict {
        // A print strictly better than the limit means the level cleared:
        // everything resting at this price and better was taken.
        let traded_through = match direction {
            Direction::Long => print_price < limit_price,
            Direction::Short => print_price > limit_price,
        };
        if traded_through {
            self.orders.remove(&order_id);
            return QueueVerdict::FilledThrough;
        }

        // Prints away from the limit tell us nothing about our queue.
        if print_price != limit_price {
            return QueueVerdict::Resting;
        }

        let side = match direction {
            Direction::Long => BookSide::Bid,
            Direction::Short => BookSide::Ask,
        };
        let state = match self.orders.get_mut(&order_id) {
            // A modified price is a new place in line.
            Some(state) if state.price == limit_price => state,
            _ => {
                let Some(ahead) = initial_queue(book, side, limit_price) else {
                    return QueueVerdict::Unknown;
                };
                self.orders
                    .insert(order_id, QueueState { ahead, traded: 0.0, price: limit_price });
                self.orders.get_mut(&order_id).expect("just inserted")
            }
        };

        state.traded += print_size;
        if state.traded > state.ahead {
            self.orders.remove(&order_id);
            QueueVerdict::FilledInQueue
        } else {
            QueueVerdict::Resting
        }
    }
}

/// Size to assume is queued ahead of a newly resting order.
///
/// A depth feed answers this exactly: join the back of what is displayed.
/// A quote-only book knows the price but not the size, so it cannot — the
/// caller falls back to its probabilistic model rather than inventing a
/// number. An order resting *better* than the touch is alone at its price
/// and has nothing ahead of it.
fn initial_queue(book: &OrderBook, side: BookSide, limit_price: Price) -> Option<f64> {
    if let Some(size) = book.size_at(side, limit_price) {
        // NaN means the level is visible but unquantified (L1 quote).
        return if size.is_nan() { None } else { Some(size) };
    }
    let touch = match side {
        BookSide::Bid => book.best_bid(),
        BookSide::Ask => book.best_ask(),
    }?;
    let improves = match side {
        BookSide::Bid => limit_price > touch,
        BookSide::Ask => limit_price < touch,
    };
    // Better than the touch: a new price level, nothing queued ahead.
    // Worse than the touch and outside the visible window: unknown.
    improves.then_some(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{BookLevel, DepthTick};

    fn depth_book(bid: (Price, f64), ask: (Price, f64)) -> OrderBook {
        let mut book = OrderBook::new();
        book.apply_depth(&DepthTick::from_levels(
            0,
            &[BookLevel { price: bid.0, size: bid.1 }],
            &[BookLevel { price: ask.0, size: ask.1 }],
        ));
        book
    }

    #[test]
    fn a_print_through_the_limit_fills_regardless_of_queue() {
        let mut tracker = QueueTracker::new();
        let book = depth_book((99.0, 10_000.0), (101.0, 500.0));
        // Huge queue ahead, but the print cleared the level entirely.
        let verdict =
            tracker.observe_print(1, 99.0, Direction::Long, &book, 98.5, 1.0);
        assert_eq!(verdict, QueueVerdict::FilledThrough);
    }

    #[test]
    fn prints_at_the_limit_accumulate_until_the_queue_clears() {
        let mut tracker = QueueTracker::new();
        let book = depth_book((99.0, 300.0), (101.0, 500.0));

        // 300 displayed ahead: 100 + 150 is not enough.
        assert_eq!(
            tracker.observe_print(1, 99.0, Direction::Long, &book, 99.0, 100.0),
            QueueVerdict::Resting
        );
        assert_eq!(
            tracker.observe_print(1, 99.0, Direction::Long, &book, 99.0, 150.0),
            QueueVerdict::Resting
        );
        assert_eq!(tracker.ahead_of(1), Some(50.0));
        // The third print exhausts it.
        assert_eq!(
            tracker.observe_print(1, 99.0, Direction::Long, &book, 99.0, 60.0),
            QueueVerdict::FilledInQueue
        );
        assert_eq!(tracker.ahead_of(1), None, "filled orders are forgotten");
    }

    #[test]
    fn progress_is_monotone_across_prints() {
        // The scalar model this replaces has no memory: an order passed over
        // repeatedly is no likelier to fill. Here it strictly improves.
        let mut tracker = QueueTracker::new();
        let book = depth_book((99.0, 100.0), (101.0, 100.0));
        let mut previous = f64::INFINITY;
        for _ in 0..4 {
            tracker.observe_print(1, 99.0, Direction::Long, &book, 99.0, 20.0);
            let ahead = tracker.ahead_of(1).unwrap_or(0.0);
            assert!(ahead < previous, "queue must shrink: {ahead} !< {previous}");
            previous = ahead;
        }
    }

    #[test]
    fn prints_away_from_the_limit_do_not_advance_the_queue() {
        let mut tracker = QueueTracker::new();
        let book = depth_book((99.0, 300.0), (101.0, 500.0));
        tracker.observe_print(1, 99.0, Direction::Long, &book, 99.0, 100.0);
        // A print at 100.0 is neither at nor through a 99.0 bid.
        assert_eq!(
            tracker.observe_print(1, 99.0, Direction::Long, &book, 100.0, 9_999.0),
            QueueVerdict::Resting
        );
        assert_eq!(tracker.ahead_of(1), Some(200.0), "unchanged");
    }

    #[test]
    fn a_quote_only_book_cannot_estimate_the_queue() {
        let mut tracker = QueueTracker::new();
        let mut book = OrderBook::new();
        book.apply_quote(0, 99.0, 101.0);
        // The price is visible but its size is not; the caller must fall
        // back rather than guess.
        assert_eq!(
            tracker.observe_print(1, 99.0, Direction::Long, &book, 99.0, 100.0),
            QueueVerdict::Unknown
        );
    }

    #[test]
    fn an_order_better_than_the_touch_has_nothing_ahead() {
        let mut tracker = QueueTracker::new();
        let book = depth_book((99.0, 300.0), (101.0, 500.0));
        // Resting at 99.5 improves the bid: a new level, alone in it.
        assert_eq!(
            tracker.observe_print(1, 99.5, Direction::Long, &book, 99.5, 1.0),
            QueueVerdict::FilledInQueue
        );
    }

    #[test]
    fn a_short_rests_on_the_ask() {
        let mut tracker = QueueTracker::new();
        let book = depth_book((99.0, 300.0), (101.0, 400.0));
        assert_eq!(
            tracker.observe_print(1, 101.0, Direction::Short, &book, 101.0, 100.0),
            QueueVerdict::Resting
        );
        // Upward through a resting sell clears it.
        assert_eq!(
            tracker.observe_print(1, 101.0, Direction::Short, &book, 101.5, 1.0),
            QueueVerdict::FilledThrough
        );
    }
}
