//! Renko bricks: bars driven by price movement alone.
//!
//! A brick is emitted every time price travels a full brick height from the
//! last brick's close. Time and volume do not close a brick, so a quiet
//! hour produces nothing and a fast move produces several at once.
//!
//! This is the symmetric one-brick grid (what charting packages call
//! "traditional" Renko with a fixed box size): a reversal needs one brick
//! against the trend, not two. Bricks carry no wicks — `high` and `low` are
//! the brick's own bounds — because a brick is a price interval, not a
//! summary of a time window.

use std::collections::VecDeque;

use crate::core::types::{OhlcvBar, Price};
use crate::data::aggregate::{BarBuilder, SourceRecord};

/// Bricks one record may complete before the builder gives up and resyncs.
///
/// A brick size far below the instrument's scale (`0.0001` on an index)
/// would otherwise turn one print into millions of bars.
const MAX_BRICKS_PER_RECORD: usize = 10_000;

/// Fixed-height price bricks.
#[derive(Debug)]
pub struct RenkoBarBuilder {
    brick: Price,
    /// Close of the last emitted brick; the grid line price must cross.
    anchor: Option<Price>,
    /// Bricks completed by one record, drained one per call.
    pending: VecDeque<OhlcvBar>,
}

impl RenkoBarBuilder {
    pub fn new(brick: Price) -> Self {
        debug_assert!(brick > 0.0, "brick height must be positive");
        Self { brick, anchor: None, pending: VecDeque::new() }
    }

    /// Snap a price down to the brick grid, so bricks land on round
    /// multiples rather than wherever the first record happened to fall.
    fn snap(&self, price: Price) -> Price {
        (price / self.brick).floor() * self.brick
    }
}

impl BarBuilder for RenkoBarBuilder {
    fn push(&mut self, rec: &SourceRecord) -> Option<OhlcvBar> {
        let anchor = match self.anchor {
            Some(anchor) => anchor,
            None => {
                // The first record only establishes the grid.
                self.anchor = Some(self.snap(rec.close));
                return None;
            }
        };

        let close = rec.close;
        let steps = ((close - anchor).abs() / self.brick).floor() as usize;
        if steps == 0 {
            return None;
        }
        let up = close > anchor;
        let bricks = steps.min(MAX_BRICKS_PER_RECORD);

        let mut level = anchor;
        for i in 0..bricks {
            let next = if up { level + self.brick } else { level - self.brick };
            // Volume rides entirely on the brick that the record completed
            // last: splitting it across bricks would invent a distribution
            // the data does not have, and dropping it would lose it.
            let volume = if i + 1 == bricks { rec.volume } else { 0.0 };
            self.pending.push_back(OhlcvBar {
                timestamp: rec.timestamp,
                open: level,
                high: level.max(next),
                low: level.min(next),
                close: next,
                volume,
            });
            level = next;
        }
        // Resync past a capped burst so the grid tracks price rather than
        // trailing it forever.
        self.anchor = Some(if bricks == steps { level } else { self.snap(close) });
        self.pending.pop_front()
    }

    /// Renko has no partial bar: a brick that never completed is not a brick.
    fn flush(&mut self) -> Option<OhlcvBar> {
        self.pending.pop_front()
    }

    fn next_pending(&mut self) -> Option<OhlcvBar> {
        self.pending.pop_front()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(ts: i64, price: Price, volume: f64) -> SourceRecord {
        SourceRecord::trade(ts, price, volume)
    }

    fn drain(builder: &mut RenkoBarBuilder, rec: &SourceRecord) -> Vec<OhlcvBar> {
        let mut out = Vec::new();
        if let Some(bar) = builder.push(rec) {
            out.push(bar);
        }
        while let Some(bar) = builder.next_pending() {
            out.push(bar);
        }
        out
    }

    #[test]
    fn the_first_record_only_sets_the_grid() {
        let mut builder = RenkoBarBuilder::new(1.0);
        assert!(builder.push(&rec(0, 100.0, 5.0)).is_none());
    }

    #[test]
    fn one_brick_per_full_move() {
        let mut builder = RenkoBarBuilder::new(1.0);
        builder.push(&rec(0, 100.0, 1.0));
        // Half a brick is not a brick.
        assert!(drain(&mut builder, &rec(1, 100.5, 1.0)).is_empty());
        let bars = drain(&mut builder, &rec(2, 101.0, 1.0));
        assert_eq!(bars.len(), 1);
        assert_eq!(bars[0].open, 100.0);
        assert_eq!(bars[0].close, 101.0);
    }

    #[test]
    fn a_burst_emits_every_brick_it_crossed() {
        let mut builder = RenkoBarBuilder::new(1.0);
        builder.push(&rec(0, 100.0, 1.0));
        // A jump of three bricks must not collapse into one bar.
        let bars = drain(&mut builder, &rec(1, 103.0, 9.0));
        assert_eq!(bars.len(), 3);
        assert_eq!(
            bars.iter().map(|b| (b.open, b.close)).collect::<Vec<_>>(),
            vec![(100.0, 101.0), (101.0, 102.0), (102.0, 103.0)]
        );
        // Volume lands on the completing brick, and is neither split nor lost.
        assert_eq!(bars.iter().map(|b| b.volume).sum::<f64>(), 9.0);
        assert_eq!(bars[2].volume, 9.0);
    }

    #[test]
    fn bricks_have_no_wicks() {
        let mut builder = RenkoBarBuilder::new(1.0);
        builder.push(&rec(0, 100.0, 1.0));
        for bar in drain(&mut builder, &rec(1, 102.0, 1.0)) {
            assert_eq!(bar.high, bar.open.max(bar.close));
            assert_eq!(bar.low, bar.open.min(bar.close));
        }
    }

    #[test]
    fn a_reversal_emits_down_bricks() {
        let mut builder = RenkoBarBuilder::new(1.0);
        builder.push(&rec(0, 100.0, 1.0));
        drain(&mut builder, &rec(1, 102.0, 1.0));
        let bars = drain(&mut builder, &rec(2, 100.0, 1.0));
        assert_eq!(bars.len(), 2);
        assert!(bars.iter().all(|b| b.close < b.open), "down bricks: {bars:?}");
        assert_eq!(bars[1].close, 100.0);
    }

    #[test]
    fn time_and_volume_never_close_a_brick() {
        let mut builder = RenkoBarBuilder::new(1.0);
        builder.push(&rec(0, 100.0, 1.0));
        // Hours pass and volume piles up, but price does not move.
        for i in 1..50 {
            assert!(drain(&mut builder, &rec(i * 3_600_000_000_000, 100.2, 1e6)).is_empty());
        }
    }

    #[test]
    fn a_partial_brick_is_discarded_at_end_of_data() {
        let mut builder = RenkoBarBuilder::new(1.0);
        builder.push(&rec(0, 100.0, 1.0));
        drain(&mut builder, &rec(1, 100.9, 1.0));
        assert!(builder.flush().is_none(), "an incomplete brick is not a brick");
    }

    #[test]
    fn bricks_land_on_a_stable_grid() {
        // Two builders starting from different prices in the same brick
        // must agree on where the brick boundaries are.
        let mut a = RenkoBarBuilder::new(10.0);
        let mut b = RenkoBarBuilder::new(10.0);
        a.push(&rec(0, 101.0, 1.0));
        b.push(&rec(0, 108.0, 1.0));
        let bars_a = drain(&mut a, &rec(1, 115.0, 1.0));
        let bars_b = drain(&mut b, &rec(1, 115.0, 1.0));
        assert_eq!(bars_a.len(), 1);
        assert_eq!(bars_b.len(), 1);
        assert_eq!(bars_a[0].close, 110.0);
        assert_eq!(bars_b[0].close, 110.0);
    }

    #[test]
    fn a_pathological_burst_is_capped_and_resyncs() {
        // A brick far below the instrument's scale must not try to emit
        // millions of bars from one print.
        let mut builder = RenkoBarBuilder::new(0.000_1);
        builder.push(&rec(0, 100.0, 1.0));
        let bars = drain(&mut builder, &rec(1, 200.0, 1.0));
        assert_eq!(bars.len(), MAX_BRICKS_PER_RECORD);
        // The grid resyncs to price rather than trailing a million bricks behind.
        let next = drain(&mut builder, &rec(2, 200.000_3, 1.0));
        assert!(next.len() <= 4, "grid should have caught up, got {}", next.len());
    }
}
