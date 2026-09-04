//! Order fill simulation models.

use crate::core::types::{Direction, FillTiming, OhlcvBar, Price};

/// Fill price model determining at what price orders are executed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FillPrice {
    /// Execute at close price (end of bar).
    #[default]
    Close,
    /// Execute at open price (start of next bar).
    Open,
    /// Execute at OHLC average.
    Average,
    /// Execute at typical price (H+L+C)/3.
    Typical,
    /// Execute at VWAP (if available, otherwise typical).
    Vwap,
    /// Execute at worst price (high for buys, low for sells).
    Worst,
    /// Execute at best price (low for buys, high for sells).
    Best,
}

impl FillPrice {
    /// Price source implied by an execution-timing policy.
    ///
    /// `NextBarOpen` and `SameBarOpenLookahead` both read the open — the
    /// policy decides *which bar* that open belongs to, which is the
    /// kernel's job, not this table's.
    pub fn for_timing(timing: FillTiming) -> Self {
        match timing {
            FillTiming::SameBarClose => FillPrice::Close,
            FillTiming::NextBarOpen | FillTiming::SameBarOpenLookahead => FillPrice::Open,
        }
    }

    /// Get execution price from OHLCV bar.
    ///
    /// # Arguments
    /// * `bar` - OHLCV bar data
    /// * `direction` - Trade direction
    /// * `is_entry` - Whether this is an entry or exit
    ///
    /// # Returns
    /// Execution price
    pub fn get_price(&self, bar: &OhlcvBar, direction: Direction, is_entry: bool) -> Price {
        match self {
            FillPrice::Close => bar.close,
            FillPrice::Open => bar.open,
            FillPrice::Average => (bar.open + bar.high + bar.low + bar.close) / 4.0,
            FillPrice::Typical => (bar.high + bar.low + bar.close) / 3.0,
            FillPrice::Vwap => (bar.high + bar.low + bar.close) / 3.0, // Simplified
            FillPrice::Worst => {
                // Worst price for the trade
                match (direction, is_entry) {
                    (Direction::Long, true) => bar.high,   // Buy high
                    (Direction::Long, false) => bar.low,   // Sell low
                    (Direction::Short, true) => bar.low,   // Short at low (bad)
                    (Direction::Short, false) => bar.high, // Cover at high (bad)
                }
            }
            FillPrice::Best => {
                // Best price for the trade
                match (direction, is_entry) {
                    (Direction::Long, true) => bar.low,   // Buy low
                    (Direction::Long, false) => bar.high, // Sell high
                    (Direction::Short, true) => bar.high, // Short at high (good)
                    (Direction::Short, false) => bar.low, // Cover at low (good)
                }
            }
        }
    }

    /// Get execution price from separate arrays.
    ///
    /// # Arguments
    /// * `open` - Open price
    /// * `high` - High price
    /// * `low` - Low price
    /// * `close` - Close price
    /// * `direction` - Trade direction
    /// * `is_entry` - Whether this is an entry or exit
    ///
    /// # Returns
    /// Execution price
    pub fn get_price_from_arrays(
        &self,
        open: Price,
        high: Price,
        low: Price,
        close: Price,
        direction: Direction,
        is_entry: bool,
    ) -> Price {
        match self {
            FillPrice::Close => close,
            FillPrice::Open => open,
            FillPrice::Average => (open + high + low + close) / 4.0,
            FillPrice::Typical => (high + low + close) / 3.0,
            FillPrice::Vwap => (high + low + close) / 3.0,
            FillPrice::Worst => match (direction, is_entry) {
                (Direction::Long, true) => high,
                (Direction::Long, false) => low,
                (Direction::Short, true) => low,
                (Direction::Short, false) => high,
            },
            FillPrice::Best => match (direction, is_entry) {
                (Direction::Long, true) => low,
                (Direction::Long, false) => high,
                (Direction::Short, true) => high,
                (Direction::Short, false) => low,
            },
        }
    }
}

/// Deterministic SplitMix64 stream for stochastic fills.
///
/// Hand-rolled (like the Monte Carlo module's generator) to avoid an RNG
/// dependency; the same seed always produces the same fill sequence.
#[derive(Debug, Clone)]
pub struct FillRng(u64);

impl FillRng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// Next uniform draw in [0, 1).
    pub fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Fill model combining price model with execution rules.
#[derive(Debug, Clone)]
pub struct FillModel {
    /// Price model for fills.
    pub fill_price: FillPrice,
    /// Adverse price adjustment applied to limit fills, as a fraction of
    /// the limit price. `0.0` (default) fills exactly at the limit.
    ///
    /// Models being filled only when the market is about to move through
    /// you — adverse selection on a resting order. It does *not* apply
    /// when the queue model granted the fill: observed volume traded ahead
    /// of you is evidence you genuinely held that price.
    pub limit_slippage: f64,
}

impl Default for FillModel {
    fn default() -> Self {
        Self { fill_price: FillPrice::Close, limit_slippage: 0.0 }
    }
}

impl FillModel {
    /// Create a fill model that executes at close.
    pub fn at_close() -> Self {
        Self { fill_price: FillPrice::Close, ..Self::default() }
    }

    /// Check if a limit order would be filled.
    ///
    /// # Arguments
    /// * `limit_price` - Limit price
    /// * `bar` - OHLCV bar
    /// * `direction` - Trade direction
    /// * `is_entry` - Whether this is an entry or exit
    ///
    /// # Returns
    /// True if order would be filled
    pub fn would_fill_limit(
        &self,
        limit_price: Price,
        bar: &OhlcvBar,
        direction: Direction,
        is_entry: bool,
    ) -> bool {
        match (direction, is_entry) {
            // Long entry: buy at or below limit
            (Direction::Long, true) => bar.low <= limit_price,
            // Long exit: sell at or above limit
            (Direction::Long, false) => bar.high >= limit_price,
            // Short entry: sell at or above limit
            (Direction::Short, true) => bar.high >= limit_price,
            // Short exit: buy at or below limit
            (Direction::Short, false) => bar.low <= limit_price,
        }
    }

    /// Get fill price for a limit order.
    ///
    /// Returns limit price if filled, None if not filled.
    ///
    /// # Arguments
    /// * `limit_price` - Limit price
    /// * `bar` - OHLCV bar
    /// * `direction` - Trade direction
    /// * `is_entry` - Whether this is an entry or exit
    ///
    /// # Returns
    /// Fill price or None
    pub fn get_limit_fill_price(
        &self,
        limit_price: Price,
        bar: &OhlcvBar,
        direction: Direction,
        is_entry: bool,
    ) -> Option<Price> {
        if !self.would_fill_limit(limit_price, bar, direction, is_entry) {
            return None;
        }
        if self.limit_slippage == 0.0 {
            return Some(limit_price);
        }
        // Slip against the trader: a buy pays more, a sell receives less.
        let adjust = limit_price * self.limit_slippage;
        Some(match (direction, is_entry) {
            (Direction::Long, true) | (Direction::Short, false) => limit_price + adjust,
            _ => limit_price - adjust,
        })
    }

    /// Check if a stop order would be triggered.
    ///
    /// # Arguments
    /// * `stop_price` - Stop price
    /// * `bar` - OHLCV bar
    /// * `direction` - Trade direction
    /// * `is_entry` - Whether this is an entry or exit
    ///
    /// # Returns
    /// True if stop would be triggered
    pub fn would_trigger_stop(
        &self,
        stop_price: Price,
        bar: &OhlcvBar,
        direction: Direction,
        is_entry: bool,
    ) -> bool {
        match (direction, is_entry) {
            // Long entry stop: buy when price rises to stop
            (Direction::Long, true) => bar.high >= stop_price,
            // Long exit stop: sell when price falls to stop
            (Direction::Long, false) => bar.low <= stop_price,
            // Short entry stop: sell when price falls to stop
            (Direction::Short, true) => bar.low <= stop_price,
            // Short exit stop: buy when price rises to stop
            (Direction::Short, false) => bar.high >= stop_price,
        }
    }

    /// Get fill price for a stop order.
    ///
    /// Returns fill price if triggered, None if not.
    /// Uses worst-case scenario (stop price or worse).
    ///
    /// # Arguments
    /// * `stop_price` - Stop price
    /// * `bar` - OHLCV bar
    /// * `direction` - Trade direction
    /// * `is_entry` - Whether this is an entry or exit
    ///
    /// # Returns
    /// Fill price or None
    pub fn get_stop_fill_price(
        &self,
        stop_price: Price,
        bar: &OhlcvBar,
        direction: Direction,
        is_entry: bool,
    ) -> Option<Price> {
        if !self.would_trigger_stop(stop_price, bar, direction, is_entry) {
            return None;
        }

        // Check for gap through stop
        match (direction, is_entry) {
            (Direction::Long, true) => {
                // Buy stop: fill at stop or worse (gap up through stop)
                if bar.open >= stop_price {
                    Some(bar.open) // Gap up, fill at open
                } else {
                    Some(stop_price)
                }
            }
            (Direction::Long, false) => {
                // Sell stop: fill at stop or worse (gap down through stop)
                if bar.open <= stop_price {
                    Some(bar.open) // Gap down, fill at open
                } else {
                    Some(stop_price)
                }
            }
            (Direction::Short, true) => {
                // Short stop: fill at stop or worse (gap down through stop)
                if bar.open <= stop_price {
                    Some(bar.open)
                } else {
                    Some(stop_price)
                }
            }
            (Direction::Short, false) => {
                // Cover stop: fill at stop or worse (gap up through stop)
                if bar.open >= stop_price {
                    Some(bar.open)
                } else {
                    Some(stop_price)
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn limit_fills_at_the_limit_by_default() {
        let model = FillModel::default();
        let bar = OhlcvBar {
            timestamp: 0,
            open: 100.0,
            high: 101.0,
            low: 98.0,
            close: 100.0,
            volume: 1.0,
        };
        assert_eq!(model.get_limit_fill_price(99.0, &bar, Direction::Long, true), Some(99.0));
    }

    #[test]
    fn limit_slippage_moves_against_the_trader() {
        let model = FillModel { limit_slippage: 0.01, ..FillModel::default() };
        let bar = OhlcvBar {
            timestamp: 0,
            open: 100.0,
            high: 101.0,
            low: 98.0,
            close: 100.0,
            volume: 1.0,
        };
        // A buy pays more than its limit.
        assert_eq!(model.get_limit_fill_price(100.0, &bar, Direction::Long, true), Some(101.0));
        // A sell receives less.
        assert_eq!(model.get_limit_fill_price(100.0, &bar, Direction::Short, true), Some(99.0));
    }

    #[test]
    fn limit_slippage_does_not_create_fills() {
        // A price the market never reached stays unfilled however the
        // slippage is configured.
        let model = FillModel { limit_slippage: 0.05, ..FillModel::default() };
        let bar = OhlcvBar {
            timestamp: 0,
            open: 100.0,
            high: 101.0,
            low: 99.0,
            close: 100.0,
            volume: 1.0,
        };
        assert_eq!(model.get_limit_fill_price(90.0, &bar, Direction::Long, true), None);
    }

    fn test_bar() -> OhlcvBar {
        OhlcvBar { timestamp: 0, open: 100.0, high: 105.0, low: 95.0, close: 102.0, volume: 1000.0 }
    }

    #[test]
    fn test_fill_price_close() {
        let bar = test_bar();
        let fp = FillPrice::Close;
        assert!((fp.get_price(&bar, Direction::Long, true) - 102.0).abs() < 1e-10);
    }

    #[test]
    fn test_fill_price_worst() {
        let bar = test_bar();
        let fp = FillPrice::Worst;

        // Long entry: high (105)
        assert!((fp.get_price(&bar, Direction::Long, true) - 105.0).abs() < 1e-10);

        // Long exit: low (95)
        assert!((fp.get_price(&bar, Direction::Long, false) - 95.0).abs() < 1e-10);
    }

    #[test]
    fn test_limit_fill() {
        let fill = FillModel::default();
        let bar = test_bar();

        // Limit buy at 96 should fill (low is 95)
        assert!(fill.would_fill_limit(96.0, &bar, Direction::Long, true));

        // Limit buy at 94 should not fill (low is 95)
        assert!(!fill.would_fill_limit(94.0, &bar, Direction::Long, true));
    }

    #[test]
    fn test_stop_fill() {
        let fill = FillModel::default();
        let bar = test_bar();

        // Stop sell at 96 should trigger (low is 95)
        assert!(fill.would_trigger_stop(96.0, &bar, Direction::Long, false));

        // Stop sell at 94 should not trigger (low is 95)
        assert!(!fill.would_trigger_stop(94.0, &bar, Direction::Long, false));
    }

    #[test]
    fn test_gap_through_stop() {
        let fill = FillModel::default();

        // Gap down through stop
        let gap_bar = OhlcvBar {
            timestamp: 0,
            open: 90.0, // Gap down from stop at 95
            high: 92.0,
            low: 88.0,
            close: 91.0,
            volume: 1000.0,
        };

        let fill_price = fill.get_stop_fill_price(95.0, &gap_bar, Direction::Long, false);
        // Should fill at open (90) not stop (95)
        assert_eq!(fill_price, Some(90.0));
    }
}
