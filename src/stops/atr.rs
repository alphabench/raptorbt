//! ATR-based stop-loss and take-profit.

use super::{StopCalculator, TargetCalculator};
use crate::core::types::{Direction, Price};

/// ATR-based stop-loss.
#[derive(Debug, Clone)]
pub struct AtrStop {
    /// ATR multiplier.
    pub multiplier: f64,
    /// Current ATR value.
    pub atr: f64,
}

impl AtrStop {
    /// Create a new ATR stop.
    pub fn new(multiplier: f64, atr: f64) -> Self {
        Self { multiplier, atr }
    }

    /// Update ATR value.
    pub fn update_atr(&mut self, atr: f64) {
        self.atr = atr;
    }
}

impl StopCalculator for AtrStop {
    fn calculate_stop(&self, entry_price: Price, direction: Direction) -> Option<Price> {
        if self.atr <= 0.0 {
            return None;
        }

        let distance = self.atr * self.multiplier;
        let stop = match direction {
            Direction::Long => entry_price - distance,
            Direction::Short => entry_price + distance,
        };
        Some(stop)
    }

    fn update_stop(
        &self,
        current_stop: Option<Price>,
        _current_price: Price,
        _high: Price,
        _low: Price,
        _direction: Direction,
    ) -> Option<Price> {
        // ATR stop doesn't trail by default
        current_stop
    }
}

/// ATR-based take-profit.
#[derive(Debug, Clone)]
pub struct AtrTarget {
    /// ATR multiplier.
    pub multiplier: f64,
    /// Current ATR value.
    pub atr: f64,
}

impl AtrTarget {
    /// Create a new ATR target.
    pub fn new(multiplier: f64, atr: f64) -> Self {
        Self { multiplier, atr }
    }

    /// Update ATR value.
    pub fn update_atr(&mut self, atr: f64) {
        self.atr = atr;
    }
}

impl TargetCalculator for AtrTarget {
    fn calculate_target(
        &self,
        entry_price: Price,
        _stop_price: Option<Price>,
        direction: Direction,
    ) -> Option<Price> {
        if self.atr <= 0.0 {
            return None;
        }

        let distance = self.atr * self.multiplier;
        let target = match direction {
            Direction::Long => entry_price + distance,
            Direction::Short => entry_price - distance,
        };
        Some(target)
    }
}

/// Chandelier exit (ATR-based trailing stop from high/low).
#[derive(Debug, Clone)]
pub struct ChandelierExit {
    /// ATR multiplier.
    pub multiplier: f64,
    /// Current ATR value.
    pub atr: f64,
    /// Highest high since entry (for long).
    pub highest_high: f64,
    /// Lowest low since entry (for short).
    pub lowest_low: f64,
}

impl ChandelierExit {
    /// Create a new Chandelier exit.
    pub fn new(multiplier: f64, atr: f64) -> Self {
        Self {
            multiplier,
            atr,
            highest_high: 0.0,
            lowest_low: f64::MAX,
        }
    }

    /// Reset for new position.
    pub fn reset(&mut self, entry_price: Price) {
        self.highest_high = entry_price;
        self.lowest_low = entry_price;
    }

    /// Update with new bar data.
    pub fn update(&mut self, high: Price, low: Price, atr: f64) {
        if high > self.highest_high {
            self.highest_high = high;
        }
        if low < self.lowest_low {
            self.lowest_low = low;
        }
        self.atr = atr;
    }

    /// Get current stop level.
    pub fn stop_level(&self, direction: Direction) -> Option<Price> {
        if self.atr <= 0.0 {
            return None;
        }

        let distance = self.atr * self.multiplier;
        let stop = match direction {
            Direction::Long => self.highest_high - distance,
            Direction::Short => self.lowest_low + distance,
        };
        Some(stop)
    }
}

impl StopCalculator for ChandelierExit {
    fn calculate_stop(&self, entry_price: Price, direction: Direction) -> Option<Price> {
        if self.atr <= 0.0 {
            return None;
        }

        let distance = self.atr * self.multiplier;
        let stop = match direction {
            Direction::Long => entry_price - distance,
            Direction::Short => entry_price + distance,
        };
        Some(stop)
    }

    fn update_stop(
        &self,
        current_stop: Option<Price>,
        _current_price: Price,
        high: Price,
        low: Price,
        direction: Direction,
    ) -> Option<Price> {
        if self.atr <= 0.0 {
            return current_stop;
        }

        let distance = self.atr * self.multiplier;
        let new_stop = match direction {
            Direction::Long => {
                let proposed = high - distance;
                current_stop.map(|cs| cs.max(proposed)).or(Some(proposed))
            }
            Direction::Short => {
                let proposed = low + distance;
                current_stop.map(|cs| cs.min(proposed)).or(Some(proposed))
            }
        };

        new_stop
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atr_stop_long() {
        let stop = AtrStop::new(2.0, 5.0);
        let result = stop.calculate_stop(100.0, Direction::Long);
        // 100 - (2 * 5) = 90
        assert!((result.unwrap() - 90.0).abs() < 1e-10);
    }

    #[test]
    fn test_atr_stop_short() {
        let stop = AtrStop::new(2.0, 5.0);
        let result = stop.calculate_stop(100.0, Direction::Short);
        // 100 + (2 * 5) = 110
        assert!((result.unwrap() - 110.0).abs() < 1e-10);
    }

    #[test]
    fn test_atr_target() {
        let target = AtrTarget::new(3.0, 5.0);
        let result = target.calculate_target(100.0, None, Direction::Long);
        // 100 + (3 * 5) = 115
        assert!((result.unwrap() - 115.0).abs() < 1e-10);
    }

    #[test]
    fn test_chandelier_exit() {
        let mut chandelier = ChandelierExit::new(3.0, 2.0);
        chandelier.reset(100.0);

        // Simulate price movement up
        chandelier.update(105.0, 99.0, 2.0);
        chandelier.update(110.0, 103.0, 2.0);

        // Long stop should trail from highest high
        // 110 - (3 * 2) = 104
        let stop = chandelier.stop_level(Direction::Long);
        assert!((stop.unwrap() - 104.0).abs() < 1e-10);
    }

    #[test]
    fn test_atr_zero() {
        let stop = AtrStop::new(2.0, 0.0);
        let result = stop.calculate_stop(100.0, Direction::Long);
        assert!(result.is_none());
    }
}
