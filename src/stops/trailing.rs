//! Trailing stop implementations.

use super::StopCalculator;
use crate::core::types::{Direction, Price};

/// Percentage-based trailing stop.
#[derive(Debug, Clone, Copy)]
pub struct TrailingStop {
    /// Trail percentage (e.g., 0.05 for 5%).
    pub percent: f64,
    /// Activation threshold (optional - start trailing after this profit %).
    pub activation_threshold: Option<f64>,
}

impl TrailingStop {
    /// Create a new trailing stop.
    pub fn new(percent: f64) -> Self {
        Self {
            percent: percent.abs(),
            activation_threshold: None,
        }
    }

    /// Create with activation threshold.
    pub fn with_activation(mut self, threshold: f64) -> Self {
        self.activation_threshold = Some(threshold.abs());
        self
    }

    /// Check if trailing should be activated.
    #[allow(dead_code)]
    fn should_activate(
        &self,
        entry_price: Price,
        current_price: Price,
        direction: Direction,
    ) -> bool {
        if let Some(threshold) = self.activation_threshold {
            let profit_pct = match direction {
                Direction::Long => (current_price - entry_price) / entry_price,
                Direction::Short => (entry_price - current_price) / entry_price,
            };
            profit_pct >= threshold
        } else {
            true // Always active if no threshold
        }
    }
}

impl StopCalculator for TrailingStop {
    fn calculate_stop(&self, entry_price: Price, direction: Direction) -> Option<Price> {
        let stop = match direction {
            Direction::Long => entry_price * (1.0 - self.percent),
            Direction::Short => entry_price * (1.0 + self.percent),
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
        match direction {
            Direction::Long => {
                // Trail below the high
                let new_stop = high * (1.0 - self.percent);
                current_stop.map(|cs| cs.max(new_stop)).or(Some(new_stop))
            }
            Direction::Short => {
                // Trail above the low
                let new_stop = low * (1.0 + self.percent);
                current_stop.map(|cs| cs.min(new_stop)).or(Some(new_stop))
            }
        }
    }
}

/// Point-based trailing stop (fixed point distance).
#[derive(Debug, Clone, Copy)]
pub struct PointTrailingStop {
    /// Trail distance in points.
    pub points: f64,
}

impl PointTrailingStop {
    /// Create a new point-based trailing stop.
    pub fn new(points: f64) -> Self {
        Self {
            points: points.abs(),
        }
    }
}

impl StopCalculator for PointTrailingStop {
    fn calculate_stop(&self, entry_price: Price, direction: Direction) -> Option<Price> {
        let stop = match direction {
            Direction::Long => entry_price - self.points,
            Direction::Short => entry_price + self.points,
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
        match direction {
            Direction::Long => {
                let new_stop = high - self.points;
                current_stop.map(|cs| cs.max(new_stop)).or(Some(new_stop))
            }
            Direction::Short => {
                let new_stop = low + self.points;
                current_stop.map(|cs| cs.min(new_stop)).or(Some(new_stop))
            }
        }
    }
}

/// Step trailing stop (moves in discrete steps).
#[derive(Debug, Clone, Copy)]
pub struct StepTrailingStop {
    /// Step size percentage.
    pub step_percent: f64,
    /// Trail percentage from each step.
    pub trail_percent: f64,
}

impl StepTrailingStop {
    /// Create a new step trailing stop.
    pub fn new(step_percent: f64, trail_percent: f64) -> Self {
        Self {
            step_percent: step_percent.abs(),
            trail_percent: trail_percent.abs(),
        }
    }

    /// Calculate stop for a given step level.
    fn stop_for_step(&self, entry_price: Price, step: usize, direction: Direction) -> Price {
        let step_gain = self.step_percent * step as f64;
        match direction {
            Direction::Long => {
                let step_price = entry_price * (1.0 + step_gain);
                step_price * (1.0 - self.trail_percent)
            }
            Direction::Short => {
                let step_price = entry_price * (1.0 - step_gain);
                step_price * (1.0 + self.trail_percent)
            }
        }
    }

    /// Determine current step level.
    #[allow(dead_code)]
    fn current_step(
        &self,
        entry_price: Price,
        extreme_price: Price,
        direction: Direction,
    ) -> usize {
        let gain = match direction {
            Direction::Long => (extreme_price - entry_price) / entry_price,
            Direction::Short => (entry_price - extreme_price) / entry_price,
        };

        if gain <= 0.0 {
            return 0;
        }

        (gain / self.step_percent).floor() as usize
    }
}

impl StopCalculator for StepTrailingStop {
    fn calculate_stop(&self, entry_price: Price, direction: Direction) -> Option<Price> {
        Some(self.stop_for_step(entry_price, 0, direction))
    }

    fn update_stop(
        &self,
        current_stop: Option<Price>,
        _current_price: Price,
        high: Price,
        low: Price,
        direction: Direction,
    ) -> Option<Price> {
        // This is a simplified version - full implementation would need entry price
        // For now, just use regular trailing behavior
        match direction {
            Direction::Long => {
                let new_stop = high * (1.0 - self.trail_percent);
                current_stop.map(|cs| cs.max(new_stop)).or(Some(new_stop))
            }
            Direction::Short => {
                let new_stop = low * (1.0 + self.trail_percent);
                current_stop.map(|cs| cs.min(new_stop)).or(Some(new_stop))
            }
        }
    }
}

/// Parabolic SAR style trailing stop.
#[derive(Debug, Clone)]
pub struct ParabolicStop {
    /// Initial acceleration factor.
    pub af_start: f64,
    /// Acceleration factor increment.
    pub af_step: f64,
    /// Maximum acceleration factor.
    pub af_max: f64,
    /// Current acceleration factor.
    current_af: f64,
    /// Current extreme point.
    extreme_point: f64,
    /// Current SAR value.
    current_sar: f64,
}

impl ParabolicStop {
    /// Create a new Parabolic SAR stop with default parameters.
    pub fn new() -> Self {
        Self::with_params(0.02, 0.02, 0.2)
    }

    /// Create with custom parameters.
    pub fn with_params(af_start: f64, af_step: f64, af_max: f64) -> Self {
        Self {
            af_start,
            af_step,
            af_max,
            current_af: af_start,
            extreme_point: 0.0,
            current_sar: 0.0,
        }
    }

    /// Initialize for new position.
    pub fn init(&mut self, entry_price: Price, direction: Direction) {
        self.current_af = self.af_start;
        self.extreme_point = entry_price;
        self.current_sar = match direction {
            Direction::Long => entry_price * 0.99,  // Slightly below entry
            Direction::Short => entry_price * 1.01, // Slightly above entry
        };
    }

    /// Update SAR with new bar data.
    pub fn update_sar(&mut self, high: Price, low: Price, direction: Direction) -> Price {
        // Update extreme point
        let new_ep = match direction {
            Direction::Long => {
                if high > self.extreme_point {
                    self.current_af = (self.current_af + self.af_step).min(self.af_max);
                    high
                } else {
                    self.extreme_point
                }
            }
            Direction::Short => {
                if low < self.extreme_point {
                    self.current_af = (self.current_af + self.af_step).min(self.af_max);
                    low
                } else {
                    self.extreme_point
                }
            }
        };
        self.extreme_point = new_ep;

        // Calculate new SAR
        let new_sar = self.current_sar + self.current_af * (self.extreme_point - self.current_sar);

        // Ensure SAR doesn't cross price
        self.current_sar = match direction {
            Direction::Long => new_sar.min(low),
            Direction::Short => new_sar.max(high),
        };

        self.current_sar
    }
}

impl Default for ParabolicStop {
    fn default() -> Self {
        Self::new()
    }
}

impl StopCalculator for ParabolicStop {
    fn calculate_stop(&self, _entry_price: Price, _direction: Direction) -> Option<Price> {
        if self.current_sar > 0.0 {
            Some(self.current_sar)
        } else {
            None
        }
    }

    fn update_stop(
        &self,
        _current_stop: Option<Price>,
        _current_price: Price,
        _high: Price,
        _low: Price,
        _direction: Direction,
    ) -> Option<Price> {
        // Parabolic stop is updated via update_sar method
        if self.current_sar > 0.0 {
            Some(self.current_sar)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trailing_stop_long() {
        let stop = TrailingStop::new(0.05);

        // Initial stop
        let initial = stop.calculate_stop(100.0, Direction::Long);
        assert!((initial.unwrap() - 95.0).abs() < 1e-10);

        // Update with higher high
        let updated = stop.update_stop(initial, 108.0, 110.0, 105.0, Direction::Long);
        // 110 * 0.95 = 104.5
        assert!((updated.unwrap() - 104.5).abs() < 1e-10);
    }

    #[test]
    fn test_trailing_stop_short() {
        let stop = TrailingStop::new(0.05);

        // Initial stop
        let initial = stop.calculate_stop(100.0, Direction::Short);
        assert!((initial.unwrap() - 105.0).abs() < 1e-10);

        // Update with lower low
        let updated = stop.update_stop(initial, 92.0, 95.0, 90.0, Direction::Short);
        // 90 * 1.05 = 94.5
        assert!((updated.unwrap() - 94.5).abs() < 1e-10);
    }

    #[test]
    fn test_trailing_stop_only_tightens() {
        let stop = TrailingStop::new(0.05);

        let initial = stop.calculate_stop(100.0, Direction::Long);

        // Move up
        let moved_up = stop.update_stop(initial, 110.0, 110.0, 108.0, Direction::Long);
        // 110 * 0.95 = 104.5
        assert!((moved_up.unwrap() - 104.5).abs() < 1e-10);

        // Move down - stop should NOT move down
        let moved_down = stop.update_stop(moved_up, 105.0, 106.0, 103.0, Direction::Long);
        // Should still be 104.5 (not 106 * 0.95 = 100.7)
        assert!((moved_down.unwrap() - 104.5).abs() < 1e-10);
    }

    #[test]
    fn test_point_trailing_stop() {
        let stop = PointTrailingStop::new(5.0);

        // Initial stop
        let initial = stop.calculate_stop(100.0, Direction::Long);
        assert!((initial.unwrap() - 95.0).abs() < 1e-10);

        // Update with higher high
        let updated = stop.update_stop(initial, 108.0, 110.0, 105.0, Direction::Long);
        // 110 - 5 = 105
        assert!((updated.unwrap() - 105.0).abs() < 1e-10);
    }

    #[test]
    fn test_parabolic_stop() {
        let mut stop = ParabolicStop::new();
        stop.init(100.0, Direction::Long);

        // Simulate uptrend
        let sar1 = stop.update_sar(102.0, 99.0, Direction::Long);
        let sar2 = stop.update_sar(105.0, 101.0, Direction::Long);
        let sar3 = stop.update_sar(108.0, 103.0, Direction::Long);

        // SAR should be increasing
        assert!(sar2 > sar1);
        assert!(sar3 > sar2);

        // SAR should be below current low
        assert!(sar3 < 103.0);
    }
}
