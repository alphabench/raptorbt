//! Fixed percentage stop-loss and take-profit.

use super::{StopCalculator, TargetCalculator};
use crate::core::types::{Direction, Price};

/// Fixed percentage stop-loss.
#[derive(Debug, Clone, Copy)]
pub struct FixedStop {
    /// Stop percentage (e.g., 0.02 for 2%).
    pub percent: f64,
}

impl FixedStop {
    /// Create a new fixed stop with given percentage.
    pub fn new(percent: f64) -> Self {
        Self { percent: percent.abs() }
    }

    /// Create a 1% stop.
    pub fn one_percent() -> Self {
        Self::new(0.01)
    }

    /// Create a 2% stop.
    pub fn two_percent() -> Self {
        Self::new(0.02)
    }

    /// Create a 5% stop.
    pub fn five_percent() -> Self {
        Self::new(0.05)
    }
}

impl StopCalculator for FixedStop {
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
        _high: Price,
        _low: Price,
        _direction: Direction,
    ) -> Option<Price> {
        // Fixed stop doesn't update
        current_stop
    }
}

/// Fixed percentage take-profit.
#[derive(Debug, Clone, Copy)]
pub struct FixedTarget {
    /// Target percentage (e.g., 0.04 for 4%).
    pub percent: f64,
}

impl FixedTarget {
    /// Create a new fixed target with given percentage.
    pub fn new(percent: f64) -> Self {
        Self { percent: percent.abs() }
    }
}

impl TargetCalculator for FixedTarget {
    fn calculate_target(
        &self,
        entry_price: Price,
        _stop_price: Option<Price>,
        direction: Direction,
    ) -> Option<Price> {
        let target = match direction {
            Direction::Long => entry_price * (1.0 + self.percent),
            Direction::Short => entry_price * (1.0 - self.percent),
        };
        Some(target)
    }
}

/// Risk-reward based take-profit.
#[derive(Debug, Clone, Copy)]
pub struct RiskRewardTarget {
    /// Risk-reward ratio (e.g., 2.0 for 2:1 reward:risk).
    pub ratio: f64,
}

impl RiskRewardTarget {
    /// Create a new risk-reward target.
    pub fn new(ratio: f64) -> Self {
        Self { ratio }
    }

    /// Create a 2:1 target.
    pub fn two_to_one() -> Self {
        Self::new(2.0)
    }

    /// Create a 3:1 target.
    pub fn three_to_one() -> Self {
        Self::new(3.0)
    }
}

impl TargetCalculator for RiskRewardTarget {
    fn calculate_target(
        &self,
        entry_price: Price,
        stop_price: Option<Price>,
        direction: Direction,
    ) -> Option<Price> {
        let stop = stop_price?;
        let risk = (entry_price - stop).abs();
        let reward = risk * self.ratio;

        let target = match direction {
            Direction::Long => entry_price + reward,
            Direction::Short => entry_price - reward,
        };
        Some(target)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixed_stop_long() {
        let stop = FixedStop::new(0.02);
        let result = stop.calculate_stop(100.0, Direction::Long);
        assert!((result.unwrap() - 98.0).abs() < 1e-10);
    }

    #[test]
    fn test_fixed_stop_short() {
        let stop = FixedStop::new(0.02);
        let result = stop.calculate_stop(100.0, Direction::Short);
        assert!((result.unwrap() - 102.0).abs() < 1e-10);
    }

    #[test]
    fn test_fixed_target_long() {
        let target = FixedTarget::new(0.04);
        let result = target.calculate_target(100.0, None, Direction::Long);
        assert!((result.unwrap() - 104.0).abs() < 1e-10);
    }

    #[test]
    fn test_risk_reward_target() {
        let target = RiskRewardTarget::new(2.0);
        // Entry at 100, stop at 98 (2% risk), target should be at 104 (4% reward)
        let result = target.calculate_target(100.0, Some(98.0), Direction::Long);
        assert!((result.unwrap() - 104.0).abs() < 1e-10);
    }

    #[test]
    fn test_risk_reward_no_stop() {
        let target = RiskRewardTarget::new(2.0);
        let result = target.calculate_target(100.0, None, Direction::Long);
        assert!(result.is_none());
    }
}
