//! Fee calculation models.

use crate::core::types::{Direction, Price};
use crate::execution::indian_costs::{self, FeeBreakdown, Segment};

/// Fee model for calculating transaction costs.
#[derive(Debug, Clone)]
pub enum FeeModel {
    /// No fees.
    None,
    /// Fixed percentage of trade value.
    Percentage(f64),
    /// Fixed fee per trade.
    Fixed(f64),
    /// Per-share/contract fee.
    PerShare(f64),
    /// Tiered fee structure based on trade value.
    Tiered(Vec<(f64, f64)>), // (threshold, rate)
    /// Custom fee function (stored as percentage for simplicity).
    Custom { base: f64, per_share: f64 },
    /// Itemized Indian regulatory costs for a market segment.
    ///
    /// Unlike the other variants this produces a per-component breakdown, so
    /// the equity curve and the reported costs are the same numbers.
    Indian { segment: Segment },
}

impl Default for FeeModel {
    fn default() -> Self {
        FeeModel::Percentage(0.001) // 0.1% default
    }
}

impl FeeModel {
    /// Create a new percentage fee model.
    pub fn percentage(rate: f64) -> Self {
        FeeModel::Percentage(rate)
    }

    /// Create a new fixed fee model.
    pub fn fixed(amount: f64) -> Self {
        FeeModel::Fixed(amount)
    }

    /// Create a new per-share fee model.
    pub fn per_share(rate: f64) -> Self {
        FeeModel::PerShare(rate)
    }

    /// Calculate fee for a trade.
    ///
    /// # Arguments
    /// * `price` - Trade price
    /// * `size` - Position size (shares/contracts)
    /// * `direction` - Trade direction (for asymmetric fees if needed)
    ///
    /// # Returns
    /// Fee amount
    pub fn calculate(&self, price: Price, size: f64, _direction: Direction) -> f64 {
        let trade_value = price * size.abs();

        match self {
            FeeModel::None => 0.0,
            FeeModel::Percentage(rate) => trade_value * rate,
            FeeModel::Fixed(amount) => *amount,
            FeeModel::PerShare(rate) => size.abs() * rate,
            FeeModel::Tiered(tiers) => {
                // Find applicable tier
                let mut applicable_rate = 0.0;
                for (threshold, rate) in tiers {
                    if trade_value >= *threshold {
                        applicable_rate = *rate;
                    } else {
                        break;
                    }
                }
                trade_value * applicable_rate
            }
            FeeModel::Custom { base, per_share } => base + size.abs() * per_share,
            FeeModel::Indian { segment } => {
                indian_costs::calculate_side(*segment, trade_value, _direction, true).total()
            }
        }
    }

    /// Itemized costs for one side of a trade.
    ///
    /// Returns `None` for the flat models, which have no component structure.
    /// Unlike [`FeeModel::calculate`], this distinguishes entry from exit,
    /// which matters because STT lands on the sell leg and stamp duty on the
    /// buy leg.
    pub fn breakdown(
        &self,
        price: Price,
        size: f64,
        direction: Direction,
        is_entry: bool,
    ) -> Option<FeeBreakdown> {
        match self {
            FeeModel::Indian { segment } => Some(indian_costs::calculate_side(
                *segment,
                price * size.abs(),
                direction,
                is_entry,
            )),
            _ => None,
        }
    }

    /// Fee for one side, honoring entry/exit asymmetry where the model has any.
    pub fn calculate_side(
        &self,
        price: Price,
        size: f64,
        direction: Direction,
        is_entry: bool,
    ) -> f64 {
        match self.breakdown(price, size, direction, is_entry) {
            Some(b) => b.total(),
            None => self.calculate(price, size, direction),
        }
    }

    /// Create an itemized Indian cost model for a segment.
    pub fn indian(segment: Segment) -> Self {
        FeeModel::Indian { segment }
    }

    /// Calculate round-trip fees (entry + exit).
    pub fn round_trip(
        &self,
        entry_price: Price,
        exit_price: Price,
        size: f64,
        direction: Direction,
    ) -> f64 {
        self.calculate(entry_price, size, direction) + self.calculate(exit_price, size, direction)
    }
}

/// Broker-specific fee configurations.
pub struct BrokerFees;

impl BrokerFees {
    /// Interactive Brokers tiered pricing (approximate).
    pub fn interactive_brokers() -> FeeModel {
        FeeModel::Custom { base: 1.0, per_share: 0.005 }
    }

    /// Zero commission broker (like Robinhood).
    pub fn zero_commission() -> FeeModel {
        FeeModel::None
    }

    /// Indian broker (Zerodha-like).
    pub fn india_equity() -> FeeModel {
        // 0.03% or Rs 20 per trade, whichever is lower
        // Simplified as 0.03%
        FeeModel::Percentage(0.0003)
    }

    /// Crypto exchange (typical).
    pub fn crypto_exchange() -> FeeModel {
        FeeModel::Percentage(0.001) // 0.1% maker/taker
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_percentage_fee() {
        let fee = FeeModel::percentage(0.001);
        let result = fee.calculate(100.0, 100.0, Direction::Long);
        assert!((result - 10.0).abs() < 1e-10); // 100 * 100 * 0.001 = 10
    }

    #[test]
    fn test_fixed_fee() {
        let fee = FeeModel::fixed(5.0);
        let result = fee.calculate(100.0, 100.0, Direction::Long);
        assert!((result - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_per_share_fee() {
        let fee = FeeModel::per_share(0.01);
        let result = fee.calculate(100.0, 100.0, Direction::Long);
        assert!((result - 1.0).abs() < 1e-10); // 100 * 0.01 = 1
    }

    #[test]
    fn test_round_trip() {
        let fee = FeeModel::percentage(0.001);
        let result = fee.round_trip(100.0, 110.0, 100.0, Direction::Long);
        // Entry: 100 * 100 * 0.001 = 10
        // Exit: 110 * 100 * 0.001 = 11
        // Total: 21
        assert!((result - 21.0).abs() < 1e-10);
    }

    #[test]
    fn test_no_fee() {
        let fee = FeeModel::None;
        let result = fee.calculate(100.0, 100.0, Direction::Long);
        assert!((result - 0.0).abs() < 1e-10);
    }
}
