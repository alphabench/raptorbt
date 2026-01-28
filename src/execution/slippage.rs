//! Slippage models for realistic trade execution.

use crate::core::types::{Direction, Price};

/// Slippage model for simulating execution price deviation.
#[derive(Debug, Clone)]
pub enum SlippageModel {
    /// No slippage.
    None,
    /// Fixed percentage slippage.
    Percentage(f64),
    /// Fixed point slippage.
    Fixed(f64),
    /// Volume-based slippage (higher volume = lower slippage).
    VolumeBased { base: f64, volume_factor: f64 },
    /// Spread-based slippage (uses bid-ask spread).
    SpreadBased { half_spread: f64 },
}

impl Default for SlippageModel {
    fn default() -> Self {
        SlippageModel::None
    }
}

impl SlippageModel {
    /// Create a new percentage slippage model.
    pub fn percentage(rate: f64) -> Self {
        SlippageModel::Percentage(rate)
    }

    /// Create a new fixed slippage model.
    pub fn fixed(points: f64) -> Self {
        SlippageModel::Fixed(points)
    }

    /// Create a volume-based slippage model.
    pub fn volume_based(base: f64, volume_factor: f64) -> Self {
        SlippageModel::VolumeBased {
            base,
            volume_factor,
        }
    }

    /// Calculate slippage for a trade.
    ///
    /// For long entries and short exits: slippage is ADDED to price (pay more/receive less)
    /// For short entries and long exits: slippage is SUBTRACTED from price
    ///
    /// # Arguments
    /// * `price` - Base execution price
    /// * `direction` - Trade direction
    /// * `is_entry` - Whether this is an entry or exit
    /// * `volume` - Optional volume for volume-based models
    ///
    /// # Returns
    /// Slippage amount (positive = unfavorable)
    pub fn calculate(
        &self,
        price: Price,
        direction: Direction,
        is_entry: bool,
        volume: Option<f64>,
    ) -> f64 {
        let base_slippage = match self {
            SlippageModel::None => 0.0,
            SlippageModel::Percentage(rate) => price * rate,
            SlippageModel::Fixed(points) => *points,
            SlippageModel::VolumeBased {
                base,
                volume_factor,
            } => {
                if let Some(vol) = volume {
                    if vol > 0.0 {
                        base * (1.0 / (1.0 + vol * volume_factor))
                    } else {
                        *base
                    }
                } else {
                    *base
                }
            }
            SlippageModel::SpreadBased { half_spread } => *half_spread,
        };

        // Determine sign based on trade type
        // Long entry: pay higher price (positive slippage)
        // Long exit: receive lower price (negative slippage)
        // Short entry: receive higher price (negative slippage means worse)
        // Short exit: pay higher price
        match (direction, is_entry) {
            (Direction::Long, true) => base_slippage,   // Pay more
            (Direction::Long, false) => -base_slippage, // Receive less
            (Direction::Short, true) => -base_slippage, // Receive less
            (Direction::Short, false) => base_slippage, // Pay more
        }
    }

    /// Apply slippage to get execution price.
    ///
    /// # Arguments
    /// * `price` - Base price
    /// * `direction` - Trade direction
    /// * `is_entry` - Whether this is an entry or exit
    /// * `volume` - Optional volume for volume-based models
    ///
    /// # Returns
    /// Execution price after slippage
    pub fn apply(
        &self,
        price: Price,
        direction: Direction,
        is_entry: bool,
        volume: Option<f64>,
    ) -> Price {
        price + self.calculate(price, direction, is_entry, volume)
    }
}

/// Market impact model for large orders.
#[derive(Debug, Clone)]
pub struct MarketImpact {
    /// Temporary impact coefficient.
    pub temporary_impact: f64,
    /// Permanent impact coefficient.
    pub permanent_impact: f64,
    /// Average daily volume for normalization.
    pub avg_daily_volume: f64,
}

impl MarketImpact {
    /// Create a new market impact model.
    pub fn new(temporary: f64, permanent: f64, adv: f64) -> Self {
        Self {
            temporary_impact: temporary,
            permanent_impact: permanent,
            avg_daily_volume: adv,
        }
    }

    /// Calculate market impact for an order.
    ///
    /// Uses simplified square-root model: impact = sigma * sqrt(Q / ADV)
    ///
    /// # Arguments
    /// * `order_size` - Number of shares/contracts
    /// * `price` - Current price
    /// * `volatility` - Price volatility (sigma)
    ///
    /// # Returns
    /// Total market impact in price terms
    pub fn calculate(&self, order_size: f64, price: Price, volatility: f64) -> f64 {
        if self.avg_daily_volume <= 0.0 {
            return 0.0;
        }

        let participation_rate = order_size / self.avg_daily_volume;
        let sqrt_participation = participation_rate.sqrt();

        let temporary = self.temporary_impact * volatility * price * sqrt_participation;
        let permanent = self.permanent_impact * volatility * price * participation_rate;

        temporary + permanent
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_percentage_slippage() {
        let slip = SlippageModel::percentage(0.001);

        // Long entry: pay more
        let entry_slip = slip.calculate(100.0, Direction::Long, true, None);
        assert!((entry_slip - 0.1).abs() < 1e-10);

        // Long exit: receive less
        let exit_slip = slip.calculate(100.0, Direction::Long, false, None);
        assert!((exit_slip - (-0.1)).abs() < 1e-10);
    }

    #[test]
    fn test_apply_slippage() {
        let slip = SlippageModel::percentage(0.001);

        // Long entry at 100 should pay 100.1
        let entry_price = slip.apply(100.0, Direction::Long, true, None);
        assert!((entry_price - 100.1).abs() < 1e-10);

        // Long exit at 100 should receive 99.9
        let exit_price = slip.apply(100.0, Direction::Long, false, None);
        assert!((exit_price - 99.9).abs() < 1e-10);
    }

    #[test]
    fn test_no_slippage() {
        let slip = SlippageModel::None;
        let result = slip.apply(100.0, Direction::Long, true, None);
        assert!((result - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_volume_based_slippage() {
        let slip = SlippageModel::volume_based(0.1, 0.0001);

        // High volume should have lower slippage
        let high_vol = slip.calculate(100.0, Direction::Long, true, Some(100000.0));
        let low_vol = slip.calculate(100.0, Direction::Long, true, Some(1000.0));

        assert!(high_vol < low_vol);
    }
}
