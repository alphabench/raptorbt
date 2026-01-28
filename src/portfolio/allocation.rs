//! Capital allocation strategies for portfolio management.

/// Allocation strategy for distributing capital across instruments.
#[derive(Debug, Clone)]
pub enum AllocationStrategy {
    /// Equal weight across all instruments.
    EqualWeight,
    /// Fixed weight for each instrument.
    FixedWeight(Vec<f64>),
    /// Volatility-based weighting (inverse volatility).
    InverseVolatility,
    /// Risk parity (equal risk contribution).
    RiskParity,
    /// Maximum weight per instrument.
    MaxWeight(f64),
    /// Custom weights.
    Custom(Vec<(String, f64)>),
}

impl Default for AllocationStrategy {
    fn default() -> Self {
        AllocationStrategy::EqualWeight
    }
}

/// Capital allocator for managing position sizing and capital distribution.
#[derive(Debug, Clone)]
pub struct CapitalAllocator {
    /// Total capital.
    pub total_capital: f64,
    /// Available capital (not in positions).
    pub available_capital: f64,
    /// Allocation strategy.
    pub strategy: AllocationStrategy,
    /// Maximum position size as fraction of capital.
    pub max_position_size: f64,
    /// Minimum position size (absolute).
    pub min_position_size: f64,
    /// Reserve capital fraction (never allocate).
    pub reserve_fraction: f64,
}

impl CapitalAllocator {
    /// Create a new capital allocator.
    pub fn new(total_capital: f64) -> Self {
        Self {
            total_capital,
            available_capital: total_capital,
            strategy: AllocationStrategy::EqualWeight,
            max_position_size: 1.0,
            min_position_size: 0.0,
            reserve_fraction: 0.0,
        }
    }

    /// Set allocation strategy.
    pub fn with_strategy(mut self, strategy: AllocationStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set maximum position size.
    pub fn with_max_position(mut self, max_fraction: f64) -> Self {
        self.max_position_size = max_fraction.clamp(0.0, 1.0);
        self
    }

    /// Set reserve fraction.
    pub fn with_reserve(mut self, reserve: f64) -> Self {
        self.reserve_fraction = reserve.clamp(0.0, 1.0);
        self
    }

    /// Calculate position size for a single instrument.
    ///
    /// # Arguments
    /// * `price` - Entry price
    /// * `num_instruments` - Total number of instruments in portfolio
    /// * `instrument_weight` - Optional custom weight for this instrument
    ///
    /// # Returns
    /// Position size in shares/contracts
    pub fn calculate_position_size(
        &self,
        price: f64,
        num_instruments: usize,
        instrument_weight: Option<f64>,
    ) -> f64 {
        if price <= 0.0 || num_instruments == 0 {
            return 0.0;
        }

        // Calculate allocatable capital
        let allocatable = self.available_capital * (1.0 - self.reserve_fraction);

        // Calculate weight
        let weight = match &self.strategy {
            AllocationStrategy::EqualWeight => 1.0 / num_instruments as f64,
            AllocationStrategy::FixedWeight(weights) => {
                if weights.is_empty() {
                    1.0 / num_instruments as f64
                } else {
                    weights[0].min(self.max_position_size)
                }
            }
            AllocationStrategy::MaxWeight(max) => (*max).min(1.0 / num_instruments as f64),
            _ => instrument_weight.unwrap_or(1.0 / num_instruments as f64),
        };

        // Calculate allocation
        let allocation = allocatable * weight.min(self.max_position_size);

        // Convert to shares
        let shares = allocation / price;

        // Apply minimum size constraint
        if shares * price < self.min_position_size {
            return 0.0;
        }

        shares
    }

    /// Calculate position sizes for multiple instruments.
    ///
    /// # Arguments
    /// * `prices` - Entry prices for each instrument
    /// * `weights` - Optional weights for each instrument
    ///
    /// # Returns
    /// Position sizes for each instrument
    pub fn calculate_portfolio_sizes(&self, prices: &[f64], weights: Option<&[f64]>) -> Vec<f64> {
        let n = prices.len();
        if n == 0 {
            return vec![];
        }

        let allocatable = self.available_capital * (1.0 - self.reserve_fraction);

        // Get weights
        let instrument_weights: Vec<f64> = match &self.strategy {
            AllocationStrategy::EqualWeight => vec![1.0 / n as f64; n],
            AllocationStrategy::FixedWeight(w) => {
                if w.len() == n {
                    w.clone()
                } else {
                    vec![1.0 / n as f64; n]
                }
            }
            AllocationStrategy::MaxWeight(max) => {
                let equal = 1.0 / n as f64;
                vec![equal.min(*max); n]
            }
            _ => weights
                .map(|w| w.to_vec())
                .unwrap_or_else(|| vec![1.0 / n as f64; n]),
        };

        // Normalize weights
        let total_weight: f64 = instrument_weights.iter().sum();
        let normalized_weights: Vec<f64> = if total_weight > 0.0 {
            instrument_weights
                .iter()
                .map(|w| w / total_weight)
                .collect()
        } else {
            vec![1.0 / n as f64; n]
        };

        // Calculate sizes
        prices
            .iter()
            .zip(normalized_weights.iter())
            .map(|(&price, &weight)| {
                if price <= 0.0 {
                    return 0.0;
                }
                let allocation = allocatable * weight.min(self.max_position_size);
                let shares = allocation / price;
                if shares * price < self.min_position_size {
                    0.0
                } else {
                    shares
                }
            })
            .collect()
    }

    /// Calculate volatility-adjusted position size.
    ///
    /// # Arguments
    /// * `price` - Entry price
    /// * `volatility` - Instrument volatility (e.g., ATR)
    /// * `risk_per_trade` - Risk per trade as fraction of capital
    ///
    /// # Returns
    /// Position size
    pub fn calculate_volatility_sized(
        &self,
        price: f64,
        volatility: f64,
        risk_per_trade: f64,
    ) -> f64 {
        if price <= 0.0 || volatility <= 0.0 {
            return 0.0;
        }

        let risk_amount = self.available_capital * risk_per_trade;
        let size = risk_amount / volatility;

        // Apply maximum constraint
        let max_allocation = self.available_capital * self.max_position_size;
        let max_shares = max_allocation / price;

        size.min(max_shares)
    }

    /// Allocate capital to a position.
    ///
    /// # Arguments
    /// * `amount` - Amount to allocate
    ///
    /// # Returns
    /// True if allocation succeeded
    pub fn allocate(&mut self, amount: f64) -> bool {
        if amount > self.available_capital {
            return false;
        }
        self.available_capital -= amount;
        true
    }

    /// Release capital from a closed position.
    ///
    /// # Arguments
    /// * `amount` - Amount to release (including P&L)
    pub fn release(&mut self, amount: f64) {
        self.available_capital += amount;
    }

    /// Update total capital (e.g., after deposit/withdrawal or daily mark-to-market).
    pub fn update_capital(&mut self, new_capital: f64) {
        let diff = new_capital - self.total_capital;
        self.total_capital = new_capital;
        self.available_capital += diff;
    }

    /// Get current utilization rate.
    pub fn utilization(&self) -> f64 {
        if self.total_capital <= 0.0 {
            return 0.0;
        }
        1.0 - (self.available_capital / self.total_capital)
    }

    /// Reset allocator to initial state.
    pub fn reset(&mut self) {
        self.available_capital = self.total_capital;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_equal_weight() {
        let allocator = CapitalAllocator::new(100_000.0);

        // 4 instruments, equal weight = 25% each
        let size = allocator.calculate_position_size(100.0, 4, None);

        // Expected: 100000 * 0.25 / 100 = 250 shares
        assert!((size - 250.0).abs() < 1e-10);
    }

    #[test]
    fn test_max_position() {
        let allocator = CapitalAllocator::new(100_000.0).with_max_position(0.1);

        // Even with 1 instrument, max is 10%
        let size = allocator.calculate_position_size(100.0, 1, None);

        // Expected: 100000 * 0.1 / 100 = 100 shares
        assert!((size - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_portfolio_sizes() {
        let allocator = CapitalAllocator::new(100_000.0);

        let prices = vec![100.0, 50.0, 200.0];
        let sizes = allocator.calculate_portfolio_sizes(&prices, None);

        assert_eq!(sizes.len(), 3);

        // Equal weight, each gets 1/3 of capital
        // Instrument 1: 33333 / 100 = 333.33
        // Instrument 2: 33333 / 50 = 666.66
        // Instrument 3: 33333 / 200 = 166.66
        assert!((sizes[0] - 333.33).abs() < 1.0);
        assert!((sizes[1] - 666.66).abs() < 1.0);
        assert!((sizes[2] - 166.66).abs() < 1.0);
    }

    #[test]
    fn test_allocate_release() {
        let mut allocator = CapitalAllocator::new(100_000.0);

        // Allocate 30000
        assert!(allocator.allocate(30_000.0));
        assert!((allocator.available_capital - 70_000.0).abs() < 1e-10);

        // Try to allocate more than available
        assert!(!allocator.allocate(80_000.0));

        // Release with profit
        allocator.release(35_000.0);
        assert!((allocator.available_capital - 105_000.0).abs() < 1e-10);
    }

    #[test]
    fn test_utilization() {
        let mut allocator = CapitalAllocator::new(100_000.0);

        assert!((allocator.utilization() - 0.0).abs() < 1e-10);

        allocator.allocate(50_000.0);
        assert!((allocator.utilization() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_volatility_sizing() {
        let allocator = CapitalAllocator::new(100_000.0).with_max_position(0.2);

        // Risk 1% per trade with ATR of 2
        let size = allocator.calculate_volatility_sized(100.0, 2.0, 0.01);

        // Risk amount: 100000 * 0.01 = 1000
        // Size: 1000 / 2 = 500 shares
        // Max: 100000 * 0.2 / 100 = 200 shares
        // Should be capped at max
        assert!((size - 200.0).abs() < 1e-10);
    }
}
