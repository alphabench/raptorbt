//! Position tracking for portfolio management.

use crate::core::types::{Direction, ExitReason, Position, Price, Timestamp, Trade};

/// Position manager for tracking open positions.
#[derive(Debug, Clone)]
pub struct PositionManager {
    /// Current position state.
    pub position: Position,
    /// Trade counter for generating unique IDs.
    trade_counter: u64,
    /// Symbol being traded.
    pub symbol: String,
}

impl PositionManager {
    /// Create a new position manager.
    pub fn new(symbol: String) -> Self {
        Self {
            position: Position::new(),
            trade_counter: 0,
            symbol,
        }
    }

    /// Check if currently in a position.
    #[inline]
    pub fn is_in_position(&self) -> bool {
        self.position.is_open
    }

    /// Get current position direction.
    pub fn current_direction(&self) -> Option<Direction> {
        if self.position.is_open {
            Some(self.position.direction)
        } else {
            None
        }
    }

    /// Open a new position.
    ///
    /// # Arguments
    /// * `idx` - Bar index
    /// * `timestamp` - Entry timestamp
    /// * `price` - Entry price
    /// * `size` - Position size
    /// * `direction` - Trade direction
    /// * `stop_price` - Optional stop-loss price
    /// * `target_price` - Optional take-profit price
    /// * `entry_fees` - Entry fees (to track for PnL calculation)
    ///
    /// # Returns
    /// True if position was opened, false if already in position
    pub fn open_position(
        &mut self,
        idx: usize,
        _timestamp: Timestamp,
        price: Price,
        size: f64,
        direction: Direction,
        stop_price: Option<Price>,
        target_price: Option<Price>,
        entry_fees: f64,
    ) -> bool {
        if self.position.is_open {
            return false;
        }

        self.position.open(
            idx,
            price,
            size,
            direction,
            stop_price,
            target_price,
            entry_fees,
        );
        true
    }

    /// Close current position and generate a trade record.
    ///
    /// # Arguments
    /// * `idx` - Bar index
    /// * `timestamp` - Exit timestamp
    /// * `price` - Exit price
    /// * `entry_timestamp` - Entry timestamp (for trade record)
    /// * `exit_reason` - Reason for exit
    /// * `fees` - Transaction fees
    ///
    /// # Returns
    /// Trade record if position was closed, None if no position
    pub fn close_position(
        &mut self,
        idx: usize,
        timestamp: Timestamp,
        price: Price,
        entry_timestamp: Timestamp,
        exit_reason: ExitReason,
        fees: f64,
    ) -> Option<Trade> {
        if !self.position.is_open {
            return None;
        }

        let trade = self.create_trade(idx, timestamp, price, entry_timestamp, exit_reason, fees);
        self.position.close();
        self.trade_counter += 1;

        Some(trade)
    }

    /// Create a trade record from current position.
    fn create_trade(
        &self,
        exit_idx: usize,
        exit_timestamp: Timestamp,
        exit_price: Price,
        entry_timestamp: Timestamp,
        exit_reason: ExitReason,
        exit_fees: f64,
    ) -> Trade {
        let pos = &self.position;
        let multiplier = pos.direction.multiplier();

        // Calculate P&L (matching VectorBT: gross - entry_fees - exit_fees)
        let gross_pnl = (exit_price - pos.entry_price) * pos.size * multiplier;
        let total_fees = pos.entry_fees + exit_fees;
        let pnl = gross_pnl - total_fees;

        // Calculate return percentage
        let cost_basis = pos.entry_price * pos.size;
        let return_pct = if cost_basis > 0.0 {
            pnl / cost_basis * 100.0
        } else {
            0.0
        };

        Trade {
            id: self.trade_counter,
            symbol: self.symbol.clone(),
            entry_idx: pos.entry_idx,
            exit_idx,
            entry_price: pos.entry_price,
            exit_price,
            size: pos.size,
            direction: pos.direction,
            pnl,
            return_pct,
            entry_time: entry_timestamp,
            exit_time: exit_timestamp,
            fees: total_fees,
            exit_reason,
        }
    }

    /// Update position with new price data (for trailing stops).
    ///
    /// # Arguments
    /// * `high` - Current bar high
    /// * `low` - Current bar low
    pub fn update_price(&mut self, high: Price, low: Price) {
        if self.position.is_open {
            self.position.update_extremes(high, low);
        }
    }

    /// Calculate unrealized P&L at current price.
    pub fn unrealized_pnl(&self, current_price: Price) -> f64 {
        self.position.unrealized_pnl(current_price)
    }

    /// Get current position value (market value of position).
    pub fn position_value(&self, current_price: Price) -> f64 {
        if !self.position.is_open {
            return 0.0;
        }
        current_price * self.position.size
    }

    /// Calculate position exposure (notional value as fraction of given capital).
    pub fn exposure(&self, current_price: Price, capital: f64) -> f64 {
        if capital <= 0.0 {
            return 0.0;
        }
        self.position_value(current_price) / capital
    }

    /// Check if stop-loss is hit.
    pub fn is_stop_hit(&self, low: Price, high: Price) -> bool {
        if !self.position.is_open {
            return false;
        }

        if let Some(stop) = self.position.stop_price {
            match self.position.direction {
                Direction::Long => low <= stop,
                Direction::Short => high >= stop,
            }
        } else {
            false
        }
    }

    /// Check if take-profit is hit.
    pub fn is_target_hit(&self, low: Price, high: Price) -> bool {
        if !self.position.is_open {
            return false;
        }

        if let Some(target) = self.position.target_price {
            match self.position.direction {
                Direction::Long => high >= target,
                Direction::Short => low <= target,
            }
        } else {
            false
        }
    }

    /// Update trailing stop.
    ///
    /// # Arguments
    /// * `trail_percent` - Trailing stop percentage
    pub fn update_trailing_stop(&mut self, trail_percent: f64) {
        if !self.position.is_open {
            return;
        }

        match self.position.direction {
            Direction::Long => {
                // Trail below highest price since entry
                let new_stop = self.position.highest_since_entry * (1.0 - trail_percent);
                if let Some(current_stop) = self.position.stop_price {
                    if new_stop > current_stop {
                        self.position.stop_price = Some(new_stop);
                    }
                } else {
                    self.position.stop_price = Some(new_stop);
                }
            }
            Direction::Short => {
                // Trail above lowest price since entry
                let new_stop = self.position.lowest_since_entry * (1.0 + trail_percent);
                if let Some(current_stop) = self.position.stop_price {
                    if new_stop < current_stop {
                        self.position.stop_price = Some(new_stop);
                    }
                } else {
                    self.position.stop_price = Some(new_stop);
                }
            }
        }
    }

    /// Reset position manager for new backtest.
    pub fn reset(&mut self) {
        self.position = Position::new();
        self.trade_counter = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_open_close_position() {
        let mut pm = PositionManager::new("TEST".to_string());

        // Open position
        assert!(pm.open_position(0, 1000, 100.0, 10.0, Direction::Long, None, None));
        assert!(pm.is_in_position());

        // Try to open another - should fail
        assert!(!pm.open_position(1, 1001, 101.0, 10.0, Direction::Long, None, None));

        // Close position with profit
        let trade = pm
            .close_position(5, 1005, 110.0, 1000, ExitReason::Signal, 2.0)
            .unwrap();

        assert!(!pm.is_in_position());
        assert_eq!(trade.entry_idx, 0);
        assert_eq!(trade.exit_idx, 5);
        assert!((trade.entry_price - 100.0).abs() < 1e-10);
        assert!((trade.exit_price - 110.0).abs() < 1e-10);

        // P&L: (110 - 100) * 10 - 2 = 98
        assert!((trade.pnl - 98.0).abs() < 1e-10);
    }

    #[test]
    fn test_short_position() {
        let mut pm = PositionManager::new("TEST".to_string());

        pm.open_position(0, 1000, 100.0, 10.0, Direction::Short, None, None);

        // Close with profit (price went down)
        let trade = pm
            .close_position(5, 1005, 90.0, 1000, ExitReason::Signal, 2.0)
            .unwrap();

        // P&L: (100 - 90) * 10 * -(-1) - 2 = 98
        // For short: (entry - exit) * size = (100 - 90) * 10 = 100 gross, minus 2 fees = 98
        assert!((trade.pnl - 98.0).abs() < 1e-10);
    }

    #[test]
    fn test_stop_loss() {
        let mut pm = PositionManager::new("TEST".to_string());

        pm.open_position(
            0,
            1000,
            100.0,
            10.0,
            Direction::Long,
            Some(95.0), // Stop at 95
            None,
        );

        // Check stop not hit
        assert!(!pm.is_stop_hit(96.0, 102.0));

        // Check stop hit
        assert!(pm.is_stop_hit(94.0, 102.0));
    }

    #[test]
    fn test_trailing_stop() {
        let mut pm = PositionManager::new("TEST".to_string());

        pm.open_position(0, 1000, 100.0, 10.0, Direction::Long, None, None);

        // Update with higher price
        pm.update_price(110.0, 98.0);
        pm.update_trailing_stop(0.05); // 5% trail

        // Stop should be at 110 * 0.95 = 104.5
        assert!((pm.position.stop_price.unwrap() - 104.5).abs() < 1e-10);

        // Update with even higher price
        pm.update_price(120.0, 108.0);
        pm.update_trailing_stop(0.05);

        // Stop should move up to 120 * 0.95 = 114
        assert!((pm.position.stop_price.unwrap() - 114.0).abs() < 1e-10);
    }

    #[test]
    fn test_unrealized_pnl() {
        let mut pm = PositionManager::new("TEST".to_string());

        pm.open_position(0, 1000, 100.0, 10.0, Direction::Long, None, None);

        // Price up
        let pnl = pm.unrealized_pnl(110.0);
        assert!((pnl - 100.0).abs() < 1e-10); // (110 - 100) * 10 = 100

        // Price down
        let pnl = pm.unrealized_pnl(95.0);
        assert!((pnl - (-50.0)).abs() < 1e-10); // (95 - 100) * 10 = -50
    }
}
