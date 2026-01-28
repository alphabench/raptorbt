//! Trade statistics calculation.

use crate::core::types::Trade;

/// Comprehensive trade statistics.
#[derive(Debug, Clone, Default)]
pub struct TradeStatistics {
    /// Total number of trades.
    pub total_trades: usize,
    /// Number of winning trades.
    pub winning_trades: usize,
    /// Number of losing trades.
    pub losing_trades: usize,
    /// Number of breakeven trades.
    pub breakeven_trades: usize,
    /// Win rate (as percentage).
    pub win_rate: f64,
    /// Average win amount.
    pub avg_win: f64,
    /// Average loss amount.
    pub avg_loss: f64,
    /// Largest win.
    pub largest_win: f64,
    /// Largest loss.
    pub largest_loss: f64,
    /// Total profit.
    pub total_profit: f64,
    /// Total loss.
    pub total_loss: f64,
    /// Net profit.
    pub net_profit: f64,
    /// Profit factor.
    pub profit_factor: f64,
    /// Expected value per trade.
    pub expectancy: f64,
    /// Average trade return percentage.
    pub avg_return_pct: f64,
    /// Average holding period (bars).
    pub avg_holding_period: f64,
    /// Max consecutive wins.
    pub max_consecutive_wins: usize,
    /// Max consecutive losses.
    pub max_consecutive_losses: usize,
    /// Average win/loss ratio.
    pub avg_win_loss_ratio: f64,
    /// Recovery factor (net profit / max loss).
    pub recovery_factor: f64,
    /// Payoff ratio (avg win / avg loss).
    pub payoff_ratio: f64,
}

impl TradeStatistics {
    /// Calculate statistics from a list of trades.
    pub fn from_trades(trades: &[Trade]) -> Self {
        let mut stats = Self::default();

        if trades.is_empty() {
            return stats;
        }

        stats.total_trades = trades.len();

        // Categorize trades
        for trade in trades {
            if trade.pnl > 0.0 {
                stats.winning_trades += 1;
                stats.total_profit += trade.pnl;
                if trade.pnl > stats.largest_win {
                    stats.largest_win = trade.pnl;
                }
            } else if trade.pnl < 0.0 {
                stats.losing_trades += 1;
                stats.total_loss += trade.pnl.abs();
                if trade.pnl.abs() > stats.largest_loss {
                    stats.largest_loss = trade.pnl.abs();
                }
            } else {
                stats.breakeven_trades += 1;
            }
        }

        // Calculate ratios
        stats.net_profit = stats.total_profit - stats.total_loss;

        if stats.total_trades > 0 {
            stats.win_rate = stats.winning_trades as f64 / stats.total_trades as f64 * 100.0;
        }

        if stats.winning_trades > 0 {
            stats.avg_win = stats.total_profit / stats.winning_trades as f64;
        }

        if stats.losing_trades > 0 {
            stats.avg_loss = stats.total_loss / stats.losing_trades as f64;
        }

        if stats.total_loss > 0.0 {
            stats.profit_factor = stats.total_profit / stats.total_loss;
        } else if stats.total_profit > 0.0 {
            stats.profit_factor = f64::INFINITY;
        }

        if stats.avg_loss > 0.0 {
            stats.payoff_ratio = stats.avg_win / stats.avg_loss;
        }

        // Expectancy
        if stats.total_trades > 0 {
            stats.expectancy = stats.net_profit / stats.total_trades as f64;
        }

        // Average return percentage
        if stats.total_trades > 0 {
            stats.avg_return_pct =
                trades.iter().map(|t| t.return_pct).sum::<f64>() / stats.total_trades as f64;
        }

        // Average holding period
        if stats.total_trades > 0 {
            stats.avg_holding_period = trades
                .iter()
                .map(|t| t.holding_period() as f64)
                .sum::<f64>()
                / stats.total_trades as f64;
        }

        // Consecutive wins/losses
        let (max_wins, max_losses) = calculate_consecutive(trades);
        stats.max_consecutive_wins = max_wins;
        stats.max_consecutive_losses = max_losses;

        // Recovery factor
        if stats.largest_loss > 0.0 {
            stats.recovery_factor = stats.net_profit / stats.largest_loss;
        }

        // Win/loss ratio
        if stats.losing_trades > 0 {
            stats.avg_win_loss_ratio = stats.winning_trades as f64 / stats.losing_trades as f64;
        }

        stats
    }

    /// Get summary as formatted string.
    pub fn summary(&self) -> String {
        format!(
            "Trades: {} | Win Rate: {:.1}% | Profit Factor: {:.2} | Net: {:.2}",
            self.total_trades, self.win_rate, self.profit_factor, self.net_profit
        )
    }

    /// Check if strategy is profitable.
    pub fn is_profitable(&self) -> bool {
        self.net_profit > 0.0
    }

    /// Get edge (expected value as percentage of average trade).
    pub fn edge(&self) -> f64 {
        if self.total_trades == 0 {
            return 0.0;
        }
        let avg_trade = self.net_profit / self.total_trades as f64;
        let avg_cost = (self.total_profit + self.total_loss) / self.total_trades as f64;
        if avg_cost > 0.0 {
            avg_trade / avg_cost * 100.0
        } else {
            0.0
        }
    }
}

/// Calculate maximum consecutive wins and losses.
fn calculate_consecutive(trades: &[Trade]) -> (usize, usize) {
    let mut max_wins = 0;
    let mut max_losses = 0;
    let mut current_wins = 0;
    let mut current_losses = 0;

    for trade in trades {
        if trade.pnl > 0.0 {
            current_wins += 1;
            current_losses = 0;
            max_wins = max_wins.max(current_wins);
        } else if trade.pnl < 0.0 {
            current_losses += 1;
            current_wins = 0;
            max_losses = max_losses.max(current_losses);
        }
    }

    (max_wins, max_losses)
}

/// Monthly returns breakdown.
#[derive(Debug, Clone, Default)]
pub struct MonthlyReturns {
    /// Year.
    pub year: i32,
    /// Month (1-12).
    pub month: u8,
    /// Return percentage.
    pub return_pct: f64,
    /// Number of trades.
    pub trade_count: usize,
}

/// Calculate trade statistics by exit reason.
pub fn stats_by_exit_reason(
    trades: &[Trade],
) -> std::collections::HashMap<crate::core::types::ExitReason, TradeStatistics> {
    use crate::core::types::ExitReason;
    use std::collections::HashMap;

    let mut grouped: HashMap<ExitReason, Vec<&Trade>> = HashMap::new();

    for trade in trades {
        grouped.entry(trade.exit_reason).or_default().push(trade);
    }

    grouped
        .into_iter()
        .map(|(reason, trade_refs)| {
            let owned_trades: Vec<Trade> = trade_refs.into_iter().cloned().collect();
            (reason, TradeStatistics::from_trades(&owned_trades))
        })
        .collect()
}

/// Calculate statistics for long vs short trades.
pub fn stats_by_direction(trades: &[Trade]) -> (TradeStatistics, TradeStatistics) {
    use crate::core::types::Direction;

    let long_trades: Vec<Trade> = trades
        .iter()
        .filter(|t| t.direction == Direction::Long)
        .cloned()
        .collect();

    let short_trades: Vec<Trade> = trades
        .iter()
        .filter(|t| t.direction == Direction::Short)
        .cloned()
        .collect();

    (
        TradeStatistics::from_trades(&long_trades),
        TradeStatistics::from_trades(&short_trades),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::{Direction, ExitReason};

    fn sample_trades() -> Vec<Trade> {
        vec![
            Trade {
                id: 1,
                symbol: "TEST".to_string(),
                entry_idx: 0,
                exit_idx: 5,
                entry_price: 100.0,
                exit_price: 110.0,
                size: 10.0,
                direction: Direction::Long,
                pnl: 100.0, // Win
                return_pct: 10.0,
                entry_time: 0,
                exit_time: 5,
                fees: 0.0,
                exit_reason: ExitReason::Signal,
            },
            Trade {
                id: 2,
                symbol: "TEST".to_string(),
                entry_idx: 10,
                exit_idx: 15,
                entry_price: 100.0,
                exit_price: 95.0,
                size: 10.0,
                direction: Direction::Long,
                pnl: -50.0, // Loss
                return_pct: -5.0,
                entry_time: 10,
                exit_time: 15,
                fees: 0.0,
                exit_reason: ExitReason::StopLoss,
            },
            Trade {
                id: 3,
                symbol: "TEST".to_string(),
                entry_idx: 20,
                exit_idx: 25,
                entry_price: 100.0,
                exit_price: 108.0,
                size: 10.0,
                direction: Direction::Long,
                pnl: 80.0, // Win
                return_pct: 8.0,
                entry_time: 20,
                exit_time: 25,
                fees: 0.0,
                exit_reason: ExitReason::TakeProfit,
            },
        ]
    }

    #[test]
    fn test_basic_stats() {
        let trades = sample_trades();
        let stats = TradeStatistics::from_trades(&trades);

        assert_eq!(stats.total_trades, 3);
        assert_eq!(stats.winning_trades, 2);
        assert_eq!(stats.losing_trades, 1);
        assert!((stats.win_rate - 66.67).abs() < 0.1);
    }

    #[test]
    fn test_profit_calculations() {
        let trades = sample_trades();
        let stats = TradeStatistics::from_trades(&trades);

        assert!((stats.total_profit - 180.0).abs() < 1e-10);
        assert!((stats.total_loss - 50.0).abs() < 1e-10);
        assert!((stats.net_profit - 130.0).abs() < 1e-10);
        assert!((stats.profit_factor - 3.6).abs() < 0.1);
    }

    #[test]
    fn test_consecutive() {
        let trades = sample_trades();
        let (max_wins, max_losses) = calculate_consecutive(&trades);

        // W, L, W -> max consecutive wins = 1, max consecutive losses = 1
        assert_eq!(max_wins, 1);
        assert_eq!(max_losses, 1);
    }

    #[test]
    fn test_stats_by_exit_reason() {
        let trades = sample_trades();
        let by_reason = stats_by_exit_reason(&trades);

        // Should have 3 different exit reasons
        assert!(by_reason.contains_key(&ExitReason::Signal));
        assert!(by_reason.contains_key(&ExitReason::StopLoss));
        assert!(by_reason.contains_key(&ExitReason::TakeProfit));
    }

    #[test]
    fn test_empty_trades() {
        let stats = TradeStatistics::from_trades(&[]);

        assert_eq!(stats.total_trades, 0);
        assert!((stats.win_rate - 0.0).abs() < 1e-10);
        assert!((stats.profit_factor - 0.0).abs() < 1e-10);
    }
}
