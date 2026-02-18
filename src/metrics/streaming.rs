//! Streaming metrics calculation using Welford's algorithm.
//!
//! Enables single-pass calculation of mean, variance, Sharpe ratio, and Sortino ratio.

use crate::core::types::BacktestMetrics;

/// Streaming metrics calculator using Welford's algorithm.
///
/// Allows incremental calculation of statistics without storing all values.
/// Also tracks equity and drawdown for backtesting.
#[derive(Debug, Clone)]
pub struct StreamingMetrics {
    /// Number of observations.
    count: usize,
    /// Running mean.
    mean: f64,
    /// Running M2 for variance calculation.
    m2: f64,
    /// Running M2 for downside variance (Sortino).
    m2_downside: f64,
    /// Target return for Sortino (default: 0).
    target_return: f64,
    /// Sum of returns (for total return calculation).
    sum: f64,
    /// Sum of positive returns.
    sum_positive: f64,
    /// Sum of negative returns.
    sum_negative: f64,
    /// Count of positive returns.
    count_positive: usize,
    /// Count of negative returns.
    count_negative: usize,

    // === Equity and drawdown tracking ===
    /// Initial capital.
    #[allow(dead_code)]
    initial_capital: f64,
    /// Peak equity value (for drawdown calculation).
    peak_equity: f64,
    /// Current equity value.
    current_equity: f64,
    /// Maximum drawdown percentage.
    max_drawdown_pct: f64,
    /// Current drawdown percentage.
    current_drawdown: f64,
    /// Bars since peak (for max drawdown duration).
    bars_since_peak: usize,
    /// Maximum drawdown duration in bars.
    max_drawdown_duration: usize,

    // === Trade tracking ===
    /// Number of trades.
    trade_count: usize,
    /// Number of winning trades.
    winning_trades: usize,
    /// Number of losing trades.
    losing_trades: usize,
    /// Sum of winning trade P&L.
    sum_wins: f64,
    /// Sum of losing trade P&L.
    sum_losses: f64,
    /// Sum of trade return percentages.
    sum_trade_returns: f64,
    /// Sum of squared trade return percentages (for SQN).
    sum_trade_returns_sq: f64,
    /// Best trade return percentage.
    best_trade_pct: f64,
    /// Worst trade return percentage.
    worst_trade_pct: f64,
    /// Sum of winning trade durations.
    sum_winning_duration: usize,
    /// Sum of losing trade durations.
    sum_losing_duration: usize,
    /// Current consecutive wins.
    current_consecutive_wins: usize,
    /// Current consecutive losses.
    current_consecutive_losses: usize,
    /// Maximum consecutive wins.
    max_consecutive_wins: usize,
    /// Maximum consecutive losses.
    max_consecutive_losses: usize,
    /// Total holding period (bars).
    total_holding_period: usize,
    /// Total fees paid.
    total_fees: f64,
}

impl Default for StreamingMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamingMetrics {
    /// Create a new streaming metrics calculator.
    pub fn new() -> Self {
        Self::with_initial_capital(0.0)
    }

    /// Create a new streaming metrics calculator with initial capital.
    pub fn with_initial_capital(initial_capital: f64) -> Self {
        Self {
            count: 0,
            mean: 0.0,
            m2: 0.0,
            m2_downside: 0.0,
            target_return: 0.0,
            sum: 0.0,
            sum_positive: 0.0,
            sum_negative: 0.0,
            count_positive: 0,
            count_negative: 0,
            // Equity tracking
            initial_capital,
            peak_equity: initial_capital,
            current_equity: initial_capital,
            max_drawdown_pct: 0.0,
            current_drawdown: 0.0,
            bars_since_peak: 0,
            max_drawdown_duration: 0,
            // Trade tracking
            trade_count: 0,
            winning_trades: 0,
            losing_trades: 0,
            sum_wins: 0.0,
            sum_losses: 0.0,
            sum_trade_returns: 0.0,
            sum_trade_returns_sq: 0.0,
            best_trade_pct: f64::NEG_INFINITY,
            worst_trade_pct: f64::INFINITY,
            sum_winning_duration: 0,
            sum_losing_duration: 0,
            current_consecutive_wins: 0,
            current_consecutive_losses: 0,
            max_consecutive_wins: 0,
            max_consecutive_losses: 0,
            total_holding_period: 0,
            total_fees: 0.0,
        }
    }

    /// Create with a custom target return for Sortino calculation.
    pub fn with_target_return(mut self, target: f64) -> Self {
        self.target_return = target;
        self
    }

    /// Update metrics with a new return value.
    ///
    /// Uses Welford's online algorithm for numerically stable variance calculation.
    pub fn update(&mut self, return_value: f64) {
        self.count += 1;
        self.sum += return_value;

        // Track positive/negative
        if return_value > 0.0 {
            self.sum_positive += return_value;
            self.count_positive += 1;
        } else if return_value < 0.0 {
            self.sum_negative += return_value;
            self.count_negative += 1;
        }

        // Welford's algorithm for mean and variance
        let delta = return_value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = return_value - self.mean;
        self.m2 += delta * delta2;

        // Downside variance (for Sortino)
        let downside = (return_value - self.target_return).min(0.0);
        let _delta_down = downside - (self.m2_downside / self.count.max(1) as f64).sqrt();
        self.m2_downside += downside * downside;
    }

    /// Get the number of observations.
    #[inline]
    pub fn count(&self) -> usize {
        self.count
    }

    /// Get the running mean.
    #[inline]
    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// Get the sample variance.
    pub fn variance(&self) -> f64 {
        if self.count < 2 {
            return 0.0;
        }
        self.m2 / (self.count - 1) as f64
    }

    /// Get the population variance.
    pub fn variance_population(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        self.m2 / self.count as f64
    }

    /// Get the sample standard deviation.
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Get the downside standard deviation (for Sortino).
    pub fn downside_std_dev(&self) -> f64 {
        if self.count < 2 {
            return 0.0;
        }
        (self.m2_downside / (self.count - 1) as f64).sqrt()
    }

    /// Calculate Sharpe ratio.
    ///
    /// # Arguments
    /// * `periods_per_year` - Number of periods per year (e.g., 252 for daily)
    /// * `risk_free_rate` - Annual risk-free rate (default: 0)
    ///
    /// # Returns
    /// Annualized Sharpe ratio
    pub fn sharpe_ratio(&self, periods_per_year: f64) -> f64 {
        self.sharpe_ratio_with_rf(periods_per_year, 0.0)
    }

    /// Calculate Sharpe ratio with custom risk-free rate.
    pub fn sharpe_ratio_with_rf(&self, periods_per_year: f64, risk_free_rate: f64) -> f64 {
        let std = self.std_dev();
        if std == 0.0 || self.count < 2 {
            return 0.0;
        }

        let rf_per_period = risk_free_rate / periods_per_year;
        let excess_return = self.mean - rf_per_period;
        let annualized_excess = excess_return * periods_per_year;
        let annualized_std = std * periods_per_year.sqrt();

        annualized_excess / annualized_std
    }

    /// Calculate Sortino ratio.
    ///
    /// # Arguments
    /// * `periods_per_year` - Number of periods per year (e.g., 252 for daily)
    ///
    /// # Returns
    /// Annualized Sortino ratio
    pub fn sortino_ratio(&self, periods_per_year: f64) -> f64 {
        let downside_std = self.downside_std_dev();
        if downside_std == 0.0 || self.count < 2 {
            return if self.mean > 0.0 { f64::INFINITY } else { 0.0 };
        }

        let excess_return = self.mean - self.target_return;
        let annualized_excess = excess_return * periods_per_year;
        let annualized_downside_std = downside_std * periods_per_year.sqrt();

        annualized_excess / annualized_downside_std
    }

    /// Get total return.
    pub fn total_return(&self) -> f64 {
        self.sum
    }

    /// Get average positive return.
    pub fn avg_positive_return(&self) -> f64 {
        if self.count_positive == 0 {
            return 0.0;
        }
        self.sum_positive / self.count_positive as f64
    }

    /// Get average negative return.
    pub fn avg_negative_return(&self) -> f64 {
        if self.count_negative == 0 {
            return 0.0;
        }
        self.sum_negative / self.count_negative as f64
    }

    /// Get win rate (fraction of positive returns).
    pub fn win_rate(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        self.count_positive as f64 / self.count as f64
    }

    /// Get profit factor (sum of profits / sum of losses).
    pub fn profit_factor(&self) -> f64 {
        if self.sum_negative == 0.0 {
            return if self.sum_positive > 0.0 { f64::INFINITY } else { 0.0 };
        }
        self.sum_positive / self.sum_negative.abs()
    }

    /// Get omega ratio (same as profit factor for return-based calculation).
    /// Omega = (sum of returns above threshold) / |sum of returns below threshold|
    /// With threshold = 0, this equals profit_factor.
    pub fn omega_ratio(&self) -> f64 {
        self.profit_factor()
    }

    // === Equity tracking methods ===

    /// Update equity and calculate drawdown.
    pub fn update_equity(&mut self, equity: f64) {
        self.current_equity = equity;

        if equity > self.peak_equity {
            self.peak_equity = equity;
            self.bars_since_peak = 0;
        } else {
            self.bars_since_peak += 1;
            if self.bars_since_peak > self.max_drawdown_duration {
                self.max_drawdown_duration = self.bars_since_peak;
            }
        }

        // Calculate current drawdown percentage
        if self.peak_equity > 0.0 {
            self.current_drawdown = (self.peak_equity - equity) / self.peak_equity * 100.0;
            if self.current_drawdown > self.max_drawdown_pct {
                self.max_drawdown_pct = self.current_drawdown;
            }
        }
    }

    /// Get current drawdown percentage.
    #[inline]
    pub fn current_drawdown_pct(&self) -> f64 {
        self.current_drawdown
    }

    /// Get maximum drawdown percentage.
    #[inline]
    pub fn max_drawdown_pct(&self) -> f64 {
        self.max_drawdown_pct
    }

    // === Trade tracking methods ===

    /// Record a completed trade.
    ///
    /// # Arguments
    /// * `pnl` - Trade profit/loss
    /// * `return_pct` - Trade return percentage
    /// * `duration` - Trade duration in bars
    pub fn record_trade(&mut self, pnl: f64, return_pct: f64, duration: usize) {
        self.trade_count += 1;
        self.sum_trade_returns += return_pct;
        self.sum_trade_returns_sq += return_pct * return_pct;
        self.total_holding_period += duration;

        // Track best/worst trades
        if return_pct > self.best_trade_pct {
            self.best_trade_pct = return_pct;
        }
        if return_pct < self.worst_trade_pct {
            self.worst_trade_pct = return_pct;
        }

        if pnl > 0.0 {
            self.winning_trades += 1;
            self.sum_wins += pnl;
            self.sum_winning_duration += duration;
            self.current_consecutive_wins += 1;
            self.current_consecutive_losses = 0;
            if self.current_consecutive_wins > self.max_consecutive_wins {
                self.max_consecutive_wins = self.current_consecutive_wins;
            }
        } else if pnl < 0.0 {
            self.losing_trades += 1;
            self.sum_losses += pnl.abs();
            self.sum_losing_duration += duration;
            self.current_consecutive_losses += 1;
            self.current_consecutive_wins = 0;
            if self.current_consecutive_losses > self.max_consecutive_losses {
                self.max_consecutive_losses = self.current_consecutive_losses;
            }
        }
    }

    /// Record fees paid.
    pub fn record_fees(&mut self, fees: f64) {
        self.total_fees += fees;
    }

    /// Finalize metrics and produce BacktestMetrics.
    ///
    /// # Arguments
    /// * `initial_capital` - Starting capital
    /// * `final_value` - Ending portfolio value
    /// * `returns` - Array of period returns for ratio calculations
    pub fn finalize(
        &self,
        initial_capital: f64,
        final_value: f64,
        returns: &[f64],
    ) -> BacktestMetrics {
        // Calculate return metrics from the returns array
        let mut return_metrics = StreamingMetrics::new();
        for &r in returns {
            if !r.is_nan() {
                return_metrics.update(r);
            }
        }

        let total_return_pct = if initial_capital > 0.0 {
            (final_value - initial_capital) / initial_capital * 100.0
        } else {
            0.0
        };

        // Calculate trade-based metrics
        let win_rate_pct = if self.trade_count > 0 {
            self.winning_trades as f64 / self.trade_count as f64 * 100.0
        } else {
            0.0
        };

        let profit_factor = if self.sum_losses > 0.0 {
            self.sum_wins / self.sum_losses
        } else if self.sum_wins > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };

        let avg_trade_return_pct = if self.trade_count > 0 {
            self.sum_trade_returns / self.trade_count as f64
        } else {
            0.0
        };

        let avg_win_pct = if self.winning_trades > 0 {
            self.sum_wins / self.winning_trades as f64 / initial_capital * 100.0
        } else {
            0.0
        };

        let avg_loss_pct = if self.losing_trades > 0 {
            -(self.sum_losses / self.losing_trades as f64 / initial_capital * 100.0)
        } else {
            0.0
        };

        let avg_winning_duration = if self.winning_trades > 0 {
            self.sum_winning_duration as f64 / self.winning_trades as f64
        } else {
            0.0
        };

        let avg_losing_duration = if self.losing_trades > 0 {
            self.sum_losing_duration as f64 / self.losing_trades as f64
        } else {
            0.0
        };

        let avg_holding_period = if self.trade_count > 0 {
            self.total_holding_period as f64 / self.trade_count as f64
        } else {
            0.0
        };

        // Expectancy: average profit per trade
        let expectancy = if self.trade_count > 0 {
            (self.sum_wins - self.sum_losses) / self.trade_count as f64
        } else {
            0.0
        };

        // SQN (System Quality Number)
        let sqn = if self.trade_count > 1 {
            let mean_return = self.sum_trade_returns / self.trade_count as f64;
            let variance =
                (self.sum_trade_returns_sq / self.trade_count as f64) - (mean_return * mean_return);
            let std_dev = variance.max(0.0).sqrt();
            if std_dev > 0.0 {
                (mean_return / std_dev) * (self.trade_count as f64).sqrt()
            } else {
                0.0
            }
        } else {
            0.0
        };

        // Sharpe ratio (annualized, assuming 252 trading days)
        let sharpe_ratio = return_metrics.sharpe_ratio(252.0);

        // Sortino ratio (annualized)
        let sortino_ratio = return_metrics.sortino_ratio(252.0);

        // Calmar ratio (annualized return / max drawdown)
        let calmar_ratio = if self.max_drawdown_pct > 0.0 {
            total_return_pct / self.max_drawdown_pct
        } else if total_return_pct > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };

        // Omega ratio
        let omega_ratio = return_metrics.omega_ratio();

        // Best/worst trade handling (handle edge cases)
        let best_trade_pct =
            if self.best_trade_pct == f64::NEG_INFINITY { 0.0 } else { self.best_trade_pct };
        let worst_trade_pct =
            if self.worst_trade_pct == f64::INFINITY { 0.0 } else { self.worst_trade_pct };

        // Payoff ratio: average win / average loss (absolute value)
        let payoff_ratio = if avg_loss_pct.abs() > 0.0 {
            avg_win_pct / avg_loss_pct.abs()
        } else if avg_win_pct > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };

        // Recovery factor: net profit / max drawdown (absolute value)
        let net_profit = final_value - initial_capital;
        let recovery_factor = if self.max_drawdown_pct > 0.0 && initial_capital > 0.0 {
            let max_dd_absolute = self.max_drawdown_pct / 100.0 * initial_capital;
            if max_dd_absolute > 0.0 {
                net_profit / max_dd_absolute
            } else {
                0.0
            }
        } else if net_profit > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };

        BacktestMetrics {
            total_return_pct,
            sharpe_ratio,
            sortino_ratio,
            calmar_ratio,
            omega_ratio,
            max_drawdown_pct: self.max_drawdown_pct,
            max_drawdown_duration: self.max_drawdown_duration,
            win_rate_pct,
            profit_factor,
            expectancy,
            sqn,
            total_trades: self.trade_count,
            total_closed_trades: self.trade_count,
            total_open_trades: 0,
            open_trade_pnl: 0.0,
            winning_trades: self.winning_trades,
            losing_trades: self.losing_trades,
            start_value: initial_capital,
            end_value: final_value,
            total_fees_paid: self.total_fees,
            best_trade_pct,
            worst_trade_pct,
            avg_trade_return_pct,
            avg_win_pct,
            avg_loss_pct,
            avg_winning_duration,
            avg_losing_duration,
            max_consecutive_wins: self.max_consecutive_wins,
            max_consecutive_losses: self.max_consecutive_losses,
            avg_holding_period,
            exposure_pct: 0.0, // TODO: calculate based on time in market
            payoff_ratio,
            recovery_factor,
        }
    }

    /// Reset all metrics.
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Merge two streaming metrics (for parallel computation).
    pub fn merge(&mut self, other: &StreamingMetrics) {
        if other.count == 0 {
            return;
        }
        if self.count == 0 {
            *self = other.clone();
            return;
        }

        let combined_count = self.count + other.count;
        let delta = other.mean - self.mean;

        // Merge means
        let combined_mean = self.mean + delta * other.count as f64 / combined_count as f64;

        // Merge M2 (parallel variance)
        let combined_m2 = self.m2
            + other.m2
            + delta * delta * self.count as f64 * other.count as f64 / combined_count as f64;

        // Update state
        self.count = combined_count;
        self.mean = combined_mean;
        self.m2 = combined_m2;
        self.sum += other.sum;
        self.sum_positive += other.sum_positive;
        self.sum_negative += other.sum_negative;
        self.count_positive += other.count_positive;
        self.count_negative += other.count_negative;
        self.m2_downside += other.m2_downside; // Approximation
    }
}

/// Calculate Sharpe ratio from a slice of returns.
pub fn sharpe_ratio(returns: &[f64], periods_per_year: f64, risk_free_rate: f64) -> f64 {
    let mut metrics = StreamingMetrics::new();
    for &r in returns {
        if !r.is_nan() {
            metrics.update(r);
        }
    }
    metrics.sharpe_ratio_with_rf(periods_per_year, risk_free_rate)
}

/// Calculate Sortino ratio from a slice of returns.
pub fn sortino_ratio(returns: &[f64], periods_per_year: f64) -> f64 {
    let mut metrics = StreamingMetrics::new();
    for &r in returns {
        if !r.is_nan() {
            metrics.update(r);
        }
    }
    metrics.sortino_ratio(periods_per_year)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_statistics() {
        let mut metrics = StreamingMetrics::new();
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        for v in &values {
            metrics.update(*v);
        }

        assert_eq!(metrics.count(), 5);
        assert!((metrics.mean() - 3.0).abs() < 1e-10);

        // Sample variance of [1,2,3,4,5] = 2.5
        assert!((metrics.variance() - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_welford_numerical_stability() {
        let mut metrics = StreamingMetrics::new();

        // Large values that might cause numerical issues with naive algorithm
        let base = 1e10;
        let values = vec![base + 1.0, base + 2.0, base + 3.0];

        for v in &values {
            metrics.update(*v);
        }

        // Mean should be base + 2
        assert!((metrics.mean() - (base + 2.0)).abs() < 1e-5);

        // Variance should be 1.0 (same as [1, 2, 3])
        assert!((metrics.variance() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_sharpe_ratio() {
        let mut metrics = StreamingMetrics::new();

        // Daily returns: 1%, 2%, -1%, 1.5%, 0.5%
        let returns = vec![0.01, 0.02, -0.01, 0.015, 0.005];

        for r in &returns {
            metrics.update(*r);
        }

        // Should produce a positive Sharpe ratio
        let sharpe = metrics.sharpe_ratio(252.0);
        assert!(sharpe > 0.0);
    }

    #[test]
    fn test_sortino_ratio() {
        let mut metrics = StreamingMetrics::new();

        // Mix of positive and negative returns
        let returns = vec![0.02, -0.01, 0.03, -0.02, 0.01];

        for r in &returns {
            metrics.update(*r);
        }

        // Sortino should be different from Sharpe
        let sharpe = metrics.sharpe_ratio(252.0);
        let sortino = metrics.sortino_ratio(252.0);

        // With negative returns, Sortino penalizes only downside
        assert!(sortino != sharpe);
    }

    #[test]
    fn test_win_rate_and_profit_factor() {
        let mut metrics = StreamingMetrics::new();

        // 3 wins, 2 losses
        let returns = vec![0.02, -0.01, 0.03, -0.02, 0.01];

        for r in &returns {
            metrics.update(*r);
        }

        // Win rate should be 60%
        assert!((metrics.win_rate() - 0.6).abs() < 1e-10);

        // Profit factor = 0.06 / 0.03 = 2.0
        assert!((metrics.profit_factor() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_merge() {
        let mut m1 = StreamingMetrics::new();
        let mut m2 = StreamingMetrics::new();

        // Split data between two calculators
        for v in &[1.0, 2.0, 3.0] {
            m1.update(*v);
        }
        for v in &[4.0, 5.0] {
            m2.update(*v);
        }

        // Merge
        m1.merge(&m2);

        // Should match single calculator with all data
        let mut combined = StreamingMetrics::new();
        for v in &[1.0, 2.0, 3.0, 4.0, 5.0] {
            combined.update(*v);
        }

        assert_eq!(m1.count(), combined.count());
        assert!((m1.mean() - combined.mean()).abs() < 1e-10);
        assert!((m1.variance() - combined.variance()).abs() < 1e-10);
    }
}
