//! Streaming metrics calculation using Welford's algorithm.
//!
//! Enables single-pass calculation of mean, variance, Sharpe ratio, and Sortino ratio.

/// Streaming metrics calculator using Welford's algorithm.
///
/// Allows incremental calculation of statistics without storing all values.
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
}

impl Default for StreamingMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamingMetrics {
    /// Create a new streaming metrics calculator.
    pub fn new() -> Self {
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
