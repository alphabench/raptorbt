//! Incremental drawdown tracking.

/// Drawdown tracker for incremental portfolio value updates.
#[derive(Debug, Clone)]
pub struct DrawdownTracker {
    /// Current peak value.
    peak: f64,
    /// Current drawdown value.
    current_drawdown: f64,
    /// Maximum drawdown seen.
    max_drawdown: f64,
    /// Current drawdown duration (bars since peak).
    current_duration: usize,
    /// Maximum drawdown duration.
    max_duration: usize,
    /// Value at drawdown start.
    drawdown_start_value: f64,
    /// Index at drawdown start.
    drawdown_start_idx: usize,
    /// Index at max drawdown.
    max_drawdown_idx: usize,
    /// Total count of updates.
    count: usize,
}

impl Default for DrawdownTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl DrawdownTracker {
    /// Create a new drawdown tracker.
    pub fn new() -> Self {
        Self {
            peak: 0.0,
            current_drawdown: 0.0,
            max_drawdown: 0.0,
            current_duration: 0,
            max_duration: 0,
            drawdown_start_value: 0.0,
            drawdown_start_idx: 0,
            max_drawdown_idx: 0,
            count: 0,
        }
    }

    /// Create with initial value.
    pub fn with_initial(initial_value: f64) -> Self {
        Self {
            peak: initial_value,
            current_drawdown: 0.0,
            max_drawdown: 0.0,
            current_duration: 0,
            max_duration: 0,
            drawdown_start_value: initial_value,
            drawdown_start_idx: 0,
            max_drawdown_idx: 0,
            count: 1,
        }
    }

    /// Update with new portfolio value.
    pub fn update(&mut self, value: f64) {
        self.count += 1;

        if value > self.peak {
            // New peak - reset drawdown
            self.peak = value;
            self.current_drawdown = 0.0;
            self.current_duration = 0;
            self.drawdown_start_value = value;
            self.drawdown_start_idx = self.count - 1;
        } else {
            // In drawdown
            self.current_drawdown = (self.peak - value) / self.peak;
            self.current_duration += 1;

            if self.current_drawdown > self.max_drawdown {
                self.max_drawdown = self.current_drawdown;
                self.max_drawdown_idx = self.count - 1;
            }

            if self.current_duration > self.max_duration {
                self.max_duration = self.current_duration;
            }
        }
    }

    /// Get current drawdown as percentage.
    #[inline]
    pub fn current_drawdown_pct(&self) -> f64 {
        self.current_drawdown * 100.0
    }

    /// Get maximum drawdown as percentage.
    #[inline]
    pub fn max_drawdown_pct(&self) -> f64 {
        self.max_drawdown * 100.0
    }

    /// Get maximum drawdown as fraction.
    #[inline]
    pub fn max_drawdown(&self) -> f64 {
        self.max_drawdown
    }

    /// Get current peak value.
    #[inline]
    pub fn peak(&self) -> f64 {
        self.peak
    }

    /// Get current drawdown duration.
    #[inline]
    pub fn current_duration(&self) -> usize {
        self.current_duration
    }

    /// Get maximum drawdown duration.
    #[inline]
    pub fn max_duration(&self) -> usize {
        self.max_duration
    }

    /// Check if currently in drawdown.
    #[inline]
    pub fn in_drawdown(&self) -> bool {
        self.current_drawdown > 0.0
    }

    /// Get index where max drawdown occurred.
    #[inline]
    pub fn max_drawdown_idx(&self) -> usize {
        self.max_drawdown_idx
    }

    /// Reset the tracker.
    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

/// Calculate drawdown curve from equity curve.
///
/// # Arguments
/// * `equity_curve` - Portfolio values over time
///
/// # Returns
/// Drawdown percentages at each point
pub fn calculate_drawdown_curve(equity_curve: &[f64]) -> Vec<f64> {
    let n = equity_curve.len();
    if n == 0 {
        return vec![];
    }

    let mut drawdown_curve = vec![0.0; n];
    let mut peak = equity_curve[0];

    for i in 0..n {
        if equity_curve[i] > peak {
            peak = equity_curve[i];
        }
        if peak > 0.0 {
            drawdown_curve[i] = (peak - equity_curve[i]) / peak * 100.0;
        }
    }

    drawdown_curve
}

/// Calculate maximum drawdown from equity curve.
///
/// # Arguments
/// * `equity_curve` - Portfolio values over time
///
/// # Returns
/// Maximum drawdown as percentage
pub fn max_drawdown(equity_curve: &[f64]) -> f64 {
    let dd = calculate_drawdown_curve(equity_curve);
    dd.iter().fold(0.0f64, |a, &b| a.max(b))
}

/// Calculate average drawdown from equity curve.
///
/// # Arguments
/// * `equity_curve` - Portfolio values over time
///
/// # Returns
/// Average drawdown as percentage
pub fn avg_drawdown(equity_curve: &[f64]) -> f64 {
    let dd = calculate_drawdown_curve(equity_curve);
    if dd.is_empty() {
        return 0.0;
    }
    dd.iter().sum::<f64>() / dd.len() as f64
}

/// Find drawdown periods.
///
/// # Arguments
/// * `equity_curve` - Portfolio values over time
///
/// # Returns
/// Vector of (start_idx, end_idx, max_drawdown) tuples for each drawdown period
pub fn drawdown_periods(equity_curve: &[f64]) -> Vec<(usize, usize, f64)> {
    let n = equity_curve.len();
    if n < 2 {
        return vec![];
    }

    let mut periods = Vec::new();
    let mut peak = equity_curve[0];
    let mut peak_idx = 0;
    let mut in_dd = false;
    let mut dd_start = 0;
    let mut max_dd = 0.0;

    for i in 1..n {
        if equity_curve[i] > peak {
            if in_dd {
                // End of drawdown period
                periods.push((dd_start, i - 1, max_dd));
                in_dd = false;
                max_dd = 0.0;
            }
            peak = equity_curve[i];
            peak_idx = i;
        } else if peak > 0.0 {
            let dd = (peak - equity_curve[i]) / peak * 100.0;
            if !in_dd && dd > 0.0 {
                in_dd = true;
                dd_start = peak_idx;
            }
            if dd > max_dd {
                max_dd = dd;
            }
        }
    }

    // Handle ongoing drawdown at end
    if in_dd {
        periods.push((dd_start, n - 1, max_dd));
    }

    periods
}

/// Calculate Calmar ratio.
///
/// # Arguments
/// * `total_return` - Total return as percentage
/// * `max_drawdown` - Maximum drawdown as percentage
///
/// # Returns
/// Calmar ratio
pub fn calmar_ratio(total_return: f64, max_drawdown: f64) -> f64 {
    if max_drawdown <= 0.0 {
        return if total_return > 0.0 { f64::INFINITY } else { 0.0 };
    }
    total_return / max_drawdown
}

/// Calculate Ulcer Index (root mean square of drawdowns).
///
/// # Arguments
/// * `equity_curve` - Portfolio values over time
///
/// # Returns
/// Ulcer Index
pub fn ulcer_index(equity_curve: &[f64]) -> f64 {
    let dd = calculate_drawdown_curve(equity_curve);
    if dd.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = dd.iter().map(|d| d * d).sum();
    (sum_sq / dd.len() as f64).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_tracking() {
        let mut tracker = DrawdownTracker::new();

        tracker.update(100.0);
        tracker.update(110.0);
        tracker.update(105.0); // 4.5% drawdown
        tracker.update(120.0);
        tracker.update(100.0); // 16.67% drawdown

        assert!((tracker.max_drawdown_pct() - 16.67).abs() < 0.1);
        assert!((tracker.peak() - 120.0).abs() < 1e-10);
    }

    #[test]
    fn test_drawdown_curve() {
        let equity = vec![100.0, 110.0, 105.0, 120.0, 100.0];
        let dd = calculate_drawdown_curve(&equity);

        assert_eq!(dd.len(), 5);
        assert!((dd[0] - 0.0).abs() < 1e-10);
        assert!((dd[1] - 0.0).abs() < 1e-10);
        assert!((dd[2] - 4.545).abs() < 0.1); // (110-105)/110 * 100
        assert!((dd[3] - 0.0).abs() < 1e-10);
        assert!((dd[4] - 16.67).abs() < 0.1); // (120-100)/120 * 100
    }

    #[test]
    fn test_max_drawdown() {
        let equity = vec![100.0, 120.0, 90.0, 110.0, 85.0];
        let max_dd = max_drawdown(&equity);

        // Max DD should be (120-85)/120 = 29.17%
        assert!((max_dd - 29.17).abs() < 0.1);
    }

    #[test]
    fn test_drawdown_periods() {
        let equity = vec![100.0, 110.0, 105.0, 115.0, 100.0, 120.0];
        let periods = drawdown_periods(&equity);

        // Should have 2 drawdown periods
        assert_eq!(periods.len(), 2);
    }

    #[test]
    fn test_calmar_ratio() {
        // 50% return with 10% max drawdown
        let calmar = calmar_ratio(50.0, 10.0);
        assert!((calmar - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_ulcer_index() {
        let equity = vec![100.0, 95.0, 90.0, 95.0, 100.0];
        let ui = ulcer_index(&equity);

        // Should be positive (there were drawdowns)
        assert!(ui > 0.0);
    }
}
