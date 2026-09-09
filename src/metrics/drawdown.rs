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

    for (i, &equity) in equity_curve.iter().enumerate().take(n).skip(1) {
        if equity > peak {
            if in_dd {
                // End of drawdown period
                periods.push((dd_start, i - 1, max_dd));
                in_dd = false;
                max_dd = 0.0;
            }
            peak = equity;
            peak_idx = i;
        } else if peak > 0.0 {
            let dd = (peak - equity) / peak * 100.0;
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
    fn calmar_does_not_annualize_a_short_window() {
        // Measured 2026-08-30 on a real 5.27-day options backtest: total return
        // 15.4971%, max drawdown 18.6952%. The portfolio engine used to annualize
        // this into a CAGR before dividing, reporting Calmar = 115_906.80 -- an
        // artifact of the window length, not a property of the strategy. The
        // plain ratio is 0.83, and the same number must come out no matter how
        // short the run is, because this function is given no notion of time.
        let calmar = calmar_ratio(15.4971, 18.6952);
        assert!(
            (calmar - 0.8289).abs() < 1e-4,
            "expected the un-annualized ratio ~0.8289, got {calmar}"
        );
        assert!(calmar < 5.0, "a plausible Calmar cannot be in the thousands");
    }

    #[test]
    fn an_average_over_an_empty_population_is_undefined() {
        // Not a drawdown test, but the same rule, and this module is where the
        // "undefined is not zero" convention is pinned.
        //
        // Measured 2026-08-30: a two-leg straddle closed two winning trades and
        // no losing ones, and the stored row read avg_losing_duration = 0.00
        // and avg_loss_pct = 0.00 -- figures describing trades that do not
        // exist. BacktestMetrics now carries Option for all four, so an empty
        // population reports "cannot say" instead of a number.
        let no_trades: Vec<f64> = Vec::new();
        let mean = |xs: &[f64]| -> Option<f64> {
            if xs.is_empty() {
                None
            } else {
                Some(xs.iter().sum::<f64>() / xs.len() as f64)
            }
        };
        assert_eq!(mean(&no_trades), None, "an average of nothing is not zero");
        assert_eq!(mean(&[2.0, 4.0]), Some(3.0), "a real population still averages");
    }

    #[test]
    fn calmar_is_undefined_rather_than_zero_without_drawdown() {
        // A profitable run that never drew down has no defined Calmar. Returning
        // 0.0 (as three of the strategy runners used to) reads as "terrible",
        // which is the opposite of the truth; INFINITY maps to None at the
        // Python boundary via `finite()`.
        assert!(calmar_ratio(12.0, 0.0).is_infinite());
        // No return and no drawdown is genuinely zero, not undefined.
        assert_eq!(calmar_ratio(0.0, 0.0), 0.0);
        assert_eq!(calmar_ratio(-5.0, 0.0), 0.0);
    }

    #[test]
    fn ulcer_index_matches_a_hand_computed_curve() {
        // Drawdowns: 0, 5, 10, 5, 0 percent. Mean of squares = 150/5 = 30.
        let equity = vec![100.0, 95.0, 90.0, 95.0, 100.0];
        assert!((ulcer_index(&equity) - 30.0f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn ulcer_index_divides_by_the_full_length_including_the_zeros() {
        // Two of four samples are at the high-water mark. Dividing by the
        // underwater count instead of the length would give sqrt(50) here
        // rather than sqrt(25) -- a ~1.41x overstatement on this curve, and
        // roughly 2x on a curve that is mostly flat.
        let equity = vec![100.0, 100.0, 90.0, 100.0];
        // dd = [0, 0, 10, 0]; mean of squares = 100/4 = 25.
        assert!((ulcer_index(&equity) - 5.0).abs() < 1e-12);
    }

    #[test]
    fn ulcer_index_punishes_a_long_shallow_drawdown_more_than_a_brief_deep_one() {
        // The reason this metric exists. Both curves bottom at -10%, so
        // max_drawdown cannot tell them apart; the second holds that loss for
        // ten samples instead of one.
        let brief =
            vec![100.0, 90.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0];
        let long = vec![100.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 100.0];

        assert!((max_drawdown(&brief) - max_drawdown(&long)).abs() < 1e-12);
        assert!(ulcer_index(&long) > ulcer_index(&brief) * 2.0);
    }

    #[test]
    fn ulcer_index_never_exceeds_max_drawdown() {
        // An RMS over a series bounded by its own maximum cannot exceed it.
        // This pins the percentage-point scaling: a fraction-vs-percent slip
        // in either function breaks it by 100x.
        let equity = vec![100.0, 110.0, 105.0, 120.0, 100.0, 118.0, 95.0];
        assert!(ulcer_index(&equity) <= max_drawdown(&equity) + 1e-12);
    }

    #[test]
    fn an_empty_curve_reports_no_ulcer_rather_than_a_nan() {
        // 0.0/0.0 would ship a NaN across the PyO3 boundary, where it is not
        // JSON-serializable.
        assert_eq!(ulcer_index(&[]), 0.0);
        assert!(!ulcer_index(&[100.0]).is_nan());
    }

    #[test]
    fn drawdown_periods_counts_a_high_water_plateau_as_underwater() {
        // Records current behaviour, which OVERSTATES time under water:
        // `dd_start` is set to `peak_idx`, and `peak_idx` only advances on a
        // strictly new high, so a flat stretch at the high-water mark is
        // folded into the following drawdown.
        //
        // Here exactly one sample (index 4, at 105) is below its running peak,
        // but the reported span runs 1..=4. Flat equity is the normal state of
        // a backtest holding no position, so summing durations off these
        // tuples inflates badly.
        //
        // This is why `time_under_water_pct` in PortfolioEngine::calculate_metrics
        // counts samples off the streamed drawdown curve instead of using this
        // function. Fixing it changes a public function's return values and is
        // deferred to a release that can carry the caller audit.
        let equity = vec![100.0, 110.0, 110.0, 110.0, 105.0, 120.0];
        let periods = drawdown_periods(&equity);
        assert_eq!(periods.len(), 1);
        assert_eq!(periods[0].0, 1, "start is pinned to the plateau, not the decline");
        assert_eq!(periods[0].1, 4);
    }
}
