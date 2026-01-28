//! Expression evaluation for signal generation.
//!
//! Provides a Rust-native expression evaluator for generating trading signals
//! from indicator values.

/// Comparison operators for signal generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompareOp {
    /// Greater than.
    Gt,
    /// Greater than or equal.
    Gte,
    /// Less than.
    Lt,
    /// Less than or equal.
    Lte,
    /// Equal (within tolerance).
    Eq,
    /// Not equal.
    Ne,
}

/// Crossover/crossunder detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrossType {
    /// Line A crosses above line B.
    CrossOver,
    /// Line A crosses below line B.
    CrossUnder,
}

/// Compare two series element-wise.
///
/// # Arguments
/// * `a` - First series
/// * `b` - Second series
/// * `op` - Comparison operator
///
/// # Returns
/// Boolean series indicating where comparison is true
pub fn compare(a: &[f64], b: &[f64], op: CompareOp) -> Vec<bool> {
    let n = a.len();
    assert_eq!(n, b.len());

    let tolerance = 1e-10;

    let mut result = vec![false; n];
    for i in 0..n {
        if a[i].is_nan() || b[i].is_nan() {
            continue;
        }
        result[i] = match op {
            CompareOp::Gt => a[i] > b[i],
            CompareOp::Gte => a[i] >= b[i],
            CompareOp::Lt => a[i] < b[i],
            CompareOp::Lte => a[i] <= b[i],
            CompareOp::Eq => (a[i] - b[i]).abs() < tolerance,
            CompareOp::Ne => (a[i] - b[i]).abs() >= tolerance,
        };
    }

    result
}

/// Compare series with a scalar value.
///
/// # Arguments
/// * `a` - Series
/// * `value` - Scalar value to compare against
/// * `op` - Comparison operator
///
/// # Returns
/// Boolean series indicating where comparison is true
pub fn compare_scalar(a: &[f64], value: f64, op: CompareOp) -> Vec<bool> {
    let n = a.len();
    let tolerance = 1e-10;

    let mut result = vec![false; n];
    for i in 0..n {
        if a[i].is_nan() {
            continue;
        }
        result[i] = match op {
            CompareOp::Gt => a[i] > value,
            CompareOp::Gte => a[i] >= value,
            CompareOp::Lt => a[i] < value,
            CompareOp::Lte => a[i] <= value,
            CompareOp::Eq => (a[i] - value).abs() < tolerance,
            CompareOp::Ne => (a[i] - value).abs() >= tolerance,
        };
    }

    result
}

/// Detect crossover/crossunder between two series.
///
/// Crossover: a crosses above b (a[i-1] < b[i-1] and a[i] > b[i])
/// Crossunder: a crosses below b (a[i-1] > b[i-1] and a[i] < b[i])
///
/// # Arguments
/// * `a` - First series
/// * `b` - Second series
/// * `cross_type` - Type of cross to detect
///
/// # Returns
/// Boolean series indicating where cross occurs
pub fn cross(a: &[f64], b: &[f64], cross_type: CrossType) -> Vec<bool> {
    let n = a.len();
    assert_eq!(n, b.len());

    let mut result = vec![false; n];
    if n < 2 {
        return result;
    }

    for i in 1..n {
        if a[i].is_nan() || b[i].is_nan() || a[i - 1].is_nan() || b[i - 1].is_nan() {
            continue;
        }

        result[i] = match cross_type {
            CrossType::CrossOver => a[i - 1] <= b[i - 1] && a[i] > b[i],
            CrossType::CrossUnder => a[i - 1] >= b[i - 1] && a[i] < b[i],
        };
    }

    result
}

/// Detect crossover with a scalar value.
///
/// # Arguments
/// * `a` - Series
/// * `value` - Scalar value to cross
/// * `cross_type` - Type of cross to detect
///
/// # Returns
/// Boolean series indicating where cross occurs
pub fn cross_scalar(a: &[f64], value: f64, cross_type: CrossType) -> Vec<bool> {
    let n = a.len();
    let mut result = vec![false; n];

    if n < 2 {
        return result;
    }

    for i in 1..n {
        if a[i].is_nan() || a[i - 1].is_nan() {
            continue;
        }

        result[i] = match cross_type {
            CrossType::CrossOver => a[i - 1] <= value && a[i] > value,
            CrossType::CrossUnder => a[i - 1] >= value && a[i] < value,
        };
    }

    result
}

/// Check if value is in a range.
///
/// # Arguments
/// * `a` - Series
/// * `low` - Lower bound
/// * `high` - Upper bound
///
/// # Returns
/// Boolean series indicating where value is in range [low, high]
pub fn in_range(a: &[f64], low: f64, high: f64) -> Vec<bool> {
    let n = a.len();
    let mut result = vec![false; n];

    for i in 0..n {
        if a[i].is_nan() {
            continue;
        }
        result[i] = a[i] >= low && a[i] <= high;
    }

    result
}

/// Check if series is rising (current > previous).
///
/// # Arguments
/// * `a` - Series
/// * `periods` - Number of periods to look back (default: 1)
///
/// # Returns
/// Boolean series indicating where value is rising
pub fn is_rising(a: &[f64], periods: usize) -> Vec<bool> {
    let n = a.len();
    let mut result = vec![false; n];

    if periods >= n {
        return result;
    }

    for i in periods..n {
        if a[i].is_nan() || a[i - periods].is_nan() {
            continue;
        }
        result[i] = a[i] > a[i - periods];
    }

    result
}

/// Check if series is falling (current < previous).
///
/// # Arguments
/// * `a` - Series
/// * `periods` - Number of periods to look back (default: 1)
///
/// # Returns
/// Boolean series indicating where value is falling
pub fn is_falling(a: &[f64], periods: usize) -> Vec<bool> {
    let n = a.len();
    let mut result = vec![false; n];

    if periods >= n {
        return result;
    }

    for i in periods..n {
        if a[i].is_nan() || a[i - periods].is_nan() {
            continue;
        }
        result[i] = a[i] < a[i - periods];
    }

    result
}

/// Check if value has been above a threshold for n consecutive bars.
///
/// # Arguments
/// * `a` - Series
/// * `threshold` - Threshold value
/// * `consecutive` - Number of consecutive bars required
///
/// # Returns
/// Boolean series indicating where condition is met
pub fn above_for(a: &[f64], threshold: f64, consecutive: usize) -> Vec<bool> {
    let n = a.len();
    let mut result = vec![false; n];

    if consecutive > n {
        return result;
    }

    for i in (consecutive - 1)..n {
        let mut all_above = true;
        for j in 0..consecutive {
            let idx = i - j;
            if a[idx].is_nan() || a[idx] <= threshold {
                all_above = false;
                break;
            }
        }
        result[i] = all_above;
    }

    result
}

/// Check if value has been below a threshold for n consecutive bars.
///
/// # Arguments
/// * `a` - Series
/// * `threshold` - Threshold value
/// * `consecutive` - Number of consecutive bars required
///
/// # Returns
/// Boolean series indicating where condition is met
pub fn below_for(a: &[f64], threshold: f64, consecutive: usize) -> Vec<bool> {
    let n = a.len();
    let mut result = vec![false; n];

    if consecutive > n {
        return result;
    }

    for i in (consecutive - 1)..n {
        let mut all_below = true;
        for j in 0..consecutive {
            let idx = i - j;
            if a[idx].is_nan() || a[idx] >= threshold {
                all_below = false;
                break;
            }
        }
        result[i] = all_below;
    }

    result
}

/// Detect highest value in rolling window.
///
/// # Arguments
/// * `a` - Series
/// * `window` - Window size
///
/// # Returns
/// Boolean series indicating where current value is highest in window
pub fn is_highest(a: &[f64], window: usize) -> Vec<bool> {
    let n = a.len();
    let mut result = vec![false; n];

    if window > n || window == 0 {
        return result;
    }

    for i in (window - 1)..n {
        let start = i + 1 - window;
        let current = a[i];
        if current.is_nan() {
            continue;
        }

        let max_in_window = a[start..=i]
            .iter()
            .filter(|v| !v.is_nan())
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        result[i] = (current - max_in_window).abs() < 1e-10;
    }

    result
}

/// Detect lowest value in rolling window.
///
/// # Arguments
/// * `a` - Series
/// * `window` - Window size
///
/// # Returns
/// Boolean series indicating where current value is lowest in window
pub fn is_lowest(a: &[f64], window: usize) -> Vec<bool> {
    let n = a.len();
    let mut result = vec![false; n];

    if window > n || window == 0 {
        return result;
    }

    for i in (window - 1)..n {
        let start = i + 1 - window;
        let current = a[i];
        if current.is_nan() {
            continue;
        }

        let min_in_window = a[start..=i]
            .iter()
            .filter(|v| !v.is_nan())
            .fold(f64::INFINITY, |a, &b| a.min(b));

        result[i] = (current - min_in_window).abs() < 1e-10;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compare() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 2.0, 2.0, 2.0];

        let result = compare(&a, &b, CompareOp::Gt);
        assert!(!result[0]); // 1 > 2 = false
        assert!(!result[1]); // 2 > 2 = false
        assert!(result[2]); // 3 > 2 = true
        assert!(result[3]); // 4 > 2 = true
    }

    #[test]
    fn test_crossover() {
        let a = vec![1.0, 1.5, 2.5, 3.0, 2.5];
        let b = vec![2.0, 2.0, 2.0, 2.0, 2.0];

        let result = cross(&a, &b, CrossType::CrossOver);
        assert!(!result[0]); // No previous
        assert!(!result[1]); // 1.0 < 2.0, 1.5 < 2.0 - still below
        assert!(result[2]); // 1.5 < 2.0, 2.5 > 2.0 - crossed over!
        assert!(!result[3]); // 2.5 > 2.0, 3.0 > 2.0 - already above
        assert!(!result[4]); // 3.0 > 2.0, 2.5 > 2.0 - still above
    }

    #[test]
    fn test_crossunder() {
        let a = vec![3.0, 2.5, 1.5, 1.0, 1.5];
        let b = vec![2.0, 2.0, 2.0, 2.0, 2.0];

        let result = cross(&a, &b, CrossType::CrossUnder);
        assert!(!result[0]); // No previous
        assert!(!result[1]); // 3.0 > 2.0, 2.5 > 2.0 - still above
        assert!(result[2]); // 2.5 > 2.0, 1.5 < 2.0 - crossed under!
        assert!(!result[3]); // 1.5 < 2.0, 1.0 < 2.0 - already below
        assert!(!result[4]); // 1.0 < 2.0, 1.5 < 2.0 - still below
    }

    #[test]
    fn test_in_range() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let result = in_range(&a, 2.0, 4.0);
        assert!(!result[0]); // 1 not in [2, 4]
        assert!(result[1]); // 2 in [2, 4]
        assert!(result[2]); // 3 in [2, 4]
        assert!(result[3]); // 4 in [2, 4]
        assert!(!result[4]); // 5 not in [2, 4]
    }

    #[test]
    fn test_is_rising() {
        let a = vec![1.0, 2.0, 3.0, 2.5, 3.5];

        let result = is_rising(&a, 1);
        assert!(!result[0]); // No previous
        assert!(result[1]); // 2 > 1
        assert!(result[2]); // 3 > 2
        assert!(!result[3]); // 2.5 < 3
        assert!(result[4]); // 3.5 > 2.5
    }

    #[test]
    fn test_above_for() {
        let a = vec![1.0, 3.0, 3.5, 4.0, 2.0, 3.0];
        let threshold = 2.5;

        let result = above_for(&a, threshold, 3);
        assert!(!result[0]);
        assert!(!result[1]);
        assert!(!result[2]); // 1.0 < 2.5
        assert!(result[3]); // 3.0, 3.5, 4.0 all > 2.5
        assert!(!result[4]); // 2.0 < 2.5
        assert!(!result[5]);
    }

    #[test]
    fn test_is_highest() {
        let a = vec![1.0, 3.0, 2.0, 4.0, 3.5];

        let result = is_highest(&a, 3);
        assert!(!result[0]);
        assert!(!result[1]);
        assert!(result[2] == false); // 2.0 is not highest in [1.0, 3.0, 2.0]
        assert!(result[3]); // 4.0 is highest in [3.0, 2.0, 4.0]
        assert!(!result[4]); // 3.5 is not highest in [2.0, 4.0, 3.5]
    }
}
