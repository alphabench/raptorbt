//! Rolling min/max indicators for LLV/HHV support.
//!
//! Provides rolling minimum and maximum calculations for Lowest Low Value (LLV)
//! and Highest High Value (HHV) expressions.

use crate::core::error::RaptorError;

/// Calculate rolling minimum (Lowest Low Value) over a period.
///
/// Returns NaN for the first (period - 1) values where insufficient data exists.
///
/// # Arguments
/// * `data` - Input data slice
/// * `period` - Lookback period
///
/// # Returns
/// Vec of rolling minimum values
pub fn rolling_min(data: &[f64], period: usize) -> Result<Vec<f64>, RaptorError> {
    if period == 0 {
        return Err(RaptorError::invalid_parameter("period must be at least 1"));
    }

    let n = data.len();
    let mut result = vec![f64::NAN; n];

    for i in (period - 1)..n {
        let start = i + 1 - period;
        let min_val =
            data[start..=i]
                .iter()
                .fold(f64::INFINITY, |a, &b| if b.is_nan() { a } else { a.min(b) });
        result[i] = if min_val == f64::INFINITY { f64::NAN } else { min_val };
    }

    Ok(result)
}

/// Calculate rolling maximum (Highest High Value) over a period.
///
/// Returns NaN for the first (period - 1) values where insufficient data exists.
///
/// # Arguments
/// * `data` - Input data slice
/// * `period` - Lookback period
///
/// # Returns
/// Vec of rolling maximum values
pub fn rolling_max(data: &[f64], period: usize) -> Result<Vec<f64>, RaptorError> {
    if period == 0 {
        return Err(RaptorError::invalid_parameter("period must be at least 1"));
    }

    let n = data.len();
    let mut result = vec![f64::NAN; n];

    for i in (period - 1)..n {
        let start = i + 1 - period;
        let max_val =
            data[start..=i]
                .iter()
                .fold(f64::NEG_INFINITY, |a, &b| if b.is_nan() { a } else { a.max(b) });
        result[i] = if max_val == f64::NEG_INFINITY { f64::NAN } else { max_val };
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rolling_min() {
        let data = vec![5.0, 3.0, 8.0, 2.0, 7.0, 1.0, 9.0];
        let result = rolling_min(&data, 3).unwrap();

        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert!((result[2] - 3.0).abs() < f64::EPSILON); // min(5, 3, 8)
        assert!((result[3] - 2.0).abs() < f64::EPSILON); // min(3, 8, 2)
        assert!((result[4] - 2.0).abs() < f64::EPSILON); // min(8, 2, 7)
        assert!((result[5] - 1.0).abs() < f64::EPSILON); // min(2, 7, 1)
        assert!((result[6] - 1.0).abs() < f64::EPSILON); // min(7, 1, 9)
    }

    #[test]
    fn test_rolling_max() {
        let data = vec![5.0, 3.0, 8.0, 2.0, 7.0, 1.0, 9.0];
        let result = rolling_max(&data, 3).unwrap();

        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert!((result[2] - 8.0).abs() < f64::EPSILON); // max(5, 3, 8)
        assert!((result[3] - 8.0).abs() < f64::EPSILON); // max(3, 8, 2)
        assert!((result[4] - 8.0).abs() < f64::EPSILON); // max(8, 2, 7)
        assert!((result[5] - 7.0).abs() < f64::EPSILON); // max(2, 7, 1)
        assert!((result[6] - 9.0).abs() < f64::EPSILON); // max(7, 1, 9)
    }

    #[test]
    fn test_invalid_period() {
        let data = vec![1.0, 2.0, 3.0];
        assert!(rolling_min(&data, 0).is_err());
        assert!(rolling_max(&data, 0).is_err());
    }
}
