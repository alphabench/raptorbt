//! Rolling min/max indicators for LLV/HHV support.
//!
//! Provides rolling minimum and maximum calculations for Lowest Low Value (LLV)
//! and Highest High Value (HHV) expressions.

use std::cmp::Ordering;
use std::collections::VecDeque;

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
    rolling_extremum(data, period, Ordering::Less, f64::INFINITY)
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
    rolling_extremum(data, period, Ordering::Greater, f64::NEG_INFINITY)
}

/// O(n) sliding extremum over a monotonic deque of indices; NaN is skipped.
///
/// `want` is the ordering that makes a value better (Less for the minimum). Values compare by
/// `total_cmp`, which puts -0.0 below +0.0: the minimum of a window holding both zeros is -0.0
/// and the maximum +0.0, whatever their order -- the result `f64::min`/`f64::max` give on
/// aarch64 (the standard library leaves it unspecified). `identity` is the fold seed of the
/// loop this replaced: a window whose extremum equals it, including one of only infinities of
/// that sign, yields NaN.
fn rolling_extremum(
    data: &[f64],
    period: usize,
    want: Ordering,
    identity: f64,
) -> Result<Vec<f64>, RaptorError> {
    if period == 0 {
        return Err(RaptorError::invalid_parameter("period must be at least 1"));
    }

    let n = data.len();
    let mut result = vec![f64::NAN; n];
    // Indices with strictly better-or-equal-later values: the front is the window's extremum.
    let mut dq: VecDeque<usize> = VecDeque::with_capacity(period.min(n) + 1);

    for (i, &x) in data.iter().enumerate() {
        if !x.is_nan() {
            while dq.back().is_some_and(|&j| x.total_cmp(&data[j]) != want.reverse()) {
                dq.pop_back();
            }
            dq.push_back(i);
        }
        while dq.front().is_some_and(|&j| j + period <= i) {
            dq.pop_front();
        }
        if i + 1 >= period {
            if let Some(&j) = dq.front() {
                if data[j] != identity {
                    result[i] = data[j];
                }
            }
        }
    }

    Ok(result)
}

/// The loops `rolling_min` and `rolling_max` replaced, kept as the oracle for the deque.
#[cfg(test)]
fn naive(data: &[f64], period: usize, max: bool) -> Vec<f64> {
    let n = data.len();
    let mut result = vec![f64::NAN; n];
    for i in (period - 1)..n {
        let w = &data[i + 1 - period..=i];
        if max {
            let v = w.iter().fold(f64::NEG_INFINITY, |a, &b| if b.is_nan() { a } else { a.max(b) });
            result[i] = if v == f64::NEG_INFINITY { f64::NAN } else { v };
        } else {
            let v = w.iter().fold(f64::INFINITY, |a, &b| if b.is_nan() { a } else { a.min(b) });
            result[i] = if v == f64::INFINITY { f64::NAN } else { v };
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

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

    fn same(a: &[f64], b: &[f64]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x.is_nan() && y.is_nan()) || x == y)
    }

    #[cfg(target_arch = "aarch64")]
    fn same_bits(a: &[f64], b: &[f64]) -> bool {
        a.len() == b.len()
            && a.iter()
                .zip(b)
                .all(|(x, y)| (x.is_nan() && y.is_nan()) || x.to_bits() == y.to_bits())
    }

    fn value() -> impl Strategy<Value = f64> {
        prop_oneof![
            4 => -10.0f64..10.0,
            2 => prop::sample::select(vec![0.0, -0.0, 1.0, -1.0]),
            2 => Just(f64::NAN),
            1 => Just(f64::INFINITY),
            1 => Just(f64::NEG_INFINITY),
        ]
    }

    proptest! {
        #[test]
        fn deque_equals_the_loop(
            data in prop::collection::vec(value(), 0..200),
            period in 1usize..40,
        ) {
            prop_assert!(same(&rolling_min(&data, period).unwrap(), &naive(&data, period, false)));
            prop_assert!(same(&rolling_max(&data, period).unwrap(), &naive(&data, period, true)));
        }

        // Bitwise, so 0.0 against -0.0 counts. Only where the loop's own answer is defined:
        // f64::min/max pick a zero by hardware rule, which is order-independent on aarch64.
        #[cfg(target_arch = "aarch64")]
        #[test]
        fn deque_equals_the_loop_bitwise(
            data in prop::collection::vec(value(), 0..200),
            period in 1usize..40,
        ) {
            prop_assert!(same_bits(&rolling_min(&data, period).unwrap(), &naive(&data, period, false)));
            prop_assert!(same_bits(&rolling_max(&data, period).unwrap(), &naive(&data, period, true)));
        }
    }

    #[test]
    fn signed_zero_follows_ieee_minimum_and_maximum() {
        for data in [[0.0, -0.0, 5.0], [-0.0, 0.0, 5.0]] {
            let lo = rolling_min(&data, 2).unwrap();
            let hi = rolling_max(&data, 2).unwrap();
            assert!(lo[1] == 0.0 && lo[1].is_sign_negative());
            assert!(hi[1] == 0.0 && hi[1].is_sign_positive());
        }
    }

    #[test]
    fn all_infinite_window_is_nan_as_before() {
        let data = [f64::INFINITY, f64::INFINITY, 1.0];
        assert!(rolling_min(&data, 2).unwrap()[1].is_nan());
        assert_eq!(rolling_max(&data, 2).unwrap()[1], f64::INFINITY);
    }

    #[test]
    fn test_invalid_period() {
        let data = vec![1.0, 2.0, 3.0];
        assert!(rolling_min(&data, 0).is_err());
        assert!(rolling_max(&data, 0).is_err());
    }
}
