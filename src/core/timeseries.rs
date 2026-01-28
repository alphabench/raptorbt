//! Time-indexed array wrapper for efficient operations.

use super::types::Timestamp;

/// A time-indexed series of values.
#[derive(Debug, Clone)]
pub struct TimeSeries<T> {
    /// Timestamps for each value.
    pub timestamps: Vec<Timestamp>,
    /// Values.
    pub values: Vec<T>,
}

impl<T: Clone> TimeSeries<T> {
    /// Create a new time series.
    pub fn new(timestamps: Vec<Timestamp>, values: Vec<T>) -> Self {
        debug_assert_eq!(timestamps.len(), values.len());
        Self { timestamps, values }
    }

    /// Create from values only (no timestamps).
    pub fn from_values(values: Vec<T>) -> Self {
        let timestamps = (0..values.len() as i64).collect();
        Self { timestamps, values }
    }

    /// Get the length.
    #[inline]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Check if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Get value at index.
    #[inline]
    pub fn get(&self, index: usize) -> Option<&T> {
        self.values.get(index)
    }

    /// Get timestamp at index.
    #[inline]
    pub fn get_timestamp(&self, index: usize) -> Option<Timestamp> {
        self.timestamps.get(index).copied()
    }

    /// Get slice of values.
    pub fn slice(&self, start: usize, end: usize) -> Self {
        Self {
            timestamps: self.timestamps[start..end].to_vec(),
            values: self.values[start..end].to_vec(),
        }
    }

    /// Map values to a new type.
    pub fn map<U, F>(&self, f: F) -> TimeSeries<U>
    where
        F: Fn(&T) -> U,
    {
        TimeSeries {
            timestamps: self.timestamps.clone(),
            values: self.values.iter().map(f).collect(),
        }
    }

    /// Iterator over (timestamp, value) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (Timestamp, &T)> {
        self.timestamps.iter().copied().zip(self.values.iter())
    }
}

impl<T: Clone + Default> TimeSeries<T> {
    /// Create with default values.
    pub fn with_default(timestamps: Vec<Timestamp>) -> Self {
        let len = timestamps.len();
        Self {
            timestamps,
            values: vec![T::default(); len],
        }
    }
}

impl TimeSeries<f64> {
    /// Create a series filled with NaN.
    pub fn with_nan(len: usize) -> Self {
        Self {
            timestamps: (0..len as i64).collect(),
            values: vec![f64::NAN; len],
        }
    }

    /// Calculate sum of all values.
    pub fn sum(&self) -> f64 {
        self.values.iter().filter(|v| !v.is_nan()).sum()
    }

    /// Calculate mean of all values.
    pub fn mean(&self) -> f64 {
        let valid: Vec<_> = self.values.iter().filter(|v| !v.is_nan()).collect();
        if valid.is_empty() {
            return f64::NAN;
        }
        valid.iter().copied().sum::<f64>() / valid.len() as f64
    }

    /// Calculate standard deviation.
    pub fn std(&self) -> f64 {
        let mean = self.mean();
        if mean.is_nan() {
            return f64::NAN;
        }
        let valid: Vec<_> = self.values.iter().filter(|v| !v.is_nan()).collect();
        if valid.len() < 2 {
            return f64::NAN;
        }
        let variance =
            valid.iter().map(|v| (*v - mean).powi(2)).sum::<f64>() / (valid.len() - 1) as f64;
        variance.sqrt()
    }

    /// Get minimum value.
    pub fn min(&self) -> f64 {
        self.values
            .iter()
            .filter(|v| !v.is_nan())
            .copied()
            .fold(f64::INFINITY, f64::min)
    }

    /// Get maximum value.
    pub fn max(&self) -> f64 {
        self.values
            .iter()
            .filter(|v| !v.is_nan())
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// Shift values by n positions (positive = shift forward, fill with NaN).
    pub fn shift(&self, n: isize) -> Self {
        let len = self.values.len();
        let mut result = vec![f64::NAN; len];

        if n >= 0 {
            let n = n as usize;
            if n < len {
                for i in n..len {
                    result[i] = self.values[i - n];
                }
            }
        } else {
            let n = (-n) as usize;
            if n < len {
                for i in 0..len - n {
                    result[i] = self.values[i + n];
                }
            }
        }

        Self {
            timestamps: self.timestamps.clone(),
            values: result,
        }
    }

    /// Calculate difference from previous value.
    pub fn diff(&self) -> Self {
        let mut result = vec![f64::NAN; self.values.len()];
        for i in 1..self.values.len() {
            if !self.values[i].is_nan() && !self.values[i - 1].is_nan() {
                result[i] = self.values[i] - self.values[i - 1];
            }
        }
        Self {
            timestamps: self.timestamps.clone(),
            values: result,
        }
    }

    /// Calculate percentage change from previous value.
    pub fn pct_change(&self) -> Self {
        let mut result = vec![f64::NAN; self.values.len()];
        for i in 1..self.values.len() {
            if !self.values[i].is_nan() && !self.values[i - 1].is_nan() && self.values[i - 1] != 0.0
            {
                result[i] = (self.values[i] - self.values[i - 1]) / self.values[i - 1];
            }
        }
        Self {
            timestamps: self.timestamps.clone(),
            values: result,
        }
    }

    /// Apply rolling window function.
    pub fn rolling<F>(&self, window: usize, f: F) -> Self
    where
        F: Fn(&[f64]) -> f64,
    {
        let mut result = vec![f64::NAN; self.values.len()];
        if window == 0 || window > self.values.len() {
            return Self {
                timestamps: self.timestamps.clone(),
                values: result,
            };
        }

        for i in (window - 1)..self.values.len() {
            let slice = &self.values[i + 1 - window..=i];
            result[i] = f(slice);
        }

        Self {
            timestamps: self.timestamps.clone(),
            values: result,
        }
    }

    /// Calculate rolling sum.
    pub fn rolling_sum(&self, window: usize) -> Self {
        self.rolling(window, |slice| slice.iter().sum())
    }

    /// Calculate rolling mean.
    pub fn rolling_mean(&self, window: usize) -> Self {
        self.rolling(window, |slice| {
            slice.iter().sum::<f64>() / slice.len() as f64
        })
    }

    /// Calculate rolling standard deviation.
    pub fn rolling_std(&self, window: usize) -> Self {
        self.rolling(window, |slice| {
            let mean = slice.iter().sum::<f64>() / slice.len() as f64;
            let variance =
                slice.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (slice.len() - 1) as f64;
            variance.sqrt()
        })
    }

    /// Calculate rolling maximum.
    pub fn rolling_max(&self, window: usize) -> Self {
        self.rolling(window, |slice| {
            slice.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        })
    }

    /// Calculate rolling minimum.
    pub fn rolling_min(&self, window: usize) -> Self {
        self.rolling(window, |slice| {
            slice.iter().copied().fold(f64::INFINITY, f64::min)
        })
    }
}

impl TimeSeries<bool> {
    /// Count true values.
    pub fn count_true(&self) -> usize {
        self.values.iter().filter(|&&v| v).count()
    }

    /// Get indices of true values.
    pub fn true_indices(&self) -> Vec<usize> {
        self.values
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| if v { Some(i) } else { None })
            .collect()
    }

    /// Logical AND with another series.
    pub fn and(&self, other: &Self) -> Self {
        debug_assert_eq!(self.len(), other.len());
        Self {
            timestamps: self.timestamps.clone(),
            values: self
                .values
                .iter()
                .zip(other.values.iter())
                .map(|(&a, &b)| a && b)
                .collect(),
        }
    }

    /// Logical OR with another series.
    pub fn or(&self, other: &Self) -> Self {
        debug_assert_eq!(self.len(), other.len());
        Self {
            timestamps: self.timestamps.clone(),
            values: self
                .values
                .iter()
                .zip(other.values.iter())
                .map(|(&a, &b)| a || b)
                .collect(),
        }
    }

    /// Logical NOT.
    pub fn not(&self) -> Self {
        Self {
            timestamps: self.timestamps.clone(),
            values: self.values.iter().map(|&v| !v).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rolling_mean() {
        let ts = TimeSeries::from_values(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let result = ts.rolling_mean(3);
        assert!(result.values[0].is_nan());
        assert!(result.values[1].is_nan());
        assert!((result.values[2] - 2.0).abs() < 1e-10);
        assert!((result.values[3] - 3.0).abs() < 1e-10);
        assert!((result.values[4] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_shift() {
        let ts = TimeSeries::from_values(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let shifted = ts.shift(2);
        assert!(shifted.values[0].is_nan());
        assert!(shifted.values[1].is_nan());
        assert!((shifted.values[2] - 1.0).abs() < 1e-10);
        assert!((shifted.values[3] - 2.0).abs() < 1e-10);
        assert!((shifted.values[4] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_pct_change() {
        let ts = TimeSeries::from_values(vec![100.0, 110.0, 99.0]);
        let pct = ts.pct_change();
        assert!(pct.values[0].is_nan());
        assert!((pct.values[1] - 0.1).abs() < 1e-10);
        assert!((pct.values[2] - (-0.1)).abs() < 1e-10);
    }
}
