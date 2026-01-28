//! Trend indicators: SMA, EMA, Supertrend.

use crate::core::error::RaptorError;
use crate::core::Result;

/// Simple Moving Average.
///
/// # Arguments
/// * `data` - Price data
/// * `period` - Lookback period
///
/// # Returns
/// Vector of SMA values (NaN for warmup period)
pub fn sma(data: &[f64], period: usize) -> Result<Vec<f64>> {
    if period == 0 {
        return Err(RaptorError::invalid_parameter("SMA period must be > 0"));
    }
    if data.is_empty() {
        return Ok(vec![]);
    }

    let n = data.len();
    let mut result = vec![f64::NAN; n];

    if period > n {
        return Ok(result);
    }

    // Calculate first SMA
    let mut sum: f64 = data[..period].iter().sum();
    result[period - 1] = sum / period as f64;

    // Sliding window for remaining values
    for i in period..n {
        sum = sum - data[i - period] + data[i];
        result[i] = sum / period as f64;
    }

    Ok(result)
}

/// Exponential Moving Average.
///
/// # Arguments
/// * `data` - Price data
/// * `period` - Lookback period (used to calculate smoothing factor)
///
/// # Returns
/// Vector of EMA values (NaN for warmup period)
pub fn ema(data: &[f64], period: usize) -> Result<Vec<f64>> {
    if period == 0 {
        return Err(RaptorError::invalid_parameter("EMA period must be > 0"));
    }
    if data.is_empty() {
        return Ok(vec![]);
    }

    let n = data.len();
    let mut result = vec![f64::NAN; n];

    if period > n {
        return Ok(result);
    }

    // Smoothing factor
    let alpha = 2.0 / (period as f64 + 1.0);

    // Initialize with SMA of first 'period' values
    let initial_sma: f64 = data[..period].iter().sum::<f64>() / period as f64;
    result[period - 1] = initial_sma;

    // Calculate EMA for remaining values
    for i in period..n {
        result[i] = alpha * data[i] + (1.0 - alpha) * result[i - 1];
    }

    Ok(result)
}

/// EMA with custom smoothing factor (internal use).
#[allow(dead_code)]
pub(crate) fn ema_with_alpha(data: &[f64], alpha: f64, initial: f64) -> Vec<f64> {
    let n = data.len();
    let mut result = vec![f64::NAN; n];

    if n == 0 {
        return result;
    }

    result[0] = initial;
    for i in 1..n {
        if data[i].is_nan() {
            result[i] = result[i - 1];
        } else {
            result[i] = alpha * data[i] + (1.0 - alpha) * result[i - 1];
        }
    }

    result
}

/// Supertrend indicator result.
#[derive(Debug, Clone)]
pub struct SupertrendResult {
    /// Supertrend line values.
    pub supertrend: Vec<f64>,
    /// Direction: 1 = bullish (below price), -1 = bearish (above price).
    pub direction: Vec<i8>,
}

/// Supertrend indicator.
///
/// # Arguments
/// * `high` - High prices
/// * `low` - Low prices
/// * `close` - Close prices
/// * `period` - ATR period
/// * `multiplier` - ATR multiplier
///
/// # Returns
/// SupertrendResult with supertrend line and direction
pub fn supertrend(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    multiplier: f64,
) -> Result<SupertrendResult> {
    let n = close.len();
    if n != high.len() || n != low.len() {
        return Err(RaptorError::length_mismatch(n, high.len()));
    }
    if period == 0 {
        return Err(RaptorError::invalid_parameter("Supertrend period must be > 0"));
    }

    let mut supertrend = vec![f64::NAN; n];
    let mut direction = vec![0i8; n];

    if period >= n {
        return Ok(SupertrendResult { supertrend, direction });
    }

    // Calculate ATR
    let atr_values = super::volatility::atr(high, low, close, period)?;

    // Calculate basic upper and lower bands
    let mut upper_band = vec![f64::NAN; n];
    let mut lower_band = vec![f64::NAN; n];

    for i in (period - 1)..n {
        let hl2 = (high[i] + low[i]) / 2.0;
        let atr_val = atr_values[i];
        if !atr_val.is_nan() {
            upper_band[i] = hl2 + multiplier * atr_val;
            lower_band[i] = hl2 - multiplier * atr_val;
        }
    }

    // Calculate final bands with carryover logic
    let mut final_upper = vec![f64::NAN; n];
    let mut final_lower = vec![f64::NAN; n];

    for i in (period - 1)..n {
        if i == period - 1 {
            final_upper[i] = upper_band[i];
            final_lower[i] = lower_band[i];
        } else {
            // Final upper band: use lower of current upper or previous final upper
            // if previous close was below previous final upper
            if !upper_band[i].is_nan() && !final_upper[i - 1].is_nan() {
                if close[i - 1] <= final_upper[i - 1] {
                    final_upper[i] = upper_band[i].min(final_upper[i - 1]);
                } else {
                    final_upper[i] = upper_band[i];
                }
            } else {
                final_upper[i] = upper_band[i];
            }

            // Final lower band: use higher of current lower or previous final lower
            // if previous close was above previous final lower
            if !lower_band[i].is_nan() && !final_lower[i - 1].is_nan() {
                if close[i - 1] >= final_lower[i - 1] {
                    final_lower[i] = lower_band[i].max(final_lower[i - 1]);
                } else {
                    final_lower[i] = lower_band[i];
                }
            } else {
                final_lower[i] = lower_band[i];
            }
        }
    }

    // Calculate supertrend and direction
    for i in (period - 1)..n {
        if i == period - 1 {
            // Initial direction based on price vs bands
            if close[i] <= final_upper[i] {
                supertrend[i] = final_upper[i];
                direction[i] = -1; // bearish
            } else {
                supertrend[i] = final_lower[i];
                direction[i] = 1; // bullish
            }
        } else {
            let _prev_st = supertrend[i - 1];
            let prev_dir = direction[i - 1];

            if prev_dir == 1 {
                // Was bullish
                if close[i] < final_lower[i] {
                    // Switch to bearish
                    supertrend[i] = final_upper[i];
                    direction[i] = -1;
                } else {
                    // Stay bullish
                    supertrend[i] = final_lower[i];
                    direction[i] = 1;
                }
            } else {
                // Was bearish
                if close[i] > final_upper[i] {
                    // Switch to bullish
                    supertrend[i] = final_lower[i];
                    direction[i] = 1;
                } else {
                    // Stay bearish
                    supertrend[i] = final_upper[i];
                    direction[i] = -1;
                }
            }
        }
    }

    Ok(SupertrendResult { supertrend, direction })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sma() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = sma(&data, 3).unwrap();
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert!((result[2] - 2.0).abs() < 1e-10);
        assert!((result[3] - 3.0).abs() < 1e-10);
        assert!((result[4] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_ema() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = ema(&data, 3).unwrap();
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert!(!result[2].is_nan());
        assert!(!result[3].is_nan());
        assert!(!result[4].is_nan());
        // EMA should be between min and max of data
        assert!(result[4] >= 1.0 && result[4] <= 5.0);
    }

    #[test]
    fn test_sma_invalid_period() {
        let data = vec![1.0, 2.0, 3.0];
        let result = sma(&data, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_ema_period_larger_than_data() {
        let data = vec![1.0, 2.0, 3.0];
        let result = ema(&data, 10).unwrap();
        assert!(result.iter().all(|v| v.is_nan()));
    }
}
