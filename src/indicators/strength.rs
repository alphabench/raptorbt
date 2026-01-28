//! Strength indicators: ADX.

use crate::core::error::RaptorError;
use crate::core::Result;

/// Average Directional Index (ADX).
///
/// # Arguments
/// * `high` - High prices
/// * `low` - Low prices
/// * `close` - Close prices
/// * `period` - Lookback period (default: 14)
///
/// # Returns
/// Vector of ADX values (0-100 scale, NaN for warmup period)
pub fn adx(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Result<Vec<f64>> {
    let n = close.len();
    if n != high.len() || n != low.len() {
        return Err(RaptorError::length_mismatch(n, high.len()));
    }
    if period == 0 {
        return Err(RaptorError::invalid_parameter("ADX period must be > 0"));
    }

    let mut result = vec![f64::NAN; n];

    // Need at least 2 * period for meaningful ADX
    if 2 * period > n {
        return Ok(result);
    }

    // Calculate directional movement
    let mut plus_dm = vec![0.0; n];
    let mut minus_dm = vec![0.0; n];
    let mut tr = vec![0.0; n];

    for i in 1..n {
        let up_move = high[i] - high[i - 1];
        let down_move = low[i - 1] - low[i];

        // +DM
        if up_move > down_move && up_move > 0.0 {
            plus_dm[i] = up_move;
        }

        // -DM
        if down_move > up_move && down_move > 0.0 {
            minus_dm[i] = down_move;
        }

        // True Range
        let hl = high[i] - low[i];
        let hc = (high[i] - close[i - 1]).abs();
        let lc = (low[i] - close[i - 1]).abs();
        tr[i] = hl.max(hc).max(lc);
    }

    // Smooth DM and TR using Wilder's smoothing
    let mut smooth_plus_dm = vec![0.0; n];
    let mut smooth_minus_dm = vec![0.0; n];
    let mut smooth_tr = vec![0.0; n];

    // Initial sums
    let sum_plus_dm: f64 = plus_dm[1..=period].iter().sum();
    let sum_minus_dm: f64 = minus_dm[1..=period].iter().sum();
    let sum_tr: f64 = tr[1..=period].iter().sum();

    smooth_plus_dm[period] = sum_plus_dm;
    smooth_minus_dm[period] = sum_minus_dm;
    smooth_tr[period] = sum_tr;

    // Wilder's smoothing for remaining values
    for i in (period + 1)..n {
        smooth_plus_dm[i] =
            smooth_plus_dm[i - 1] - (smooth_plus_dm[i - 1] / period as f64) + plus_dm[i];
        smooth_minus_dm[i] =
            smooth_minus_dm[i - 1] - (smooth_minus_dm[i - 1] / period as f64) + minus_dm[i];
        smooth_tr[i] = smooth_tr[i - 1] - (smooth_tr[i - 1] / period as f64) + tr[i];
    }

    // Calculate DI+ and DI-
    let mut plus_di = vec![0.0; n];
    let mut minus_di = vec![0.0; n];
    let mut dx = vec![0.0; n];

    for i in period..n {
        if smooth_tr[i] > 0.0 {
            plus_di[i] = 100.0 * smooth_plus_dm[i] / smooth_tr[i];
            minus_di[i] = 100.0 * smooth_minus_dm[i] / smooth_tr[i];

            // Calculate DX
            let di_sum = plus_di[i] + minus_di[i];
            if di_sum > 0.0 {
                dx[i] = 100.0 * (plus_di[i] - minus_di[i]).abs() / di_sum;
            }
        }
    }

    // Calculate ADX (smoothed DX)
    let adx_start = 2 * period - 1;
    if adx_start < n {
        // Initial ADX is average of first 'period' DX values
        let initial_adx: f64 = dx[period..=adx_start].iter().sum::<f64>() / period as f64;
        result[adx_start] = initial_adx;

        // Smooth ADX for remaining values
        for i in (adx_start + 1)..n {
            result[i] = (result[i - 1] * (period - 1) as f64 + dx[i]) / period as f64;
        }
    }

    Ok(result)
}

/// Directional Index result including +DI, -DI, and ADX.
#[derive(Debug, Clone)]
pub struct DirectionalIndexResult {
    /// +DI values.
    pub plus_di: Vec<f64>,
    /// -DI values.
    pub minus_di: Vec<f64>,
    /// ADX values.
    pub adx: Vec<f64>,
}

/// Full Directional Movement System (DI+, DI-, ADX).
///
/// # Arguments
/// * `high` - High prices
/// * `low` - Low prices
/// * `close` - Close prices
/// * `period` - Lookback period (default: 14)
///
/// # Returns
/// DirectionalIndexResult with +DI, -DI, and ADX
pub fn directional_movement(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
) -> Result<DirectionalIndexResult> {
    let n = close.len();
    if n != high.len() || n != low.len() {
        return Err(RaptorError::length_mismatch(n, high.len()));
    }
    if period == 0 {
        return Err(RaptorError::invalid_parameter("Period must be > 0"));
    }

    let mut plus_di = vec![f64::NAN; n];
    let mut minus_di = vec![f64::NAN; n];
    let mut adx_values = vec![f64::NAN; n];

    if 2 * period > n {
        return Ok(DirectionalIndexResult {
            plus_di,
            minus_di,
            adx: adx_values,
        });
    }

    // Calculate directional movement
    let mut plus_dm = vec![0.0; n];
    let mut minus_dm = vec![0.0; n];
    let mut tr = vec![0.0; n];

    for i in 1..n {
        let up_move = high[i] - high[i - 1];
        let down_move = low[i - 1] - low[i];

        if up_move > down_move && up_move > 0.0 {
            plus_dm[i] = up_move;
        }
        if down_move > up_move && down_move > 0.0 {
            minus_dm[i] = down_move;
        }

        let hl = high[i] - low[i];
        let hc = (high[i] - close[i - 1]).abs();
        let lc = (low[i] - close[i - 1]).abs();
        tr[i] = hl.max(hc).max(lc);
    }

    // Smooth using Wilder's method
    let mut smooth_plus_dm: f64 = plus_dm[1..=period].iter().sum();
    let mut smooth_minus_dm: f64 = minus_dm[1..=period].iter().sum();
    let mut smooth_tr: f64 = tr[1..=period].iter().sum();

    let mut dx = vec![0.0; n];

    for i in period..n {
        if i > period {
            smooth_plus_dm = smooth_plus_dm - (smooth_plus_dm / period as f64) + plus_dm[i];
            smooth_minus_dm = smooth_minus_dm - (smooth_minus_dm / period as f64) + minus_dm[i];
            smooth_tr = smooth_tr - (smooth_tr / period as f64) + tr[i];
        }

        if smooth_tr > 0.0 {
            plus_di[i] = 100.0 * smooth_plus_dm / smooth_tr;
            minus_di[i] = 100.0 * smooth_minus_dm / smooth_tr;

            let di_sum = plus_di[i] + minus_di[i];
            if di_sum > 0.0 {
                dx[i] = 100.0 * (plus_di[i] - minus_di[i]).abs() / di_sum;
            }
        }
    }

    // Calculate ADX
    let adx_start = 2 * period - 1;
    if adx_start < n {
        let initial_adx: f64 = dx[period..=adx_start].iter().sum::<f64>() / period as f64;
        adx_values[adx_start] = initial_adx;

        for i in (adx_start + 1)..n {
            adx_values[i] = (adx_values[i - 1] * (period - 1) as f64 + dx[i]) / period as f64;
        }
    }

    Ok(DirectionalIndexResult {
        plus_di,
        minus_di,
        adx: adx_values,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adx() {
        // Generate some trending data
        let n = 50;
        let high: Vec<f64> = (0..n).map(|i| 100.0 + i as f64 + 2.0).collect();
        let low: Vec<f64> = (0..n).map(|i| 100.0 + i as f64 - 2.0).collect();
        let close: Vec<f64> = (0..n).map(|i| 100.0 + i as f64).collect();

        let result = adx(&high, &low, &close, 14).unwrap();

        // ADX should be valid from index 27 (2 * period - 1)
        assert!(result[26].is_nan());
        assert!(!result[27].is_nan());

        // ADX should be positive and <= 100
        assert!(result[27] >= 0.0 && result[27] <= 100.0);
    }

    #[test]
    fn test_directional_movement() {
        let n = 50;
        let high: Vec<f64> = (0..n).map(|i| 100.0 + i as f64 + 2.0).collect();
        let low: Vec<f64> = (0..n).map(|i| 100.0 + i as f64 - 2.0).collect();
        let close: Vec<f64> = (0..n).map(|i| 100.0 + i as f64).collect();

        let result = directional_movement(&high, &low, &close, 14).unwrap();

        // Check DI values are valid
        assert!(!result.plus_di[20].is_nan());
        assert!(!result.minus_di[20].is_nan());

        // In an uptrend, +DI should be greater than -DI
        assert!(result.plus_di[40] > result.minus_di[40]);
    }
}
