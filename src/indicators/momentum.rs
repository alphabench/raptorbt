//! Momentum indicators: RSI, MACD, Stochastic.

use super::trend::ema;
use crate::core::error::RaptorError;
use crate::core::Result;

/// Relative Strength Index (RSI).
///
/// # Arguments
/// * `data` - Price data (typically close prices)
/// * `period` - Lookback period (default: 14)
///
/// # Returns
/// Vector of RSI values (0-100 scale, NaN for warmup period)
pub fn rsi(data: &[f64], period: usize) -> Result<Vec<f64>> {
    if period == 0 {
        return Err(RaptorError::invalid_parameter("RSI period must be > 0"));
    }
    if data.len() < 2 {
        return Ok(vec![f64::NAN; data.len()]);
    }

    let n = data.len();
    let mut result = vec![f64::NAN; n];

    // Calculate price changes
    let mut gains = vec![0.0; n];
    let mut losses = vec![0.0; n];

    for i in 1..n {
        let change = data[i] - data[i - 1];
        if change > 0.0 {
            gains[i] = change;
        } else {
            losses[i] = -change;
        }
    }

    if period >= n {
        return Ok(result);
    }

    // Calculate initial average gain/loss using SMA
    let mut avg_gain: f64 = gains[1..=period].iter().sum::<f64>() / period as f64;
    let mut avg_loss: f64 = losses[1..=period].iter().sum::<f64>() / period as f64;

    // First RSI value
    if avg_loss == 0.0 {
        result[period] = 100.0;
    } else {
        let rs = avg_gain / avg_loss;
        result[period] = 100.0 - (100.0 / (1.0 + rs));
    }

    // Smoothed moving average for remaining values (Wilder's smoothing)
    let alpha = 1.0 / period as f64;
    for i in (period + 1)..n {
        avg_gain = alpha * gains[i] + (1.0 - alpha) * avg_gain;
        avg_loss = alpha * losses[i] + (1.0 - alpha) * avg_loss;

        if avg_loss == 0.0 {
            result[i] = 100.0;
        } else {
            let rs = avg_gain / avg_loss;
            result[i] = 100.0 - (100.0 / (1.0 + rs));
        }
    }

    Ok(result)
}

/// MACD result structure.
#[derive(Debug, Clone)]
pub struct MacdResult {
    /// MACD line (fast EMA - slow EMA).
    pub macd_line: Vec<f64>,
    /// Signal line (EMA of MACD line).
    pub signal_line: Vec<f64>,
    /// Histogram (MACD line - signal line).
    pub histogram: Vec<f64>,
}

/// Moving Average Convergence Divergence (MACD).
///
/// # Arguments
/// * `data` - Price data (typically close prices)
/// * `fast_period` - Fast EMA period (default: 12)
/// * `slow_period` - Slow EMA period (default: 26)
/// * `signal_period` - Signal line EMA period (default: 9)
///
/// # Returns
/// MacdResult with MACD line, signal line, and histogram
pub fn macd(
    data: &[f64],
    fast_period: usize,
    slow_period: usize,
    signal_period: usize,
) -> Result<MacdResult> {
    if fast_period == 0 || slow_period == 0 || signal_period == 0 {
        return Err(RaptorError::invalid_parameter("MACD periods must be > 0"));
    }
    if fast_period >= slow_period {
        return Err(RaptorError::invalid_parameter("MACD fast period must be < slow period"));
    }

    let n = data.len();
    let mut macd_line = vec![f64::NAN; n];
    let mut signal_line = vec![f64::NAN; n];
    let mut histogram = vec![f64::NAN; n];

    if slow_period > n {
        return Ok(MacdResult { macd_line, signal_line, histogram });
    }

    // Calculate fast and slow EMAs
    let fast_ema = ema(data, fast_period)?;
    let slow_ema = ema(data, slow_period)?;

    // Calculate MACD line
    for i in (slow_period - 1)..n {
        if !fast_ema[i].is_nan() && !slow_ema[i].is_nan() {
            macd_line[i] = fast_ema[i] - slow_ema[i];
        }
    }

    // Calculate signal line (EMA of MACD line)
    // Need at least signal_period valid MACD values
    let signal_start = slow_period - 1 + signal_period - 1;
    if signal_start < n {
        // Calculate initial signal using SMA of first signal_period MACD values
        let mut sum = 0.0;
        let mut count = 0;
        for i in (slow_period - 1)..=(slow_period - 1 + signal_period - 1) {
            if i < n && !macd_line[i].is_nan() {
                sum += macd_line[i];
                count += 1;
            }
        }
        if count == signal_period {
            let initial_signal = sum / signal_period as f64;
            signal_line[signal_start] = initial_signal;

            // EMA for remaining signal values
            let alpha = 2.0 / (signal_period as f64 + 1.0);
            for i in (signal_start + 1)..n {
                if !macd_line[i].is_nan() {
                    signal_line[i] = alpha * macd_line[i] + (1.0 - alpha) * signal_line[i - 1];
                }
            }
        }
    }

    // Calculate histogram
    for i in 0..n {
        if !macd_line[i].is_nan() && !signal_line[i].is_nan() {
            histogram[i] = macd_line[i] - signal_line[i];
        }
    }

    Ok(MacdResult { macd_line, signal_line, histogram })
}

/// Stochastic oscillator result.
#[derive(Debug, Clone)]
pub struct StochasticResult {
    /// %K line (fast stochastic).
    pub k: Vec<f64>,
    /// %D line (slow stochastic, SMA of %K).
    pub d: Vec<f64>,
}

/// Stochastic Oscillator.
///
/// # Arguments
/// * `high` - High prices
/// * `low` - Low prices
/// * `close` - Close prices
/// * `k_period` - %K lookback period (default: 14)
/// * `d_period` - %D smoothing period (default: 3)
///
/// # Returns
/// StochasticResult with %K and %D lines (0-100 scale)
pub fn stochastic(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    k_period: usize,
    d_period: usize,
) -> Result<StochasticResult> {
    let n = close.len();
    if n != high.len() || n != low.len() {
        return Err(RaptorError::length_mismatch(n, high.len()));
    }
    if k_period == 0 || d_period == 0 {
        return Err(RaptorError::invalid_parameter("Stochastic periods must be > 0"));
    }

    let mut k = vec![f64::NAN; n];
    let mut d = vec![f64::NAN; n];

    if k_period > n {
        return Ok(StochasticResult { k, d });
    }

    // Calculate %K
    for i in (k_period - 1)..n {
        let start = i + 1 - k_period;

        // Find highest high and lowest low in window
        let mut highest_high = f64::NEG_INFINITY;
        let mut lowest_low = f64::INFINITY;
        for j in start..=i {
            if high[j] > highest_high {
                highest_high = high[j];
            }
            if low[j] < lowest_low {
                lowest_low = low[j];
            }
        }

        let range = highest_high - lowest_low;
        if range > 0.0 {
            k[i] = ((close[i] - lowest_low) / range) * 100.0;
        } else {
            k[i] = 50.0; // Default to middle when range is zero
        }
    }

    // Calculate %D (SMA of %K)
    let d_start = k_period - 1 + d_period - 1;
    if d_start < n {
        for i in d_start..n {
            let start = i + 1 - d_period;
            let mut sum = 0.0;
            let mut count = 0;
            for j in start..=i {
                if !k[j].is_nan() {
                    sum += k[j];
                    count += 1;
                }
            }
            if count == d_period {
                d[i] = sum / d_period as f64;
            }
        }
    }

    Ok(StochasticResult { k, d })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rsi() {
        // Test with simple increasing data
        let data = vec![
            44.0, 44.25, 44.5, 43.75, 44.5, 44.25, 44.0, 44.0, 44.25, 45.0, 45.5, 46.0, 46.5, 47.0,
            47.5,
        ];
        let result = rsi(&data, 14).unwrap();

        // RSI should be valid from index 14
        assert!(result[13].is_nan());
        assert!(!result[14].is_nan());
        assert!(result[14] >= 0.0 && result[14] <= 100.0);
    }

    #[test]
    fn test_macd() {
        let data: Vec<f64> = (1..=50).map(|x| x as f64).collect();
        let result = macd(&data, 12, 26, 9).unwrap();

        // MACD line should be valid from index 25 (slow_period - 1)
        assert!(result.macd_line[24].is_nan());
        assert!(!result.macd_line[25].is_nan());

        // Signal line should be valid later
        assert!(result.signal_line[33].is_nan());
        assert!(!result.signal_line[34].is_nan());
    }

    #[test]
    fn test_stochastic() {
        let high = vec![50.0, 51.0, 52.0, 51.5, 50.5, 51.0, 52.0, 53.0, 52.5, 51.5];
        let low = vec![48.0, 49.0, 50.0, 49.5, 48.5, 49.0, 50.0, 51.0, 50.5, 49.5];
        let close = vec![49.0, 50.0, 51.0, 50.0, 49.0, 50.0, 51.0, 52.0, 51.0, 50.0];

        let result = stochastic(&high, &low, &close, 5, 3).unwrap();

        // %K should be valid from index 4
        assert!(result.k[3].is_nan());
        assert!(!result.k[4].is_nan());
        assert!(result.k[4] >= 0.0 && result.k[4] <= 100.0);

        // %D should be valid from index 6
        assert!(result.d[5].is_nan());
        assert!(!result.d[6].is_nan());
    }
}
