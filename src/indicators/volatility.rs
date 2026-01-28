//! Volatility indicators: ATR, Bollinger Bands.

use super::trend::sma;
use crate::core::error::RaptorError;
use crate::core::Result;

/// Average True Range (ATR).
///
/// # Arguments
/// * `high` - High prices
/// * `low` - Low prices
/// * `close` - Close prices
/// * `period` - Lookback period (default: 14)
///
/// # Returns
/// Vector of ATR values (NaN for warmup period)
pub fn atr(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Result<Vec<f64>> {
    let n = close.len();
    if n != high.len() || n != low.len() {
        return Err(RaptorError::length_mismatch(n, high.len()));
    }
    if period == 0 {
        return Err(RaptorError::invalid_parameter("ATR period must be > 0"));
    }

    let mut result = vec![f64::NAN; n];

    if period >= n {
        return Ok(result);
    }

    // Calculate True Range
    let mut tr = vec![0.0; n];
    tr[0] = high[0] - low[0]; // First TR is just high - low

    for i in 1..n {
        let hl = high[i] - low[i];
        let hc = (high[i] - close[i - 1]).abs();
        let lc = (low[i] - close[i - 1]).abs();
        tr[i] = hl.max(hc).max(lc);
    }

    // Calculate initial ATR using SMA of first 'period' TR values
    let initial_atr: f64 = tr[..period].iter().sum::<f64>() / period as f64;
    result[period - 1] = initial_atr;

    // Use Wilder's smoothing (exponential) for remaining values
    let alpha = 1.0 / period as f64;
    for i in period..n {
        result[i] = alpha * tr[i] + (1.0 - alpha) * result[i - 1];
    }

    Ok(result)
}

/// True Range calculation (single bar).
#[inline]
pub fn true_range(high: f64, low: f64, prev_close: f64) -> f64 {
    let hl = high - low;
    let hc = (high - prev_close).abs();
    let lc = (low - prev_close).abs();
    hl.max(hc).max(lc)
}

/// Bollinger Bands result.
#[derive(Debug, Clone)]
pub struct BollingerBandsResult {
    /// Middle band (SMA).
    pub middle: Vec<f64>,
    /// Upper band (SMA + std_dev * multiplier).
    pub upper: Vec<f64>,
    /// Lower band (SMA - std_dev * multiplier).
    pub lower: Vec<f64>,
    /// Bandwidth: (upper - lower) / middle.
    pub bandwidth: Vec<f64>,
    /// %B: (price - lower) / (upper - lower).
    pub percent_b: Vec<f64>,
}

/// Bollinger Bands.
///
/// # Arguments
/// * `data` - Price data (typically close prices)
/// * `period` - Lookback period (default: 20)
/// * `std_dev` - Standard deviation multiplier (default: 2.0)
///
/// # Returns
/// BollingerBandsResult with middle, upper, lower bands, bandwidth, and %B
pub fn bollinger_bands(data: &[f64], period: usize, std_dev: f64) -> Result<BollingerBandsResult> {
    if period == 0 {
        return Err(RaptorError::invalid_parameter(
            "Bollinger Bands period must be > 0",
        ));
    }
    if std_dev <= 0.0 {
        return Err(RaptorError::invalid_parameter(
            "Bollinger Bands std_dev must be > 0",
        ));
    }

    let n = data.len();
    let mut middle = vec![f64::NAN; n];
    let mut upper = vec![f64::NAN; n];
    let mut lower = vec![f64::NAN; n];
    let mut bandwidth = vec![f64::NAN; n];
    let mut percent_b = vec![f64::NAN; n];

    if period > n {
        return Ok(BollingerBandsResult {
            middle,
            upper,
            lower,
            bandwidth,
            percent_b,
        });
    }

    // Calculate SMA for middle band
    middle = sma(data, period)?;

    // Calculate standard deviation and bands
    for i in (period - 1)..n {
        let mean = middle[i];

        // Skip if mean is NaN (warmup period)
        if mean.is_nan() {
            continue;
        }

        let start = i + 1 - period;

        // Calculate standard deviation using population variance
        let variance: f64 = data[start..=i]
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>()
            / period as f64;
        let std = variance.sqrt();

        // Calculate bands (std is always non-negative from sqrt)
        upper[i] = mean + std_dev * std;
        lower[i] = mean - std_dev * std;

        // Calculate bandwidth (as percentage of middle)
        if mean.abs() > f64::EPSILON {
            bandwidth[i] = (upper[i] - lower[i]) / mean.abs();
        }

        // Calculate %B (position within bands)
        let band_width = upper[i] - lower[i];
        if band_width > f64::EPSILON {
            percent_b[i] = (data[i] - lower[i]) / band_width;
        }
    }

    Ok(BollingerBandsResult {
        middle,
        upper,
        lower,
        bandwidth,
        percent_b,
    })
}

/// Keltner Channels (ATR-based bands).
///
/// # Arguments
/// * `high` - High prices
/// * `low` - Low prices
/// * `close` - Close prices
/// * `ema_period` - EMA period for middle band
/// * `atr_period` - ATR period
/// * `multiplier` - ATR multiplier
///
/// # Returns
/// Tuple of (middle, upper, lower) bands
pub fn keltner_channels(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    ema_period: usize,
    atr_period: usize,
    multiplier: f64,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    let n = close.len();
    if n != high.len() || n != low.len() {
        return Err(RaptorError::length_mismatch(n, high.len()));
    }

    // Calculate EMA for middle band
    let middle = super::trend::ema(close, ema_period)?;

    // Calculate ATR
    let atr_values = atr(high, low, close, atr_period)?;

    // Calculate bands
    let mut upper = vec![f64::NAN; n];
    let mut lower = vec![f64::NAN; n];

    for i in 0..n {
        if !middle[i].is_nan() && !atr_values[i].is_nan() {
            upper[i] = middle[i] + multiplier * atr_values[i];
            lower[i] = middle[i] - multiplier * atr_values[i];
        }
    }

    Ok((middle, upper, lower))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atr() {
        let high = vec![50.0, 51.0, 52.0, 51.5, 50.5, 51.0, 52.0, 53.0, 52.5, 51.5];
        let low = vec![48.0, 49.0, 50.0, 49.5, 48.5, 49.0, 50.0, 51.0, 50.5, 49.5];
        let close = vec![49.0, 50.0, 51.0, 50.0, 49.0, 50.0, 51.0, 52.0, 51.0, 50.0];

        let result = atr(&high, &low, &close, 5).unwrap();

        // ATR should be valid from index 4
        assert!(result[3].is_nan());
        assert!(!result[4].is_nan());
        assert!(result[4] > 0.0);
    }

    #[test]
    fn test_bollinger_bands() {
        let data: Vec<f64> = (1..=30)
            .map(|x| x as f64 + (x as f64 * 0.1).sin())
            .collect();

        let result = bollinger_bands(&data, 20, 2.0).unwrap();

        // Bands should be valid from index 19
        assert!(result.middle[18].is_nan());
        assert!(!result.middle[19].is_nan());

        // Upper > Middle > Lower
        assert!(result.upper[19] > result.middle[19]);
        assert!(result.middle[19] > result.lower[19]);

        // %B should be between 0 and 1 for data within bands
        assert!(result.percent_b[19] >= -0.5 && result.percent_b[19] <= 1.5);
    }

    #[test]
    fn test_true_range() {
        // Simple case
        assert!((true_range(52.0, 48.0, 50.0) - 4.0).abs() < 1e-10);

        // Gap up case
        assert!((true_range(55.0, 53.0, 50.0) - 5.0).abs() < 1e-10);

        // Gap down case
        assert!((true_range(48.0, 45.0, 50.0) - 5.0).abs() < 1e-10);
    }
}
