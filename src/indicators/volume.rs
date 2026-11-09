//! Volume indicators: VWAP, OBV.

use crate::core::error::RaptorError;
use crate::core::Result;

/// Volume Weighted Average Price (VWAP).
///
/// # Arguments
/// * `high` - High prices
/// * `low` - Low prices
/// * `close` - Close prices
/// * `volume` - Volume data
///
/// # Returns
/// Vector of VWAP values
pub fn vwap(high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> Result<Vec<f64>> {
    let n = close.len();
    if n != high.len() || n != low.len() || n != volume.len() {
        return Err(RaptorError::length_mismatch(n, high.len()));
    }

    if n == 0 {
        return Ok(vec![]);
    }

    let mut result = vec![f64::NAN; n];
    let mut cumulative_tp_vol = 0.0;
    let mut cumulative_vol = 0.0;

    for i in 0..n {
        // Typical price
        let tp = (high[i] + low[i] + close[i]) / 3.0;

        cumulative_tp_vol += tp * volume[i];
        cumulative_vol += volume[i];

        if cumulative_vol > 0.0 {
            result[i] = cumulative_tp_vol / cumulative_vol;
        }
    }

    Ok(result)
}

/// VWAP with session reset (e.g., daily reset).
///
/// # Arguments
/// * `high` - High prices
/// * `low` - Low prices
/// * `close` - Close prices
/// * `volume` - Volume data
/// * `session_starts` - Boolean array indicating session start (true = reset VWAP)
///
/// # Returns
/// Vector of VWAP values with session resets
pub fn vwap_session(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    session_starts: &[bool],
) -> Result<Vec<f64>> {
    let n = close.len();
    if n != high.len() || n != low.len() || n != volume.len() || n != session_starts.len() {
        return Err(RaptorError::length_mismatch(n, high.len()));
    }

    if n == 0 {
        return Ok(vec![]);
    }

    let mut result = vec![f64::NAN; n];
    let mut cumulative_tp_vol = 0.0;
    let mut cumulative_vol = 0.0;

    for i in 0..n {
        // Reset on session start
        if session_starts[i] {
            cumulative_tp_vol = 0.0;
            cumulative_vol = 0.0;
        }

        // Typical price
        let tp = (high[i] + low[i] + close[i]) / 3.0;

        cumulative_tp_vol += tp * volume[i];
        cumulative_vol += volume[i];

        if cumulative_vol > 0.0 {
            result[i] = cumulative_tp_vol / cumulative_vol;
        }
    }

    Ok(result)
}

/// On Balance Volume (OBV).
///
/// # Arguments
/// * `close` - Close prices
/// * `volume` - Volume data
///
/// # Returns
/// Vector of OBV values
pub fn obv(close: &[f64], volume: &[f64]) -> Result<Vec<f64>> {
    let n = close.len();
    if n != volume.len() {
        return Err(RaptorError::length_mismatch(n, volume.len()));
    }

    if n == 0 {
        return Ok(vec![]);
    }

    let mut result = vec![0.0; n];
    result[0] = volume[0];

    for i in 1..n {
        if close[i] > close[i - 1] {
            result[i] = result[i - 1] + volume[i];
        } else if close[i] < close[i - 1] {
            result[i] = result[i - 1] - volume[i];
        } else {
            result[i] = result[i - 1];
        }
    }

    Ok(result)
}

/// Volume Rate of Change.
///
/// # Arguments
/// * `volume` - Volume data
/// * `period` - Lookback period
///
/// # Returns
/// Vector of volume rate of change values
pub fn volume_roc(volume: &[f64], period: usize) -> Result<Vec<f64>> {
    if period == 0 {
        return Err(RaptorError::invalid_parameter("Period must be > 0"));
    }

    let n = volume.len();
    let mut result = vec![f64::NAN; n];

    if period >= n {
        return Ok(result);
    }

    for i in period..n {
        if volume[i - period] != 0.0 {
            result[i] = (volume[i] - volume[i - period]) / volume[i - period] * 100.0;
        }
    }

    Ok(result)
}

/// Money Flow Index (volume-weighted RSI).
///
/// # Arguments
/// * `high` - High prices
/// * `low` - Low prices
/// * `close` - Close prices
/// * `volume` - Volume data
/// * `period` - Lookback period (default: 14)
///
/// # Returns
/// Vector of MFI values (0-100 scale)
pub fn mfi(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    period: usize,
) -> Result<Vec<f64>> {
    let n = close.len();
    if n != high.len() || n != low.len() || n != volume.len() {
        return Err(RaptorError::length_mismatch(n, high.len()));
    }
    if period == 0 {
        return Err(RaptorError::invalid_parameter("MFI period must be > 0"));
    }

    let mut result = vec![f64::NAN; n];

    if period >= n {
        return Ok(result);
    }

    // Calculate typical price and raw money flow
    let mut typical_price = vec![0.0; n];
    let mut raw_money_flow = vec![0.0; n];

    for i in 0..n {
        typical_price[i] = (high[i] + low[i] + close[i]) / 3.0;
        raw_money_flow[i] = typical_price[i] * volume[i];
    }

    // Calculate MFI for each period
    for (i, out) in result.iter_mut().enumerate().take(n).skip(period) {
        let mut positive_flow = 0.0;
        let mut negative_flow = 0.0;

        for (j, flow) in raw_money_flow.iter().enumerate().take(i + 1).skip(i - period + 1) {
            if typical_price[j] > typical_price[j - 1] {
                positive_flow += flow;
            } else if typical_price[j] < typical_price[j - 1] {
                negative_flow += flow;
            }
        }

        if negative_flow == 0.0 {
            *out = 100.0;
        } else {
            let money_ratio = positive_flow / negative_flow;
            *out = 100.0 - (100.0 / (1.0 + money_ratio));
        }
    }

    Ok(result)
}

/// Accumulation/Distribution Line.
///
/// # Arguments
/// * `high` - High prices
/// * `low` - Low prices
/// * `close` - Close prices
/// * `volume` - Volume data
///
/// # Returns
/// Vector of A/D line values
pub fn ad_line(high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> Result<Vec<f64>> {
    let n = close.len();
    if n != high.len() || n != low.len() || n != volume.len() {
        return Err(RaptorError::length_mismatch(n, high.len()));
    }

    if n == 0 {
        return Ok(vec![]);
    }

    let mut result = vec![0.0; n];

    for i in 0..n {
        let hl_range = high[i] - low[i];

        // Money Flow Multiplier
        let mfm = if hl_range > 0.0 {
            ((close[i] - low[i]) - (high[i] - close[i])) / hl_range
        } else {
            0.0
        };

        // Money Flow Volume
        let mfv = mfm * volume[i];

        // Accumulate
        if i == 0 {
            result[i] = mfv;
        } else {
            result[i] = result[i - 1] + mfv;
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vwap() {
        let high = vec![52.0, 53.0, 54.0, 53.0, 52.0];
        let low = vec![50.0, 51.0, 52.0, 51.0, 50.0];
        let close = vec![51.0, 52.0, 53.0, 52.0, 51.0];
        let volume = vec![1000.0, 1500.0, 2000.0, 1500.0, 1000.0];

        let result = vwap(&high, &low, &close, &volume).unwrap();

        // VWAP should be valid for all bars
        assert!(!result[0].is_nan());
        assert!(!result[4].is_nan());

        // VWAP should be between low and high range
        assert!(result[4] >= 50.0 && result[4] <= 54.0);
    }

    /// Two sessions of three bars; typical price tp = (h + l + c) / 3, all exact in binary.
    /// Session 1 (start at 0): tp 11, 12, 13 with volume 10, 30, 10.
    ///   bar 0: 110 / 10 = 11; bar 1: (110 + 360) / 40 = 11.75; bar 2: (470 + 130) / 50 = 12.
    /// Session 2 (start at 3, sums reset): tp 20, 21, 30 with volume 0, 5, 15.
    ///   bar 3: volume 0 so NaN; bar 4: 105 / 5 = 21; bar 5: (105 + 450) / 20 = 27.75.
    #[test]
    fn test_vwap_session_two_sessions_by_hand() {
        let high = vec![12.0, 13.0, 14.0, 21.0, 24.0, 33.0];
        let low = vec![10.0, 11.0, 12.0, 19.0, 18.0, 27.0];
        let close = vec![11.0, 12.0, 13.0, 20.0, 21.0, 30.0];
        let volume = vec![10.0, 30.0, 10.0, 0.0, 5.0, 15.0];
        let starts = vec![true, false, false, true, false, false];

        let r = vwap_session(&high, &low, &close, &volume, &starts).unwrap();

        assert_eq!(r[0], 11.0);
        assert_eq!(r[1], 11.75);
        assert_eq!(r[2], 12.0);
        assert!(r[3].is_nan());
        assert_eq!(r[4], 21.0);
        assert_eq!(r[5], 27.75);
    }

    /// One session start at index 0 is the cumulative VWAP, bit for bit.
    #[test]
    fn test_vwap_session_single_start_equals_vwap() {
        let high = vec![52.0, 53.1, 54.7, 53.3, 52.9, 0.0];
        let low = vec![50.0, 51.3, 52.2, 51.7, 50.1, 0.0];
        let close = vec![51.0, 52.7, 53.9, 52.1, 51.3, 0.0];
        let volume = vec![0.0, 1500.0, 2000.0, 1500.0, 1000.0, 0.0];
        let mut starts = vec![false; 6];
        starts[0] = true;

        let a = vwap(&high, &low, &close, &volume).unwrap();
        let b = vwap_session(&high, &low, &close, &volume, &starts).unwrap();

        for (x, y) in a.iter().zip(&b) {
            assert_eq!(x.to_bits(), y.to_bits());
        }
    }

    #[test]
    fn test_vwap_session_length_mismatch() {
        let x = vec![1.0, 2.0];
        assert!(vwap_session(&x, &x, &x, &x, &[true]).is_err());
    }

    #[test]
    fn test_obv() {
        let close = vec![50.0, 51.0, 50.5, 52.0, 51.0];
        let volume = vec![1000.0, 1500.0, 1200.0, 1800.0, 1300.0];

        let result = obv(&close, &volume).unwrap();

        // OBV starts with first volume
        assert!((result[0] - 1000.0).abs() < 1e-10);

        // Price up -> add volume
        assert!((result[1] - 2500.0).abs() < 1e-10);

        // Price down -> subtract volume
        assert!((result[2] - 1300.0).abs() < 1e-10);
    }

    #[test]
    fn test_mfi() {
        let high = vec![
            52.0, 53.0, 54.0, 53.0, 52.0, 53.0, 54.0, 55.0, 54.0, 53.0, 52.0, 53.0, 54.0, 55.0,
            56.0,
        ];
        let low = vec![
            50.0, 51.0, 52.0, 51.0, 50.0, 51.0, 52.0, 53.0, 52.0, 51.0, 50.0, 51.0, 52.0, 53.0,
            54.0,
        ];
        let close = vec![
            51.0, 52.0, 53.0, 52.0, 51.0, 52.0, 53.0, 54.0, 53.0, 52.0, 51.0, 52.0, 53.0, 54.0,
            55.0,
        ];
        let volume = vec![1000.0; 15];

        let result = mfi(&high, &low, &close, &volume, 14).unwrap();

        // MFI should be valid from index 14
        assert!(result[13].is_nan());
        assert!(!result[14].is_nan());
        assert!(result[14] >= 0.0 && result[14] <= 100.0);
    }
}
