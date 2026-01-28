//! Integration tests for RaptorBT indicators.

use raptorbt::indicators::momentum::{macd, rsi, stochastic};
use raptorbt::indicators::strength::adx;
use raptorbt::indicators::trend::{ema, sma, supertrend};
use raptorbt::indicators::volatility::{atr, bollinger_bands};
use raptorbt::indicators::volume::vwap;

fn sample_ohlcv() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    // Create sample OHLCV data with 50 bars
    let n = 50;
    let mut close: Vec<f64> = vec![100.0];
    let mut high: Vec<f64> = vec![101.0];
    let mut low: Vec<f64> = vec![99.0];
    let mut open: Vec<f64> = vec![100.0];
    let volume: Vec<f64> = vec![1000.0; n];

    // Generate trending data
    for i in 1..n {
        let prev_close = close[i - 1];
        let change = ((i as f64 * 0.2).sin() * 2.0) + 0.5; // Slight uptrend with oscillation
        let new_close = prev_close + change;
        close.push(new_close);
        open.push(prev_close);
        high.push(new_close.max(prev_close) + 0.5);
        low.push(new_close.min(prev_close) - 0.5);
    }

    (open, high, low, close, volume)
}

#[test]
fn test_sma_correctness() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let result = sma(&data, 3).unwrap();

    // First 2 values should be NaN
    assert!(result[0].is_nan());
    assert!(result[1].is_nan());

    // SMA(3) for [1,2,3] = 2.0
    assert!((result[2] - 2.0).abs() < 1e-10);
    // SMA(3) for [2,3,4] = 3.0
    assert!((result[3] - 3.0).abs() < 1e-10);
    // SMA(3) for [8,9,10] = 9.0
    assert!((result[9] - 9.0).abs() < 1e-10);
}

#[test]
fn test_ema_correctness() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let result = ema(&data, 3).unwrap();

    // First 2 values should be NaN
    assert!(result[0].is_nan());
    assert!(result[1].is_nan());

    // EMA should be valid from index 2
    assert!(!result[2].is_nan());
    assert!(!result[9].is_nan());

    // EMA should be between min and max
    assert!(result[9] >= 1.0 && result[9] <= 10.0);
}

#[test]
fn test_rsi_range() {
    let (_, _, _, close, _) = sample_ohlcv();
    let result = rsi(&close, 14).unwrap();

    // Check RSI is in valid range [0, 100]
    for (i, &value) in result.iter().enumerate() {
        if !value.is_nan() {
            assert!(
                value >= 0.0 && value <= 100.0,
                "RSI at index {} is out of range: {}",
                i,
                value
            );
        }
    }
}

#[test]
fn test_macd_structure() {
    let (_, _, _, close, _) = sample_ohlcv();
    let result = macd(&close, 12, 26, 9).unwrap();

    assert_eq!(result.macd_line.len(), close.len());
    assert_eq!(result.signal_line.len(), close.len());
    assert_eq!(result.histogram.len(), close.len());

    // MACD line should be valid from index 25 (slow_period - 1)
    assert!(result.macd_line[24].is_nan());
    assert!(!result.macd_line[25].is_nan());
}

#[test]
fn test_stochastic_range() {
    let (_, high, low, close, _) = sample_ohlcv();
    let result = stochastic(&high, &low, &close, 14, 3).unwrap();

    // %K and %D should be in [0, 100]
    for (i, &k) in result.k.iter().enumerate() {
        if !k.is_nan() {
            assert!(
                k >= 0.0 && k <= 100.0,
                "%K at index {} is out of range: {}",
                i,
                k
            );
        }
    }

    for (i, &d) in result.d.iter().enumerate() {
        if !d.is_nan() {
            assert!(
                d >= 0.0 && d <= 100.0,
                "%D at index {} is out of range: {}",
                i,
                d
            );
        }
    }
}

#[test]
fn test_atr_positive() {
    let (_, high, low, close, _) = sample_ohlcv();
    let result = atr(&high, &low, &close, 14).unwrap();

    // ATR should always be non-negative
    for (i, &value) in result.iter().enumerate() {
        if !value.is_nan() {
            assert!(value >= 0.0, "ATR at index {} is negative: {}", i, value);
        }
    }
}

#[test]
fn test_bollinger_bands_ordering() {
    let (_, _, _, close, _) = sample_ohlcv();
    let result = bollinger_bands(&close, 20, 2.0).unwrap();

    // Upper > Middle > Lower
    for i in 19..close.len() {
        if !result.upper[i].is_nan() {
            assert!(
                result.upper[i] >= result.middle[i],
                "Upper band should be >= middle at index {}",
                i
            );
            assert!(
                result.middle[i] >= result.lower[i],
                "Middle band should be >= lower at index {}",
                i
            );
        }
    }
}

#[test]
fn test_adx_range() {
    let (_, high, low, close, _) = sample_ohlcv();
    let result = adx(&high, &low, &close, 14).unwrap();

    // ADX should be in [0, 100]
    for (i, &value) in result.iter().enumerate() {
        if !value.is_nan() {
            assert!(
                value >= 0.0 && value <= 100.0,
                "ADX at index {} is out of range: {}",
                i,
                value
            );
        }
    }
}

#[test]
fn test_vwap_bounds() {
    let (_, high, low, close, volume) = sample_ohlcv();
    let result = vwap(&high, &low, &close, &volume).unwrap();

    // VWAP should be between the overall min low and max high
    let min_low = low.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_high = high.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    for (i, &value) in result.iter().enumerate() {
        if !value.is_nan() {
            assert!(
                value >= min_low && value <= max_high,
                "VWAP at index {} is out of bounds: {} (should be between {} and {})",
                i,
                value,
                min_low,
                max_high
            );
        }
    }
}

#[test]
fn test_supertrend_direction() {
    let (_, high, low, close, _) = sample_ohlcv();
    let result = supertrend(&high, &low, &close, 10, 3.0).unwrap();

    // Direction should be either 1 or -1
    for (i, &dir) in result.direction.iter().enumerate() {
        if dir != 0 {
            assert!(
                dir == 1 || dir == -1,
                "Supertrend direction at index {} is invalid: {}",
                i,
                dir
            );
        }
    }
}

#[test]
fn test_invalid_period() {
    let data = vec![1.0, 2.0, 3.0];

    // Period of 0 should error
    assert!(sma(&data, 0).is_err());
    assert!(ema(&data, 0).is_err());
    assert!(rsi(&data, 0).is_err());
}

#[test]
fn test_empty_data() {
    let empty: Vec<f64> = vec![];

    let result = sma(&empty, 10).unwrap();
    assert!(result.is_empty());

    let result = ema(&empty, 10).unwrap();
    assert!(result.is_empty());
}
