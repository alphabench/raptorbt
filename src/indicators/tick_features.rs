//! Tick-level feature extraction functions.
//!
//! All functions accept parallel arrays (one element per tick) and return a
//! Vec<f64> of the same length. NaN is used where the feature is undefined
//! (e.g. insufficient history for a lookback window).
//!
//! These are building blocks for the signal generation layer — compute features
//! once on the full tick window, then pass the resulting arrays to
//! `tick_signals::tick_momentum_entry`.

/// Per-tick bid/ask spread as a percentage of the mid price.
///
/// Returns 0.0 where both bid and ask are zero.
pub fn spread_pct(bid: &[f64], ask: &[f64]) -> Vec<f64> {
    bid.iter()
        .zip(ask.iter())
        .map(|(&b, &a)| {
            let mid = (b + a) / 2.0;
            if mid > 0.0 {
                (a - b) / mid * 100.0
            } else {
                0.0
            }
        })
        .collect()
}

/// Per-tick delta BSI from Zerodha cumulative session totals.
///
/// Zerodha's `total_buy_qty` / `total_sell_qty` are running sums that grow
/// monotonically from market open. Computing BSI from raw cumulative values
/// yields ~0.95 for the whole day (artefact of early-session buy-side dominance).
///
/// This function computes the imbalance of the most recent tick's activity only:
///   `bsi[i] = Δbuy[i] / (Δbuy[i] + Δsell[i])` where `Δbuy[i] = max(0, buy[i] - buy[i-1])`
///
/// Returns 0.5 (neutral) where the total delta is zero (no activity).
pub fn buy_sell_imbalance_delta(
    buy_qty_cumulative: &[f64],
    sell_qty_cumulative: &[f64],
) -> Vec<f64> {
    let n = buy_qty_cumulative.len();
    let mut out = vec![0.5_f64; n];
    for i in 1..n {
        let db = (buy_qty_cumulative[i] - buy_qty_cumulative[i - 1]).max(0.0);
        let ds = (sell_qty_cumulative[i] - sell_qty_cumulative[i - 1]).max(0.0);
        let total = db + ds;
        if total > 0.0 {
            out[i] = db / total;
        }
    }
    out
}

/// Per-tick lookback return over a fixed time window.
///
/// For each tick i, finds the latest tick whose timestamp is at most
/// `timestamps_ns[i] - window_seconds * 1e9` and computes:
///   `(ltp[i] - ltp_ref) / ltp_ref * 100`
///
/// Returns `f64::NAN` for ticks where no reference tick exists (start of series
/// or insufficient history).
///
/// Uses binary search → O(N log N) total.
pub fn return_window(timestamps_ns: &[i64], ltp: &[f64], window_seconds: f64) -> Vec<f64> {
    let n = timestamps_ns.len();
    let window_ns = (window_seconds * 1_000_000_000.0) as i64;
    let mut out = vec![f64::NAN; n];

    for i in 0..n {
        let cutoff = timestamps_ns[i] - window_ns;
        // Binary search for the last index with ts <= cutoff
        let pos = timestamps_ns[..i].partition_point(|&ts| ts <= cutoff);
        // pos is the first index > cutoff; we want pos.saturating_sub(1)
        if pos > 0 {
            let ref_idx = pos - 1;
            let ltp_ref = ltp[ref_idx];
            if ltp_ref > 0.0 {
                out[i] = (ltp[i] - ltp_ref) / ltp_ref * 100.0;
            }
        }
    }
    out
}

/// Rolling realized volatility proxy: annualized stddev of log returns.
///
/// For each tick i, computes stddev of log-returns over all ticks within
/// the preceding `window_seconds`. Returns `f64::NAN` if fewer than 2 ticks
/// in the window.
///
/// O(N²) worst case but typical windows are short (60–300 s at ~80 ticks/min
/// = 80–400 ticks), making the inner loop fast in practice.
pub fn realized_vol_rolling(timestamps_ns: &[i64], ltp: &[f64], window_seconds: f64) -> Vec<f64> {
    let n = timestamps_ns.len();
    let window_ns = (window_seconds * 1_000_000_000.0) as i64;
    let mut out = vec![f64::NAN; n];

    for i in 1..n {
        let cutoff = timestamps_ns[i] - window_ns;
        // Find the first tick inside the window
        let start = timestamps_ns[..i].partition_point(|&ts| ts < cutoff);
        // We need log returns from start..=i
        let count = i - start;
        if count < 1 {
            continue;
        }
        let mut log_rets = Vec::with_capacity(count);
        for j in (start + 1)..=i {
            if ltp[j - 1] > 0.0 {
                log_rets.push((ltp[j] / ltp[j - 1]).ln());
            }
        }
        if log_rets.len() < 2 {
            continue;
        }
        let mean = log_rets.iter().sum::<f64>() / log_rets.len() as f64;
        let variance =
            log_rets.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (log_rets.len() - 1) as f64;
        out[i] = variance.sqrt() * 100.0; // as percentage of price
    }
    out
}

/// Per-tick OI position within the day's high/low range.
///
/// Returns `(oi[i] - oi_day_low) / (oi_day_high - oi_day_low) * 100` ∈ [0, 100].
/// Returns `f64::NAN` where `oi_day_high <= oi_day_low`.
pub fn oi_position_pct(oi: &[f64], oi_day_high: f64, oi_day_low: f64) -> Vec<f64> {
    let range = oi_day_high - oi_day_low;
    if range <= 0.0 {
        return vec![f64::NAN; oi.len()];
    }
    oi.iter().map(|&o| (o - oi_day_low) / range * 100.0).collect()
}

/// Rolling tick velocity: number of ticks per minute in the preceding window.
///
/// For each tick i, counts ticks in (timestamps_ns[i] - window_seconds*1e9, timestamps_ns[i]].
/// Returns 0.0 for the first tick.
pub fn tick_velocity(timestamps_ns: &[i64], window_seconds: f64) -> Vec<f64> {
    let n = timestamps_ns.len();
    let window_ns = (window_seconds * 1_000_000_000.0) as i64;
    let mut out = vec![0.0_f64; n];

    for i in 1..n {
        let cutoff = timestamps_ns[i] - window_ns;
        let start = timestamps_ns[..i].partition_point(|&ts| ts <= cutoff);
        let count = (i - start + 1) as f64; // include current tick
        let minutes = window_seconds / 60.0;
        out[i] = if minutes > 0.0 { count / minutes } else { 0.0 };
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spread_pct_basic() {
        let bid = vec![100.0, 200.0];
        let ask = vec![101.0, 202.0];
        let s = spread_pct(&bid, &ask);
        // (101-100)/100.5 * 100 ≈ 0.995
        assert!((s[0] - 0.9950248756218905).abs() < 1e-9);
        // (202-200)/201 * 100 ≈ 0.995
        assert!((s[1] - 0.9950248756218905).abs() < 1e-9);
    }

    #[test]
    fn test_spread_pct_zero_bid_ask() {
        let bid = vec![0.0];
        let ask = vec![0.0];
        let s = spread_pct(&bid, &ask);
        assert_eq!(s[0], 0.0);
    }

    #[test]
    fn test_bsi_delta_basic() {
        // Cumulative: buy grows by 100, sell by 0 → bsi = 1.0
        let buy = vec![1000.0, 1100.0, 1100.0, 1150.0];
        let sell = vec![800.0, 800.0, 850.0, 850.0];
        let bsi = buy_sell_imbalance_delta(&buy, &sell);
        assert_eq!(bsi[0], 0.5); // first tick always neutral
        assert_eq!(bsi[1], 1.0); // all buy
        assert_eq!(bsi[2], 0.0); // all sell
        assert_eq!(bsi[3], 1.0); // all buy
    }

    #[test]
    fn test_bsi_delta_no_activity() {
        // No change → neutral 0.5
        let buy = vec![1000.0, 1000.0];
        let sell = vec![800.0, 800.0];
        let bsi = buy_sell_imbalance_delta(&buy, &sell);
        assert_eq!(bsi[1], 0.5);
    }

    #[test]
    fn test_return_window_basic() {
        // Ticks at 0s, 30s, 61s, 90s (nanoseconds)
        let sec = 1_000_000_000_i64;
        let ts = vec![0, 30 * sec, 61 * sec, 90 * sec];
        let ltp = vec![100.0, 102.0, 101.0, 105.0];
        let ret = return_window(&ts, &ltp, 60.0);
        // ts[0]: no history → NAN
        assert!(ret[0].is_nan());
        // ts[1] at 30s: no tick <= -30s → NAN
        assert!(ret[1].is_nan());
        // ts[2] at 61s: cutoff = 1s, ts[0]=0 ≤ 1s → ref = ltp[0]=100.0
        // (101 - 100) / 100 * 100 = 1.0
        assert!((ret[2] - 1.0).abs() < 1e-9);
        // ts[3] at 90s: cutoff = 30s, ts[1]=30s ≤ 30s → ref = ltp[1]=102.0
        // (105 - 102) / 102 * 100 ≈ 2.941
        assert!((ret[3] - (3.0 / 102.0 * 100.0)).abs() < 1e-9);
    }

    #[test]
    fn test_oi_position_pct() {
        let oi = vec![50.0, 100.0, 150.0];
        let result = oi_position_pct(&oi, 200.0, 0.0);
        assert_eq!(result, vec![25.0, 50.0, 75.0]);
    }

    #[test]
    fn test_oi_position_pct_no_range() {
        let oi = vec![100.0, 100.0];
        let result = oi_position_pct(&oi, 100.0, 100.0);
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
    }

    #[test]
    fn test_tick_velocity_basic() {
        // 4 ticks at 0s, 10s, 20s, 30s; window=60s
        let sec = 1_000_000_000_i64;
        let ts = vec![0, 10 * sec, 20 * sec, 30 * sec];
        let vel = tick_velocity(&ts, 60.0);
        // At i=3 (30s): ticks in (−30s, 30s] = all 4 → 4 ticks / 1 min = 4.0
        assert_eq!(vel[0], 0.0);
        assert!((vel[3] - 4.0).abs() < 1e-9);
    }
}
