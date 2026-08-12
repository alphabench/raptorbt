//! Tick-level signal generation for momentum entry/exit.
//!
//! Converts precomputed feature arrays (one scalar per tick) into entry and
//! exit boolean arrays that can be fed directly into `run_tick_backtest`.
//!
//! All functions are O(N) single-pass — no backward linear search, no nested
//! loops. The return_1m feature array must be precomputed by the caller
//! (via `tick_features::return_window` or equivalent).

/// Generate momentum entry signals from per-tick feature arrays.
///
/// All input slices must have the same length N.
///
/// Rules applied in order (a failing rule sets entry[i] = false):
///   1. spread gate: `spread_pct[i] <= spread_pct_max`
///   2. BSI gate: if `bsi_min > 0.0`, `bsi_delta[i] >= bsi_min`
///   3. return gate: if `return_1m_min_abs > 0.0`, direction-aligned
///      `return_1m[i]` must have `abs >= return_1m_min_abs` and correct sign.
///      NaN return_1m always fails the gate.
///   4. cooldown: after each entry, suppress the next `cooldown_ticks` ticks.
///
/// `return_direction`: +1 for long (return_1m must be positive), -1 for short
/// (return_1m must be negative).
// Three parallel series plus the five thresholds that gate them. Grouping
// the thresholds would separate them from the series they apply to.
#[allow(clippy::too_many_arguments)]
pub fn tick_momentum_entry(
    spread_pct: &[f64],
    bsi_delta: &[f64],
    return_1m: &[f64],
    spread_pct_max: f64,
    bsi_min: f64,
    return_1m_min_abs: f64,
    return_direction: i8,
    cooldown_ticks: usize,
) -> Vec<bool> {
    let n = spread_pct.len();
    let mut entries = vec![false; n];
    let mut cooldown_until: usize = 0;

    for i in 0..n {
        if i < cooldown_until {
            continue;
        }

        // Spread gate
        if spread_pct[i] > spread_pct_max {
            continue;
        }

        // BSI delta gate (disabled when bsi_min == 0.0)
        if bsi_min > 0.0 {
            let b = if i < bsi_delta.len() { bsi_delta[i] } else { continue };
            if b < bsi_min {
                continue;
            }
        }

        // 1-minute return gate (disabled when return_1m_min_abs == 0.0)
        if return_1m_min_abs > 0.0 {
            let r = if i < return_1m.len() { return_1m[i] } else { continue };
            if r.is_nan() {
                continue;
            }
            let abs_r = r.abs();
            if abs_r < return_1m_min_abs {
                continue;
            }
            // Direction alignment: long needs positive return, short needs negative
            if return_direction > 0 && r < 0.0 {
                continue;
            }
            if return_direction < 0 && r > 0.0 {
                continue;
            }
        }

        entries[i] = true;
        cooldown_until = i + 1 + cooldown_ticks;
    }

    entries
}

/// Generate time-based exit signals (EOD / session-end).
///
/// Sets exit[i] = true for every tick at or after `eod_exit_time_ns`.
/// When `eod_exit_time_ns == 0` all exits are false (disabled).
///
/// `timestamps_ns`: nanoseconds-since-epoch timestamp for each tick.
pub fn tick_momentum_exit(timestamps_ns: &[i64], eod_exit_time_ns: i64) -> Vec<bool> {
    let n = timestamps_ns.len();
    if eod_exit_time_ns == 0 {
        return vec![false; n];
    }
    timestamps_ns.iter().map(|&ts| ts >= eod_exit_time_ns).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entry_spread_gate() {
        // All spreads above max → no entries
        let spread = vec![3.0, 4.0, 6.0];
        let bsi = vec![0.6, 0.7, 0.8];
        let ret = vec![1.0, 1.0, 1.0];
        let entries = tick_momentum_entry(&spread, &bsi, &ret, 2.0, 0.0, 0.0, 1, 0);
        assert_eq!(entries, vec![false, false, false]);
    }

    #[test]
    fn test_entry_bsi_gate() {
        let spread = vec![1.0, 1.0, 1.0];
        let bsi = vec![0.3, 0.6, 0.4]; // only index 1 passes bsi_min=0.5
        let ret = vec![0.5, 0.5, 0.5];
        let entries = tick_momentum_entry(&spread, &bsi, &ret, 5.0, 0.5, 0.0, 1, 0);
        assert_eq!(entries, vec![false, true, false]);
    }

    #[test]
    fn test_entry_return_gate_long() {
        let spread = vec![1.0, 1.0, 1.0, 1.0];
        let bsi = vec![0.6, 0.6, 0.6, 0.6];
        // positive, positive, too small, negative
        let ret = vec![0.5, 1.0, 0.1, -0.5];
        let entries = tick_momentum_entry(&spread, &bsi, &ret, 5.0, 0.0, 0.3, 1, 0);
        assert_eq!(entries, vec![true, true, false, false]);
    }

    #[test]
    fn test_entry_return_gate_short() {
        let spread = vec![1.0, 1.0, 1.0];
        let bsi = vec![0.6, 0.6, 0.6];
        // negative enough, positive (fails direction), nan
        let ret = vec![-0.5, 0.5, f64::NAN];
        let entries = tick_momentum_entry(&spread, &bsi, &ret, 5.0, 0.0, 0.3, -1, 0);
        assert_eq!(entries, vec![true, false, false]);
    }

    #[test]
    fn test_entry_cooldown() {
        // cooldown_ticks=2: after entry at i=0, next eligible at i=3
        let spread = vec![1.0; 6];
        let bsi = vec![0.6; 6];
        let ret = vec![0.0; 6];
        let entries = tick_momentum_entry(&spread, &bsi, &ret, 5.0, 0.0, 0.0, 1, 2);
        assert!(entries[0]);
        assert!(!entries[1]);
        assert!(!entries[2]);
        assert!(entries[3]);
        assert!(!entries[4]);
        assert!(!entries[5]);
    }

    #[test]
    fn test_exit_disabled() {
        let ts = vec![1_000_000_i64, 2_000_000, 3_000_000];
        let exits = tick_momentum_exit(&ts, 0);
        assert_eq!(exits, vec![false, false, false]);
    }

    #[test]
    fn test_exit_eod_fires() {
        let ts = vec![1_000_i64, 2_000, 3_000, 4_000];
        let exits = tick_momentum_exit(&ts, 3_000);
        assert_eq!(exits, vec![false, false, true, true]);
    }
}
