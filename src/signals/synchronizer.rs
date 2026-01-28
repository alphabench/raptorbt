//! Signal synchronization for multi-instrument strategies.
//!
//! Handles combining signals from multiple instruments with different sync modes.

use crate::core::types::CompiledSignals;

/// Synchronization mode for combining signals from multiple instruments.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncMode {
    /// All instruments must signal (AND logic).
    All,
    /// Any instrument can signal (OR logic).
    Any,
    /// Majority of instruments must signal.
    Majority,
    /// Use first instrument's signals as master.
    Master,
}

impl Default for SyncMode {
    fn default() -> Self {
        SyncMode::All
    }
}

/// Signal synchronizer for multi-instrument backtests.
#[derive(Debug, Clone)]
pub struct SignalSynchronizer {
    /// Synchronization mode.
    pub mode: SyncMode,
    /// Minimum number of instruments that must signal (for custom thresholds).
    pub min_signals: Option<usize>,
}

impl Default for SignalSynchronizer {
    fn default() -> Self {
        Self { mode: SyncMode::All, min_signals: None }
    }
}

impl SignalSynchronizer {
    /// Create a new signal synchronizer with the given mode.
    pub fn new(mode: SyncMode) -> Self {
        Self { mode, min_signals: None }
    }

    /// Create a synchronizer with a custom minimum signal threshold.
    pub fn with_min_signals(min: usize) -> Self {
        Self { mode: SyncMode::Majority, min_signals: Some(min) }
    }

    /// Synchronize entry signals from multiple instruments.
    ///
    /// # Arguments
    /// * `signals` - Slice of signal arrays from each instrument
    ///
    /// # Returns
    /// Combined entry signals based on sync mode
    pub fn sync_entries(&self, signals: &[&[bool]]) -> Vec<bool> {
        if signals.is_empty() {
            return vec![];
        }

        let n = signals[0].len();
        for sig in signals.iter() {
            assert_eq!(sig.len(), n, "All signal arrays must have same length");
        }

        let num_instruments = signals.len();
        let mut result = vec![false; n];

        for i in 0..n {
            let count = signals.iter().filter(|s| s[i]).count();

            result[i] = match self.mode {
                SyncMode::All => count == num_instruments,
                SyncMode::Any => count > 0,
                SyncMode::Majority => {
                    let threshold = self.min_signals.unwrap_or((num_instruments + 1) / 2);
                    count >= threshold
                }
                SyncMode::Master => signals[0][i],
            };
        }

        result
    }

    /// Synchronize exit signals from multiple instruments.
    ///
    /// Exit logic is typically inverse of entry:
    /// - All mode -> exit on Any
    /// - Any mode -> exit on All
    /// - Majority mode -> exit when majority want to exit
    /// - Master mode -> use master's exit signals
    ///
    /// # Arguments
    /// * `signals` - Slice of signal arrays from each instrument
    ///
    /// # Returns
    /// Combined exit signals based on sync mode
    pub fn sync_exits(&self, signals: &[&[bool]]) -> Vec<bool> {
        if signals.is_empty() {
            return vec![];
        }

        let n = signals[0].len();
        for sig in signals.iter() {
            assert_eq!(sig.len(), n, "All signal arrays must have same length");
        }

        let num_instruments = signals.len();
        let mut result = vec![false; n];

        for i in 0..n {
            let count = signals.iter().filter(|s| s[i]).count();

            result[i] = match self.mode {
                // For All entry mode, exit when ANY wants to exit
                SyncMode::All => count > 0,
                // For Any entry mode, exit when ALL want to exit
                SyncMode::Any => count == num_instruments,
                SyncMode::Majority => {
                    let threshold = self.min_signals.unwrap_or((num_instruments + 1) / 2);
                    count >= threshold
                }
                SyncMode::Master => signals[0][i],
            };
        }

        result
    }

    /// Synchronize signals from CompiledSignals objects.
    ///
    /// # Arguments
    /// * `compiled_signals` - Slice of CompiledSignals from each instrument
    ///
    /// # Returns
    /// Tuple of (synchronized_entries, synchronized_exits)
    pub fn sync_compiled_signals(
        &self,
        compiled_signals: &[&CompiledSignals],
    ) -> (Vec<bool>, Vec<bool>) {
        if compiled_signals.is_empty() {
            return (vec![], vec![]);
        }

        let entries: Vec<&[bool]> =
            compiled_signals.iter().map(|cs| cs.entries.as_slice()).collect();

        let exits: Vec<&[bool]> = compiled_signals.iter().map(|cs| cs.exits.as_slice()).collect();

        let synced_entries = self.sync_entries(&entries);
        let synced_exits = self.sync_exits(&exits);

        (synced_entries, synced_exits)
    }

    /// Calculate signal agreement score (0.0 to 1.0).
    ///
    /// # Arguments
    /// * `signals` - Slice of signal arrays from each instrument
    ///
    /// # Returns
    /// Vector of agreement scores for each bar
    pub fn signal_agreement(&self, signals: &[&[bool]]) -> Vec<f64> {
        if signals.is_empty() {
            return vec![];
        }

        let n = signals[0].len();
        let num_instruments = signals.len() as f64;

        let mut result = vec![0.0; n];

        for i in 0..n {
            let count = signals.iter().filter(|s| s[i]).count() as f64;
            result[i] = count / num_instruments;
        }

        result
    }
}

/// Align signals to a common time axis.
///
/// Useful when instruments have different trading hours or missing data.
///
/// # Arguments
/// * `signals` - Signal array to align
/// * `source_timestamps` - Timestamps of the signal array
/// * `target_timestamps` - Target timestamp grid
/// * `fill_value` - Value to use for missing timestamps
///
/// # Returns
/// Aligned signal array
pub fn align_signals(
    signals: &[bool],
    source_timestamps: &[i64],
    target_timestamps: &[i64],
    fill_value: bool,
) -> Vec<bool> {
    let n = target_timestamps.len();
    let mut result = vec![fill_value; n];

    // Create a map of source timestamps to indices
    let mut source_map = std::collections::HashMap::new();
    for (i, &ts) in source_timestamps.iter().enumerate() {
        source_map.insert(ts, i);
    }

    // Fill in values where timestamps match
    for (i, &ts) in target_timestamps.iter().enumerate() {
        if let Some(&source_idx) = source_map.get(&ts) {
            result[i] = signals[source_idx];
        }
    }

    result
}

/// Forward-fill signals (carry forward last signal).
pub fn forward_fill_signals(signals: &[bool]) -> Vec<bool> {
    let mut result = signals.to_vec();
    let mut last_value = false;

    for i in 0..result.len() {
        if result[i] {
            last_value = true;
        }
        result[i] = last_value;
    }

    result
}

/// Create synchronized position signals.
///
/// Returns a position signal where:
/// - 1 = in position
/// - 0 = out of position
///
/// # Arguments
/// * `entries` - Entry signals (cleaned)
/// * `exits` - Exit signals (cleaned)
///
/// # Returns
/// Position state array
pub fn position_signals(entries: &[bool], exits: &[bool]) -> Vec<i8> {
    let n = entries.len();
    assert_eq!(n, exits.len());

    let mut result = vec![0i8; n];
    let mut in_position = false;

    for i in 0..n {
        if entries[i] {
            in_position = true;
        }
        if exits[i] {
            in_position = false;
        }
        result[i] = if in_position { 1 } else { 0 };
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sync_all() {
        let sync = SignalSynchronizer::new(SyncMode::All);

        let sig1 = vec![true, true, false, true];
        let sig2 = vec![true, false, false, true];
        let sig3 = vec![true, true, false, true];

        let result = sync.sync_entries(&[&sig1, &sig2, &sig3]);

        assert!(result[0]); // All true
        assert!(!result[1]); // Not all true
        assert!(!result[2]); // All false
        assert!(result[3]); // All true
    }

    #[test]
    fn test_sync_any() {
        let sync = SignalSynchronizer::new(SyncMode::Any);

        let sig1 = vec![true, false, false, false];
        let sig2 = vec![false, true, false, false];
        let sig3 = vec![false, false, false, false];

        let result = sync.sync_entries(&[&sig1, &sig2, &sig3]);

        assert!(result[0]); // At least one true
        assert!(result[1]); // At least one true
        assert!(!result[2]); // All false
        assert!(!result[3]); // All false
    }

    #[test]
    fn test_sync_majority() {
        let sync = SignalSynchronizer::new(SyncMode::Majority);

        let sig1 = vec![true, true, false, true];
        let sig2 = vec![true, false, false, true];
        let sig3 = vec![false, true, false, false];

        let result = sync.sync_entries(&[&sig1, &sig2, &sig3]);

        assert!(result[0]); // 2 out of 3
        assert!(result[1]); // 2 out of 3
        assert!(!result[2]); // 0 out of 3
        assert!(result[3]); // 2 out of 3
    }

    #[test]
    fn test_sync_master() {
        let sync = SignalSynchronizer::new(SyncMode::Master);

        let sig1 = vec![true, false, true, false]; // Master
        let sig2 = vec![false, true, false, true];
        let sig3 = vec![true, true, true, true];

        let result = sync.sync_entries(&[&sig1, &sig2, &sig3]);

        // Should follow master (sig1)
        assert!(result[0]);
        assert!(!result[1]);
        assert!(result[2]);
        assert!(!result[3]);
    }

    #[test]
    fn test_exit_inverse_logic() {
        // For All entry mode, exit should be Any
        let sync = SignalSynchronizer::new(SyncMode::All);

        let exit1 = vec![true, false, false];
        let exit2 = vec![false, false, false];
        let exit3 = vec![false, false, false];

        let result = sync.sync_exits(&[&exit1, &exit2, &exit3]);

        assert!(result[0]); // Any true -> exit
        assert!(!result[1]);
        assert!(!result[2]);
    }

    #[test]
    fn test_signal_agreement() {
        let sync = SignalSynchronizer::new(SyncMode::All);

        let sig1 = vec![true, true, false, true];
        let sig2 = vec![true, false, false, true];
        let sig3 = vec![false, true, false, true];

        let result = sync.signal_agreement(&[&sig1, &sig2, &sig3]);

        assert!((result[0] - 2.0 / 3.0).abs() < 1e-10);
        assert!((result[1] - 2.0 / 3.0).abs() < 1e-10);
        assert!((result[2] - 0.0).abs() < 1e-10);
        assert!((result[3] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_position_signals() {
        let entries = vec![false, true, false, false, true, false];
        let exits = vec![false, false, false, true, false, true];

        let result = position_signals(&entries, &exits);

        assert_eq!(result[0], 0);
        assert_eq!(result[1], 1);
        assert_eq!(result[2], 1);
        assert_eq!(result[3], 0);
        assert_eq!(result[4], 1);
        assert_eq!(result[5], 0);
    }
}
