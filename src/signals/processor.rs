//! Signal processor for cleaning entry/exit signals.
//!
//! Ensures proper alternation between entries and exits to prevent
//! overlapping positions or orphaned signals.

use crate::core::types::Direction;

/// Signal processor for cleaning raw entry/exit signals.
#[derive(Debug, Clone)]
pub struct SignalProcessor {
    /// Whether to allow multiple entries before an exit (pyramiding).
    pub allow_pyramiding: bool,
    /// Maximum number of pyramid entries.
    pub max_pyramid_entries: usize,
}

impl Default for SignalProcessor {
    fn default() -> Self {
        Self { allow_pyramiding: false, max_pyramid_entries: 1 }
    }
}

impl SignalProcessor {
    /// Create a new signal processor.
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable pyramiding with a maximum number of entries.
    pub fn with_pyramiding(mut self, max_entries: usize) -> Self {
        self.allow_pyramiding = max_entries > 1;
        self.max_pyramid_entries = max_entries;
        self
    }

    /// Clean entry/exit signals to ensure proper alternation.
    ///
    /// Rules (matching VectorBT behavior):
    /// 1. First signal must be an entry
    /// 2. After an entry, ignore further entries (unless pyramiding)
    /// 3. After an exit, ignore further exits
    /// 4. Entries and exits must alternate properly
    /// 5. Same-bar conflict: If both entry AND exit signals are True on the same bar
    ///    when in position, VectorBT stays in position (ignores the exit).
    ///    This matches VectorBT's "entry takes priority" behavior.
    ///
    /// # Arguments
    /// * `entries` - Raw entry signals
    /// * `exits` - Raw exit signals
    ///
    /// # Returns
    /// Tuple of (cleaned_entries, cleaned_exits)
    pub fn clean_signals(&self, entries: &[bool], exits: &[bool]) -> (Vec<bool>, Vec<bool>) {
        let n = entries.len();
        assert_eq!(n, exits.len(), "Entry and exit arrays must have same length");

        let mut clean_entries = vec![false; n];
        let mut clean_exits = vec![false; n];

        if n == 0 {
            return (clean_entries, clean_exits);
        }

        let mut in_position = false;
        let mut position_count = 0;

        for i in 0..n {
            if !in_position {
                // Not in position - looking for entry
                if entries[i] {
                    clean_entries[i] = true;
                    in_position = true;
                    position_count = 1;
                }
                // Ignore exits when not in position
            } else {
                // In position - looking for exit (or pyramid entry)
                // VectorBT behavior: If both entry and exit are True, stay in position
                // (entry signal "cancels" the exit signal)
                if exits[i] && !entries[i] {
                    // Only exit if there's no conflicting entry signal
                    clean_exits[i] = true;
                    if self.allow_pyramiding {
                        position_count -= 1;
                        if position_count == 0 {
                            in_position = false;
                        }
                    } else {
                        in_position = false;
                        position_count = 0;
                    }
                } else if entries[i]
                    && self.allow_pyramiding
                    && position_count < self.max_pyramid_entries
                {
                    // Pyramid entry
                    clean_entries[i] = true;
                    position_count += 1;
                }
                // If both entry and exit are True, we stay in position (ignore both)
                // If only entry is True and not pyramiding, ignore entry (already in position)
            }
        }

        (clean_entries, clean_exits)
    }

    /// Clean signals with direction awareness (for strategies that can go long/short).
    ///
    /// # Arguments
    /// * `long_entries` - Long entry signals
    /// * `long_exits` - Long exit signals
    /// * `short_entries` - Short entry signals
    /// * `short_exits` - Short exit signals
    ///
    /// # Returns
    /// Tuple of (clean_long_entries, clean_long_exits, clean_short_entries, clean_short_exits)
    pub fn clean_signals_bidirectional(
        &self,
        long_entries: &[bool],
        long_exits: &[bool],
        short_entries: &[bool],
        short_exits: &[bool],
    ) -> (Vec<bool>, Vec<bool>, Vec<bool>, Vec<bool>) {
        let n = long_entries.len();
        assert_eq!(n, long_exits.len());
        assert_eq!(n, short_entries.len());
        assert_eq!(n, short_exits.len());

        let mut clean_long_entries = vec![false; n];
        let mut clean_long_exits = vec![false; n];
        let mut clean_short_entries = vec![false; n];
        let mut clean_short_exits = vec![false; n];

        if n == 0 {
            return (clean_long_entries, clean_long_exits, clean_short_entries, clean_short_exits);
        }

        let mut current_direction: Option<Direction> = None;

        for i in 0..n {
            match current_direction {
                None => {
                    // Not in any position - look for entry
                    if long_entries[i] {
                        clean_long_entries[i] = true;
                        current_direction = Some(Direction::Long);
                    } else if short_entries[i] {
                        clean_short_entries[i] = true;
                        current_direction = Some(Direction::Short);
                    }
                }
                Some(Direction::Long) => {
                    // In long position - look for exit or reversal
                    if long_exits[i] {
                        clean_long_exits[i] = true;
                        current_direction = None;
                    } else if short_entries[i] {
                        // Reversal: exit long and enter short
                        clean_long_exits[i] = true;
                        clean_short_entries[i] = true;
                        current_direction = Some(Direction::Short);
                    }
                }
                Some(Direction::Short) => {
                    // In short position - look for exit or reversal
                    if short_exits[i] {
                        clean_short_exits[i] = true;
                        current_direction = None;
                    } else if long_entries[i] {
                        // Reversal: exit short and enter long
                        clean_short_exits[i] = true;
                        clean_long_entries[i] = true;
                        current_direction = Some(Direction::Long);
                    }
                }
            }
        }

        (clean_long_entries, clean_long_exits, clean_short_entries, clean_short_exits)
    }

    /// Generate exit-on-opposite-entry signals.
    ///
    /// Useful for strategies where an entry in opposite direction
    /// should automatically close the current position.
    ///
    /// # Arguments
    /// * `entries` - Entry signals
    /// * `direction` - Current position direction
    ///
    /// # Returns
    /// Modified exit signals that include opposite-direction entries as exits
    pub fn exits_from_opposite_entries(
        &self,
        long_entries: &[bool],
        short_entries: &[bool],
    ) -> (Vec<bool>, Vec<bool>) {
        let n = long_entries.len();
        assert_eq!(n, short_entries.len());

        // Long exits when short entry
        // Short exits when long entry
        (short_entries.to_vec(), long_entries.to_vec())
    }

    /// Count the number of trades that would be generated from signals.
    ///
    /// # Arguments
    /// * `entries` - Entry signals (already cleaned)
    /// * `exits` - Exit signals (already cleaned)
    ///
    /// # Returns
    /// Number of complete trades (entry + exit pairs)
    pub fn count_trades(_entries: &[bool], exits: &[bool]) -> usize {
        exits.iter().filter(|&&e| e).count()
    }

    /// Get indices of entries and exits.
    ///
    /// # Arguments
    /// * `entries` - Entry signals
    /// * `exits` - Exit signals
    ///
    /// # Returns
    /// Tuple of (entry_indices, exit_indices)
    pub fn get_trade_indices(entries: &[bool], exits: &[bool]) -> (Vec<usize>, Vec<usize>) {
        let entry_indices: Vec<usize> = entries
            .iter()
            .enumerate()
            .filter_map(|(i, &e)| if e { Some(i) } else { None })
            .collect();

        let exit_indices: Vec<usize> =
            exits.iter().enumerate().filter_map(|(i, &e)| if e { Some(i) } else { None }).collect();

        (entry_indices, exit_indices)
    }
}

/// Shift signals forward by n bars (delays execution).
pub fn shift_signals(signals: &[bool], n: usize) -> Vec<bool> {
    let len = signals.len();
    let mut result = vec![false; len];

    if n >= len {
        return result;
    }

    for i in n..len {
        result[i] = signals[i - n];
    }

    result
}

/// Combine multiple signal arrays with AND logic.
pub fn combine_signals_and(signals: &[&[bool]]) -> Vec<bool> {
    if signals.is_empty() {
        return vec![];
    }

    let n = signals[0].len();
    for sig in signals.iter() {
        assert_eq!(sig.len(), n, "All signal arrays must have same length");
    }

    let mut result = vec![true; n];
    for sig in signals.iter() {
        for i in 0..n {
            result[i] = result[i] && sig[i];
        }
    }

    result
}

/// Combine multiple signal arrays with OR logic.
pub fn combine_signals_or(signals: &[&[bool]]) -> Vec<bool> {
    if signals.is_empty() {
        return vec![];
    }

    let n = signals[0].len();
    for sig in signals.iter() {
        assert_eq!(sig.len(), n, "All signal arrays must have same length");
    }

    let mut result = vec![false; n];
    for sig in signals.iter() {
        for i in 0..n {
            result[i] = result[i] || sig[i];
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clean_signals_basic() {
        let processor = SignalProcessor::new();

        let entries = vec![true, false, true, false, true, false];
        let exits = vec![false, true, false, true, false, true];

        let (clean_e, clean_x) = processor.clean_signals(&entries, &exits);

        // First entry should be kept
        assert!(clean_e[0]);
        // First exit should be kept
        assert!(clean_x[1]);
        // Second entry should be kept
        assert!(clean_e[2]);
        // Second exit should be kept
        assert!(clean_x[3]);
    }

    #[test]
    fn test_clean_signals_consecutive_entries() {
        let processor = SignalProcessor::new();

        let entries = vec![true, true, true, false, false];
        let exits = vec![false, false, false, true, false];

        let (clean_e, clean_x) = processor.clean_signals(&entries, &exits);

        // Only first entry should be kept
        assert!(clean_e[0]);
        assert!(!clean_e[1]);
        assert!(!clean_e[2]);
        // Exit should be kept
        assert!(clean_x[3]);
    }

    #[test]
    fn test_clean_signals_consecutive_exits() {
        let processor = SignalProcessor::new();

        let entries = vec![true, false, false, false, false];
        let exits = vec![false, true, true, true, false];

        let (clean_e, clean_x) = processor.clean_signals(&entries, &exits);

        // Entry should be kept
        assert!(clean_e[0]);
        // Only first exit should be kept
        assert!(clean_x[1]);
        assert!(!clean_x[2]);
        assert!(!clean_x[3]);
    }

    #[test]
    fn test_clean_signals_exit_before_entry() {
        let processor = SignalProcessor::new();

        let entries = vec![false, false, true, false, false];
        let exits = vec![true, true, false, true, false];

        let (clean_e, clean_x) = processor.clean_signals(&entries, &exits);

        // Exits before first entry should be ignored
        assert!(!clean_x[0]);
        assert!(!clean_x[1]);
        // Entry should be kept
        assert!(clean_e[2]);
        // Exit after entry should be kept
        assert!(clean_x[3]);
    }

    #[test]
    fn test_pyramiding() {
        let processor = SignalProcessor::new().with_pyramiding(3);

        let entries = vec![true, true, true, false, false];
        let exits = vec![false, false, false, true, true];

        let (clean_e, clean_x) = processor.clean_signals(&entries, &exits);

        // All three entries should be kept (pyramiding)
        assert!(clean_e[0]);
        assert!(clean_e[1]);
        assert!(clean_e[2]);
        // Both exits should be kept
        assert!(clean_x[3]);
        assert!(clean_x[4]);
    }

    #[test]
    fn test_shift_signals() {
        let signals = vec![true, false, true, false, true];
        let shifted = shift_signals(&signals, 2);

        assert!(!shifted[0]);
        assert!(!shifted[1]);
        assert!(shifted[2]); // Original [0]
        assert!(!shifted[3]); // Original [1]
        assert!(shifted[4]); // Original [2]
    }

    #[test]
    fn test_combine_signals_and() {
        let sig1 = vec![true, true, false, false];
        let sig2 = vec![true, false, true, false];

        let combined = combine_signals_and(&[&sig1, &sig2]);

        assert!(combined[0]); // true && true
        assert!(!combined[1]); // true && false
        assert!(!combined[2]); // false && true
        assert!(!combined[3]); // false && false
    }

    #[test]
    fn test_combine_signals_or() {
        let sig1 = vec![true, true, false, false];
        let sig2 = vec![true, false, true, false];

        let combined = combine_signals_or(&[&sig1, &sig2]);

        assert!(combined[0]); // true || true
        assert!(combined[1]); // true || false
        assert!(combined[2]); // false || true
        assert!(!combined[3]); // false || false
    }
}
