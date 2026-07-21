//! Account modes: fully-funded cash vs leveraged margin.
//!
//! [`AccountMode::Cash`] is the historical model — entries debit full
//! notional, exits credit it back, equity marks positions at `price * size`.
//! Its arithmetic lives in the kernel unchanged and is pinned by the golden
//! fixture suite.
//!
//! [`AccountMode::Margin`] locks initial margin instead of full notional,
//! marks equity as balance plus direction-aware unrealized PnL (which fixes
//! short cash-flow), and emits a margin call when equity falls below the
//! maintenance requirement.

use std::collections::HashMap;

/// How the account funds positions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AccountMode {
    /// Fully funded: notional debited on entry (historical behavior).
    Cash,
    /// Leveraged: initial margin locked per position.
    ///
    /// The per-position margin rate is the instrument's `margin_init` when
    /// set, else `1 / leverage`.
    Margin { leverage: f64 },
}

impl Default for AccountMode {
    fn default() -> Self {
        AccountMode::Cash
    }
}

/// Per-position margin bookkeeping for [`AccountMode::Margin`].
#[derive(Debug, Default)]
pub struct MarginBook {
    /// Initial margin locked per open position id.
    locked: HashMap<u64, f64>,
    /// Latched once a margin call fires; blocks further entries.
    halted: bool,
}

impl MarginBook {
    /// Total locked initial margin.
    pub fn total_locked(&self) -> f64 {
        self.locked.values().sum()
    }

    /// Lock margin for a newly opened position.
    pub fn lock(&mut self, position_id: u64, amount: f64) {
        self.locked.insert(position_id, amount);
    }

    /// Release a closed position's margin, returning the amount.
    pub fn release(&mut self, position_id: u64) -> f64 {
        self.locked.remove(&position_id).unwrap_or(0.0)
    }

    /// Whether the margin-call kill-switch has tripped.
    pub fn is_halted(&self) -> bool {
        self.halted
    }

    /// Trip the margin-call kill-switch (latching).
    pub fn halt(&mut self) {
        self.halted = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lock_release_roundtrip() {
        let mut book = MarginBook::default();
        book.lock(0, 1_000.0);
        book.lock(1, 500.0);
        assert_eq!(book.total_locked(), 1_500.0);
        assert_eq!(book.release(0), 1_000.0);
        assert_eq!(book.release(0), 0.0);
        assert_eq!(book.total_locked(), 500.0);
    }

    #[test]
    fn halt_latches() {
        let mut book = MarginBook::default();
        assert!(!book.is_halted());
        book.halt();
        assert!(book.is_halted());
    }
}
