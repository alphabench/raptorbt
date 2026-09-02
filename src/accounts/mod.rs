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
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum AccountMode {
    /// Fully funded: notional debited on entry (historical behavior).
    #[default]
    Cash,
    /// Leveraged: initial margin locked per position.
    ///
    /// The per-position margin rate is the instrument's `margin_init` when
    /// set, else `1 / leverage`.
    Margin { leverage: f64 },
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

    /// Margin locked for one open position; 0.0 for an unknown id.
    pub fn locked_for(&self, position_id: u64) -> f64 {
        self.locked.get(&position_id).copied().unwrap_or(0.0)
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

/// One account funding several kernels that trade as a portfolio.
///
/// The multi-instrument session owns exactly one of these. Kernels keep
/// their own [`MarginBook`]s — nothing is shared by reference — so this
/// holds only the aggregate figures the session needs to re-point each
/// kernel at the portfolio's capital before stepping it:
///
/// - `balance` is cash. Locking margin does not debit it (see the kernel's
///   entry funding), so `balance + unrealized` is portfolio equity with no
///   double-count.
/// - `locked` mirrors the sum of every kernel's locked initial margin as a
///   scalar. Per-position maps stay in the kernels, which is why position
///   ids colliding across kernels is harmless.
///
/// In [`AccountMode::Cash`] `locked` is always 0.0 and every operation
/// reduces to the historical single-pool arithmetic exactly.
#[derive(Debug)]
pub struct SharedAccount {
    mode: AccountMode,
    balance: f64,
    locked: f64,
    halted: bool,
    halted_at: Option<usize>,
}

impl SharedAccount {
    pub fn new(mode: AccountMode, initial_capital: f64) -> Self {
        Self { mode, balance: initial_capital, locked: 0.0, halted: false, halted_at: None }
    }

    pub fn mode(&self) -> AccountMode {
        self.mode
    }

    /// Cash balance, including amounts notionally locked as margin.
    pub fn balance(&self) -> f64 {
        self.balance
    }

    /// Total initial margin locked across every kernel; 0.0 in cash mode.
    pub fn locked(&self) -> f64 {
        self.locked
    }

    /// Capital available to open new positions.
    pub fn free(&self) -> f64 {
        self.balance - self.locked
    }

    /// Whether the margin-call kill-switch has tripped.
    pub fn is_halted(&self) -> bool {
        self.halted
    }

    /// Where the halt first latched, in the caller's index space.
    pub fn halted_at(&self) -> Option<usize> {
        self.halted_at
    }

    /// Trip the kill-switch (latching); only the first `idx` is recorded.
    pub fn halt(&mut self, idx: usize) {
        if !self.halted {
            self.halted = true;
            self.halted_at = Some(idx);
        }
    }

    /// Apply a stepped kernel's cash and locked-margin movement.
    pub fn reconcile(&mut self, delta_cash: f64, delta_locked: f64) {
        self.balance += delta_cash;
        self.locked += delta_locked;
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

    #[test]
    fn cash_account_never_locks() {
        let mut account = SharedAccount::new(AccountMode::Cash, 100_000.0);
        assert_eq!(account.locked(), 0.0);
        assert_eq!(account.free(), 100_000.0);
        // A cash-mode step only ever moves cash.
        account.reconcile(-25_000.0, 0.0);
        assert_eq!(account.balance(), 75_000.0);
        assert_eq!(account.free(), 75_000.0);
    }

    #[test]
    fn margin_locks_do_not_debit_balance() {
        let mut account = SharedAccount::new(AccountMode::Margin { leverage: 5.0 }, 100_000.0);
        // Entry: locks margin, debits only fees.
        account.reconcile(-50.0, 20_000.0);
        assert_eq!(account.balance(), 99_950.0);
        assert_eq!(account.locked(), 20_000.0);
        assert_eq!(account.free(), 79_950.0);
        // Exit: releases the lock, books PnL.
        account.reconcile(1_000.0, -20_000.0);
        assert_eq!(account.locked(), 0.0);
        assert_eq!(account.free(), 100_950.0);
    }

    #[test]
    fn shared_halt_records_first_index_only() {
        let mut account = SharedAccount::new(AccountMode::Margin { leverage: 10.0 }, 10_000.0);
        assert!(!account.is_halted());
        assert_eq!(account.halted_at(), None);
        account.halt(7);
        account.halt(9);
        assert!(account.is_halted());
        assert_eq!(account.halted_at(), Some(7));
    }
}
