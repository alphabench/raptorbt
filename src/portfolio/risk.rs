//! Pre-trade risk gating.
//!
//! Constraints are checked *before* an entry opens, so a rejected entry never
//! reaches the equity curve and the reported metrics describe the constrained
//! run. Filtering trades after the fact -- which is what the backend does today
//! -- leaves `summary_metrics` describing an unconstrained run that was never
//! actually tradeable.

/// Why an entry was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RejectReason {
    /// Already at the concurrent-position limit.
    MaxPositions,
    /// The drawdown kill-switch has tripped; no further entries this run.
    DrawdownHalt,
    /// Requested size rounded/computed to zero units (e.g. size fraction too
    /// small for the lot size, or insufficient capital at this price).
    ZeroSize,
    /// The instrument has expired; no further entries are possible.
    Expired,
    /// The instrument is not yet active at this bar's timestamp.
    Inactive,
    /// An explicit-size order costs more than the available capital.
    InsufficientCapital,
    /// The margin-call kill-switch has tripped; no further entries.
    MarginCall,
}

impl RejectReason {
    /// Stable identifier for reporting.
    pub fn as_str(&self) -> &'static str {
        match self {
            RejectReason::MaxPositions => "max_positions",
            RejectReason::DrawdownHalt => "drawdown_halt",
            RejectReason::ZeroSize => "zero_size",
            RejectReason::Expired => "expired",
            RejectReason::Inactive => "inactive",
            RejectReason::InsufficientCapital => "insufficient_capital",
            RejectReason::MarginCall => "margin_call",
        }
    }
}

/// Portfolio-level constraints applied between signal and execution.
#[derive(Debug, Clone, Copy, Default)]
pub struct RiskGate {
    /// Maximum concurrent open positions. `None` is unlimited.
    max_positions: Option<usize>,
    /// Peak-to-trough drawdown percent that halts new entries. `None` disables.
    max_drawdown_pct: Option<f64>,
    /// Set once the drawdown limit is breached; never cleared.
    halted: bool,
    /// Count of entries refused, for reporting.
    rejected_entries: usize,
}

impl RiskGate {
    /// A gate that permits everything.
    pub fn unconstrained() -> Self {
        Self::default()
    }

    /// Build a gate from optional limits.
    ///
    /// Non-positive limits are treated as absent rather than as "reject
    /// everything", so a stray `max_positions=0` cannot silently produce a
    /// zero-trade backtest.
    pub fn new(max_positions: Option<usize>, max_drawdown_pct: Option<f64>) -> Self {
        Self {
            max_positions: max_positions.filter(|&n| n > 0),
            max_drawdown_pct: max_drawdown_pct.filter(|&d| d > 0.0),
            halted: false,
            rejected_entries: 0,
        }
    }

    /// Whether any constraint is active.
    #[inline]
    pub fn is_active(&self) -> bool {
        self.max_positions.is_some() || self.max_drawdown_pct.is_some()
    }

    /// Whether the kill-switch has tripped.
    #[inline]
    pub fn is_halted(&self) -> bool {
        self.halted
    }

    /// How many entries have been refused.
    #[inline]
    pub fn rejected_entries(&self) -> usize {
        self.rejected_entries
    }

    /// Test whether a new entry may open, given the current open-position count.
    ///
    /// Pure: call [`RiskGate::record_rejection`] to count a refusal.
    pub fn check_entry(&self, open_positions: usize) -> Result<(), RejectReason> {
        if self.halted {
            return Err(RejectReason::DrawdownHalt);
        }
        if let Some(max) = self.max_positions {
            if open_positions >= max {
                return Err(RejectReason::MaxPositions);
            }
        }
        Ok(())
    }

    /// Note that an entry was refused.
    #[inline]
    pub fn record_rejection(&mut self) {
        self.rejected_entries += 1;
    }

    /// Update the kill-switch from the current equity and running peak.
    ///
    /// Once tripped it stays tripped: a kill-switch that re-arms when equity
    /// recovers is a different (and much less conservative) policy than the one
    /// a max-drawdown limit is usually meant to express.
    pub fn on_equity(&mut self, equity: f64, peak_equity: f64) {
        if self.halted {
            return;
        }
        let Some(limit) = self.max_drawdown_pct else {
            return;
        };
        if peak_equity <= 0.0 {
            return;
        }

        let drawdown_pct = (peak_equity - equity) / peak_equity * 100.0;
        if drawdown_pct >= limit {
            self.halted = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unconstrained_permits_everything() {
        let gate = RiskGate::unconstrained();
        assert!(!gate.is_active());
        assert!(gate.check_entry(0).is_ok());
        assert!(gate.check_entry(1_000).is_ok());
    }

    #[test]
    fn max_positions_blocks_at_the_limit() {
        let gate = RiskGate::new(Some(2), None);
        assert!(gate.check_entry(0).is_ok());
        assert!(gate.check_entry(1).is_ok());
        assert_eq!(gate.check_entry(2), Err(RejectReason::MaxPositions));
        assert_eq!(gate.check_entry(3), Err(RejectReason::MaxPositions));
    }

    #[test]
    fn drawdown_halt_trips_and_latches() {
        let mut gate = RiskGate::new(None, Some(20.0));
        assert!(gate.check_entry(0).is_ok());

        // 10% drawdown: under the limit.
        gate.on_equity(90.0, 100.0);
        assert!(!gate.is_halted());
        assert!(gate.check_entry(0).is_ok());

        // 25% drawdown: trips.
        gate.on_equity(75.0, 100.0);
        assert!(gate.is_halted());
        assert_eq!(gate.check_entry(0), Err(RejectReason::DrawdownHalt));

        // Full recovery does not re-arm.
        gate.on_equity(100.0, 100.0);
        assert!(gate.is_halted());
        assert_eq!(gate.check_entry(0), Err(RejectReason::DrawdownHalt));
    }

    #[test]
    fn drawdown_trips_exactly_at_the_limit() {
        let mut gate = RiskGate::new(None, Some(20.0));
        gate.on_equity(80.0, 100.0); // exactly 20%
        assert!(gate.is_halted());
    }

    #[test]
    fn nonpositive_limits_are_treated_as_absent() {
        let gate = RiskGate::new(Some(0), Some(0.0));
        assert!(!gate.is_active(), "0 limits must not mean 'reject everything'");
        assert!(gate.check_entry(50).is_ok());
    }

    #[test]
    fn zero_peak_equity_does_not_trip() {
        let mut gate = RiskGate::new(None, Some(10.0));
        gate.on_equity(0.0, 0.0);
        assert!(!gate.is_halted());
    }

    #[test]
    fn rejections_are_counted() {
        let mut gate = RiskGate::new(Some(1), None);
        assert_eq!(gate.rejected_entries(), 0);
        gate.record_rejection();
        gate.record_rejection();
        assert_eq!(gate.rejected_entries(), 2);
    }
}
