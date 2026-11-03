//! Pre-trade risk gating.
//!
//! Constraints are checked *before* an entry opens, so a rejected entry never
//! reaches the equity curve and the reported metrics describe the constrained
//! run. Filtering trades after the fact leaves `summary_metrics` describing an unconstrained run that was never
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
    /// Capital-fraction sizing produced zero units because the instrument's
    /// margin requirement (a short option's SPAN-style deposit, a future's
    /// initial margin) exceeds the available capital — the lot itself was
    /// affordable on notional, the margin was not.
    InsufficientMargin,
    /// The margin model chosen for the account has no rule for this
    /// instrument (a class it does not list, or an instrument without a
    /// market definition). Never read as a requirement of zero.
    MarginUnpriced,
    /// The margin-call kill-switch has tripped; no further entries.
    MarginCall,
    /// The instrument's trading session is closed (outside a window, or on a
    /// holiday).
    MarketClosed,
    /// An entry arrived on the event that squares the session off.
    Squareoff,
    /// The order's resolved quantity or notional is above `max_order_qty` /
    /// `max_order_notional`.
    OrderSize,
    /// The fill would take the instrument's position above `max_position_qty`.
    MaxPosition,
    /// The entry would take gross notional exposure above `max_gross_exposure`.
    MaxExposure,
    /// The day's loss limit has been reached; no new entries until the next day.
    DailyLoss,
    /// The strategy was halted by hand (`ctx.halt` / `LiveNode.halt`).
    Killed,
    /// Netting holds a position and the order would open a second entry on
    /// its side.
    PositionOpen,
    /// A reduce-only order that cannot reduce the position (none held, or
    /// held on the order's own side).
    ReduceOnly,
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
            RejectReason::InsufficientMargin => "insufficient_margin",
            RejectReason::MarginUnpriced => "margin_unpriced",
            RejectReason::MarginCall => "margin_call",
            RejectReason::MarketClosed => "market_closed",
            RejectReason::Squareoff => "squareoff",
            RejectReason::OrderSize => "order_size",
            RejectReason::MaxPosition => "max_position",
            RejectReason::PositionOpen => "position_open",
            RejectReason::ReduceOnly => "reduce_only",
            RejectReason::MaxExposure => "max_exposure",
            RejectReason::DailyLoss => "daily_loss",
            RejectReason::Killed => "killed",
        }
    }

    /// Whether this refusal comes from one of the optional size, exposure,
    /// daily-loss or kill-switch limits (and so also raises a `risk_rejected`
    /// event).
    pub fn is_risk_limit(&self) -> bool {
        matches!(
            self,
            RejectReason::OrderSize
                | RejectReason::MaxPosition
                | RejectReason::MaxExposure
                | RejectReason::DailyLoss
                | RejectReason::Killed
        )
    }

    /// [`RejectReason::is_risk_limit`] for a stable identifier string.
    pub fn is_risk_limit_str(reason: &str) -> bool {
        matches!(reason, "order_size" | "max_position" | "max_exposure" | "daily_loss" | "killed")
    }
}

const NS_PER_DAY: i64 = 86_400_000_000_000;

/// The trading day a timestamp belongs to, as a day number.
///
/// This is the fallback for an instrument with no session schedule: the local
/// calendar date, the UTC timestamp shifted by the configured
/// `session_tz_offset_ns`. An instrument with a schedule uses the schedule's
/// session day instead (see `EngineKernel::risk_day`).
#[inline]
pub fn day_of(timestamp_ns: i64, tz_offset_ns: i64) -> i64 {
    timestamp_ns.saturating_add(tz_offset_ns).div_euclid(NS_PER_DAY)
}

/// Optional order-size, position-size and exposure limits, checked when a
/// fill is about to open or add to a position. `None` disables each.
///
/// Non-positive and NaN limits are treated as absent, matching
/// [`RiskGate::new`].
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SizeLimits {
    max_order_qty: Option<f64>,
    max_order_notional: Option<f64>,
    max_position_qty: Option<f64>,
    max_gross_exposure: Option<f64>,
}

impl SizeLimits {
    pub fn new(
        max_order_qty: Option<f64>,
        max_order_notional: Option<f64>,
        max_position_qty: Option<f64>,
        max_gross_exposure: Option<f64>,
    ) -> Self {
        let positive = |v: Option<f64>| v.filter(|&x| x > 0.0);
        Self {
            max_order_qty: positive(max_order_qty),
            max_order_notional: positive(max_order_notional),
            max_position_qty: positive(max_position_qty),
            max_gross_exposure: positive(max_gross_exposure),
        }
    }

    /// Whether any limit is set.
    #[inline]
    pub fn is_active(&self) -> bool {
        self.max_order_qty.is_some()
            || self.max_order_notional.is_some()
            || self.max_position_qty.is_some()
            || self.max_gross_exposure.is_some()
    }

    /// Whether `max_gross_exposure` is set.
    #[inline]
    pub fn tracks_exposure(&self) -> bool {
        self.max_gross_exposure.is_some()
    }

    /// An order of `qty` units worth `notional` is refused only when it is
    /// strictly above a limit; equal passes.
    pub fn check_order(&self, qty: f64, notional: f64) -> Result<(), RejectReason> {
        if self.max_order_qty.is_some_and(|max| qty > max)
            || self.max_order_notional.is_some_and(|max| notional > max)
        {
            return Err(RejectReason::OrderSize);
        }
        Ok(())
    }

    /// A fill taking the signed net position from `net_before` to
    /// `net_after` is refused when it ends strictly above the limit *and*
    /// larger than it was, so a fill that shrinks an oversized (adopted)
    /// position is never blocked.
    pub fn check_position(&self, net_before: f64, net_after: f64) -> Result<(), RejectReason> {
        match self.max_position_qty {
            Some(max) if net_after.abs() > max && net_after.abs() > net_before.abs() => {
                Err(RejectReason::MaxPosition)
            }
            _ => Ok(()),
        }
    }

    /// Gross exposure after the entry (`existing + added`, notional) must not
    /// be strictly above the limit.
    pub fn check_exposure(&self, existing: f64, added: f64) -> Result<(), RejectReason> {
        match self.max_gross_exposure {
            Some(max) if existing + added > max => Err(RejectReason::MaxExposure),
            _ => Ok(()),
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
    /// Daily-loss limits and what to do on a breach.
    daily: DailyLoss,
    /// Trading day currently being tracked.
    day: Option<i64>,
    /// Equity at the start of that day (the previous day's last observation).
    day_start_equity: f64,
    /// Most recent observed equity; seeded with the starting capital.
    last_equity: f64,
    /// Set when the day's loss limit is reached; cleared when the day rolls.
    day_blocked: bool,
    /// Set by an explicit halt; never cleared.
    killed: bool,
    /// Count of entries refused, for reporting.
    rejected_entries: usize,
}

/// Daily-loss limit: an absolute amount, a percent of the day's starting
/// equity, or both (the tighter wins).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct DailyLoss {
    amount: Option<f64>,
    pct: Option<f64>,
    flatten: bool,
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
            daily: DailyLoss::default(),
            day: None,
            day_start_equity: 0.0,
            last_equity: 0.0,
            day_blocked: false,
            killed: false,
            rejected_entries: 0,
        }
    }

    /// Add a daily-loss limit. `initial_equity` is the day-one starting
    /// equity. Non-positive limits are treated as absent.
    pub fn with_daily_loss(
        mut self,
        amount: Option<f64>,
        pct: Option<f64>,
        flatten: bool,
        initial_equity: f64,
    ) -> Self {
        self.daily = DailyLoss {
            amount: amount.filter(|&a| a > 0.0),
            pct: pct.filter(|&p| p > 0.0),
            flatten,
        };
        self.last_equity = initial_equity;
        self.day_start_equity = initial_equity;
        self
    }

    /// Whether any constraint is active.
    #[inline]
    pub fn is_active(&self) -> bool {
        self.max_positions.is_some() || self.max_drawdown_pct.is_some() || self.daily_active()
    }

    /// Whether a daily-loss limit is set.
    #[inline]
    pub fn daily_active(&self) -> bool {
        self.daily.amount.is_some() || self.daily.pct.is_some()
    }

    /// Whether a daily-loss breach also flattens open positions.
    #[inline]
    pub fn flatten_on_daily_loss(&self) -> bool {
        self.daily.flatten
    }

    /// Whether new entries are currently refused for the day's loss.
    #[inline]
    pub fn is_day_blocked(&self) -> bool {
        self.day_blocked
    }

    /// Whether a latching halt has tripped: the drawdown limit or an
    /// explicit kill. A daily-loss block clears at the next day, so it is
    /// not a halt.
    #[inline]
    pub fn is_halted(&self) -> bool {
        self.halted || self.killed
    }

    /// Whether the drawdown limit, as distinct from an explicit kill, has
    /// tripped.
    #[inline]
    pub fn is_drawdown_halted(&self) -> bool {
        self.halted
    }

    /// Latch an explicit halt; entries are refused with `Killed`.
    #[inline]
    pub fn kill(&mut self) {
        self.killed = true;
    }

    /// Whether an explicit halt has been requested.
    #[inline]
    pub fn is_killed(&self) -> bool {
        self.killed
    }

    /// Move to `day`, if it is a new one: the day's starting equity is the
    /// last equity seen, and the loss block lifts. Call before an entry check
    /// so the first bar of a day is judged against that day.
    pub fn roll_day(&mut self, day: i64) {
        if self.day != Some(day) {
            self.day = Some(day);
            self.day_start_equity = self.last_equity;
            self.day_blocked = false;
        }
    }

    /// Observe equity on `day`. Returns true on the edge where the day's loss
    /// limit is first reached (P&L at or below minus the limit).
    ///
    /// `day` is `None` when the event has no trading day (a weekend or
    /// holiday under a session schedule): the day does not roll, and the
    /// equity is judged against the day already running.
    pub fn on_day_equity(&mut self, day: Option<i64>, equity: f64) -> bool {
        if !self.daily_active() {
            return false;
        }
        if let Some(day) = day {
            self.roll_day(day);
        }
        let mut tripped = false;
        if !self.day_blocked {
            let loss = self.day_start_equity - equity;
            let by_amount = self.daily.amount.is_some_and(|a| loss >= a);
            let by_pct = self.daily.pct.is_some_and(|p| {
                self.day_start_equity > 0.0 && loss >= p / 100.0 * self.day_start_equity
            });
            if by_amount || by_pct {
                self.day_blocked = true;
                tripped = true;
            }
        }
        self.last_equity = equity;
        tripped
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
        if self.killed {
            return Err(RejectReason::Killed);
        }
        if self.halted {
            return Err(RejectReason::DrawdownHalt);
        }
        if self.day_blocked {
            return Err(RejectReason::DailyLoss);
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
    fn size_limits_refuse_only_strictly_above() {
        let l = SizeLimits::new(Some(10.0), Some(1_000.0), Some(10.0), Some(5_000.0));
        assert_eq!(l.check_order(10.0, 1_000.0), Ok(()));
        assert_eq!(l.check_order(10.01, 1_000.0), Err(RejectReason::OrderSize));
        assert_eq!(l.check_order(10.0, 1_000.01), Err(RejectReason::OrderSize));
        assert_eq!(l.check_position(4.0, 10.0), Ok(()));
        assert_eq!(l.check_position(4.0, 10.01), Err(RejectReason::MaxPosition));
        assert_eq!(l.check_position(0.0, -10.01), Err(RejectReason::MaxPosition));
        // Shrinking an oversized position is never blocked.
        assert_eq!(l.check_position(12.0, 11.0), Ok(()));
        assert_eq!(l.check_exposure(4_000.0, 1_000.0), Ok(()));
        assert_eq!(l.check_exposure(4_000.0, 1_000.01), Err(RejectReason::MaxExposure));
    }

    #[test]
    fn nonpositive_size_limits_are_absent() {
        let l = SizeLimits::new(Some(0.0), Some(-1.0), Some(f64::NAN), Some(0.0));
        assert!(!l.is_active());
    }

    #[test]
    fn daily_loss_trips_at_the_limit_and_resets_next_day() {
        let mut g =
            RiskGate::new(None, None).with_daily_loss(Some(1_000.0), None, false, 100_000.0);
        assert!(!g.on_day_equity(Some(0), 99_000.01));
        assert!(g.check_entry(0).is_ok());
        assert!(g.on_day_equity(Some(0), 99_000.0)); // exactly at the limit
        assert_eq!(g.check_entry(0), Err(RejectReason::DailyLoss));
        assert!(!g.on_day_equity(Some(0), 98_000.0)); // already tripped: no second edge
        assert!(!g.is_halted(), "a daily block is not a latching halt");
        // Next day: measured from the previous close, not from day one.
        g.roll_day(1);
        assert!(g.check_entry(0).is_ok());
        assert!(!g.on_day_equity(Some(1), 97_500.0)); // -500 on the day
        assert!(g.on_day_equity(Some(1), 97_000.0)); // -1000 on the day
    }

    #[test]
    fn an_event_with_no_trading_day_does_not_roll_the_day() {
        let mut g =
            RiskGate::new(None, None).with_daily_loss(Some(1_000.0), None, false, 100_000.0);
        assert!(!g.on_day_equity(Some(0), 99_400.0));
        // A weekend event: still day 0, so the loss accumulates (-1200).
        assert!(g.on_day_equity(None, 98_800.0));
        assert!(g.is_day_blocked());
        // No day, no reset either.
        assert!(!g.on_day_equity(None, 98_800.0));
        assert!(g.is_day_blocked());
    }

    #[test]
    fn daily_loss_percent_uses_the_days_starting_equity() {
        let mut g = RiskGate::new(None, None).with_daily_loss(None, Some(2.0), false, 50_000.0);
        assert!(!g.on_day_equity(Some(0), 49_000.01));
        assert!(g.on_day_equity(Some(0), 49_000.0));
    }

    #[test]
    fn kill_latches_and_outranks_other_reasons() {
        let mut g = RiskGate::new(Some(1), None);
        g.kill();
        assert!(g.is_halted());
        assert_eq!(g.check_entry(5), Err(RejectReason::Killed));
    }

    #[test]
    fn day_of_follows_the_utc_offset() {
        let hour = 3_600_000_000_000;
        assert_eq!(day_of(0, 0), 0);
        assert_eq!(day_of(23 * hour, 0), 0);
        assert_eq!(day_of(23 * hour, 2 * hour), 1);
        assert_eq!(day_of(-1, 0), -1);
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
