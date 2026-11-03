//! Trading-session enforcement for [`EngineKernel`].
//!
//! Active only for an instrument whose spec carries a
//! [`SessionSchedule`](crate::core::session_schedule::SessionSchedule); with
//! none, every method here is a no-op and the step is exactly the step it
//! always was. Living in the kernel means every runner gets the same rules.
//!
//! * **Entries outside the session** are refused with `market_closed`:
//!   typed orders at submission, signal and intent entries at the decision
//!   event (judged at the event's instant), and a deferred next-bar-open
//!   entry again at its fill bar (judged on the bar's data).
//! * **Resting orders** are not matched against bars or prints outside the
//!   session; the event is skipped for matching (see `OrderEngine`). Nothing
//!   is dropped: the order waits, and a DAY order still lapses at the close.
//! * **Square-off**: on the first event at or after the schedule's
//!   square-off minute on each trading day, every open position is closed
//!   with [`ExitReason::Squareoff`] at that event's close, and entries on
//!   squaring off also cancels the working entry orders (reduce-only orders
//!   are left alone). From then to the end of that trading day every entry is
//!   refused with `squareoff`, so a squared-off intraday position cannot
//!   re-open and carry overnight. The square-off fires once per session.
//!
//! A bar is stamped at its close, so the data it carries ends one nanosecond
//! earlier: the bar closing at 15:30 is inside a session that closes at
//! 15:30, while an *action* taken at 15:30 is not.

use crate::core::session_schedule::SessionSchedule;
use crate::core::types::ExitReason;
use crate::portfolio::kernel::{EngineEvent, EngineKernel, KernelBar, StepMode};
use crate::portfolio::risk::RejectReason;

/// Per-kernel square-off bookkeeping.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct SessionState {
    /// Local day number whose square-off has already fired.
    squareoff_day: Option<i64>,
    /// Whether the event being stepped is the one that squares off.
    pub(crate) squareoff_now: bool,
    /// Whether the event falls on a trading day already squared off, from the
    /// square-off event to the end of that day: no entry may open.
    pub(crate) entries_blocked: bool,
}

/// What the calendar says about one event.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SessionGate {
    /// Whether an action taken at the event's instant would be in session.
    pub(crate) decision_open: bool,
    /// Whether the data the event carries lies in the session.
    pub(crate) data_open: bool,
}

impl EngineKernel {
    pub(crate) fn schedule(&self) -> Option<&SessionSchedule> {
        self.spec.as_ref().and_then(|s| s.session.as_ref())
    }

    /// The time in force an order gets when it names none: `"day"` on an
    /// instrument with an exchange session (an order is cancelled at that
    /// session's close, as an exchange does), `"gtc"` on one without. Put
    /// `tif="gtc"` on the order to carry it across sessions where the venue
    /// allows that.
    pub fn default_tif(&self) -> &'static str {
        if self.schedule().is_some()
            || self.spec.as_ref().and_then(|s| s.native_gtc()) == Some(false)
        {
            "day"
        } else {
            "gtc"
        }
    }

    /// Refuse a validity the instrument's exchange segment does not hold: a
    /// `gtc` or `gtd` where only day orders are established, or a `gtd` dated
    /// past the contract's maturity where the segment holds them. The message
    /// says why; `None` without a venue allows everything.
    pub fn check_validity(&self, tif: &str, expire_ns: Option<i64>) -> Result<(), String> {
        let Some(spec) = &self.spec else { return Ok(()) };
        // A day order ends with the session; a venue no preset covers, given no session, has no
        // day to end, and trading it round the clock would be a silent guess.
        if tif == "day" && spec.session.is_none() {
            if let Some(venue) = spec.venue.as_deref() {
                return Err(format!(
                    "day is not a validity {:?} has: its venue {venue:?} implies no session (the \
                     presets cover NSE, BSE, NFO, BFO, MCX, CDS and BCD) and none was given, so \
                     there is no session end to cancel the order at; give the InstrumentSpec a \
                     session=",
                    spec.symbol
                ));
            }
        }
        if tif != "gtc" && tif != "gtd" {
            return Ok(());
        }
        match spec.native_gtc() {
            Some(false) => Err(format!(
                "{tif} is not a validity the {} segment of {:?} holds: its orders are day orders \
                 (cancelled at the session's end), or ioc/fok; a broker that keeps an order alive \
                 resubmits it each day, which is the host's business, not the exchange's",
                spec.venue.as_deref().unwrap_or_default(),
                spec.symbol
            )),
            Some(true) => match (tif, expire_ns, spec.expiration_ns) {
                ("gtd", Some(until), Some(maturity)) if until > maturity => Err(format!(
                    "gtd expiry {until} is past the maturity of {:?} ({maturity}): the exchange \
                     holds a good-till-date order no later than the contract's maturity",
                    spec.symbol
                )),
                _ => Ok(()),
            },
            None => Ok(()),
        }
    }

    /// Evaluate the calendar for the event about to be stepped, and arm the
    /// square-off if it is due. Called once at the top of every step.
    pub(crate) fn session_gate(&mut self, ts: i64, mode: StepMode) -> SessionGate {
        self.session_state.squareoff_now = false;
        self.session_state.entries_blocked = false;
        if self.schedule().is_none() {
            self.configured_squareoff(ts);
            return SessionGate { decision_open: true, data_open: true };
        }
        let Some(schedule) = self.schedule() else {
            return SessionGate { decision_open: true, data_open: true };
        };
        let data_ts = if mode == StepMode::Bar { ts.saturating_sub(1) } else { ts };
        let gate = SessionGate {
            decision_open: schedule.is_open(ts)
                || (mode == StepMode::Bar && self.session_bar_at(ts)),
            data_open: schedule.is_open(data_ts),
        };
        // The run's own `squareoff_time` is read in the schedule's zone and
        // beats the schedule's default minute.
        let last = self.session_state.squareoff_day;
        let due = match self.config.squareoff_time_minutes {
            Some(minute) => schedule.squareoff_due_at(minute, ts, last),
            None => schedule.squareoff_due(ts, last),
        };
        let today = schedule.session_day(ts);
        if let Some(day) = due {
            self.session_state.squareoff_day = Some(day);
            self.session_state.squareoff_now = true;
        }
        self.session_state.entries_blocked =
            today.is_some() && today == self.session_state.squareoff_day;
        gate
    }

    /// The square-off `BacktestConfig.squareoff_time` asks for, on an
    /// instrument with no calendar: the first event at or after that local
    /// time of day closes every position, and entries are refused with
    /// `squareoff` until the day ends. An instrument's own `SessionSchedule`
    /// reads the minute in its own zone instead (see `session_gate`).
    fn configured_squareoff(&mut self, ts: i64) {
        let Some(minutes) = self.config.squareoff_time_minutes else { return };
        let local_s = (ts + self.config.session_tz_offset_ns).div_euclid(1_000_000_000);
        let day = local_s.div_euclid(86_400);
        let minute_of_day = (local_s.rem_euclid(86_400) / 60) as u32;
        if minute_of_day >= minutes && self.session_state.squareoff_day != Some(day) {
            self.session_state.squareoff_day = Some(day);
            self.session_state.squareoff_now = true;
        }
        self.session_state.entries_blocked = self.session_state.squareoff_day == Some(day);
    }

    /// Why an entry is refused right now, given whether `open` holds for the
    /// instant it is judged at.
    pub(crate) fn session_refusal(&self, open: bool) -> Option<RejectReason> {
        if self.session_state.entries_blocked {
            Some(RejectReason::Squareoff)
        } else if !open {
            Some(RejectReason::MarketClosed)
        } else {
            None
        }
    }

    /// Whether a typed order submitted at `ts` falls outside the session. A bar stamped at the
    /// session's close that is the first bar of its session (a daily bar) is that session's bar:
    /// a decision on it is in session, and fills at that close (see [`Self::session_bar_at`]).
    pub(crate) fn closed_at(&self, ts: i64) -> bool {
        self.schedule().is_some_and(|s| !s.is_open(ts)) && !self.session_bar_at(ts)
    }

    /// Whether `ts` is the stamp of a bar that covers its whole session: it is stamped at the
    /// instant the session closes (the last instant of data inside it) and the bar stepped before
    /// it belongs to an earlier session, so nothing else of this session came first. A minute bar
    /// stamped at the close follows another bar of its session and is not one.
    pub(crate) fn session_bar_at(&self, ts: i64) -> bool {
        let Some(schedule) = self.schedule() else { return false };
        if schedule.is_open(ts) || !schedule.is_open(ts.saturating_sub(1)) {
            return false;
        }
        let before = match self.last_step {
            Some((_, b)) if b.timestamp == ts => self.prev_bar_ts,
            Some((_, b)) => Some(b.timestamp),
            None => None,
        };
        before.is_none_or(|p| {
            schedule.session_day(p.saturating_sub(1)) != schedule.session_day(ts.saturating_sub(1))
        })
    }

    /// Close every open position for the session square-off, at the event's
    /// close, the way a forced liquidation is marked, and cancel the working
    /// entry orders.
    pub(crate) fn squareoff_all(&mut self, idx: usize, bar: &KernelBar) -> Vec<EngineEvent> {
        if self.venue_managed {
            // A venue holds the positions: the runtime sends the closing orders.
            self.squareoff_outbox = true;
            return Vec::new();
        }
        let ids: Vec<u64> = self.ledger.positions().iter().map(|p| p.id).collect();
        let mut events = Vec::new();
        for position_id in ids {
            if let Some(event) =
                self.close_at(idx, bar, position_id, bar.close, ExitReason::Squareoff, false)
            {
                events.push(event);
            }
        }
        // What is still working can only open something now.
        let entries: Vec<u64> =
            self.orders.working().filter(|o| !o.reduce_only).map(|o| o.id).collect();
        for id in entries {
            self.cancel_order(idx, id);
        }
        events.append(&mut self.pending_events);
        events
    }
}

#[cfg(test)]
#[path = "kernel_session_tests.rs"]
mod tests;
