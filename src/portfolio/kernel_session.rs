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
    fn schedule(&self) -> Option<&SessionSchedule> {
        self.spec.as_ref().and_then(|s| s.session.as_ref())
    }

    /// Evaluate the calendar for the event about to be stepped, and arm the
    /// square-off if it is due. Called once at the top of every step.
    pub(crate) fn session_gate(&mut self, ts: i64, mode: StepMode) -> SessionGate {
        self.session_state.squareoff_now = false;
        self.session_state.entries_blocked = false;
        let Some(schedule) = self.schedule() else {
            return SessionGate { decision_open: true, data_open: true };
        };
        let data_ts = if mode == StepMode::Bar { ts.saturating_sub(1) } else { ts };
        let gate = SessionGate {
            decision_open: schedule.is_open(ts),
            data_open: schedule.is_open(data_ts),
        };
        let due = schedule.squareoff_due(ts, self.session_state.squareoff_day);
        let today = schedule.session_day(ts);
        if let Some(day) = due {
            self.session_state.squareoff_day = Some(day);
            self.session_state.squareoff_now = true;
        }
        self.session_state.entries_blocked =
            today.is_some() && today == self.session_state.squareoff_day;
        gate
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

    /// Whether a typed order submitted at `ts` falls outside the session.
    pub(crate) fn closed_at(&self, ts: i64) -> bool {
        self.schedule().is_some_and(|s| !s.is_open(ts))
    }

    /// Close every open position for the session square-off, at the event's
    /// close, the way a forced liquidation is marked, and cancel the working
    /// entry orders.
    pub(crate) fn squareoff_all(&mut self, idx: usize, bar: &KernelBar) -> Vec<EngineEvent> {
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
