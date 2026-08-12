//! Execution algorithms: schedules that release child orders over time.
//!
//! A TWAP is not an order — it is a plan to submit several. Modelling it as
//! a parent order would deadlock its children: the one-triggers-other gate
//! holds a child until its parent *fills*, and a schedule never fills. So a
//! schedule is its own entity, and the orders it releases are completely
//! ordinary ones that the matcher needs to know nothing about.
//!
//! Slicing is timed, never counted in bars. `idx` is a bar ordinal in a bar
//! session and an event ordinal in a tick session, so "every 1 bar" would
//! silently mean "every 1 print" on a tick feed — turning a five-slice TWAP
//! into a single burst. An interval in nanoseconds means the same thing on
//! both.

use crate::core::types::Timestamp;
use crate::execution::orders::{OrderKind, OrderSide, TimeInForce};

/// How a schedule releases its slices.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExecAlgorithm {
    /// Equal slices at a fixed interval.
    Twap { slices: u32, interval_ns: i64 },
}

/// One slice a schedule wants submitted.
#[derive(Debug, Clone, PartialEq)]
pub struct PendingSlice {
    pub algo_id: u64,
    pub client_id: String,
    pub side: OrderSide,
    pub kind: OrderKind,
    pub tif: TimeInForce,
    pub units: f64,
    pub reduce_only: bool,
}

/// A working execution schedule.
#[derive(Debug, Clone)]
pub struct AlgoSchedule {
    pub id: u64,
    pub client_id: String,
    pub side: OrderSide,
    pub kind: OrderKind,
    pub tif: TimeInForce,
    pub total_units: f64,
    pub algo: ExecAlgorithm,
    pub reduce_only: bool,
    /// Earliest timestamp the next slice may be released at.
    next_release_ns: Timestamp,
    released: u32,
    active: bool,
}

impl AlgoSchedule {
    /// Units in slice `n`, allocated so the parts sum to the total exactly.
    ///
    /// Splitting by repeated division leaves a residue that either loses or
    /// invents size; carving cumulative boundaries cannot.
    fn slice_units(&self, n: u32) -> f64 {
        let ExecAlgorithm::Twap { slices, .. } = self.algo;
        let cut = |k: u32| self.total_units * k as f64 / slices as f64;
        if n + 1 == slices {
            // The last slice takes whatever remains, so rounding never
            // strands a fraction.
            self.total_units - cut(n)
        } else {
            cut(n + 1) - cut(n)
        }
    }

    pub fn is_active(&self) -> bool {
        self.active
    }

    pub fn released(&self) -> u32 {
        self.released
    }

    pub fn is_complete(&self) -> bool {
        let ExecAlgorithm::Twap { slices, .. } = self.algo;
        self.released >= slices
    }
}

/// Errors registering a schedule.
#[derive(Debug, Clone, PartialEq)]
pub enum AlgoError {
    /// `slices` must be at least one.
    ZeroSlices,
    /// The interval between slices must be positive.
    NonPositiveInterval,
    /// Total size must be a positive, finite unit count.
    InvalidUnits,
}

impl std::fmt::Display for AlgoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AlgoError::ZeroSlices => write!(f, "slices must be >= 1"),
            AlgoError::NonPositiveInterval => write!(f, "interval must be > 0"),
            AlgoError::InvalidUnits => write!(f, "total units must be finite and > 0"),
        }
    }
}

/// Registry of working schedules.
#[derive(Debug, Default)]
pub struct AlgoEngine {
    schedules: Vec<AlgoSchedule>,
    next_id: u64,
}

impl AlgoEngine {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.schedules.is_empty()
    }

    /// Register a schedule; the first slice is due immediately.
    #[allow(clippy::too_many_arguments)]
    pub fn submit(
        &mut self,
        client_id: String,
        side: OrderSide,
        kind: OrderKind,
        tif: TimeInForce,
        total_units: f64,
        algo: ExecAlgorithm,
        reduce_only: bool,
        now_ns: Timestamp,
    ) -> Result<u64, AlgoError> {
        let ExecAlgorithm::Twap { slices, interval_ns } = algo;
        if slices == 0 {
            return Err(AlgoError::ZeroSlices);
        }
        if interval_ns <= 0 {
            return Err(AlgoError::NonPositiveInterval);
        }
        if !total_units.is_finite() || total_units <= 0.0 {
            return Err(AlgoError::InvalidUnits);
        }
        let id = self.next_id;
        self.next_id += 1;
        self.schedules.push(AlgoSchedule {
            id,
            client_id,
            side,
            kind,
            tif,
            total_units,
            algo,
            reduce_only,
            next_release_ns: now_ns,
            released: 0,
            active: true,
        });
        Ok(id)
    }

    /// Stop a schedule from releasing further slices.
    ///
    /// Slices already released keep their own lifecycle: cancelling a TWAP
    /// halts the remainder, it does not unwind what already traded.
    pub fn cancel(&mut self, algo_id: u64) -> bool {
        match self.schedules.iter_mut().find(|s| s.id == algo_id) {
            Some(schedule) if schedule.active => {
                schedule.active = false;
                true
            }
            _ => false,
        }
    }

    pub fn get(&self, algo_id: u64) -> Option<&AlgoSchedule> {
        self.schedules.iter().find(|s| s.id == algo_id)
    }

    /// Slices due at this timestamp, at most one per schedule.
    ///
    /// A gap in the data — a weekend, a halt — can leave several intervals
    /// due at once. Releasing them all would dump the size the schedule
    /// exists to spread, so the schedule stretches instead.
    pub fn release_due(&mut self, now_ns: Timestamp) -> Vec<PendingSlice> {
        let mut due = Vec::new();
        for schedule in &mut self.schedules {
            if !schedule.active || schedule.is_complete() || now_ns < schedule.next_release_ns {
                continue;
            }
            let ExecAlgorithm::Twap { interval_ns, .. } = schedule.algo;
            let n = schedule.released;
            let units = schedule.slice_units(n);
            schedule.released += 1;
            schedule.next_release_ns = now_ns + interval_ns;
            due.push(PendingSlice {
                algo_id: schedule.id,
                client_id: format!("{}#{}", schedule.client_id, n),
                side: schedule.side,
                kind: schedule.kind,
                tif: schedule.tif,
                units,
                reduce_only: schedule.reduce_only,
            });
        }
        due
    }

    /// Ids of schedules that have released every slice since last asked.
    pub fn drain_completed(&mut self) -> Vec<u64> {
        let done: Vec<u64> =
            self.schedules.iter().filter(|s| s.is_complete() || !s.active).map(|s| s.id).collect();
        self.schedules.retain(|s| !s.is_complete() && s.active);
        done
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SEC: i64 = 1_000_000_000;

    fn engine_with(total: f64, slices: u32, interval_ns: i64) -> (AlgoEngine, u64) {
        let mut engine = AlgoEngine::new();
        let id = engine
            .submit(
                "twap".to_string(),
                OrderSide::Buy,
                OrderKind::Market,
                TimeInForce::Gtc,
                total,
                ExecAlgorithm::Twap { slices, interval_ns },
                false,
                0,
            )
            .expect("valid schedule");
        (engine, id)
    }

    #[test]
    fn one_slice_per_interval() {
        let (mut engine, _) = engine_with(100.0, 4, 60 * SEC);
        assert_eq!(engine.release_due(0).len(), 1, "the first slice is due at once");
        // Before the interval elapses, nothing more.
        assert!(engine.release_due(30 * SEC).is_empty());
        assert_eq!(engine.release_due(60 * SEC).len(), 1);
        assert_eq!(engine.release_due(120 * SEC).len(), 1);
        assert_eq!(engine.release_due(180 * SEC).len(), 1);
        // Four slices released; the schedule is spent.
        assert!(engine.release_due(240 * SEC).is_empty());
    }

    #[test]
    fn slices_sum_to_the_total_exactly() {
        for (total, slices) in [(100.0, 4u32), (100.0, 3), (7.0, 3), (1.0, 7)] {
            let (mut engine, _) = engine_with(total, slices, SEC);
            let mut sum = 0.0;
            for i in 0..slices as i64 {
                for slice in engine.release_due(i * SEC) {
                    sum += slice.units;
                }
            }
            assert_eq!(sum, total, "{slices} slices of {total} must sum exactly");
        }
    }

    #[test]
    fn slices_are_evenly_sized_when_they_divide() {
        let (mut engine, _) = engine_with(100.0, 4, SEC);
        let mut units = Vec::new();
        for i in 0..4 {
            units.extend(engine.release_due(i * SEC).into_iter().map(|s| s.units));
        }
        assert_eq!(units, vec![25.0, 25.0, 25.0, 25.0]);
    }

    #[test]
    fn slice_client_ids_derive_from_the_parent() {
        let (mut engine, _) = engine_with(10.0, 2, SEC);
        assert_eq!(engine.release_due(0)[0].client_id, "twap#0");
        assert_eq!(engine.release_due(SEC)[0].client_id, "twap#1");
    }

    #[test]
    fn a_gap_releases_one_slice_not_the_backlog() {
        // A weekend passes between two bars. Dumping every missed slice
        // would defeat the point of spreading the order.
        let (mut engine, _) = engine_with(100.0, 4, 60 * SEC);
        engine.release_due(0);
        let after_gap = engine.release_due(3_600 * SEC);
        assert_eq!(after_gap.len(), 1, "the schedule stretches, it does not burst");
    }

    #[test]
    fn cancelling_stops_future_slices() {
        let (mut engine, id) = engine_with(100.0, 4, SEC);
        engine.release_due(0);
        engine.release_due(SEC);
        assert!(engine.cancel(id));
        assert!(engine.release_due(2 * SEC).is_empty(), "no slice after cancel");
        assert_eq!(engine.get(id).map(|s| s.released()), Some(2), "two already went out");
    }

    #[test]
    fn cancelling_an_unknown_schedule_is_false() {
        let (mut engine, _) = engine_with(10.0, 2, SEC);
        assert!(!engine.cancel(999));
    }

    #[test]
    fn invalid_schedules_are_refused() {
        let mut engine = AlgoEngine::new();
        let submit = |engine: &mut AlgoEngine, units, slices, interval| {
            engine.submit(
                "x".to_string(),
                OrderSide::Buy,
                OrderKind::Market,
                TimeInForce::Gtc,
                units,
                ExecAlgorithm::Twap { slices, interval_ns: interval },
                false,
                0,
            )
        };
        assert_eq!(submit(&mut engine, 10.0, 0, SEC), Err(AlgoError::ZeroSlices));
        assert_eq!(submit(&mut engine, 10.0, 2, 0), Err(AlgoError::NonPositiveInterval));
        assert_eq!(submit(&mut engine, 0.0, 2, SEC), Err(AlgoError::InvalidUnits));
        assert_eq!(submit(&mut engine, f64::NAN, 2, SEC), Err(AlgoError::InvalidUnits));
    }

    #[test]
    fn a_single_slice_releases_the_whole_order() {
        let (mut engine, _) = engine_with(100.0, 1, SEC);
        let due = engine.release_due(0);
        assert_eq!(due.len(), 1);
        assert_eq!(due[0].units, 100.0);
    }

    #[test]
    fn completed_and_cancelled_schedules_drain() {
        let (mut engine, id) = engine_with(10.0, 1, SEC);
        engine.release_due(0);
        assert_eq!(engine.drain_completed(), vec![id]);
        assert!(engine.is_empty(), "spent schedules do not accumulate");
    }
}
