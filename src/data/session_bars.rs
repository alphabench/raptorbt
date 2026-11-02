//! Day-and-longer bars that follow an instrument's trading calendar.
//!
//! A plain day builder cuts windows at midnight in whatever offset it is
//! given, and builds a bar from every record it sees. With a
//! [`SessionSchedule`] the offset is the schedule's own, so a day bar covers
//! one local trading date, and records that fall on a holiday or weekend are
//! ignored, so such a date produces no bar.
//!
//! Intraday units are untouched: a bar that opens inside a window cannot
//! straddle a holiday, and out-of-session prints are already the caller's
//! to filter or keep.

use crate::core::session_schedule::SessionSchedule;
use crate::core::types::OhlcvBar;
use crate::data::aggregate::{builder_for_with, BarBuilder, SourceRecord};
use crate::data::bar_spec::{AggregationUnit, BarSpec, BuilderParams, SpecError};

/// Whether bars of `unit` are cut at local midnights, which is where a
/// calendar matters.
pub fn follows_calendar(unit: AggregationUnit) -> bool {
    matches!(
        unit,
        AggregationUnit::Day
            | AggregationUnit::Week
            | AggregationUnit::Month
            | AggregationUnit::Year
    )
}

/// Skips records on non-trading days, then defers to the wrapped builder.
#[derive(Debug)]
struct TradingDaysOnly {
    inner: Box<dyn BarBuilder + Send + Sync>,
    schedule: SessionSchedule,
}

impl BarBuilder for TradingDaysOnly {
    fn push(&mut self, rec: &SourceRecord) -> Option<OhlcvBar> {
        self.schedule.session_date(rec.timestamp)?;
        self.inner.push(rec)
    }

    fn flush(&mut self) -> Option<OhlcvBar> {
        self.inner.flush()
    }

    fn next_pending(&mut self) -> Option<OhlcvBar> {
        self.inner.next_pending()
    }
}

/// [`builder_for_with`] that honours a trading calendar for day-and-longer
/// units. Any other unit, or no calendar, builds exactly as before at
/// `align_offset_ns`.
pub fn builder_for_session(
    spec: BarSpec,
    align_offset_ns: i64,
    params: BuilderParams,
    schedule: Option<&SessionSchedule>,
) -> Result<Box<dyn BarBuilder + Send + Sync>, SpecError> {
    match schedule {
        Some(schedule) if follows_calendar(spec.unit) => {
            let inner = builder_for_with(spec, schedule.tz_offset_ns(), params)?;
            Ok(Box::new(TradingDaysOnly { inner, schedule: schedule.clone() }))
        }
        _ => builder_for_with(spec, align_offset_ns, params),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DAY: i64 = 86_400_000_000_000;
    const IST: i64 = (5 * 3600 + 30 * 60) * 1_000_000_000;

    fn rec(ts: i64, price: f64) -> SourceRecord {
        SourceRecord {
            timestamp: ts,
            open: price,
            high: price,
            low: price,
            close: price,
            volume: 1.0,
            signed_volume: 0.0,
        }
    }

    /// Thu 15, Fri 16 (a holiday), Sat 17, Mon 19 Jan 2026 at 11:00 IST, then
    /// Tue 20 so Monday's bar completes. Returns the completed bars.
    fn run(schedule: Option<&SessionSchedule>) -> Vec<i64> {
        let thursday = 20_468 * DAY;
        let spec = BarSpec::new(1, AggregationUnit::Day).unwrap();
        let mut builder =
            builder_for_session(spec, IST, BuilderParams::default(), schedule).unwrap();
        [0, 1, 2, 4, 5]
            .into_iter()
            .filter_map(|d| builder.push(&rec(thursday + d * DAY + 5 * 3600, 100.0)))
            .map(|bar| bar.timestamp)
            .collect()
    }

    #[test]
    fn a_holiday_or_weekend_makes_no_day_bar() {
        let schedule = SessionSchedule::nse_equity().with_holidays([20260116]).unwrap();
        let with = run(Some(&schedule));
        let without = run(None);
        // Without a calendar every date with data closes a bar: Thu, Fri, Sat, Mon.
        assert_eq!(without.len(), 4);
        // With one, only Thursday's and Monday's dates trade.
        assert_eq!(with.len(), 2);
        assert!(with.iter().all(|t| without.contains(t)));
    }

    #[test]
    fn intraday_units_ignore_the_calendar() {
        let spec = BarSpec::new(5, AggregationUnit::Minute).unwrap();
        let schedule = SessionSchedule::nse_equity();
        let mut plain = builder_for_with(spec, 0, BuilderParams::default()).unwrap();
        let mut wrapped =
            builder_for_session(spec, 0, BuilderParams::default(), Some(&schedule)).unwrap();
        let saturday = 20_470 * DAY;
        for i in 0..20 {
            let r = rec(saturday + i * 60_000_000_000, 100.0 + i as f64);
            assert_eq!(plain.push(&r).map(|b| b.timestamp), wrapped.push(&r).map(|b| b.timestamp));
        }
    }
}
