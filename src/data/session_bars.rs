//! Day-and-longer bars that follow an instrument's trading calendar.
//!
//! A plain day builder cuts windows at midnight in whatever offset it is
//! given, and builds a bar from every record it sees. With a
//! [`SessionSchedule`] the offset is the schedule's own, so a day bar covers
//! one local trading date, and records that fall on a holiday or weekend are
//! ignored, so such a date produces no bar.
//!
//! Intraday time units are cut from the session open, not from the clock: a
//! 30-minute bar of a 09:15 session runs 09:15-09:45, the way a venue's own
//! candles do, and the last bar of a window ends at its close (a 60-minute
//! bar of 09:15-15:30 ends 15:15-15:30, not at 16:15). Each trading window of
//! a day is anchored at its own open, so a lunch break or an evening session
//! starts a fresh grid. A print outside every window falls back to the clock
//! grid of the schedule's zone and never joins a session bar.

use crate::core::session_schedule::SessionSchedule;
use crate::core::types::{OhlcvBar, Timestamp};
use crate::data::aggregate::{builder_for_with, BarBuilder, Partial, SourceRecord};
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

/// Whether bars of `unit` depend on the instrument's calendar: the day and
/// longer units cut at its midnight, the intraday time units at its session
/// open.
pub fn uses_calendar(unit: AggregationUnit) -> bool {
    follows_calendar(unit) || unit.fixed_ns().is_some()
}

/// Time bars of a fixed width anchored at the open of the trading window a
/// record falls in.
#[derive(Debug)]
struct SessionAnchored {
    width_ns: i64,
    schedule: SessionSchedule,
    /// `[start, end)` of the bar in progress.
    window: Option<(Timestamp, Timestamp)>,
    partial: Option<Partial>,
}

impl SessionAnchored {
    fn window_of(&self, ts: Timestamp) -> (Timestamp, Timestamp) {
        for (open, close) in self.schedule.windows_ns_on_day_of(ts) {
            if open <= ts && ts < close {
                let start = open + (ts - open).div_euclid(self.width_ns) * self.width_ns;
                return (start, (start + self.width_ns).min(close));
            }
        }
        let offset = self.schedule.tz_offset_ns();
        let start = (ts + offset).div_euclid(self.width_ns) * self.width_ns - offset;
        (start, start + self.width_ns)
    }
}

impl SessionAnchored {
    /// Whether a record bucketed at `ts` is the last the window `end` can take, and that end is
    /// a trading window's close: a close-stamped bar reaches the close (it is bucketed by the
    /// last nanosecond it covers), so nothing more of the window follows. A bar stamped at its
    /// open cannot say so and waits for the next window's first record, as every bucket does.
    fn reaches_close(&self, ts: Timestamp, end: Timestamp) -> bool {
        ts + 1 >= end
            && self.schedule.windows_ns_on_day_of(ts).iter().any(|&(_, close)| close == end)
    }
}

impl BarBuilder for SessionAnchored {
    fn push(&mut self, rec: &SourceRecord) -> Option<OhlcvBar> {
        let window = self.window_of(rec.timestamp);
        let mut done = None;
        match (self.window, &mut self.partial) {
            (Some(current), Some(partial)) if current == window => partial.absorb(rec),
            (Some((_, end)), slot) => {
                done = slot.take().map(|p| p.into_bar(end));
                self.window = Some(window);
                self.partial = Some(Partial::start(rec));
            }
            (None, _) => {
                self.window = Some(window);
                self.partial = Some(Partial::start(rec));
            }
        }
        // The session's last bucket is complete at the close: it is emitted there, not with the
        // first record of the next session (a live strategy would get it the next morning). A
        // record that also closed an earlier bucket returns that one, so a tail made of that
        // one record waits for the next push or the flush.
        if done.is_none() && self.reaches_close(rec.timestamp, window.1) {
            self.window = None;
            return self.partial.take().map(|p| p.into_bar(window.1));
        }
        done
    }

    fn flush(&mut self) -> Option<OhlcvBar> {
        let (_, end) = self.window.take()?;
        self.partial.take().map(|p| p.into_bar(end))
    }
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

/// [`builder_for_with`] that honours a trading calendar: day-and-longer units
/// cut at its midnight and skip non-trading days, intraday time units anchor
/// at the session open. Any other unit, or no calendar, builds exactly as
/// before at `align_offset_ns`.
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
        Some(schedule) if spec.unit.fixed_ns().is_some() => {
            let width_ns = spec.unit.fixed_ns().unwrap_or(1) * i64::from(spec.step);
            Ok(Box::new(SessionAnchored {
                width_ns,
                schedule: schedule.clone(),
                window: None,
                partial: None,
            }))
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

    const MIN: i64 = 60_000_000_000;

    /// Bar end times from one-minute records opening at `open_min` (local
    /// minutes of 2026-01-15, an IST Thursday) for `n` minutes.
    fn ends(spec: BarSpec, schedule: &SessionSchedule, open_min: i64, n: i64) -> Vec<i64> {
        let thursday = 20_468 * DAY - IST;
        let mut b = builder_for_session(spec, 0, BuilderParams::default(), Some(schedule)).unwrap();
        let mut out: Vec<i64> = (0..n)
            .filter_map(|i| b.push(&rec(thursday + (open_min + i) * MIN, 100.0 + i as f64)))
            .map(|bar| (bar.timestamp - thursday) / MIN)
            .collect();
        out.extend(b.flush().map(|bar| (bar.timestamp - thursday) / MIN));
        out
    }

    #[test]
    fn intraday_bars_are_cut_from_the_session_open_and_end_at_the_close() {
        let nse = SessionSchedule::nse_equity();
        let open = 9 * 60 + 15;
        let thirty = BarSpec::new(30, AggregationUnit::Minute).unwrap();
        let got = ends(thirty, &nse, open, 375);
        // 09:45, 10:15, ... : twelve whole bars and the last 15 minutes.
        assert_eq!(got.len(), 13);
        assert_eq!(&got[..2], &[open + 30, open + 60]);
        assert_eq!(*got.last().unwrap(), 15 * 60 + 30, "the last bar closes with the session");
        let hour = BarSpec::new(1, AggregationUnit::Hour).unwrap();
        let got = ends(hour, &nse, open, 375);
        assert_eq!(got.len(), 7);
        assert_eq!(*got.last().unwrap(), 15 * 60 + 30, "15:15-15:30, not 15:15-16:15");
    }

    #[test]
    fn a_window_that_closes_off_the_grid_ends_its_last_bar_there() {
        // The equity-derivatives close after 2026-08-03 is 15:40: 09:15 + 77 x 5 minutes.
        let fo = SessionSchedule::nse_fo();
        let thursday_after = 20_671 * DAY - IST; // a weekday after the change
        let five = BarSpec::new(5, AggregationUnit::Minute).unwrap();
        let mut b = builder_for_session(five, 0, BuilderParams::default(), Some(&fo)).unwrap();
        let mut last = None;
        for i in 0..385 {
            if let Some(bar) = b.push(&rec(thursday_after + (555 + i) * MIN, 100.0)) {
                last = Some(bar.timestamp);
            }
        }
        last = b.flush().map(|bar| bar.timestamp).or(last);
        assert_eq!(last, Some(thursday_after + (15 * 60 + 40) * MIN));
    }

    #[test]
    fn an_evening_session_closing_at_2355_ends_its_last_bar_at_the_close_not_after_midnight() {
        // MCX in US standard time: the evening window 17:00-23:55. A bucket of 30, 60 or 75
        // minutes from 17:00 that would end after 23:55 ends at the close, in its own day.
        let mcx = SessionSchedule::mcx();
        for (width, unit) in [
            (30, AggregationUnit::Minute),
            (1, AggregationUnit::Hour),
            (75, AggregationUnit::Minute),
        ] {
            let spec = BarSpec::new(width, unit).unwrap();
            let got = ends(spec, &mcx, 17 * 60, 415); // one-minute records 17:00 to 23:54
            assert_eq!(*got.last().unwrap(), 23 * 60 + 55, "{width} {unit:?}");
            assert!(got.iter().all(|&m| m <= 23 * 60 + 55));
        }
    }

    #[test]
    fn the_last_bucket_of_a_close_stamped_session_is_emitted_at_the_close() {
        // One-minute bars stamped at their close, 17:01 to 23:55, bucketed by their last
        // nanosecond as the runtime does: the 23:25-23:55 tail comes out with the bar that
        // reaches the close, not with the first record of the next session or a flush.
        let mcx = SessionSchedule::mcx();
        let thursday = 20_468 * DAY - IST;
        let spec = BarSpec::new(30, AggregationUnit::Minute).unwrap();
        let mut b = builder_for_session(spec, 0, BuilderParams::default(), Some(&mcx)).unwrap();
        let mut ends = Vec::new();
        for close in (17 * 60 + 1)..=(23 * 60 + 55) {
            if let Some(bar) = b.push(&rec(thursday + close * MIN - 1, 100.0)) {
                ends.push((bar.timestamp - thursday) / MIN);
                assert!(close <= 23 * 60 + 55);
            }
        }
        assert_eq!(*ends.last().unwrap(), 23 * 60 + 55, "{ends:?}");
        assert!(b.flush().is_none(), "nothing is left to wait for the next session");
        // a bar stamped at its open cannot say it reaches the close: it waits to be flushed
        let mut b = builder_for_session(spec, 0, BuilderParams::default(), Some(&mcx)).unwrap();
        for open in (17 * 60)..(23 * 60 + 55) {
            if let Some(bar) = b.push(&rec(thursday + open * MIN, 100.0)) {
                assert!(bar.timestamp <= thursday + (23 * 60 + 30) * MIN, "before the tail");
            }
        }
        assert_eq!(b.flush().map(|bar| (bar.timestamp - thursday) / MIN), Some(23 * 60 + 55));
    }

    #[test]
    fn without_a_calendar_intraday_bars_are_cut_on_the_clock_grid() {
        let spec = BarSpec::new(30, AggregationUnit::Minute).unwrap();
        let mut b = builder_for_session(spec, IST, BuilderParams::default(), None).unwrap();
        let thursday = 20_468 * DAY - IST;
        let mut first = None;
        for i in 0..60 {
            first = first.or(b.push(&rec(thursday + (555 + i) * MIN, 100.0)));
        }
        assert_eq!((first.unwrap().timestamp - thursday) / MIN, 9 * 60 + 30);
    }
}
