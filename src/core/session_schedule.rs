//! Trading-session calendars: when an instrument can trade.
//!
//! A [`SessionSchedule`] answers "is this instant inside a trading window?"
//! for one venue. It holds a UTC offset, trading windows for each weekday
//! (several per day, so a lunch break is two windows), a set of holiday dates
//! supplied by the caller, and an optional square-off minute for intraday
//! positions.
//!
//! No holiday list ships with the crate. Exchange calendars change every year
//! and are published by the exchange; the caller passes the dates it trusts.
//!
//! All calendar arithmetic is integer arithmetic on days since the Unix epoch
//! (the civil-date algorithms of H. Hinnant), so there is no date dependency
//! and no floating-point rounding at a day boundary.
//!
//! Conventions:
//!
//! * A window is half-open: `open <= t < close`. A print stamped exactly at
//!   the close is outside the session; one stamped exactly at the open is in.
//! * Minutes are local minutes from midnight (`9 * 60 + 15` is 09:15).
//! * Dates are `yyyymmdd` integers (`20260115`).
//! * Weekday windows are indexed Monday = 0 through Sunday = 6.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::core::error::RaptorError;
use crate::core::Result;

const NS_PER_MINUTE: i64 = 60_000_000_000;
const NS_PER_DAY: i64 = 86_400_000_000_000;
const MINUTES_PER_DAY: u32 = 24 * 60;

/// India Standard Time, UTC+5:30, in nanoseconds.
const IST_OFFSET_NS: i64 = (5 * 3600 + 30 * 60) * 1_000_000_000;

/// How many days `next_open_ns` looks ahead before giving up, so a schedule
/// whose every day is a holiday ends the search instead of looping.
const NEXT_OPEN_HORIZON_DAYS: i64 = 400;

/// Days since 1970-01-01 for a proleptic Gregorian date.
fn days_from_civil(year: i64, month: u32, day: u32) -> i64 {
    let y = if month <= 2 { year - 1 } else { year };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = (month as i64 + 9) % 12;
    let doy = (153 * mp + 2) / 5 + day as i64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// The `(year, month, day)` of a day count since 1970-01-01.
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let month = (if mp < 10 { mp + 3 } else { mp - 9 }) as u32;
    let year = yoe + era * 400 + i64::from(month <= 2);
    (year, month, day)
}

/// Day count for a `yyyymmdd` date, or `None` if it is not a real date in
/// 1900..=2200.
pub fn day_from_yyyymmdd(date: i32) -> Option<i64> {
    let (year, month, day) = (date / 10_000, (date / 100) % 100, date % 100);
    if !(1900..=2200).contains(&year) || !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return None;
    }
    let days = days_from_civil(year as i64, month as u32, day as u32);
    // Round-trip rejects 20260231 and friends.
    (civil_from_days(days) == (year as i64, month as u32, day as u32)).then_some(days)
}

/// `yyyymmdd` for a day count since 1970-01-01.
pub fn yyyymmdd_from_day(day: i64) -> i32 {
    let (y, m, d) = civil_from_days(day);
    (y * 10_000 + m as i64 * 100 + d as i64) as i32
}

/// Monday = 0 .. Sunday = 6 for a day count (1970-01-01 was a Thursday).
fn weekday_of(day: i64) -> usize {
    (day + 3).rem_euclid(7) as usize
}

/// One venue's trading calendar.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionSchedule {
    tz_offset_ns: i64,
    /// `(open_minute, close_minute)` windows per weekday, Monday first.
    windows: [Vec<(u32, u32)>; 7],
    holidays: BTreeSet<i32>,
    squareoff_minute: Option<u32>,
}

impl SessionSchedule {
    /// Build and validate a schedule.
    ///
    /// Errors name the offending field: an offset outside +-24h, a window that
    /// does not open before it closes, overlapping windows on one day, a bad
    /// holiday date, a square-off minute outside every window, or a schedule
    /// with no trading window at all.
    pub fn new(
        tz_offset_ns: i64,
        windows: [Vec<(u32, u32)>; 7],
        holidays: impl IntoIterator<Item = i32>,
        squareoff_minute: Option<u32>,
    ) -> Result<Self> {
        if tz_offset_ns.abs() >= NS_PER_DAY {
            return Err(RaptorError::invalid_parameter(format!(
                "tz_offset_ns must be within +-24 hours, got {tz_offset_ns}"
            )));
        }
        const NAMES: [&str; 7] = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"];
        let mut windows = windows;
        for (weekday, day_windows) in windows.iter_mut().enumerate() {
            day_windows.sort_unstable();
            let mut previous_close = 0;
            for &(open, close) in day_windows.iter() {
                if close > MINUTES_PER_DAY {
                    return Err(RaptorError::invalid_parameter(format!(
                        "{} window ({open}, {close}): close minute must be <= {MINUTES_PER_DAY}",
                        NAMES[weekday]
                    )));
                }
                if open >= close {
                    return Err(RaptorError::invalid_parameter(format!(
                        "{} window ({open}, {close}): open minute must be before close minute",
                        NAMES[weekday]
                    )));
                }
                if open < previous_close {
                    return Err(RaptorError::invalid_parameter(format!(
                        "{} windows overlap at minute {open}",
                        NAMES[weekday]
                    )));
                }
                previous_close = close;
            }
        }
        if windows.iter().all(Vec::is_empty) {
            return Err(RaptorError::invalid_parameter(
                "a session schedule needs at least one trading window",
            ));
        }
        let holidays = Self::checked_holidays(holidays)?;
        let schedule = Self { tz_offset_ns, windows, holidays, squareoff_minute: None };
        schedule.with_squareoff_minute(squareoff_minute)
    }

    fn checked_holidays(dates: impl IntoIterator<Item = i32>) -> Result<BTreeSet<i32>> {
        let mut set = BTreeSet::new();
        for date in dates {
            if day_from_yyyymmdd(date).is_none() {
                return Err(RaptorError::invalid_parameter(format!(
                    "holiday {date} is not a yyyymmdd date between 1900 and 2200"
                )));
            }
            set.insert(date);
        }
        Ok(set)
    }

    /// The same schedule with exactly these holidays (replacing any set
    /// before). Dates are `yyyymmdd`.
    pub fn with_holidays(mut self, dates: impl IntoIterator<Item = i32>) -> Result<Self> {
        self.holidays = Self::checked_holidays(dates)?;
        Ok(self)
    }

    /// The same schedule with another square-off minute (`None` disables it).
    ///
    /// The minute must lie inside a trading window or at its close: a
    /// square-off that can never be reached is a typo, not a setting.
    pub fn with_squareoff_minute(mut self, minute: Option<u32>) -> Result<Self> {
        if let Some(m) = minute {
            let reachable =
                self.windows.iter().flatten().any(|&(open, close)| open <= m && m <= close);
            if !reachable {
                return Err(RaptorError::invalid_parameter(format!(
                    "squareoff_minute {m} is outside every trading window"
                )));
            }
        }
        self.squareoff_minute = minute;
        Ok(self)
    }

    /// Monday-to-Friday, one `[open, close)` window, at an offset.
    fn weekdays(
        tz_offset_ns: i64,
        open: u32,
        close: u32,
        squareoff_minute: Option<u32>,
    ) -> Result<Self> {
        let day = vec![(open, close)];
        Self::new(
            tz_offset_ns,
            [day.clone(), day.clone(), day.clone(), day.clone(), day, vec![], vec![]],
            [],
            squareoff_minute,
        )
    }

    /// NSE cash equities: Mon-Fri 09:15-15:30 IST, square-off 15:20.
    ///
    /// The regular session only; the pre-open and post-close auctions are not
    /// modelled. 15:20 is a typical broker cut-off for intraday (MIS)
    /// positions, not an exchange rule: it varies by broker and product, so
    /// override it with [`SessionSchedule::with_squareoff_minute`].
    pub fn nse_equity() -> Self {
        Self::weekdays(IST_OFFSET_NS, 9 * 60 + 15, 15 * 60 + 30, Some(15 * 60 + 20))
            .expect("preset is valid")
    }

    /// NSE futures and options: Mon-Fri 09:15-15:30 IST, square-off 15:20.
    pub fn nse_fo() -> Self {
        Self::nse_equity()
    }

    /// BSE cash equities: Mon-Fri 09:15-15:30 IST, square-off 15:20.
    pub fn bse_equity() -> Self {
        Self::nse_equity()
    }

    pub fn tz_offset_ns(&self) -> i64 {
        self.tz_offset_ns
    }

    /// Windows for a weekday, Monday = 0.
    pub fn windows(&self, weekday: usize) -> &[(u32, u32)] {
        &self.windows[weekday]
    }

    pub fn holidays(&self) -> impl Iterator<Item = i32> + '_ {
        self.holidays.iter().copied()
    }

    pub fn squareoff_minute(&self) -> Option<u32> {
        self.squareoff_minute
    }

    /// `(local day number, nanoseconds into that local day)`.
    #[inline]
    fn local(&self, ts_ns: i64) -> (i64, i64) {
        let local = ts_ns + self.tz_offset_ns;
        (local.div_euclid(NS_PER_DAY), local.rem_euclid(NS_PER_DAY))
    }

    /// Windows of a local day, or `None` when it is a holiday or has none.
    fn trading_windows(&self, day: i64) -> Option<&[(u32, u32)]> {
        let windows = &self.windows[weekday_of(day)];
        if windows.is_empty() || self.holidays.contains(&yyyymmdd_from_day(day)) {
            return None;
        }
        Some(windows)
    }

    /// Whether `ts_ns` falls inside a trading window (`open <= t < close`).
    pub fn is_open(&self, ts_ns: i64) -> bool {
        let (day, tod) = self.local(ts_ns);
        self.trading_windows(day).is_some_and(|windows| {
            windows.iter().any(|&(open, close)| {
                open as i64 * NS_PER_MINUTE <= tod && tod < close as i64 * NS_PER_MINUTE
            })
        })
    }

    /// Local trading day number of `ts_ns`, or `None` on a weekend, a
    /// holiday or any day without a window. Time of day is ignored, so a
    /// print before the open or after the close still belongs to its day.
    pub(crate) fn session_day(&self, ts_ns: i64) -> Option<i64> {
        let (day, _) = self.local(ts_ns);
        self.trading_windows(day).map(|_| day)
    }

    /// `yyyymmdd` of the trading day `ts_ns` falls on; see
    /// [`SessionSchedule::session_day`] for what has no session.
    pub fn session_date(&self, ts_ns: i64) -> Option<i32> {
        self.session_day(ts_ns).map(yyyymmdd_from_day)
    }

    /// UTC nanoseconds of the last close on a trading date, or `None` when
    /// the date is not a real date or is not a trading day.
    pub fn session_close_ns(&self, date: i32) -> Option<i64> {
        let day = day_from_yyyymmdd(date)?;
        self.close_of_day(day)
    }

    fn close_of_day(&self, day: i64) -> Option<i64> {
        let &(_, close) = self.trading_windows(day)?.last()?;
        Some(day * NS_PER_DAY + close as i64 * NS_PER_MINUTE - self.tz_offset_ns)
    }

    /// The first window-open instant strictly after `ts_ns`, or `None` if
    /// there is none within a year (every day a holiday).
    ///
    /// Inside a window this is the next window's open: after a lunch break,
    /// or the next trading day.
    pub fn next_open_ns(&self, ts_ns: i64) -> Option<i64> {
        let (first_day, _) = self.local(ts_ns);
        for day in first_day..first_day + NEXT_OPEN_HORIZON_DAYS {
            let Some(windows) = self.trading_windows(day) else { continue };
            for &(open, _) in windows {
                let open_ns = day * NS_PER_DAY + open as i64 * NS_PER_MINUTE - self.tz_offset_ns;
                if open_ns > ts_ns {
                    return Some(open_ns);
                }
            }
        }
        None
    }

    /// When a DAY order placed at `submitted_ts_ns` lapses: the close of its
    /// session, or of the next session when it was placed after the close or
    /// on a non-trading day.
    pub(crate) fn day_order_expiry_ns(&self, submitted_ts_ns: i64) -> Option<i64> {
        let (first_day, _) = self.local(submitted_ts_ns);
        (first_day..first_day + NEXT_OPEN_HORIZON_DAYS)
            .filter_map(|day| self.close_of_day(day))
            .find(|&close| close > submitted_ts_ns)
    }

    /// The trading day whose square-off is due at `ts_ns`, or `None`.
    ///
    /// Due on the first event at or after the square-off minute on a trading
    /// day. `last_done` is the day number last returned; passing it back
    /// makes the answer fire once per day, which is the contract of
    /// `squareoff_flags` applied to a calendar with holidays and weekends.
    pub(crate) fn squareoff_due(&self, ts_ns: i64, last_done: Option<i64>) -> Option<i64> {
        let minute = self.squareoff_minute?;
        let (day, tod) = self.local(ts_ns);
        if tod < minute as i64 * NS_PER_MINUTE || last_done == Some(day) {
            return None;
        }
        self.trading_windows(day).map(|_| day)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::session::squareoff_flags;

    const HOUR: i64 = 3_600 * 1_000_000_000;
    const MIN: i64 = NS_PER_MINUTE;

    /// UTC ns of an IST wall-clock time.
    fn ist(date: i32, hour: i64, minute: i64) -> i64 {
        day_from_yyyymmdd(date).unwrap() * NS_PER_DAY + hour * HOUR + minute * MIN - IST_OFFSET_NS
    }

    #[test]
    fn civil_dates_round_trip_and_know_the_weekday() {
        assert_eq!(days_from_civil(1970, 1, 1), 0);
        assert_eq!(weekday_of(0), 3); // Thursday
        assert_eq!(day_from_yyyymmdd(20260115), Some(days_from_civil(2026, 1, 15)));
        assert_eq!(yyyymmdd_from_day(day_from_yyyymmdd(20240229).unwrap()), 20240229);
        // 2026-01-15 is a Thursday, 2026-01-17 a Saturday.
        assert_eq!(weekday_of(day_from_yyyymmdd(20260115).unwrap()), 3);
        assert_eq!(weekday_of(day_from_yyyymmdd(20260117).unwrap()), 5);
        assert_eq!(day_from_yyyymmdd(20260231), None);
        assert_eq!(day_from_yyyymmdd(20251301), None);
        assert_eq!(day_from_yyyymmdd(20230229), None);
        assert!(day_from_yyyymmdd(20240229).is_some());
    }

    #[test]
    fn nse_window_is_half_open_and_weekday_only() {
        let s = SessionSchedule::nse_equity();
        assert!(!s.is_open(ist(20260115, 9, 14)));
        assert!(s.is_open(ist(20260115, 9, 15)));
        assert!(s.is_open(ist(20260115, 15, 29)));
        assert!(!s.is_open(ist(20260115, 15, 30)));
        assert!(!s.is_open(ist(20260117, 10, 0)), "Saturday");
        assert!(!s.is_open(ist(20260118, 10, 0)), "Sunday");
        assert!(s.is_open(ist(20260119, 10, 0)), "Monday");
    }

    #[test]
    fn timezone_decides_the_local_day() {
        // 03:45 UTC is 09:15 IST; the same instant is closed in UTC.
        let utc_instant = days_from_civil(2026, 1, 15) * NS_PER_DAY + 3 * HOUR + 45 * MIN;
        assert!(SessionSchedule::nse_equity().is_open(utc_instant));
        let utc = SessionSchedule::weekdays(0, 9 * 60 + 15, 15 * 60 + 30, None).unwrap();
        assert!(!utc.is_open(utc_instant));
        // Negative offset (New York winter): 14:30 UTC is 09:30 local.
        let ny = SessionSchedule::weekdays(-5 * HOUR, 9 * 60 + 30, 16 * 60, None).unwrap();
        assert!(ny.is_open(days_from_civil(2026, 1, 15) * NS_PER_DAY + 14 * HOUR + 30 * MIN));
        assert!(!ny.is_open(days_from_civil(2026, 1, 15) * NS_PER_DAY + 14 * HOUR + 29 * MIN));
    }

    #[test]
    fn a_lunch_break_is_a_gap_between_two_windows() {
        let day = vec![(9 * 60, 12 * 60), (13 * 60, 15 * 60)];
        let s = SessionSchedule::new(
            0,
            [day.clone(), day.clone(), day.clone(), day.clone(), day, vec![], vec![]],
            [],
            None,
        )
        .unwrap();
        let at = |h, m| ist(20260115, h, m) + IST_OFFSET_NS;
        assert!(s.is_open(at(11, 59)));
        assert!(!s.is_open(at(12, 0)));
        assert!(!s.is_open(at(12, 59)));
        assert!(s.is_open(at(13, 0)));
        assert_eq!(s.next_open_ns(at(10, 0)), Some(at(13, 0)));
        assert_eq!(s.next_open_ns(at(12, 30)), Some(at(13, 0)));
        assert_eq!(s.session_close_ns(20260115), Some(at(15, 0)));
    }

    #[test]
    fn holidays_close_the_day_and_are_caller_supplied() {
        let plain = SessionSchedule::nse_equity();
        assert!(plain.is_open(ist(20260126, 10, 0)), "no calendar ships with the crate");
        let s = plain.with_holidays([20260126]).unwrap();
        assert!(!s.is_open(ist(20260126, 10, 0)));
        assert_eq!(s.session_date(ist(20260126, 10, 0)), None);
        assert_eq!(s.session_close_ns(20260126), None);
        // Monday 26th is a holiday, so the open after Friday's close is Tuesday's.
        assert_eq!(s.next_open_ns(ist(20260123, 15, 30)), Some(ist(20260127, 9, 15)));
        // with_holidays replaces rather than accumulates.
        let replaced = s.with_holidays([20260127]).unwrap();
        assert!(replaced.is_open(ist(20260126, 10, 0)));
        assert!(!replaced.is_open(ist(20260127, 10, 0)));
    }

    #[test]
    fn session_date_ignores_time_of_day_but_not_weekends() {
        let s = SessionSchedule::nse_equity();
        assert_eq!(s.session_date(ist(20260115, 3, 0)), Some(20260115));
        assert_eq!(s.session_date(ist(20260115, 23, 59)), Some(20260115));
        assert_eq!(s.session_date(ist(20260117, 12, 0)), None);
    }

    #[test]
    fn next_open_is_strictly_after() {
        let s = SessionSchedule::nse_equity();
        assert_eq!(s.next_open_ns(ist(20260115, 8, 0)), Some(ist(20260115, 9, 15)));
        assert_eq!(s.next_open_ns(ist(20260115, 9, 15)), Some(ist(20260116, 9, 15)));
        assert_eq!(s.next_open_ns(ist(20260116, 16, 0)), Some(ist(20260119, 9, 15)));
    }

    #[test]
    fn next_open_gives_up_when_every_day_is_a_holiday() {
        let first = day_from_yyyymmdd(20260101).unwrap();
        let all = (first - 5..first + 500).map(yyyymmdd_from_day);
        let s = SessionSchedule::nse_equity().with_holidays(all).unwrap();
        assert_eq!(s.next_open_ns(ist(20260105, 10, 0)), None);
    }

    #[test]
    fn presets_match_their_documented_times() {
        for s in [
            SessionSchedule::nse_equity(),
            SessionSchedule::nse_fo(),
            SessionSchedule::bse_equity(),
        ] {
            assert_eq!(s.tz_offset_ns(), IST_OFFSET_NS);
            for weekday in 0..5 {
                assert_eq!(s.windows(weekday), &[(555, 930)]);
            }
            assert!(s.windows(5).is_empty() && s.windows(6).is_empty());
            assert_eq!(s.squareoff_minute(), Some(920));
            assert_eq!(s.holidays().count(), 0);
        }
    }

    #[test]
    fn validation_names_the_problem() {
        let msg = |r: Result<SessionSchedule>| r.unwrap_err().to_string();
        let ok = vec![(540, 900)];
        let week = |w: Vec<(u32, u32)>| [w, vec![], vec![], vec![], vec![], vec![], vec![]];
        assert!(msg(SessionSchedule::new(25 * HOUR, week(ok.clone()), [], None))
            .contains("tz_offset_ns"));
        assert!(
            msg(SessionSchedule::new(0, week(vec![(900, 540)]), [], None)).contains("before close")
        );
        assert!(msg(SessionSchedule::new(0, week(vec![(0, 1500)]), [], None)).contains("<= 1440"));
        assert!(msg(SessionSchedule::new(0, week(vec![(540, 700), (600, 900)]), [], None))
            .contains("overlap"));
        assert!(msg(SessionSchedule::new(0, week(vec![]), [], None)).contains("at least one"));
        assert!(
            msg(SessionSchedule::new(0, week(ok.clone()), [20260230], None)).contains("holiday")
        );
        assert!(msg(SessionSchedule::new(0, week(ok), [], Some(1000))).contains("squareoff_minute"));
    }

    #[test]
    fn day_orders_lapse_at_the_close_of_their_session() {
        let s = SessionSchedule::nse_equity();
        assert_eq!(s.day_order_expiry_ns(ist(20260115, 10, 0)), Some(ist(20260115, 15, 30)));
        // Placed after the close, or at the weekend: the next session's close.
        assert_eq!(s.day_order_expiry_ns(ist(20260115, 16, 0)), Some(ist(20260116, 15, 30)));
        assert_eq!(s.day_order_expiry_ns(ist(20260117, 10, 0)), Some(ist(20260119, 15, 30)));
    }

    #[test]
    fn squareoff_fires_once_per_trading_day_and_agrees_with_squareoff_flags() {
        let s = SessionSchedule::nse_equity();
        let sq = s.squareoff_minute().unwrap();
        assert_eq!(sq, 15 * 60 + 20);
        let stamps: Vec<i64> = [
            ist(20260115, 15, 19),
            ist(20260115, 15, 20),
            ist(20260115, 15, 25),
            ist(20260116, 15, 19),
            ist(20260116, 15, 21),
        ]
        .to_vec();
        let flags = squareoff_flags(&stamps, sq, IST_OFFSET_NS);
        let mut last = None;
        let ours: Vec<bool> = stamps
            .iter()
            .map(|&t| match s.squareoff_due(t, last) {
                Some(day) => {
                    last = Some(day);
                    true
                }
                None => false,
            })
            .collect();
        assert_eq!(ours, flags);
        assert_eq!(ours, [false, true, false, false, true]);
    }

    #[test]
    fn squareoff_skips_holidays_and_weekends() {
        let s = SessionSchedule::nse_equity().with_holidays([20260126]).unwrap();
        assert_eq!(s.squareoff_due(ist(20260126, 15, 21), None), None);
        assert_eq!(s.squareoff_due(ist(20260117, 15, 21), None), None);
        let no_squareoff = SessionSchedule::nse_equity().with_squareoff_minute(None).unwrap();
        assert_eq!(no_squareoff.squareoff_due(ist(20260115, 15, 21), None), None);
    }
}
