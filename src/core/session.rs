//! Session tracking for intraday strategies.
//!
//! Handles:
//! - Session boundary detection (market open/close)
//! - Squareoff time enforcement
//! - Session high/low tracking for ORB and session-based indicators
//! - Timezone handling for IST (India Standard Time)

use serde::{Deserialize, Serialize};

/// Session configuration for trading hours.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionConfig {
    /// Market open hour (24-hour format).
    pub market_open_hour: u32,
    /// Market open minute.
    pub market_open_minute: u32,
    /// Market close hour (24-hour format).
    pub market_close_hour: u32,
    /// Market close minute.
    pub market_close_minute: u32,
    /// Squareoff minutes before market close.
    pub squareoff_minutes_before_close: u32,
    /// Timezone offset in hours from UTC (5 for IST = UTC+5:30).
    pub timezone_offset_hours: i32,
    /// Timezone offset minutes (30 for IST).
    pub timezone_offset_minutes: i32,
}

impl Default for SessionConfig {
    fn default() -> Self {
        // Default: NSE equity session (9:15 - 15:30 IST, squareoff at 15:25)
        Self {
            market_open_hour: 9,
            market_open_minute: 15,
            market_close_hour: 15,
            market_close_minute: 30,
            squareoff_minutes_before_close: 5,
            timezone_offset_hours: 5,
            timezone_offset_minutes: 30,
        }
    }
}

impl SessionConfig {
    /// Create NSE equity session config (9:15 - 15:30).
    pub fn nse_equity() -> Self {
        Self::default()
    }

    /// Create MCX commodity session config (9:00 - 23:30).
    pub fn mcx_commodity() -> Self {
        Self {
            market_open_hour: 9,
            market_open_minute: 0,
            market_close_hour: 23,
            market_close_minute: 30,
            squareoff_minutes_before_close: 5,
            timezone_offset_hours: 5,
            timezone_offset_minutes: 30,
        }
    }

    /// Create CDS currency session config (9:00 - 17:00).
    pub fn cds_currency() -> Self {
        Self {
            market_open_hour: 9,
            market_open_minute: 0,
            market_close_hour: 17,
            market_close_minute: 0,
            squareoff_minutes_before_close: 5,
            timezone_offset_hours: 5,
            timezone_offset_minutes: 30,
        }
    }

    /// Get market open time in minutes from midnight.
    pub fn market_open_minutes(&self) -> u32 {
        self.market_open_hour * 60 + self.market_open_minute
    }

    /// Get market close time in minutes from midnight.
    pub fn market_close_minutes(&self) -> u32 {
        self.market_close_hour * 60 + self.market_close_minute
    }

    /// Get squareoff time in minutes from midnight.
    pub fn squareoff_minutes(&self) -> u32 {
        self.market_close_minutes().saturating_sub(self.squareoff_minutes_before_close)
    }

    /// Session length in minutes.
    ///
    /// Used to annualize intraday returns: an intraday series only accrues
    /// returns while the market is open, so bars-per-session drives the period
    /// count. NSE equity is 375 minutes; MCX commodity is 870, which is why
    /// this cannot be a single hardcoded constant.
    pub fn session_minutes(&self) -> u32 {
        self.market_close_minutes().saturating_sub(self.market_open_minutes())
    }

    /// Get timezone offset in seconds.
    pub fn timezone_offset_seconds(&self) -> i64 {
        (self.timezone_offset_hours as i64 * 3600) + (self.timezone_offset_minutes as i64 * 60)
    }
}

/// Session tracker for managing intraday session state.
#[derive(Debug, Clone)]
pub struct SessionTracker {
    config: SessionConfig,
    /// Current session date (days since epoch in local timezone).
    current_session_date: i64,
    /// Session high price.
    session_high: f64,
    /// Session low price.
    session_low: f64,
    /// Session open price.
    session_open: f64,
    /// Bar index at session start.
    session_start_idx: usize,
    /// Whether we're currently in a trading session.
    in_session: bool,
    /// Whether squareoff has been triggered today.
    squareoff_triggered: bool,
}

impl SessionTracker {
    /// Create a new session tracker.
    pub fn new(config: SessionConfig) -> Self {
        Self {
            config,
            current_session_date: -1,
            session_high: f64::NEG_INFINITY,
            session_low: f64::INFINITY,
            session_open: 0.0,
            session_start_idx: 0,
            in_session: false,
            squareoff_triggered: false,
        }
    }

    /// Convert nanosecond timestamp to local time components.
    fn timestamp_to_local(&self, timestamp_ns: i64) -> (i64, u32, u32, u32) {
        // Convert to seconds
        let timestamp_s = timestamp_ns / 1_000_000_000;

        // Apply timezone offset
        let local_s = timestamp_s + self.config.timezone_offset_seconds();

        // Calculate date (days since epoch)
        let days = local_s / 86400;

        // Calculate time within day
        let time_in_day = (local_s % 86400) as u32;
        let hours = time_in_day / 3600;
        let minutes = (time_in_day % 3600) / 60;
        let seconds = time_in_day % 60;

        (days, hours, minutes, seconds)
    }

    /// Get minutes from midnight for a timestamp.
    fn get_minutes_from_midnight(&self, timestamp_ns: i64) -> u32 {
        let (_, hours, minutes, _) = self.timestamp_to_local(timestamp_ns);
        hours * 60 + minutes
    }

    /// Check if timestamp is within trading hours.
    pub fn is_within_trading_hours(&self, timestamp_ns: i64) -> bool {
        let minutes = self.get_minutes_from_midnight(timestamp_ns);
        minutes >= self.config.market_open_minutes() && minutes < self.config.market_close_minutes()
    }

    /// Check if it's squareoff time.
    pub fn is_squareoff_time(&self, timestamp_ns: i64) -> bool {
        let minutes = self.get_minutes_from_midnight(timestamp_ns);
        minutes >= self.config.squareoff_minutes()
    }

    /// Check if this bar starts a new session.
    pub fn is_session_start(&self, prev_ts_ns: i64, curr_ts_ns: i64) -> bool {
        let (prev_date, prev_h, prev_m, _) = self.timestamp_to_local(prev_ts_ns);
        let (curr_date, curr_h, curr_m, _) = self.timestamp_to_local(curr_ts_ns);

        // New day
        if curr_date != prev_date {
            let curr_minutes = curr_h * 60 + curr_m;
            return curr_minutes >= self.config.market_open_minutes();
        }

        // Same day, but crossed market open
        let prev_minutes = prev_h * 60 + prev_m;
        let curr_minutes = curr_h * 60 + curr_m;

        prev_minutes < self.config.market_open_minutes()
            && curr_minutes >= self.config.market_open_minutes()
    }

    /// Check if this bar ends the session.
    pub fn is_session_end(&self, curr_ts_ns: i64, next_ts_ns: Option<i64>) -> bool {
        let (curr_date, curr_h, curr_m, _) = self.timestamp_to_local(curr_ts_ns);
        let curr_minutes = curr_h * 60 + curr_m;

        // At or past market close
        if curr_minutes >= self.config.market_close_minutes() {
            return true;
        }

        // Check if next bar is in a new session
        if let Some(next_ts) = next_ts_ns {
            let (next_date, _, _, _) = self.timestamp_to_local(next_ts);
            if next_date != curr_date {
                return true;
            }
        }

        false
    }

    /// Update session state for a new bar.
    ///
    /// Returns tuple of (is_new_session, is_squareoff_time, is_session_end).
    // A bar is 6 values plus its neighbours; bundling them into a struct at
    // this call rate would allocate per bar for no reader benefit.
    #[allow(clippy::too_many_arguments)]
    pub fn update(
        &mut self,
        idx: usize,
        timestamp_ns: i64,
        open: f64,
        high: f64,
        low: f64,
        _close: f64,
        prev_timestamp_ns: Option<i64>,
        next_timestamp_ns: Option<i64>,
    ) -> (bool, bool, bool) {
        let (date, hours, minutes, _) = self.timestamp_to_local(timestamp_ns);
        let time_minutes = hours * 60 + minutes;

        // Check for new session
        let is_new_session = if self.current_session_date != date {
            // New date - check if within trading hours
            if time_minutes >= self.config.market_open_minutes()
                && time_minutes < self.config.market_close_minutes()
            {
                self.reset_session(idx, date, open);
                true
            } else {
                false
            }
        } else if let Some(prev_ts) = prev_timestamp_ns {
            if self.is_session_start(prev_ts, timestamp_ns) {
                self.reset_session(idx, date, open);
                true
            } else {
                false
            }
        } else {
            // First bar - start session if within hours
            if time_minutes >= self.config.market_open_minutes()
                && time_minutes < self.config.market_close_minutes()
            {
                self.reset_session(idx, date, open);
                true
            } else {
                false
            }
        };

        // Update session high/low
        if self.in_session {
            if high > self.session_high {
                self.session_high = high;
            }
            if low < self.session_low {
                self.session_low = low;
            }
        }

        // Check squareoff time
        let is_squareoff =
            if time_minutes >= self.config.squareoff_minutes() && !self.squareoff_triggered {
                self.squareoff_triggered = true;
                self.in_session
            } else {
                false
            };

        // Check session end
        let is_session_end = self.is_session_end(timestamp_ns, next_timestamp_ns);
        if is_session_end {
            self.in_session = false;
        }

        (is_new_session, is_squareoff, is_session_end)
    }

    /// Reset session state for a new trading day.
    fn reset_session(&mut self, idx: usize, date: i64, open_price: f64) {
        self.current_session_date = date;
        self.session_start_idx = idx;
        self.session_open = open_price;
        self.session_high = open_price;
        self.session_low = open_price;
        self.in_session = true;
        self.squareoff_triggered = false;
    }

    /// Get current session high.
    pub fn session_high(&self) -> f64 {
        self.session_high
    }

    /// Get current session low.
    pub fn session_low(&self) -> f64 {
        self.session_low
    }

    /// Get current session open.
    pub fn session_open(&self) -> f64 {
        self.session_open
    }

    /// Get session start index.
    pub fn session_start_idx(&self) -> usize {
        self.session_start_idx
    }

    /// Check if currently in a trading session.
    pub fn in_session(&self) -> bool {
        self.in_session
    }

    /// Get opening range (high - low) for the session.
    pub fn opening_range(&self) -> f64 {
        self.session_high - self.session_low
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_timestamp(year: i32, month: u32, day: u32, hour: u32, minute: u32) -> i64 {
        // Simplified: calculate seconds from 1970-01-01 and convert to nanoseconds
        // This is approximate for testing
        let days_since_epoch = (year - 1970) as i64 * 365 + (month - 1) as i64 * 30 + day as i64;
        let seconds = days_since_epoch * 86400 + hour as i64 * 3600 + minute as i64 * 60;
        // Subtract IST offset to get UTC
        let utc_seconds = seconds - (5 * 3600 + 30 * 60);
        utc_seconds * 1_000_000_000
    }

    #[test]
    fn test_session_config_defaults() {
        let config = SessionConfig::default();
        assert_eq!(config.market_open_hour, 9);
        assert_eq!(config.market_open_minute, 15);
        assert_eq!(config.market_close_hour, 15);
        assert_eq!(config.market_close_minute, 30);
        assert_eq!(config.squareoff_minutes_before_close, 5);
    }

    #[test]
    fn test_squareoff_minutes() {
        let config = SessionConfig::default();
        // 15:30 - 5 minutes = 15:25 = 925 minutes
        assert_eq!(config.squareoff_minutes(), 925);
    }

    #[test]
    fn test_mcx_session() {
        let config = SessionConfig::mcx_commodity();
        assert_eq!(config.market_open_hour, 9);
        assert_eq!(config.market_close_hour, 23);
        assert_eq!(config.market_close_minute, 30);
    }

    #[test]
    fn test_session_tracker_new_session() {
        let config = SessionConfig::default();
        let mut tracker = SessionTracker::new(config);

        // Simulate market open at 9:15 IST
        let ts = make_timestamp(2024, 1, 15, 9, 15);
        let (is_new, _, _) = tracker.update(0, ts, 100.0, 101.0, 99.0, 100.5, None, None);

        assert!(is_new);
        assert!(tracker.in_session());
        assert_eq!(tracker.session_open(), 100.0);
    }

    #[test]
    fn test_session_high_low() {
        let config = SessionConfig::default();
        let mut tracker = SessionTracker::new(config);

        // First bar
        let ts1 = make_timestamp(2024, 1, 15, 9, 15);
        tracker.update(0, ts1, 100.0, 105.0, 95.0, 102.0, None, None);

        // Second bar
        let ts2 = make_timestamp(2024, 1, 15, 9, 30);
        tracker.update(1, ts2, 102.0, 110.0, 100.0, 108.0, Some(ts1), None);

        assert_eq!(tracker.session_high(), 110.0);
        assert_eq!(tracker.session_low(), 95.0);
    }
}

/// Bars at which an intraday position must be force-closed for squareoff.
///
/// **In plain words: marks the last bar of each trading day at or after the
/// squareoff time, so the engine can flatten before the market closes.**
///
/// `squareoff_time_minutes` is a local time-of-day in minutes from midnight
/// (925 = 15:25); `tz_offset_ns` shifts UTC timestamps into that local day.
/// Returns one flag per timestamp: `true` on the FIRST bar at or after the
/// squareoff time within each local day.
///
/// Only the first qualifying bar per day is marked. Marking every late bar
/// would be harmless for a single position but wrong as a general signal: a
/// strategy that re-enters after squareoff would be flattened on every
/// subsequent bar, which is not what a broker's square-off does.
///
/// Timestamps are assumed non-decreasing, as everywhere else in the engine.
pub fn squareoff_flags(
    timestamps_ns: &[i64],
    squareoff_time_minutes: u32,
    tz_offset_ns: i64,
) -> Vec<bool> {
    let mut flags = vec![false; timestamps_ns.len()];
    let mut last_marked_day: Option<i64> = None;

    for (i, &ts) in timestamps_ns.iter().enumerate() {
        // Floor-divide so pre-epoch timestamps land in the right local day
        // rather than truncating toward zero.
        let local_s = (ts + tz_offset_ns).div_euclid(1_000_000_000);
        let day = local_s.div_euclid(86_400);
        let minute_of_day = (local_s.rem_euclid(86_400) / 60) as u32;

        if minute_of_day >= squareoff_time_minutes && last_marked_day != Some(day) {
            flags[i] = true;
            last_marked_day = Some(day);
        }
    }

    flags
}

#[cfg(test)]
mod squareoff_tests {
    use super::*;

    /// IST is UTC+5:30.
    const IST: i64 = (5 * 3600 + 30 * 60) * 1_000_000_000;
    /// 15:25 local, NSE's squareoff.
    const AT_1525: u32 = 15 * 60 + 25;

    /// Build a UTC nanosecond timestamp for an IST wall-clock time.
    fn ist(day: i64, hour: i64, minute: i64) -> i64 {
        (day * 86_400 + hour * 3600 + minute * 60) * 1_000_000_000 - IST
    }

    #[test]
    fn marks_the_first_bar_at_or_after_squareoff() {
        let ts = vec![
            ist(0, 9, 15),
            ist(0, 15, 24),
            ist(0, 15, 25), // <- exactly at squareoff
            ist(0, 15, 26),
        ];
        assert_eq!(squareoff_flags(&ts, AT_1525, IST), vec![false, false, true, false]);
    }

    #[test]
    fn marks_once_per_day_across_sessions() {
        let ts = vec![ist(0, 9, 15), ist(0, 15, 29), ist(1, 9, 15), ist(1, 15, 29), ist(2, 9, 15)];
        // Each day's late bar is marked; the fresh morning bars are not.
        assert_eq!(squareoff_flags(&ts, AT_1525, IST), vec![false, true, false, true, false]);
    }

    /// The defect this whole feature exists to fix, stated as a test.
    ///
    /// A session that ends before squareoff (a half day, or a series that
    /// simply stops early) must NOT be marked -- there is no bar at which the
    /// close could have happened, and inventing one would report a fill at a
    /// price that never traded.
    #[test]
    fn a_session_ending_before_squareoff_is_not_marked() {
        let ts = vec![ist(0, 9, 15), ist(0, 12, 0), ist(1, 9, 15)];
        assert_eq!(squareoff_flags(&ts, AT_1525, IST), vec![false, false, false]);
    }

    #[test]
    fn a_bar_after_midnight_utc_stays_in_its_ist_day() {
        // 15:25 IST is 09:55 UTC, but a 23:00 IST bar is 17:30 UTC the same
        // IST day. Without the offset this would split one session in two.
        let ts = vec![ist(0, 15, 25), ist(0, 23, 0)];
        assert_eq!(squareoff_flags(&ts, AT_1525, IST), vec![true, false]);
    }

    #[test]
    fn utc_market_needs_no_offset() {
        let ts = vec![ist(0, 9, 0), ist(0, 16, 0)];
        let flags = squareoff_flags(&ts, 16 * 60, 0);
        assert_eq!(flags.len(), 2);
    }

    #[test]
    fn empty_input_is_empty_output() {
        assert!(squareoff_flags(&[], AT_1525, IST).is_empty());
    }
}
