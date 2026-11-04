//! Annualization factors derived from bar spacing.
//!
//! Through 0.4.1 these were hardcoded and inconsistent: the single-instrument
//! path annualized at 365, the basket/pairs/options/multi paths at 252, and
//! Calmar derived years from bar *count* over 365.25. The same daily return
//! stream therefore produced Sharpe values differing by a factor of ~1.204
//! depending on which runner was called, and Calmar was meaningless on
//! intraday data (11k 1-minute bars read as ~31 "years").
//!
//! This module derives the factor from actual timestamp spacing instead.

/// Nanoseconds in a 365-day calendar year.
const NANOS_PER_YEAR: f64 = 365.0 * 24.0 * 60.0 * 60.0 * 1e9;

/// Legacy annualization constant used by the single-instrument path.
pub const LEGACY_PERIODS_SINGLE: f64 = 365.0;

/// Legacy annualization constant used by the multi-instrument paths.
pub const LEGACY_PERIODS_STRATEGIES: f64 = 252.0;

/// Legacy day count used for Calmar's CAGR.
pub const LEGACY_CALMAR_DAYS: f64 = 365.25;

/// Trading days per year, used when intraday spacing implies a session-based
/// series rather than a continuous one.
const TRADING_DAYS: f64 = 252.0;

/// How an intraday series maps onto trading time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SessionSpec {
    /// Session-based market: `minutes` of trading per session, 252 sessions/year.
    Session { minutes: f64 },
    /// Continuously traded market (24x7), named as such: annualize on calendar time.
    Continuous,
    /// No session named: the trading day is what the data shows. Periods per
    /// year are the median number of observations per local day (the day cut at
    /// `tz_offset_ns`) times 252 sessions. Nights and weekends add no periods.
    Inferred { tz_offset_ns: i64 },
}

/// No session is assumed: the data says how many observations a trading day has.
impl Default for SessionSpec {
    fn default() -> Self {
        SessionSpec::Inferred { tz_offset_ns: 0 }
    }
}

/// Median number of observations per local day over the days that have any.
fn median_observations_per_day(timestamps: &[i64], tz_offset_ns: i64) -> Option<f64> {
    const DAY_NANOS: i64 = 24 * 60 * 60 * 1_000_000_000;
    let mut counts: Vec<u32> = Vec::new();
    let mut current: Option<i64> = None;
    for &t in timestamps {
        let day = (t + tz_offset_ns).div_euclid(DAY_NANOS);
        if current == Some(day) {
            if let Some(last) = counts.last_mut() {
                *last += 1;
            }
        } else {
            current = Some(day);
            counts.push(1);
        }
    }
    if counts.is_empty() {
        return None;
    }
    counts.sort_unstable();
    let mid = counts.len() / 2;
    Some(if counts.len().is_multiple_of(2) {
        (f64::from(counts[mid - 1]) + f64::from(counts[mid])) / 2.0
    } else {
        f64::from(counts[mid])
    })
}

/// Median spacing between consecutive timestamps, in nanoseconds.
///
/// Median rather than mean so that overnight and weekend gaps in an intraday
/// series do not distort the estimate.
pub fn median_spacing_nanos(timestamps: &[i64]) -> Option<f64> {
    if timestamps.len() < 2 {
        return None;
    }

    let mut deltas: Vec<i64> =
        timestamps.windows(2).map(|w| w[1] - w[0]).filter(|&d| d > 0).collect();

    if deltas.is_empty() {
        return None;
    }

    deltas.sort_unstable();
    let mid = deltas.len() / 2;
    let median = if deltas.len().is_multiple_of(2) {
        (deltas[mid - 1] as f64 + deltas[mid] as f64) / 2.0
    } else {
        deltas[mid] as f64
    };

    Some(median)
}

/// Infer periods per year from bar timestamps, assuming no session: the data's
/// own trading day (see [`SessionSpec::Inferred`]).
pub fn infer_periods_per_year(timestamps: &[i64]) -> Option<f64> {
    infer_periods_per_year_with_session(timestamps, SessionSpec::default())
}

/// The largest share of a daily series' bars two adjacent weekday classes may hold for the series
/// to count as leaving weekends out. A session market with weekday sessions still has a handful
/// of special sessions on a weekend (a Budget-day Saturday, a Muhurat Sunday), which must not
/// turn its trading-day count into calendar time; a seven-day series puts 2/7 of its bars there.
const WEEKEND_SHARE: f64 = 0.05;

/// Whether a daily series leaves out weekends: some two adjacent days of the week together carry
/// at most [`WEEKEND_SHARE`] of its bars. Read from the day numbers alone, so it needs no time
/// zone (a constant shift of the day cut moves every bar by the same residue). Fewer than six bars
/// cannot show a weekend.
fn skips_weekends(timestamps: &[i64]) -> bool {
    const DAY: i64 = 24 * 60 * 60 * 1_000_000_000;
    if timestamps.len() < 6 {
        return false;
    }
    let mut per_day = [0usize; 7];
    for &t in timestamps {
        per_day[t.div_euclid(DAY).rem_euclid(7) as usize] += 1;
    }
    let quiet = (0..7).map(|d| per_day[d] + per_day[(d + 1) % 7]).min().unwrap_or(0);
    // A short series is allowed one special session: five percent of under twenty bars is less
    // than one, and a seven-day series of seven bars or more always puts two bars on a weekend.
    let allowed = (WEEKEND_SHARE * timestamps.len() as f64).max(if timestamps.len() >= 7 {
        1.0
    } else {
        0.0
    });
    (quiet as f64) <= allowed
}

/// Infer periods per year from bar timestamps and a session specification.
///
/// Daily bars of a session market (no weekend bars) annualize on 252 trading days; coarser
/// spacing, a market named continuous and a seven-day series annualize on calendar time. Intraday spacing
/// annualizes on trading sessions, since an intraday series only accrues
/// returns during market hours -- treating a 1-minute NSE bar as 1/525600 of a
/// calendar year would overstate the period count by roughly 4x.
///
/// [`SessionSpec::Continuous`] annualizes intraday bars on calendar time, which
/// is correct for 24x7 markets.
pub fn infer_periods_per_year_with_session(
    timestamps: &[i64],
    session: SessionSpec,
) -> Option<f64> {
    let spacing = median_spacing_nanos(timestamps)?;
    if spacing <= 0.0 {
        return None;
    }

    // A day or more between bars. Daily bars of a session market (no weekend bar in the
    // series) accrue returns on trading days, as intraday bars do: 252 a year. Coarser bars,
    // a market named continuous (`session_minutes=0`) and a seven-day series count calendar time.
    const DAY_NANOS: f64 = 24.0 * 60.0 * 60.0 * 1e9;
    if spacing >= DAY_NANOS {
        let calendar_market = match session {
            SessionSpec::Continuous => true,
            SessionSpec::Session { minutes } => minutes <= 0.0,
            SessionSpec::Inferred { .. } => false,
        };
        if spacing < 1.5 * DAY_NANOS && !calendar_market && skips_weekends(timestamps) {
            return Some(TRADING_DAYS * DAY_NANOS / spacing);
        }
        return Some(NANOS_PER_YEAR / spacing);
    }

    match session {
        // Intraday on a session market: bars per session times sessions per year.
        SessionSpec::Session { minutes } if minutes > 0.0 => {
            let bars_per_session = (minutes * 60.0 * 1e9 / spacing).max(1.0);
            Some(bars_per_session * TRADING_DAYS)
        }
        // The data's own trading day: observations per day times sessions per year.
        SessionSpec::Inferred { tz_offset_ns } => {
            median_observations_per_day(timestamps, tz_offset_ns).map(|n| n * TRADING_DAYS)
        }
        // Continuous market, named: every bar counts against calendar time.
        SessionSpec::Session { .. } | SessionSpec::Continuous => Some(NANOS_PER_YEAR / spacing),
    }
}

/// Elapsed years between the first and last timestamp.
///
/// Used for CAGR so that Calmar reflects real time rather than bar count.
pub fn elapsed_years(timestamps: &[i64]) -> Option<f64> {
    if timestamps.len() < 2 {
        return None;
    }
    let span = (*timestamps.last()? - *timestamps.first()?) as f64;
    if span <= 0.0 {
        return None;
    }
    Some(span / NANOS_PER_YEAR)
}

/// Resolve the annualization factor for a run.
///
/// Precedence: explicit config value, then inference from timestamps, then the
/// supplied legacy fallback for series too short to infer from.
pub fn resolve_periods_per_year(explicit: Option<f64>, timestamps: &[i64], fallback: f64) -> f64 {
    resolve_periods_per_year_with_session(explicit, timestamps, SessionSpec::default(), fallback)
}

/// Resolve the annualization factor for a run, honoring a session spec.
pub fn resolve_periods_per_year_with_session(
    explicit: Option<f64>,
    timestamps: &[i64],
    session: SessionSpec,
    fallback: f64,
) -> f64 {
    explicit
        .filter(|v| *v > 0.0)
        .or_else(|| infer_periods_per_year_with_session(timestamps, session))
        .unwrap_or(fallback)
}

#[cfg(test)]
mod tests {
    use super::*;

    const DAY: i64 = 86_400_000_000_000;
    const MIN: i64 = 60_000_000_000;

    #[test]
    fn daily_bars_annualize_at_365() {
        let ts: Vec<i64> = (0..100).map(|i| i * DAY).collect();
        let ppy = infer_periods_per_year(&ts).unwrap();
        assert!((ppy - 365.0).abs() < 0.01, "expected ~365, got {ppy}");
    }

    #[test]
    fn weekly_bars_annualize_at_52() {
        let ts: Vec<i64> = (0..100).map(|i| i * 7 * DAY).collect();
        let ppy = infer_periods_per_year(&ts).unwrap();
        assert!((ppy - 52.14).abs() < 0.1, "expected ~52, got {ppy}");
    }

    #[test]
    fn one_minute_bars_use_session_count() {
        // NSE: 375 bars/session * 252 sessions = 94_500.
        let ts: Vec<i64> = (0..1000).map(|i| i * MIN).collect();
        let ppy = infer_periods_per_year_with_session(&ts, SessionSpec::Session { minutes: 375.0 })
            .unwrap();
        assert!((ppy - 94_500.0).abs() < 100.0, "expected ~94500, got {ppy}");
    }

    #[test]
    fn with_no_session_given_the_trading_day_is_what_the_data_shows() {
        // Five days of 375 one-minute bars, with nights and a weekend between them:
        // 375 observations a day times 252 days, as a 375-minute session would give.
        let mut ts = Vec::new();
        for day in [0, 1, 2, 3, 6] {
            ts.extend((0..375).map(|i| day * DAY + 9 * 60 * MIN + i * MIN));
        }
        let ppy = infer_periods_per_year(&ts).unwrap();
        assert_eq!(ppy, 375.0 * 252.0);
        assert_eq!(SessionSpec::default(), SessionSpec::Inferred { tz_offset_ns: 0 });
        // The same bars named as an NSE session, or as the data's own day, agree.
        let named =
            infer_periods_per_year_with_session(&ts, SessionSpec::Session { minutes: 375.0 })
                .unwrap();
        assert_eq!(named, ppy);
        // Only a market named continuous counts nights and weekends.
        let continuous = infer_periods_per_year_with_session(&ts, SessionSpec::Continuous).unwrap();
        assert_eq!(continuous, 525_600.0);
    }

    #[test]
    fn mcx_session_yields_more_periods_than_nse() {
        // MCX runs 09:00-23:30 = 870 minutes, vs NSE's 375.
        let ts: Vec<i64> = (0..1000).map(|i| i * MIN).collect();
        let nse = infer_periods_per_year_with_session(&ts, SessionSpec::Session { minutes: 375.0 })
            .unwrap();
        let mcx = infer_periods_per_year_with_session(&ts, SessionSpec::Session { minutes: 870.0 })
            .unwrap();

        assert!((mcx - 870.0 * 252.0).abs() < 100.0, "expected ~219240, got {mcx}");
        // Sharpe scales with sqrt(periods), so an NSE assumption on MCX data
        // understates it by this factor.
        let sharpe_ratio_error = (mcx / nse).sqrt();
        assert!(
            (sharpe_ratio_error - 1.523).abs() < 0.01,
            "expected ~1.52x understatement, got {sharpe_ratio_error}"
        );
    }

    #[test]
    fn continuous_market_annualizes_on_calendar_time() {
        // 24x7: a 1-minute bar is 1/525600 of a year.
        let ts: Vec<i64> = (0..1000).map(|i| i * MIN).collect();
        let ppy = infer_periods_per_year_with_session(&ts, SessionSpec::Continuous).unwrap();
        assert!((ppy - 525_600.0).abs() < 100.0, "expected ~525600, got {ppy}");
    }

    #[test]
    fn weekday_daily_bars_count_trading_days() {
        // 60 weekday bars at 15:30 IST (10:00 UTC): 252 whatever the session says, unless the
        // market is named continuous.
        let mut ts = Vec::new();
        let mut day = 20_600;
        while ts.len() < 60 {
            if day % 7 != 2 && day % 7 != 3 {
                // epoch day 0 is a Thursday: 2 and 3 are the weekend
                ts.push(day * DAY + 10 * 3600 * 1_000_000_000);
            }
            day += 1;
        }
        for spec in [
            SessionSpec::default(),
            SessionSpec::Session { minutes: 375.0 },
            SessionSpec::Inferred { tz_offset_ns: 19_800 * 1_000_000_000 },
        ] {
            let ppy = infer_periods_per_year_with_session(&ts, spec).unwrap();
            assert!((ppy - 252.0).abs() < 1e-9, "expected 252 for {spec:?}, got {ppy}");
        }
        for spec in [SessionSpec::Session { minutes: 0.0 }, SessionSpec::Continuous] {
            let ppy = infer_periods_per_year_with_session(&ts, spec).unwrap();
            assert!((ppy - 365.0).abs() < 1e-9, "calendar for {spec:?}, got {ppy}");
        }
        // Weekly bars stay on calendar time.
        let weekly: Vec<i64> = (0..60).map(|i| i * 7 * DAY).collect();
        let ppy = infer_periods_per_year(&weekly).unwrap();
        assert!((ppy - 52.14).abs() < 0.1);
    }

    #[test]
    fn a_handful_of_weekend_sessions_do_not_make_a_session_market_a_calendar_one() {
        // A year of weekday bars plus a Budget-day Saturday and a Muhurat Sunday: 252 trading days.
        let mut ts = Vec::new();
        let mut day = 20_000;
        while ts.len() < 250 {
            let weekend = day % 7 == 2 || day % 7 == 3;
            if !weekend || ts.len() == 40 || ts.len() == 200 {
                ts.push(day * DAY + 10 * 3600 * 1_000_000_000);
            }
            day += 1;
        }
        let ppy = infer_periods_per_year(&ts).unwrap();
        assert!((ppy - 252.0).abs() < 1e-9, "two special sessions: expected 252, got {ppy}");
        // A series that trades the weekend regularly counts calendar time.
        let every: Vec<i64> = (0..250).map(|i| (20_000 + i) * DAY).collect();
        let ppy = infer_periods_per_year(&every).unwrap();
        assert!((ppy - 365.0).abs() < 1e-9, "seven-day series: expected 365, got {ppy}");
    }

    #[test]
    fn session_spec_does_not_affect_daily_bars() {
        // Daily-or-coarser is calendar-based regardless of session length.
        let ts: Vec<i64> = (0..100).map(|i| i * DAY).collect();
        for spec in [
            SessionSpec::Session { minutes: 375.0 },
            SessionSpec::Session { minutes: 870.0 },
            SessionSpec::Continuous,
        ] {
            let ppy = infer_periods_per_year_with_session(&ts, spec).unwrap();
            assert!((ppy - 365.0).abs() < 0.01, "expected 365 for {spec:?}, got {ppy}");
        }
    }

    #[test]
    fn median_ignores_overnight_gaps() {
        // Two sessions of 1-minute bars separated by an overnight gap.
        let mut ts: Vec<i64> = (0..100).map(|i| i * MIN).collect();
        let next_day = DAY;
        ts.extend((0..100).map(|i| next_day + i * MIN));
        let spacing = median_spacing_nanos(&ts).unwrap();
        assert_eq!(spacing, MIN as f64, "gap should not move the median");
    }

    #[test]
    fn elapsed_years_tracks_wall_clock_not_bar_count() {
        // 11_250 one-minute bars spanning 30 calendar days.
        let ts: Vec<i64> = (0..11_250).map(|i| i * MIN).collect();
        let years = elapsed_years(&ts).unwrap();
        // ~7.8 days of continuous minutes, not 11250/365.25 = 30.8 "years".
        assert!(years < 0.05, "expected a small fraction of a year, got {years}");
    }

    #[test]
    fn explicit_value_wins_over_inference() {
        let ts: Vec<i64> = (0..100).map(|i| i * DAY).collect();
        assert_eq!(resolve_periods_per_year(Some(12.0), &ts, 252.0), 12.0);
    }

    #[test]
    fn falls_back_when_too_short_to_infer() {
        assert_eq!(resolve_periods_per_year(None, &[], 252.0), 252.0);
        assert_eq!(resolve_periods_per_year(None, &[1], 365.0), 365.0);
    }
}
