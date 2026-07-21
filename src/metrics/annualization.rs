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

/// Default session length in minutes: NSE equity, 09:15-15:30.
///
/// Only a default. MCX runs 09:00-23:30 (870 minutes), so a hardcoded NSE
/// session would understate MCX intraday Sharpe by roughly sqrt(870/375) ~ 1.5x.
/// Callers pass the real session length via [`SessionSpec`].
pub const DEFAULT_SESSION_MINUTES: f64 = 375.0;

/// How an intraday series maps onto trading time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SessionSpec {
    /// Session-based market: `minutes` of trading per session, 252 sessions/year.
    Session { minutes: f64 },
    /// Continuously traded market (24x7): annualize on calendar time.
    Continuous,
}

impl Default for SessionSpec {
    fn default() -> Self {
        SessionSpec::Session { minutes: DEFAULT_SESSION_MINUTES }
    }
}

impl SessionSpec {
    /// Session length in nanoseconds, or `None` when continuous.
    fn session_nanos(&self) -> Option<f64> {
        match self {
            SessionSpec::Session { minutes } if *minutes > 0.0 => Some(minutes * 60.0 * 1e9),
            _ => None,
        }
    }
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

/// Infer periods per year from bar timestamps, assuming the default NSE session.
pub fn infer_periods_per_year(timestamps: &[i64]) -> Option<f64> {
    infer_periods_per_year_with_session(timestamps, SessionSpec::default())
}

/// Infer periods per year from bar timestamps and a session specification.
///
/// Daily-or-coarser spacing annualizes on calendar time. Intraday spacing
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

    // A day or more between bars: calendar-based.
    const DAY_NANOS: f64 = 24.0 * 60.0 * 60.0 * 1e9;
    if spacing >= DAY_NANOS {
        return Some(NANOS_PER_YEAR / spacing);
    }

    match session.session_nanos() {
        // Intraday on a session market: bars per session times sessions per year.
        Some(session_nanos) => {
            let bars_per_session = (session_nanos / spacing).max(1.0);
            Some(bars_per_session * TRADING_DAYS)
        }
        // Continuous market: every bar counts against calendar time.
        None => Some(NANOS_PER_YEAR / spacing),
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
        let ppy = infer_periods_per_year(&ts).unwrap();
        assert!((ppy - 94_500.0).abs() < 100.0, "expected ~94500, got {ppy}");
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
