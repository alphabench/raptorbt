//! Event study by the market model: abnormal returns around dated events, cumulative (CAR) per
//! event and averaged over events (CAAR), each with a t-statistic.
//!
//! Dates are day numbers (days since 1970-01-01, any strictly increasing integers will do);
//! [`days_from_civil`] and [`iso_date`] convert. A fit with no residual noise (the estimation
//! returns lie on the market line to rounding) has an undefined t (dividing by a sigma of
//! 1e-18 would give t of about 5e15). Here `t` is NaN and `degenerate_fit` is set, the CAR and
//! the CAAR keep their values.

use crate::math::fsum::fsum;
use crate::math::special::t_two_sided_p;

use super::error::StatsError;
use super::hac::{finite_series, mean, peak_to_peak};

pub const MAX_EVENTS: usize = 100;
pub const MIN_ESTIMATION: usize = 60;
pub const MAX_WINDOW: usize = 20;
pub const MIN_CROSS_EVENTS: usize = 2;
/// A fit whose residual sigma is below this fraction of the estimation returns' own standard
/// deviation is exact to rounding: its t is undefined.
pub const DEGENERATE_SIGMA_RATIO: f64 = 1e-10;

pub const REASON_AFTER_LAST: &str = "after the last bar";
pub const REASON_NO_WINDOW: &str = "no full estimation and event window";
pub const REASON_FLAT_MARKET: &str = "market constant in estimation";

#[derive(Debug, Clone, PartialEq)]
pub struct EventRow {
    /// The event date as given.
    pub event: i64,
    /// The bar the event falls on (the next bar when the date is off the calendar).
    pub event_bar: i64,
    pub alpha: f64,
    pub beta: f64,
    pub residual_sigma: f64,
    pub abnormal_returns: Vec<f64>,
    pub car: f64,
    /// NaN when `degenerate_fit`.
    pub t: f64,
    pub degenerate_fit: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Skipped {
    pub event: i64,
    pub reason: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EventWindow {
    pub from: i64,
    pub to: i64,
    pub periods: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Estimation {
    pub periods: usize,
    pub gap: usize,
}

/// The result of [`event_study`], full precision. The cross-sectional fields are `None` with
/// fewer than two events or no spread among the CARs.
#[derive(Debug, Clone, PartialEq)]
pub struct EventStudy {
    pub n_bars: usize,
    pub n_events: usize,
    pub event_window: EventWindow,
    pub estimation: Estimation,
    pub caar: f64,
    pub events: Vec<EventRow>,
    pub skipped: Vec<Skipped>,
    pub caar_t: Option<f64>,
    pub caar_p_value: Option<f64>,
    pub positive_share: Option<f64>,
}

/// Days since 1970-01-01 of a proleptic Gregorian date (Hinnant's algorithm).
pub fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let y = if month <= 2 { year - 1 } else { year };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = (month + 9) % 12;
    let doy = (153 * mp + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// The inverse of [`days_from_civil`]: (year, month, day).
pub fn civil_from_days(days: i64) -> (i64, i64, i64) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    (yoe + era * 400 + i64::from(month <= 2), month, day)
}

/// `YYYY-MM-DD` of a day number.
pub fn iso_date(days: i64) -> String {
    let (y, m, d) = civil_from_days(days);
    format!("{y:04}-{m:02}-{d:02}")
}

/// The windows of an event study, in periods. Typical values are 120 and 5 for the estimation
/// and the gap, 1 and 1 for the window; the caller passes them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EventParams {
    /// Length of the estimation window.
    pub estimation: usize,
    /// Periods between the estimation window and the event window.
    pub gap: usize,
    /// Periods before the event bar in the event window.
    pub pre: usize,
    /// Periods after the event bar in the event window.
    pub post: usize,
}

/// `dates`, `asset` and `market` are aligned per-period returns, oldest first. For each event
/// the market model (OLS) is fitted on the `estimation` periods ending `gap` periods before the
/// event window `[-pre, +post]`; an event off the calendar moves to the next period on it. Events
/// without a full estimation and event window are listed as skipped.
pub fn event_study(
    dates: &[i64],
    asset: &[f64],
    market: &[f64],
    events: &[i64],
    params: EventParams,
) -> Result<EventStudy, StatsError> {
    let EventParams { estimation, gap, pre, post } = params;
    if !(1..=MAX_EVENTS).contains(&events.len()) {
        return Err(StatsError::Invalid(format!("give 1 to {MAX_EVENTS} event dates")));
    }
    if pre > MAX_WINDOW || post > MAX_WINDOW {
        return Err(StatsError::Invalid(format!("pre and post are 0 to {MAX_WINDOW} periods")));
    }
    if estimation < MIN_ESTIMATION {
        return Err(StatsError::Invalid(format!(
            "estimation is at least {MIN_ESTIMATION} periods, gap not negative"
        )));
    }
    let n = dates.len();
    let r = finite_series(asset, "asset", 2)?;
    let mk = finite_series(market, "market", 2)?;
    if !(r.len() == mk.len() && mk.len() == n) {
        return Err(StatsError::Invalid("dates, asset and market differ in length".into()));
    }
    if dates.windows(2).any(|w| w[0] >= w[1]) {
        return Err(StatsError::Invalid("dates are not strictly increasing".into()));
    }
    let (n_i, pre_i, post_i) = (n as i64, pre as i64, post as i64);
    let length = pre + post + 1;

    let mut cars = Vec::new();
    let mut rows = Vec::new();
    let mut skipped = Vec::new();
    for &ev in events {
        let idx = dates.partition_point(|d| *d < ev);
        if idx >= n {
            skipped.push(Skipped { event: ev, reason: REASON_AFTER_LAST });
            continue;
        }
        let lo = idx as i64 - pre_i;
        let hi = idx as i64 + post_i;
        let est_hi = lo - gap as i64;
        let est_lo = est_hi - estimation as i64;
        if est_lo < 0 || hi >= n_i {
            skipped.push(Skipped { event: ev, reason: REASON_NO_WINDOW });
            continue;
        }
        let (lo, hi, est_lo, est_hi) = (lo as usize, hi as usize, est_lo as usize, est_hi as usize);
        let (x, y) = (&mk[est_lo..est_hi], &r[est_lo..est_hi]);
        if peak_to_peak(x) == 0.0 {
            skipped.push(Skipped { event: ev, reason: REASON_FLAT_MARKET });
            continue;
        }
        let (xm, ym) = (mean(x), mean(y));
        let sxx: f64 = x.iter().map(|v| (v - xm) * (v - xm)).sum();
        let sxy: f64 = x.iter().zip(y).map(|(a, b)| (a - xm) * (b - ym)).sum();
        let beta = sxy / sxx;
        let alpha = ym - beta * xm;
        let resid: Vec<f64> = x.iter().zip(y).map(|(a, b)| b - alpha - beta * a).collect();
        let rm = mean(&resid);
        let m = resid.len() as f64;
        let sigma = (resid.iter().map(|e| (e - rm) * (e - rm)).sum::<f64>() / (m - 2.0)).sqrt();
        let y_sd = (y.iter().map(|v| (v - ym) * (v - ym)).sum::<f64>() / (m - 1.0)).sqrt();
        let ar: Vec<f64> = (lo..=hi).map(|i| r[i] - alpha - beta * mk[i]).collect();
        let car: f64 = ar.iter().sum();
        let degenerate = sigma.is_nan() || sigma <= DEGENERATE_SIGMA_RATIO * y_sd;
        let t = if degenerate { f64::NAN } else { car / (sigma * (length as f64).sqrt()) };
        cars.push(car);
        rows.push(EventRow {
            event: ev,
            event_bar: dates[idx],
            alpha,
            beta,
            residual_sigma: sigma,
            abnormal_returns: ar,
            car,
            t,
            degenerate_fit: degenerate,
        });
    }
    if rows.is_empty() {
        let listed: Vec<String> = skipped
            .iter()
            .map(|s| format!("{{'event': '{}', 'reason': '{}'}}", iso_date(s.event), s.reason))
            .collect();
        return Err(StatsError::Invalid(format!(
            "no event had a full window: [{}]",
            listed.join(", ")
        )));
    }
    let m = cars.len();
    let mf = m as f64;
    let caar = fsum(cars.iter().copied()) / mf;
    let sd0 = (cars.iter().map(|c| (c - caar) * (c - caar)).sum::<f64>() / mf).sqrt();
    let (mut caar_t, mut caar_p, mut share) = (None, None, None);
    if m >= MIN_CROSS_EVENTS && sd0 > 0.0 {
        let sd1 = (cars.iter().map(|c| (c - caar) * (c - caar)).sum::<f64>() / (mf - 1.0)).sqrt();
        let t_x = caar / (sd1 / mf.sqrt());
        caar_t = Some(t_x);
        caar_p = Some(t_two_sided_p(t_x.abs(), mf - 1.0)?);
        share = Some(cars.iter().filter(|c| **c > 0.0).count() as f64 / mf);
    }
    Ok(EventStudy {
        n_bars: n,
        n_events: m,
        event_window: EventWindow { from: -pre_i, to: post_i, periods: length },
        estimation: Estimation { periods: estimation, gap },
        caar,
        events: rows,
        skipped,
        caar_t,
        caar_p_value: caar_p,
        positive_share: share,
    })
}
