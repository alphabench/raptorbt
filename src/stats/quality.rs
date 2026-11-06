//! Is a stretch of bars fit to research on? Pure checks over OHLCV arrays, no I/O.
//!
//! It reports what is wrong and where, with counts and the first examples, and says what it
//! cannot judge: it holds no exchange holiday calendar, so a gap is a stretch with no bars, not
//! a missing session. A suspected unadjusted corporate action is a close-to-close ratio near a
//! whole split factor; it is a flag to check against the corporate actions, not a finding.
//!
//! Times are epoch seconds and are written in IST (UTC+5:30, no daylight saving). Every
//! threshold is a field of [`AuditRules`]; [`AuditRules::default`] holds the usual values, which
//! the caller states rather than inherits. Amounts such as `return_pct`, `close_ratio` and
//! `days` are full precision.
//!
//! Bars that are flat and carry no volume are counted (`flat_zero_volume_bars`) and the series
//! is not `clean` when they are more than `dead_bar_share_max` of the bars.

use super::error::StatsError;
use super::events::civil_from_days;

const DAY_SECONDS: f64 = 86_400.0;
const IST_OFFSET_SECONDS: i64 = 19_800;

/// The thresholds of the audit.
#[derive(Debug, Clone, PartialEq)]
pub struct AuditRules {
    /// A bar whose robust z of return (median / MAD) is at least this is listed.
    pub outlier_z: f64,
    /// MAD to a standard deviation for normal data.
    pub mad_scale: f64,
    /// Examples named per finding.
    pub listed: usize,
    /// A daily stretch with no bar longer than this many days is a gap.
    pub daily_gap_days: f64,
    /// An intraday hole wider than this many bars, inside one session, is a gap.
    pub intraday_gap_bars: f64,
    /// Close-to-close ratios near one of these (or its inverse) are suspected splits.
    pub split_factors: Vec<f64>,
    /// "Near" is within this fraction of the factor.
    pub split_tolerance: f64,
    /// Fewest bars worth auditing.
    pub min_bars: usize,
    /// The series is not clean when more than this share of its bars are flat with no volume. The
    /// default, one half, is a judgment: review it.
    pub dead_bar_share_max: f64,
}

impl Default for AuditRules {
    fn default() -> Self {
        AuditRules {
            outlier_z: 8.0,
            mad_scale: 1.4826,
            listed: 5,
            daily_gap_days: 4.0,
            intraday_gap_bars: 3.0,
            split_factors: vec![1.5, 2.0, 3.0, 4.0, 5.0, 10.0],
            split_tolerance: 0.03,
            min_bars: 5,
            dead_bar_share_max: 0.5,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ImpossibleOhlc {
    pub count: usize,
    pub first: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ReturnOutlier {
    pub at: String,
    pub return_pct: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SplitSuspect {
    pub at: String,
    pub close_ratio: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Gap {
    Daily { from: String, to: String, days: f64 },
    Intraday { from: String, to: String, missing_bars: usize },
}

#[derive(Debug, Clone, PartialEq)]
pub struct Findings {
    pub duplicate_timestamps: usize,
    pub out_of_order: usize,
    pub impossible_ohlc: ImpossibleOhlc,
    pub zero_volume_bars: usize,
    pub flat_bars: usize,
    pub flat_zero_volume_bars: usize,
    pub robust_z_over: f64,
    pub return_outliers: Vec<ReturnOutlier>,
    /// At most `listed`; `clean` counts them all.
    pub possible_unadjusted_splits: Vec<SplitSuspect>,
    pub gaps: Vec<Gap>,
}

/// The result of [`audit_bars`].
#[derive(Debug, Clone, PartialEq)]
pub struct Audit {
    pub bars: usize,
    pub first: String,
    pub last: String,
    pub findings: Findings,
    pub clean: bool,
}

/// `YYYY-MM-DD HH:MM` in IST of epoch seconds (the minute is truncated).
fn at_str(ts: f64) -> String {
    let secs = ts.floor() as i64 + IST_OFFSET_SECONDS;
    let sod = secs.rem_euclid(86_400);
    let (y, m, d) = civil_from_days(secs.div_euclid(86_400));
    format!("{y:04}-{m:02}-{d:02} {:02}:{:02}", sod / 3600, sod % 3600 / 60)
}

/// The IST calendar day of epoch seconds, as a day number.
fn ist_day(ts: f64) -> i64 {
    (ts.floor() as i64 + IST_OFFSET_SECONDS).div_euclid(86_400)
}

fn median(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    }
}

/// Bars as parallel columns, oldest first: `t` in epoch seconds.
#[derive(Debug, Clone, Copy)]
pub struct Bars<'a> {
    pub t: &'a [f64],
    pub open: &'a [f64],
    pub high: &'a [f64],
    pub low: &'a [f64],
    pub close: &'a [f64],
    pub volume: &'a [f64],
}

/// Audits `bars` of `interval_seconds` each against `rules`.
pub fn audit_bars(
    bars: &Bars,
    interval_seconds: u64,
    rules: &AuditRules,
) -> Result<Audit, StatsError> {
    let Bars { t, open: o, high: h, low, close: c, volume: v } = *bars;
    let n = t.len();
    if [o.len(), h.len(), low.len(), c.len(), v.len()].iter().any(|l| *l != n) {
        return Err(StatsError::Invalid("the columns of the bars differ in length".into()));
    }
    if n < rules.min_bars {
        return Err(StatsError::InsufficientObservations {
            what: "bars to audit".into(),
            found: n,
            needed: rules.min_bars,
        });
    }
    if interval_seconds == 0 {
        return Err(StatsError::Invalid("interval_seconds must be positive".into()));
    }
    if t.iter().any(|x| !x.is_finite()) {
        return Err(StatsError::NonFinite { what: "t".into() });
    }
    let interval = interval_seconds as f64;
    let daily = interval >= DAY_SECONDS;
    let dt: Vec<f64> = t.windows(2).map(|w| w[1] - w[0]).collect();
    let listed = rules.listed;

    let bad: Vec<bool> = (0..n)
        .map(|i| {
            let (oi, hi, li, ci) = (o[i], h[i], low[i], c[i]);
            !(oi.is_finite() && hi.is_finite() && li.is_finite() && ci.is_finite())
                || oi.min(hi).min(li).min(ci) <= 0.0
                || hi < oi.max(ci)
                || li > oi.min(ci)
        })
        .collect();
    let impossible = ImpossibleOhlc {
        count: bad.iter().filter(|b| **b).count(),
        first: (0..n).filter(|&i| bad[i]).take(listed).map(|i| at_str(t[i])).collect(),
    };
    let flat = |i: usize| o[i] == h[i] && h[i] == low[i] && low[i] == c[i];
    let flat_bars = (0..n).filter(|&i| flat(i)).count();
    let flat_zero_volume_bars = (0..n).filter(|&i| flat(i) && v[i] == 0.0).count();

    let ok_idx: Vec<usize> = (0..n).filter(|&i| !bad[i]).collect();
    let ret: Vec<f64> = if ok_idx.len() > 1 {
        ok_idx.windows(2).map(|w| c[w[1]].ln() - c[w[0]].ln()).collect()
    } else {
        Vec::new()
    };
    // the bar each return ends on
    let ret_at: Vec<usize> = ok_idx.iter().skip(1).copied().collect();

    let mut outliers = Vec::new();
    if ret.len() >= rules.min_bars {
        let mut sorted = ret.clone();
        sorted.sort_by(f64::total_cmp);
        let med = median(&sorted);
        let mut dev: Vec<f64> = ret.iter().map(|r| (r - med).abs()).collect();
        dev.sort_by(f64::total_cmp);
        let mad = median(&dev) * rules.mad_scale;
        if mad > 0.0 {
            let z: Vec<f64> = ret.iter().map(|r| (r - med) / mad).collect();
            let mut by_size: Vec<usize> = (0..z.len()).collect();
            by_size.sort_by(|a, b| z[*b].abs().total_cmp(&z[*a].abs()).then(a.cmp(b)));
            for &i in by_size.iter().take(listed) {
                if z[i].abs() >= rules.outlier_z {
                    outliers.push(ReturnOutlier {
                        at: at_str(t[ret_at[i]]),
                        return_pct: ret[i].exp_m1() * 100.0,
                    });
                }
            }
        }
    }

    let mut splits = Vec::new();
    if daily {
        for (i, lr) in ret.iter().enumerate() {
            let ratio = lr.exp();
            for f in &rules.split_factors {
                if (ratio - f).abs() / f < rules.split_tolerance
                    || (ratio - 1.0 / f).abs() * f < rules.split_tolerance
                {
                    splits.push(SplitSuspect { at: at_str(t[ret_at[i]]), close_ratio: ratio });
                    break;
                }
            }
        }
    }

    let mut gaps = Vec::new();
    if daily {
        for i in (0..n - 1).filter(|&i| dt[i] > rules.daily_gap_days * DAY_SECONDS).take(listed) {
            gaps.push(Gap::Daily {
                from: at_str(t[i]),
                to: at_str(t[i + 1]),
                days: dt[i] / DAY_SECONDS,
            });
        }
    } else {
        let hole = |i: usize| {
            ist_day(t[i]) == ist_day(t[i + 1]) && dt[i] > rules.intraday_gap_bars * interval
        };
        for i in (0..n - 1).filter(|&i| hole(i)).take(listed) {
            gaps.push(Gap::Intraday {
                from: at_str(t[i]),
                to: at_str(t[i + 1]),
                missing_bars: (dt[i] / interval).floor() as usize - 1,
            });
        }
    }

    let duplicate_timestamps = dt.iter().filter(|d| **d == 0.0).count();
    let out_of_order = dt.iter().filter(|d| **d < 0.0).count();
    let dead_share = flat_zero_volume_bars as f64 / n as f64;
    let clean = !(duplicate_timestamps > 0
        || out_of_order > 0
        || impossible.count > 0
        || !outliers.is_empty()
        || !splits.is_empty()
        || !gaps.is_empty()
        || dead_share > rules.dead_bar_share_max);
    // keep only the first examples of splits once `clean` has counted them all
    splits.truncate(listed);

    Ok(Audit {
        bars: n,
        first: at_str(t[0]),
        last: at_str(t[n - 1]),
        findings: Findings {
            duplicate_timestamps,
            out_of_order,
            impossible_ohlc: impossible,
            zero_volume_bars: v.iter().filter(|x| **x == 0.0).count(),
            flat_bars,
            flat_zero_volume_bars,
            robust_z_over: rules.outlier_z,
            return_outliers: outliers,
            possible_unadjusted_splits: splits,
            gaps,
        },
        clean,
    })
}
