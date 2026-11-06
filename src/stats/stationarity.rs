//! Is a series stationary: the augmented Dickey-Fuller and the KPSS test, read together. They
//! test opposite nulls (ADF: a unit root; KPSS: stationarity), so one alone is weak evidence.
//!
//! Follows the parts of statsmodels it replaces:
//!
//! - `adfuller`: the lag search fits `d x_t` on the lagged level, `0..=maxlag` lagged
//!   differences and the deterministic terms, every fit on the same sample (the rows left after
//!   `maxlag` lags are taken), picks the lag with the smallest AIC (the lowest lag on a tie),
//!   then refits on the sample that lag leaves. The statistic is the t-value of the lagged
//!   level; the p-value is MacKinnon (1994) and the critical values MacKinnon (2010).
//! - `kpss`: level or trend stationarity, the long-run variance from a Bartlett (Newey-West)
//!   kernel, the lag from Hobijn, Franses and Ooms (1998) when asked for `auto`. The p-value is
//!   read off Kwiatkowski et al.'s table by linear interpolation, so it is bounded to
//!   `[0.01, 0.10]`; `p_value_clipped` says that the bound, not the data, set it.
//!
//! Results are plain structs whose field names are the result keys, at full precision:
//! a caller that displays them rounds. The critical values are arrays in the order of [`ADF_CRITICAL_KEYS`]
//! and [`KPSS_CRITICAL_KEYS`].
//!
//! Differences from statsmodels, each on purpose:
//! - the ADF regression centres the series before regressions that carry a constant (the
//!   statistic, the lag and the AIC differences are exactly those of the uncentred regression,
//!   and a series like `1e9 + noise` no longer loses its digits to the level-constant
//!   collinearity);
//! - a regression whose columns are collinear (a series that is an exact line) is a typed
//!   [`StatsError::DegenerateSeries`], where statsmodels returns a minimum-norm answer.

use super::error::StatsError;
use super::tables;
use crate::math::linalg::{qr_pivoted, Mat};
use crate::pricing::norm::norm_cdf;

/// Observations the stationarity tests need.
pub const MIN_OBS: usize = 50;
/// The significance level the tests are read at.
pub const ALPHA: f64 = 0.05;
/// Keys of [`Adf::critical_values`].
pub const ADF_CRITICAL_KEYS: [&str; 3] = ["1%", "5%", "10%"];
/// Keys of [`Kpss::critical_values`].
pub const KPSS_CRITICAL_KEYS: [&str; 4] = ["10%", "5%", "2.5%", "1%"];

/// The deterministic terms of a unit-root regression (MacKinnon's `n`, `c`, `ct`, `ctt`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Regression {
    /// None.
    N,
    /// A constant.
    C,
    /// A constant and a linear trend.
    Ct,
    /// A constant, a linear and a quadratic trend.
    Ctt,
}

impl Regression {
    /// The position in the first axis of the `TAU_*` tables.
    pub fn index(self) -> usize {
        match self {
            Regression::N => 0,
            Regression::C => 1,
            Regression::Ct => 2,
            Regression::Ctt => 3,
        }
    }

    /// How many deterministic columns the regression adds.
    pub fn n_trend(self) -> usize {
        self.index()
    }

    pub fn name(self) -> &'static str {
        tables::REGRESSIONS[self.index()]
    }
}

/// `Horner` with coefficients in rising order, as numpy's `polyval` on the reversed vector.
fn horner_rising(coefs: &[f64], x: f64) -> f64 {
    coefs.iter().rev().fold(0.0, |y, c| y * x + c)
}

/// MacKinnon's (1994) approximate p-value of an ADF or cointegration statistic for `n_series`
/// I(1) series (1 for the ADF test). `None` for `n_series` above 6, where the 1994 tables stop:
/// statsmodels raises `IndexError` there, the 2010 critical values ([`mackinnon_crit`]) still
/// answer up to 12.
pub fn mackinnon_p(stat: f64, regression: Regression, n_series: usize) -> Option<f64> {
    if n_series == 0 || n_series > tables::MACKINNON_1994_MAX_N {
        return None;
    }
    let (r, i) = (regression.index(), n_series - 1);
    if stat > tables::TAU_MAX[r][i] {
        return Some(1.0);
    }
    if stat < tables::TAU_MIN[r][i] {
        return Some(0.0);
    }
    let z = if stat <= tables::TAU_STAR[r][i] {
        horner_rising(&tables::TAU_SMALLP[r][i], stat)
    } else {
        horner_rising(&tables::TAU_LARGEP[r][i], stat)
    };
    Some(norm_cdf(z))
}

/// MacKinnon's (2010) 1%, 5% and 10% critical values at sample size `nobs` (`f64::INFINITY` for
/// the asymptotic values). `None` for `n_series` outside 1 to 12, or above 1 for the
/// no-constant regression, whose surface exists for one series only.
pub fn mackinnon_crit(regression: Regression, n_series: usize, nobs: f64) -> Option<[f64; 3]> {
    if n_series == 0 || n_series > tables::MACKINNON_2010_MAX_N {
        return None;
    }
    let surface: &[[f64; 4]; 3] = match regression {
        Regression::N if n_series == 1 => &tables::TAU_2010_N[0],
        Regression::N => return None,
        Regression::C => &tables::TAU_2010_C[n_series - 1],
        Regression::Ct => &tables::TAU_2010_CT[n_series - 1],
        Regression::Ctt => &tables::TAU_2010_CTT[n_series - 1],
    };
    let mut out = [0.0; 3];
    for (o, beta) in out.iter_mut().zip(surface) {
        *o = if nobs.is_infinite() {
            beta[0]
        } else {
            // beta_inf + beta_1/T + beta_2/T^2 + beta_3/T^3, as numpy's polyval at 1/T
            let x = 1.0 / nobs;
            (((0.0 * x + beta[3]) * x + beta[2]) * x + beta[1]) * x + beta[0]
        };
    }
    Some(out)
}

pub(crate) fn require_finite(x: &[f64], what: &str) -> Result<(), StatsError> {
    if x.iter().all(|v| v.is_finite()) {
        Ok(())
    } else {
        Err(StatsError::NonFinite { what: what.to_string() })
    }
}

pub(crate) fn is_constant(x: &[f64]) -> bool {
    x.iter().all(|v| *v == x[0])
}

/// numpy's pairwise summation (eight accumulators below 128 values, halves above), so a sum
/// lands on the digits `numpy.sum` gives: a mean subtracted from a series whose spread is 1e-11
/// of its level must be the same mean.
pub(crate) fn pairwise_sum(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        return a.iter().fold(0.0, |s, v| s + v);
    }
    if n <= 128 {
        let mut r = [0.0; 8];
        r.copy_from_slice(&a[..8]);
        let mut i = 8;
        while i < n - n % 8 {
            for j in 0..8 {
                r[j] += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for v in &a[i..] {
            res += v;
        }
        return res;
    }
    let mut n2 = n / 2;
    n2 -= n2 % 8;
    pairwise_sum(&a[..n2]) + pairwise_sum(&a[n2..])
}

pub(crate) fn mean(x: &[f64]) -> f64 {
    pairwise_sum(x) / x.len() as f64
}

fn sum_of_squares(x: &[f64]) -> f64 {
    pairwise_sum(&x.iter().map(|v| v * v).collect::<Vec<f64>>())
}

/// A least-squares fit by pivoted QR.
#[derive(Debug, Clone)]
pub(crate) struct OlsFit {
    /// In the order of the columns given.
    pub coef: Vec<f64>,
    /// `y - X coef`, formed explicitly as statsmodels does.
    pub resid: Vec<f64>,
    pub ssr: f64,
    pub rank: usize,
    pub nobs: usize,
    /// The diagonal of `(X'X)^-1`, in the order of the columns given.
    pub inv_diag: Vec<f64>,
}

impl OlsFit {
    /// The t-value of column `j`.
    pub fn tvalue(&self, j: usize) -> Result<f64, StatsError> {
        let df = self.nobs - self.rank;
        if df == 0 {
            return Err(StatsError::InsufficientObservations {
                what: "regression".to_string(),
                found: self.nobs,
                needed: self.rank + 1,
            });
        }
        let t = self.coef[j] / (self.ssr / df as f64 * self.inv_diag[j]).sqrt();
        if t.is_finite() {
            Ok(t)
        } else {
            Err(StatsError::DegenerateSeries(
                "the regression fits exactly: the test statistic is not finite".to_string(),
            ))
        }
    }
}

/// Regresses `y` on `cols`. Collinear columns are a [`StatsError::DegenerateSeries`].
pub(crate) fn ols(cols: &[&[f64]], y: &[f64]) -> Result<OlsFit, StatsError> {
    let (m, k) = (y.len(), cols.len());
    if m <= k {
        return Err(StatsError::InsufficientObservations {
            what: "regression".to_string(),
            found: m,
            needed: k + 1,
        });
    }
    let mut data = vec![0.0; m * k];
    for (j, c) in cols.iter().enumerate() {
        for i in 0..m {
            data[i * k + j] = c[i];
        }
    }
    let x = Mat::from_vec(m, k, data)?;
    let qr = qr_pivoted(&x)?;
    if qr.rank() < k {
        return Err(StatsError::DegenerateSeries(format!(
            "the regressors are collinear (rank {} of {k})",
            qr.rank()
        )));
    }
    let coef = qr.solve_basic(y)?.x;
    let mut resid = Vec::with_capacity(m);
    for i in 0..m {
        let fitted: f64 = (0..k).map(|j| x[(i, j)] * coef[j]).sum();
        resid.push(y[i] - fitted);
    }
    let ssr: f64 = resid.iter().map(|r| r * r).sum();
    // (X'X)^-1 = P R^-1 R^-T P'; its diagonal at pivot position p is the squared norm of row p
    // of R^-1
    let r = qr.r();
    let mut rinv = vec![0.0; k * k];
    for c in 0..k {
        for i in (0..=c).rev() {
            let mut s = if i == c { 1.0 } else { 0.0 };
            for j in i + 1..=c {
                s -= r[(i, j)] * rinv[j * k + c];
            }
            rinv[i * k + c] = s / r[(i, i)];
        }
    }
    let mut inv_diag = vec![0.0; k];
    for (p, orig) in qr.perm().iter().enumerate() {
        inv_diag[*orig] = (p..k).map(|c| rinv[p * k + c].powi(2)).sum();
    }
    Ok(OlsFit { coef, resid, ssr, rank: k, nobs: m, inv_diag })
}

// ------------------------------------------------------------------------------------- ADF

/// An augmented Dickey-Fuller test.
#[derive(Debug, Clone, PartialEq)]
pub struct Adf {
    pub statistic: f64,
    pub p_value: f64,
    /// Lagged differences used.
    pub lags: usize,
    /// Observations of the final regression.
    pub nobs: usize,
    /// 1%, 5%, 10% ([`ADF_CRITICAL_KEYS`]).
    pub critical_values: [f64; 3],
    /// The smallest AIC, when the lag was searched.
    pub icbest: Option<f64>,
    /// AIC of the fit with `k` lagged differences at index `k`, every fit on the common sample.
    /// Empty when the lag was not searched.
    pub aic_by_lag: Vec<f64>,
    /// The longest lag searched, or the lag used when it was not.
    pub maxlag: usize,
}

fn trend_columns(regression: Regression, nobs: usize) -> Vec<Vec<f64>> {
    let t: Vec<f64> = (1..=nobs).map(|i| i as f64).collect();
    let mut cols = Vec::new();
    if regression.n_trend() >= 1 {
        cols.push(vec![1.0; nobs]);
    }
    if regression.n_trend() >= 2 {
        cols.push(t.clone());
    }
    if regression.n_trend() >= 3 {
        cols.push(t.iter().map(|v| v * v).collect());
    }
    cols
}

/// `d x_t`, the lagged level `x_t` and the lagged differences `d x_{t-1} .. d x_{t-l}` for the
/// rows `t = l .. n - 2`.
fn adf_rows(x: &[f64], xd: &[f64], l: usize) -> (Vec<f64>, Vec<f64>, Vec<Vec<f64>>) {
    let n = x.len();
    let y = xd[l..].to_vec();
    let level = x[l..n - 1].to_vec();
    let lags = (1..=l).map(|j| (l..n - 1).map(|t| xd[t - j]).collect()).collect();
    (y, level, lags)
}

/// `aic` of statsmodels' `OLS`: `-2 llf + 2 rank`.
fn aic(fit: &OlsFit) -> Result<f64, StatsError> {
    if fit.ssr <= 0.0 {
        return Err(StatsError::DegenerateSeries(
            "the regression fits exactly: the lag cannot be chosen".to_string(),
        ));
    }
    let n = fit.nobs as f64;
    let llf =
        -(n / 2.0) * (2.0 * std::f64::consts::PI).ln() - (n / 2.0) * (fit.ssr / n).ln() - n / 2.0;
    Ok(-2.0 * llf + 2.0 * fit.rank as f64)
}

/// The augmented Dickey-Fuller test, as `statsmodels.tsa.stattools.adfuller`.
///
/// `maxlag` defaults to Schwert's `ceil(12 (n/100)^(1/4))`, capped at `n/2 - ntrend - 1`. With
/// `autolag` the lag is chosen by AIC among `0..=maxlag`, otherwise `maxlag` is used.
pub fn adfuller(
    x: &[f64],
    regression: Regression,
    maxlag: Option<usize>,
    autolag: bool,
) -> Result<Adf, StatsError> {
    require_finite(x, "series")?;
    let n = x.len();
    if n == 0 || is_constant(x) {
        return Err(StatsError::DegenerateSeries("series is constant".to_string()));
    }
    let nt = regression.n_trend();
    let cap = (n / 2) as isize - nt as isize - 1;
    let maxlag = match maxlag {
        None => {
            if cap < 0 {
                return Err(StatsError::InsufficientObservations {
                    what: format!("series (regression '{}')", regression.name()),
                    found: n,
                    needed: 2 * (nt + 1),
                });
            }
            let schwert = (12.0 * (n as f64 / 100.0).powf(0.25)).ceil() as usize;
            schwert.min(cap as usize)
        }
        Some(m) if m as isize > cap => {
            return Err(StatsError::Invalid(format!(
                "maxlag {m} must not exceed nobs/2 - 1 - ntrend = {cap}"
            )))
        }
        Some(m) => m,
    };
    // A constant is in the span of every regression but 'n': the statistic does not change when
    // the level is centred, and the fit keeps its digits when the level is far from zero.
    let centred: Vec<f64>;
    let x = if regression == Regression::N {
        x
    } else {
        let m = mean(x);
        centred = x.iter().map(|v| v - m).collect();
        &centred
    };
    let xd: Vec<f64> = x.windows(2).map(|w| w[1] - w[0]).collect();

    let mut aic_by_lag = Vec::new();
    let mut icbest = None;
    let used = if autolag {
        let (y, level, lags) = adf_rows(x, &xd, maxlag);
        let trend = trend_columns(regression, y.len());
        let mut full: Vec<&[f64]> = trend.iter().map(|c| c.as_slice()).collect();
        full.push(&level);
        full.extend(lags.iter().map(|c| c.as_slice()));
        for k in 0..=maxlag {
            aic_by_lag.push(aic(&ols(&full[..nt + 1 + k], &y)?)?);
        }
        let mut best = 0;
        for (k, v) in aic_by_lag.iter().enumerate() {
            if *v < aic_by_lag[best] {
                best = k;
            }
        }
        icbest = Some(aic_by_lag[best]);
        best
    } else {
        maxlag
    };

    let (y, level, lags) = adf_rows(x, &xd, used);
    let trend = trend_columns(regression, y.len());
    let mut cols: Vec<&[f64]> = vec![&level];
    cols.extend(lags.iter().map(|c| c.as_slice()));
    cols.extend(trend.iter().map(|c| c.as_slice()));
    let fit = ols(&cols, &y)?;
    let statistic = fit.tvalue(0)?;
    let nobs = y.len();
    Ok(Adf {
        statistic,
        p_value: mackinnon_p(statistic, regression, 1).expect("N = 1 is in the 1994 tables"),
        lags: used,
        nobs,
        critical_values: mackinnon_crit(regression, 1, nobs as f64)
            .expect("N = 1 is in the 2010 tables"),
        icbest,
        aic_by_lag,
        maxlag,
    })
}

// ------------------------------------------------------------------------------------ KPSS

/// What the KPSS null holds stationary around.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KpssTrend {
    /// A level (`c`).
    Level,
    /// A linear trend (`ct`).
    Trend,
}

/// How many autocovariances the long-run variance takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KpssLags {
    /// Hobijn, Franses and Ooms (1998).
    Auto,
    /// Schwert's `ceil(12 (n/100)^(1/4))`.
    Legacy,
    Fixed(usize),
}

/// A KPSS test.
#[derive(Debug, Clone, PartialEq)]
pub struct Kpss {
    pub statistic: f64,
    /// Interpolated from the table, so within `[0.01, 0.10]`.
    pub p_value: f64,
    /// The p-value sits on the table's bound: the true value is smaller than 0.01 or greater
    /// than 0.10 (statsmodels' `InterpolationWarning`).
    pub p_value_clipped: bool,
    pub lags: usize,
    /// 10%, 5%, 2.5%, 1% ([`KPSS_CRITICAL_KEYS`]).
    pub critical_values: [f64; 4],
}

fn kpss_critical(trend: KpssTrend) -> [f64; 4] {
    match trend {
        KpssTrend::Level => tables::KPSS_CRIT_C,
        KpssTrend::Trend => tables::KPSS_CRIT_CT,
    }
}

/// `numpy.interp` for increasing `xp`, clamped to the end values.
fn interp(x: f64, xp: &[f64; 4], fp: &[f64; 4]) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x < xp[0] {
        return fp[0];
    }
    if x >= xp[3] {
        return fp[3];
    }
    let j = (0..3).rev().find(|j| x >= xp[*j]).unwrap_or(0);
    let slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]);
    slope * (x - xp[j]) + fp[j]
}

/// The KPSS p-value of a statistic: linear interpolation between the table's critical values,
/// 0.10 below the first and 0.01 above the last.
pub fn kpss_p_value(statistic: f64, trend: KpssTrend) -> f64 {
    interp(statistic, &kpss_critical(trend), &tables::KPSS_PVALS)
}

/// The lag of Hobijn, Franses and Ooms (1998) for a Bartlett kernel.
fn kpss_autolag(resids: &[f64]) -> usize {
    let n = resids.len();
    let nf = n as f64;
    let covlags = nf.powf(2.0 / 9.0) as usize;
    let mut s0 = sum_of_squares(resids) / nf;
    let mut s1 = 0.0;
    for i in 1..=covlags {
        let prod = autocov_sum(resids, i) / (nf / 2.0);
        s0 += prod;
        s1 += i as f64 * prod;
    }
    let s_hat = s1 / s0;
    let gamma_hat = 1.1447 * (s_hat * s_hat).powf(1.0 / 3.0);
    (gamma_hat * nf.powf(1.0 / 3.0)) as usize
}

/// `sum_t r_t r_{t-i}`.
fn autocov_sum(r: &[f64], i: usize) -> f64 {
    r[i..].iter().zip(&r[..r.len() - i]).map(|(a, b)| a * b).sum()
}

/// The KPSS test, as `statsmodels.tsa.stattools.kpss`.
pub fn kpss(x: &[f64], trend: KpssTrend, lags: KpssLags) -> Result<Kpss, StatsError> {
    require_finite(x, "series")?;
    let n = x.len();
    if n < 2 {
        return Err(StatsError::InsufficientObservations {
            what: "series".to_string(),
            found: n,
            needed: 2,
        });
    }
    let resids: Vec<f64> = match trend {
        KpssTrend::Level => {
            let m = mean(x);
            x.iter().map(|v| v - m).collect()
        }
        KpssTrend::Trend => {
            let ones = vec![1.0; n];
            let t: Vec<f64> = (1..=n).map(|i| i as f64).collect();
            ols(&[&ones, &t], x)?.resid
        }
    };
    let nf = n as f64;
    let lags = match lags {
        KpssLags::Legacy => ((12.0 * (nf / 100.0).powf(0.25)).ceil() as usize).min(n - 1),
        KpssLags::Auto => kpss_autolag(&resids).min(n - 1),
        KpssLags::Fixed(k) if k >= n => {
            return Err(StatsError::Invalid(format!(
                "lags ({k}) must be < number of observations ({n})"
            )))
        }
        KpssLags::Fixed(k) => k,
    };
    let mut cum = 0.0;
    let cumulative: Vec<f64> = resids
        .iter()
        .map(|r| {
            cum += r;
            cum
        })
        .collect();
    let eta = sum_of_squares(&cumulative) / (nf * nf);
    let mut s_hat = sum_of_squares(&resids);
    for i in 1..=lags {
        s_hat += 2.0 * autocov_sum(&resids, i) * (1.0 - (i as f64 / (lags as f64 + 1.0)));
    }
    let s_hat = s_hat / nf;
    if s_hat <= 0.0 {
        return Err(StatsError::DegenerateSeries(
            "the long-run variance is zero: the series is constant".to_string(),
        ));
    }
    let statistic = eta / s_hat;
    let p_value = kpss_p_value(statistic, trend);
    Ok(Kpss {
        statistic,
        p_value,
        // statsmodels compares the interpolated value with the table's end points
        p_value_clipped: p_value == tables::KPSS_PVALS[3] || p_value == tables::KPSS_PVALS[0],
        lags,
        critical_values: kpss_critical(trend),
    })
}

// ------------------------------------------------------------------------------- together

/// What the two tests say together.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reading {
    /// ADF rejects a unit root and KPSS does not reject stationarity.
    Stationary,
    /// ADF cannot reject a unit root and KPSS rejects stationarity.
    UnitRoot,
    /// Both reject: typically a trend or a break, not a clean unit root.
    Contradictory,
    /// Neither rejects.
    Inconclusive,
}

impl Reading {
    /// The `reading` value.
    pub fn as_str(self) -> &'static str {
        match self {
            Reading::Stationary => "stationary",
            Reading::UnitRoot => "unit_root",
            Reading::Contradictory => "contradictory",
            Reading::Inconclusive => "inconclusive",
        }
    }
}

/// The `stationarity` result: the ADF and KPSS tests and their combined reading.
#[derive(Debug, Clone, PartialEq)]
pub struct Stationarity {
    pub n: usize,
    pub alpha: f64,
    pub reading: Reading,
    pub adf: Adf,
    pub rejects_unit_root: bool,
    pub kpss: Kpss,
    pub rejects_stationarity: bool,
}

/// ADF (constant, AIC lags) and KPSS (level, automatic lags) on one series, read at `alpha`.
pub fn stationarity(x: &[f64], alpha: f64) -> Result<Stationarity, StatsError> {
    if x.len() < MIN_OBS {
        return Err(StatsError::InsufficientObservations {
            what: "series".to_string(),
            found: x.len(),
            needed: MIN_OBS,
        });
    }
    require_finite(x, "series")?;
    if is_constant(x) {
        return Err(StatsError::DegenerateSeries("series is constant".to_string()));
    }
    let adf = adfuller(x, Regression::C, None, true)?;
    let kp = kpss(x, KpssTrend::Level, KpssLags::Auto)?;
    let rejects_unit_root = adf.p_value < alpha;
    let rejects_stationarity = kp.p_value < alpha;
    let reading = match (rejects_unit_root, rejects_stationarity) {
        (true, false) => Reading::Stationary,
        (false, true) => Reading::UnitRoot,
        (true, true) => Reading::Contradictory,
        (false, false) => Reading::Inconclusive,
    };
    Ok(Stationarity {
        n: x.len(),
        alpha,
        reading,
        adf,
        rejects_unit_root,
        kpss: kp,
        rejects_stationarity,
    })
}
