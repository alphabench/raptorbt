//! The mean of a return series against zero with a Newey-West (HAC) standard error, and the
//! input checks the statistics of this family share.
//!
//! The standard error is statsmodels' `OLS(r, 1).fit(cov_type="HAC")` with the Bartlett kernel
//! and no small-sample correction: `Var = (S0 + 2 sum_j w_j G_j) / n^2` with `w_j = 1 - j/(L+1)`
//! and `G_j = sum_t e_t e_{t-j}`. The HAC p-value uses the normal reference; the naive one the
//! t distribution with `n - 1` degrees of freedom.

use crate::math::fsum::fsum;
use crate::math::special::t_two_sided_p;
use crate::pricing::norm::erfc;

use super::error::StatsError;

/// Fewest observations the t-statistic is computed on.
pub const MIN_HAC_OBS: usize = 30;

/// A series that is long enough and finite: the length is checked first.
pub(crate) fn finite_series<'a>(
    x: &'a [f64],
    name: &str,
    min_n: usize,
) -> Result<&'a [f64], StatsError> {
    if x.len() < min_n {
        return Err(StatsError::InsufficientObservations {
            what: name.to_string(),
            found: x.len(),
            needed: min_n,
        });
    }
    if x.iter().any(|v| !v.is_finite()) {
        return Err(StatsError::NonFinite { what: name.to_string() });
    }
    Ok(x)
}

/// `max - min` of a non-empty finite series.
pub(crate) fn peak_to_peak(x: &[f64]) -> f64 {
    let (lo, hi) =
        x.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| (lo.min(v), hi.max(v)));
    hi - lo
}

/// Whether a finite series varies by no more than rounding noise: `max - min` at most
/// [`CONSTANT_REL`] of its largest magnitude (a series of ulps around a level is constant, and
/// a fit to it prints numbers that are the noise).
pub(crate) fn is_numerically_constant(x: &[f64]) -> bool {
    let level = x.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    peak_to_peak(x) <= CONSTANT_REL * level
}

/// Relative spread at or below which a series counts as constant.
pub(crate) const CONSTANT_REL: f64 = 1e-12;

pub(crate) fn mean(x: &[f64]) -> f64 {
    fsum(x.iter().copied()) / x.len() as f64
}

/// The result of [`hac_tstat`], full precision. Rounding and wording are the caller's.
#[derive(Debug, Clone, PartialEq)]
pub struct Hac {
    pub n: usize,
    pub mean: f64,
    pub hac_std_error: f64,
    pub hac_t: f64,
    pub hac_p_value: f64,
    pub naive_t: f64,
    pub naive_p_value: f64,
    pub lags: usize,
    pub per_period_sharpe: f64,
    /// NaN when the first `n - 1` returns are constant.
    pub lag1_autocorrelation: f64,
}

/// The mean of `returns` against zero with a Newey-West standard error. `lags` defaults to the
/// rule `floor(4 (n/100)^(2/9))`; a given value must lie in `0..n/2`.
pub fn hac_tstat(returns: &[f64], lags: Option<i64>) -> Result<Hac, StatsError> {
    let r = finite_series(returns, "returns", MIN_HAC_OBS)?;
    let n = r.len();
    if peak_to_peak(r) == 0.0 {
        return Err(StatsError::DegenerateSeries("returns are constant".into()));
    }
    let used = match lags {
        Some(l) => l,
        None => (4.0 * (n as f64 / 100.0).powf(2.0 / 9.0)).floor() as i64,
    };
    if !(0 <= used && (used as usize) < n / 2) {
        return Err(StatsError::Invalid(format!(
            "lags must be 0 to {} for {} observations",
            n / 2 - 1,
            n
        )));
    }
    let used = used as usize;
    let nf = n as f64;
    let m = mean(r);
    // `r - m` is exact for returns near their mean; its sum is the rounding error of `m`, which is
    // taken out again so that the residuals are centred to far below one ulp of the mean (a
    // series with a mean of 1 and noise of 1e-9 would otherwise lose 8 digits in the variance)
    let shift = fsum(r.iter().map(|v| v - m)) / n as f64;
    let e: Vec<f64> = r.iter().map(|v| (v - m) - shift).collect();

    let s0: f64 = e.iter().map(|v| v * v).sum();
    let mut long_run = s0;
    for j in 1..=used {
        let w = 1.0 - j as f64 / (used as f64 + 1.0);
        let gamma: f64 = (j..n).map(|t| e[t] * e[t - j]).sum();
        long_run += 2.0 * w * gamma;
    }
    let hac_se = (long_run / (nf * nf)).sqrt();
    let hac_t = m / hac_se;
    let hac_p = erfc(hac_t.abs() / std::f64::consts::SQRT_2);

    let naive_se = (s0 / (nf - 1.0) / nf).sqrt();
    let naive_t = m / naive_se;
    let naive_p = t_two_sided_p(naive_t, nf - 1.0)?;

    let sd = (s0 / (nf - 1.0)).sqrt();
    Ok(Hac {
        n,
        mean: m,
        hac_std_error: hac_se,
        hac_t,
        hac_p_value: hac_p,
        naive_t,
        naive_p_value: naive_p,
        lags: used,
        per_period_sharpe: m / sd,
        lag1_autocorrelation: correlation(&r[..n - 1], &r[1..]),
    })
}

/// Pearson correlation of two equal-length series.
pub(crate) fn correlation(x: &[f64], y: &[f64]) -> f64 {
    let (mx, my) = (mean(x), mean(y));
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for (a, b) in x.iter().zip(y) {
        let (da, db) = (a - mx, b - my);
        sxy += da * db;
        sxx += da * da;
        syy += db * db;
    }
    sxy / (sxx.sqrt() * syy.sqrt())
}
