//! Principal components of a panel of return series (of the correlation matrix: each series is
//! standardised) and, optionally, what a target series loads on the first components' scores.
//!
//! Sign rule. A singular vector is only defined up to sign, so each component is oriented to
//! load up on net: its loadings sum to a positive number. When that sum is within `SIGN_TOL` of
//! zero (a pair of opposite signs, a component that is orthogonal to the common factor) the sign
//! of the sum is numerical noise, and the rule falls back to making the loading of largest
//! magnitude positive (the first one, when several are within `SIGN_TOL` of the largest).
//! The orientation is therefore the same whichever SVD routine produced the vectors. When a sum is exactly
//! zero the sign of the raw vector is not used.
//!
//! A component with a zero singular value (a perfectly collinear panel) has no information: its
//! exposure in the target regression is reported as beta 0 and t 0, as a minimum-norm
//! least squares gives, and the regression is fitted on the other components. (Inverting the
//! nearly singular `X'X` instead gives every component a wrong t-statistic: 0.138 where
//! ordinary least squares gives 0.932 for a panel of two identical series and a third.)

use crate::math::linalg::{svd, Mat};

use super::error::StatsError;
use super::hac::{finite_series, peak_to_peak};

pub const MIN_OBS: usize = 60;
pub const MAX_SERIES: usize = 40;
pub const MAX_COMPONENTS: usize = 5;
/// A loading sum within this of zero is treated as zero by the sign rule.
pub const SIGN_TOL: f64 = 1e-9;

/// One component's exposure in the target regression: the coefficient on its score and its OLS
/// t-statistic (not HAC).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Exposure {
    pub beta: f64,
    pub t: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PcaTarget {
    pub r_squared: f64,
    pub alpha_per_period: f64,
    pub alpha_t: f64,
    /// One per component, in order.
    pub exposures_per_unit_score: Vec<Exposure>,
}

/// The result of [`pca_factors`], full precision. `loadings[j][i]` is series `i` on component
/// `j`; the series names stay with the caller.
#[derive(Debug, Clone, PartialEq)]
pub struct Pca {
    pub n: usize,
    pub components: usize,
    pub explained_variance_ratio: Vec<f64>,
    pub cumulative_explained: f64,
    pub loadings: Vec<Vec<f64>>,
    pub first_component_is_common_factor: bool,
    pub target: Option<PcaTarget>,
}

/// Orients a component by the sign rule in the module doc.
fn orient(row: &mut [f64]) {
    let sum: f64 = row.iter().sum();
    let negative = if sum.abs() > SIGN_TOL {
        sum < 0.0
    } else {
        let biggest = row.iter().fold(0.0f64, |m, x| m.max(x.abs()));
        let pivot = row.iter().position(|x| x.abs() >= biggest - SIGN_TOL).unwrap_or(0);
        row[pivot] < 0.0
    };
    if negative {
        row.iter_mut().for_each(|x| *x = -*x);
    }
}

/// Components of the correlation matrix of `cols` (equal-length series). `components` is capped
/// at 5 and at the number of series. With `target` (same length), its regression on the scores
/// of those components.
pub fn pca_factors<C: AsRef<[f64]>>(
    cols: &[C],
    components: usize,
    target: Option<&[f64]>,
) -> Result<Pca, StatsError> {
    if cols.len() > MAX_SERIES {
        return Err(StatsError::Invalid(format!("{} series, at most {MAX_SERIES}", cols.len())));
    }
    if cols.len() < 2 {
        return Err(StatsError::InsufficientObservations {
            what: "returns".into(),
            found: cols.len(),
            needed: 2,
        });
    }
    let series = cols
        .iter()
        .enumerate()
        .map(|(i, c)| finite_series(c.as_ref(), &format!("returns[{i}]"), MIN_OBS))
        .collect::<Result<Vec<_>, _>>()?;
    let n = series[0].len();
    if series.iter().any(|s| s.len() != n) {
        let mut lens: Vec<usize> = series.iter().map(|s| s.len()).collect();
        lens.sort_unstable();
        lens.dedup();
        return Err(StatsError::Invalid(format!(
            "returns: the series differ in length ({lens:?})"
        )));
    }
    if series.iter().any(|s| peak_to_peak(s) == 0.0) {
        return Err(StatsError::DegenerateSeries("a series is constant".into()));
    }
    let k = series.len();
    let c = components.min(MAX_COMPONENTS).min(k);
    let nf = n as f64;

    // z: n x k, each column centred and divided by its sample standard deviation
    let mut z = vec![0.0; n * k];
    for (j, s) in series.iter().enumerate() {
        let m = super::hac::mean(s);
        let sd = (s.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (nf - 1.0)).sqrt();
        for (t, x) in s.iter().enumerate() {
            z[t * k + j] = (x - m) / sd;
        }
    }
    let zm = Mat::from_vec(n, k, z).map_err(|e| StatsError::Invalid(format!("pca: {e}")))?;
    let f = svd(&zm).map_err(|e| StatsError::DegenerateSeries(format!("pca: {e}")))?;

    let var: Vec<f64> = f.s.iter().map(|s| s * s / (nf - 1.0)).collect();
    let total: f64 = var.iter().sum();
    let ratio: Vec<f64> = var.iter().map(|v| v / total).collect();
    let mut vt: Vec<Vec<f64>> = (0..k).map(|j| (0..k).map(|i| f.v[(i, j)]).collect()).collect();
    vt.iter_mut().for_each(|r| orient(r));

    let first = &vt[0];
    let common = first.iter().all(|x| *x > 0.0) || first.iter().all(|x| *x < 0.0);
    let loadings: Vec<Vec<f64>> = vt[..c].to_vec();

    let target = match target {
        None => None,
        Some(t) => {
            let t = finite_series(t, "target", n)?;
            if t.len() != n {
                return Err(StatsError::Invalid("target and panel differ in length".into()));
            }
            // a score whose singular value is at rounding level carries no information
            let null: Vec<bool> =
                (0..c).map(|j| f.s[j] <= n.max(k) as f64 * f64::EPSILON * f.s[0]).collect();
            Some(regress(&zm, &loadings, &null, t, n)?)
        }
    };
    Ok(Pca {
        n,
        components: c,
        explained_variance_ratio: ratio[..c].to_vec(),
        cumulative_explained: ratio[..c].iter().sum(),
        loadings,
        first_component_is_common_factor: common,
        target,
    })
}

/// OLS of `target` on a constant and the scores `z . loading_j` of the components that are not
/// `null`; a null component gets coefficient 0 and t 0 (see the module doc). The residual
/// variance divides by `n - c - 1` with `c` counting the null components.
fn regress(
    z: &Mat,
    loadings: &[Vec<f64>],
    null: &[bool],
    target: &[f64],
    n: usize,
) -> Result<PcaTarget, StatsError> {
    let c = loadings.len();
    let kept: Vec<usize> = (0..c).filter(|j| !null[*j]).collect();
    let p = kept.len() + 1;
    let mut x = vec![0.0; n * p];
    for t in 0..n {
        x[t * p] = 1.0;
        let row = z.row(t);
        for (col, &j) in kept.iter().enumerate() {
            x[t * p + 1 + col] = row.iter().zip(&loadings[j]).map(|(a, b)| a * b).sum();
        }
    }
    let xm = Mat::from_vec(n, p, x).map_err(|e| StatsError::Invalid(format!("pca: {e}")))?;
    let f = svd(&xm).map_err(|e| StatsError::DegenerateSeries(format!("pca: {e}")))?;
    let rank = f.rank(n.max(p) as f64 * f64::EPSILON);

    let mut beta = vec![0.0; p];
    let mut cov_diag = vec![0.0; p];
    for r in 0..rank {
        let coef = (0..n).map(|t| f.u[(t, r)] * target[t]).sum::<f64>() / f.s[r];
        for i in 0..p {
            beta[i] += coef * f.v[(i, r)];
            cov_diag[i] += f.v[(i, r)] * f.v[(i, r)] / (f.s[r] * f.s[r]);
        }
    }
    let mut rss = 0.0;
    for (t, y) in target.iter().enumerate() {
        let fit: f64 = (0..p).map(|i| xm.row(t)[i] * beta[i]).sum();
        rss += (y - fit) * (y - fit);
    }
    let tm = super::hac::mean(target);
    let sst: f64 = target.iter().map(|y| (y - tm) * (y - tm)).sum();
    if sst == 0.0 {
        return Err(StatsError::DegenerateSeries("target is constant".into()));
    }
    let s2 = rss / (n - c - 1) as f64;
    let t_of = |i: usize| {
        let se = (s2 * cov_diag[i]).sqrt();
        if se == 0.0 && beta[i] == 0.0 {
            0.0
        } else {
            beta[i] / se
        }
    };
    let mut exposures = vec![Exposure { beta: 0.0, t: 0.0 }; c];
    for (col, &j) in kept.iter().enumerate() {
        exposures[j] = Exposure { beta: beta[col + 1], t: t_of(col + 1) };
    }
    Ok(PcaTarget {
        r_squared: 1.0 - rss / sst,
        alpha_per_period: beta[0],
        alpha_t: t_of(0),
        exposures_per_unit_score: exposures,
    })
}
