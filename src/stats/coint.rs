//! Do series share a long-run equilibrium: Engle-Granger (two-step, one cointegrating vector)
//! and Johansen (the rank of the system), with the hedge ratio and the spread's half-life.
//!
//! Follows the statsmodels functions it replaces:
//!
//! - `coint` (Engle-Granger): regress the first series on a constant and the others, run the
//!   no-constant ADF test with AIC lags on the residual, read the statistic against MacKinnon's
//!   cointegration p-value for `N` series and the 2010 critical values at `nobs - 1`
//!   observations. When the regression's R-squared is above `1 - 100 sqrt(eps)` the series are
//!   (almost) one line and the statistic is `-inf`.
//! - `coint_johansen`: the trace and maximum-eigenvalue statistics of the VECM with
//!   `k_ar_diff` lagged differences and `det_order` -1, 0 or 1, and the table critical values
//!   of MacKinnon, Haug and Michelis (1999).
//!
//! Johansen is solved in its symmetric form. With `S_kk = L L'` (Cholesky), `S_00 = L0 L0'` and
//! `S_k0` the moment matrices of the residuals, the eigenvalues of `S_kk^-1 S_k0 S_00^-1 S_0k`
//! are those of the symmetric `C = G G'`, `G = L^-1 S_k0 L0^-T`, found by Jacobi, and the
//! eigenvectors are `d = L^-T w`, which satisfy `d' S_kk d = I` as in statsmodels. statsmodels
//! solves the non-symmetric problem with `eig`, which can return a complex answer or a matrix
//! that is not positive definite; this port has no such outcome: a singular moment matrix is
//! [`StatsError::DegenerateSeries`]. The residual series are scaled to unit root-mean-square
//! first, and a moment matrix whose condition number exceeds [`MAX_CONDITION`] is refused (the eigenvalues do not change, the vectors are scaled back), so a panel with columns
//! of very different units is not mistaken for a singular one.
//!
//! The sign of an eigenvector is arbitrary. This port fixes it: each column of `evec` is signed
//! so that its entry of largest magnitude is positive (the first on a tie). statsmodels signs
//! the whole matrix by its first entry and leaves the columns to LAPACK; a comparison
//! normalises the oracle's columns by the same rule.
//!
//! Differences from statsmodels, each on purpose:
//! - `p_value` is `None` for 7 to 12 series (the 1994 table stops at 6, where statsmodels
//!   raises `IndexError`); the 2010 critical values and Johansen still answer, and
//!   `cointegrated` falls back to the statistic against the critical value at `alpha` when
//!   `alpha` is 0.01, 0.05 or 0.10, else `None`;
//! - Johansen's singular panels are [`StatsError::DegenerateSeries`] instead of numpy's
//!   `LinAlgError`.

use super::error::StatsError;
use super::stationarity::{
    adfuller, is_constant, mackinnon_crit, mackinnon_p, mean, ols, require_finite, Regression,
};
use super::tables;
use crate::math::linalg::{cholesky, qr_pivoted, sym_eigen, Cholesky, Mat};

/// Observations per series Engle-Granger and Johansen need.
pub const MIN_OBS: usize = 60;
/// Lagged differences of the Johansen fit in [`cointegration`].
const KAR_DIFF: usize = 1;

/// Below this scale factor the correction removes over a tenth of the statistic: a small sample.
const SMALL_SAMPLE_SCALE: f64 = 0.9;

/// Johansen's critical values stop at 12 series.
pub const MAX_SERIES: usize = 12;
/// The significance level the tests are read at.
pub const ALPHA: f64 = 0.05;
/// Keys of [`EngleGranger::critical_values`].
pub const CRITICAL_KEYS: [&str; 3] = ["1%", "5%", "10%"];

/// Periods for a deviation to halve, from the AR(1) of the spread (change on lagged level);
/// `None` when the spread does not revert (a non-negative slope) or has nothing to regress on
/// (a constant spread, under three values).
pub fn half_life(spread: &[f64]) -> Option<f64> {
    if spread.len() < 3 {
        return None;
    }
    let lag = &spread[..spread.len() - 1];
    let diff: Vec<f64> = spread.windows(2).map(|w| w[1] - w[0]).collect();
    let ones = vec![1.0; lag.len()];
    let fit = ols(&[&ones, lag], &diff).ok()?;
    let phi = 1.0 + fit.coef[1];
    if !(0.0 < phi && phi < 1.0) {
        return None;
    }
    Some(-(2.0f64.ln()) / phi.ln())
}

/// The Engle-Granger two-step test and the regression it rests on.
#[derive(Debug, Clone, PartialEq)]
pub struct EngleGranger {
    /// The ADF statistic of the residual; `-inf` when the series are (almost) collinear.
    pub statistic: f64,
    /// MacKinnon's p-value; `None` for 7 or more series.
    pub p_value: Option<f64>,
    /// 1%, 5%, 10% ([`CRITICAL_KEYS`]).
    pub critical_values: [f64; 3],
    pub cointegrated: Option<bool>,
    /// Lags of the residual ADF test; `None` in the collinear branch.
    pub adf_lags: Option<usize>,
    pub intercept: f64,
    /// One per regressor, in the order given.
    pub hedge_ratios: Vec<f64>,
    pub spread_mean: f64,
    pub spread_std: f64,
    pub spread_last_z: f64,
    pub half_life_periods: Option<f64>,
}

/// Engle-Granger of `y` on the columns `x` (levels), read at `alpha`.
pub fn engle_granger(y: &[f64], x: &[&[f64]], alpha: f64) -> Result<EngleGranger, StatsError> {
    let nobs = y.len();
    if x.is_empty() || x.iter().any(|c| c.len() != nobs) {
        return Err(StatsError::Invalid(
            "engle_granger needs at least one regressor, each as long as y".to_string(),
        ));
    }
    let k_vars = x.len() + 1;
    let ones = vec![1.0; nobs];
    let mut cols: Vec<&[f64]> = vec![&ones];
    cols.extend_from_slice(x);
    let fit = ols(&cols, y)?;
    let ybar = mean(y);
    let tss: f64 = y.iter().map(|v| (v - ybar) * (v - ybar)).sum();
    if tss == 0.0 {
        return Err(StatsError::DegenerateSeries("series is constant".to_string()));
    }
    let rsquared = 1.0 - fit.ssr / tss;
    let (statistic, adf_lags) = if rsquared < 1.0 - 100.0 * f64::EPSILON.sqrt() {
        let adf = adfuller(&fit.resid, Regression::N, None, true)?;
        (adf.statistic, Some(adf.lags))
    } else {
        (f64::NEG_INFINITY, None)
    };
    let p_value = mackinnon_p(statistic, Regression::C, k_vars);
    let critical_values = mackinnon_crit(Regression::C, k_vars, (nobs - 1) as f64)
        .ok_or_else(|| StatsError::Invalid(format!("{k_vars} series, at most {MAX_SERIES}")))?;
    let cointegrated = match p_value {
        Some(p) => Some(p < alpha),
        None => [0.01, 0.05, 0.10]
            .iter()
            .position(|a| *a == alpha)
            .map(|i| statistic < critical_values[i]),
    };
    let spread = &fit.resid;
    let spread_mean = mean(spread);
    let spread_std =
        (spread.iter().map(|v| (v - spread_mean).powi(2)).sum::<f64>() / (nobs - 1) as f64).sqrt();
    Ok(EngleGranger {
        statistic,
        p_value,
        critical_values,
        cointegrated,
        adf_lags,
        intercept: fit.coef[0],
        hedge_ratios: fit.coef[1..].to_vec(),
        spread_mean,
        spread_std,
        spread_last_z: (spread[nobs - 1] - spread_mean) / spread_std,
        half_life_periods: half_life(spread),
    })
}

// -------------------------------------------------------------------------------- Johansen

/// Johansen's test of a VECM.
#[derive(Debug, Clone)]
pub struct Johansen {
    /// Eigenvalues (squared canonical correlations), descending.
    pub eig: Vec<f64>,
    /// Eigenvectors in the columns, normalised `d' S_kk d = I` and signed as the module says.
    pub evec: Mat,
    /// Trace statistics, hypothesis `rank <= i`.
    pub lr1: Vec<f64>,
    /// Maximum-eigenvalue statistics.
    pub lr2: Vec<f64>,
    /// Trace critical values at 90%, 95%, 99%, one row per hypothesis.
    pub cvt: Vec<[f64; 3]>,
    /// Maximum-eigenvalue critical values at 90%, 95%, 99%.
    pub cvm: Vec<[f64; 3]>,
    /// Observations in the moment matrices.
    pub nobs: usize,
}

/// The largest condition number of an equilibrated moment matrix Johansen is solved at. The
/// eigenvalues carry an error of about `eps * cond`, so `1e-8 / eps` (4.5e7) is the point beyond
/// which the 1e-8 the statistic is quoted to is not determined by the data. Measured on the
/// fixtures: well-posed panels sit at 3 to 2.5e3; the 5-series rank-2 panel at 6e7 had eigenvalues
/// 1.5e-8 off a 60-digit evaluation; panels at 1e11 and beyond returned numbers (a negative trace
/// statistic, an eigenvalue of -124) that are not eigenvalues of anything.
pub const MAX_CONDITION: f64 = 1e-8 / f64::EPSILON;

/// `(max l_ii / min l_ii)^2` of a Cholesky factor: the condition number of `L L'` to within a
/// small factor.
fn condition_estimate(l: &Cholesky) -> f64 {
    let l = l.l();
    let d: Vec<f64> = (0..l.rows()).map(|i| l[(i, i)]).collect();
    let hi = d.iter().cloned().fold(0.0, f64::max);
    let lo = d.iter().cloned().fold(f64::INFINITY, f64::min);
    (hi / lo).powi(2)
}

fn degenerate(e: impl std::fmt::Display) -> StatsError {
    StatsError::DegenerateSeries(format!(
        "Johansen's moment matrices are singular to working precision: {e}"
    ))
}

/// The columns of `a` as vectors.
fn columns(a: &Mat) -> Vec<Vec<f64>> {
    (0..a.cols()).map(|j| a.col(j)).collect()
}

fn from_columns(rows: usize, cols: &[Vec<f64>]) -> Result<Mat, StatsError> {
    let k = cols.len();
    let mut data = vec![0.0; rows * k];
    for (j, c) in cols.iter().enumerate() {
        for i in 0..rows {
            data[i * k + j] = c[i];
        }
    }
    Ok(Mat::from_vec(rows, k, data)?)
}

/// `y - X b` for the least-squares `b`, column by column: the residual of `y` after projecting
/// it on the columns of `x` (rank deficiency is harmless: the projection is the same).
fn project_out(y: &Mat, x: &Mat) -> Result<Mat, StatsError> {
    let (m, k) = (x.rows(), x.cols());
    let qr = qr_pivoted(x)?;
    let mut out = Vec::with_capacity(y.cols());
    for j in 0..y.cols() {
        let col = y.col(j);
        let b = qr.solve_basic(&col)?.x;
        out.push(
            (0..m)
                .map(|i| col[i] - (0..k).map(|c| x[(i, c)] * b[c]).sum::<f64>())
                .collect::<Vec<f64>>(),
        );
    }
    from_columns(m, &out)
}

/// Removes a polynomial trend of `order` (`np.vander(np.linspace(-1, 1, n), order + 1)`) from
/// each column; a negative order leaves the data alone.
fn detrend(a: &Mat, order: i32) -> Result<Mat, StatsError> {
    if order < 0 {
        return Ok(a.clone());
    }
    let n = a.rows();
    let step = 2.0 / (n - 1) as f64;
    let grid: Vec<f64> =
        (0..n).map(|i| if i == n - 1 { 1.0 } else { -1.0 + i as f64 * step }).collect();
    let cols: Vec<Vec<f64>> =
        (0..=order).rev().map(|p| grid.iter().map(|g| g.powi(p)).collect()).collect();
    project_out(a, &from_columns(n, &cols)?)
}

/// `L^-1 b` for lower triangular `L`.
fn forward(l: &Cholesky, b: &[f64]) -> Vec<f64> {
    let l = l.l();
    let mut x = vec![0.0; b.len()];
    for i in 0..b.len() {
        let s: f64 = (0..i).map(|k| l[(i, k)] * x[k]).sum();
        x[i] = (b[i] - s) / l[(i, i)];
    }
    x
}

/// `L^-T b` for lower triangular `L`.
fn back_transposed(l: &Cholesky, b: &[f64]) -> Vec<f64> {
    let l = l.l();
    let n = b.len();
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let s: f64 = (i + 1..n).map(|k| l[(k, i)] * x[k]).sum();
        x[i] = (b[i] - s) / l[(i, i)];
    }
    x
}

/// `a'b / t`.
fn moment(a: &Mat, b: &Mat, t: f64) -> Result<Mat, StatsError> {
    let p = a.transpose().matmul(b)?;
    let data = p.as_slice().iter().map(|v| v / t).collect();
    Ok(Mat::from_vec(p.rows(), p.cols(), data)?)
}

/// Divides each column by its root-mean-square; returns the scales.
fn equilibrate(a: &Mat) -> Result<(Mat, Vec<f64>), StatsError> {
    let t = a.rows() as f64;
    let scales: Vec<f64> =
        (0..a.cols()).map(|j| (a.col(j).iter().map(|v| v * v).sum::<f64>() / t).sqrt()).collect();
    if scales.iter().any(|s| *s == 0.0 || !s.is_finite()) {
        return Err(StatsError::DegenerateSeries(
            "Johansen's residuals of a series are identically zero".to_string(),
        ));
    }
    let cols: Vec<Vec<f64>> =
        (0..a.cols()).map(|j| a.col(j).iter().map(|v| v / scales[j]).collect()).collect();
    Ok((from_columns(a.rows(), &cols)?, scales))
}

/// Johansen's test on the levels `m` (one series per column), as
/// `statsmodels.tsa.vector_ar.vecm.coint_johansen(m, det_order, k_ar_diff)`.
pub fn johansen(m: &Mat, det_order: i32, k_ar_diff: usize) -> Result<Johansen, StatsError> {
    if !(-1..=1).contains(&det_order) {
        return Err(StatsError::Invalid(format!(
            "det_order {det_order}: critical values exist for -1, 0 and 1"
        )));
    }
    let (n, q) = (m.rows(), m.cols());
    if q > MAX_SERIES {
        return Err(StatsError::Invalid(format!("{q} series, at most {MAX_SERIES}")));
    }
    let k = k_ar_diff;
    let needed = q * k + q + k + 2;
    if n < needed {
        return Err(StatsError::InsufficientObservations {
            what: "series".to_string(),
            found: n,
            needed,
        });
    }
    // statsmodels detrends the levels by det_order, then the differences, lags and lagged
    // levels by a constant (f = 0) unless there is no deterministic term at all
    let f = if det_order > -1 { 0 } else { det_order };
    let endog = detrend(m, det_order)?;
    let rows = n - 1 - k;
    let dx = Mat::from_vec(
        n - 1,
        q,
        (0..(n - 1) * q).map(|p| endog[(p / q + 1, p % q)] - endog[(p / q, p % q)]).collect(),
    )?;
    let dx_tail = Mat::from_vec(rows, q, dx.as_slice()[k * q..].to_vec())?;
    let r0t_in = detrend(&dx_tail, f)?;
    let lx = Mat::from_vec(rows, q, endog.as_slice()[q..(rows + 1) * q].to_vec())?;
    let rkt_in = detrend(&lx, f)?;
    let (r0t, rkt) = if k == 0 {
        (r0t_in, rkt_in)
    } else {
        // the lagged differences: row r holds d x_{k+r-1}, .., d x_r
        let z = Mat::from_vec(
            rows,
            q * k,
            (0..rows * q * k)
                .map(|p| {
                    let (r, c) = (p / (q * k), p % (q * k));
                    dx[(k + r - 1 - c / q, c % q)]
                })
                .collect(),
        )?;
        let z = detrend(&z, f)?;
        (project_out(&r0t_in, &z)?, project_out(&rkt_in, &z)?)
    };
    let t = rkt.rows() as f64;
    let (r0s, _) = equilibrate(&r0t)?;
    let (rks, sk) = equilibrate(&rkt)?;
    let skk = moment(&rks, &rks, t)?;
    let sk0 = moment(&rks, &r0s, t)?;
    let s00 = moment(&r0s, &r0s, t)?;
    let l = cholesky(&skk).map_err(degenerate)?;
    let l0 = cholesky(&s00).map_err(degenerate)?;
    for (name, f) in [("S_kk", &l), ("S_00", &l0)] {
        let cond = condition_estimate(f);
        if cond > MAX_CONDITION {
            return Err(StatsError::DegenerateSeries(format!(
                "Johansen's {name} has condition number about {cond:.1e}, above {MAX_CONDITION:.1e}: \
                 the eigenvalues cannot be found to 1e-8 in double precision"
            )));
        }
    }
    // C = (L^-1 S_k0 L0^-T)(..)'
    let y_cols: Vec<Vec<f64>> = columns(&sk0).iter().map(|c| forward(&l, c)).collect();
    let y = from_columns(q, &y_cols)?;
    let g: Vec<Vec<f64>> = (0..q).map(|i| forward(&l0, y.row(i))).collect();
    let mut c = Mat::zeros(q, q)?;
    for i in 0..q {
        for j in i..q {
            let v: f64 = g[i].iter().zip(&g[j]).map(|(a, b)| a * b).sum();
            c[(i, j)] = v;
            c[(j, i)] = v;
        }
    }
    let eigen = sym_eigen(&c)?;
    let eig = eigen.values;
    if eig.iter().any(|a| !a.is_finite() || *a >= 1.0) {
        return Err(StatsError::DegenerateSeries(
            "Johansen's canonical correlations reach 1: the series are exactly dependent"
                .to_string(),
        ));
    }
    let mut cols = Vec::with_capacity(q);
    for j in 0..q {
        let w = eigen.vectors.col(j);
        let mut d: Vec<f64> = back_transposed(&l, &w).iter().zip(&sk).map(|(v, s)| v / s).collect();
        let big = (0..q).fold(0, |b, i| if d[i].abs() > d[b].abs() { i } else { b });
        if d[big] < 0.0 {
            d.iter_mut().for_each(|v| *v = -*v);
        }
        cols.push(d);
    }
    let evec = from_columns(q, &cols)?;
    let (mut lr1, mut lr2) = (Vec::with_capacity(q), Vec::with_capacity(q));
    let (mut cvt, mut cvm) = (Vec::with_capacity(q), Vec::with_capacity(q));
    let det = (det_order + 1) as usize;
    for i in 0..q {
        lr1.push(-t * eig[i..].iter().map(|a| (1.0 - a).ln()).sum::<f64>());
        lr2.push(-t * (1.0 - eig[i]).ln());
        cvt.push(tables::JOHANSEN_TRACE[det][q - i - 1]);
        cvm.push(tables::JOHANSEN_MAX_EIG[det][q - i - 1]);
    }
    Ok(Johansen { eig, evec, lr1, lr2, cvt, cvm, nobs: rows })
}

// ----------------------------------------------------------------------------- together

/// The Johansen part of the result.
#[derive(Debug, Clone, PartialEq)]
pub struct JohansenSummary {
    /// Hypotheses rejected in turn at 5% by the raw trace statistic against the asymptotic
    /// critical values, until one is not. The oracle pins it; it over-rejects at small `n / q`
    /// (see `rank`).
    pub rank_at_5pct: usize,
    pub trace_statistics: Vec<f64>,
    pub trace_critical_5pct: Vec<f64>,
    /// The same rule on the Reinsel-Ahn scaled statistics: the rank to read.
    pub rank: usize,
    /// `trace_statistics * scale_factor`.
    pub trace_statistics_scaled: Vec<f64>,
    /// Reinsel and Ahn's `(T - q k) / T`, `T` the observations used and `k = k_ar_diff + 1` the
    /// order of the VAR in levels.
    pub scale_factor: f64,
    /// Observations the statistics were computed on.
    pub effective_observations: usize,
    /// True when the correction removes more than a tenth of the statistic (`T < 20 q` for the
    /// one lagged difference used): the critical values are then far from the finite-sample
    /// distribution and even the scaled rank is a rough guide.
    pub small_sample: bool,
    /// The first eigenvector over its first entry (the raw vector when that entry is 0).
    pub cointegrating_vector: Vec<f64>,
}

/// The cointegration result: Engle-Granger and Johansen together.
#[derive(Debug, Clone, PartialEq)]
pub struct Cointegration {
    pub n: usize,
    pub alpha: f64,
    pub engle_granger: EngleGranger,
    pub johansen: JohansenSummary,
}

/// Engle-Granger (the first series dependent) and Johansen (constant in the relation, one lagged
/// difference) on the levels `cols`, read at `alpha`.
pub fn cointegration(cols: &[&[f64]], alpha: f64) -> Result<Cointegration, StatsError> {
    if cols.len() > MAX_SERIES {
        return Err(StatsError::Invalid(format!("{} series, at most {MAX_SERIES}", cols.len())));
    }
    if cols.len() < 2 {
        return Err(StatsError::Invalid(format!(
            "series: {} series, this statistic needs 2",
            cols.len()
        )));
    }
    for (i, c) in cols.iter().enumerate() {
        let what = format!("series[{i}]");
        if c.len() < MIN_OBS {
            return Err(StatsError::InsufficientObservations {
                what,
                found: c.len(),
                needed: MIN_OBS,
            });
        }
        require_finite(c, &what)?;
    }
    let n = cols[0].len();
    if cols.iter().any(|c| c.len() != n) {
        return Err(StatsError::Invalid("series: the series differ in length".to_string()));
    }
    if let Some(i) = cols.iter().position(|c| is_constant(c)) {
        return Err(StatsError::DegenerateSeries(format!("series[{i}] is constant")));
    }
    let eg = engle_granger(cols[0], &cols[1..], alpha)?;
    let levels = from_columns(n, &cols.iter().map(|c| c.to_vec()).collect::<Vec<_>>())?;
    let jo = johansen(&levels, 0, KAR_DIFF)?;
    let q = cols.len();
    let rank_of =
        |stats: &[f64]| stats.iter().zip(&jo.cvt).take_while(|(stat, cv)| **stat > cv[1]).count();
    let rank = rank_of(&jo.lr1);
    // Reinsel and Ahn (1992): the asymptotic critical values are for T -> infinity at fixed q;
    // `(T - q k) / T` removes the bias of the statistic when q k is not small against T.
    let t_obs = jo.nobs;
    let k_var = KAR_DIFF + 1;
    let scale_factor = (t_obs as f64 - (q * k_var) as f64).max(0.0) / t_obs as f64;
    let trace_scaled: Vec<f64> = jo.lr1.iter().map(|v| v * scale_factor).collect();
    let first = jo.evec.col(0);
    let vector = if first[0] != 0.0 { first.iter().map(|v| v / first[0]).collect() } else { first };
    Ok(Cointegration {
        n,
        alpha,
        engle_granger: eg,
        johansen: JohansenSummary {
            rank_at_5pct: rank,
            trace_statistics: jo.lr1,
            trace_critical_5pct: jo.cvt.iter().map(|r| r[1]).collect(),
            rank: rank_of(&trace_scaled),
            trace_statistics_scaled: trace_scaled,
            scale_factor,
            effective_observations: t_obs,
            small_sample: scale_factor < SMALL_SAMPLE_SCALE,
            cointegrating_vector: vector,
        },
    })
}
