//! Conditional volatility: GARCH(1,1) or GJR-GARCH(1,1,1) with a constant mean, fitted by
//! maximum likelihood under normal or standardised Student-t innovations; the persistence, the
//! half-life of a shock, the forecast variance path and the robust (sandwich) standard errors.
//!
//! Result fields hold full precision in the units the caller passed (returns as fractions); a
//! caller that displays them rounds. The fit is on returns in percent, so `log_likelihood` and `aic`
//! are those of the percent series.
//!
//! What is the same as the reference fit (the `arch` package, NCSA licence, read for its
//! behaviour; none of its code is here):
//! - the variance recursion `s_t = omega + alpha e_{t-1}^2 + gamma e_{t-1}^2 [e_{t-1} < 0] +
//!   beta s_{t-1}`, started from a backcast, the 0.94-EWMA of the first 75 squared residuals
//!   (weights `0.94^i`, normalised), which the shock terms and `s_{-1}` all take;
//! - the conditional variance kept between loose bounds built from an EWMA of the residuals at
//!   the starting mean (past the upper bound it grows by the log of the excess, so the
//!   likelihood stays finite);
//! - the start: a 4 x 4 x 4 grid of (alpha, gamma, alpha + beta) scored by the Gaussian
//!   likelihood, the first maximum in grid order, the mean at the sample mean, and for the t the
//!   moment estimate `max((4k - 6) / (k - 3), 4)` of the shape from the kurtosis of the
//!   standardised residuals at that start (12 when `k <= 3.75`);
//! - the shape of the t bounded to 2.05..500, and `variance = omega / (1 - persistence)`;
//! - the analytic forecast `s_{T+1} = omega + alpha e_T^2 + gamma e_T^2 [e_T < 0] + beta s_T`,
//!   then `s_{T+h+1} = omega + persistence s_{T+h}`.
//!
//! What is different, and why:
//! - the optimiser is BFGS with an analytic gradient on unconstrained coordinates, not SLSQP:
//!   `(alpha, beta)` (and for GJR `(alpha / 2, (alpha + gamma) / 2, beta)`) are the weights of a
//!   softmax whose last weight is the slack `1 - persistence`, so `alpha, beta >= 0`,
//!   `alpha + gamma >= 0` and `persistence < 1` hold at every step. A softmax only approaches the
//!   boundary (the IGARCH edge, `alpha = 0`): where the data want the boundary the fit stops
//!   within the gradient tolerance of it, and the likelihood is the maximum's to that tolerance.
//!   `omega` is `exp` of a coordinate, the t shape a logistic map onto 2.05..500, and the upper
//!   bound `alpha <= 1` and the lower bounds on `omega` are not enforced (they never bind on
//!   returns);
//! - a series whose percent variance is outside `0.1..10000` is rescaled by a power of ten
//!   (arch's rule) and every result is returned in the original units, so a 1e-7-scale series
//!   fits where the reference did not converge. The likelihood moves by `n ln(scale)` back;
//! - the robust standard errors are returned (the reference does not return them): `H^-1 S H^-1 / n`, with
//!   `H` the central difference of the analytic gradient and `S` the covariance of the per-bar
//!   scores. `None` means that `H` could not be inverted.
//!
//! Deterministic: one thread, fixed order of operations, no randomness.

use thiserror::Error;

use crate::math::linalg::{inverse, Mat};
use crate::math::optimize::{bfgs, BfgsOptions};

/// Fewest returns that are fitted.
pub const MIN_OBS: usize = 250;
/// Longest forecast horizon, in bars.
pub const MAX_HORIZON: usize = 60;
/// Returns are fitted in percent.
pub const SCALE: f64 = 100.0;
/// Bars of `recent_conditional_vol`.
pub const TIMELINE_MAX: usize = 60;

const BACKCAST_BARS: usize = 75;
const EWMA_LAMBDA: f64 = 0.94;
const NU_MIN: f64 = 2.05;
const NU_MAX: f64 = 500.0;
const LN_2PI: f64 = 1.837_877_066_409_345_3;
/// BFGS runs from the previous end point before giving up (each resets the inverse Hessian).
const MAX_RESTARTS: usize = 6;

/// Variance model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Model {
    Garch,
    Gjr,
}

/// Innovation distribution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dist {
    Normal,
    StudentT,
}

/// The model, innovation distribution and forecast horizon of a fit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GarchOptions {
    pub model: Model,
    pub dist: Dist,
    /// Bars to forecast, 1 to [`MAX_HORIZON`].
    pub horizon: usize,
    /// Annualises the volatilities when positive.
    pub periods_per_year: Option<u32>,
}

impl Default for GarchOptions {
    fn default() -> Self {
        GarchOptions {
            model: Model::Garch,
            dist: Dist::StudentT,
            horizon: 10,
            periods_per_year: None,
        }
    }
}

/// Why the fit was refused.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum GarchError {
    #[error("horizon must be 1 to {MAX_HORIZON}")]
    BadHorizon { horizon: usize },
    #[error("returns: {found} observations, this statistic needs {needed}")]
    InsufficientObservations { found: usize, needed: usize },
    #[error("returns holds a NaN or infinite value")]
    NonFinite,
    #[error("returns are constant")]
    DegenerateSeries,
    #[error("the GARCH fit did not converge: use a longer or cleaner sample")]
    DidNotConverge,
}

impl GarchError {
    /// The error code.
    pub fn code(&self) -> &'static str {
        match self {
            GarchError::BadHorizon { .. } | GarchError::NonFinite => "invalid_argument",
            GarchError::InsufficientObservations { .. } => "insufficient_observations",
            GarchError::DegenerateSeries => "degenerate_series",
            GarchError::DidNotConverge => "did_not_converge",
        }
    }
}

/// Fitted parameters in the units of the returns passed in (`mu` a return, `omega` a squared
/// return); `alpha`, `gamma`, `beta` and `nu` are pure numbers.
#[derive(Debug, Clone, PartialEq)]
pub struct GarchParameters {
    pub mu: f64,
    pub omega: f64,
    pub alpha: f64,
    /// GJR only.
    pub gamma: Option<f64>,
    pub beta: f64,
    /// Student-t only.
    pub nu: Option<f64>,
}

/// Robust standard errors, same fields and units as [`GarchParameters`].
#[derive(Debug, Clone, PartialEq)]
pub struct GarchStdErrors {
    pub mu: f64,
    pub omega: f64,
    pub alpha: f64,
    pub gamma: Option<f64>,
    pub beta: f64,
    pub nu: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GarchResult {
    pub n: usize,
    pub model: &'static str,
    pub distribution: &'static str,
    /// Always true: a fit that did not converge is an error.
    pub converged: bool,
    pub parameters: GarchParameters,
    /// Robust (sandwich) standard errors, new. `None`: the Hessian could not be inverted.
    pub std_errors: Option<GarchStdErrors>,
    /// The robust covariance in the order of `parameter_names`, row-major, same units as
    /// `parameters`. `None` with `std_errors`.
    pub covariance_robust: Option<Vec<Vec<f64>>>,
    pub parameter_names: Vec<&'static str>,
    pub persistence: f64,
    /// `persistence < 1`
    pub stationary_variance: bool,
    pub half_life_periods: Option<f64>,
    pub unconditional_vol_per_period: Option<f64>,
    /// Of the series in percent.
    pub log_likelihood: f64,
    pub aic: f64,
    pub last_vol_per_period: f64,
    pub forecast_vol_per_period: Vec<f64>,
    pub forecast_cumulative_vol: f64,
    pub recent_conditional_vol: Vec<f64>,
    pub last_vol_annualised: Option<f64>,
    pub forecast_vol_annualised: Option<Vec<f64>>,
    pub periods_per_year: Option<u32>,
    /// The power of ten the percent series was multiplied by to fit (1.0: none).
    pub rescale: f64,
}

// ------------------------------------------------------------------------------ the likelihood

/// Everything one pass over the series gives.
struct Eval {
    /// Negative log-likelihood (sum).
    nll: f64,
    /// Its gradient in natural parameters.
    grad: Vec<f64>,
    sigma2: Vec<f64>,
    /// Per-bar scores of the negative log-likelihood, row-major `n x k`; empty unless asked.
    scores: Vec<f64>,
}

struct Fitter<'a> {
    y: &'a [f64],
    gjr: bool,
    backcast: f64,
    lo: Vec<f64>,
    hi: Vec<f64>,
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

fn variance(v: &[f64]) -> f64 {
    let m = mean(v);
    v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / v.len() as f64
}

/// The 0.94-EWMA of the first 75 squared residuals, weights `0.94^i` normalised.
fn backcast_of(e: &[f64]) -> f64 {
    let tau = BACKCAST_BARS.min(e.len());
    let mut wsum = 0.0;
    let mut acc = 0.0;
    let mut w = 1.0;
    for x in e.iter().take(tau) {
        acc += w * x * x;
        wsum += w;
        w *= EWMA_LAMBDA;
    }
    acc / wsum
}

/// Lower and upper bounds of each conditional variance, from the residuals at the starting mean.
fn variance_bounds(e: &[f64], backcast: f64) -> (Vec<f64>, Vec<f64>) {
    let n = e.len();
    let mut ewma = vec![0.0; n];
    ewma[0] = backcast;
    for t in 1..n {
        ewma[t] = (1.0 - EWMA_LAMBDA) * e[t - 1] * e[t - 1] + EWMA_LAMBDA * ewma[t - 1];
    }
    let max_sq = e.iter().fold(0.0f64, |m, x| m.max(x * x));
    let min_upper = 1.0 + max_sq;
    let lower_floor = variance(e) / 1e8;
    let upper_cap = 1e7 * (1.0 + max_sq);
    let mut lo = Vec::with_capacity(n);
    let mut hi = Vec::with_capacity(n);
    for &v in &ewma {
        let mut l = v / 1e6;
        let mut h = v * 1e6;
        if l < lower_floor {
            l = lower_floor;
        }
        if h < min_upper {
            h = min_upper;
        }
        if h > upper_cap {
            h = upper_cap;
        }
        lo.push(l);
        hi.push(h);
    }
    (lo, hi)
}

/// The digamma function: recurrence up to 10, then DLMF 5.11.2.
fn digamma(mut x: f64) -> f64 {
    let mut acc = 0.0;
    while x < 10.0 {
        acc -= 1.0 / x;
        x += 1.0;
    }
    let i2 = 1.0 / (x * x);
    acc + x.ln()
        - 0.5 / x
        - i2 * (1.0 / 12.0
            - i2 * (1.0 / 120.0 - i2 * (1.0 / 252.0 - i2 * (1.0 / 240.0 - i2 * (1.0 / 132.0)))))
}

/// `ln G(a + 1/2) - ln G(a)` for `a > 1/2`, computed as one smooth quantity: the two `ln_gamma`
/// values are each good to 1e-14 absolute at `a` near 30, which is noise the line search of a
/// 1e-8 gradient tolerance cannot see through. Shifts `a` up to `b >= 15` (the shift terms are
/// ratios), then Stirling's series for the difference (DLMF 5.11.1).
fn lgamma_half_step(a: f64) -> f64 {
    let mut shift = 0.0;
    let mut b = a;
    while b < 15.0 {
        shift += ((b + 0.5) / b).ln();
        b += 1.0;
    }
    let stirling = |z: f64| {
        let i2 = 1.0 / (z * z);
        (1.0 / 12.0
            - i2 * (1.0 / 360.0
                - i2 * (1.0 / 1260.0
                    - i2 * (1.0 / 1680.0 - i2 * (1.0 / 1188.0 - i2 * (691.0 / 360360.0))))))
            / z
    };
    b * (0.5 / b).ln_1p() + 0.5 * b.ln() - 0.5 + stirling(b + 0.5) - stirling(b) - shift
}

impl Fitter<'_> {
    fn k(&self, t: bool) -> usize {
        4 + usize::from(self.gjr) + usize::from(t)
    }

    /// Natural parameters, in order: `mu, omega, alpha, [gamma], beta, [nu]`.
    fn eval(&self, th: &[f64], t_dist: bool, want_scores: bool) -> Eval {
        let n = self.y.len();
        let g = usize::from(self.gjr);
        let (mu, om, al) = (th[0], th[1], th[2]);
        let ga = if self.gjr { th[3] } else { 0.0 };
        let be = th[3 + g];
        let nu = if t_dist { th[4 + g] } else { f64::NAN };
        let k = th.len();
        let (c_nu, psi_a, psi_b) = if t_dist {
            (
                lgamma_half_step(nu / 2.0) - 0.5 * (std::f64::consts::PI * (nu - 2.0)).ln(),
                digamma((nu + 1.0) / 2.0),
                digamma(nu / 2.0),
            )
        } else {
            (0.0, 0.0, 0.0)
        };
        let mut sigma2 = Vec::with_capacity(n);
        let mut scores = Vec::new();
        if want_scores {
            scores.reserve(n * k);
        }
        // gradient of the log-likelihood in [mu, omega, alpha, gamma, beta, nu]
        let mut gsum = [0.0f64; 6];
        let mut ll_sum = 0.0;
        let mut prev_d = [0.0f64; 5];
        let mut prev_s = 0.0;
        for t in 0..n {
            let e = self.y[t] - mu;
            // d holds the derivative of s_t in [mu, omega, alpha, gamma, beta]
            let (raw, d) = if t == 0 {
                let bc = self.backcast;
                (om + al * bc + ga * 0.5 * bc + be * bc, [0.0, 1.0, bc, 0.5 * bc, bc])
            } else {
                let e1 = self.y[t - 1] - mu;
                let e2 = e1 * e1;
                let neg = if e1 < 0.0 { 1.0 } else { 0.0 };
                (
                    om + al * e2 + ga * e2 * neg + be * prev_s,
                    [
                        -2.0 * (al + ga * neg) * e1 + be * prev_d[0],
                        1.0 + be * prev_d[1],
                        e2 + be * prev_d[2],
                        e2 * neg + be * prev_d[3],
                        prev_s + be * prev_d[4],
                    ],
                )
            };
            let (s, d) = if raw < self.lo[t] {
                (self.lo[t], [0.0; 5])
            } else if raw > self.hi[t] {
                if raw.is_infinite() {
                    (self.hi[t] + 1000.0, [0.0; 5])
                } else {
                    (self.hi[t] + (raw / self.hi[t]).ln(), d.map(|v| v / raw))
                }
            } else {
                (raw, d)
            };
            sigma2.push(s);
            prev_d = d;
            prev_s = s;
            // the observation's log-likelihood and its derivatives in s, e (and nu)
            let (ll, dls, dle, dnu);
            if t_dist {
                let q = e * e / (s * (nu - 2.0));
                let l1 = q.ln_1p();
                ll = c_nu - 0.5 * s.ln() - (nu + 1.0) / 2.0 * l1;
                dls = -0.5 / s + (nu + 1.0) / 2.0 * q / s / (1.0 + q);
                dle = -(nu + 1.0) / 2.0 * (2.0 * e / (s * (nu - 2.0))) / (1.0 + q);
                dnu = 0.5 * psi_a - 0.5 * psi_b - 0.5 / (nu - 2.0) - 0.5 * l1
                    + (nu + 1.0) / 2.0 * q / (nu - 2.0) / (1.0 + q);
            } else {
                ll = -0.5 * (LN_2PI + s.ln() + e * e / s);
                dls = -0.5 * (1.0 / s - e * e / (s * s));
                dle = -e / s;
                dnu = 0.0;
            }
            ll_sum += ll;
            let row = [dls * d[0] - dle, dls * d[1], dls * d[2], dls * d[3], dls * d[4], dnu];
            for (a, b) in gsum.iter_mut().zip(row) {
                *a += b;
            }
            if want_scores {
                scores.extend(self.natural(row, t_dist).into_iter().map(|v| -v));
            }
        }
        Eval {
            nll: -ll_sum,
            grad: self.natural(gsum, t_dist).into_iter().map(|v| -v).collect(),
            sigma2,
            scores,
        }
    }

    /// `[mu, omega, alpha, gamma, beta, nu]` reduced to the parameters of the model.
    fn natural(&self, v: [f64; 6], t_dist: bool) -> Vec<f64> {
        let mut out = vec![v[0], v[1], v[2]];
        if self.gjr {
            out.push(v[3]);
        }
        out.push(v[4]);
        if t_dist {
            out.push(v[5]);
        }
        out
    }
}

/// The log-likelihood and its gradient (natural parameters `mu, omega, alpha, [gamma], beta,
/// [nu]`) of a series **in percent**, with the backcast and variance bounds built, as the fit
/// builds them, from the residuals at the sample mean. For checking the gradient against
/// differences and the likelihood against a hand calculation.
pub fn log_likelihood_gradient(
    returns_percent: &[f64],
    model: Model,
    dist: Dist,
    params: &[f64],
) -> (f64, Vec<f64>) {
    let fitter = fitter_for(returns_percent, model);
    let t = dist == Dist::StudentT;
    assert_eq!(params.len(), fitter.k(t), "params: wrong length for the model");
    let ev = fitter.eval(params, t, false);
    (-ev.nll, ev.grad.into_iter().map(|v| -v).collect())
}

fn fitter_for(y: &[f64], model: Model) -> Fitter<'_> {
    let m = mean(y);
    let e: Vec<f64> = y.iter().map(|v| v - m).collect();
    let backcast = backcast_of(&e);
    let (lo, hi) = variance_bounds(&e, backcast);
    Fitter { y, gjr: model == Model::Gjr, backcast, lo, hi }
}

// ----------------------------------------------------------------------------- the coordinates

/// Unconstrained coordinates `x`: `[mu, ln omega, logits.., eta]`. The logits are the log-odds
/// against the slack of the weights of `(alpha, beta)` or `(alpha / 2, (alpha + gamma) / 2,
/// beta)`; `eta` maps the shape onto `NU_MIN..NU_MAX`.
struct Coords {
    gjr: bool,
    t: bool,
}

impl Coords {
    fn len(&self) -> usize {
        4 + usize::from(self.gjr) + usize::from(self.t)
    }

    fn weights(&self, x: &[f64]) -> Vec<f64> {
        let nl = if self.gjr { 3 } else { 2 };
        let mut z = vec![0.0; nl + 1];
        z[1..].copy_from_slice(&x[2..2 + nl]);
        let top = z.iter().fold(f64::NEG_INFINITY, |m, v| m.max(*v));
        let e: Vec<f64> = z.iter().map(|v| (v - top).exp()).collect();
        let s: f64 = e.iter().sum();
        e.into_iter().map(|v| v / s).collect()
    }

    fn logistic(v: f64) -> f64 {
        1.0 / (1.0 + (-v).exp())
    }

    /// Natural parameters and the softmax weights `[slack, ..]`.
    fn to_natural(&self, x: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let w = self.weights(x);
        let mut th = vec![x[0], x[1].exp()];
        if self.gjr {
            let alpha = 2.0 * w[1];
            th.push(alpha);
            th.push(2.0 * w[2] - alpha);
            th.push(w[3]);
        } else {
            th.push(w[1]);
            th.push(w[2]);
        }
        if self.t {
            th.push(NU_MIN + (NU_MAX - NU_MIN) * Self::logistic(x[x.len() - 1]));
        }
        (th, w)
    }

    fn to_coords(&self, th: &[f64]) -> Vec<f64> {
        let mut x = vec![th[0], th[1].ln()];
        if self.gjr {
            let (alpha, gamma, beta) = (th[2], th[3], th[4]);
            let (wa, wc, wb) = (alpha / 2.0, (alpha + gamma) / 2.0, beta);
            let ws = 1.0 - wa - wc - wb;
            x.extend([(wa / ws).ln(), (wc / ws).ln(), (wb / ws).ln()]);
        } else {
            let ws = 1.0 - th[2] - th[3];
            x.extend([(th[2] / ws).ln(), (th[3] / ws).ln()]);
        }
        if self.t {
            let u = (th[th.len() - 1] - NU_MIN) / (NU_MAX - NU_MIN);
            x.push((u / (1.0 - u)).ln());
        }
        x
    }

    /// The gradient in coordinates from the gradient of the same function in natural parameters.
    fn chain(&self, x: &[f64], w: &[f64], gn: &[f64]) -> Vec<f64> {
        let mut gx = vec![0.0; x.len()];
        gx[0] = gn[0];
        gx[1] = gn[1] * x[1].exp();
        // derivative in each weight (slack first, which no parameter reads)
        let dw: Vec<f64> = if self.gjr {
            vec![0.0, 2.0 * gn[2] - 2.0 * gn[3], 2.0 * gn[3], gn[4]]
        } else {
            vec![0.0, gn[2], gn[3]]
        };
        let dot: f64 = dw.iter().zip(w).map(|(a, b)| a * b).sum();
        for j in 1..w.len() {
            gx[1 + j] = w[j] * (dw[j] - dot);
        }
        if self.t {
            let l = Self::logistic(x[x.len() - 1]);
            let last = gx.len() - 1;
            gx[last] = gn[gn.len() - 1] * (NU_MAX - NU_MIN) * l * (1.0 - l);
        }
        gx
    }
}

// ------------------------------------------------------------------------------------ the start

/// The arch start: the best of the 64 (alpha, gamma, alpha + beta) grid points by the Gaussian
/// likelihood, then the shape from the kurtosis.
fn start_values(fitter: &Fitter<'_>, t_dist: bool) -> Vec<f64> {
    let y = fitter.y;
    let mu0 = mean(y);
    let target = {
        let e: Vec<f64> = y.iter().map(|v| v - mu0).collect();
        e.iter().map(|v| v * v).sum::<f64>() / e.len() as f64
    };
    let alphas = [0.01, 0.05, 0.1, 0.2];
    let abg = [0.5, 0.7, 0.9, 0.98];
    let gammas: &[f64] = if fitter.gjr { &alphas } else { &[0.0] };
    let mut best: Option<(f64, Vec<f64>)> = None;
    for &a in &alphas {
        for &gm in gammas {
            for &s in &abg {
                let mut beta = s - a;
                let mut th = vec![mu0, (1.0 - s) * target, a];
                if fitter.gjr {
                    beta -= gm / 2.0;
                    th.push(gm);
                }
                th.push(beta);
                let ll = -fitter.eval(&th, false, false).nll;
                // strict: the first maximum in grid order, as the reference
                if best.as_ref().is_none_or(|(b, _)| ll > *b) {
                    best = Some((ll, th));
                }
            }
        }
    }
    let mut th = best.map(|(_, th)| th).unwrap_or_default();
    if t_dist {
        let ev = fitter.eval(&th, false, false);
        let z: Vec<f64> = y.iter().zip(&ev.sigma2).map(|(v, s)| (v - mu0) / s.sqrt()).collect();
        let zm = mean(&z);
        let m2 = z.iter().map(|v| (v - zm) * (v - zm)).sum::<f64>() / z.len() as f64;
        let m4 = z.iter().map(|v| (v - zm).powi(4)).sum::<f64>() / z.len() as f64;
        let kurt = m4 / (m2 * m2);
        let nu = if kurt > 3.75 { (4.0 * kurt - 6.0) / (kurt - 3.0) } else { 12.0 };
        th.push(nu.max(4.0));
    }
    th
}

/// arch's rule: the power of ten that brings the variance of the residuals into `0.1..10000`.
fn rescale_factor(y: &[f64]) -> f64 {
    let orig = variance(y);
    let mut factor = 1.0f64;
    let mut scale = orig;
    while !(0.1..10000.0).contains(&scale) && scale > 0.0 {
        if scale < 1.0 {
            factor *= 10.0;
        } else {
            factor /= 10.0;
        }
        scale = orig * factor * factor;
    }
    factor
}

// ---------------------------------------------------------------------------------------- the fit

struct Fit {
    th: Vec<f64>,
    llf: f64,
    sigma2: Vec<f64>,
    cov: Option<Mat>,
}

fn fit_scaled(y: &[f64], model: Model, dist: Dist) -> Result<Fit, GarchError> {
    let t_dist = dist == Dist::StudentT;
    let fitter = fitter_for(y, model);
    let coords = Coords { gjr: fitter.gjr, t: t_dist };
    debug_assert_eq!(coords.len(), fitter.k(t_dist));
    let n = y.len() as f64;
    let th0 = start_values(&fitter, t_dist);
    let mut x = coords.to_coords(&th0);
    let mut converged = false;
    let opts = BfgsOptions::default();
    for _ in 0..MAX_RESTARTS {
        let res = bfgs(
            |x: &[f64], grad: &mut [f64]| {
                let (th, w) = coords.to_natural(x);
                let ev = fitter.eval(&th, t_dist, false);
                let gx = coords.chain(x, &w, &ev.grad);
                for (o, v) in grad.iter_mut().zip(gx) {
                    *o = v / n;
                }
                ev.nll / n
            },
            &x,
            &opts,
        )
        .map_err(|_| GarchError::DidNotConverge)?;
        x = res.x.clone();
        if res.converged() {
            converged = true;
            break;
        }
    }
    if !converged {
        return Err(GarchError::DidNotConverge);
    }
    let (th, _) = coords.to_natural(&x);
    let ev = fitter.eval(&th, t_dist, true);
    if !ev.nll.is_finite() {
        return Err(GarchError::DidNotConverge);
    }
    let cov = robust_covariance(&fitter, &th, t_dist, &ev);
    Ok(Fit { th, llf: -ev.nll, sigma2: ev.sigma2, cov })
}

/// `H^-1 S H^-1 / n`: `H` the central difference of the analytic gradient of the negative
/// log-likelihood (per bar), `S` the covariance (n - 1) of the per-bar scores.
fn robust_covariance(fitter: &Fitter<'_>, th: &[f64], t_dist: bool, ev: &Eval) -> Option<Mat> {
    let k = th.len();
    let n = fitter.y.len();
    let mut h = vec![0.0; k * k];
    for j in 0..k {
        let step = 1e-5 * th[j].abs().max(1e-2);
        let mut up = th.to_vec();
        let mut dn = th.to_vec();
        up[j] += step;
        dn[j] -= step;
        let gu = fitter.eval(&up, t_dist, false).grad;
        let gd = fitter.eval(&dn, t_dist, false).grad;
        for i in 0..k {
            h[i * k + j] = (gu[i] - gd[i]) / (2.0 * step);
        }
    }
    let mut sym = vec![0.0; k * k];
    for i in 0..k {
        for j in 0..k {
            sym[i * k + j] = 0.5 * (h[i * k + j] + h[j * k + i]) / n as f64;
        }
    }
    let hinv = inverse(&Mat::from_vec(k, k, sym).ok()?).ok()?;
    let mut means = vec![0.0; k];
    for row in ev.scores.chunks(k) {
        for (m, v) in means.iter_mut().zip(row) {
            *m += v;
        }
    }
    means.iter_mut().for_each(|m| *m /= n as f64);
    let mut s = vec![0.0; k * k];
    for row in ev.scores.chunks(k) {
        for i in 0..k {
            for j in 0..k {
                s[i * k + j] += (row[i] - means[i]) * (row[j] - means[j]);
            }
        }
    }
    s.iter_mut().for_each(|v| *v /= (n - 1) as f64);
    let smat = Mat::from_vec(k, k, s).ok()?;
    let cov = hinv.matmul(&smat).ok()?.matmul(&hinv).ok()?;
    let mut data = cov.as_slice().to_vec();
    data.iter_mut().for_each(|v| *v /= n as f64);
    // an estimator of a covariance: symmetric, with a non-negative diagonal
    for i in 0..k {
        for j in 0..i {
            let m = 0.5 * (data[i * k + j] + data[j * k + i]);
            data[i * k + j] = m;
            data[j * k + i] = m;
        }
        if data[i * k + i].is_nan() || data[i * k + i] < 0.0 {
            return None;
        }
    }
    Mat::from_vec(k, k, data).ok()
}

/// Fits the model to `returns` (fractions) and reports in the units passed in.
pub fn garch(returns: &[f64], opts: &GarchOptions) -> Result<GarchResult, GarchError> {
    if !(1..=MAX_HORIZON).contains(&opts.horizon) {
        return Err(GarchError::BadHorizon { horizon: opts.horizon });
    }
    if returns.len() < MIN_OBS {
        return Err(GarchError::InsufficientObservations { found: returns.len(), needed: MIN_OBS });
    }
    if !returns.iter().all(|v| v.is_finite()) {
        return Err(GarchError::NonFinite);
    }
    let (lo, hi) =
        returns.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), v| (l.min(*v), h.max(*v)));
    if hi - lo == 0.0 {
        return Err(GarchError::DegenerateSeries);
    }
    let n = returns.len();
    let pct: Vec<f64> = returns.iter().map(|v| v * SCALE).collect();
    let rescale = rescale_factor(&pct);
    let y: Vec<f64> = pct.iter().map(|v| v * rescale).collect();
    let gjr = opts.model == Model::Gjr;
    let t_dist = opts.dist == Dist::StudentT;
    let fit = fit_scaled(&y, opts.model, opts.dist)?;
    let th = &fit.th;

    // units: the fitted series is `returns * q`
    let q = SCALE * rescale;
    let g = usize::from(gjr);
    let (mu, omega_s, alpha) = (th[0], th[1], th[2]);
    let gamma_v = if gjr { th[3] } else { 0.0 };
    let beta = th[3 + g];
    let nu = if t_dist { Some(th[4 + g]) } else { None };
    let persistence = alpha + beta + gamma_v / 2.0;
    let omega = omega_s / (q * q);
    let (uncond, half) = if persistence < 1.0 {
        let u = omega_s / (1.0 - persistence) / (q * q);
        let h = if persistence > 0.0 { 0.5f64.ln() / persistence.ln() } else { 0.0 };
        (Some(u), Some(h))
    } else {
        (None, None)
    };

    // forecast from the last bar
    let e_last = y[n - 1] - mu;
    let neg = if e_last < 0.0 { 1.0 } else { 0.0 };
    let mut var_path = Vec::with_capacity(opts.horizon);
    var_path.push(
        omega_s
            + alpha * e_last * e_last
            + gamma_v * e_last * e_last * neg
            + beta * fit.sigma2[n - 1],
    );
    for h in 1..opts.horizon {
        var_path.push(omega_s + persistence * var_path[h - 1]);
    }
    let var_path: Vec<f64> = var_path.into_iter().map(|v| v / (q * q)).collect();
    let vol: Vec<f64> = fit.sigma2.iter().map(|v| v.sqrt() / q).collect();
    let ppy = opts.periods_per_year.filter(|p| *p > 0);
    let ann = ppy.map(|p| f64::from(p).sqrt());
    let forecast_vol: Vec<f64> = var_path.iter().map(|v| v.sqrt()).collect();
    let last_vol = vol[n - 1];

    let k = th.len();
    let llf = fit.llf + n as f64 * rescale.ln();
    // parameters, in the units of the returns passed in
    let mut names = vec!["mu", "omega", "alpha"];
    if gjr {
        names.push("gamma");
    }
    names.push("beta");
    if t_dist {
        names.push("nu");
    }
    let unit = |name: &str| match name {
        "mu" => 1.0 / q,
        "omega" => 1.0 / (q * q),
        _ => 1.0,
    };
    let (std_errors, covariance_robust) = match &fit.cov {
        Some(c) => {
            let scaled: Vec<Vec<f64>> = (0..k)
                .map(|i| (0..k).map(|j| c[(i, j)] * (unit(names[i]) * unit(names[j]))).collect())
                .collect();
            let se: Vec<f64> = (0..k).map(|i| scaled[i][i].sqrt()).collect();
            (
                Some(GarchStdErrors {
                    mu: se[0],
                    omega: se[1],
                    alpha: se[2],
                    gamma: gjr.then(|| se[3]),
                    beta: se[3 + g],
                    nu: t_dist.then(|| se[4 + g]),
                }),
                Some(scaled),
            )
        }
        None => (None, None),
    };

    Ok(GarchResult {
        n,
        model: if gjr { "GJR-GARCH(1,1,1)" } else { "GARCH(1,1)" },
        distribution: if t_dist { "t" } else { "normal" },
        converged: true,
        parameters: GarchParameters {
            mu: mu / q,
            omega,
            alpha,
            gamma: gjr.then_some(gamma_v),
            beta,
            nu,
        },
        std_errors,
        covariance_robust,
        parameter_names: names,
        persistence,
        stationary_variance: persistence < 1.0,
        half_life_periods: half,
        unconditional_vol_per_period: uncond.map(f64::sqrt),
        log_likelihood: llf,
        aic: 2.0 * k as f64 - 2.0 * llf,
        last_vol_per_period: last_vol,
        forecast_cumulative_vol: var_path.iter().sum::<f64>().sqrt(),
        forecast_vol_annualised: ann.map(|a| forecast_vol.iter().map(|v| v * a).collect()),
        forecast_vol_per_period: forecast_vol,
        recent_conditional_vol: vol[n.saturating_sub(TIMELINE_MAX)..].to_vec(),
        last_vol_annualised: ann.map(|a| last_vol * a),
        periods_per_year: ppy,
        rescale,
    })
}

#[cfg(test)]
mod tests {
    use super::digamma;

    #[test]
    fn digamma_known_values() {
        // psi(1) = -gamma, psi(1/2) = -gamma - 2 ln 2, psi(x + 1) = psi(x) + 1/x
        let euler = 0.577_215_664_901_532_9;
        assert!((digamma(1.0) + euler).abs() < 1e-13);
        assert!((digamma(0.5) + euler + 2.0 * 2.0f64.ln()).abs() < 1e-13);
        for x in [0.3, 1.7, 9.99, 10.0, 55.5, 250.0] {
            assert!((digamma(x + 1.0) - digamma(x) - 1.0 / x).abs() < 1e-12, "x={x}");
        }
    }
}
