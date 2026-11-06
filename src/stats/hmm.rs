//! Hidden Markov regimes: a Markov-switching mean and variance fitted to returns by maximum
//! likelihood (the model of statsmodels' `MarkovRegression(trend="c", switching_variance=True)`
//! that it follows), the probability of each regime at each bar, and the timeline.
//!
//! Result fields hold full precision; a caller that displays them rounds. The fit is on returns
//! in percent; means and volatilities are reported per period as fractions.
//!
//! # Two kinds of probability, one of them non-causal
//!
//! - `filtered_probabilities` (and `recent_filtered_probabilities`) are `P(regime_t | returns up
//!   to and including t)`: **causal**, what a live reader would have had at bar `t`. A strategy
//!   may read these.
//! - `smoothed_probabilities` (and `recent_probabilities`, `current_probabilities`,
//!   `regime_stats.share_of_periods`, `timeline`, `current_regime`) are
//!   `P(regime_t | all returns)`: **non-causal**. Bar `t`'s value uses returns after `t`. They
//!   describe the past; a strategy that read them as a signal would see the future. Only the
//!   last bar's smoothed and filtered values agree. [`SMOOTHED_NOTE`] says so in the result
//!   (`smoothed_is_causal` is false).
//!
//! `recent_probabilities` and `regimes` are the smoothed ones.
//!
//! # The model and the fit
//!
//! `y_t | s_t = i ~ N(mu_i, sigma2_i)`, `s_t` a first-order Markov chain with `P[to][from]`
//! column-stochastic, and `s_{-1}` distributed as the chain's steady state (as statsmodels'
//! default initialisation does). The likelihood is the Hamilton filter in probability space,
//! scaled at each bar (the densities are shifted by their maximum before `exp`, so a 30-sigma bar
//! does not underflow). The smoother is the exact backward recursion
//! `P(s_t = i | all) = a_t(i) * sum_m P[m][i] g_{t+1}(m) / pred_{t+1}(m)` (Kim 1994), whose
//! pairwise form also gives the expected transition counts.
//!
//! As the reference fit: start values of equal transition probabilities, means
//! `mean(y) * i / k` and variances `linspace(v / 10, v, k)` (`v` the variance of the series),
//! improved by **5 EM steps** (the exact steps of the reference, including its use of the
//! smoothed marginals over all `T` bars as the denominator of the transition update), then BFGS on
//! the likelihood. What is different:
//! - the BFGS gradient is analytic: by Fisher's identity it is the expected complete-data score
//!   under the smoothed probabilities, plus the derivative of the steady-state initial
//!   distribution, from one more pass of the smoother (checked against differences in the tests);
//! - the variances are optimised as `ln sigma2` (the reference uses `sqrt`), the transitions as
//!   multinomial-logit coordinates, as the reference does;
//! - **three starts**: the reference's, and two deterministic partitions of the bars into `k`
//!   equal groups by rank, by squared deviation from the mean (a variance split) and by return
//!   (a level split), each with persistence 0.9 and each through the same 5 EM steps. The best
//!   likelihood wins; a tie goes to the earlier start. No randomness anywhere;
//! - a softmax only approaches a transition probability of 0 or 1, where the coordinate's gradient
//!   vanishes before the likelihood is maximal: after a fit with a transition probability under
//!   0.02, the probabilities are lifted to 0.02 and the fit is repeated (at most twice), and the
//!   better of the two is kept;
//! - a fit with a regime variance under a thousandth of the series' variance is dropped: a regime
//!   that has collapsed onto a few bars sends the likelihood to infinity, which is not an answer.
//!   If every start collapses the error is [`HmmError::DegenerateFit`].
//!
//! # Low support
//!
//! Maximum likelihood can reward a regime fitted to a handful of bars. Each entry of
//! `regime_stats` carries `support_bars` (its smoothed probabilities summed) and `low_support`,
//! true when that is under `HmmOptions::min_support_bars` (default 10) or the expected duration
//! is under `min_duration_periods` (default 2). `low_support_regimes` lists the flagged indices.
//! The fit and the other keys are unchanged; the flag is for the reader.
//!
//! # Regime labels
//!
//! Regimes are ordered by variance, calmest first; equal variances by mean, then by the fit's own
//! order. That is `argsort(var)` made total. A label, a probability column and a
//! transition-matrix row of the same index describe the same regime.
//!
//! Deterministic: one thread, fixed order of operations, no randomness.

use thiserror::Error;

use crate::math::linalg::{inverse, Mat};
use crate::math::optimize::{bfgs, BfgsOptions};

/// Returns are fitted in percent.
pub const SCALE: f64 = 100.0;
/// Observations needed per regime beyond the first: the fit needs `200 * (k - 1)`.
pub const MIN_OBS_PER_REGIME: usize = 200;
/// Segments of the timeline kept (the latest).
pub const SEGMENTS_MAX: usize = 40;
/// Bars of `recent_probabilities`.
pub const PROB_TAIL: usize = 20;
/// The words that go with the smoothed probabilities.
pub const SMOOTHED_NOTE: &str = "smoothed probabilities are conditioned on the whole sample, so an earlier bar's value uses returns after it: not causal, never a signal. filtered probabilities use only returns up to the bar";

const EM_STEPS: usize = 5;
const PROB_FLOOR: f64 = 1e-20;
const PRED_FLOOR: f64 = 1e-300;
const LN_2PI: f64 = 1.837_877_066_409_345_3;
const SOFT_FLOOR: f64 = 0.02;
const POLISH_ROUNDS: usize = 2;
const MAX_RESTARTS: usize = 3;
const DEGENERATE_VARIANCE_RATIO: f64 = 1e-3;
const PARTITION_STAY: f64 = 0.9;
const STARTS: usize = 3;

/// Why the fit was refused.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum HmmError {
    #[error("k must be 2 or 3 regimes")]
    BadRegimes { regimes: usize },
    #[error("returns: {found} observations, this statistic needs {needed}")]
    InsufficientObservations { found: usize, needed: usize },
    #[error("returns holds a NaN or infinite value")]
    NonFinite,
    #[error("returns are constant")]
    DegenerateSeries,
    #[error("the regime fit did not converge: use a longer sample or fewer regimes")]
    DidNotConverge,
    #[error(
        "every start collapsed a regime onto a few bars: use a longer sample or fewer regimes"
    )]
    DegenerateFit,
}

impl HmmError {
    /// The error code.
    pub fn code(&self) -> &'static str {
        match self {
            HmmError::BadRegimes { .. } | HmmError::NonFinite => "invalid_argument",
            HmmError::InsufficientObservations { .. } => "insufficient_observations",
            HmmError::DegenerateSeries | HmmError::DegenerateFit => "degenerate_series",
            HmmError::DidNotConverge => "did_not_converge",
        }
    }
}

/// The number of regimes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HmmOptions {
    /// 2 or 3.
    pub regimes: usize,
    /// A regime whose smoothed probabilities sum to fewer bars than this is flagged
    /// `low_support`: it is a few bars, not a regime to act on.
    pub min_support_bars: f64,
    /// A regime whose expected duration (`1 / (1 - stay)`) is under this many bars is flagged
    /// `low_support`.
    pub min_duration_periods: f64,
}

impl Default for HmmOptions {
    fn default() -> Self {
        HmmOptions { regimes: 2, min_support_bars: 10.0, min_duration_periods: 2.0 }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RegimeStat {
    pub regime: usize,
    pub mean_per_period: f64,
    pub vol_per_period: f64,
    /// Share of bars whose smoothed argmax is this regime (non-causal).
    pub share_of_periods: f64,
    pub stay_probability: f64,
    /// `1 / (1 - stay)`; `None` when the regime never ends.
    pub expected_duration_periods: Option<f64>,
    /// Bars of evidence for the regime: its smoothed probabilities summed (non-causal).
    pub support_bars: f64,
    /// True when `support_bars < min_support_bars` or the expected duration is under
    /// `min_duration_periods`. Such a regime can be a spike on a few bars fitted at a higher
    /// likelihood (on data with no regimes the best fit can be a 7-bar regime), so a reader should
    /// not present it as a regime without saying so.
    pub low_support: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TimelineSegment {
    pub regime: usize,
    pub start_index: usize,
    pub end_index: usize,
    pub periods: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HmmResult {
    pub n: usize,
    pub regimes: usize,
    /// Always true: a fit that did not converge is an error.
    pub converged: bool,
    /// One per regime, calmest first; each carries its own `low_support` flag.
    pub regime_stats: Vec<RegimeStat>,
    /// The regimes flagged `low_support`, by index (empty when none is).
    pub low_support_regimes: Vec<usize>,
    /// `[from][to]`, regimes in the order of `regime_stats`.
    pub transition_matrix: Vec<Vec<f64>>,
    /// The smoothed argmax at the last bar. At the last bar smoothed and filtered agree.
    pub current_regime: usize,
    pub current_probabilities: Vec<f64>,
    /// SMOOTHED (non-causal), the last `PROB_TAIL` bars.
    pub recent_probabilities: Vec<Vec<f64>>,
    /// Filtered (causal), the last `PROB_TAIL` bars.
    pub recent_filtered_probabilities: Vec<Vec<f64>>,
    /// Runs of equal smoothed argmax over the whole series.
    pub timeline_total: usize,
    /// The latest `SEGMENTS_MAX` runs.
    pub timeline: Vec<TimelineSegment>,
    /// Of the series in percent.
    pub log_likelihood: f64,
    pub aic: f64,
    /// `n x regimes`, SMOOTHED: non-causal, see [`SMOOTHED_NOTE`].
    pub smoothed_probabilities: Vec<Vec<f64>>,
    /// `n x regimes`, filtered: causal.
    pub filtered_probabilities: Vec<Vec<f64>>,
    /// Always false. Says in the result what the doc says.
    pub smoothed_is_causal: bool,
    /// Always true.
    pub filtered_is_causal: bool,
    /// The fit of each of the three starts: its log-likelihood, or `None` if the start failed to
    /// converge or collapsed a regime.
    pub start_log_likelihoods: Vec<Option<f64>>,
    pub winning_start: usize,
}

// ------------------------------------------------------------------------------- the parameters

/// Natural parameters. `p[to * k + from]`, column-stochastic.
#[derive(Debug, Clone)]
struct Nat {
    k: usize,
    p: Vec<f64>,
    mu: Vec<f64>,
    var: Vec<f64>,
}

fn n_unc(k: usize) -> usize {
    k * (k - 1) + 2 * k
}

impl Nat {
    /// From the unconstrained coordinates `[logits (k-1 x k, row-major), mu, ln var]`.
    fn from_unc(u: &[f64], k: usize) -> Nat {
        let km = k - 1;
        let mut p = vec![0.0; k * k];
        for c in 0..k {
            let mut z = vec![0.0; k];
            for m in 0..km {
                z[m] = u[m * k + c];
            }
            let top = z.iter().fold(f64::NEG_INFINITY, |a, b| a.max(*b));
            let e: Vec<f64> = z.iter().map(|v| (v - top).exp()).collect();
            let s: f64 = e.iter().sum();
            for m in 0..k {
                p[m * k + c] = e[m] / s;
            }
        }
        let mu = u[k * km..k * km + k].to_vec();
        let var = u[k * km + k..].iter().map(|v| v.exp()).collect();
        Nat { k, p, mu, var }
    }

    fn to_unc(&self) -> Vec<f64> {
        let k = self.k;
        let mut u = Vec::with_capacity(n_unc(k));
        for m in 0..k - 1 {
            for c in 0..k {
                let top = self.p[(k - 1) * k + c].max(1e-10);
                u.push((self.p[m * k + c].max(1e-10) / top).ln());
            }
        }
        u.extend(&self.mu);
        u.extend(self.var.iter().map(|v| v.ln()));
        u
    }

    /// Any transition probability under `SOFT_FLOOR` lifted to it, columns renormalised.
    fn softened(&self) -> Option<Nat> {
        if self.p.iter().all(|v| *v >= SOFT_FLOOR) {
            return None;
        }
        let k = self.k;
        let mut out = self.clone();
        for c in 0..k {
            let mut s = 0.0;
            for m in 0..k {
                out.p[m * k + c] = out.p[m * k + c].max(SOFT_FLOOR);
                s += out.p[m * k + c];
            }
            for m in 0..k {
                out.p[m * k + c] /= s;
            }
        }
        Some(out)
    }
}

fn mat_of(data: Vec<f64>, k: usize) -> Option<Mat> {
    Mat::from_vec(k, k, data).ok()
}

/// The steady state `pi` of `P` and `M^-1` with `M = I - P + 11'` (so `pi = M^-1 1`, and a change
/// `dP` moves it by `M^-1 dP pi`). `None` when `M` is singular (a reducible chain).
fn steady_state(p: &[f64], k: usize) -> Option<(Vec<f64>, Mat)> {
    let mut m = vec![0.0; k * k];
    for i in 0..k {
        for j in 0..k {
            m[i * k + j] = (if i == j { 1.0 } else { 0.0 }) - p[i * k + j] + 1.0;
        }
    }
    let minv = inverse(&mat_of(m, k)?).ok()?;
    let pi: Vec<f64> =
        (0..k).map(|i| (0..k).map(|j| minv[(i, j)]).sum::<f64>().max(PROB_FLOOR)).collect();
    Some((pi, minv))
}

// ----------------------------------------------------------------------- filter and smoother

struct Filtered {
    /// Predicted probabilities `P(s_t | y_<t)`, `n x k` row-major.
    pred: Vec<f64>,
    /// Filtered probabilities `P(s_t | y_<=t)`.
    alpha: Vec<f64>,
    ll: f64,
}

/// The scaled Hamilton filter. `None` if a bar has no usable density (the likelihood is not
/// finite there).
fn filter(y: &[f64], nat: &Nat, pi0: &[f64]) -> Option<Filtered> {
    let k = nat.k;
    let n = y.len();
    let mut pred = vec![0.0; n * k];
    let mut alpha = vec![0.0; n * k];
    let mut prev = pi0.to_vec();
    let mut ll = 0.0;
    let mut f = vec![0.0; k];
    for t in 0..n {
        for i in 0..k {
            pred[t * k + i] = (0..k).map(|j| nat.p[i * k + j] * prev[j]).sum();
        }
        let mut top = f64::NEG_INFINITY;
        for i in 0..k {
            let d = y[t] - nat.mu[i];
            f[i] = -0.5 * (LN_2PI + nat.var[i].ln()) - 0.5 * d * d / nat.var[i];
            top = top.max(f[i]);
        }
        let mut c = 0.0;
        for i in 0..k {
            f[i] = (f[i] - top).exp();
            c += pred[t * k + i] * f[i];
        }
        if c.is_nan() || c <= 0.0 || !c.is_finite() || !top.is_finite() {
            return None;
        }
        for i in 0..k {
            alpha[t * k + i] = pred[t * k + i] * f[i] / c;
            prev[i] = alpha[t * k + i];
        }
        ll += c.ln() + top;
    }
    Some(Filtered { pred, alpha, ll })
}

struct Smoothed {
    /// `P(s_t | all y)`, `n x k`.
    gamma: Vec<f64>,
    /// Expected transitions, `xi[to * k + from]`, summed over every bar including the pair
    /// (`s_{-1}`, `s_0`).
    xi: Vec<f64>,
    /// `P(s_{-1} = i | all y)`.
    w0: Vec<f64>,
}

fn smooth(nat: &Nat, f: &Filtered, pi0: &[f64], n: usize) -> Smoothed {
    let k = nat.k;
    let mut gamma = vec![0.0; n * k];
    let mut xi = vec![0.0; k * k];
    gamma[(n - 1) * k..].copy_from_slice(&f.alpha[(n - 1) * k..]);
    let mut r = vec![0.0; k];
    for t in (0..n - 1).rev() {
        for m in 0..k {
            r[m] = gamma[(t + 1) * k + m] / f.pred[(t + 1) * k + m].max(PRED_FLOOR);
        }
        for i in 0..k {
            let a = f.alpha[t * k + i];
            let mut s = 0.0;
            for m in 0..k {
                s += nat.p[m * k + i] * r[m];
                xi[m * k + i] += nat.p[m * k + i] * r[m] * a;
            }
            gamma[t * k + i] = a * s;
        }
    }
    for m in 0..k {
        r[m] = gamma[m] / f.pred[m].max(PRED_FLOOR);
    }
    let mut w0 = vec![0.0; k];
    for i in 0..k {
        for m in 0..k {
            let v = nat.p[m * k + i] * r[m] * pi0[i];
            xi[m * k + i] += v;
            w0[i] += v;
        }
    }
    Smoothed { gamma, xi, w0 }
}

/// Filtered and smoothed probabilities (`n x k` each) and the log-likelihood.
pub type FilterSmooth = (Vec<Vec<f64>>, Vec<Vec<f64>>, f64);

/// The filter and smoother at given parameters, for checking and for reading regimes at
/// parameters the caller holds. `transition[from][to]` (each row sums to one), `mean` and
/// `variance` per regime in the units of `returns`. Returns `(filtered, smoothed,
/// log_likelihood)`, probabilities `n x k`. The filtered rows depend only on `returns[..=t]`; the
/// smoothed rows depend on all of `returns`.
pub fn filter_and_smooth(
    returns: &[f64],
    transition: &[Vec<f64>],
    mean: &[f64],
    variance: &[f64],
) -> Option<FilterSmooth> {
    let k = mean.len();
    if returns.is_empty() || k < 2 || transition.len() != k || variance.len() != k {
        return None;
    }
    let mut p = vec![0.0; k * k];
    for from in 0..k {
        if transition[from].len() != k {
            return None;
        }
        for to in 0..k {
            p[to * k + from] = transition[from][to];
        }
    }
    let nat = Nat { k, p, mu: mean.to_vec(), var: variance.to_vec() };
    let (pi0, _) = steady_state(&nat.p, k)?;
    let f = filter(returns, &nat, &pi0)?;
    let s = smooth(&nat, &f, &pi0, returns.len());
    let rows = |v: &[f64]| v.chunks(k).map(<[f64]>::to_vec).collect::<Vec<_>>();
    Some((rows(&f.alpha), rows(&s.gamma), f.ll))
}

// ---------------------------------------------------------------- likelihood and its gradient

/// Mean negative log-likelihood and its gradient in the unconstrained coordinates. `None` where
/// the likelihood is not finite or the chain is reducible.
fn nll_grad(y: &[f64], u: &[f64], k: usize) -> Option<(f64, Vec<f64>)> {
    let n = y.len();
    let nat = Nat::from_unc(u, k);
    if nat.var.iter().any(|v| !v.is_finite() || *v <= 0.0) {
        return None;
    }
    let (pi0, minv) = steady_state(&nat.p, k)?;
    let f = filter(y, &nat, &pi0)?;
    let s = smooth(&nat, &f, &pi0, n);
    let km = k - 1;
    let mut g = vec![0.0; n_unc(k)];
    for c in 0..k {
        let tot: f64 = (0..k).map(|m| s.xi[m * k + c]).sum();
        for m in 0..km {
            // the transitions out of `c` and the initial distribution's dependence on them
            let g_trans = s.xi[m * k + c] - nat.p[m * k + c] * tot;
            let v: Vec<f64> = (0..k)
                .map(|r| {
                    let delta = if r == m { 1.0 } else { 0.0 };
                    pi0[c] * nat.p[r * k + c] * (delta - nat.p[m * k + c])
                })
                .collect();
            let mut g_init = 0.0;
            for i in 0..k {
                let dpi: f64 = (0..k).map(|j| minv[(i, j)] * v[j]).sum();
                g_init += s.w0[i] / pi0[i] * dpi;
            }
            g[m * k + c] = g_trans + g_init;
        }
    }
    for i in 0..k {
        let (mut gm, mut gv) = (0.0, 0.0);
        for t in 0..n {
            let d = y[t] - nat.mu[i];
            let w = s.gamma[t * k + i];
            gm += w * d / nat.var[i];
            gv += w * (-0.5 + 0.5 * d * d / nat.var[i]);
        }
        g[k * km + i] = gm;
        g[k * km + k + i] = gv;
    }
    let nn = n as f64;
    Some((-f.ll / nn, g.into_iter().map(|v| -v / nn).collect()))
}

/// The negative log-likelihood and its gradient at unconstrained coordinates, `[logits (k - 1 x
/// k, row-major), mu, ln sigma2]`, of a series in percent. For checking the gradient against
/// differences. The value is per bar.
pub fn mean_nll_gradient(
    returns_percent: &[f64],
    coords: &[f64],
    regimes: usize,
) -> Option<(f64, Vec<f64>)> {
    if !(2..=3).contains(&regimes) || coords.len() != n_unc(regimes) || returns_percent.is_empty() {
        return None;
    }
    nll_grad(returns_percent, coords, regimes)
}

// ------------------------------------------------------------------------------------- the EM

/// One EM step of the reference (smooth, then update), and the log-likelihood of the parameters
/// it started from.
fn em_step(y: &[f64], nat: &Nat) -> Option<(f64, Nat)> {
    let k = nat.k;
    let n = y.len();
    let (pi0, _) = steady_state(&nat.p, k)?;
    let f = filter(y, nat, &pi0)?;
    let s = smooth(nat, &f, &pi0, n);
    let mass: Vec<f64> = (0..k).map(|i| (0..n).map(|t| s.gamma[t * k + i]).sum()).collect();
    if mass.iter().any(|m| m.is_nan() || *m <= 0.0) {
        return None;
    }
    let mut p = vec![0.0; k * k];
    for from in 0..k {
        let mut row: Vec<f64> = (0..k - 1).map(|to| s.xi[to * k + from] / mass[from]).collect();
        let delta = row.iter().sum::<f64>() - 1.0;
        if delta > 0.0 {
            // rounding made the row exceed one: the reference rescales it the same way
            row.iter_mut().for_each(|v| *v /= 1.0 + delta + 1e-6);
        }
        let mut last = 1.0;
        for (to, v) in row.iter().enumerate() {
            p[to * k + from] = *v;
            last -= v;
        }
        p[(k - 1) * k + from] = last;
    }
    let mut mu = vec![0.0; k];
    let mut var = vec![0.0; k];
    for i in 0..k {
        mu[i] = (0..n).map(|t| s.gamma[t * k + i] * y[t]).sum::<f64>() / mass[i];
        var[i] = (0..n)
            .map(|t| {
                let d = y[t] - mu[i];
                s.gamma[t * k + i] * d * d
            })
            .sum::<f64>()
            / mass[i];
    }
    Some((f.ll, Nat { k, p, mu, var }))
}

/// The reference's `_fit_em` with `maxiter = 5`, `tolerance = 0`: it stops early only if the
/// likelihood stopped rising after the second step.
fn em(y: &[f64], start: Nat) -> Option<Nat> {
    let mut llf: Vec<f64> = Vec::new();
    let mut cur = start;
    let mut delta = 0.0;
    let mut i = 0;
    while i < EM_STEPS && (i < 2 || delta > 0.0) {
        let (ll, next) = em_step(y, &cur)?;
        llf.push(ll);
        cur = next;
        if i > 0 {
            delta = 2.0 * (llf[i] - llf[i - 1]) / (llf[i] + llf[i - 1]).abs();
        }
        i += 1;
    }
    Some(cur)
}

// -------------------------------------------------------------------------------- the starts

fn variance_of(y: &[f64]) -> f64 {
    let m = y.iter().sum::<f64>() / y.len() as f64;
    y.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / y.len() as f64
}

/// The reference's start: equal transitions, means `mean * i / k`, `linspace(v / 10, v, k)`.
fn reference_start(y: &[f64], k: usize) -> Nat {
    let v = variance_of(y);
    let m = y.iter().sum::<f64>() / y.len() as f64;
    let lo = v / 10.0;
    Nat {
        k,
        p: vec![1.0 / k as f64; k * k],
        mu: (0..k).map(|i| m * (i as f64 / k as f64)).collect(),
        var: (0..k).map(|i| lo + (v - lo) * i as f64 / (k - 1) as f64).collect(),
    }
}

/// The bars cut into `k` equal groups by the rank of `key` (ties by position); each group's mean
/// and variance are a regime's, and each regime stays with probability `PARTITION_STAY`.
fn partition_start(y: &[f64], k: usize, key: &[f64]) -> Nat {
    let n = y.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|a, b| key[*a].total_cmp(&key[*b]).then(a.cmp(b)));
    let floor = 1e-6 * variance_of(y);
    let off = (1.0 - PARTITION_STAY) / (k - 1) as f64;
    let mut p = vec![off; k * k];
    for i in 0..k {
        p[i * k + i] = PARTITION_STAY;
    }
    let (mut mu, mut var) = (Vec::new(), Vec::new());
    for g in 0..k {
        let group: Vec<f64> = order[g * n / k..(g + 1) * n / k].iter().map(|i| y[*i]).collect();
        mu.push(group.iter().sum::<f64>() / group.len() as f64);
        var.push(variance_of(&group).max(floor));
    }
    Nat { k, p, mu, var }
}

fn starts(y: &[f64], k: usize) -> Vec<Nat> {
    let m = y.iter().sum::<f64>() / y.len() as f64;
    let dev: Vec<f64> = y.iter().map(|v| (v - m) * (v - m)).collect();
    vec![reference_start(y, k), partition_start(y, k, &dev), partition_start(y, k, y)]
}

// ------------------------------------------------------------------------------------ the fit

struct Candidate {
    nat: Nat,
    llf: f64,
}

/// BFGS from `u`, restarted (inverse Hessian reset) while it has not converged. `None` unless it
/// converged.
fn run_bfgs(y: &[f64], k: usize, mut u: Vec<f64>) -> Option<(Vec<f64>, f64)> {
    let n = y.len() as f64;
    let opts = BfgsOptions::default();
    for _ in 0..MAX_RESTARTS {
        let res = bfgs(
            |x: &[f64], grad: &mut [f64]| match nll_grad(y, x, k) {
                Some((f, g)) => {
                    grad.copy_from_slice(&g);
                    f
                }
                None => {
                    grad.iter_mut().for_each(|v| *v = f64::NAN);
                    f64::NAN
                }
            },
            &u,
            &opts,
        )
        .ok()?;
        u = res.x.clone();
        if res.converged() {
            return Some((u, -res.f * n));
        }
    }
    None
}

fn fit_start(y: &[f64], start: Nat) -> Option<Candidate> {
    let k = start.k;
    let after_em = em(y, start)?;
    let (mut u, mut llf) = run_bfgs(y, k, after_em.to_unc())?;
    for _ in 0..POLISH_ROUNDS {
        let Some(soft) = Nat::from_unc(&u, k).softened() else { break };
        match run_bfgs(y, k, soft.to_unc()) {
            Some((u2, l2)) if l2 > llf => {
                u = u2;
                llf = l2;
            }
            _ => break,
        }
    }
    let nat = Nat::from_unc(&u, k);
    let floor = DEGENERATE_VARIANCE_RATIO * variance_of(y);
    if nat.var.iter().any(|v| v.is_nan() || *v < floor) || !llf.is_finite() {
        return None;
    }
    Some(Candidate { nat, llf })
}

/// Fits `regimes` regimes to `returns` (fractions) and reports in the units passed in.
pub fn markov_regimes(returns: &[f64], opts: &HmmOptions) -> Result<HmmResult, HmmError> {
    let k = opts.regimes;
    if !(2..=3).contains(&k) {
        return Err(HmmError::BadRegimes { regimes: k });
    }
    let needed = MIN_OBS_PER_REGIME * (k - 1);
    if returns.len() < needed {
        return Err(HmmError::InsufficientObservations { found: returns.len(), needed });
    }
    if !returns.iter().all(|v| v.is_finite()) {
        return Err(HmmError::NonFinite);
    }
    let (lo, hi) =
        returns.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), v| (l.min(*v), h.max(*v)));
    if hi - lo == 0.0 {
        return Err(HmmError::DegenerateSeries);
    }
    let y: Vec<f64> = returns.iter().map(|v| v * SCALE).collect();

    let mut candidates: Vec<Option<Candidate>> = Vec::new();
    for start in starts(&y, k) {
        candidates.push(fit_start(&y, start));
    }
    debug_assert_eq!(candidates.len(), STARTS);
    let start_log_likelihoods: Vec<Option<f64>> =
        candidates.iter().map(|c| c.as_ref().map(|c| c.llf)).collect();
    let mut winner: Option<usize> = None;
    for (i, c) in candidates.iter().enumerate() {
        if let Some(c) = c {
            // strict: a tie goes to the earlier start
            if winner
                .is_none_or(|w| c.llf > candidates[w].as_ref().map_or(f64::NEG_INFINITY, |b| b.llf))
            {
                winner = Some(i);
            }
        }
    }
    let Some(w) = winner else {
        return Err(if start_log_likelihoods.iter().all(Option::is_none) {
            HmmError::DidNotConverge
        } else {
            HmmError::DegenerateFit
        });
    };
    let best = candidates[w].take().expect("winner is a fit");
    assemble(&y, best.nat, best.llf, w, start_log_likelihoods, opts).ok_or(HmmError::DidNotConverge)
}

fn assemble(
    y: &[f64],
    nat: Nat,
    llf: f64,
    winning_start: usize,
    start_log_likelihoods: Vec<Option<f64>>,
    opts: &HmmOptions,
) -> Option<HmmResult> {
    let k = nat.k;
    let n = y.len();
    let (pi0, _) = steady_state(&nat.p, k)?;
    let f = filter(y, &nat, &pi0)?;
    let s = smooth(&nat, &f, &pi0, n);

    // calmest first; equal variances by mean, then by the fit's own order
    let mut order: Vec<usize> = (0..k).collect();
    order.sort_by(|a, b| {
        nat.var[*a].total_cmp(&nat.var[*b]).then(nat.mu[*a].total_cmp(&nat.mu[*b])).then(a.cmp(b))
    });
    let reorder =
        |v: &[f64], t: usize| -> Vec<f64> { order.iter().map(|o| v[t * k + o]).collect() };
    let smoothed: Vec<Vec<f64>> = (0..n).map(|t| reorder(&s.gamma, t)).collect();
    let filtered: Vec<Vec<f64>> = (0..n).map(|t| reorder(&f.alpha, t)).collect();
    // trans[from][to] in the new order
    let transition: Vec<Vec<f64>> =
        order.iter().map(|from| order.iter().map(|to| nat.p[to * k + from]).collect()).collect();

    let label: Vec<usize> = smoothed
        .iter()
        .map(|row| {
            let mut best = 0;
            for (i, v) in row.iter().enumerate() {
                if *v > row[best] {
                    best = i;
                }
            }
            best
        })
        .collect();
    let mut runs: Vec<TimelineSegment> = Vec::new();
    for (i, g) in label.iter().enumerate() {
        match runs.last_mut() {
            Some(r) if r.regime == *g => {
                r.end_index = i;
                r.periods += 1;
            }
            _ => {
                runs.push(TimelineSegment { regime: *g, start_index: i, end_index: i, periods: 1 })
            }
        }
    }
    let timeline_total = runs.len();
    let timeline = runs[timeline_total.saturating_sub(SEGMENTS_MAX)..].to_vec();

    let regime_stats: Vec<RegimeStat> = (0..k)
        .map(|i| {
            let o = order[i];
            let stay = transition[i][i];
            let duration = (stay < 1.0).then(|| 1.0 / (1.0 - stay));
            let support: f64 = smoothed.iter().map(|row| row[i]).sum();
            RegimeStat {
                regime: i,
                mean_per_period: nat.mu[o] / SCALE,
                vol_per_period: nat.var[o].sqrt() / SCALE,
                share_of_periods: label.iter().filter(|g| **g == i).count() as f64 / n as f64,
                stay_probability: stay,
                expected_duration_periods: duration,
                support_bars: support,
                low_support: support < opts.min_support_bars
                    || duration.is_some_and(|d| d < opts.min_duration_periods),
            }
        })
        .collect();
    let tail = n.saturating_sub(PROB_TAIL);
    let low_support_regimes =
        regime_stats.iter().filter(|r| r.low_support).map(|r| r.regime).collect();
    Some(HmmResult {
        n,
        regimes: k,
        converged: true,
        regime_stats,
        low_support_regimes,
        transition_matrix: transition,
        current_regime: label[n - 1],
        current_probabilities: smoothed[n - 1].clone(),
        recent_probabilities: smoothed[tail..].to_vec(),
        recent_filtered_probabilities: filtered[tail..].to_vec(),
        timeline_total,
        timeline,
        log_likelihood: llf,
        aic: 2.0 * n_unc(k) as f64 - 2.0 * llf,
        smoothed_probabilities: smoothed,
        filtered_probabilities: filtered,
        smoothed_is_causal: false,
        filtered_is_causal: true,
        start_log_likelihoods,
        winning_start,
    })
}
