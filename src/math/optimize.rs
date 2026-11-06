//! BFGS minimisation with an analytic gradient and a strong-Wolfe line search.
//!
//! Nocedal and Wright, *Numerical Optimization* (2nd ed.): the BFGS update of the inverse Hessian
//! (6.17) with the initial scaling (6.20), and the line search of Algorithms 3.5 and 3.6 with a
//! cubic interpolation (3.59) safeguarded by bisection. Deterministic: no random start, no
//! threads, the same order of operations on every run.
//!
//! An unusable start (empty, non-finite value or gradient) is an [`OptError`]. A run that stops
//! short of its tolerances is *not* an error: it returns the best point with a [`BfgsStatus`]
//! that says why it stopped, so the caller (a GARCH fit, the M-step of an EM) can use the point
//! or report `did_not_converge`.

use thiserror::Error;

const BRACKET_STEPS: usize = 30;
const ZOOM_STEPS: usize = 40;
const MAX_STEP: f64 = 1e12;

/// An optimisation that could not start.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum OptError {
    #[error("bfgs: no parameters")]
    ZeroSize,
    #[error("bfgs: the {what} is NaN or infinite at the starting point")]
    NonFinite { what: &'static str },
    #[error("bfgs: option {name} = {value} is not usable")]
    BadOption { name: &'static str, value: f64 },
}

/// Tolerances and limits.
#[derive(Debug, Clone, PartialEq)]
pub struct BfgsOptions {
    /// Iteration cap.
    pub max_iter: usize,
    /// Converged when `max |g_i| <= gtol * max(1, |f|)`.
    pub gtol: f64,
    /// A step that moves no coordinate by more than `xtol * max(1, max |x_i|)` ends the run:
    /// `Converged(Step)` when the gradient is within `stall_gtol` too, `Stalled` otherwise.
    pub xtol: f64,
    /// Sufficient-decrease constant of the Wolfe conditions.
    pub c1: f64,
    /// Curvature constant of the strong Wolfe conditions.
    pub c2: f64,
    /// A run whose line search fails along steepest descent, with `max |g_i| <= stall_gtol *
    /// max(1, |f|)`, has reached the precision of the objective (no step lowers `f` by more
    /// than its rounding noise) and is reported `Converged(Precision)`. With a larger gradient
    /// the same failure is `LineSearchFailed`.
    pub stall_gtol: f64,
}

impl Default for BfgsOptions {
    fn default() -> Self {
        BfgsOptions { max_iter: 500, gtol: 1e-8, xtol: 1e-14, c1: 1e-4, c2: 0.9, stall_gtol: 1e-6 }
    }
}

/// How a converged run ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Convergence {
    /// The gradient is within `gtol`.
    Gradient,
    /// The step fell below `xtol` and the gradient is within `stall_gtol`.
    Step,
    /// No descent step exists within the objective's rounding noise, and the gradient is
    /// within `stall_gtol`.
    Precision,
}

/// Why the run stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BfgsStatus {
    Converged(Convergence),
    /// `max_iter` iterations without meeting a tolerance.
    MaxIterations,
    /// No step along the steepest-descent direction satisfied the Wolfe conditions.
    LineSearchFailed,
    /// The step fell below `xtol` while the gradient is still above `stall_gtol`: the iterate
    /// stopped moving, not at a stationary point.
    Stalled,
}

/// The best point found.
#[derive(Debug, Clone, PartialEq)]
pub struct BfgsResult {
    pub x: Vec<f64>,
    pub f: f64,
    pub grad: Vec<f64>,
    pub iterations: usize,
    pub evaluations: usize,
    pub status: BfgsStatus,
}

impl BfgsResult {
    pub fn converged(&self) -> bool {
        matches!(self.status, BfgsStatus::Converged(_))
    }
}

#[derive(Clone, Copy)]
struct Point {
    alpha: f64,
    f: f64,
    d: f64,
}

/// The objective along `x + alpha p`, keeping the gradient of the latest evaluation.
struct Line<'a, F> {
    func: &'a mut F,
    x: &'a [f64],
    p: &'a [f64],
    xt: Vec<f64>,
    g: Vec<f64>,
    evals: usize,
}

impl<F: FnMut(&[f64], &mut [f64]) -> f64> Line<'_, F> {
    /// Value and slope at `alpha`; a non-finite value or gradient reads as an infinite value, so
    /// the search backs away from it.
    fn at(&mut self, alpha: f64) -> Point {
        for i in 0..self.x.len() {
            self.xt[i] = self.x[i] + alpha * self.p[i];
        }
        let f = (self.func)(&self.xt, &mut self.g);
        self.evals += 1;
        if !f.is_finite() || !self.g.iter().all(|v| v.is_finite()) {
            return Point { alpha, f: f64::INFINITY, d: 0.0 };
        }
        let d = self.g.iter().zip(self.p).map(|(g, p)| g * p).sum();
        Point { alpha, f, d }
    }
}

/// Cubic minimiser between two points (N&W 3.59), or `None` when it does not exist.
fn cubic(lo: Point, hi: Point) -> Option<f64> {
    if !lo.f.is_finite() || !hi.f.is_finite() {
        return None;
    }
    let d1 = lo.d + hi.d - 3.0 * (lo.f - hi.f) / (lo.alpha - hi.alpha);
    let disc = d1 * d1 - lo.d * hi.d;
    if disc < 0.0 {
        return None;
    }
    let d2 = (hi.alpha - lo.alpha).signum() * disc.sqrt();
    let a = hi.alpha - (hi.alpha - lo.alpha) * (hi.d + d2 - d1) / (hi.d - lo.d + 2.0 * d2);
    a.is_finite().then_some(a)
}

/// Algorithm 3.6: a point between `lo` (satisfies sufficient decrease, lowest value so far) and
/// `hi` that meets the strong Wolfe conditions.
fn zoom<F: FnMut(&[f64], &mut [f64]) -> f64>(
    line: &mut Line<'_, F>,
    f0: f64,
    d0: f64,
    mut lo: Point,
    mut hi: Point,
    c1: f64,
    c2: f64,
) -> Option<Point> {
    for _ in 0..ZOOM_STEPS {
        let (a, b) = if lo.alpha < hi.alpha { (lo.alpha, hi.alpha) } else { (hi.alpha, lo.alpha) };
        let width = b - a;
        if width <= f64::EPSILON * b.abs().max(1e-300) {
            return None;
        }
        let trial = match cubic(lo, hi) {
            Some(t) if t > a + 0.1 * width && t < b - 0.1 * width => t,
            _ => 0.5 * (a + b),
        };
        let cur = line.at(trial);
        if cur.f > f0 + c1 * trial * d0 || cur.f >= lo.f {
            hi = cur;
        } else {
            if cur.d.abs() <= -c2 * d0 {
                return Some(cur);
            }
            if cur.d * (hi.alpha - lo.alpha) >= 0.0 {
                hi = lo;
            }
            lo = cur;
        }
    }
    None
}

/// Algorithm 3.5: a step along `p` meeting the strong Wolfe conditions, starting from
/// `alpha_init`. On `Some`, `line.g` holds the gradient at the returned point.
fn wolfe_search<F: FnMut(&[f64], &mut [f64]) -> f64>(
    line: &mut Line<'_, F>,
    f0: f64,
    d0: f64,
    alpha_init: f64,
    c1: f64,
    c2: f64,
) -> Option<Point> {
    let mut prev = Point { alpha: 0.0, f: f0, d: d0 };
    let mut alpha = alpha_init;
    for i in 0..BRACKET_STEPS {
        let cur = line.at(alpha);
        if cur.f > f0 + c1 * alpha * d0 || (i > 0 && cur.f >= prev.f) {
            return zoom(line, f0, d0, prev, cur, c1, c2);
        }
        if cur.d.abs() <= -c2 * d0 {
            return Some(cur);
        }
        if cur.d >= 0.0 {
            return zoom(line, f0, d0, cur, prev, c1, c2);
        }
        prev = cur;
        alpha = (2.0 * alpha).min(MAX_STEP);
    }
    None
}

fn inf_norm(v: &[f64]) -> f64 {
    v.iter().fold(0.0, |m, x| m.max(x.abs()))
}

fn check_option(name: &'static str, value: f64, ok: bool) -> Result<(), OptError> {
    if ok && value.is_finite() {
        Ok(())
    } else {
        Err(OptError::BadOption { name, value })
    }
}

/// Minimises `func` from `x0`. `func(x, grad)` writes the gradient into `grad` and returns the
/// value.
pub fn bfgs<F>(mut func: F, x0: &[f64], opts: &BfgsOptions) -> Result<BfgsResult, OptError>
where
    F: FnMut(&[f64], &mut [f64]) -> f64,
{
    let n = x0.len();
    if n == 0 {
        return Err(OptError::ZeroSize);
    }
    check_option("gtol", opts.gtol, opts.gtol >= 0.0)?;
    check_option("xtol", opts.xtol, opts.xtol >= 0.0)?;
    check_option("c1", opts.c1, opts.c1 > 0.0 && opts.c1 < opts.c2)?;
    check_option("c2", opts.c2, opts.c2 < 1.0)?;
    check_option("stall_gtol", opts.stall_gtol, opts.stall_gtol >= 0.0)?;
    if !x0.iter().all(|v| v.is_finite()) {
        return Err(OptError::NonFinite { what: "starting point" });
    }
    let mut x = x0.to_vec();
    let mut g = vec![0.0; n];
    let mut fx = func(&x, &mut g);
    let mut evals = 1;
    if !fx.is_finite() {
        return Err(OptError::NonFinite { what: "objective" });
    }
    if !g.iter().all(|v| v.is_finite()) {
        return Err(OptError::NonFinite { what: "gradient" });
    }
    let mut h = identity(n);
    let mut fresh = true; // h is the identity and no update has been made yet
    let mut status = BfgsStatus::MaxIterations;
    let mut iterations = 0;
    while iterations < opts.max_iter {
        if inf_norm(&g) <= opts.gtol * fx.abs().max(1.0) {
            status = BfgsStatus::Converged(Convergence::Gradient);
            break;
        }
        let mut p = direction(&h, &g, n);
        let mut d0: f64 = g.iter().zip(&p).map(|(a, b)| a * b).sum();
        if d0 >= 0.0 || !d0.is_finite() {
            h = identity(n);
            fresh = true;
            p = g.iter().map(|v| -v).collect();
            d0 = -g.iter().map(|v| v * v).sum::<f64>();
        }
        let alpha_init =
            if fresh { (1.0 / g.iter().map(|v| v.abs()).sum::<f64>()).min(1.0) } else { 1.0 };
        let mut line =
            Line { func: &mut func, x: &x, p: &p, xt: vec![0.0; n], g: vec![0.0; n], evals: 0 };
        let mut step = wolfe_search(&mut line, fx, d0, alpha_init, opts.c1, opts.c2);
        if step.is_none() && !fresh {
            // the quasi-Newton direction failed: retry along steepest descent
            evals += line.evals;
            drop(line);
            h = identity(n);
            fresh = true;
            p = g.iter().map(|v| -v).collect();
            d0 = -g.iter().map(|v| v * v).sum::<f64>();
            line =
                Line { func: &mut func, x: &x, p: &p, xt: vec![0.0; n], g: vec![0.0; n], evals: 0 };
            let init = (1.0 / g.iter().map(|v| v.abs()).sum::<f64>()).min(1.0);
            step = wolfe_search(&mut line, fx, d0, init, opts.c1, opts.c2);
            evals += line.evals;
        } else {
            evals += line.evals;
        }
        iterations += 1;
        let Some(pt) = step else {
            status = if inf_norm(&g) <= opts.stall_gtol * fx.abs().max(1.0) {
                BfgsStatus::Converged(Convergence::Precision)
            } else {
                BfgsStatus::LineSearchFailed
            };
            break;
        };
        let s: Vec<f64> = p.iter().map(|v| pt.alpha * v).collect();
        let g_new = line.g.clone();
        let y: Vec<f64> = g_new.iter().zip(&g).map(|(a, b)| a - b).collect();
        for i in 0..n {
            x[i] += s[i];
        }
        fx = pt.f;
        g = g_new;
        if inf_norm(&s) <= opts.xtol * inf_norm(&x).max(1.0) {
            status = if inf_norm(&g) <= opts.stall_gtol * fx.abs().max(1.0) {
                BfgsStatus::Converged(Convergence::Step)
            } else {
                BfgsStatus::Stalled
            };
            break;
        }
        let sy: f64 = s.iter().zip(&y).map(|(a, b)| a * b).sum();
        let yy: f64 = y.iter().map(|v| v * v).sum();
        let ss: f64 = s.iter().map(|v| v * v).sum();
        // the Wolfe curvature condition makes sy positive; a numerically zero one skips the update
        if sy > f64::EPSILON * (ss * yy).sqrt() && sy > 0.0 {
            if fresh {
                let scale = sy / yy;
                for i in 0..n {
                    h[i * n + i] = scale;
                }
                fresh = false;
            }
            bfgs_update(&mut h, &s, &y, sy, n);
        }
    }
    Ok(BfgsResult { x, f: fx, grad: g, iterations, evaluations: evals, status })
}

fn identity(n: usize) -> Vec<f64> {
    let mut h = vec![0.0; n * n];
    for i in 0..n {
        h[i * n + i] = 1.0;
    }
    h
}

/// `-H g`.
fn direction(h: &[f64], g: &[f64], n: usize) -> Vec<f64> {
    (0..n).map(|i| -(0..n).map(|j| h[i * n + j] * g[j]).sum::<f64>()).collect()
}

/// `H <- (I - rho s y')H(I - rho y s') + rho s s'` in its expanded form (N&W 6.17).
fn bfgs_update(h: &mut [f64], s: &[f64], y: &[f64], sy: f64, n: usize) {
    let rho = 1.0 / sy;
    let hy: Vec<f64> = (0..n).map(|i| (0..n).map(|j| h[i * n + j] * y[j]).sum()).collect();
    let yhy: f64 = y.iter().zip(&hy).map(|(a, b)| a * b).sum();
    let k = (sy + yhy) * rho * rho;
    for i in 0..n {
        for j in 0..n {
            h[i * n + j] += k * s[i] * s[j] - rho * (hy[i] * s[j] + s[i] * hy[j]);
        }
    }
}
