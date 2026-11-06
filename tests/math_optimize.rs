//! BFGS on problems with a known minimiser, determinism, and the typed refusals.

use raptorbt::math::optimize::*;

fn rosenbrock(x: &[f64], g: &mut [f64]) -> f64 {
    let (a, b) = (x[0], x[1]);
    g[0] = -400.0 * a * (b - a * a) - 2.0 * (1.0 - a);
    g[1] = 200.0 * (b - a * a);
    100.0 * (b - a * a).powi(2) + (1.0 - a).powi(2)
}

/// 1/2 x'Ax - b'x with A = [[4,1,0],[1,3,1],[0,1,2]], b = [1,2,3].
///
/// The minimiser solves A x = b. By hand: 4x + y = 1, x + 3y + z = 2, y + 2z = 3 give
/// y = 1 - 4x and z = 1 + 2x from the first and last; the middle then reads
/// x + 3 - 12x + 1 + 2x = 2, so x = 2/9, y = 1/9, z = 13/9. The value there is -b'x/2.
fn quadratic(x: &[f64], g: &mut [f64]) -> f64 {
    let a = [[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]];
    let b = [1.0, 2.0, 3.0];
    let mut f = 0.0;
    for i in 0..3 {
        let ax: f64 = (0..3).map(|j| a[i][j] * x[j]).sum();
        g[i] = ax - b[i];
        f += 0.5 * x[i] * ax - b[i] * x[i];
    }
    f
}

#[test]
fn rosenbrock_reaches_its_minimum() {
    let r = bfgs(rosenbrock, &[-1.2, 1.0], &BfgsOptions::default()).unwrap();
    assert!(r.converged(), "status {:?}", r.status);
    assert!((r.x[0] - 1.0).abs() < 1e-6 && (r.x[1] - 1.0).abs() < 1e-6, "x = {:?}", r.x);
    assert!(r.f < 1e-12, "f = {:e}", r.f);
    assert!(r.evaluations >= r.iterations);
}

#[test]
fn quadratic_reaches_the_hand_computed_minimiser() {
    let r = bfgs(quadratic, &[0.0, 0.0, 0.0], &BfgsOptions::default()).unwrap();
    assert!(r.converged(), "status {:?}", r.status);
    let want = [2.0 / 9.0, 1.0 / 9.0, 13.0 / 9.0];
    for (i, w) in want.iter().enumerate() {
        assert!((r.x[i] - w).abs() < 1e-7, "x[{i}] = {} want {w}", r.x[i]);
    }
    // f(x*) = -b'x*/2 = -(2/9 + 2/9 + 39/9)/2 = -43/18
    assert!((r.f + 43.0 / 18.0).abs() < 1e-12, "f = {}", r.f);
    // a quadratic of dimension 3 needs few iterations with a Wolfe line search
    assert!(r.iterations <= 30, "{} iterations", r.iterations);
}

#[test]
fn starting_at_the_minimum_stops_at_once() {
    let want = [2.0 / 9.0, 1.0 / 9.0, 13.0 / 9.0];
    let r = bfgs(quadratic, &want, &BfgsOptions::default()).unwrap();
    assert_eq!(r.status, BfgsStatus::Converged(Convergence::Gradient));
    assert_eq!(r.iterations, 0);
    assert_eq!(r.evaluations, 1);
}

#[test]
fn identical_runs_are_bit_identical() {
    let a = bfgs(rosenbrock, &[-1.2, 1.0], &BfgsOptions::default()).unwrap();
    let b = bfgs(rosenbrock, &[-1.2, 1.0], &BfgsOptions::default()).unwrap();
    assert_eq!(a.iterations, b.iterations);
    assert_eq!(a.evaluations, b.evaluations);
    assert_eq!(a.status, b.status);
    assert_eq!(a.f.to_bits(), b.f.to_bits());
    for i in 0..2 {
        assert_eq!(a.x[i].to_bits(), b.x[i].to_bits());
        assert_eq!(a.grad[i].to_bits(), b.grad[i].to_bits());
    }
}

#[test]
fn an_iteration_cap_is_a_status_not_an_error() {
    let opts = BfgsOptions { max_iter: 2, ..BfgsOptions::default() };
    let r = bfgs(rosenbrock, &[-1.2, 1.0], &opts).unwrap();
    assert_eq!(r.status, BfgsStatus::MaxIterations);
    assert!(!r.converged());
    assert_eq!(r.iterations, 2);
    // the point returned is the best so far, below the start
    let mut g = [0.0; 2];
    assert!(r.f < rosenbrock(&[-1.2, 1.0], &mut g));
}

#[test]
fn a_non_finite_wall_is_backed_away_from() {
    // f = x^2 with NaN for x < -0.2. From 0.3 the first trial step (-0.6) lands on the wall; the
    // line search must read it as an infinite value and bisect back, not return NaN.
    let f = |x: &[f64], g: &mut [f64]| {
        if x[0] < -0.2 {
            g[0] = f64::NAN;
            return f64::NAN;
        }
        g[0] = 2.0 * x[0];
        x[0] * x[0]
    };
    let r = bfgs(f, &[0.3], &BfgsOptions::default()).unwrap();
    assert!(r.converged(), "status {:?}", r.status);
    assert!(r.x[0].abs() < 1e-7, "x = {}", r.x[0]);
    assert!(r.evaluations >= 3, "the wall was never reached: {} evaluations", r.evaluations);
}

#[test]
fn unusable_starts_are_typed_errors() {
    let opts = BfgsOptions::default();
    assert_eq!(bfgs(rosenbrock, &[], &opts).unwrap_err(), OptError::ZeroSize);
    assert!(matches!(
        bfgs(rosenbrock, &[f64::NAN, 1.0], &opts),
        Err(OptError::NonFinite { what: "starting point" })
    ));
    assert!(matches!(
        bfgs(rosenbrock, &[f64::INFINITY, 1.0], &opts),
        Err(OptError::NonFinite { what: "starting point" })
    ));
    let nan_value = |_: &[f64], g: &mut [f64]| {
        g[0] = 0.0;
        f64::NAN
    };
    assert!(matches!(
        bfgs(nan_value, &[1.0], &opts),
        Err(OptError::NonFinite { what: "objective" })
    ));
    let nan_grad = |x: &[f64], g: &mut [f64]| {
        g[0] = f64::NAN;
        x[0] * x[0]
    };
    assert!(matches!(bfgs(nan_grad, &[1.0], &opts), Err(OptError::NonFinite { what: "gradient" })));
}

#[test]
fn unusable_options_are_typed_errors() {
    let bad = [
        BfgsOptions { gtol: -1.0, ..BfgsOptions::default() },
        BfgsOptions { gtol: f64::NAN, ..BfgsOptions::default() },
        BfgsOptions { xtol: -1.0, ..BfgsOptions::default() },
        BfgsOptions { c1: 0.0, ..BfgsOptions::default() },
        BfgsOptions { c1: 0.95, ..BfgsOptions::default() },
        BfgsOptions { c2: 1.0, ..BfgsOptions::default() },
    ];
    for opts in bad {
        assert!(matches!(bfgs(rosenbrock, &[-1.2, 1.0], &opts), Err(OptError::BadOption { .. })));
    }
}

/// An objective whose value never changes while it reports a gradient: no step lowers it, as at
/// the rounding noise of a real likelihood. The line search fails along steepest descent for any
/// gradient, so the status depends only on the gradient against `stall_gtol`.
fn flat_with_gradient(slope: f64) -> impl FnMut(&[f64], &mut [f64]) -> f64 {
    move |_x, g| {
        g[0] = slope;
        1.0
    }
}

#[test]
fn a_line_search_failure_with_a_large_gradient_stays_a_failure() {
    // |g| = 1 against stall_gtol * max(1, |f|) = 1e-6: this is not a minimum
    let r = bfgs(flat_with_gradient(1.0), &[3.0], &BfgsOptions::default()).unwrap();
    assert_eq!(r.status, BfgsStatus::LineSearchFailed);
    assert!(!r.converged());
    assert_eq!(r.x, vec![3.0], "no step was taken");
}

#[test]
fn a_stall_with_a_small_gradient_reports_precision() {
    // |g| = 1e-7 is above gtol (1e-8) so the run is not converged by its gradient, and below
    // stall_gtol (1e-6): the objective has no resolution left to find a lower point
    let r = bfgs(flat_with_gradient(1e-7), &[3.0], &BfgsOptions::default()).unwrap();
    assert_eq!(r.status, BfgsStatus::Converged(Convergence::Precision));
    assert!(r.converged());
    assert_eq!(r.x, vec![3.0]);
}

#[test]
fn stall_gtol_alone_separates_the_two() {
    // the same objective and gradient; only the threshold moves, in both directions
    let tight = BfgsOptions { stall_gtol: 1e-8, ..BfgsOptions::default() };
    let r = bfgs(flat_with_gradient(1e-7), &[3.0], &tight).unwrap();
    assert_eq!(r.status, BfgsStatus::LineSearchFailed);
    let loose = BfgsOptions { stall_gtol: 2.0, ..BfgsOptions::default() };
    let r = bfgs(flat_with_gradient(1.0), &[3.0], &loose).unwrap();
    assert_eq!(r.status, BfgsStatus::Converged(Convergence::Precision));
    // zero is allowed: no stall is ever read as converged
    let none = BfgsOptions { stall_gtol: 0.0, ..BfgsOptions::default() };
    let r = bfgs(flat_with_gradient(1e-7), &[3.0], &none).unwrap();
    assert_eq!(r.status, BfgsStatus::LineSearchFailed);
}

#[test]
fn a_step_below_xtol_at_a_large_gradient_is_stalled_not_converged() {
    // An xtol this loose ends the run after one step, far from the minimiser of the quadratic.
    let opts = BfgsOptions { xtol: 1e3, ..BfgsOptions::default() };
    let r = bfgs(quadratic, &[0.0, 0.0, 0.0], &opts).unwrap();
    assert_eq!(r.status, BfgsStatus::Stalled);
    assert!(!r.converged());
    assert!(r.grad.iter().fold(0.0_f64, |m, g| m.max(g.abs())) > 1e-3, "grad {:?}", r.grad);
}
