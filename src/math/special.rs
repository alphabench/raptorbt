//! Special functions for the statistics: `ln_gamma`, the regularised incomplete beta and
//! gamma functions, and the t, chi-squared and F distributions built on them.
//!
//! Written from the published formulas, not from a numerical library:
//! - `ln_gamma`: Stirling's series (DLMF 5.11.1) after shifting the argument to 15 or more with
//!   the recurrence `ln G(z) = ln G(z + 1) - ln z`; only `x > 0` is a valid argument.
//! - incomplete beta: the continued fraction DLMF 8.17.22, evaluated by the modified Lentz
//!   algorithm, on the side `x <= (a + 1) / (a + b + 2)` where it converges quickly.
//! - incomplete gamma: the series DLMF 8.7.1 for `x < a + 1`, otherwise the continued fraction
//!   for `G(a, x)` (DLMF 8.9.2, in its even contraction).
//!
//! The prefactors of the incomplete functions are formed so that no term of the size of
//! `a ln a` is subtracted (see `ln_gamma_ratio`, `ln1p_minus_x`).
//!
//! Every function returns the small tail without cancellation: `*_pq` gives `(P, Q)` with
//! `P + Q = 1`, the smaller of the two computed directly. The relative error is a few `eps`
//! times the exponent of the prefactor (under 3e-14 against a 40-digit evaluation at degrees of
//! freedom of 1,000). The normal distribution is `pricing::norm`.
//!
//! An argument outside a function's domain, a NaN, or an iteration that does not converge is a
//! [`SpecialError`]; no function returns NaN.

pub use crate::pricing::norm::{norm_cdf, norm_pdf, norm_ppf};

use thiserror::Error;

/// A special function was asked for something it cannot answer.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum SpecialError {
    /// An argument is NaN, or infinite where a finite number is required.
    #[error("{function}: argument {name} is not a number")]
    NotANumber { function: &'static str, name: &'static str },
    /// An argument is outside the function's domain.
    #[error("{function}: argument {name} = {value} is outside the domain ({domain})")]
    Domain { function: &'static str, name: &'static str, value: f64, domain: &'static str },
    /// A series or continued fraction did not settle within its iteration cap.
    #[error("{function}: no convergence in {iterations} iterations")]
    NoConvergence { function: &'static str, iterations: usize },
}

const TINY: f64 = 1e-300;
const CF_EPS: f64 = 1e-16;
const BETA_CF_MAX_ITER: usize = 100_000;
const GAMMA_MAX_ITER: usize = 1_000_000;
const LN_SQRT_TWO_PI: f64 = 0.918_938_533_204_672_7;

/// B(2k) / (2k (2k - 1)) for k = 1..=10, the coefficients of Stirling's series.
const STIRLING: [f64; 10] = [
    1.0 / 12.0,
    -1.0 / 360.0,
    1.0 / 1260.0,
    -1.0 / 1680.0,
    1.0 / 1188.0,
    -691.0 / 360_360.0,
    1.0 / 156.0,
    -3617.0 / 122_400.0,
    43_867.0 / 244_188.0,
    -174_611.0 / 125_400.0,
];

fn check_number(function: &'static str, name: &'static str, v: f64) -> Result<(), SpecialError> {
    if v.is_nan() {
        return Err(SpecialError::NotANumber { function, name });
    }
    Ok(())
}

fn check_positive(function: &'static str, name: &'static str, v: f64) -> Result<(), SpecialError> {
    check_number(function, name, v)?;
    if !(v > 0.0 && v.is_finite()) {
        return Err(SpecialError::Domain { function, name, value: v, domain: "finite and > 0" });
    }
    Ok(())
}

/// Natural log of the gamma function for `x > 0`.
pub fn ln_gamma(x: f64) -> Result<f64, SpecialError> {
    check_positive("ln_gamma", "x", x)?;
    Ok(ln_gamma_unchecked(x))
}

fn ln_gamma_unchecked(x: f64) -> f64 {
    // G(1) = G(2) = 1: the series loses the relative accuracy of a result near zero.
    if x == 1.0 || x == 2.0 {
        return 0.0;
    }
    let mut z = x;
    let mut prod = 1.0f64;
    let mut shift = 0.0f64;
    while z < 15.0 {
        prod *= z;
        z += 1.0;
        if prod > 1e280 {
            shift += prod.ln();
            prod = 1.0;
        }
    }
    shift += prod.ln();
    (z - 0.5) * z.ln() - z + LN_SQRT_TWO_PI + stirling_correction(z) - shift
}

/// The tail of Stirling's series, `ln G(z) - ((z - 1/2) ln z - z + ln sqrt(2 pi))`, for `z >= 15`.
fn stirling_correction(z: f64) -> f64 {
    let zi = 1.0 / z;
    let z2 = zi * zi;
    let mut s = 0.0;
    for &c in STIRLING.iter().rev() {
        s = s * z2 + c;
    }
    s * zi
}

/// `ln G(b) - ln G(a + b)` for `b >= 15`, with the large terms combined analytically so nothing
/// of the size of `b ln b` is subtracted: `-(b - 1/2) ln(1 + a/b) - a (ln(a + b) - 1)` plus the
/// difference of the two Stirling corrections.
fn ln_gamma_ratio(a: f64, b: f64) -> f64 {
    -(b - 0.5) * (a / b).ln_1p() - a * ((a + b).ln() - 1.0) + stirling_correction(b)
        - stirling_correction(a + b)
}

/// `ln(1 + d) - d` for `d > -1`, by its series where the subtraction would cancel.
fn ln1p_minus_x(d: f64) -> f64 {
    if d.abs() >= 0.5 {
        return d.ln_1p() - d;
    }
    // sum_{k >= 2} (-1)^(k-1) d^k / k
    let mut power = d;
    let mut sum = 0.0;
    for k in 2..200 {
        power *= d;
        let term = if k % 2 == 0 { -power } else { power } / k as f64;
        sum += term;
        if term.abs() < 1e-18 * sum.abs() {
            break;
        }
    }
    sum
}

/// `ln B(a, b)` for `a, b > 0`.
pub fn ln_beta(a: f64, b: f64) -> Result<f64, SpecialError> {
    check_positive("ln_beta", "a", a)?;
    check_positive("ln_beta", "b", b)?;
    Ok(ln_beta_unchecked(a, b))
}

fn ln_beta_unchecked(a: f64, b: f64) -> f64 {
    let (small, large) = if a <= b { (a, b) } else { (b, a) };
    if large < 15.0 {
        return ln_gamma_unchecked(a) + ln_gamma_unchecked(b) - ln_gamma_unchecked(a + b);
    }
    ln_gamma_unchecked(small) + ln_gamma_ratio(small, large)
}

/// DLMF 8.17.22 by the modified Lentz algorithm: the continued fraction of
/// `I_x(a, b) = x^a (1-x)^b / (a B(a, b)) * 1 / (1 + d1 / (1 + d2 / (1 + ...)))`.
fn beta_cf(a: f64, b: f64, x: f64) -> Result<f64, SpecialError> {
    let mut f = TINY;
    let mut c = f;
    let mut d = 0.0f64;
    for i in 0..BETA_CF_MAX_ITER {
        let m = (i / 2) as f64;
        let num = if i == 0 {
            1.0
        } else if i % 2 == 0 {
            m * (b - m) * x / ((a + 2.0 * m - 1.0) * (a + 2.0 * m))
        } else {
            -(a + m) * (a + b + m) * x / ((a + 2.0 * m) * (a + 2.0 * m + 1.0))
        };
        d = 1.0 + num * d;
        if d == 0.0 {
            d = TINY;
        }
        c = 1.0 + num / c;
        if c == 0.0 {
            c = TINY;
        }
        d = 1.0 / d;
        let delta = c * d;
        f *= delta;
        if (1.0 - delta).abs() < CF_EPS {
            return Ok(f);
        }
    }
    Err(SpecialError::NoConvergence { function: "inc_beta", iterations: BETA_CF_MAX_ITER })
}

/// `(I_x(a, b), 1 - I_x(a, b))` where `y = 1 - x` is given alongside `x` so that a caller that
/// holds the small one exactly (a t statistic's `x^2 / (nu + x^2)`) loses nothing to `1 - x`.
/// The smaller of the two is computed directly.
pub fn inc_beta_pq(a: f64, b: f64, x: f64, y: f64) -> Result<(f64, f64), SpecialError> {
    check_positive("inc_beta", "a", a)?;
    check_positive("inc_beta", "b", b)?;
    check_number("inc_beta", "x", x)?;
    check_number("inc_beta", "y", y)?;
    if !(0.0..=1.0).contains(&x) {
        return Err(SpecialError::Domain {
            function: "inc_beta",
            name: "x",
            value: x,
            domain: "0 <= x <= 1",
        });
    }
    if !(0.0..=1.0).contains(&y) {
        return Err(SpecialError::Domain {
            function: "inc_beta",
            name: "y",
            value: y,
            domain: "0 <= y <= 1",
        });
    }
    if x == 0.0 {
        return Ok((0.0, 1.0));
    }
    if y == 0.0 {
        return Ok((1.0, 0.0));
    }
    if x > (a + 1.0) / (a + b + 2.0) {
        let (q, p) = beta_direct(b, a, y, x)?;
        return Ok((p, q));
    }
    beta_direct(a, b, x, y)
}

fn beta_direct(a: f64, b: f64, x: f64, y: f64) -> Result<(f64, f64), SpecialError> {
    let ln_front = a * x.ln() + b * y.ln() - ln_beta_unchecked(a, b);
    let p = ln_front.exp() / a * beta_cf(a, b, x)?;
    Ok((p, 1.0 - p))
}

/// Regularised incomplete beta function `I_x(a, b)`, `a, b > 0`, `0 <= x <= 1`.
pub fn inc_beta(a: f64, b: f64, x: f64) -> Result<f64, SpecialError> {
    Ok(inc_beta_pq(a, b, x, 1.0 - x)?.0)
}

/// `(P(a, x), Q(a, x))`, the regularised lower and upper incomplete gamma functions, the
/// smaller computed directly. `a > 0`, `x >= 0` (`x = inf` gives `(1, 0)`).
pub fn inc_gamma_pq(a: f64, x: f64) -> Result<(f64, f64), SpecialError> {
    check_positive("inc_gamma", "a", a)?;
    check_number("inc_gamma", "x", x)?;
    if x < 0.0 {
        return Err(SpecialError::Domain {
            function: "inc_gamma",
            name: "x",
            value: x,
            domain: ">= 0",
        });
    }
    if x == 0.0 {
        return Ok((0.0, 1.0));
    }
    if x == f64::INFINITY {
        return Ok((1.0, 0.0));
    }
    // ln(x^a e^-x / G(a)); for a >= 15 the Stirling form a (ln(x/a) - (x/a - 1)) + ln(a)/2 -
    // ln sqrt(2 pi) - correction, which stays accurate when x is near a
    let ln_front = if a >= 15.0 {
        a * ln1p_minus_x((x - a) / a) + 0.5 * a.ln() - LN_SQRT_TWO_PI - stirling_correction(a)
    } else {
        a * x.ln() - x - ln_gamma_unchecked(a)
    };
    if x < a + 1.0 {
        // DLMF 8.7.1: P = x^a e^-x * sum_k x^k / G(a + k + 1)
        let mut term = 1.0 / a;
        let mut sum = term;
        let mut k = 1.0f64;
        for _ in 0..GAMMA_MAX_ITER {
            term *= x / (a + k);
            sum += term;
            if term.abs() < sum.abs() * 1e-17 {
                let p = ln_front.exp() * sum;
                return Ok((p, 1.0 - p));
            }
            k += 1.0;
        }
        return Err(SpecialError::NoConvergence {
            function: "inc_gamma",
            iterations: GAMMA_MAX_ITER,
        });
    }
    // Q = x^a e^-x / G(a) * 1 / (x + 1 - a - 1(1-a) / (x + 3 - a - 2(2-a) / (x + 5 - a - ...)))
    let b0 = x + 1.0 - a;
    let mut f = if b0 == 0.0 { TINY } else { b0 };
    let mut c = f;
    let mut d = 0.0f64;
    for n in 1..GAMMA_MAX_ITER {
        let nf = n as f64;
        let an = -nf * (nf - a);
        let bn = x + 2.0 * nf + 1.0 - a;
        d = bn + an * d;
        if d == 0.0 {
            d = TINY;
        }
        c = bn + an / c;
        if c == 0.0 {
            c = TINY;
        }
        d = 1.0 / d;
        let delta = c * d;
        f *= delta;
        if (delta - 1.0).abs() < CF_EPS {
            let q = ln_front.exp() / f;
            return Ok((1.0 - q, q));
        }
    }
    Err(SpecialError::NoConvergence { function: "inc_gamma", iterations: GAMMA_MAX_ITER })
}

/// Regularised lower incomplete gamma function `P(a, x)`.
pub fn inc_gamma_p(a: f64, x: f64) -> Result<f64, SpecialError> {
    Ok(inc_gamma_pq(a, x)?.0)
}

/// Regularised upper incomplete gamma function `Q(a, x)`.
pub fn inc_gamma_q(a: f64, x: f64) -> Result<f64, SpecialError> {
    Ok(inc_gamma_pq(a, x)?.1)
}

/// Student t survival function `P(T > x)` with `df > 0` degrees of freedom. Infinite `x` gives
/// 0 or 1.
pub fn t_sf(x: f64, df: f64) -> Result<f64, SpecialError> {
    check_positive("t_sf", "df", df)?;
    check_number("t_sf", "x", x)?;
    if x.is_infinite() {
        return Ok(if x > 0.0 { 0.0 } else { 1.0 });
    }
    let two_sided = t_two_sided(x, df)?;
    Ok(if x >= 0.0 { 0.5 * two_sided } else { 1.0 - 0.5 * two_sided })
}

/// `P(|T| > |x|) = I_z(df / 2, 1 / 2)` with `z = df / (df + x^2)`, for finite `x`.
fn t_two_sided(x: f64, df: f64) -> Result<f64, SpecialError> {
    let x2 = x * x;
    let (z, w) = if x2.is_finite() {
        let denom = df + x2;
        (df / denom, x2 / denom)
    } else {
        ((df / x) / x, 1.0)
    };
    Ok(inc_beta_pq(0.5 * df, 0.5, z, w)?.0)
}

/// Student t distribution function `P(T <= x)`.
pub fn t_cdf(x: f64, df: f64) -> Result<f64, SpecialError> {
    check_number("t_cdf", "x", x)?;
    t_sf(-x, df)
}

/// Two-sided p-value of a t statistic, `P(|T| > |x|)`.
pub fn t_two_sided_p(x: f64, df: f64) -> Result<f64, SpecialError> {
    check_positive("t_two_sided_p", "df", df)?;
    check_number("t_two_sided_p", "x", x)?;
    if x.is_infinite() {
        return Ok(0.0);
    }
    t_two_sided(x, df)
}

/// Chi-squared distribution function with `df > 0` degrees of freedom.
pub fn chi2_cdf(x: f64, df: f64) -> Result<f64, SpecialError> {
    check_positive("chi2_cdf", "df", df)?;
    check_number("chi2_cdf", "x", x)?;
    if x <= 0.0 {
        return Ok(0.0);
    }
    inc_gamma_p(0.5 * df, 0.5 * x)
}

/// Chi-squared survival function.
pub fn chi2_sf(x: f64, df: f64) -> Result<f64, SpecialError> {
    check_positive("chi2_sf", "df", df)?;
    check_number("chi2_sf", "x", x)?;
    if x <= 0.0 {
        return Ok(1.0);
    }
    inc_gamma_q(0.5 * df, 0.5 * x)
}

/// F distribution function with `d1, d2 > 0` degrees of freedom.
pub fn f_cdf(x: f64, d1: f64, d2: f64) -> Result<f64, SpecialError> {
    Ok(f_pq(x, d1, d2, "f_cdf")?.0)
}

/// F survival function.
pub fn f_sf(x: f64, d1: f64, d2: f64) -> Result<f64, SpecialError> {
    Ok(f_pq(x, d1, d2, "f_sf")?.1)
}

fn f_pq(x: f64, d1: f64, d2: f64, function: &'static str) -> Result<(f64, f64), SpecialError> {
    check_positive(function, "d1", d1)?;
    check_positive(function, "d2", d2)?;
    check_number(function, "x", x)?;
    if x <= 0.0 {
        return Ok((0.0, 1.0));
    }
    if x == f64::INFINITY {
        return Ok((1.0, 0.0));
    }
    let num = d1 * x;
    let denom = num + d2;
    inc_beta_pq(0.5 * d1, 0.5 * d2, num / denom, d2 / denom)
}
