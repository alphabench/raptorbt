//! Implied volatility by a safeguarded Newton iteration.
//!
//! The solver brackets the answer in `[lo, hi]`, takes Newton steps on the
//! analytic vega while they stay inside the bracket and make progress, and
//! bisects otherwise. The same inputs give the same answer on every platform.
//!
//! A price with no volatility in the bracket is reported with the reason
//! ([`IvStatus`]); the result is NaN, never a clamped or guessed number.

use super::{price_raw, Model, OptionInputs, OptionRight, PricingError};

/// Why an implied volatility exists or does not.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum IvStatus {
    /// Solved: the returned volatility reprices the target.
    Ok = 0,
    /// `tau == 0`: an expired option has no volatility.
    Expired = 1,
    /// An input is invalid or not finite.
    InvalidInput = 2,
    /// The target is negative or below the discounted intrinsic value: no
    /// volatility gives it (an arbitrage).
    BelowIntrinsic = 3,
    /// The target is at or above the model's upper bound (the discounted
    /// forward for a call, strike for a put).
    AboveUpperBound = 4,
    /// The target needs a volatility at or below `lo` (zero time value, or a
    /// price of exactly zero, counts).
    BelowBracket = 5,
    /// The target needs a volatility at or above `hi`.
    AboveBracket = 6,
    /// The iteration budget ran out inside the bracket.
    NoConvergence = 7,
}

impl IvStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            IvStatus::Ok => "ok",
            IvStatus::Expired => "expired",
            IvStatus::InvalidInput => "invalid_input",
            IvStatus::BelowIntrinsic => "below_intrinsic",
            IvStatus::AboveUpperBound => "above_upper_bound",
            IvStatus::BelowBracket => "below_bracket",
            IvStatus::AboveBracket => "above_bracket",
            IvStatus::NoConvergence => "no_convergence",
        }
    }

    pub fn from_code(code: u8) -> Option<Self> {
        Some(match code {
            0 => IvStatus::Ok,
            1 => IvStatus::Expired,
            2 => IvStatus::InvalidInput,
            3 => IvStatus::BelowIntrinsic,
            4 => IvStatus::AboveUpperBound,
            5 => IvStatus::BelowBracket,
            6 => IvStatus::AboveBracket,
            7 => IvStatus::NoConvergence,
            _ => return None,
        })
    }
}

/// Bracket and stopping rule. Defaults: `[1e-4, 5]` (0.01% to 500%),
/// a volatility tolerance of `1e-12`, 200 iterations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IvSearch {
    pub lo: f64,
    pub hi: f64,
    /// Stop when the bracket or the step is narrower than this (volatility
    /// units).
    pub tol: f64,
    pub max_iter: u32,
}

impl Default for IvSearch {
    fn default() -> Self {
        Self { lo: 1e-4, hi: 5.0, tol: 1e-12, max_iter: 200 }
    }
}

impl IvSearch {
    pub fn validate(&self) -> Result<(), PricingError> {
        let ok = self.lo.is_finite()
            && self.hi.is_finite()
            && self.lo > 0.0
            && self.hi > self.lo
            && self.tol.is_finite()
            && self.tol > 0.0
            && self.max_iter >= 1;
        if ok {
            Ok(())
        } else {
            Err(PricingError::InvalidInput(
                "search needs 0 < lo < hi, tol > 0 and max_iter >= 1".into(),
            ))
        }
    }
}

/// Implied volatility of `target` for the option in `base` (its `vol` is
/// ignored). `Err(status)` carries the reason there is none.
pub fn implied_vol(
    model: Model,
    base: &OptionInputs,
    target: f64,
    search: &IvSearch,
) -> Result<f64, IvStatus> {
    let (iv, status) = implied_vol_raw(model, base, target, search);
    if status == IvStatus::Ok {
        Ok(iv)
    } else {
        Err(status)
    }
}

/// The solver with a flat result: `(NaN, status)` when there is none.
pub fn implied_vol_raw(
    model: Model,
    base: &OptionInputs,
    target: f64,
    search: &IvSearch,
) -> (f64, IvStatus) {
    let fail = |s: IvStatus| (f64::NAN, s);
    if base.validate_market(model).is_err() || !target.is_finite() {
        return fail(IvStatus::InvalidInput);
    }
    if base.tau == 0.0 {
        return fail(IvStatus::Expired);
    }
    let (k, tau) = (base.strike, base.tau);
    let fwd = base.forward(model);
    let df = (-base.rate * tau).exp();
    let call = base.right == OptionRight::Call;
    let lower = df * if call { (fwd - k).max(0.0) } else { (k - fwd).max(0.0) };
    let upper = df * if call { fwd } else { k };
    if target < lower * (1.0 - 1e-12) || target < 0.0 {
        return fail(IvStatus::BelowIntrinsic);
    }
    if target == 0.0 {
        // An out-of-the-money option worth nothing: no volatility is small
        // enough, and a price that underflowed is not a quote.
        return fail(IvStatus::BelowBracket);
    }
    if target >= upper {
        return fail(IvStatus::AboveUpperBound);
    }
    let at = |vol: f64| price_raw(model, &OptionInputs { vol, ..*base }) - target;
    let (mut a, mut b) = (search.lo, search.hi);
    let (fa, fb) = (at(a), at(b));
    if fa.is_nan() || fb.is_nan() {
        return fail(IvStatus::InvalidInput);
    }
    // The target must lie strictly between the prices at the bracket's ends;
    // a price equal to the one at `lo` is zero time value, not a volatility.
    if fa >= 0.0 {
        return fail(IvStatus::BelowBracket);
    }
    if fb <= 0.0 {
        return fail(IvStatus::AboveBracket);
    }

    let sqrt_tau = tau.sqrt();
    // Brenner-Subrahmanyam start, clamped into the bracket.
    let guess = (target / df / fwd.min(k)) * (2.0 * std::f64::consts::PI / tau).sqrt();
    let mut sigma = guess.clamp(a, b);
    if !(sigma > a && sigma < b) {
        sigma = 0.5 * (a + b);
    }
    let mut prev_abs = f64::INFINITY;
    // Bracket widths two iterations back: when Newton is not halving the
    // bracket, bisect, so the bracket always closes.
    let (mut w1, mut w2) = (f64::INFINITY, f64::INFINITY);
    for _ in 0..search.max_iter {
        let f = at(sigma);
        if f == 0.0 {
            return (sigma, IvStatus::Ok);
        }
        if f > 0.0 {
            b = sigma;
        } else {
            a = sigma;
        }
        if b - a <= search.tol {
            return (0.5 * (a + b), IvStatus::Ok);
        }
        // Newton step on the analytic vega, kept only while it stays inside
        // the bracket and the residual is shrinking at least by half.
        let mut next = f64::NAN;
        let closing = (b - a) <= 0.5 * w2;
        w2 = w1;
        w1 = b - a;
        if closing && f.abs() <= 0.5 * prev_abs {
            let sd = sigma * sqrt_tau;
            let d1 = ((fwd / k).ln() + 0.5 * sd * sd) / sd;
            let vega = df * fwd * super::norm_pdf(d1) * sqrt_tau;
            if vega > 0.0 && vega.is_finite() {
                next = sigma - f / vega;
            }
        }
        prev_abs = f.abs();
        if !(next > a && next < b) {
            next = 0.5 * (a + b);
        }
        if (next - sigma).abs() <= search.tol {
            return (next, IvStatus::Ok);
        }
        sigma = next;
    }
    fail(IvStatus::NoConvergence)
}
