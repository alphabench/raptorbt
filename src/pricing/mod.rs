//! European option pricing, Greeks and implied volatility.
//!
//! Two models, both lognormal:
//!
//! * [`Model::BlackScholes`]: the underlying is a spot price; the forward is
//!   `S * exp((r - q) * tau)` with a continuous rate `r` and dividend (or
//!   other carry) yield `q` (Black-Scholes-Merton).
//! * [`Model::Black76`]: the underlying is already a forward (a futures
//!   price); no carry is applied and `dividend_yield` must be zero. The rate
//!   only discounts the premium.
//!
//! Conventions: `tau` is in years; rates and yields are continuously
//! compounded; `vega` is per 1.00 of volatility, `rho` per 1.00 of rate and
//! `theta` per year (negative when the option decays); `delta` is with
//! respect to the underlying handed in (the forward for Black-76).
//!
//! Edge cases are defined, not guessed:
//!
//! * `tau == 0` (expiry): the price is the intrinsic value; delta is +-1 when
//!   in the money and 0 otherwise; the other Greeks are 0.
//! * `vol == 0` with `tau > 0`: the price is the discounted intrinsic value on
//!   the forward and every Greek is its exact derivative of that payoff
//!   (delta jumps at the forward; gamma and vega are 0).
//! * Invalid inputs (non-positive spot or strike, negative tau or vol, any
//!   non-finite number) are an error from the checked functions and NaN from
//!   the `*_raw` kernels. NaN always means "absent", never zero.
//!
//! Implied volatility is an explicit outcome: [`IvStatus`] says why no
//! volatility exists or was found. Nothing is clamped or guessed.

mod chain;
mod iv;
pub mod norm;

use thiserror::Error;

pub use crate::instruments::OptionRight;
pub use chain::{ChainError, ExpiryPick, OptionChain, YEAR_NS};
pub use iv::{implied_vol, implied_vol_raw, IvSearch, IvStatus};
pub use norm::{norm_cdf, norm_pdf};

/// Which pricing formula to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Model {
    /// Spot underlying, continuous rate and dividend yield.
    BlackScholes,
    /// Forward underlying (futures); no carry.
    Black76,
}

impl Model {
    /// Parse `"black_scholes"` or `"black76"`.
    pub fn parse(text: &str) -> Result<Self, PricingError> {
        match text {
            "black_scholes" => Ok(Model::BlackScholes),
            "black76" => Ok(Model::Black76),
            other => Err(PricingError::InvalidInput(format!(
                "model is one of black_scholes, black76, got {other:?}"
            ))),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Model::BlackScholes => "black_scholes",
            Model::Black76 => "black76",
        }
    }
}

/// A rejected input.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum PricingError {
    #[error("{0}")]
    InvalidInput(String),
}

/// Everything a model prices. `underlying` is the spot for
/// [`Model::BlackScholes`] and the forward for [`Model::Black76`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OptionInputs {
    pub right: OptionRight,
    pub underlying: f64,
    pub strike: f64,
    /// Years to expiry.
    pub tau: f64,
    /// Lognormal volatility per square-root year.
    pub vol: f64,
    pub rate: f64,
    pub dividend_yield: f64,
}

impl OptionInputs {
    /// Refuse inputs no model can price. `vol` is not checked here so the
    /// implied-volatility solver can reuse the same checks for the rest.
    pub fn validate_market(&self, model: Model) -> Result<(), PricingError> {
        let bad = |m: &str| Err(PricingError::InvalidInput(m.to_string()));
        if !(self.underlying.is_finite() && self.strike.is_finite()) {
            return bad("underlying and strike must be finite");
        }
        if self.underlying <= 0.0 || self.strike <= 0.0 {
            return bad("underlying and strike must be positive");
        }
        if !self.tau.is_finite() || self.tau < 0.0 {
            return bad("tau must be finite and non-negative");
        }
        if !(self.rate.is_finite() && self.dividend_yield.is_finite()) {
            return bad("rate and dividend_yield must be finite");
        }
        if model == Model::Black76 && self.dividend_yield != 0.0 {
            return bad("black76 prices a forward: dividend_yield must be 0");
        }
        Ok(())
    }

    pub fn validate(&self, model: Model) -> Result<(), PricingError> {
        self.validate_market(model)?;
        if !self.vol.is_finite() || self.vol < 0.0 {
            return Err(PricingError::InvalidInput("vol must be finite and non-negative".into()));
        }
        Ok(())
    }

    /// The forward the model prices on.
    #[inline]
    pub fn forward(&self, model: Model) -> f64 {
        match model {
            Model::BlackScholes => {
                self.underlying * ((self.rate - self.dividend_yield) * self.tau).exp()
            }
            Model::Black76 => self.underlying,
        }
    }
}

/// A price and its sensitivities (units in the module docs).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Greeks {
    pub price: f64,
    pub delta: f64,
    pub gamma: f64,
    pub vega: f64,
    pub theta: f64,
    pub rho: f64,
}

impl Greeks {
    pub const NAN: Greeks = Greeks {
        price: f64::NAN,
        delta: f64::NAN,
        gamma: f64::NAN,
        vega: f64::NAN,
        theta: f64::NAN,
        rho: f64::NAN,
    };
}

#[inline]
fn is_call(right: OptionRight) -> bool {
    right == OptionRight::Call
}

/// Normal-CDF terms of the model, or their limit when `vol * sqrt(tau)` is 0.
struct Terms {
    /// N(d1), N(d2), N(-d1), N(-d2).
    n_d1: f64,
    n_d2: f64,
    n_md1: f64,
    n_md2: f64,
    /// phi(d1), 0 in the zero-vol limit.
    pdf_d1: f64,
}

/// Terms for `tau > 0`. With `vol == 0` they are the step functions of the
/// payoff: 1 for in the money on the forward, 0 otherwise.
#[inline]
fn terms(fwd: f64, k: f64, tau: f64, vol: f64) -> Terms {
    if vol == 0.0 {
        let up = if fwd > k { 1.0 } else { 0.0 };
        let down = if fwd < k { 1.0 } else { 0.0 };
        return Terms { n_d1: up, n_d2: up, n_md1: down, n_md2: down, pdf_d1: 0.0 };
    }
    let sd = vol * tau.sqrt();
    let d1 = ((fwd / k).ln() + 0.5 * sd * sd) / sd;
    let d2 = d1 - sd;
    Terms {
        n_d1: norm_cdf(d1),
        n_d2: norm_cdf(d2),
        n_md1: norm_cdf(-d1),
        n_md2: norm_cdf(-d2),
        pdf_d1: norm_pdf(d1),
    }
}

/// The price for inputs that passed [`OptionInputs::validate`]; NaN inputs
/// give NaN. Use [`price`] to get an error instead.
pub fn price_raw(model: Model, i: &OptionInputs) -> f64 {
    let (k, tau) = (i.strike, i.tau);
    let fwd = i.forward(model);
    let call = is_call(i.right);
    if tau == 0.0 {
        return if call { (fwd - k).max(0.0) } else { (k - fwd).max(0.0) };
    }
    let df = (-i.rate * tau).exp();
    let t = terms(fwd, k, tau, i.vol);
    if call {
        df * (fwd * t.n_d1 - k * t.n_d2)
    } else {
        df * (k * t.n_md2 - fwd * t.n_md1)
    }
}

/// The price and Greeks for validated inputs; NaN inputs give NaN.
pub fn greeks_raw(model: Model, i: &OptionInputs) -> Greeks {
    let (k, tau, vol) = (i.strike, i.tau, i.vol);
    let (r, q) = (i.rate, i.dividend_yield);
    let fwd = i.forward(model);
    let call = is_call(i.right);
    if tau == 0.0 {
        let itm = if call { fwd > k } else { fwd < k };
        let sign = if call { 1.0 } else { -1.0 };
        let price = if call { (fwd - k).max(0.0) } else { (k - fwd).max(0.0) };
        return Greeks {
            price,
            delta: if itm { sign } else { 0.0 },
            gamma: 0.0,
            vega: 0.0,
            theta: 0.0,
            rho: 0.0,
        };
    }
    let df = (-r * tau).exp();
    let sqrt_tau = tau.sqrt();
    let t = terms(fwd, k, tau, vol);
    let spot_model = model == Model::BlackScholes;
    // d(forward)/d(underlying): carry for a spot, 1 for a forward.
    let carry = if spot_model { ((r - q) * tau).exp() } else { 1.0 };

    let price =
        if call { df * (fwd * t.n_d1 - k * t.n_d2) } else { df * (k * t.n_md2 - fwd * t.n_md1) };
    let delta = if call { df * carry * t.n_d1 } else { -df * carry * t.n_md1 };
    let gamma =
        if vol == 0.0 { 0.0 } else { df * t.pdf_d1 / (fwd * vol * sqrt_tau) * carry * carry };
    let vega = df * fwd * t.pdf_d1 * sqrt_tau;
    let decay = -df * fwd * t.pdf_d1 * vol / (2.0 * sqrt_tau);
    let (theta, rho) = match (call, spot_model) {
        (true, true) => {
            (decay - r * k * df * t.n_d2 + q * df * fwd * t.n_d1, k * tau * df * t.n_d2)
        }
        (false, true) => {
            (decay + r * k * df * t.n_md2 - q * df * fwd * t.n_md1, -k * tau * df * t.n_md2)
        }
        // With the forward held fixed the premium accretes at the rate:
        // theta = -dP/dtau = r * P + decay.
        (_, false) => (decay + r * price, -tau * price),
    };
    Greeks { price, delta, gamma, vega, theta, rho }
}

/// The price, or an error for inputs no model can price.
pub fn price(model: Model, i: &OptionInputs) -> Result<f64, PricingError> {
    i.validate(model)?;
    Ok(price_raw(model, i))
}

/// The price and Greeks, or an error for inputs no model can price.
pub fn greeks(model: Model, i: &OptionInputs) -> Result<Greeks, PricingError> {
    i.validate(model)?;
    Ok(greeks_raw(model, i))
}

/// Like [`price_raw`] but NaN for any invalid input instead of trusting the
/// caller: the entry point for arrays.
pub fn price_or_nan(model: Model, i: &OptionInputs) -> f64 {
    if i.validate(model).is_ok() {
        price_raw(model, i)
    } else {
        f64::NAN
    }
}

/// Like [`greeks_raw`] but all-NaN for any invalid input.
pub fn greeks_or_nan(model: Model, i: &OptionInputs) -> Greeks {
    if i.validate(model).is_ok() {
        greeks_raw(model, i)
    } else {
        Greeks::NAN
    }
}

/// `S * exp((r - q) * tau)`.
pub fn forward_price(spot: f64, tau: f64, rate: f64, dividend_yield: f64) -> f64 {
    spot * ((rate - dividend_yield) * tau).exp()
}

#[cfg(test)]
mod tests;
