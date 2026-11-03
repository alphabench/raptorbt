//! Python bindings for option pricing, Greeks, implied volatility and the
//! option chain.
//!
//! Array entry points borrow the NumPy buffers without copying, release the
//! GIL while they compute and split large inputs across threads. Invalid
//! elements come back as NaN (and an implied-volatility status), never zero;
//! the scalar entry points raise `ValueError` instead.

use std::cell::RefCell;

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

use super::errors::{invalid_input, Fail};
use super::instrument_bindings::PyInstrumentSpec;
use super::numpy_bridge::{numpy_cow_f64, vec_to_numpy_f64};
use crate::pricing::{
    self, ExpiryPick, Greeks, IvSearch, IvStatus, Model, OptionChain, OptionInputs, OptionRight,
    PricingError,
};

/// Arrays shorter than this are computed on the calling thread.
const PARALLEL_MIN: usize = 4096;

fn value_err(e: impl std::fmt::Display) -> PyErr {
    invalid_input(e)
}

const EXPIRY_PICKS: [&str; 5] = ["nearest", "nth", "weekly", "monthly", "date"];

fn model_of(text: &str) -> PyResult<Model> {
    Model::parse(text).map_err(|PricingError::InvalidInput(m)| value_err(m))
}

fn right_of(is_call: bool) -> OptionRight {
    if is_call {
        OptionRight::Call
    } else {
        OptionRight::Put
    }
}

fn parse_right(text: &str) -> PyResult<OptionRight> {
    match text.to_ascii_lowercase().as_str() {
        "call" | "c" => Ok(OptionRight::Call),
        "put" | "p" => Ok(OptionRight::Put),
        _ => Err(value_err(format!("right is call or put, got {text:?}"))),
    }
}

/// A price and its sensitivities.
#[pyclass(frozen, name = "Greeks", module = "raptorbt._raptorbt", get_all)]
#[derive(Debug, Clone, Copy)]
pub struct PyGreeks {
    price: f64,
    delta: f64,
    gamma: f64,
    vega: f64,
    theta: f64,
    rho: f64,
}

#[pymethods]
impl PyGreeks {
    fn __repr__(&self) -> String {
        format!(
            "Greeks(price={:?}, delta={:?}, gamma={:?}, vega={:?}, theta={:?}, rho={:?})",
            self.price, self.delta, self.gamma, self.vega, self.theta, self.rho
        )
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other.cast::<Self>().is_ok_and(|o| {
            let o = o.get();
            [self.price, self.delta, self.gamma, self.vega, self.theta, self.rho]
                == [o.price, o.delta, o.gamma, o.vega, o.theta, o.rho]
        })
    }

    fn as_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("price", self.price)?;
        d.set_item("delta", self.delta)?;
        d.set_item("gamma", self.gamma)?;
        d.set_item("vega", self.vega)?;
        d.set_item("theta", self.theta)?;
        d.set_item("rho", self.rho)?;
        Ok(d)
    }
}

impl From<Greeks> for PyGreeks {
    fn from(g: Greeks) -> Self {
        Self {
            price: g.price,
            delta: g.delta,
            gamma: g.gamma,
            vega: g.vega,
            theta: g.theta,
            rho: g.rho,
        }
    }
}

fn inputs(
    is_call: bool,
    underlying: f64,
    strike: f64,
    tau: f64,
    vol: f64,
    rate: f64,
    dividend_yield: f64,
) -> OptionInputs {
    OptionInputs { right: right_of(is_call), underlying, strike, tau, vol, rate, dividend_yield }
}

/// Price of one option; raises `ValueError` for inputs no model can price.
#[pyfunction]
#[pyo3(signature = (model, is_call, underlying, strike, tau, vol, rate=0.0, dividend_yield=0.0))]
#[allow(clippy::too_many_arguments)]
pub fn pricing_price(
    model: &str,
    is_call: bool,
    underlying: f64,
    strike: f64,
    tau: f64,
    vol: f64,
    rate: f64,
    dividend_yield: f64,
) -> PyResult<f64> {
    let i = inputs(is_call, underlying, strike, tau, vol, rate, dividend_yield);
    pricing::price(model_of(model)?, &i).map_err(|PricingError::InvalidInput(m)| value_err(m))
}

/// Price and Greeks of one option; raises `ValueError` for bad inputs.
#[pyfunction]
#[pyo3(signature = (model, is_call, underlying, strike, tau, vol, rate=0.0, dividend_yield=0.0))]
#[allow(clippy::too_many_arguments)]
pub fn pricing_greeks(
    model: &str,
    is_call: bool,
    underlying: f64,
    strike: f64,
    tau: f64,
    vol: f64,
    rate: f64,
    dividend_yield: f64,
) -> PyResult<PyGreeks> {
    let i = inputs(is_call, underlying, strike, tau, vol, rate, dividend_yield);
    pricing::greeks(model_of(model)?, &i)
        .map(PyGreeks::from)
        .map_err(|PricingError::InvalidInput(m)| value_err(m))
}

/// Implied volatility of one price as `(iv, status_code)`; `iv` is NaN unless
/// the status is 0. Raises `ValueError` only for a bad search bracket or model.
#[pyfunction]
#[pyo3(signature = (model, price, is_call, underlying, strike, tau, rate=0.0,
                    dividend_yield=0.0, lo=1e-4, hi=5.0, tol=1e-12, max_iter=200))]
#[allow(clippy::too_many_arguments)]
pub fn pricing_implied_vol(
    model: &str,
    price: f64,
    is_call: bool,
    underlying: f64,
    strike: f64,
    tau: f64,
    rate: f64,
    dividend_yield: f64,
    lo: f64,
    hi: f64,
    tol: f64,
    max_iter: u32,
) -> PyResult<(f64, u8)> {
    let search = IvSearch { lo, hi, tol, max_iter };
    search.validate().map_err(|PricingError::InvalidInput(m)| value_err(m))?;
    let base = inputs(is_call, underlying, strike, tau, 0.0, rate, dividend_yield);
    let (iv, status) = pricing::implied_vol_raw(model_of(model)?, &base, price, &search);
    Ok((iv, status as u8))
}

/// Run `f` over `0..n`, in parallel when `n` is large.
fn map_indexed<T: Send>(n: usize, f: impl Fn(usize) -> T + Sync + Send) -> Vec<T> {
    if n >= PARALLEL_MIN {
        (0..n).into_par_iter().map(f).collect()
    } else {
        (0..n).map(f).collect()
    }
}

fn same_len(lens: &[usize]) -> PyResult<usize> {
    let n = lens[0];
    if lens.iter().any(|&l| l != n) {
        return Err(value_err("all input arrays must have the same length"));
    }
    Ok(n)
}

/// Borrow a bool array when contiguous, copy it when it is not.
fn bools<'a>(arr: &'a PyReadonlyArray1<'_, bool>) -> std::borrow::Cow<'a, [bool]> {
    match arr.as_slice() {
        Ok(slice) => std::borrow::Cow::Borrowed(slice),
        Err(_) => std::borrow::Cow::Owned(arr.as_array().iter().copied().collect()),
    }
}

/// `(iv, status)`.
type IvArrays<'py> = (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<u8>>);

type Greeks6<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
);

/// Prices of many options. Inputs are equal-length arrays (a strided view is copied);
/// an invalid element prices as NaN.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn pricing_price_array<'py>(
    py: Python<'py>,
    model: &str,
    is_call: PyReadonlyArray1<'py, bool>,
    underlying: PyReadonlyArray1<'py, f64>,
    strike: PyReadonlyArray1<'py, f64>,
    tau: PyReadonlyArray1<'py, f64>,
    vol: PyReadonlyArray1<'py, f64>,
    rate: PyReadonlyArray1<'py, f64>,
    dividend_yield: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let model = model_of(model)?;
    let (c, s, k, t, v, r, q) = (
        bools(&is_call),
        numpy_cow_f64(&underlying),
        numpy_cow_f64(&strike),
        numpy_cow_f64(&tau),
        numpy_cow_f64(&vol),
        numpy_cow_f64(&rate),
        numpy_cow_f64(&dividend_yield),
    );
    let n = same_len(&[c.len(), s.len(), k.len(), t.len(), v.len(), r.len(), q.len()])?;
    let out = py.detach(|| {
        map_indexed(n, |i| {
            pricing::price_or_nan(model, &inputs(c[i], s[i], k[i], t[i], v[i], r[i], q[i]))
        })
    });
    Ok(vec_to_numpy_f64(py, out))
}

/// Price and Greeks of many options, as six arrays
/// `(price, delta, gamma, vega, theta, rho)`; an invalid element is NaN in all.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn pricing_greeks_array<'py>(
    py: Python<'py>,
    model: &str,
    is_call: PyReadonlyArray1<'py, bool>,
    underlying: PyReadonlyArray1<'py, f64>,
    strike: PyReadonlyArray1<'py, f64>,
    tau: PyReadonlyArray1<'py, f64>,
    vol: PyReadonlyArray1<'py, f64>,
    rate: PyReadonlyArray1<'py, f64>,
    dividend_yield: PyReadonlyArray1<'py, f64>,
) -> PyResult<Greeks6<'py>> {
    let model = model_of(model)?;
    let (c, s, k, t, v, r, q) = (
        bools(&is_call),
        numpy_cow_f64(&underlying),
        numpy_cow_f64(&strike),
        numpy_cow_f64(&tau),
        numpy_cow_f64(&vol),
        numpy_cow_f64(&rate),
        numpy_cow_f64(&dividend_yield),
    );
    let n = same_len(&[c.len(), s.len(), k.len(), t.len(), v.len(), r.len(), q.len()])?;
    let g = py.detach(|| {
        map_indexed(n, |i| {
            pricing::greeks_or_nan(model, &inputs(c[i], s[i], k[i], t[i], v[i], r[i], q[i]))
        })
    });
    let col = |f: fn(&Greeks) -> f64| vec_to_numpy_f64(py, g.iter().map(f).collect());
    Ok((
        col(|g| g.price),
        col(|g| g.delta),
        col(|g| g.gamma),
        col(|g| g.vega),
        col(|g| g.theta),
        col(|g| g.rho),
    ))
}

/// Implied volatilities of many prices as `(iv, status)` arrays; `iv` is NaN
/// wherever `status` is not 0.
#[pyfunction]
#[pyo3(signature = (model, price, is_call, underlying, strike, tau, rate, dividend_yield,
                    lo=1e-4, hi=5.0, tol=1e-12, max_iter=200))]
#[allow(clippy::too_many_arguments)]
pub fn pricing_implied_vol_array<'py>(
    py: Python<'py>,
    model: &str,
    price: PyReadonlyArray1<'py, f64>,
    is_call: PyReadonlyArray1<'py, bool>,
    underlying: PyReadonlyArray1<'py, f64>,
    strike: PyReadonlyArray1<'py, f64>,
    tau: PyReadonlyArray1<'py, f64>,
    rate: PyReadonlyArray1<'py, f64>,
    dividend_yield: PyReadonlyArray1<'py, f64>,
    lo: f64,
    hi: f64,
    tol: f64,
    max_iter: u32,
) -> PyResult<IvArrays<'py>> {
    let model = model_of(model)?;
    let search = IvSearch { lo, hi, tol, max_iter };
    search.validate().map_err(|PricingError::InvalidInput(m)| value_err(m))?;
    let (p, c, s, k, t, r, q) = (
        numpy_cow_f64(&price),
        bools(&is_call),
        numpy_cow_f64(&underlying),
        numpy_cow_f64(&strike),
        numpy_cow_f64(&tau),
        numpy_cow_f64(&rate),
        numpy_cow_f64(&dividend_yield),
    );
    let n = same_len(&[p.len(), c.len(), s.len(), k.len(), t.len(), r.len(), q.len()])?;
    let res = py.detach(|| {
        map_indexed(n, |i| {
            let base = inputs(c[i], s[i], k[i], t[i], 0.0, r[i], q[i]);
            pricing::implied_vol_raw(model, &base, p[i], &search)
        })
    });
    let iv: Vec<f64> = res.iter().map(|&(v, _)| v).collect();
    let status: Vec<u8> = res.iter().map(|&(_, s)| s as u8).collect();
    Ok((vec_to_numpy_f64(py, iv), PyArray1::from_vec(py, status)))
}

/// The name of an implied-volatility status code.
#[pyfunction]
pub fn pricing_status_name(code: u8) -> PyResult<&'static str> {
    IvStatus::from_code(code)
        .map(IvStatus::as_str)
        .ok_or_else(|| value_err(format!("no implied-volatility status {code}")))
}

/// Years between two timestamps (nanoseconds), on a 365-day year.
#[pyfunction]
pub fn pricing_year_fraction(now_ns: i64, expiry_ns: i64) -> f64 {
    (expiry_ns - now_ns) as f64 / pricing::YEAR_NS as f64
}

/// The forward `S * exp((rate - dividend_yield) * tau)`.
#[pyfunction]
#[pyo3(signature = (spot, tau, rate=0.0, dividend_yield=0.0))]
pub fn pricing_forward(spot: f64, tau: f64, rate: f64, dividend_yield: f64) -> f64 {
    pricing::forward_price(spot, tau, rate, dividend_yield)
}

/// Option contracts by underlying, expiry and strike, visible only as they
/// are listed. Every query takes the current event time `now_ns`.
#[pyclass(frozen, name = "OptionChain", module = "raptorbt._raptorbt")]
pub struct PyOptionChain {
    inner: OptionChain,
}

fn spec_out(s: Option<&crate::instruments::InstrumentSpec>) -> Option<PyInstrumentSpec> {
    s.map(|s| PyInstrumentSpec { inner: s.clone() })
}

fn right_opt(right: Option<&str>) -> PyResult<Option<OptionRight>> {
    right.map(parse_right).transpose()
}

#[pymethods]
impl PyOptionChain {
    /// Build from `InstrumentSpec` objects. Specs that are not options are
    /// ignored; two contracts with the same terms are refused.
    #[new]
    fn new(specs: Vec<PyRef<'_, PyInstrumentSpec>>) -> PyResult<Self> {
        let specs = specs.iter().map(|s| s.inner.clone());
        OptionChain::new(specs).map(|inner| Self { inner }).map_err(value_err)
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn underlyings(&self) -> Vec<String> {
        self.inner.underlyings().map(str::to_string).collect()
    }

    /// Expiries with a contract listed at `now_ns`, ascending.
    fn expiries(&self, underlying: &str, now_ns: i64) -> Vec<i64> {
        self.inner.expiries(underlying, now_ns)
    }

    /// Strikes listed at `now_ns` for one expiry, ascending; one right if given.
    #[pyo3(signature = (underlying, expiry_ns, now_ns, right=None))]
    fn strikes(
        &self,
        underlying: &str,
        expiry_ns: i64,
        now_ns: i64,
        right: Option<&str>,
    ) -> PyResult<Vec<f64>> {
        Ok(self.inner.strikes(underlying, expiry_ns, now_ns, right_opt(right)?))
    }

    /// The contract at exactly these terms, or None if not listed at `now_ns`.
    fn contract(
        &self,
        underlying: &str,
        expiry_ns: i64,
        strike: f64,
        right: &str,
        now_ns: i64,
    ) -> PyResult<Option<PyInstrumentSpec>> {
        let right = parse_right(right)?;
        Ok(spec_out(self.inner.contract(underlying, expiry_ns, strike, right, now_ns)))
    }

    /// Contracts of one expiry listed at `now_ns`, by ascending strike.
    #[pyo3(signature = (underlying, expiry_ns, now_ns, right=None))]
    fn contracts(
        &self,
        underlying: &str,
        expiry_ns: i64,
        now_ns: i64,
        right: Option<&str>,
    ) -> PyResult<Vec<PyInstrumentSpec>> {
        let right = right_opt(right)?;
        Ok(self
            .inner
            .contracts(underlying, expiry_ns, now_ns, right)
            .into_iter()
            .map(|s| PyInstrumentSpec { inner: s.clone() })
            .collect())
    }

    /// Pick an expiry listed at `now_ns`. `which` is `"nearest"` (first at or
    /// after `min_days` days from now), `"nth"` (0 = nearest), `"weekly"` /
    /// `"monthly"` (the `n`th that is not / is the last of its calendar
    /// month) or `"date"` (the one on the same day as `date_ns`). Months and
    /// days follow `utc_offset_ns`.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (underlying, now_ns, which="nearest", n=0, min_days=0, date_ns=None,
                        utc_offset_ns=0))]
    fn pick_expiry(
        &self,
        underlying: &str,
        now_ns: i64,
        which: &str,
        n: usize,
        min_days: i64,
        date_ns: Option<i64>,
        utc_offset_ns: i64,
    ) -> PyResult<Option<i64>> {
        let pick = match which {
            "nearest" => ExpiryPick::Nearest { min_days },
            "nth" => ExpiryPick::Nth(n),
            "weekly" => ExpiryPick::Weekly(n),
            "monthly" => ExpiryPick::Monthly(n),
            "date" => ExpiryPick::Date(date_ns.ok_or_else(|| {
                Fail::new("invalid_expiry_pick", "which='date' needs date_ns")
                    .op("OptionChain.pick_expiry")
                    .param("date_ns")
                    .valid(EXPIRY_PICKS)
                    .skill("instruments")
            })?),
            other => {
                return Err(Fail::new(
                    "invalid_expiry_pick",
                    format!("which is nearest, nth, weekly, monthly or date, got {other:?}"),
                )
                .op("OptionChain.pick_expiry")
                .param("which")
                .value(other.to_string())
                .valid(EXPIRY_PICKS)
                .skill("instruments")
                .into())
            }
        };
        Ok(self.inner.pick_expiry(underlying, now_ns, pick, utc_offset_ns))
    }

    /// The listed strike nearest `reference` (the lower one on a tie).
    fn atm_strike(
        &self,
        underlying: &str,
        expiry_ns: i64,
        now_ns: i64,
        reference: f64,
    ) -> Option<f64> {
        self.inner.atm_strike(underlying, expiry_ns, now_ns, reference)
    }

    /// The listed strike nearest the forward of `spot` to this expiry.
    #[pyo3(signature = (underlying, expiry_ns, now_ns, spot, rate=0.0, dividend_yield=0.0))]
    fn atm_strike_forward(
        &self,
        underlying: &str,
        expiry_ns: i64,
        now_ns: i64,
        spot: f64,
        rate: f64,
        dividend_yield: f64,
    ) -> Option<f64> {
        let tau = (expiry_ns - now_ns) as f64 / pricing::YEAR_NS as f64;
        let fwd = pricing::forward_price(spot, tau, rate, dividend_yield);
        self.inner.atm_strike(underlying, expiry_ns, now_ns, fwd)
    }

    /// The strike `offset` listed strikes above the one nearest `reference`
    /// (negative: below); None if that runs off the chain.
    fn strike_at_offset(
        &self,
        underlying: &str,
        expiry_ns: i64,
        now_ns: i64,
        reference: f64,
        offset: i64,
    ) -> Option<f64> {
        self.inner.strike_at_offset(underlying, expiry_ns, now_ns, reference, offset)
    }

    /// The listed contract of `right` whose model delta is nearest `target`
    /// (calls positive, puts negative) at a flat `vol`.
    #[pyo3(signature = (underlying, expiry_ns, now_ns, right, target, underlying_price, vol,
                        rate=0.0, dividend_yield=0.0, model="black_scholes"))]
    #[allow(clippy::too_many_arguments)]
    fn by_delta(
        &self,
        underlying: &str,
        expiry_ns: i64,
        now_ns: i64,
        right: &str,
        target: f64,
        underlying_price: f64,
        vol: f64,
        rate: f64,
        dividend_yield: f64,
        model: &str,
    ) -> PyResult<Option<PyInstrumentSpec>> {
        Ok(spec_out(self.inner.by_delta(
            underlying,
            expiry_ns,
            now_ns,
            parse_right(right)?,
            target,
            model_of(model)?,
            underlying_price,
            vol,
            rate,
            dividend_yield,
        )))
    }

    /// The listed contract of `right` whose `measure(contract)` is nearest
    /// `target` (the lower strike on a tie). `measure` returns a number, or
    /// None / NaN to skip a contract: a quoted premium, say.
    fn closest_by(
        &self,
        underlying: &str,
        expiry_ns: i64,
        now_ns: i64,
        right: &str,
        target: f64,
        measure: &Bound<'_, PyAny>,
    ) -> PyResult<Option<PyInstrumentSpec>> {
        let failure: RefCell<Option<PyErr>> = RefCell::new(None);
        let best = self.inner.closest_by(
            underlying,
            expiry_ns,
            now_ns,
            parse_right(right)?,
            target,
            |c| {
                if failure.borrow().is_some() {
                    return None;
                }
                let spec = PyInstrumentSpec { inner: c.clone() };
                let call = measure.call1((spec,)).and_then(|v| v.extract::<Option<f64>>());
                match call {
                    Ok(v) => v,
                    Err(e) => {
                        *failure.borrow_mut() = Some(e);
                        None
                    }
                }
            },
        );
        if let Some(e) = failure.into_inner() {
            return Err(e);
        }
        Ok(spec_out(best))
    }

    fn __repr__(&self) -> String {
        format!("OptionChain(contracts={})", self.inner.len())
    }
}

/// Register the pricing and chain bindings on the extension module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGreeks>()?;
    m.add_class::<PyOptionChain>()?;
    m.add_function(wrap_pyfunction!(pricing_price, m)?)?;
    m.add_function(wrap_pyfunction!(pricing_greeks, m)?)?;
    m.add_function(wrap_pyfunction!(pricing_implied_vol, m)?)?;
    m.add_function(wrap_pyfunction!(pricing_price_array, m)?)?;
    m.add_function(wrap_pyfunction!(pricing_greeks_array, m)?)?;
    m.add_function(wrap_pyfunction!(pricing_implied_vol_array, m)?)?;
    m.add_function(wrap_pyfunction!(pricing_status_name, m)?)?;
    m.add_function(wrap_pyfunction!(pricing_year_fraction, m)?)?;
    m.add_function(wrap_pyfunction!(pricing_forward, m)?)?;
    Ok(())
}
