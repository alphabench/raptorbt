//! Python bindings for instrument market definitions.
//!
//! Exposes [`InstrumentSpec`] as a single Python class with named
//! constructors per instrument family (`Instrument.equity(...)`,
//! `.futures_contract(...)`, `.option(...)`, ...), so Python users get a
//! typed-feeling surface without a Rust class hierarchy behind it.

use pyo3::prelude::*;

use super::errors::{Fail, Val};
use super::schedule_bindings::PySessionSchedule;
use crate::instruments::{InstrumentKind, InstrumentSpec, OptionRight};

/// Python-exposed instrument market definition.
#[pyclass(name = "InstrumentSpec")]
#[derive(Debug, Clone)]
pub struct PyInstrumentSpec {
    pub(crate) inner: InstrumentSpec,
}

/// A refused argument of an `InstrumentSpec` factory.
fn refuse(
    code: &'static str,
    operation: &str,
    parameter: &str,
    message: impl Into<String>,
    value: impl Into<Val>,
) -> PyErr {
    Fail::new(code, message).op(operation).param(parameter).value(value).skill("instruments").into()
}

// The argument list IS the Python signature; collapsing it into a
// struct would change the public API for no reader benefit.
#[allow(clippy::too_many_arguments)]
fn apply_common(
    mut spec: InstrumentSpec,
    price_increment: f64,
    size_increment: f64,
    lot_size: f64,
    multiplier: f64,
    margin_init: f64,
    margin_maint: f64,
    maker_fee: f64,
    taker_fee: f64,
    activation_ns: Option<i64>,
    expiration_ns: Option<i64>,
) -> PyResult<InstrumentSpec> {
    if price_increment < 0.0 || size_increment < 0.0 || lot_size < 0.0 {
        return Err(refuse(
            "invalid_parameter",
            "InstrumentSpec",
            "price_increment",
            "increments and lot_size must be >= 0",
            price_increment.min(size_increment).min(lot_size),
        ));
    }
    if multiplier <= 0.0 {
        return Err(refuse(
            "invalid_parameter",
            "InstrumentSpec",
            "multiplier",
            "multiplier must be > 0",
            multiplier,
        ));
    }
    if let (Some(act), Some(exp)) = (activation_ns, expiration_ns) {
        if exp <= act {
            return Err(refuse(
                "invalid_parameter",
                "InstrumentSpec",
                "expiration_ns",
                "expiration_ns must be after activation_ns",
                exp,
            ));
        }
    }
    spec.price_increment = price_increment;
    spec.size_increment = size_increment;
    spec.lot_size = if lot_size > 0.0 { lot_size } else { 1.0 };
    spec.multiplier = multiplier;
    spec.margin_init = margin_init;
    spec.margin_maint = margin_maint;
    spec.maker_fee = maker_fee;
    spec.taker_fee = taker_fee;
    spec.activation_ns = activation_ns;
    spec.expiration_ns = expiration_ns;
    Ok(spec)
}

fn non_empty(field: &str, value: Option<String>) -> PyResult<Option<String>> {
    match value {
        Some(v) if v.trim().is_empty() => {
            Err(Fail::new("invalid_instrument_metadata", format!("{field} must not be empty"))
                .op("InstrumentSpec")
                .param(field)
                .into())
        }
        other => Ok(other),
    }
}

/// Attach the venue, currency and session every factory accepts.
fn with_meta(
    mut spec: InstrumentSpec,
    venue: Option<String>,
    currency: Option<String>,
    session: Option<&PySessionSchedule>,
) -> PyResult<InstrumentSpec> {
    spec.venue = non_empty("venue", venue)?;
    spec.currency = non_empty("currency", currency)?;
    spec.session = session.map(|s| s.inner.clone());
    Ok(spec)
}

#[pymethods]
impl PyInstrumentSpec {
    /// A cash equity / spot instrument.
    #[staticmethod]
    #[pyo3(signature = (symbol, price_increment=0.0, lot_size=1.0, size_increment=0.0,
                        margin_init=0.0, margin_maint=0.0, maker_fee=0.0, taker_fee=0.0, venue=None, currency=None, session=None))]
    #[allow(clippy::too_many_arguments)]
    fn equity(
        symbol: &str,
        price_increment: f64,
        lot_size: f64,
        size_increment: f64,
        margin_init: f64,
        margin_maint: f64,
        maker_fee: f64,
        taker_fee: f64,
        venue: Option<String>,
        currency: Option<String>,
        session: Option<&PySessionSchedule>,
    ) -> PyResult<Self> {
        let spec = InstrumentSpec::new(symbol, InstrumentKind::Cash);
        Ok(Self {
            inner: with_meta(
                apply_common(
                    spec,
                    price_increment,
                    size_increment,
                    lot_size,
                    1.0,
                    margin_init,
                    margin_maint,
                    maker_fee,
                    taker_fee,
                    None,
                    None,
                )?,
                venue,
                currency,
                session,
            )?,
        })
    }

    /// A dated futures contract.
    #[staticmethod]
    #[pyo3(signature = (symbol, expiration_ns, lot_size, multiplier=1.0, price_increment=0.0,
                        underlying=None, activation_ns=None, margin_init=0.0, margin_maint=0.0,
                        maker_fee=0.0, taker_fee=0.0, venue=None, currency=None, session=None))]
    #[allow(clippy::too_many_arguments)]
    fn futures_contract(
        symbol: &str,
        expiration_ns: i64,
        lot_size: f64,
        multiplier: f64,
        price_increment: f64,
        underlying: Option<String>,
        activation_ns: Option<i64>,
        margin_init: f64,
        margin_maint: f64,
        maker_fee: f64,
        taker_fee: f64,
        venue: Option<String>,
        currency: Option<String>,
        session: Option<&PySessionSchedule>,
    ) -> PyResult<Self> {
        let spec = InstrumentSpec::new(symbol, InstrumentKind::Contract { underlying });
        Ok(Self {
            inner: with_meta(
                apply_common(
                    spec,
                    price_increment,
                    0.0,
                    lot_size,
                    multiplier,
                    margin_init,
                    margin_maint,
                    maker_fee,
                    taker_fee,
                    activation_ns,
                    Some(expiration_ns),
                )?,
                venue,
                currency,
                session,
            )?,
        })
    }

    /// A perpetual contract (dated future without expiry).
    #[staticmethod]
    #[pyo3(signature = (symbol, lot_size=1.0, multiplier=1.0, price_increment=0.0,
                        size_increment=0.0, underlying=None, margin_init=0.0, margin_maint=0.0,
                        maker_fee=0.0, taker_fee=0.0, venue=None, currency=None, session=None))]
    #[allow(clippy::too_many_arguments)]
    fn perpetual(
        symbol: &str,
        lot_size: f64,
        multiplier: f64,
        price_increment: f64,
        size_increment: f64,
        underlying: Option<String>,
        margin_init: f64,
        margin_maint: f64,
        maker_fee: f64,
        taker_fee: f64,
        venue: Option<String>,
        currency: Option<String>,
        session: Option<&PySessionSchedule>,
    ) -> PyResult<Self> {
        let spec = InstrumentSpec::new(symbol, InstrumentKind::Contract { underlying });
        Ok(Self {
            inner: with_meta(
                apply_common(
                    spec,
                    price_increment,
                    size_increment,
                    lot_size,
                    multiplier,
                    margin_init,
                    margin_maint,
                    maker_fee,
                    taker_fee,
                    None,
                    None,
                )?,
                venue,
                currency,
                session,
            )?,
        })
    }

    /// A vanilla or binary option contract.    ///
    /// `span_pct` and `exposure_pct` model the deposit an exchange blocks
    /// against a SOLD option, each as a fraction of the underlying notional
    /// at the strike. Both default to `0.0`, which leaves a short option
    /// funded at its premium as in earlier releases. Bought options are
    /// unaffected: a buyer can lose only the premium.
    #[staticmethod]
    #[pyo3(signature = (symbol, strike, right, expiration_ns, lot_size, multiplier=1.0,
                        price_increment=0.0, underlying=None, binary=false, activation_ns=None,
                        margin_init=0.0, margin_maint=0.0, maker_fee=0.0, taker_fee=0.0,
                        span_pct=0.0, exposure_pct=0.0, venue=None, currency=None, session=None))]
    #[allow(clippy::too_many_arguments)]
    fn option(
        symbol: &str,
        strike: f64,
        right: &str,
        expiration_ns: i64,
        lot_size: f64,
        multiplier: f64,
        price_increment: f64,
        underlying: Option<String>,
        binary: bool,
        activation_ns: Option<i64>,
        margin_init: f64,
        margin_maint: f64,
        maker_fee: f64,
        taker_fee: f64,
        span_pct: f64,
        exposure_pct: f64,
        venue: Option<String>,
        currency: Option<String>,
        session: Option<&PySessionSchedule>,
    ) -> PyResult<Self> {
        if span_pct < 0.0 || exposure_pct < 0.0 {
            return Err(refuse(
                "invalid_parameter",
                "InstrumentSpec.option",
                "span_pct",
                "span_pct and exposure_pct must be >= 0",
                span_pct.min(exposure_pct),
            ));
        }
        let right = match right.to_ascii_lowercase().as_str() {
            "call" | "c" | "ce" => OptionRight::Call,
            "put" | "p" | "pe" => OptionRight::Put,
            other => {
                return Err(Fail::new(
                    "invalid_option_right",
                    format!("right must be 'call' or 'put', got {other:?}"),
                )
                .op("InstrumentSpec.option")
                .param("right")
                .value(other.to_string())
                .into())
            }
        };
        if strike <= 0.0 {
            return Err(refuse(
                "invalid_parameter",
                "InstrumentSpec.option",
                "strike",
                "strike must be > 0",
                strike,
            ));
        }
        let mut spec = InstrumentSpec::new(
            symbol,
            InstrumentKind::Option { strike, right, underlying, binary },
        );
        spec.span_pct = span_pct;
        spec.exposure_pct = exposure_pct;
        Ok(Self {
            inner: with_meta(
                apply_common(
                    spec,
                    price_increment,
                    0.0,
                    lot_size,
                    multiplier,
                    margin_init,
                    margin_maint,
                    maker_fee,
                    taker_fee,
                    activation_ns,
                    Some(expiration_ns),
                )?,
                venue,
                currency,
                session,
            )?,
        })
    }

    /// A spot currency pair.
    #[staticmethod]
    #[pyo3(signature = (symbol, price_increment=0.0, size_increment=0.0, lot_size=1.0,
                        margin_init=0.0, margin_maint=0.0, maker_fee=0.0, taker_fee=0.0, venue=None, currency=None, session=None))]
    #[allow(clippy::too_many_arguments)]
    fn currency_pair(
        symbol: &str,
        price_increment: f64,
        size_increment: f64,
        lot_size: f64,
        margin_init: f64,
        margin_maint: f64,
        maker_fee: f64,
        taker_fee: f64,
        venue: Option<String>,
        currency: Option<String>,
        session: Option<&PySessionSchedule>,
    ) -> PyResult<Self> {
        let spec = InstrumentSpec::new(symbol, InstrumentKind::CurrencyPair);
        Ok(Self {
            inner: with_meta(
                apply_common(
                    spec,
                    price_increment,
                    size_increment,
                    lot_size,
                    1.0,
                    margin_init,
                    margin_maint,
                    maker_fee,
                    taker_fee,
                    None,
                    None,
                )?,
                venue,
                currency,
                session,
            )?,
        })
    }

    /// A non-tradable reference index.
    #[staticmethod]
    #[pyo3(signature = (symbol, price_increment=0.0, venue=None, currency=None, session=None))]
    fn index(
        symbol: &str,
        price_increment: f64,
        venue: Option<String>,
        currency: Option<String>,
        session: Option<&PySessionSchedule>,
    ) -> PyResult<Self> {
        let mut spec = InstrumentSpec::new(symbol, InstrumentKind::Index);
        if price_increment < 0.0 {
            return Err(refuse(
                "invalid_parameter",
                "InstrumentSpec.index",
                "price_increment",
                "price_increment must be >= 0",
                price_increment,
            ));
        }
        spec.price_increment = price_increment;
        Ok(Self { inner: with_meta(spec, venue, currency, session)? })
    }

    /// Listing venue, or None. Metadata only.
    #[getter]
    fn venue(&self) -> Option<String> {
        self.inner.venue.clone()
    }

    #[setter]
    fn set_venue(&mut self, venue: Option<String>) -> PyResult<()> {
        self.inner.venue = non_empty("venue", venue)?;
        Ok(())
    }

    /// Quote currency, or None. Metadata only: nothing is converted.
    #[getter]
    fn currency(&self) -> Option<String> {
        self.inner.currency.clone()
    }

    #[setter]
    fn set_currency(&mut self, currency: Option<String>) -> PyResult<()> {
        self.inner.currency = non_empty("currency", currency)?;
        Ok(())
    }

    /// Trading calendar, or None for an instrument that trades at any time.
    #[getter]
    fn session(&self) -> Option<PySessionSchedule> {
        self.inner.session.clone().map(|inner| PySessionSchedule { inner })
    }

    #[setter]
    fn set_session(&mut self, session: Option<&PySessionSchedule>) {
        self.inner.session = session.map(|s| s.inner.clone());
    }

    #[getter]
    fn symbol(&self) -> String {
        self.inner.symbol.clone()
    }

    /// Instrument family: "cash", "currency_pair", "contract", "option", "index".
    #[getter]
    fn kind(&self) -> &'static str {
        match &self.inner.kind {
            InstrumentKind::Cash => "cash",
            InstrumentKind::CurrencyPair => "currency_pair",
            InstrumentKind::Contract { .. } => "contract",
            InstrumentKind::Option { .. } => "option",
            InstrumentKind::Index => "index",
        }
    }

    /// Fee fraction charged on settlement at expiry.
    #[getter]
    fn settlement_fee(&self) -> f64 {
        self.inner.settlement_fee
    }

    #[setter]
    fn set_settlement_fee(&mut self, fee: f64) {
        self.inner.settlement_fee = fee;
    }

    /// How the contract settles at expiry: `"cash"` (default) or
    /// `"physical"` (the underlying is delivered).
    #[getter]
    fn settlement_style(&self) -> &'static str {
        self.inner.settlement.style.as_str()
    }

    #[setter]
    fn set_settlement_style(&mut self, style: &str) -> PyResult<()> {
        self.inner.settlement.style = super::lifecycle_bindings::parse_style(style)?;
        Ok(())
    }

    /// Where the settlement price comes from: `"auto"` (default),
    /// `"last_price"`, `"underlying"`, or a number declared for the contract.
    #[getter]
    fn settlement_price(&self, py: Python<'_>) -> Py<PyAny> {
        super::lifecycle_bindings::price_to_py(py, self.inner.settlement.price)
    }

    #[setter]
    fn set_settlement_price(&mut self, price: &Bound<'_, PyAny>) -> PyResult<()> {
        self.inner.settlement.price = super::lifecycle_bindings::parse_price(price)?;
        Ok(())
    }

    #[setter]
    fn set_expiration_ns(&mut self, ns: Option<i64>) {
        self.inner.expiration_ns = ns;
    }

    #[getter]
    fn price_increment(&self) -> f64 {
        self.inner.price_increment
    }

    #[getter]
    fn size_increment(&self) -> f64 {
        self.inner.size_increment
    }

    #[getter]
    fn lot_size(&self) -> f64 {
        self.inner.lot_size
    }

    #[getter]
    fn multiplier(&self) -> f64 {
        self.inner.multiplier
    }

    #[getter]
    fn margin_init(&self) -> f64 {
        self.inner.margin_init
    }

    /// SPAN-style deposit fraction for a sold option (0.0 = not modelled).
    #[getter]
    fn span_pct(&self) -> f64 {
        self.inner.span_pct
    }

    /// Exposure margin fraction for a sold option (0.0 = not modelled).
    #[getter]
    fn exposure_pct(&self) -> f64 {
        self.inner.exposure_pct
    }

    #[getter]
    fn margin_maint(&self) -> f64 {
        self.inner.margin_maint
    }

    #[getter]
    fn maker_fee(&self) -> f64 {
        self.inner.maker_fee
    }

    #[getter]
    fn taker_fee(&self) -> f64 {
        self.inner.taker_fee
    }

    #[getter]
    fn activation_ns(&self) -> Option<i64> {
        self.inner.activation_ns
    }

    #[getter]
    fn expiration_ns(&self) -> Option<i64> {
        self.inner.expiration_ns
    }

    /// Option strike, or None for non-options.
    #[getter]
    fn strike(&self) -> Option<f64> {
        match &self.inner.kind {
            InstrumentKind::Option { strike, .. } => Some(*strike),
            _ => None,
        }
    }

    /// "call"/"put" for options, None otherwise.
    #[getter]
    fn right(&self) -> Option<&'static str> {
        match &self.inner.kind {
            InstrumentKind::Option { right: OptionRight::Call, .. } => Some("call"),
            InstrumentKind::Option { right: OptionRight::Put, .. } => Some("put"),
            _ => None,
        }
    }

    /// Underlying symbol for derivatives, None otherwise.
    #[getter]
    fn underlying(&self) -> Option<String> {
        match &self.inner.kind {
            InstrumentKind::Contract { underlying } => underlying.clone(),
            InstrumentKind::Option { underlying, .. } => underlying.clone(),
            _ => None,
        }
    }

    /// Whether orders may be placed on this instrument.
    #[getter]
    fn tradable(&self) -> bool {
        self.inner.kind.tradable()
    }

    fn __repr__(&self) -> String {
        format!(
            "InstrumentSpec(symbol={:?}, kind={:?}, lot_size={}, multiplier={})",
            self.inner.symbol,
            self.kind(),
            self.inner.lot_size,
            self.inner.multiplier
        )
    }
}

impl From<&PyInstrumentSpec> for InstrumentSpec {
    fn from(py_spec: &PyInstrumentSpec) -> Self {
        py_spec.inner.clone()
    }
}
