//! Python bindings for instrument market definitions.
//!
//! Exposes [`InstrumentSpec`] as a single Python class with named
//! constructors per instrument family (`Instrument.equity(...)`,
//! `.futures_contract(...)`, `.option(...)`, ...), so Python users get a
//! typed-feeling surface without a Rust class hierarchy behind it.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::instruments::{InstrumentKind, InstrumentSpec, OptionRight};

/// Python-exposed instrument market definition.
#[pyclass(name = "InstrumentSpec")]
#[derive(Debug, Clone)]
pub struct PyInstrumentSpec {
    pub(crate) inner: InstrumentSpec,
}

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
        return Err(PyValueError::new_err("increments and lot_size must be >= 0"));
    }
    if multiplier <= 0.0 {
        return Err(PyValueError::new_err("multiplier must be > 0"));
    }
    if let (Some(act), Some(exp)) = (activation_ns, expiration_ns) {
        if exp <= act {
            return Err(PyValueError::new_err("expiration_ns must be after activation_ns"));
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

#[pymethods]
impl PyInstrumentSpec {
    /// A cash equity / spot instrument.
    #[staticmethod]
    #[pyo3(signature = (symbol, price_increment=0.0, lot_size=1.0, size_increment=0.0,
                        margin_init=0.0, margin_maint=0.0, maker_fee=0.0, taker_fee=0.0))]
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
    ) -> PyResult<Self> {
        let spec = InstrumentSpec::new(symbol, InstrumentKind::Cash);
        Ok(Self {
            inner: apply_common(
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
        })
    }

    /// A dated futures contract.
    #[staticmethod]
    #[pyo3(signature = (symbol, expiration_ns, lot_size, multiplier=1.0, price_increment=0.0,
                        underlying=None, activation_ns=None, margin_init=0.0, margin_maint=0.0,
                        maker_fee=0.0, taker_fee=0.0))]
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
    ) -> PyResult<Self> {
        let spec = InstrumentSpec::new(symbol, InstrumentKind::Contract { underlying });
        Ok(Self {
            inner: apply_common(
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
        })
    }

    /// A perpetual contract (dated future without expiry).
    #[staticmethod]
    #[pyo3(signature = (symbol, lot_size=1.0, multiplier=1.0, price_increment=0.0,
                        size_increment=0.0, underlying=None, margin_init=0.0, margin_maint=0.0,
                        maker_fee=0.0, taker_fee=0.0))]
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
    ) -> PyResult<Self> {
        let spec = InstrumentSpec::new(symbol, InstrumentKind::Contract { underlying });
        Ok(Self {
            inner: apply_common(
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
        })
    }

    /// A vanilla or binary option contract.
    #[staticmethod]
    #[pyo3(signature = (symbol, strike, right, expiration_ns, lot_size, multiplier=1.0,
                        price_increment=0.0, underlying=None, binary=false, activation_ns=None,
                        margin_init=0.0, margin_maint=0.0, maker_fee=0.0, taker_fee=0.0))]
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
    ) -> PyResult<Self> {
        let right = match right.to_ascii_lowercase().as_str() {
            "call" | "c" | "ce" => OptionRight::Call,
            "put" | "p" | "pe" => OptionRight::Put,
            other => {
                return Err(PyValueError::new_err(format!(
                    "right must be 'call' or 'put', got {other:?}"
                )))
            }
        };
        if strike <= 0.0 {
            return Err(PyValueError::new_err("strike must be > 0"));
        }
        let spec =
            InstrumentSpec::new(symbol, InstrumentKind::Option { strike, right, underlying, binary });
        Ok(Self {
            inner: apply_common(
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
        })
    }

    /// A spot currency pair.
    #[staticmethod]
    #[pyo3(signature = (symbol, price_increment=0.0, size_increment=0.0, lot_size=1.0,
                        margin_init=0.0, margin_maint=0.0, maker_fee=0.0, taker_fee=0.0))]
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
    ) -> PyResult<Self> {
        let spec = InstrumentSpec::new(symbol, InstrumentKind::CurrencyPair);
        Ok(Self {
            inner: apply_common(
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
        })
    }

    /// A non-tradable reference index.
    #[staticmethod]
    #[pyo3(signature = (symbol, price_increment=0.0))]
    fn index(symbol: &str, price_increment: f64) -> PyResult<Self> {
        let mut spec = InstrumentSpec::new(symbol, InstrumentKind::Index);
        if price_increment < 0.0 {
            return Err(PyValueError::new_err("price_increment must be >= 0"));
        }
        spec.price_increment = price_increment;
        Ok(Self { inner: spec })
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
