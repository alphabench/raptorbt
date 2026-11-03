//! Python bindings for margin models.

use std::collections::BTreeMap;

use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::errors::Fail;
use super::instrument_bindings::PyInstrumentSpec;
use crate::accounts::margin_model::{defined_risk, LineView, MarginClass, MarginModel};
use crate::core::RaptorError;

const OP: &str = "MarginModel";

fn model_err(e: RaptorError) -> PyErr {
    Fail::new("invalid_margin_model", e.to_string()).op(OP).into()
}

/// A margin model: how much initial margin a line of an instrument demands.
#[pyclass(name = "MarginModel", module = "raptorbt", from_py_object)]
#[derive(Debug, Clone)]
pub struct PyMarginModel {
    pub(crate) inner: MarginModel,
}

fn class_map<T>(
    items: &Bound<'_, PyDict>,
    mut value: impl FnMut(&Bound<'_, PyAny>) -> PyResult<T>,
) -> PyResult<BTreeMap<MarginClass, T>> {
    let mut out = BTreeMap::new();
    for (k, v) in items.iter() {
        let name: String = k.extract()?;
        let class = MarginClass::parse(&name).map_err(model_err)?;
        out.insert(class, value(&v)?);
    }
    Ok(out)
}

fn model_of(v: &Bound<'_, PyAny>) -> PyResult<MarginModel> {
    Ok(v.extract::<PyRef<'_, PyMarginModel>>()?.inner.clone())
}

fn positive(name: &str, x: f64) -> PyResult<f64> {
    if x.is_finite() && x >= 0.0 {
        Ok(x)
    } else {
        Err(Fail::new(
            "invalid_margin_model",
            format!("{name} must be a non-negative number, got {x}"),
        )
        .op(OP)
        .param(name)
        .value(x)
        .into())
    }
}

type PyLine<'py> = (PyRef<'py, PyInstrumentSpec>, f64, f64);

fn lines<'a>(legs: &'a [PyLine<'_>]) -> Vec<LineView<'a>> {
    legs.iter().map(|(spec, q, p)| LineView::new(&spec.inner, *q, *p)).collect()
}

#[pymethods]
impl PyMarginModel {
    /// Nothing is required, nothing is refused.
    #[staticmethod]
    fn no_margin() -> Self {
        Self { inner: MarginModel::NoMargin }
    }

    /// Delivery: the full notional of a long; a short is refused.
    #[staticmethod]
    fn cash_only() -> Self {
        Self { inner: MarginModel::CashOnly }
    }

    /// Notional over a factor per instrument class: `{"CASH": 5}`.
    #[staticmethod]
    fn leverage(factors: &Bound<'_, PyDict>) -> PyResult<Self> {
        let map = class_map(factors, |v| v.extract::<f64>())?;
        Ok(Self { inner: MarginModel::Leverage(map) })
    }

    /// A percentage of the notional. Deprecated: every other rate in the
    /// library is a fraction; use `fraction_of_notional(pct / 100)`.
    #[staticmethod]
    fn percent_of_notional(py: Python<'_>, pct: f64) -> PyResult<Self> {
        PyErr::warn(
            py,
            &py.get_type::<pyo3::exceptions::PyDeprecationWarning>(),
            c"MarginModel.percent_of_notional takes a percent while every other rate takes a \
              fraction; use MarginModel.fraction_of_notional(pct / 100) (20 percent is 0.2)",
            2,
        )?;
        Ok(Self { inner: MarginModel::PercentOfNotional(positive("pct", pct)?) })
    }

    /// A fraction of the notional: `0.2` is 20 percent. A value above 1 is
    /// refused as ambiguous (it is almost always a percent).
    #[staticmethod]
    fn fraction_of_notional(fraction: f64) -> PyResult<Self> {
        let fraction = positive("fraction", fraction)?;
        if fraction > 1.0 {
            return Err(Fail::new(
                "invalid_margin_model",
                format!(
                    "fraction must be at most 1 (0.2 is 20 percent), got {fraction}; \
                     a percent such as 20 is not a fraction"
                ),
            )
            .op(OP)
            .param("fraction")
            .value(fraction)
            .into());
        }
        Ok(Self { inner: MarginModel::PercentOfNotional(fraction * 100.0) })
    }

    /// A buyer pays the premium; a seller posts `seller_pct` percent of the
    /// strike notional plus the premium.
    #[staticmethod]
    fn option_simple(seller_pct: f64) -> PyResult<Self> {
        Ok(Self {
            inner: MarginModel::OptionSimple { seller_pct: positive("seller_pct", seller_pct)? },
        })
    }

    /// Per-unit margin per instrument symbol; others on `default` or refused.
    #[staticmethod]
    #[pyo3(signature = (per_unit, default=None))]
    fn table(per_unit: BTreeMap<String, f64>, default: Option<PyRef<'_, PyMarginModel>>) -> Self {
        Self {
            inner: MarginModel::Table {
                per_unit,
                default: default.map(|d| Box::new(d.inner.clone())),
            },
        }
    }

    /// Each instrument class on its own model: `{"CASH": MarginModel.cash_only()}`.
    #[staticmethod]
    fn by_type(models: &Bound<'_, PyDict>) -> PyResult<Self> {
        Ok(Self { inner: MarginModel::ByType(class_map(models, model_of)?) })
    }

    /// The intraday quantity of a line on `intraday`, the rest on `carry`.
    #[staticmethod]
    fn by_product(intraday: PyRef<'_, PyMarginModel>, carry: PyRef<'_, PyMarginModel>) -> Self {
        Self {
            inner: MarginModel::ByProduct {
                intraday: Box::new(intraday.inner.clone()),
                carry: Box::new(carry.inner.clone()),
            },
        }
    }

    /// The worst expiry loss of a set of option legs.
    #[staticmethod]
    fn defined_risk() -> Self {
        Self { inner: MarginModel::DefinedRisk }
    }

    /// `option_simple(seller_pct)` less the credits of lines that cover each
    /// other (a defined-risk structure within an expiry; calendar and
    /// diagonal spreads across expiries, none on the short's expiry day); a
    /// spread keeps at least `spread_pct` of the short's strike notional.
    #[staticmethod]
    fn spread_credit(seller_pct: f64, spread_pct: f64) -> PyResult<Self> {
        Ok(Self {
            inner: MarginModel::SpreadCredit {
                seller_pct: positive("seller_pct", seller_pct)?,
                spread_pct: positive("spread_pct", spread_pct)?,
            },
        })
    }

    /// The engine's own rule, read from each instrument: its `margin_init`
    /// (else one over `leverage`) on notional, and the `span_pct` plus
    /// `exposure_pct` deposit for a sold option, with the sold legs of one
    /// underlying and expiry margined together as a position group.
    #[staticmethod]
    #[pyo3(signature = (leverage=1.0))]
    fn from_spec(leverage: f64) -> PyResult<Self> {
        if !(leverage.is_finite() && leverage > 0.0) {
            return Err(Fail::new(
                "invalid_margin_model",
                format!("leverage must be positive, got {leverage}"),
            )
            .op(OP)
            .param("leverage")
            .value(leverage)
            .into());
        }
        Ok(Self { inner: MarginModel::FromSpec { leverage } })
    }

    #[getter]
    fn name(&self) -> &'static str {
        self.inner.name()
    }

    /// The initial margin of one line: `quantity` signed, `price` the mark.
    fn required(
        &self,
        spec: PyRef<'_, PyInstrumentSpec>,
        quantity: f64,
        price: f64,
    ) -> PyResult<f64> {
        self.inner.required(&LineView::new(&spec.inner, quantity, price)).map_err(model_err)
    }

    /// `by_product`: the intraday part of the line on the intraday model, the
    /// rest on carry. Any other model prices the whole line.
    fn required_split(
        &self,
        spec: PyRef<'_, PyInstrumentSpec>,
        quantity: f64,
        price: f64,
        intraday_quantity: f64,
    ) -> PyResult<f64> {
        self.inner
            .required_split(&LineView::new(&spec.inner, quantity, price), intraday_quantity)
            .map_err(model_err)
    }

    /// The margin of `[(spec, quantity, price), ...]` taken together:
    /// `defined_risk` and `from_spec` margin the option legs as a group, any
    /// other model sums the legs. `spread_credit` needs `now`, the time in ns
    /// the calendar credit is judged at.
    #[pyo3(signature = (legs, now=None))]
    fn required_legs(&self, legs: Vec<PyLine<'_>>, now: Option<i64>) -> PyResult<f64> {
        let views = lines(&legs);
        match now {
            Some(now) => self.inner.required_legs_at(&views, now),
            None => self.inner.required_legs(&views),
        }
        .map_err(model_err)
    }

    /// What each leg of a group locks, in the order given.
    #[pyo3(signature = (legs, now=0))]
    fn group_shares(&self, legs: Vec<PyLine<'_>>, now: i64) -> PyResult<Vec<f64>> {
        self.inner.group_shares(&lines(&legs), now).map_err(model_err)
    }

    /// Whether the option lines are margined together by this model.
    fn groups_options(&self) -> bool {
        self.inner.groups_options()
    }

    /// Name and parameters as a dict: what a result says the margin was.
    fn params<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        py.import("json")?.call_method1("loads", (self.inner.params().to_string(),))
    }

    /// The model, nested models in full, as JSON text with sorted keys.
    fn describe(&self) -> String {
        self.inner.describe()
    }

    fn __repr__(&self) -> String {
        self.inner.describe()
    }
}

/// The worst loss of `[(spec, quantity, price), ...]` legs at the option
/// expiry over every underlying price from zero up: exact on the strikes and
/// the slope above the highest one, `inf` when the position keeps losing as
/// the price rises. Future and cash legs are included; option legs must share
/// one expiry.
#[pyfunction(name = "defined_risk")]
pub fn defined_risk_py(legs: Vec<PyLine<'_>>) -> PyResult<f64> {
    defined_risk(&lines(&legs)).map_err(model_err)
}
