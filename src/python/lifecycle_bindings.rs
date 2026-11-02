//! Python bindings for contract lifecycle terms and corporate actions.

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::errors::Fail;
use super::numpy_bridge::{numpy_to_vec_f64, numpy_to_vec_i64, vec_to_numpy_f64};
use crate::instruments::lifecycle::{
    adjust_bars, AdjustmentPolicy, CorporateAction, CorporateKind, SettlementPrice, SettlementStyle,
};

const OP: &str = "CorporateAction";

pub(crate) fn action_err(e: crate::core::RaptorError) -> PyErr {
    Fail::new("invalid_corporate_action", e.to_string()).op(OP).into()
}

pub(crate) fn policy_err(e: crate::core::RaptorError) -> PyErr {
    Fail::new("invalid_adjustment_policy", e.to_string()).param("adjustment_policy").into()
}

/// Parse the adjustment policy a runner was given. `None` is "as given" and
/// is only valid without corporate actions.
pub(crate) fn parse_policy(text: Option<&str>) -> PyResult<AdjustmentPolicy> {
    text.map(AdjustmentPolicy::parse).transpose().map_err(policy_err).map(Option::unwrap_or_default)
}

/// A corporate action on one instrument: a split, a bonus or a cash dividend.
///
/// `ts` is the ex-date's first instant in nanoseconds. A split or bonus
/// carries `ratio` (new units per old); a dividend carries `amount` per unit
/// and, optionally, `pay_ts` when the cash is credited (default: `ts`).
#[pyclass(name = "CorporateAction", frozen, from_py_object)]
#[derive(Debug, Clone)]
pub struct PyCorporateAction {
    pub(crate) inner: CorporateAction,
}

#[pymethods]
impl PyCorporateAction {
    #[new]
    #[pyo3(signature = (instrument, kind, ts, ratio=None, amount=None, pay_ts=None))]
    fn new(
        instrument: String,
        kind: &str,
        ts: i64,
        ratio: Option<f64>,
        amount: Option<f64>,
        pay_ts: Option<i64>,
    ) -> PyResult<Self> {
        let kind = CorporateKind::parse(kind).map_err(action_err)?;
        let inner = CorporateAction {
            instrument,
            kind,
            ts,
            ratio: ratio.unwrap_or(0.0),
            amount: amount.unwrap_or(0.0),
            pay_ts,
        };
        match kind {
            CorporateKind::Split | CorporateKind::Bonus if ratio.is_none() => {
                return Err(action_err(crate::core::RaptorError::invalid_parameter(format!(
                    "a {} needs a ratio",
                    kind.as_str()
                ))))
            }
            CorporateKind::Dividend if amount.is_none() => {
                return Err(action_err(crate::core::RaptorError::invalid_parameter(
                    "a dividend needs an amount",
                )))
            }
            _ => {}
        }
        inner.validate().map_err(action_err)?;
        Ok(Self { inner })
    }

    #[staticmethod]
    fn split(instrument: String, ts: i64, ratio: f64) -> PyResult<Self> {
        Self::new(instrument, "split", ts, Some(ratio), None, None)
    }

    #[staticmethod]
    fn bonus(instrument: String, ts: i64, ratio: f64) -> PyResult<Self> {
        Self::new(instrument, "bonus", ts, Some(ratio), None, None)
    }

    #[staticmethod]
    #[pyo3(signature = (instrument, ts, amount, pay_ts=None))]
    fn dividend(instrument: String, ts: i64, amount: f64, pay_ts: Option<i64>) -> PyResult<Self> {
        Self::new(instrument, "dividend", ts, None, Some(amount), pay_ts)
    }

    #[getter]
    fn instrument(&self) -> String {
        self.inner.instrument.clone()
    }

    #[getter]
    fn kind(&self) -> &'static str {
        self.inner.kind.as_str()
    }

    #[getter]
    fn ts(&self) -> i64 {
        self.inner.ts
    }

    #[getter]
    fn ratio(&self) -> Option<f64> {
        self.inner.rescales().then_some(self.inner.ratio)
    }

    #[getter]
    fn amount(&self) -> Option<f64> {
        (self.inner.kind == CorporateKind::Dividend).then_some(self.inner.amount)
    }

    #[getter]
    fn pay_ts(&self) -> Option<i64> {
        self.inner.pay_ts
    }

    fn __repr__(&self) -> String {
        let a = &self.inner;
        match a.kind {
            CorporateKind::Dividend => format!(
                "CorporateAction({:?}, dividend, ts={}, amount={}, pay_ts={:?})",
                a.instrument, a.ts, a.amount, a.pay_ts
            ),
            _ => format!(
                "CorporateAction({:?}, {}, ts={}, ratio={})",
                a.instrument,
                a.kind.as_str(),
                a.ts,
                a.ratio
            ),
        }
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

/// Extract the declared actions and the policy a runner was given, refusing
/// actions without a policy.
pub(crate) fn extract_actions(
    actions: Option<Vec<PyCorporateAction>>,
    policy: Option<&str>,
) -> PyResult<(Vec<CorporateAction>, AdjustmentPolicy)> {
    let actions: Vec<CorporateAction> =
        actions.unwrap_or_default().into_iter().map(|a| a.inner).collect();
    let policy = parse_policy(policy)?;
    if !actions.is_empty() && policy == AdjustmentPolicy::AsGiven {
        return Err(Fail::new(
            "invalid_adjustment_policy",
            "corporate_actions need an adjustment_policy: \"raw\" (prices as traded; positions \
             are rescaled at the ex-date) or \"adjusted\" (history rescaled onto the basis after \
             the last split)",
        )
        .param("adjustment_policy")
        .valid(["raw", "adjusted"])
        .into());
    }
    Ok((actions, policy))
}

/// Settlement style from its name, for an instrument spec.
pub(crate) fn parse_style(text: &str) -> PyResult<SettlementStyle> {
    SettlementStyle::parse(text).map_err(|e| {
        Fail::new("invalid_settlement", e.to_string())
            .param("settlement_style")
            .valid(["cash", "physical"])
            .into()
    })
}

/// A settlement price policy from `"auto"`, `"last_price"`, `"underlying"` or
/// a number (a price declared for the contract).
pub(crate) fn parse_price(value: &Bound<'_, PyAny>) -> PyResult<SettlementPrice> {
    if let Ok(text) = value.extract::<String>() {
        return match text.as_str() {
            "auto" => Ok(SettlementPrice::Auto),
            "last_price" => Ok(SettlementPrice::LastPrice),
            "underlying" => Ok(SettlementPrice::Underlying),
            other => Err(Fail::new(
                "invalid_settlement",
                format!(
                    "unknown settlement price {other:?}: \"auto\", \"last_price\", \
                     \"underlying\", or a number declared for the contract"
                ),
            )
            .param("settlement_price")
            .valid(["auto", "last_price", "underlying"])
            .into()),
        };
    }
    let price: f64 = value.extract().map_err(|_| {
        Fail::new(
            "invalid_settlement",
            "settlement_price is \"auto\", \"last_price\", \"underlying\" or a number",
        )
        .param("settlement_price")
    })?;
    if !(price.is_finite() && price >= 0.0) {
        return Err(Fail::new(
            "invalid_settlement",
            format!("a declared settlement price must be finite and >= 0, got {price}"),
        )
        .param("settlement_price")
        .value(price)
        .into());
    }
    Ok(SettlementPrice::Fixed(price))
}

pub(crate) fn price_to_py(py: Python<'_>, price: SettlementPrice) -> Py<PyAny> {
    match price {
        SettlementPrice::Auto => "auto".into_pyobject(py).unwrap().into_any().unbind(),
        SettlementPrice::LastPrice => "last_price".into_pyobject(py).unwrap().into_any().unbind(),
        SettlementPrice::Underlying => "underlying".into_pyobject(py).unwrap().into_any().unbind(),
        SettlementPrice::Fixed(p) => p.into_pyobject(py).unwrap().into_any().unbind(),
    }
}

/// Put one instrument's bars on the adjusted basis.
///
/// `bars` holds `timestamps` (int64 ns; `timestamp` also works), `open`, `high`, `low`, `close` and
/// `volume` arrays. Bars before each split or bonus of `instrument` in
/// `actions` have their prices divided, and volume multiplied, by the product
/// of the ratios after them. Returns a new dict; the input is untouched.
/// Dividends are not applied to prices.
#[pyfunction]
#[pyo3(signature = (instrument, actions, bars))]
pub fn adjust_history<'py>(
    py: Python<'py>,
    instrument: &str,
    actions: Vec<PyCorporateAction>,
    bars: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyDict>> {
    let col = |name: &str| -> PyResult<Bound<'py, PyAny>> {
        bars.get_item(name)?.ok_or_else(|| {
            Fail::new("invalid_data", format!("adjust_history: bars needs a {name:?} array")).into()
        })
    };
    let ts_key = if bars.contains("timestamps")? { "timestamps" } else { "timestamp" };
    let ts_arr: PyReadonlyArray1<i64> = col(ts_key)?.extract()?;
    let timestamps = numpy_to_vec_i64(ts_arr);
    let mut cols: Vec<Vec<f64>> = Vec::new();
    for name in ["open", "high", "low", "close", "volume"] {
        let a: PyReadonlyArray1<f64> = col(name)?.extract()?;
        cols.push(numpy_to_vec_f64(a));
    }
    let acts: Vec<CorporateAction> = actions.into_iter().map(|a| a.inner).collect();
    let [open, high, low, close, volume] = &mut cols[..] else { unreachable!() };
    adjust_bars(instrument, &acts, &timestamps, open, high, low, close, volume)
        .map_err(|e| Fail::new("invalid_data", e.to_string()).op("adjust_history"))?;
    let out = PyDict::new(py);
    for (name, values) in ["open", "high", "low", "close", "volume"].iter().zip(cols) {
        out.set_item(name, vec_to_numpy_f64(py, values))?;
    }
    out.set_item(ts_key, bars.get_item(ts_key)?)?;
    Ok(out)
}
