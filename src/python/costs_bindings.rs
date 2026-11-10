//! Python bindings for fee schedules and rate cards.

use std::collections::HashMap;
use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use super::errors::Fail;
use crate::core::RaptorError;
use crate::execution::costs::{FeeBreakdown, FeeRates, FeeRow, FeeSchedule, FeeSegment, RateCard};

const OP: &str = "FeeSchedule";

fn schedule_err(e: RaptorError) -> PyErr {
    match &e {
        RaptorError::InvalidConfig { .. } => {
            Fail::new("fee_schedule_gap", e.to_string()).op(OP).into()
        }
        _ => Fail::new("invalid_fee_schedule", e.to_string()).op(OP).into(),
    }
}

fn segment_arg(text: &str) -> PyResult<FeeSegment> {
    FeeSegment::parse(&text.to_ascii_uppercase()).map_err(|e| {
        Fail::new("invalid_fee_schedule", e.to_string())
            .op(OP)
            .param("segment")
            .value(text)
            .valid(["DELIVERY", "INTRADAY", "FUTURES", "OPTIONS"])
            .into()
    })
}

fn side_arg(text: &str) -> PyResult<bool> {
    match text.to_ascii_uppercase().as_str() {
        "BUY" => Ok(true),
        "SELL" => Ok(false),
        _ => Err(Fail::new(
            "invalid_fee_schedule",
            format!("side must be BUY or SELL, got {text:?}"),
        )
        .op(OP)
        .param("side")
        .value(text)
        .valid(["BUY", "SELL"])
        .into()),
    }
}

fn json_module<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyModule>> {
    py.import("json")
}

pub(crate) fn rates_dict<'py>(py: Python<'py>, r: &FeeRates) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("brokerage_flat", r.brokerage_flat)?;
    d.set_item("brokerage_rate", r.brokerage_rate)?;
    d.set_item("brokerage_cap", r.brokerage_cap)?;
    d.set_item("brokerage_min", r.brokerage_min)?;
    d.set_item("brokerage_per", r.brokerage_per.as_str())?;
    d.set_item("tax_kind", r.tax_kind.as_str())?;
    d.set_item("tax_buy_rate", r.tax_buy_rate)?;
    d.set_item("tax_sell_rate", r.tax_sell_rate)?;
    d.set_item("exchange_rate", r.exchange_rate)?;
    d.set_item("regulatory_rate", r.regulatory_rate)?;
    d.set_item("investor_fund_rate", r.investor_fund_rate)?;
    d.set_item("stamp_buy_rate", r.stamp_buy_rate)?;
    d.set_item("gst_rate", r.gst_rate)?;
    d.set_item("rounding", r.rounding.as_str())?;
    match r.day_fee {
        Some(day) => {
            let f = PyDict::new(py);
            f.set_item("amount", day.amount)?;
            f.set_item(
                "side",
                if day.side == crate::execution::costs::DaySide::Buy { "BUY" } else { "SELL" },
            )?;
            f.set_item("gst", day.gst)?;
            d.set_item("day_fee", f)?;
        }
        None => d.set_item("day_fee", py.None())?,
    }
    Ok(d)
}

fn breakdown_dict(b: &FeeBreakdown) -> HashMap<&'static str, Option<f64>> {
    HashMap::from([
        ("brokerage", Some(b.brokerage)),
        ("stt", Some(b.stt)),
        ("exchange_txn", Some(b.exchange_txn)),
        ("sebi_fee", Some(b.sebi_fee)),
        ("stamp_duty", Some(b.stamp_duty)),
        ("gst", Some(b.gst)),
        ("ipft", Some(b.ipft)),
        ("dp_charge", Some(b.dp_charge)),
        ("total", Some(b.total())),
        ("exact_total", b.exact_total),
    ])
}

/// A validated rate card: a document of fee rows with an effective date.
#[pyclass(name = "RateCard", module = "raptorbt")]
#[derive(Debug, Clone)]
pub struct PyRateCard {
    pub(crate) inner: RateCard,
}

#[pymethods]
impl PyRateCard {
    /// Parse and validate a rate card document (JSON text).
    #[staticmethod]
    fn from_json(text: &str) -> PyResult<Self> {
        RateCard::from_json(text).map(|inner| Self { inner }).map_err(schedule_err)
    }

    /// Parse and validate a rate card given as a dict.
    #[staticmethod]
    fn from_dict(py: Python<'_>, doc: &Bound<'_, PyAny>) -> PyResult<Self> {
        let text: String = json_module(py)?.call_method1("dumps", (doc,))?.extract()?;
        Self::from_json(&text)
    }

    #[getter]
    fn name(&self) -> String {
        self.inner.name.clone()
    }

    #[getter]
    fn effective_from(&self) -> String {
        self.inner.effective_from.clone()
    }

    #[getter]
    fn effective_ns(&self) -> i64 {
        self.inner.effective_ns
    }

    #[getter]
    fn source(&self) -> String {
        self.inner.source.clone()
    }

    /// The card's rows as a schedule.
    fn fee_schedule(&self) -> PyFeeSchedule {
        PyFeeSchedule { inner: Arc::new(self.inner.fee_schedule()) }
    }

    /// How many of `fill_ts` (UTC ns) fell before the card took effect.
    fn backdated_fills(&self, fill_ts: Vec<i64>) -> usize {
        self.inner.backdated_fills(&fill_ts)
    }

    /// The document, as JSON text that `from_json` reads back to the same rows.
    fn to_json(&self) -> String {
        self.inner.to_json()
    }

    /// The document, as a dict.
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        json_module(py)?.call_method1("loads", (self.inner.to_json(),))
    }

    fn __repr__(&self) -> String {
        format!(
            "RateCard({:?}, {} rows, from {})",
            self.inner.name,
            self.inner.rows.len(),
            self.inner.effective_from
        )
    }
}

/// Fees by venue and segment: the built-in exchange presets, or rows loaded
/// from rate cards.
#[pyclass(name = "FeeSchedule", module = "raptorbt", from_py_object)]
#[derive(Debug, Clone)]
pub struct PyFeeSchedule {
    pub(crate) inner: Arc<FeeSchedule>,
}

fn row_dict<'py>(py: Python<'py>, row: &FeeRow) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("venue", &row.venue)?;
    d.set_item("segment", row.segment.as_str())?;
    d.set_item("underlying", &row.underlying)?;
    d.set_item("settlement", row.settlement)?;
    d.set_item("effective_from", row.effective_from)?;
    d.set_item("effective_until", row.meta.effective_until)?;
    d.set_item("status", row.meta.status.as_str())?;
    d.set_item("base", row.meta.base.as_str())?;
    d.set_item("source", &row.meta.source)?;
    d.set_item("label", row.label())?;
    d.set_item("rates", rates_dict(py, &row.rates)?)?;
    // The same rates in rupees per crore of the base, which is how they are stored
    // and published (a fraction would carry the conversion's rounding).
    let per_crore = PyDict::new(py);
    let r = &row.rates;
    for (name, value) in [
        ("tax_sell", r.tax_sell_rate),
        ("tax_buy", r.tax_buy_rate),
        ("exchange", r.exchange_rate),
        ("ipft", r.investor_fund_rate),
        ("sebi", r.regulatory_rate),
        ("stamp_buy", r.stamp_buy_rate),
    ] {
        per_crore.set_item(name, (value * 1.0e7 * 1.0e6).round() / 1.0e6)?;
    }
    d.set_item("per_crore", per_crore)?;
    Ok(d)
}

#[pymethods]
impl PyFeeSchedule {
    /// The engine's built-in rows for NSE, BSE, NFO, BFO, MCX, CDS and BCD.
    #[staticmethod]
    fn exchange_presets() -> Self {
        Self { inner: Arc::new(FeeSchedule::exchange_presets()) }
    }

    /// One schedule from several cards, their rows in order.
    #[staticmethod]
    fn from_cards(cards: Vec<PyRef<'_, PyRateCard>>) -> PyResult<Self> {
        let parts: Vec<FeeSchedule> = cards.iter().map(|c| c.inner.fee_schedule()).collect();
        FeeSchedule::merge(&parts).map(|s| Self { inner: Arc::new(s) }).map_err(schedule_err)
    }

    #[staticmethod]
    fn from_json(text: &str) -> PyResult<Self> {
        Ok(PyRateCard::from_json(text)?.fee_schedule())
    }

    #[staticmethod]
    fn from_dict(py: Python<'_>, doc: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(PyRateCard::from_dict(py, doc)?.fee_schedule())
    }

    #[getter]
    fn name(&self) -> String {
        self.inner.name.clone()
    }

    #[getter]
    fn source(&self) -> String {
        self.inner.source.clone()
    }

    /// Every row: venue, segment, underlying, settlement, effective_from
    /// (UTC ns) and its rates.
    fn rows<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let out = PyList::empty(py);
        for row in &self.inner.rows {
            out.append(row_dict(py, row)?)?;
        }
        Ok(out)
    }

    /// The row that prices a fill; `at` is UTC ns (the latest row when
    /// omitted). A venue and segment without a row is an error.
    #[pyo3(signature = (venue, segment, underlying=None, at=None, settlement=false))]
    fn row<'py>(
        &self,
        py: Python<'py>,
        venue: &str,
        segment: &str,
        underlying: Option<&str>,
        at: Option<i64>,
        settlement: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let seg = segment_arg(segment)?;
        let row = self
            .inner
            .row_for(venue, seg, underlying, at.unwrap_or(i64::MAX), settlement)
            .map_err(schedule_err)?;
        row_dict(py, row)
    }

    /// Price one fill: pick the row and return the itemised charge as floats.
    /// `price * quantity * multiplier` is the turnover the rates apply to
    /// (the premium, for an option). `order_turnover` is what earlier fills
    /// of the same order traded (it matters when brokerage is per order);
    /// `day_fee_owed` says the instrument's per-day fee falls due here.
    #[pyo3(signature = (venue, segment, side, price, quantity, multiplier=1.0, underlying=None, at=None, order_turnover=0.0, day_fee_owed=false))]
    #[allow(clippy::too_many_arguments)]
    fn charge(
        &self,
        venue: &str,
        segment: &str,
        side: &str,
        price: f64,
        quantity: f64,
        multiplier: f64,
        underlying: Option<&str>,
        at: Option<i64>,
        order_turnover: f64,
        day_fee_owed: bool,
    ) -> PyResult<HashMap<&'static str, Option<f64>>> {
        let seg = segment_arg(segment)?;
        let is_buy = side_arg(side)?;
        let b = self
            .inner
            .charge(
                venue,
                seg,
                underlying,
                at.unwrap_or(i64::MAX),
                is_buy,
                price * quantity.abs() * multiplier,
                order_turnover,
                day_fee_owed,
            )
            .map_err(schedule_err)?;
        Ok(breakdown_dict(&b))
    }

    /// The rows in force at `at` (UTC ns): the schedule as it stood that day.
    fn in_force_at(&self, at: i64) -> Self {
        Self { inner: Arc::new(self.inner.in_force_at(at)) }
    }

    /// Every row group's earliest rates also in force from `at` (UTC ns), marked backdated in
    /// their source: a run over days before any published rate is priced on the earliest known.
    fn backdated(&self, at: i64) -> Self {
        Self { inner: Arc::new(self.inner.backdated(at)) }
    }

    /// The schedule as a rate card document (JSON text), every row dated
    /// `effective_from` (RFC 3339). A schedule whose rows take effect on
    /// several dates is refused; write `in_force_at(date)` for each.
    #[pyo3(signature = (effective_from="1970-01-01T00:00:00Z"))]
    fn to_json(&self, effective_from: &str) -> PyResult<String> {
        self.inner.to_card_json(effective_from).map_err(schedule_err)
    }

    fn __len__(&self) -> usize {
        self.inner.rows.len()
    }

    fn __repr__(&self) -> String {
        format!("FeeSchedule({:?}, {} rows)", self.inner.name, self.inner.rows.len())
    }
}

/// Resolve the rates a `fee_segment` name selects from `schedule`.
///
/// `"NFO-OPT"`, `"NSE-INTRADAY"` and `"MCX"` parse as they do for the
/// built-in presets; the venue and segment must have a row, or this is an
/// error: a schedule never silently charges nothing.
pub(crate) fn resolve_rates(schedule: &FeeSchedule, fee_segment: &str) -> PyResult<FeeRates> {
    resolve_row(schedule, fee_segment).map(|(_, _, rates)| rates)
}

/// [`resolve_rates`] with the venue and segment the name selected, which a
/// run binds to price each fill on the row in force on its date.
pub(crate) fn resolve_row(
    schedule: &FeeSchedule,
    fee_segment: &str,
) -> PyResult<(String, FeeSegment, FeeRates)> {
    let (seg, ty) = match fee_segment.split_once('-') {
        Some((s, t)) => (s, Some(t)),
        None => (fee_segment, None),
    };
    let ty_upper = ty.map(|t| t.to_ascii_uppercase());
    let intraday = !matches!(ty_upper.as_deref(), Some("DELIVERY"));
    let ty_arg = match ty_upper.as_deref() {
        Some("DELIVERY") | Some("INTRADAY") => None,
        other => other,
    };
    let venue = seg.to_ascii_uppercase();
    let is_option = matches!(ty_arg, Some("OPT") | Some("CE") | Some("PE"));
    let segment = match venue.as_str() {
        "NSE" | "BSE" if intraday => FeeSegment::Intraday,
        "NSE" | "BSE" => FeeSegment::Delivery,
        _ if is_option => FeeSegment::Options,
        _ => FeeSegment::Futures,
    };
    schedule
        .row_for(&venue, segment, None, i64::MAX, false)
        .map(|row| (venue.clone(), segment, row.rates))
        .map_err(schedule_err)
}
