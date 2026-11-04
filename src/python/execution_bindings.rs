//! Python face of the execution seam: the request a venue is sent, the
//! reports it answers with, and the bridge that lets any Python object with
//! `submit` / `cancel` / `modify` be a venue.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

use pyo3::basic::CompareOp;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use crate::execution::adapter::{AdapterError, ExecutionAdapter, OrderChange, OrderRequest};
use crate::execution::orders::{ExecutionReport, OrderKind, TimeInForce, TrailOffset};

/// One order as the engine hands it to a venue. Immutable.
#[pyclass(name = "OrderRequest", module = "raptorbt.execution", frozen)]
#[derive(Clone)]
pub struct PyOrderRequest(pub OrderRequest);

#[pymethods]
impl PyOrderRequest {
    /// Deterministic venue-facing id, `s{strategy}-{n}`.
    #[getter]
    fn client_order_id(&self) -> &str {
        &self.0.client_order_id
    }
    #[getter]
    fn symbol(&self) -> &str {
        &self.0.symbol
    }
    /// `"buy"` or `"sell"`.
    #[getter]
    fn side(&self) -> &'static str {
        self.0.side.as_str()
    }
    /// Units, positive.
    #[getter]
    fn quantity(&self) -> f64 {
        self.0.quantity
    }
    /// `"market"`, `"limit"`, `"stop_market"`, `"stop_limit"`,
    /// `"market_if_touched"`, `"limit_if_touched"`, `"market_to_limit"`,
    /// `"trailing_stop_market"` or `"trailing_stop_limit"`.
    #[getter]
    fn order_type(&self) -> &'static str {
        self.0.kind.as_str()
    }
    #[getter]
    fn limit_price(&self) -> Option<f64> {
        self.0.kind.limit_price()
    }
    #[getter]
    fn trigger_price(&self) -> Option<f64> {
        self.0.kind.trigger_price()
    }
    /// A trailing order's distance from the watermark, with `trail_kind`
    /// `"price"` or `"bps"`.
    #[getter]
    fn trail_offset(&self) -> Option<f64> {
        trail(&self.0.kind).map(|(offset, _)| offset)
    }
    #[getter]
    fn trail_kind(&self) -> Option<&'static str> {
        trail(&self.0.kind).map(|(_, kind)| kind)
    }
    /// How far through the trigger a triggered trailing-limit order rests.
    #[getter]
    fn trail_limit_offset(&self) -> Option<f64> {
        match self.0.kind {
            OrderKind::TrailingStopLimit { limit_offset, .. } => Some(limit_offset),
            _ => None,
        }
    }
    /// `"gtc"`, `"day"`, `"gtd"`, `"ioc"`, `"fok"`, `"at_open"` or `"at_close"`.
    #[getter]
    fn time_in_force(&self) -> &'static str {
        self.0.tif.as_str()
    }
    /// When a `"gtd"` order lapses (ns).
    #[getter]
    fn expire_ns(&self) -> Option<i64> {
        match self.0.tif {
            TimeInForce::Gtd { expire_ns } => Some(expire_ns),
            _ => None,
        }
    }
    #[getter]
    fn strategy_id(&self) -> u32 {
        self.0.strategy_id
    }
    #[getter]
    fn account_id(&self) -> Option<&str> {
        self.0.account_id.as_deref()
    }
    #[getter]
    fn reduce_only(&self) -> bool {
        self.0.reduce_only
    }
    #[getter]
    fn post_only(&self) -> bool {
        self.0.post_only
    }
    /// Hold until this order (a `client_order_id`) fills.
    #[getter]
    fn parent_client_order_id(&self) -> Option<&str> {
        self.0.parent_client_order_id.as_deref()
    }
    #[getter]
    fn tags(&self) -> BTreeMap<String, String> {
        self.0.tags.clone()
    }
    /// Under hedging, the engine's id of the position this order closes; `None` for an
    /// order that opens (and always on a netting account).
    #[getter]
    fn position_id(&self) -> Option<u64> {
        self.0.position_id
    }
    /// Event time (ns) of the decision.
    #[getter]
    fn submitted_ns(&self) -> i64 {
        self.0.submitted_ns
    }

    fn __repr__(&self) -> String {
        let r = &self.0;
        format!(
            "OrderRequest({} {} {} {} @ {:?}, strategy={})",
            r.client_order_id,
            r.side.as_str(),
            r.quantity,
            r.symbol,
            r.kind,
            r.strategy_id
        )
    }

    fn __richcmp__(&self, other: &Self, op: CompareOp) -> PyResult<bool> {
        match op {
            CompareOp::Eq => Ok(self.0 == other.0),
            CompareOp::Ne => Ok(self.0 != other.0),
            _ => Err(pyo3::exceptions::PyTypeError::new_err("OrderRequest supports == only")),
        }
    }
}

fn trail(kind: &OrderKind) -> Option<(f64, &'static str)> {
    match kind {
        OrderKind::TrailingStopMarket { offset } | OrderKind::TrailingStopLimit { offset, .. } => {
            Some(match offset {
                TrailOffset::Price(p) => (*p, "price"),
                TrailOffset::Bps(b) => (*b, "bps"),
            })
        }
        _ => None,
    }
}

/// What a venue says about one order. Immutable; build one with the
/// constructors (`ExecutionReport.fill(...)`) and give it to
/// `LiveNode.push_report`. Every report carries the venue's own time,
/// `ts_ns`.
#[pyclass(name = "ExecutionReport", module = "raptorbt.execution", frozen)]
#[derive(Clone)]
pub struct PyExecutionReport(pub ExecutionReport);

#[pymethods]
impl PyExecutionReport {
    #[staticmethod]
    fn accepted(venue_order_id: String, ts_ns: i64) -> Self {
        Self(ExecutionReport::Accepted { venue_order_id, ts_ns })
    }
    #[staticmethod]
    fn rejected(reason: String, ts_ns: i64) -> Self {
        Self(ExecutionReport::Rejected { reason, ts_ns })
    }
    /// A (partial) execution. `exec_id` is the venue's execution id: the same
    /// id twice counts once.
    #[staticmethod]
    #[pyo3(signature = (exec_id, qty, price, ts_ns, fee=0.0))]
    fn fill(exec_id: String, qty: f64, price: f64, ts_ns: i64, fee: f64) -> Self {
        Self(ExecutionReport::Fill { exec_id, qty, price, fee, ts_ns })
    }
    #[staticmethod]
    fn cancel_ack(ts_ns: i64) -> Self {
        Self(ExecutionReport::CancelAck { ts_ns })
    }
    #[staticmethod]
    fn cancel_rejected(reason: String, ts_ns: i64) -> Self {
        Self(ExecutionReport::CancelRejected { reason, ts_ns })
    }
    /// The venue confirmed a change. `new_qty` is the new *total* quantity.
    #[staticmethod]
    #[pyo3(signature = (ts_ns, new_qty=None, new_limit=None, new_trigger=None))]
    fn replace_ack(
        ts_ns: i64,
        new_qty: Option<f64>,
        new_limit: Option<f64>,
        new_trigger: Option<f64>,
    ) -> Self {
        Self(ExecutionReport::ReplaceAck { new_qty, new_limit, new_trigger, ts_ns })
    }
    #[staticmethod]
    fn replace_rejected(reason: String, ts_ns: i64) -> Self {
        Self(ExecutionReport::ReplaceRejected { reason, ts_ns })
    }
    #[staticmethod]
    fn expired(ts_ns: i64) -> Self {
        Self(ExecutionReport::Expired { ts_ns })
    }
    #[staticmethod]
    fn correction(exec_id: String, qty: f64, price: f64, ts_ns: i64) -> Self {
        Self(ExecutionReport::Correction { exec_id, qty, price, ts_ns })
    }
    #[staticmethod]
    fn bust(exec_id: String, ts_ns: i64) -> Self {
        Self(ExecutionReport::Bust { exec_id, ts_ns })
    }

    #[getter]
    fn kind(&self) -> &'static str {
        match self.0 {
            ExecutionReport::Accepted { .. } => "accepted",
            ExecutionReport::Rejected { .. } => "rejected",
            ExecutionReport::Fill { .. } => "fill",
            ExecutionReport::CancelAck { .. } => "cancel_ack",
            ExecutionReport::CancelRejected { .. } => "cancel_rejected",
            ExecutionReport::ReplaceAck { .. } => "replace_ack",
            ExecutionReport::ReplaceRejected { .. } => "replace_rejected",
            ExecutionReport::Expired { .. } => "expired",
            ExecutionReport::Correction { .. } => "correction",
            ExecutionReport::Bust { .. } => "bust",
        }
    }
    /// The venue's time for the report (ns).
    #[getter]
    fn ts_ns(&self) -> i64 {
        report_ts(&self.0)
    }
    #[getter]
    fn exec_id(&self) -> Option<&str> {
        match &self.0 {
            ExecutionReport::Fill { exec_id, .. }
            | ExecutionReport::Correction { exec_id, .. }
            | ExecutionReport::Bust { exec_id, .. } => Some(exec_id),
            _ => None,
        }
    }
    #[getter]
    fn qty(&self) -> Option<f64> {
        match self.0 {
            ExecutionReport::Fill { qty, .. } | ExecutionReport::Correction { qty, .. } => {
                Some(qty)
            }
            _ => None,
        }
    }
    #[getter]
    fn price(&self) -> Option<f64> {
        match self.0 {
            ExecutionReport::Fill { price, .. } | ExecutionReport::Correction { price, .. } => {
                Some(price)
            }
            _ => None,
        }
    }
    #[getter]
    fn fee(&self) -> Option<f64> {
        match self.0 {
            ExecutionReport::Fill { fee, .. } => Some(fee),
            _ => None,
        }
    }
    #[getter]
    fn reason(&self) -> Option<&str> {
        match &self.0 {
            ExecutionReport::Rejected { reason, .. }
            | ExecutionReport::CancelRejected { reason, .. }
            | ExecutionReport::ReplaceRejected { reason, .. } => Some(reason),
            _ => None,
        }
    }
    #[getter]
    fn venue_order_id(&self) -> Option<&str> {
        match &self.0 {
            ExecutionReport::Accepted { venue_order_id, .. } => Some(venue_order_id),
            _ => None,
        }
    }

    fn __repr__(&self) -> String {
        format!("ExecutionReport.{:?}", self.0)
    }

    fn __richcmp__(&self, other: &Self, op: CompareOp) -> PyResult<bool> {
        match op {
            CompareOp::Eq => Ok(self.0 == other.0),
            CompareOp::Ne => Ok(self.0 != other.0),
            _ => Err(pyo3::exceptions::PyTypeError::new_err("ExecutionReport supports == only")),
        }
    }
}

pub(crate) fn report_ts(report: &ExecutionReport) -> i64 {
    match report {
        ExecutionReport::Accepted { ts_ns, .. }
        | ExecutionReport::Rejected { ts_ns, .. }
        | ExecutionReport::Fill { ts_ns, .. }
        | ExecutionReport::CancelAck { ts_ns }
        | ExecutionReport::CancelRejected { ts_ns, .. }
        | ExecutionReport::ReplaceAck { ts_ns, .. }
        | ExecutionReport::ReplaceRejected { ts_ns, .. }
        | ExecutionReport::Expired { ts_ns }
        | ExecutionReport::Correction { ts_ns, .. }
        | ExecutionReport::Bust { ts_ns, .. } => *ts_ns,
    }
}

/// A Python object used as a venue.
///
/// The object needs `submit(request)`, `cancel(client_order_id)` and
/// `modify(client_order_id, *, quantity=None, limit_price=None,
/// trigger_price=None)`, and optionally `link_oco(client_order_ids)`.
/// `raptorbt.execution.OrderRefused` and `Unsupported` raised from them map
/// to the matching [`AdapterError`]; anything else is kept in `raised` and
/// re-raised unchanged by the runtime, so the caller sees the real exception.
pub struct PyAdapter {
    obj: Py<PyAny>,
    raised: Rc<RefCell<Option<PyErr>>>,
}

impl PyAdapter {
    pub fn new(obj: Py<PyAny>, raised: Rc<RefCell<Option<PyErr>>>) -> Self {
        Self { obj, raised }
    }

    fn call(
        &self,
        f: impl for<'py> FnOnce(Python<'py>, &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>>,
    ) -> Result<(), AdapterError> {
        Python::attach(|py| match f(py, self.obj.bind(py)) {
            Ok(_) => Ok(()),
            Err(e) => Err(self.classify(py, e)),
        })
    }

    fn classify(&self, py: Python<'_>, e: PyErr) -> AdapterError {
        let classes = py.import("raptorbt.execution").ok().and_then(|m| {
            Some((
                m.getattr("OrderRefused").ok()?,
                m.getattr("Unsupported").ok()?,
                m.getattr("Fenced").ok()?,
            ))
        });
        if let Some((refused, unsupported, fenced)) = classes {
            if e.is_instance(py, &refused) {
                return AdapterError::Refused(e.value(py).to_string());
            }
            if e.is_instance(py, &unsupported) {
                return AdapterError::Unsupported(e.value(py).to_string());
            }
            if e.is_instance(py, &fenced) {
                return AdapterError::Fenced(e.value(py).to_string());
            }
        }
        let message = e.value(py).to_string();
        *self.raised.borrow_mut() = Some(e);
        AdapterError::Unavailable(message)
    }
}

impl ExecutionAdapter for PyAdapter {
    fn submit(&mut self, req: &OrderRequest) -> Result<(), AdapterError> {
        self.call(|py, obj| {
            obj.call_method1("submit", (Py::new(py, PyOrderRequest(req.clone()))?,))
        })
    }

    fn cancel(&mut self, client_order_id: &str) -> Result<(), AdapterError> {
        self.call(|_, obj| obj.call_method1("cancel", (client_order_id,)))
    }

    fn modify(&mut self, client_order_id: &str, change: &OrderChange) -> Result<(), AdapterError> {
        self.call(|py, obj| {
            let kw = PyDict::new(py);
            kw.set_item("quantity", change.quantity)?;
            kw.set_item("limit_price", change.limit_price)?;
            kw.set_item("trigger_price", change.trigger_price)?;
            obj.call_method("modify", PyTuple::new(py, [client_order_id])?, Some(&kw))
        })
    }

    fn fence(&mut self, epoch: u64) -> Result<(), AdapterError> {
        let has = Python::attach(|py| {
            self.obj.bind(py).getattr_opt("fence").ok().flatten().is_some_and(|m| m.is_callable())
        });
        if !has {
            return Ok(());
        }
        self.call(|_, obj| obj.call_method1("fence", (epoch,)))
    }

    fn resting_exits_hold_the_line(&self) -> bool {
        Python::attach(|py| {
            self.obj
                .bind(py)
                .getattr_opt("resting_exits_hold_the_line")
                .ok()
                .flatten()
                .is_some_and(|v| v.is_truthy().unwrap_or(false))
        })
    }

    fn link_oco(&mut self, client_order_ids: &[String]) -> Result<(), AdapterError> {
        let has = Python::attach(|py| self.obj.bind(py).hasattr("link_oco").unwrap_or(false));
        if !has {
            return Err(AdapterError::Unsupported("one-cancels-other links".into()));
        }
        self.call(|_, obj| obj.call_method1("link_oco", (client_order_ids.to_vec(),)))
    }
}
