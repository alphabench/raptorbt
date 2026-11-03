//! The adapter seam of the journal: every request is written down **before**
//! the adapter is called, and what it came to is written after.
//!
//! Wrapping is by composition: the wrapper is an [`ExecutionAdapter`] around
//! the real one, so the adapter, the engine and the venue route are unchanged
//! and a node with no journal and no epoch is not wrapped at all.
//!
//! While a node is rebuilt from its journal the wrapper does not call a real
//! venue (`suppress`): a request already in the journal is checked against the
//! one the replay makes, and the answer the venue gave then is given again. A
//! request whose result was never written has an outcome nobody knows, and
//! answers "unavailable": the order stays as it was and reconciliation, not a
//! blind resend, settles it.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use super::core::Journal;
use super::record::{kind, source};
use super::value::Value;
use crate::execution::adapter::{
    AdapterError, ExecutionAdapter, OrderChange, OrderRequest, PaperVenue,
};

/// State shared between the node, the runtime and the wrapper.
#[derive(Default)]
pub struct Control {
    /// Do not call a real venue (a replay is rebuilding the node).
    pub suppress: Cell<bool>,
    /// This node's execution epoch, if it was given one.
    pub epoch: Cell<Option<u64>>,
    /// A newer epoch owns the account: no request leaves any more.
    pub fenced: Cell<bool>,
    /// The time of the event being applied: (event ns, arrival ns).
    pub now: Cell<(i64, i64)>,
    /// Adapter calls a replay answered "unavailable" for since last read.
    pub replay_errors: Cell<u32>,
    /// Orders whose request may or may not have reached the venue: sent and
    /// answered with an error, or written to the journal with no result, or
    /// not sent by a recovery. Reconciliation settles them.
    pub unconfirmed: RefCell<std::collections::BTreeSet<String>>,
}

pub struct JournalingAdapter {
    inner: Box<dyn ExecutionAdapter>,
    journal: Option<Rc<RefCell<Journal>>>,
    control: Rc<Control>,
    simulated: bool,
}

/// A request's outcome, as the journal states it.
enum Outcome {
    Ok,
    Refused(String),
    Unsupported(String),
    Error(String),
    Fenced(String),
    /// The request is in the journal and its result is not.
    Unknown,
    /// A recovery would have sent it, and did not.
    NotSent,
}

impl Outcome {
    fn of(r: &Result<(), AdapterError>) -> Outcome {
        match r {
            Ok(()) => Outcome::Ok,
            Err(AdapterError::Refused(m)) => Outcome::Refused(m.clone()),
            Err(AdapterError::Unsupported(m)) => Outcome::Unsupported(m.clone()),
            Err(AdapterError::Unavailable(m)) => Outcome::Error(m.clone()),
            Err(AdapterError::Fenced(m)) => Outcome::Fenced(m.clone()),
            Err(AdapterError::Journal(m)) => Outcome::Error(m.clone()),
        }
    }

    fn status(&self) -> (&'static str, &str) {
        match self {
            Outcome::Ok => ("ok", ""),
            Outcome::Refused(m) => ("refused", m),
            Outcome::Unsupported(m) => ("unsupported", m),
            Outcome::Error(m) => ("error", m),
            Outcome::Fenced(m) => ("fenced", m),
            Outcome::Unknown => ("unknown_outcome", "the request was written; its result was not"),
            Outcome::NotSent => ("not_sent", "recovery did not send the request"),
        }
    }

    fn from_status(status: &str, message: &str) -> Outcome {
        let m = message.to_string();
        match status {
            "ok" => Outcome::Ok,
            "refused" => Outcome::Refused(m),
            "unsupported" => Outcome::Unsupported(m),
            "fenced" => Outcome::Fenced(m),
            "unknown_outcome" => Outcome::Unknown,
            "not_sent" => Outcome::NotSent,
            "error" => Outcome::Error(m),
            // A status this code never writes: the record is not the journal's own. Say so,
            // rather than reading it as an ordinary error.
            other => Outcome::Error(format!(
                "the journal holds an adapter result with the unknown status {other:?}: {m}"
            )),
        }
    }

    fn into_result(self, control: &Control) -> Result<(), AdapterError> {
        match self {
            Outcome::Ok => Ok(()),
            Outcome::Refused(m) => Err(AdapterError::Refused(m)),
            Outcome::Unsupported(m) => Err(AdapterError::Unsupported(m)),
            Outcome::Fenced(m) => Err(AdapterError::Fenced(m)),
            Outcome::Error(m) => {
                control.replay_errors.set(control.replay_errors.get() + 1);
                Err(AdapterError::Unavailable(m))
            }
            Outcome::Unknown | Outcome::NotSent => {
                control.replay_errors.set(control.replay_errors.get() + 1);
                let (_, m) = self.status();
                Err(AdapterError::Unavailable(m.to_string()))
            }
        }
    }
}

impl JournalingAdapter {
    pub fn new(
        mut inner: Box<dyn ExecutionAdapter>,
        journal: Option<Rc<RefCell<Journal>>>,
        control: Rc<Control>,
    ) -> Self {
        let simulated = inner.as_simulation().is_some();
        JournalingAdapter { inner, journal, control, simulated }
    }

    fn append(&self, k: &str, cid: Option<&str>, payload: Value) -> Result<(), AdapterError> {
        let Some(j) = &self.journal else { return Ok(()) };
        let (ev, arr) = self.control.now.get();
        j.borrow_mut()
            .append(k, ev, arr, source::ENGINE, cid, payload)
            .map(|_| ())
            .map_err(|e| AdapterError::Journal(e.to_string()))
    }

    /// Check the epoch with the venue before a request leaves.
    fn check_fence(&mut self) -> Result<(), AdapterError> {
        if self.control.fenced.get() {
            return Err(AdapterError::Fenced("a newer execution epoch owns the account".into()));
        }
        let Some(epoch) = self.control.epoch.get() else { return Ok(()) };
        if self.control.suppress.get() && !self.simulated {
            return Ok(());
        }
        match self.inner.fence(epoch) {
            Ok(()) => Ok(()),
            Err(AdapterError::Fenced(why)) => {
                self.control.fenced.set(true);
                let _ = self.append(
                    kind::FENCE,
                    None,
                    Value::map([("epoch", Value::Int(epoch as i64)), ("reason", Value::str(&why))]),
                );
                Err(AdapterError::Fenced(why))
            }
            Err(other) => Err(other),
        }
    }

    /// One request: fence, write it, call (or replay) the adapter, write the
    /// result.
    fn request(
        &mut self,
        op: &str,
        cid: Option<&str>,
        detail: Value,
        call: impl FnOnce(&mut dyn ExecutionAdapter) -> Result<(), AdapterError>,
    ) -> Result<(), AdapterError> {
        self.check_fence()?;
        let before = self.journal.as_ref().map_or(0, |j| j.borrow().replayed());
        self.append(
            kind::ADAPTER_REQUEST,
            cid,
            Value::map([("op", Value::str(op)), ("request", detail)]),
        )?;
        let replayed_request =
            self.journal.as_ref().is_some_and(|j| j.borrow().replayed() > before);
        let suppressed = self.control.suppress.get() && !self.simulated;
        let outcome = if suppressed {
            self.replayed_outcome(replayed_request)
        } else {
            let r = call(&mut *self.inner);
            // A recovery that is not suppressed (a paper venue) still gets the
            // journal's word for errors, which a simulation does not make.
            Outcome::of(&r)
        };
        let (status, message) = outcome.status();
        // The request is durable already; a result that cannot be written
        // leaves "unknown outcome" for recovery, which is safe.
        let _ = self.append(
            kind::ADAPTER_RESULT,
            cid,
            Value::map([
                ("op", Value::str(op)),
                ("status", Value::str(status)),
                ("message", Value::str(message)),
            ]),
        );
        if let Some(id) = cid {
            if matches!(outcome, Outcome::Error(_) | Outcome::Unknown | Outcome::NotSent) {
                self.control.unconfirmed.borrow_mut().insert(id.to_string());
            } else if matches!(outcome, Outcome::Ok) {
                self.control.unconfirmed.borrow_mut().remove(id);
            }
        }
        outcome.into_result(&self.control)
    }

    /// What a replay answers for a request it cannot send.
    fn replayed_outcome(&self, request_was_in_journal: bool) -> Outcome {
        let Some(j) = &self.journal else { return Outcome::NotSent };
        let j = j.borrow();
        if !request_was_in_journal {
            return Outcome::NotSent;
        }
        match j.peek_expected() {
            Some(next) if next.record.kind == kind::ADAPTER_RESULT => {
                let payload = &next.record.payload;
                let Some(status) = payload.get("status").and_then(Value::as_str) else {
                    // An adapter_result always names its status; one that does not is damaged.
                    return Outcome::Error(
                        "the journal holds an adapter result with no status".to_string(),
                    );
                };
                let message = payload.get("message").and_then(Value::as_str).unwrap_or("");
                Outcome::from_status(status, message)
            }
            _ => Outcome::Unknown,
        }
    }
}

/// An order request as the journal states it.
pub fn request_value(req: &OrderRequest) -> Value {
    let mut value = Value::map([
        ("client_order_id", Value::str(&req.client_order_id)),
        ("symbol", Value::str(&req.symbol)),
        ("side", Value::str(format!("{:?}", req.side).to_lowercase())),
        ("quantity", Value::Float(req.quantity)),
        ("kind", from_serde(&serde_json::to_value(req.kind).unwrap_or(serde_json::Value::Null))),
        ("tif", from_serde(&serde_json::to_value(req.tif).unwrap_or(serde_json::Value::Null))),
        ("strategy_id", Value::Int(req.strategy_id as i64)),
        ("account_id", Value::opt_str(req.account_id.as_deref())),
        ("reduce_only", Value::Bool(req.reduce_only)),
        ("post_only", Value::Bool(req.post_only)),
        ("parent_client_order_id", Value::opt_str(req.parent_client_order_id.as_deref())),
        ("tags", Value::Map(req.tags.iter().map(|(k, v)| (k.clone(), Value::str(v))).collect())),
        ("submitted_ns", Value::Int(req.submitted_ns)),
        ("submitted_idx", Value::Int(req.submitted_idx as i64)),
    ]);
    // Written only when set (hedging), so the records of an order that names no position are the
    // bytes they always were and an older journal replays.
    if let (Some(position_id), Value::Map(m)) = (req.position_id, &mut value) {
        m.insert("position_id".to_string(), Value::Int(position_id as i64));
    }
    value
}

/// A change request as the journal states it.
pub fn change_value(change: &OrderChange) -> Value {
    Value::map([
        ("quantity", Value::opt_float(change.quantity)),
        ("limit_price", Value::opt_float(change.limit_price)),
        ("trigger_price", Value::opt_float(change.trigger_price)),
    ])
}

fn from_serde(v: &serde_json::Value) -> Value {
    use serde_json::Value as J;
    match v {
        J::Null => Value::Null,
        J::Bool(b) => Value::Bool(*b),
        J::Number(n) => match n.as_i64() {
            Some(i) => Value::Int(i),
            None => Value::Float(n.as_f64().unwrap_or(f64::NAN)),
        },
        J::String(s) => Value::Str(s.clone()),
        J::Array(a) => Value::List(a.iter().map(from_serde).collect()),
        J::Object(o) => Value::Map(o.iter().map(|(k, v)| (k.clone(), from_serde(v))).collect()),
    }
}

impl ExecutionAdapter for JournalingAdapter {
    fn submit(&mut self, req: &OrderRequest) -> Result<(), AdapterError> {
        let id = req.client_order_id.clone();
        self.request("submit", Some(&id), request_value(req), |a| a.submit(req))
    }

    fn cancel(&mut self, client_order_id: &str) -> Result<(), AdapterError> {
        self.request("cancel", Some(client_order_id), Value::Null, |a| a.cancel(client_order_id))
    }

    fn modify(&mut self, client_order_id: &str, change: &OrderChange) -> Result<(), AdapterError> {
        self.request("modify", Some(client_order_id), change_value(change), |a| {
            a.modify(client_order_id, change)
        })
    }

    fn link_oco(&mut self, ids: &[String]) -> Result<(), AdapterError> {
        let detail = Value::List(ids.iter().map(Value::str).collect());
        self.request("link_oco", None, detail, |a| a.link_oco(ids))
    }

    /// Check the node's epoch with the venue now (a heartbeat).
    fn fence(&mut self, _epoch: u64) -> Result<(), AdapterError> {
        self.check_fence()
    }

    fn as_simulation(&mut self) -> Option<&mut PaperVenue> {
        self.inner.as_simulation()
    }
}
