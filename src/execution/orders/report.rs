//! Venue-driven order lifecycle: execution reports and the ledger behind them.
//!
//! The in-process matcher decides fills itself. An order routed to an
//! external venue instead learns what happened from a stream of reports that
//! can repeat, arrive late, or arrive out of order. [`Order::apply_report`]
//! folds that stream into the order without trusting its delivery order.
//!
//! The design rule: the order's status is *derived* from a set of recorded
//! facts (accepted, rejected, cancel acknowledged, expired, the execution
//! ledger, outstanding requests), never advanced by "what came last". Facts
//! only accumulate, so any permutation or duplication of the same reports
//! converges to the same status, filled quantity and average price. The one
//! exception is a set of reports that contradict each other (a fill after a
//! reject), which is an error.
//!
//! Sums over the ledger run in `exec_id` order so that floating-point
//! results are bit-identical whatever order the reports arrived in.
//!
//! State diagram and the full rule table: `docs/architecture/order-lifecycle.md`.

use serde::{Deserialize, Serialize};

use super::order::{Order, OrderKind, OrderStatus, QtySpec};

/// Relative slack for float arithmetic when comparing a filled quantity with
/// the order quantity. Never used to alter a quantity, only to compare.
const QTY_EPS: f64 = 1e-9;

/// One report from the venue about one order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ExecutionReport {
    /// The venue accepted the order and gave it its own id.
    Accepted { venue_order_id: String, ts_ns: i64 },
    /// The venue refused the order.
    Rejected { reason: String, ts_ns: i64 },
    /// A (partial) execution. `exec_id` is the venue's execution id and is
    /// the idempotency key: the same id never counts twice.
    Fill { exec_id: String, qty: f64, price: f64, fee: f64, ts_ns: i64 },
    /// The venue confirmed a cancel (or cancelled on its own).
    CancelAck { ts_ns: i64 },
    /// The venue refused a cancel request; the order keeps working.
    CancelRejected { reason: String, ts_ns: i64 },
    /// The venue confirmed a replace. `new_qty` is the new *total* order
    /// quantity; a `None` field is unchanged.
    ReplaceAck {
        new_qty: Option<f64>,
        new_limit: Option<f64>,
        new_trigger: Option<f64>,
        ts_ns: i64,
    },
    /// The venue refused a replace request; the order keeps its old terms.
    ReplaceRejected { reason: String, ts_ns: i64 },
    /// Time in force ran out at the venue.
    Expired { ts_ns: i64 },
    /// The venue restated an earlier execution's quantity and price. The
    /// latest correction (by `ts_ns`) wins.
    Correction { exec_id: String, qty: f64, price: f64, ts_ns: i64 },
    /// The venue cancelled an earlier execution; it no longer counts.
    Bust { exec_id: String, ts_ns: i64 },
}

impl ExecutionReport {
    fn name(&self) -> &'static str {
        match self {
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

    fn ts_ns(&self) -> i64 {
        match self {
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
}

/// What a report did to the order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReportOutcome {
    /// The report changed the order.
    Applied,
    /// The venue repeated something already recorded (same execution id,
    /// same acknowledgement); nothing changed.
    Duplicate,
    /// The report arrived after it stopped mattering or before it can take
    /// effect. Examples: a fill while a cancel is pending (still applied,
    /// because the venue did execute it); an acknowledgement for an order
    /// that already ended (recorded, status unchanged); a correction for an
    /// execution not yet seen (held until the fill arrives).
    Late,
}

/// Why a request or report was refused. The order is left unchanged.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum OrderEventError {
    #[error("order {order_id}: illegal transition {from:?} -> {to:?}")]
    IllegalTransition { order_id: u64, from: OrderStatus, to: OrderStatus },
    #[error("order {order_id}: report `{report}` contradicts status {status:?}")]
    IllegalReport { order_id: u64, status: OrderStatus, report: &'static str },
    #[error("order {order_id}: fills of {filled} exceed the order quantity {qty}")]
    OverFill { order_id: u64, qty: f64, filled: f64 },
    #[error("order {order_id}: invalid {what}")]
    InvalidValue { order_id: u64, what: &'static str },
    #[error("order {order_id}: venue order id {got} conflicts with {known}")]
    VenueIdConflict { order_id: u64, known: String, got: String },
    #[error("order {order_id}: quantity is not an explicit unit count")]
    UnresolvedQty { order_id: u64 },
    #[error("order {order_id}: already filled or ended by the in-process matcher")]
    NotVenueDriven { order_id: u64 },
    #[error("order {order_id}: a request is already pending")]
    AlreadyPending { order_id: u64 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum EndCause {
    Canceled,
    Expired,
}

/// One execution as the venue reported it, plus later restatements.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Exec {
    exec_id: String,
    /// `(qty, price, fee)` of the original fill, once seen.
    fill: Option<(f64, f64, f64)>,
    /// Latest correction: `(ts_ns, qty, price)`.
    corr: Option<(i64, f64, f64)>,
    busted: bool,
}

impl Exec {
    /// The execution as it currently counts, with its fee.
    fn effective(&self) -> Option<(f64, f64, f64)> {
        if self.busted {
            return None;
        }
        let (q, p, fee) = self.fill?;
        Some(match self.corr {
            Some((_, cq, cp)) => (cq, cp, fee),
            None => (q, p, fee),
        })
    }
}

/// Facts recorded about a venue-driven order. Crate-private; read through
/// the accessors on [`Order`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct BrokerState {
    venue_order_id: Option<String>,
    accepted: bool,
    triggered: bool,
    rejected: bool,
    cause: Option<EndCause>,
    cancel_pending: bool,
    replace_pending: bool,
    /// Sorted by `exec_id`.
    execs: Vec<Exec>,
    last_ts_ns: i64,
}

struct Totals {
    qty: f64,
    notional: f64,
    fee: f64,
}

impl BrokerState {
    fn find(&self, exec_id: &str) -> Result<usize, usize> {
        self.execs.binary_search_by(|e| e.exec_id.as_str().cmp(exec_id))
    }

    fn entry(&mut self, exec_id: &str) -> &mut Exec {
        let i = match self.find(exec_id) {
            Ok(i) => i,
            Err(i) => {
                self.execs.insert(
                    i,
                    Exec { exec_id: exec_id.to_string(), fill: None, corr: None, busted: false },
                );
                i
            }
        };
        &mut self.execs[i]
    }

    /// Sums in `exec_id` order: float addition is not associative, so a
    /// fixed order is what makes the result independent of delivery order.
    fn totals(&self) -> Totals {
        let mut t = Totals { qty: 0.0, notional: 0.0, fee: 0.0 };
        for (q, p, fee) in self.execs.iter().filter_map(Exec::effective) {
            t.qty += q;
            t.notional += q * p;
            t.fee += fee;
        }
        t
    }

    fn ended(&self, filled: f64, qty: f64) -> bool {
        self.rejected || self.cause.is_some() || filled + qty * QTY_EPS >= qty
    }

    fn derive(&self, filled: f64, qty: f64) -> OrderStatus {
        if self.rejected {
            OrderStatus::Rejected
        } else if filled + qty * QTY_EPS >= qty {
            OrderStatus::Filled
        } else if let Some(cause) = self.cause {
            match cause {
                EndCause::Canceled => OrderStatus::Canceled,
                EndCause::Expired => OrderStatus::Expired,
            }
        } else if self.cancel_pending {
            OrderStatus::PendingCancel
        } else if self.replace_pending {
            OrderStatus::PendingReplace
        } else if filled > 0.0 {
            OrderStatus::PartiallyFilled
        } else if self.triggered {
            OrderStatus::Triggered
        } else if self.accepted {
            OrderStatus::Accepted
        } else {
            OrderStatus::Submitted
        }
    }
}

fn finite_pos(v: f64) -> bool {
    v.is_finite() && v > 0.0
}

impl Order {
    /// Make the order venue-driven from the moment it exists. The in-process
    /// matcher then never touches it: its whole life is the reports.
    pub fn into_venue_driven(mut self) -> Self {
        if self.broker.is_none() {
            self.broker = Some(Box::default());
        }
        self
    }

    /// Whether venue reports, not the in-process matcher, drive this order.
    #[inline]
    pub fn is_venue_driven(&self) -> bool {
        self.broker.is_some()
    }

    /// Venue-assigned id, once the venue has acknowledged the order.
    pub fn venue_order_id(&self) -> Option<&str> {
        self.broker.as_ref().and_then(|b| b.venue_order_id.as_deref())
    }

    /// Size-weighted mean price of the executions that currently count.
    pub fn avg_fill_price(&self) -> Option<f64> {
        let t = self.broker.as_ref()?.totals();
        (t.qty > 0.0).then(|| t.notional / t.qty)
    }

    /// Fees reported on the executions that currently count.
    pub fn fees_paid(&self) -> f64 {
        self.broker.as_ref().map_or(0.0, |b| b.totals().fee)
    }

    /// Number of executions that currently count (busted ones excluded).
    pub fn exec_count(&self) -> usize {
        self.broker
            .as_ref()
            .map_or(0, |b| b.execs.iter().filter(|e| e.effective().is_some()).count())
    }

    /// Ask the venue to cancel. The order becomes `PendingCancel`; it ends
    /// as `Canceled` only when [`ExecutionReport::CancelAck`] arrives, and
    /// may fill in between.
    pub fn request_cancel(&mut self) -> Result<(), OrderEventError> {
        use OrderStatus::*;
        match self.status {
            PendingCancel => return Err(OrderEventError::AlreadyPending { order_id: self.id }),
            Submitted | Accepted | Triggered | PartiallyFilled | PendingReplace => {}
            from => {
                return Err(OrderEventError::IllegalTransition {
                    order_id: self.id,
                    from,
                    to: PendingCancel,
                })
            }
        }
        let mut st = self.venue_state()?;
        st.cancel_pending = true;
        self.commit(st, None, false)
    }

    /// Ask the venue to modify the order. The order becomes
    /// `PendingReplace` and keeps its old terms until
    /// [`ExecutionReport::ReplaceAck`] arrives.
    pub fn request_replace(&mut self) -> Result<(), OrderEventError> {
        use OrderStatus::*;
        match self.status {
            PendingReplace => return Err(OrderEventError::AlreadyPending { order_id: self.id }),
            Accepted | Triggered | PartiallyFilled => {}
            from => {
                return Err(OrderEventError::IllegalTransition {
                    order_id: self.id,
                    from,
                    to: PendingReplace,
                })
            }
        }
        let mut st = self.venue_state()?;
        st.replace_pending = true;
        self.commit(st, None, false)
    }

    /// Fold one venue report into the order.
    ///
    /// Atomic: on `Err` the order is exactly as it was. Safe to call with
    /// duplicated, late or reordered reports; see the module docs.
    pub fn apply_report(&mut self, r: &ExecutionReport) -> Result<ReportOutcome, OrderEventError> {
        let id = self.id;
        let mut st = self.venue_state()?;
        let status = self.status;
        let total = self.total_qty()?;
        let before = st.totals().qty;
        let mut new_kind = None;
        let mut new_total = None;
        let mut exec_changing = false;
        let invalid = |what: &'static str| OrderEventError::InvalidValue { order_id: id, what };
        let terminal_fact = st.ended(before, total);

        let outcome = match r {
            ExecutionReport::Accepted { venue_order_id, .. } => {
                if venue_order_id.is_empty() {
                    return Err(invalid("venue order id"));
                }
                match &st.venue_order_id {
                    Some(known) if known != venue_order_id => {
                        return Err(OrderEventError::VenueIdConflict {
                            order_id: id,
                            known: known.clone(),
                            got: venue_order_id.clone(),
                        })
                    }
                    Some(_) => ReportOutcome::Duplicate,
                    None => {
                        st.venue_order_id = Some(venue_order_id.clone());
                        st.accepted = true;
                        if terminal_fact {
                            ReportOutcome::Late
                        } else {
                            ReportOutcome::Applied
                        }
                    }
                }
            }
            ExecutionReport::Rejected { .. } => {
                if st.rejected {
                    ReportOutcome::Duplicate
                } else if before > 0.0 {
                    return Err(OrderEventError::IllegalReport {
                        order_id: id,
                        status,
                        report: r.name(),
                    });
                } else {
                    st.rejected = true;
                    st.cancel_pending = false;
                    st.replace_pending = false;
                    ReportOutcome::Applied
                }
            }
            ExecutionReport::Fill { exec_id, qty, price, fee, .. } => {
                if exec_id.is_empty() {
                    return Err(invalid("exec id"));
                }
                if !finite_pos(*qty) || !finite_pos(*price) || !fee.is_finite() || *fee < 0.0 {
                    return Err(invalid("fill qty, price or fee"));
                }
                if st.rejected {
                    return Err(OrderEventError::IllegalReport {
                        order_id: id,
                        status,
                        report: r.name(),
                    });
                }
                let e = st.entry(exec_id);
                if e.fill.is_some() {
                    ReportOutcome::Duplicate
                } else {
                    e.fill = Some((*qty, *price, *fee));
                    exec_changing = true;
                    // A fill proves the venue accepted the order.
                    st.accepted = true;
                    if status.is_pending() || status.is_terminal() {
                        ReportOutcome::Late
                    } else {
                        ReportOutcome::Applied
                    }
                }
            }
            ExecutionReport::CancelAck { .. } => {
                st.cancel_pending = false;
                if st.cause == Some(EndCause::Canceled) {
                    ReportOutcome::Duplicate
                } else if st.cause.is_some() || st.rejected {
                    ReportOutcome::Late
                } else {
                    st.cause = Some(EndCause::Canceled);
                    st.accepted = true;
                    if terminal_fact {
                        ReportOutcome::Late
                    } else {
                        ReportOutcome::Applied
                    }
                }
            }
            ExecutionReport::CancelRejected { .. } => {
                if st.cancel_pending {
                    st.cancel_pending = false;
                    ReportOutcome::Applied
                } else {
                    ReportOutcome::Late
                }
            }
            ExecutionReport::ReplaceAck { new_qty, new_limit, new_trigger, .. } => {
                for v in [new_qty, new_limit, new_trigger].into_iter().flatten() {
                    if !finite_pos(*v) {
                        return Err(invalid("replace value"));
                    }
                }
                if !st.replace_pending {
                    ReportOutcome::Late
                } else {
                    st.replace_pending = false;
                    if terminal_fact {
                        ReportOutcome::Late
                    } else {
                        new_kind = Some(with_prices(&self.kind, *new_limit, *new_trigger, id)?);
                        new_total = *new_qty;
                        ReportOutcome::Applied
                    }
                }
            }
            ExecutionReport::ReplaceRejected { .. } => {
                if st.replace_pending {
                    st.replace_pending = false;
                    ReportOutcome::Applied
                } else {
                    ReportOutcome::Late
                }
            }
            ExecutionReport::Expired { .. } => {
                st.replace_pending = false;
                st.cancel_pending = false;
                if st.cause == Some(EndCause::Expired) {
                    ReportOutcome::Duplicate
                } else if st.cause.is_some() || st.rejected {
                    ReportOutcome::Late
                } else {
                    st.cause = Some(EndCause::Expired);
                    st.accepted = true;
                    if terminal_fact {
                        ReportOutcome::Late
                    } else {
                        ReportOutcome::Applied
                    }
                }
            }
            ExecutionReport::Correction { exec_id, qty, price, ts_ns } => {
                if exec_id.is_empty() {
                    return Err(invalid("exec id"));
                }
                if !finite_pos(*qty) || !finite_pos(*price) {
                    return Err(invalid("correction qty or price"));
                }
                let e = st.entry(exec_id);
                let incoming = (*ts_ns, *qty, *price);
                let supersedes = match e.corr {
                    None => true,
                    Some(cur) => {
                        cur.0
                            .cmp(&incoming.0)
                            .then(cur.1.total_cmp(&incoming.1))
                            .then(cur.2.total_cmp(&incoming.2))
                            == std::cmp::Ordering::Less
                    }
                };
                if e.corr == Some(incoming) || e.busted {
                    ReportOutcome::Duplicate
                } else {
                    if supersedes {
                        e.corr = Some(incoming);
                    }
                    exec_changing = true;
                    if e.fill.is_some() {
                        ReportOutcome::Applied
                    } else {
                        ReportOutcome::Late
                    }
                }
            }
            ExecutionReport::Bust { exec_id, .. } => {
                if exec_id.is_empty() {
                    return Err(invalid("exec id"));
                }
                let e = st.entry(exec_id);
                if e.busted {
                    ReportOutcome::Duplicate
                } else {
                    e.busted = true;
                    exec_changing = true;
                    if e.fill.is_some() {
                        ReportOutcome::Applied
                    } else {
                        ReportOutcome::Late
                    }
                }
            }
        };
        st.last_ts_ns = st.last_ts_ns.max(r.ts_ns());
        if outcome == ReportOutcome::Duplicate {
            return Ok(outcome);
        }
        self.commit_with(st, new_kind, new_total, exec_changing)?;
        Ok(outcome)
    }

    /// The ledger for this order, created from the current status on first
    /// use. An order the in-process matcher already moved cannot switch to
    /// venue reports midway.
    fn venue_state(&self) -> Result<BrokerState, OrderEventError> {
        if let Some(b) = &self.broker {
            return Ok((**b).clone());
        }
        if self.filled_qty != 0.0 || self.status.is_terminal() {
            return Err(OrderEventError::NotVenueDriven { order_id: self.id });
        }
        Ok(BrokerState {
            accepted: self.status != OrderStatus::Submitted,
            triggered: self.status == OrderStatus::Triggered,
            cancel_pending: self.status == OrderStatus::PendingCancel,
            replace_pending: self.status == OrderStatus::PendingReplace,
            ..BrokerState::default()
        })
    }

    fn total_qty(&self) -> Result<f64, OrderEventError> {
        match self.qty {
            QtySpec::Units(u) if finite_pos(u) => Ok(u),
            QtySpec::Units(_) => {
                Err(OrderEventError::InvalidValue { order_id: self.id, what: "order quantity" })
            }
            _ => Err(OrderEventError::UnresolvedQty { order_id: self.id }),
        }
    }

    fn commit(
        &mut self,
        st: BrokerState,
        new_kind: Option<OrderKind>,
        exec_changing: bool,
    ) -> Result<(), OrderEventError> {
        self.commit_with(st, new_kind, None, exec_changing)
    }

    /// Check the over-fill and transition rules on the would-be state, then
    /// store it. Nothing is written unless every check passes.
    fn commit_with(
        &mut self,
        st: BrokerState,
        new_kind: Option<OrderKind>,
        new_total: Option<f64>,
        exec_changing: bool,
    ) -> Result<(), OrderEventError> {
        let qty = new_total.map_or_else(|| self.total_qty(), Ok)?;
        let t = st.totals();
        if t.qty > qty + qty * QTY_EPS {
            return Err(OrderEventError::OverFill { order_id: self.id, qty, filled: t.qty });
        }
        let to = st.derive(t.qty, qty);
        self.check_move(to, exec_changing)?;
        self.status = to;
        self.filled_qty = t.qty;
        if let Some(k) = new_kind {
            self.kind = k;
        }
        if new_total.is_some() {
            self.qty = QtySpec::Units(qty);
        }
        self.broker = Some(Box::new(st));
        Ok(())
    }

    /// Legal if unchanged, in the transition table, reached through the
    /// acceptance a fill implies, or a reconciliation that an execution
    /// restatement forces (a late fill completing a cancelled order, a bust
    /// reopening a filled one).
    fn check_move(&self, to: OrderStatus, exec_changing: bool) -> Result<(), OrderEventError> {
        let from = self.status;
        let implied_accept =
            from == OrderStatus::Submitted && OrderStatus::Accepted.can_transition_to(to);
        let reconcile = exec_changing
            && (matches!(from, OrderStatus::Filled | OrderStatus::Canceled | OrderStatus::Expired)
                || matches!(
                    to,
                    OrderStatus::Accepted | OrderStatus::PartiallyFilled | OrderStatus::Filled
                ));
        if from == to || from.can_transition_to(to) || implied_accept || reconcile {
            Ok(())
        } else {
            Err(OrderEventError::IllegalTransition { order_id: self.id, from, to })
        }
    }
}

/// The order kind with its venue-acknowledged limit/trigger changes.
fn with_prices(
    kind: &OrderKind,
    limit: Option<f64>,
    trigger: Option<f64>,
    order_id: u64,
) -> Result<OrderKind, OrderEventError> {
    let bad =
        || OrderEventError::InvalidValue { order_id, what: "replace price for this order kind" };
    Ok(match *kind {
        OrderKind::Limit { price } => {
            if trigger.is_some() {
                return Err(bad());
            }
            OrderKind::Limit { price: limit.unwrap_or(price) }
        }
        OrderKind::StopMarket { trigger: t } => {
            if limit.is_some() {
                return Err(bad());
            }
            OrderKind::StopMarket { trigger: trigger.unwrap_or(t) }
        }
        OrderKind::MarketIfTouched { trigger: t } => {
            if limit.is_some() {
                return Err(bad());
            }
            OrderKind::MarketIfTouched { trigger: trigger.unwrap_or(t) }
        }
        OrderKind::StopLimit { trigger: t, price } => {
            OrderKind::StopLimit { trigger: trigger.unwrap_or(t), price: limit.unwrap_or(price) }
        }
        OrderKind::LimitIfTouched { trigger: t, price } => OrderKind::LimitIfTouched {
            trigger: trigger.unwrap_or(t),
            price: limit.unwrap_or(price),
        },
        k => {
            if limit.is_some() || trigger.is_some() {
                return Err(bad());
            }
            k
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::orders::order::{OrderSide, TimeInForce};
    use OrderStatus::*;

    fn order(qty: f64) -> Order {
        let mut o = Order::plain(
            OrderSide::Buy,
            QtySpec::Units(qty),
            OrderKind::Limit { price: 100.0 },
            TimeInForce::Gtc,
        );
        o.id = 7;
        o
    }

    fn acc() -> ExecutionReport {
        ExecutionReport::Accepted { venue_order_id: "V1".into(), ts_ns: 1 }
    }
    fn fill(id: &str, qty: f64, price: f64) -> ExecutionReport {
        ExecutionReport::Fill { exec_id: id.into(), qty, price, fee: 0.5, ts_ns: 2 }
    }
    fn ok(o: &mut Order, r: ExecutionReport) -> ReportOutcome {
        o.apply_report(&r).unwrap()
    }

    #[test]
    fn accept_then_full_fill() {
        let mut o = order(10.0);
        assert_eq!(ok(&mut o, acc()), ReportOutcome::Applied);
        assert_eq!(o.status, Accepted);
        assert_eq!(o.venue_order_id(), Some("V1"));
        assert_eq!(ok(&mut o, fill("E1", 4.0, 100.0)), ReportOutcome::Applied);
        assert_eq!(o.status, PartiallyFilled);
        assert_eq!(ok(&mut o, fill("E2", 6.0, 101.0)), ReportOutcome::Applied);
        assert_eq!(o.status, Filled);
        assert_eq!(o.filled_qty, 10.0);
        assert!((o.avg_fill_price().unwrap() - 100.6).abs() < 1e-12);
        assert_eq!(o.fees_paid(), 1.0);
    }

    #[test]
    fn reject_from_submitted() {
        let mut o = order(1.0);
        assert_eq!(
            ok(&mut o, ExecutionReport::Rejected { reason: "x".into(), ts_ns: 1 }),
            ReportOutcome::Applied
        );
        assert_eq!(o.status, Rejected);
        assert_eq!(
            ok(&mut o, ExecutionReport::Rejected { reason: "x".into(), ts_ns: 1 }),
            ReportOutcome::Duplicate
        );
        // Anything that needs a live order is refused after a reject.
        let e = o.apply_report(&fill("E", 1.0, 1.0)).unwrap_err();
        assert!(matches!(e, OrderEventError::IllegalReport { .. }));
        assert!(o.request_cancel().is_err());
    }

    #[test]
    fn fill_before_accept_implies_accept() {
        let mut o = order(10.0);
        assert_eq!(ok(&mut o, fill("E1", 3.0, 99.0)), ReportOutcome::Applied);
        assert_eq!(o.status, PartiallyFilled);
        assert_eq!(o.venue_order_id(), None);
        // The late Accepted still names the order.
        assert_eq!(ok(&mut o, acc()), ReportOutcome::Applied);
        assert_eq!(o.venue_order_id(), Some("V1"));
        assert_eq!(o.status, PartiallyFilled);
    }

    #[test]
    fn duplicate_fill_counts_once() {
        let mut o = order(10.0);
        ok(&mut o, acc());
        ok(&mut o, fill("E1", 4.0, 100.0));
        assert_eq!(ok(&mut o, fill("E1", 4.0, 100.0)), ReportOutcome::Duplicate);
        assert_eq!(o.filled_qty, 4.0);
        assert_eq!(ok(&mut o, acc()), ReportOutcome::Duplicate);
    }

    #[test]
    fn venue_id_conflict_is_an_error() {
        let mut o = order(10.0);
        ok(&mut o, acc());
        let e = o
            .apply_report(&ExecutionReport::Accepted { venue_order_id: "V2".into(), ts_ns: 3 })
            .unwrap_err();
        assert!(matches!(e, OrderEventError::VenueIdConflict { .. }));
    }

    #[test]
    fn overfill_is_an_error_and_leaves_order_unchanged() {
        let mut o = order(10.0);
        ok(&mut o, acc());
        ok(&mut o, fill("E1", 8.0, 100.0));
        let before = o.clone();
        let e = o.apply_report(&fill("E2", 3.0, 100.0)).unwrap_err();
        assert!(matches!(e, OrderEventError::OverFill { .. }));
        assert_eq!(o, before);
    }

    #[test]
    fn invalid_values_refused() {
        let mut o = order(10.0);
        for r in [
            fill("E", 0.0, 1.0),
            fill("E", f64::NAN, 1.0),
            fill("E", 1.0, -1.0),
            fill("", 1.0, 1.0),
            ExecutionReport::Accepted { venue_order_id: String::new(), ts_ns: 0 },
        ] {
            assert!(matches!(
                o.apply_report(&r).unwrap_err(),
                OrderEventError::InvalidValue { .. }
            ));
        }
        assert_eq!(o.status, Submitted);
    }

    #[test]
    fn cancel_request_is_not_cancel_ack() {
        let mut o = order(10.0);
        ok(&mut o, acc());
        o.request_cancel().unwrap();
        assert_eq!(o.status, PendingCancel);
        assert_eq!(
            o.request_cancel().unwrap_err(),
            OrderEventError::AlreadyPending { order_id: 7 }
        );
        // The venue fills it before it sees the cancel: real, so applied.
        assert_eq!(ok(&mut o, fill("E1", 4.0, 100.0)), ReportOutcome::Late);
        assert_eq!(o.status, PendingCancel);
        assert_eq!(o.filled_qty, 4.0);
        assert_eq!(ok(&mut o, ExecutionReport::CancelAck { ts_ns: 5 }), ReportOutcome::Applied);
        assert_eq!(o.status, Canceled);
        assert_eq!(o.filled_qty, 4.0);
    }

    #[test]
    fn cancel_before_ack_from_submitted() {
        let mut o = order(10.0);
        o.request_cancel().unwrap();
        assert_eq!(o.status, PendingCancel);
        ok(&mut o, acc());
        assert_eq!(o.status, PendingCancel);
        ok(&mut o, ExecutionReport::CancelAck { ts_ns: 5 });
        assert_eq!(o.status, Canceled);
    }

    #[test]
    fn cancel_rejected_returns_to_working_state() {
        let mut o = order(10.0);
        ok(&mut o, acc());
        o.request_cancel().unwrap();
        let r = ExecutionReport::CancelRejected { reason: "too late".into(), ts_ns: 4 };
        assert_eq!(ok(&mut o, r.clone()), ReportOutcome::Applied);
        assert_eq!(o.status, Accepted);
        assert_eq!(ok(&mut o, r), ReportOutcome::Late);
        ok(&mut o, fill("E1", 1.0, 100.0));
        o.request_cancel().unwrap();
        ok(&mut o, ExecutionReport::CancelRejected { reason: "x".into(), ts_ns: 6 });
        assert_eq!(o.status, PartiallyFilled);
    }

    #[test]
    fn cancel_ack_after_full_fill_is_late_and_keeps_filled() {
        let mut o = order(5.0);
        ok(&mut o, acc());
        o.request_cancel().unwrap();
        ok(&mut o, fill("E1", 5.0, 100.0));
        assert_eq!(o.status, Filled);
        assert_eq!(ok(&mut o, ExecutionReport::CancelAck { ts_ns: 9 }), ReportOutcome::Late);
        assert_eq!(o.status, Filled);
    }

    #[test]
    fn late_fill_after_cancel_ack_is_kept() {
        let mut o = order(10.0);
        ok(&mut o, acc());
        ok(&mut o, ExecutionReport::CancelAck { ts_ns: 3 });
        assert_eq!(o.status, Canceled);
        assert_eq!(ok(&mut o, fill("E1", 4.0, 100.0)), ReportOutcome::Late);
        assert_eq!(o.status, Canceled);
        assert_eq!(o.filled_qty, 4.0);
        // A fill that completes the order makes it Filled: it traded.
        ok(&mut o, fill("E2", 6.0, 100.0));
        assert_eq!(o.status, Filled);
    }

    #[test]
    fn expiry_and_conflicting_end() {
        let mut o = order(10.0);
        ok(&mut o, acc());
        ok(&mut o, fill("E1", 2.0, 100.0));
        assert_eq!(ok(&mut o, ExecutionReport::Expired { ts_ns: 8 }), ReportOutcome::Applied);
        assert_eq!(o.status, Expired);
        assert_eq!(ok(&mut o, ExecutionReport::Expired { ts_ns: 8 }), ReportOutcome::Duplicate);
        assert_eq!(ok(&mut o, ExecutionReport::CancelAck { ts_ns: 9 }), ReportOutcome::Late);
        assert_eq!(o.status, Expired);
    }

    #[test]
    fn replace_request_then_ack_changes_terms() {
        let mut o = order(10.0);
        assert!(o.request_replace().is_err()); // not yet accepted
        ok(&mut o, acc());
        o.request_replace().unwrap();
        assert_eq!(o.status, PendingReplace);
        assert_eq!(o.kind, OrderKind::Limit { price: 100.0 }, "old terms until acked");
        let ack = ExecutionReport::ReplaceAck {
            new_qty: Some(12.0),
            new_limit: Some(99.5),
            new_trigger: None,
            ts_ns: 5,
        };
        assert_eq!(ok(&mut o, ack.clone()), ReportOutcome::Applied);
        assert_eq!(o.status, Accepted);
        assert_eq!(o.kind, OrderKind::Limit { price: 99.5 });
        assert_eq!(o.qty, QtySpec::Units(12.0));
        // No request outstanding: a repeated ack changes nothing.
        assert_eq!(ok(&mut o, ack), ReportOutcome::Late);
    }

    #[test]
    fn replace_rejected_keeps_terms() {
        let mut o = order(10.0);
        ok(&mut o, acc());
        o.request_replace().unwrap();
        ok(&mut o, ExecutionReport::ReplaceRejected { reason: "x".into(), ts_ns: 4 });
        assert_eq!(o.status, Accepted);
        assert_eq!(o.qty, QtySpec::Units(10.0));
    }

    #[test]
    fn replace_wrong_price_for_kind_and_below_filled() {
        let mut o = order(10.0);
        ok(&mut o, acc());
        ok(&mut o, fill("E1", 6.0, 100.0));
        o.request_replace().unwrap();
        let bad_kind = ExecutionReport::ReplaceAck {
            new_qty: None,
            new_limit: None,
            new_trigger: Some(1.0),
            ts_ns: 5,
        };
        assert!(matches!(
            o.apply_report(&bad_kind).unwrap_err(),
            OrderEventError::InvalidValue { .. }
        ));
        let below = ExecutionReport::ReplaceAck {
            new_qty: Some(5.0),
            new_limit: None,
            new_trigger: None,
            ts_ns: 5,
        };
        assert!(matches!(o.apply_report(&below).unwrap_err(), OrderEventError::OverFill { .. }));
        assert_eq!(o.status, PendingReplace);
    }

    #[test]
    fn replace_that_equals_filled_completes_the_order() {
        let mut o = order(10.0);
        ok(&mut o, acc());
        ok(&mut o, fill("E1", 6.0, 100.0));
        o.request_replace().unwrap();
        ok(
            &mut o,
            ExecutionReport::ReplaceAck {
                new_qty: Some(6.0),
                new_limit: None,
                new_trigger: None,
                ts_ns: 5,
            },
        );
        assert_eq!(o.status, Filled);
    }

    #[test]
    fn cancel_while_replace_pending_and_back() {
        let mut o = order(10.0);
        ok(&mut o, acc());
        o.request_replace().unwrap();
        o.request_cancel().unwrap();
        assert_eq!(o.status, PendingCancel);
        assert!(o.request_replace().is_err());
        ok(&mut o, ExecutionReport::CancelRejected { reason: "x".into(), ts_ns: 4 });
        assert_eq!(o.status, PendingReplace);
        ok(&mut o, ExecutionReport::ReplaceRejected { reason: "x".into(), ts_ns: 5 });
        assert_eq!(o.status, Accepted);
    }

    #[test]
    fn correction_restates_fill() {
        let mut o = order(10.0);
        ok(&mut o, acc());
        ok(&mut o, fill("E1", 4.0, 100.0));
        let c =
            ExecutionReport::Correction { exec_id: "E1".into(), qty: 5.0, price: 102.0, ts_ns: 9 };
        assert_eq!(ok(&mut o, c.clone()), ReportOutcome::Applied);
        assert_eq!(o.filled_qty, 5.0);
        assert_eq!(o.avg_fill_price(), Some(102.0));
        assert_eq!(ok(&mut o, c), ReportOutcome::Duplicate);
        // An older correction does not displace a newer one.
        ok(
            &mut o,
            ExecutionReport::Correction { exec_id: "E1".into(), qty: 1.0, price: 1.0, ts_ns: 3 },
        );
        assert_eq!(o.filled_qty, 5.0);
    }

    #[test]
    fn correction_that_overfills_is_refused() {
        let mut o = order(10.0);
        ok(&mut o, acc());
        ok(&mut o, fill("E1", 4.0, 100.0));
        let c =
            ExecutionReport::Correction { exec_id: "E1".into(), qty: 11.0, price: 1.0, ts_ns: 9 };
        assert!(matches!(o.apply_report(&c).unwrap_err(), OrderEventError::OverFill { .. }));
        assert_eq!(o.filled_qty, 4.0);
    }

    #[test]
    fn bust_removes_fill_and_reopens_filled_order() {
        let mut o = order(10.0);
        ok(&mut o, acc());
        ok(&mut o, fill("E1", 10.0, 100.0));
        assert_eq!(o.status, Filled);
        let b = ExecutionReport::Bust { exec_id: "E1".into(), ts_ns: 9 };
        assert_eq!(ok(&mut o, b.clone()), ReportOutcome::Applied);
        assert_eq!(o.status, Accepted);
        assert_eq!(o.filled_qty, 0.0);
        assert_eq!(o.avg_fill_price(), None);
        assert_eq!(ok(&mut o, b), ReportOutcome::Duplicate);
        // The busted execution never counts again, even if re-sent.
        assert_eq!(ok(&mut o, fill("E1", 10.0, 100.0)), ReportOutcome::Duplicate);
        assert_eq!(o.filled_qty, 0.0);
    }

    #[test]
    fn bust_and_correction_before_fill_are_held() {
        let mut o = order(10.0);
        let b = ExecutionReport::Bust { exec_id: "E1".into(), ts_ns: 9 };
        assert_eq!(ok(&mut o, b), ReportOutcome::Late);
        ok(&mut o, fill("E1", 4.0, 100.0));
        assert_eq!(o.filled_qty, 0.0);
        let mut p = order(10.0);
        let c =
            ExecutionReport::Correction { exec_id: "E2".into(), qty: 3.0, price: 50.0, ts_ns: 9 };
        assert_eq!(ok(&mut p, c), ReportOutcome::Late);
        assert_eq!(p.filled_qty, 0.0);
        ok(&mut p, fill("E2", 4.0, 100.0));
        assert_eq!(p.filled_qty, 3.0);
        assert_eq!(p.avg_fill_price(), Some(50.0));
    }

    #[test]
    fn in_process_orders_cannot_switch_to_reports() {
        let mut o = order(10.0);
        o.status = Filled;
        assert!(matches!(
            o.apply_report(&acc()).unwrap_err(),
            OrderEventError::NotVenueDriven { .. }
        ));
        let mut p = order(10.0);
        p.filled_qty = 1.0;
        p.status = PartiallyFilled;
        assert!(matches!(
            p.apply_report(&acc()).unwrap_err(),
            OrderEventError::NotVenueDriven { .. }
        ));
    }

    #[test]
    fn non_unit_quantity_is_refused() {
        let mut o = order(10.0);
        o.qty = QtySpec::FullPosition;
        assert!(matches!(
            o.apply_report(&acc()).unwrap_err(),
            OrderEventError::UnresolvedQty { .. }
        ));
    }

    #[test]
    fn request_on_terminal_is_illegal() {
        let mut o = order(1.0);
        ok(&mut o, acc());
        ok(&mut o, fill("E", 1.0, 1.0));
        assert!(matches!(
            o.request_cancel().unwrap_err(),
            OrderEventError::IllegalTransition { .. }
        ));
        assert!(matches!(
            o.request_replace().unwrap_err(),
            OrderEventError::IllegalTransition { .. }
        ));
    }

    mod convergence {
        use super::*;
        use proptest::prelude::*;

        #[derive(Debug, Clone, Copy)]
        enum End {
            Open,
            Cancel,
            Expire,
        }

        /// Everything the venue says about one order, in a valid order.
        /// Fill sizes sum to at most the order quantity, and corrections
        /// only shrink a fill, so no ordering can over-fill transiently.
        fn scenario() -> impl Strategy<Value = (f64, Vec<ExecutionReport>, bool)> {
            (
                prop::collection::vec((1u32..50, 1u32..1000, any::<u8>()), 1..6),
                0u32..30,
                prop_oneof![Just(End::Open), Just(End::Cancel), Just(End::Expire)],
                any::<bool>(),
            )
                .prop_map(|(fills, extra, end, ask_cancel)| {
                    let mut qty = 0.0;
                    let mut reports =
                        vec![ExecutionReport::Accepted { venue_order_id: "V".into(), ts_ns: 1 }];
                    for (i, (q, p, flag)) in fills.iter().enumerate() {
                        let id = format!("E{i}");
                        let q = f64::from(*q) * 0.1;
                        let p = f64::from(*p) * 0.05;
                        qty += q;
                        reports.push(ExecutionReport::Fill {
                            exec_id: id.clone(),
                            qty: q,
                            price: p,
                            fee: 0.1 * f64::from(i as u32 + 1),
                            ts_ns: 10 + i as i64,
                        });
                        match flag % 4 {
                            0 => reports.push(ExecutionReport::Correction {
                                exec_id: id,
                                qty: q * 0.5,
                                price: p + 0.1,
                                ts_ns: 100 + i as i64,
                            }),
                            1 => reports
                                .push(ExecutionReport::Bust { exec_id: id, ts_ns: 200 + i as i64 }),
                            _ => {}
                        }
                    }
                    match end {
                        End::Open => {}
                        End::Cancel => reports.push(ExecutionReport::CancelAck { ts_ns: 300 }),
                        End::Expire => reports.push(ExecutionReport::Expired { ts_ns: 300 }),
                    }
                    (qty + f64::from(extra) * 0.1, reports, ask_cancel)
                })
        }

        fn run(qty: f64, ask_cancel: bool, seq: &[ExecutionReport]) -> Order {
            let mut o = order(qty);
            if ask_cancel {
                o.request_cancel().unwrap();
            }
            for r in seq {
                o.apply_report(r).unwrap_or_else(|e| panic!("{e}: {r:?} in {seq:?}"));
            }
            o
        }

        fn same(a: &Order, b: &Order) {
            assert_eq!(a.status, b.status);
            assert_eq!(a.filled_qty.to_bits(), b.filled_qty.to_bits());
            assert_eq!(a.avg_fill_price().map(f64::to_bits), b.avg_fill_price().map(f64::to_bits));
            assert_eq!(a.fees_paid().to_bits(), b.fees_paid().to_bits());
            assert_eq!(a.venue_order_id(), b.venue_order_id());
        }

        proptest! {
            #[test]
            fn any_permutation_with_duplicates_converges(
                (qty, reports, ask_cancel) in scenario(),
                dup_picks in prop::collection::vec(any::<prop::sample::Index>(), 0..6),
                seed in any::<u64>(),
            ) {
                let in_order = run(qty, ask_cancel, &reports);
                let mut noisy = reports.clone();
                for pick in &dup_picks {
                    noisy.push(reports[pick.index(reports.len())].clone());
                }
                // Deterministic shuffle from the seed (Fisher-Yates, LCG).
                let mut x = seed | 1;
                for i in (1..noisy.len()).rev() {
                    x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    noisy.swap(i, (x >> 33) as usize % (i + 1));
                }
                let shuffled = run(qty, ask_cancel, &noisy);
                same(&in_order, &shuffled);
            }
        }
    }
}
