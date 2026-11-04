//! A simulated exchange that answers with execution reports.
//!
//! [`PaperVenue`] is what a paper-trading node sends its orders to. It is not
//! a second, simpler matcher: for each strategy it holds a shadow copy of the
//! strategy's book (same configuration, same account, same instruments), and
//! steps it with the very market events the strategy sees. Its orders are
//! matched, priced, charged and sized by the same kernel a backtest runs, so
//! the same stream of data and the same decisions give the same fills.
//!
//! What differs from a backtest is the route, not the arithmetic. A backtest
//! applies a fill to the strategy's own book as the matcher produces it.
//! Here the shadow book's fills leave as [`ExecutionReport`]s, exactly as a
//! real venue's would, and the strategy's book learns of them through the
//! same path (`Order::apply_report`, then booking): the code a live venue
//! exercises is the code paper trading exercises.
//!
//! The venue speaks only in reports. A position the shadow book closes on its
//! own (a configured stop, a margin liquidation) has no order to report
//! against, so it is refused up front rather than hidden: see
//! [`PaperError::Unsolicited`]. The session square-off is sent as closing
//! orders by the runtime, and a contract's expiry, settlement or delivery is
//! the instrument's declared terms, applied alike by the strategy's book and
//! the shadow's copy.

use std::collections::HashMap;

use super::{AdapterError, ExecutionAdapter, OrderChange, OrderRequest};
use crate::core::types::ExitReason;
use crate::data::DepthTick;
use crate::execution::orders::{ExecutionReport, OrderStatus, QtySpec};
use crate::portfolio::kernel::{EngineEvent, StepInput};
use crate::portfolio::session::ScheduleEntry;
use crate::portfolio::PortfolioBook;

/// One report about one order, addressed by its venue-facing id.
pub type Report = (String, ExecutionReport);

/// The shadow book of one strategy.
struct Shadow {
    book: PortfolioBook,
    /// Symbol of each local instrument, in the strategy's order.
    symbols: Vec<String>,
}

/// Where a working order lives in the shadow books.
#[derive(Debug, Clone, Copy)]
struct Placed {
    strategy: usize,
    instrument: usize,
    order_id: u64,
}

/// Why the venue could not step.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum PaperError {
    /// The shadow book did something no order asked for, so there is no
    /// report to say it with. Nothing is hidden: the run stops.
    #[error(
        "the paper venue changed a position without an order ({0}); positions in a venue-driven \
         run move only on orders: remove position-level stops/targets, or send an exit order"
    )]
    Unsolicited(String),
    /// A fill the shadow book made could not be priced for its report.
    #[error("the paper venue could not report a fill: {0}")]
    Fill(String),
}

/// How many ended orders the venue keeps entries for.
pub const RETAINED_ENDED: usize = 1024;

/// See the module docs.
pub struct PaperVenue {
    shadows: Vec<Option<Shadow>>,
    placed: HashMap<String, Placed>,
    /// Fills reported so far per order, for execution ids.
    fills: HashMap<String, u32>,
    /// The quantity the strategy's book holds each order that the venue sizes
    /// at its fill.
    quantity: HashMap<String, f64>,
    /// Orders that ended, oldest first. Their entries in the maps above are
    /// kept for [`RETAINED_ENDED`] orders (an id sent again, or a late
    /// request for it, still finds the order) and dropped after that, so a
    /// long run holds a bounded number of orders.
    ended: std::collections::VecDeque<String>,
    /// Answers to requests that have no market event of their own (a refused
    /// cancel, an acknowledged change), delivered with the next event.
    queued: Vec<Report>,
    /// Reports for protective legs that met the rest of the bar their
    /// position was entered on: they belong to the event being processed.
    same_event: Vec<Report>,
    deciding: bool,
    /// Time of the latest event, for the answers above.
    now_ns: i64,
}

impl PaperVenue {
    /// A venue over shadow books, one per strategy, indexed by strategy id.
    /// `symbols[k]` names the instruments of strategy `k`'s book in order.
    pub fn new(books: Vec<(Vec<String>, PortfolioBook)>) -> Self {
        Self {
            shadows: books
                .into_iter()
                .map(|(symbols, mut book)| {
                    for i in 0..symbols.len() {
                        book.kernel_mut(i).set_venue_managed();
                    }
                    Some(Shadow { book, symbols })
                })
                .collect(),
            placed: HashMap::new(),
            fills: HashMap::new(),
            quantity: HashMap::new(),
            ended: std::collections::VecDeque::new(),
            queued: Vec::new(),
            same_event: Vec::new(),
            deciding: false,
            now_ns: 0,
        }
    }

    /// Give strategy `strategy`'s shadow book one more instrument, as its newest local
    /// instrument, configured as the strategy's own book configures it. The venue then steps it
    /// and takes orders on it. Used when an instrument is added to a running node.
    pub fn add_instrument(
        &mut self,
        strategy: usize,
        symbol: String,
        direction: crate::core::types::Direction,
        spec: Option<crate::instruments::InstrumentSpec>,
        config: Option<&crate::core::types::InstrumentConfig>,
        policy: crate::portfolio::ledger::PositionPolicy,
    ) -> Result<(), AdapterError> {
        let shadow = self.shadow(strategy)?;
        let local = shadow.book.add_instrument(symbol.clone(), direction, spec, config, policy);
        shadow.book.kernel_mut(local).set_venue_managed();
        shadow.symbols.push(symbol);
        debug_assert_eq!(shadow.symbols.len(), local + 1);
        Ok(())
    }

    /// Halt a strategy's shadow book: working orders are cancelled (the
    /// cancels leave with the next event) and new entries are refused. With
    /// `flatten` the shadow closes its positions at its last marks, as the
    /// strategy's own book does; the venue reports nothing for that, because
    /// the strategy's book closes the same positions itself.
    pub fn halt(&mut self, strategy: usize, reason: &str, flatten: bool) {
        if let Some(Some(shadow)) = self.shadows.get_mut(strategy) {
            shadow.book.halt(reason, flatten);
        }
    }

    fn shadow(&mut self, strategy: usize) -> Result<&mut Shadow, AdapterError> {
        self.shadows
            .get_mut(strategy)
            .and_then(Option::as_mut)
            .ok_or_else(|| AdapterError::Refused(format!("no book for strategy {strategy}")))
    }

    fn place_of(&self, id: &str) -> Result<Placed, AdapterError> {
        self.placed
            .get(id)
            .copied()
            .ok_or_else(|| AdapterError::Refused(format!("unknown order {id}")))
    }

    /// Step strategy `strategy`'s shadow book with the event it just saw and
    /// collect what the venue reports about it, answers to earlier requests
    /// first. `atr` is the strategy's ATR for the event, as in a backtest.
    pub fn on_market_event(
        &mut self,
        strategy: usize,
        entry: &ScheduleEntry,
        depth: Option<DepthTick>,
        atr: f64,
        out: &mut Vec<Report>,
    ) -> Result<(), PaperError> {
        self.now_ns = entry.timestamp();
        out.append(&mut self.queued);
        let Some(shadow) = self.shadows.get_mut(strategy).and_then(Option::as_mut) else {
            return Ok(());
        };
        let events = shadow.book.apply(entry, depth, StepInput { atr, ..StepInput::default() });
        // The strategy's own book places protection, schedule slices and the
        // square-off; the shadow only needs to not act on them itself.
        let managed = shadow.book.kernel_mut(entry.instrument);
        managed.take_protect_requests();
        managed.take_slices();
        managed.take_squareoff();
        self.report(strategy, entry.instrument, &events, entry.timestamp(), out)
    }

    /// Under `next_bar_open`, step strategy `strategy`'s shadow book to the open of the bar it is
    /// about to see: what was decided on earlier bars fills at this open, and the venue reports
    /// it before the strategy has seen the bar. [`PaperVenue::on_market_event`] carries on from
    /// here. Reports nothing when the bar cannot be opened ahead (another fill timing, a halt
    /// waiting), which `on_market_event` then does whole.
    pub fn on_open(
        &mut self,
        strategy: usize,
        entry: &ScheduleEntry,
        out: &mut Vec<Report>,
    ) -> Result<(), PaperError> {
        let Some(shadow) = self.shadows.get_mut(strategy).and_then(Option::as_mut) else {
            return Ok(());
        };
        if !shadow.book.can_open(entry) {
            return Ok(());
        }
        self.now_ns = entry.timestamp();
        out.append(&mut self.queued);
        let mut events = shadow.book.begin_event(entry.timestamp(), entry.instrument);
        events.extend(shadow.book.open_bar(entry));
        self.report(strategy, entry.instrument, &events, entry.timestamp(), out)
    }

    /// Step strategy `strategy`'s shadow book through what rests for the bar it is about to
    /// decide on: its stops, targets and resting orders trade the bar's range while the bar
    /// forms, and the venue reports it before the strategy decides. [`PaperVenue::on_market_event`]
    /// carries on from here with the decisions of the bar. Reports nothing when nothing rests or
    /// the bar cannot be met ahead (a halt waiting), which `on_market_event` then does whole.
    pub fn on_rest(
        &mut self,
        strategy: usize,
        entry: &ScheduleEntry,
        out: &mut Vec<Report>,
    ) -> Result<(), PaperError> {
        let Some(shadow) = self.shadows.get_mut(strategy).and_then(Option::as_mut) else {
            return Ok(());
        };
        if !shadow.book.can_rest(entry) {
            return Ok(());
        }
        self.now_ns = entry.timestamp();
        out.append(&mut self.queued);
        let mut events = shadow.book.begin_event(entry.timestamp(), entry.instrument);
        events.extend(shadow.book.rest_bar(entry));
        // As in `on_market_event`: the strategy's own book places protection, schedule slices and
        // the square-off; the shadow only needs to not act on them itself.
        let managed = shadow.book.kernel_mut(entry.instrument);
        managed.take_protect_requests();
        managed.take_slices();
        managed.take_squareoff();
        self.report(strategy, entry.instrument, &events, entry.timestamp(), out)
    }

    /// Turn what the shadow book did into the venue's reports.
    fn report(
        &mut self,
        strategy: usize,
        instrument: usize,
        events: &[EngineEvent],
        ts: i64,
        out: &mut Vec<Report>,
    ) -> Result<(), PaperError> {
        let Some(shadow) = self.shadows.get_mut(strategy).and_then(Option::as_mut) else {
            return Ok(());
        };
        let kernel = shadow.book.kernel(instrument);
        let mut ended_now: Vec<String> = Vec::new();
        let delivery_in_batch = events
            .iter()
            .any(|e| matches!(e, EngineEvent::Settled { deliver, .. } if *deliver != 0.0));
        let mut i = 0;
        while i < events.len() {
            match &events[i] {
                EngineEvent::OrderAccepted { order_id, client_id, .. } => {
                    out.push((
                        client_id.clone(),
                        ExecutionReport::Accepted {
                            venue_order_id: format!("P{order_id}"),
                            ts_ns: ts,
                        },
                    ));
                }
                EngineEvent::OrderFilled { order_id, client_id, price, size, .. } => {
                    // The position event right behind a fill says which way
                    // it traded and so what it cost.
                    let fee = match events.get(i + 1) {
                        Some(EngineEvent::Entered { direction, .. }) => {
                            kernel.venue_fee(*order_id, *price, *size, *direction, true)
                        }
                        Some(EngineEvent::Exited { trade, .. }) => kernel.venue_fee(
                            *order_id,
                            trade.exit_price,
                            trade.size,
                            trade.direction,
                            false,
                        ),
                        // The matcher follows every fill with its position
                        // event; one without cannot be priced.
                        other => {
                            return Err(PaperError::Fill(format!(
                                "the fill of order {order_id} is followed by {other:?}"
                            )))
                        }
                    }
                    .map_err(|e| PaperError::Fill(e.to_string()))?;
                    // An order sized by the venue at the fill: amend the
                    // strategy's quantity to what traded.
                    if let Some(held) = self.quantity.get_mut(client_id) {
                        if (*held - *size).abs() > 1e-9 * held.abs().max(1.0) {
                            *held = *size;
                            out.push((
                                client_id.clone(),
                                ExecutionReport::ReplaceAck {
                                    new_qty: Some(*size),
                                    new_limit: None,
                                    new_trigger: None,
                                    ts_ns: ts,
                                },
                            ));
                        }
                    }
                    let n = self.fills.entry(client_id.clone()).or_insert(0);
                    let exec_id = format!("{client_id}.{n}");
                    *n += 1;
                    out.push((
                        client_id.clone(),
                        ExecutionReport::Fill {
                            exec_id,
                            qty: *size,
                            price: *price,
                            fee,
                            ts_ns: ts,
                        },
                    ));
                    // The matcher ends an exit larger than the position once
                    // the position is closed, and cancels the surplus. A
                    // venue says so: the fill is followed by the cancel.
                    if let Some(order) = kernel.order(*order_id) {
                        if order.status.is_terminal() {
                            ended_now.push(client_id.clone());
                        }
                        if let QtySpec::Units(units) = order.qty {
                            if order.status == OrderStatus::Filled
                                && order.filled_qty + units * 1e-9 < units
                            {
                                out.push((
                                    client_id.clone(),
                                    ExecutionReport::CancelAck { ts_ns: ts },
                                ));
                            }
                        }
                    }
                    // Its position event is accounted for, not unsolicited.
                    i += 1;
                }
                EngineEvent::OrderCanceled { client_id, .. } => {
                    ended_now.push(client_id.clone());
                    out.push((client_id.clone(), ExecutionReport::CancelAck { ts_ns: ts }));
                }
                EngineEvent::OrderExpired { client_id, .. } => {
                    ended_now.push(client_id.clone());
                    out.push((client_id.clone(), ExecutionReport::Expired { ts_ns: ts }));
                }
                EngineEvent::OrderRejected { client_id, reason, .. } => {
                    // The in-process matcher can refuse a later slice of an
                    // order that has already traded. An exchange does not
                    // reject an order it has partly filled: it stops working
                    // the rest, which is a cancel.
                    ended_now.push(client_id.clone());
                    let report = if self.fills.contains_key(client_id) {
                        ExecutionReport::CancelAck { ts_ns: ts }
                    } else {
                        ExecutionReport::Rejected { reason: (*reason).to_string(), ts_ns: ts }
                    };
                    out.push((client_id.clone(), report));
                }
                // A stop-limit arming is internal to the venue.
                EngineEvent::OrderTriggered { .. } => {}
                // The venue's own cut of a one-cancels-other sibling: the
                // sibling's later reports carry the smaller quantity.
                EngineEvent::OrderResized { .. } => {}
                // A refused entry or a risk event with an order behind it
                // arrives as that order's rejection; the rest is the book's
                // own bookkeeping.
                EngineEvent::EntryRejected { .. }
                | EngineEvent::RiskRejected { .. }
                | EngineEvent::FillBreach { .. }
                | EngineEvent::FillDeferred { .. }
                | EngineEvent::RiskHalted { .. } => {}
                // The strategy's own book raises its margin call from the
                // same equity; the shadow's only halts its entries.
                EngineEvent::MarginCall { .. } => {}
                // Expiry, settlement and corporate actions are applied by the
                // strategy's own book from the same declared events; the
                // shadow's copy keeps the venue's position in step and sends
                // nothing.
                EngineEvent::Settled { .. }
                | EngineEvent::CorporateAction { .. }
                | EngineEvent::LifecycleRefused { .. } => {}
                // `Liquidation` reaches the shadow only from a flattening halt (`halt` above), which
                // the strategy's own book receives in the same call and closes at the same marks;
                // a margin-call liquidation is refused with a venue up front
                // (`liquidate_on_margin_call`, `test_what_a_venue_cannot_follow_is_refused_up_front`),
                // and `test_a_flattening_halt_books_the_same_trade_on_paper_and_simulated` pins
                // that the two books agree. So the exit is dropped unchecked because it cannot differ.
                // A contract's end is the exchange's act, not an order's: the
                // position settles (or is delivered against, or has its odd
                // fraction of a split paid out) on the instrument's declared
                // terms. The strategy's book reaches the same end from the
                // same terms and the same events, so the shadow keeps its
                // copy in step and reports nothing. A delivery that opens a
                // line in the underlying is the `Entered` of a batch that
                // also carries the contract's `Settled`.
                EngineEvent::Exited { trade, .. }
                    if matches!(
                        trade.exit_reason,
                        ExitReason::Settlement
                            | ExitReason::Delivery
                            | ExitReason::CorporateAction
                            | ExitReason::Liquidation
                    ) => {}
                EngineEvent::Entered { .. } if delivery_in_batch => {}
                other @ (EngineEvent::Entered { .. }
                | EngineEvent::Exited { .. }
                | EngineEvent::AlgoStarted { .. }
                | EngineEvent::AlgoCompleted { .. }) => {
                    return Err(PaperError::Unsolicited(format!("{other:?}")));
                }
            }
            i += 1;
        }
        for id in ended_now {
            self.retire(id);
        }
        Ok(())
    }

    /// The reports protective legs produced on the bar they were sent on,
    /// to be delivered with the event that is being processed.
    /// Whether the orders sent next are what the strategy's handlers decided on the current event.
    pub fn set_deciding(&mut self, deciding: bool) {
        self.deciding = deciding;
    }

    pub fn take_same_event(&mut self) -> Vec<Report> {
        std::mem::take(&mut self.same_event)
    }

    /// Note that an order ended; drop the entries of the oldest ended orders
    /// beyond [`RETAINED_ENDED`].
    fn retire(&mut self, client_order_id: String) {
        if self.ended.contains(&client_order_id) {
            return;
        }
        self.ended.push_back(client_order_id);
        while self.ended.len() > RETAINED_ENDED {
            if let Some(old) = self.ended.pop_front() {
                self.placed.remove(&old);
                self.fills.remove(&old);
                self.quantity.remove(&old);
            }
        }
    }

    /// The entries held for orders, for tests of the bound.
    #[cfg(test)]
    fn held(&self) -> usize {
        self.placed.len().max(self.fills.len()).max(self.quantity.len())
    }
}

impl ExecutionAdapter for PaperVenue {
    fn submit(&mut self, req: &OrderRequest) -> Result<(), AdapterError> {
        if self.placed.contains_key(&req.client_order_id) {
            // The same id again is the same order, not a second one.
            return Ok(());
        }
        let strategy = req.strategy_id as usize;
        let parent = match &req.parent_client_order_id {
            Some(parent) => Some(self.place_of(parent)?.order_id),
            None => None,
        };
        let shadow = self.shadow(strategy)?;
        let Some(instrument) = shadow.symbols.iter().position(|s| *s == req.symbol) else {
            return Err(AdapterError::Refused(format!(
                "strategy {strategy} does not trade {}",
                req.symbol
            )));
        };
        let order_id = shadow.book.kernel_mut(instrument).submit_order_full(
            req.side,
            req.size_frac.map_or(QtySpec::Units(req.quantity), QtySpec::CapitalFrac),
            req.kind,
            req.tif,
            req.submitted_idx,
            req.submitted_ns,
            req.client_order_id.clone(),
            None,
            None,
            req.post_only,
            req.reduce_only,
            parent,
        );
        if req.position_id.is_some() {
            shadow.book.kernel_mut(instrument).set_order_position(order_id, req.position_id);
        }
        if req.tags.get("client_id").is_some_and(|c| c.starts_with("enter-")) {
            shadow.book.kernel_mut(instrument).set_order_entry_intent(order_id);
        }
        self.now_ns = self.now_ns.max(req.submitted_ns);
        if req.size_frac.is_some() {
            self.quantity.insert(req.client_order_id.clone(), req.quantity);
        }
        self.placed.insert(req.client_order_id.clone(), Placed { strategy, instrument, order_id });
        // A protective leg sent in answer to an entry's fill meets the rest of
        // the bar the entry filled in (see `EngineKernel::match_arrived`); its
        // reports leave with the next event.
        let protective = req.tags.get("client_id").is_some_and(|c| c.starts_with("prot-"));
        // An order sent in answer to a fill the strategy just heard meets the bar that fill came
        // in (see `Order::follow_up`); its reports leave with the step.
        // An order for an instrument whose bar of that instant is already stepped (decided on
        // another instrument's event of the same timestamp) meets that bar, as in a batch run.
        let missed = self.deciding
            && self.shadow(strategy)?.book.kernel(instrument).stepped_bar_at(req.submitted_ns);
        if (req.tags.contains_key("after_fill") || missed)
            && matches!(req.kind, crate::execution::orders::OrderKind::Market)
        {
            let shadow = self.shadow(strategy)?;
            shadow.book.kernel_mut(instrument).set_order_follow_up(order_id);
            let events = shadow.book.fill_follow_up(instrument, order_id, req.submitted_ns);
            let mut reports = Vec::new();
            self.report(strategy, instrument, &events, req.submitted_ns, &mut reports)
                .map_err(|e| AdapterError::Refused(e.to_string()))?;
            self.same_event.append(&mut reports);
        }
        if req.reduce_only && protective {
            let kernel = self.shadow(strategy)?.book.kernel_mut(instrument);
            let events = kernel.match_arrived(order_id, req.submitted_ns);
            let mut reports = Vec::new();
            self.report(strategy, instrument, &events, req.submitted_ns, &mut reports)
                .map_err(|e| AdapterError::Refused(e.to_string()))?;
            self.same_event.append(&mut reports);
        }
        Ok(())
    }

    fn cancel(&mut self, client_order_id: &str) -> Result<(), AdapterError> {
        let p = self.place_of(client_order_id)?;
        let now = self.now_ns;
        let shadow = self.shadow(p.strategy)?;
        let kernel = shadow.book.kernel_mut(p.instrument);
        // The acknowledgement is the order-cancelled event the kernel queues,
        // which leaves with the next event.
        if !kernel.cancel_order(0, p.order_id) {
            self.queued.push((
                client_order_id.to_string(),
                ExecutionReport::CancelRejected { reason: "not working".into(), ts_ns: now },
            ));
        }
        Ok(())
    }

    fn modify(&mut self, client_order_id: &str, change: &OrderChange) -> Result<(), AdapterError> {
        let p = self.place_of(client_order_id)?;
        let now = self.now_ns;
        let shadow = self.shadow(p.strategy)?;
        let kernel = shadow.book.kernel_mut(p.instrument);
        let qty = change.quantity.map(QtySpec::Units);
        let applied = kernel
            .order(p.order_id)
            .is_some_and(|o| !o.status.is_terminal() && o.status != OrderStatus::Submitted)
            && kernel.modify_order_at(
                p.order_id,
                qty,
                change.limit_price,
                change.trigger_price,
                change.decided_close,
            );
        let report = if applied {
            ExecutionReport::ReplaceAck {
                new_qty: change.quantity,
                new_limit: change.limit_price,
                new_trigger: change.trigger_price,
                ts_ns: now,
            }
        } else {
            ExecutionReport::ReplaceRejected { reason: "not modifiable".into(), ts_ns: now }
        };
        self.queued.push((client_order_id.to_string(), report));
        Ok(())
    }

    fn link_oco(&mut self, ids: &[String]) -> Result<(), AdapterError> {
        let mut placed = ids.iter().map(|id| self.place_of(id)).collect::<Result<Vec<_>, _>>()?;
        placed.dedup_by_key(|p| (p.strategy, p.instrument));
        if placed.len() > 1 {
            return Err(AdapterError::Refused(
                "one-cancels-other links cannot span instruments".into(),
            ));
        }
        let Some(first) = placed.first().copied() else { return Ok(()) };
        let order_ids: Vec<u64> =
            ids.iter().map(|id| self.place_of(id).map(|p| p.order_id)).collect::<Result<_, _>>()?;
        self.shadow(first.strategy)?.book.kernel_mut(first.instrument).link_oco(&order_ids);
        Ok(())
    }

    fn as_simulation(&mut self) -> Option<&mut PaperVenue> {
        Some(self)
    }
}

#[cfg(test)]
#[path = "paper_tests.rs"]
mod tests;
