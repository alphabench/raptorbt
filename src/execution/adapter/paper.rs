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
//! own (a configured stop, a session square-off, a margin liquidation) has no
//! order to report against, so it is refused up front rather than hidden:
//! see [`PaperError::Unsolicited`].

use std::collections::HashMap;

use super::{AdapterError, ExecutionAdapter, OrderChange, OrderRequest};
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
         run move only on orders: remove position-level stops/targets and the session square-off, \
         or send an exit order"
    )]
    Unsolicited(String),
}

/// See the module docs.
pub struct PaperVenue {
    shadows: Vec<Option<Shadow>>,
    placed: HashMap<String, Placed>,
    /// Fills reported so far per order, for execution ids.
    fills: HashMap<String, u32>,
    /// Answers to requests that have no market event of their own (a refused
    /// cancel, an acknowledged change), delivered with the next event.
    queued: Vec<Report>,
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
            queued: Vec::new(),
            now_ns: 0,
        }
    }

    /// Halt a strategy's shadow book: working orders are cancelled (the
    /// cancels leave with the next event) and new entries are refused.
    pub fn halt(&mut self, strategy: usize, reason: &str) {
        if let Some(Some(shadow)) = self.shadows.get_mut(strategy) {
            shadow.book.halt(reason, false);
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
        let kernel = shadow.book.kernel(entry.instrument);
        let ts = entry.timestamp();
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
                            kernel.venue_fee(*price, *size, *direction, true)
                        }
                        Some(EngineEvent::Exited { trade, .. }) => {
                            kernel.venue_fee(trade.exit_price, trade.size, trade.direction, false)
                        }
                        _ => 0.0,
                    };
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
                    out.push((client_id.clone(), ExecutionReport::CancelAck { ts_ns: ts }));
                }
                EngineEvent::OrderExpired { client_id, .. } => {
                    out.push((client_id.clone(), ExecutionReport::Expired { ts_ns: ts }));
                }
                EngineEvent::OrderRejected { client_id, reason, .. } => {
                    // The in-process matcher can refuse a later slice of an
                    // order that has already traded. An exchange does not
                    // reject an order it has partly filled: it stops working
                    // the rest, which is a cancel.
                    let report = if self.fills.contains_key(client_id) {
                        ExecutionReport::CancelAck { ts_ns: ts }
                    } else {
                        ExecutionReport::Rejected { reason: (*reason).to_string(), ts_ns: ts }
                    };
                    out.push((client_id.clone(), report));
                }
                // A stop-limit arming is internal to the venue.
                EngineEvent::OrderTriggered { .. } => {}
                // A refused entry or a risk event with an order behind it
                // arrives as that order's rejection; the rest is the book's
                // own bookkeeping.
                EngineEvent::EntryRejected { .. }
                | EngineEvent::RiskRejected { .. }
                | EngineEvent::RiskHalted { .. } => {}
                // The strategy's own book raises its margin call from the
                // same equity; the shadow's only halts its entries.
                EngineEvent::MarginCall { .. } => {}
                other @ (EngineEvent::Entered { .. }
                | EngineEvent::Exited { .. }
                | EngineEvent::AlgoStarted { .. }
                | EngineEvent::AlgoCompleted { .. }) => {
                    return Err(PaperError::Unsolicited(format!("{other:?}")));
                }
            }
            i += 1;
        }
        Ok(())
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
            QtySpec::Units(req.quantity),
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
        self.now_ns = self.now_ns.max(req.submitted_ns);
        self.placed.insert(req.client_order_id.clone(), Placed { strategy, instrument, order_id });
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
            && kernel.modify_order(p.order_id, qty, change.limit_price, change.trigger_price);
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
