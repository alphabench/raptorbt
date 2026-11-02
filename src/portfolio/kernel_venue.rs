//! Orders that live at a venue: creation, requests, and booking what the
//! venue reports.
//!
//! The same `impl EngineKernel` as the in-process order path, in its own file
//! so the matcher's arithmetic stays where it is. A venue-driven order is
//! created venue-driven (the matcher never sees it), moves only on
//! [`ExecutionReport`]s, and its fills change the position and the cash
//! through the same `open_at` / `add_at` / `reduce_at` the matcher uses, at
//! the price and quantity the venue reported. The venue's price is final: no
//! slippage is applied on top of it.

use std::sync::Mutex;

use crate::core::types::{Direction, ExitReason};
use crate::execution::orders::{
    ExecutionReport, Order, OrderEventError, OrderKind, OrderSide, OrderStatus, QtySpec,
    ReportOutcome, TimeInForce,
};
use crate::portfolio::kernel::{EngineEvent, EngineKernel, KernelBar};
use crate::portfolio::ledger::PositionPolicy;
use crate::portfolio::risk::RejectReason;

/// Why a venue request or report was not applied.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum VenueError {
    #[error(transparent)]
    Order(#[from] OrderEventError),
    #[error("order {0} is not an order of this book")]
    UnknownOrder(u64),
    /// The report is valid for the order but cannot be applied to a position.
    #[error("{0}")]
    Unsupported(&'static str),
    /// The order took the fill, the book could not: the book and the venue
    /// now disagree about the position and need reconciling.
    #[error("the venue filled order {order_id} but the book refused the fill ({reason})")]
    FillNotBooked { order_id: u64, reason: String },
}

/// A refusal reason from a venue, as the `&'static str` the events carry.
///
/// Reasons are interned: the set a venue uses is small, and the table is
/// capped so a venue that puts an id in its reason cannot grow it without
/// bound. Past the cap a reason reads `venue_rejected`.
fn intern_reason(reason: &str) -> &'static str {
    const CAP: usize = 64;
    static TABLE: Mutex<Vec<&'static str>> = Mutex::new(Vec::new());
    let mut table = TABLE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(known) = table.iter().find(|r| **r == reason) {
        return known;
    }
    if table.len() >= CAP || reason.is_empty() {
        return "venue_rejected";
    }
    let leaked: &'static str = Box::leak(reason.to_string().into_boxed_str());
    table.push(leaked);
    leaked
}

impl EngineKernel {
    /// Create an order that a venue will work. It starts `Submitted` and
    /// becomes `Accepted` (or `Rejected`) on the venue's report; nothing is
    /// emitted here. The market-closed check of the in-process path is the
    /// venue's to make.
    #[allow(clippy::too_many_arguments)]
    pub fn submit_venue_order(
        &mut self,
        side: OrderSide,
        units: f64,
        kind: OrderKind,
        tif: TimeInForce,
        submitted_idx: usize,
        submitted_ts: i64,
        client_id: String,
        post_only: bool,
        reduce_only: bool,
        parent_id: Option<u64>,
        stop_price: Option<f64>,
        target_price: Option<f64>,
    ) -> u64 {
        let mut order = Order::plain(side, QtySpec::Units(units), kind, tif).into_venue_driven();
        order.stop_price = stop_price;
        order.target_price = target_price;
        order.client_id = client_id;
        order.submitted_idx = submitted_idx;
        order.submitted_ts = submitted_ts;
        order.post_only = post_only;
        order.reduce_only = reduce_only;
        order.parent_id = parent_id;
        self.orders.submit(order)
    }

    /// The book's own limits, applied before an order is sent to a venue
    /// that does not apply them for us: the kill-switch and halts, the
    /// drawdown and daily-loss latches, `max_positions` and the order-size
    /// and position limits. A simulated venue applies them itself, at the
    /// fill, exactly as the matcher does, so it skips this.
    ///
    /// An order that can only shrink a position is always allowed, as the
    /// matcher allows exits when entries are blocked. `price` is the price to
    /// size the order's notional at, when the caller has one.
    pub fn venue_pretrade(
        &self,
        side: OrderSide,
        units: f64,
        reduce_only: bool,
        price: Option<f64>,
    ) -> Result<(), RejectReason> {
        let direction = match side {
            OrderSide::Buy => Direction::Long,
            OrderSide::Sell => Direction::Short,
        };
        let hedging = self.ledger.policy() == PositionPolicy::Independent;
        let held = self.ledger.first().map(|m| m.position.direction);
        let opens = hedging || !matches!(held, Some(dir) if dir != direction);
        if reduce_only || !opens {
            return Ok(());
        }
        if self.margin.is_halted() {
            return Err(RejectReason::MarginCall);
        }
        self.risk.check_entry(self.gating_open_count())?;
        match price {
            Some(p) => self.check_whole_order(direction, units, p),
            None => Ok(()),
        }
    }

    /// Hand the positions' protection, schedule slices and square-off to the
    /// runtime: the kernel does not act on them in-process once it is
    /// venue-managed.
    pub fn set_venue_managed(&mut self) {
        self.venue_managed = true;
    }

    /// The protective levels of positions opened since last asked.
    pub fn take_protect_requests(&mut self) -> Vec<crate::portfolio::kernel::ProtectRequest> {
        std::mem::take(&mut self.protect_outbox)
    }

    /// Schedule slices released since last asked.
    pub fn take_slices(&mut self) -> Vec<crate::execution::algos::PendingSlice> {
        std::mem::take(&mut self.slice_outbox)
    }

    /// Whether the session square-off came due since last asked.
    pub fn take_squareoff(&mut self) -> bool {
        std::mem::take(&mut self.squareoff_outbox)
    }

    /// The units an order with no explicit size would trade, as the matcher
    /// would size it from the book as it stands on `bar`: the position, for
    /// an order that closes one; a fraction of the capital on hand, for one
    /// that opens. `ref_price` is the price the order is expected to trade at
    /// (its limit, or its trigger); a market order reads the fill-price model.
    pub fn venue_units(
        &self,
        bar: &KernelBar,
        side: OrderSide,
        frac: Option<f64>,
        ref_price: Option<f64>,
        limit_priced: bool,
    ) -> Result<f64, &'static str> {
        let direction = match side {
            OrderSide::Buy => Direction::Long,
            OrderSide::Sell => Direction::Short,
        };
        let hedging = self.ledger.policy() == PositionPolicy::Independent;
        let held = self.ledger.first().map(|m| (m.position.direction, m.position.size));
        if let (false, Some((dir, size))) = (hedging, held) {
            if dir != direction {
                return Ok(size);
            }
        }
        let Some(frac) = frac else { return Err("invalid_qty") };
        let price = ref_price.unwrap_or_else(|| self.fill_price_for(bar, direction, true));
        self.planned_entry_units(bar, direction, price, Some(frac), limit_priced)
    }

    /// The direction a signal entry opens in.
    pub fn signal_direction(&self) -> Direction {
        self.direction
    }

    /// Record that a cancel was asked of the venue. The order is cancelled
    /// only by the venue's acknowledgement.
    pub fn request_venue_cancel(&mut self, order_id: u64) -> Result<(), VenueError> {
        let order = self.orders.get_mut(order_id).ok_or(VenueError::UnknownOrder(order_id))?;
        Ok(order.request_cancel()?)
    }

    /// Record that a change was asked of the venue. The order keeps its terms
    /// until the venue acknowledges.
    pub fn request_venue_replace(&mut self, order_id: u64) -> Result<(), VenueError> {
        let order = self.orders.get_mut(order_id).ok_or(VenueError::UnknownOrder(order_id))?;
        Ok(order.request_replace()?)
    }

    /// The fee this book's fee model charges for trading `size` at `price`.
    /// `direction` is the position's: the long or short side being opened
    /// (`is_entry`) or closed.
    pub fn venue_fee(&self, price: f64, size: f64, direction: Direction, is_entry: bool) -> f64 {
        let fee_price = price * self.multiplier();
        let fee = match self.fee_model.breakdown(fee_price, size, direction, is_entry) {
            Some(b) => b.total(),
            None => self.fee_model.calculate(fee_price, size, direction),
        };
        // The amount the book would charge: rounded to its money scale.
        let scale = self.money_scale();
        scale.money(fee).map_or(fee, |m| scale.to_f64(m))
    }

    /// Fold one venue report into an order, and book what it did.
    ///
    /// Returns what the report did to the order and the events the strategy
    /// should hear, in the order the in-process path emits them (accepted;
    /// the fill followed at once by its position event; cancelled; expired;
    /// rejected). A duplicate or a report that no longer matters yields no
    /// events. A fill that settles a position is booked on the account the
    /// way the matcher books its own: cash, margin and the position ledger.
    ///
    /// Corrections and busts restate an execution after the position moved;
    /// undoing a booked fill is not supported yet and is refused before the
    /// order is touched.
    pub fn apply_venue_report(
        &mut self,
        idx: usize,
        ts: i64,
        order_id: u64,
        report: &ExecutionReport,
    ) -> Result<(ReportOutcome, Vec<EngineEvent>), VenueError> {
        if matches!(report, ExecutionReport::Correction { .. } | ExecutionReport::Bust { .. }) {
            return Err(VenueError::Unsupported(
                "a correction or bust cannot be applied to a position that already moved",
            ));
        }
        let order = self.orders.get_mut(order_id).ok_or(VenueError::UnknownOrder(order_id))?;
        let client_id = order.client_id.clone();
        let reduce_only = order.reduce_only;
        let before = order.status;
        let outcome = order.apply_report(report)?;
        let after = order.status;
        let mut events = Vec::new();
        if outcome == ReportOutcome::Duplicate {
            return Ok((outcome, events));
        }
        match report {
            ExecutionReport::Accepted { .. }
                if before == OrderStatus::Submitted && after != OrderStatus::Submitted =>
            {
                events.push(EngineEvent::OrderAccepted { idx, order_id, client_id });
            }
            ExecutionReport::Fill { qty, price, .. } => {
                events.extend(self.book_venue_fill(idx, ts, order_id, &client_id, *qty, *price)?);
            }
            ExecutionReport::CancelAck { .. }
                if after == OrderStatus::Canceled && before != OrderStatus::Canceled =>
            {
                events.push(EngineEvent::OrderCanceled { idx, order_id, client_id });
            }
            ExecutionReport::Expired { .. }
                if after == OrderStatus::Expired && before != OrderStatus::Expired =>
            {
                events.push(EngineEvent::OrderExpired { idx, order_id, client_id });
            }
            ExecutionReport::Rejected { reason, .. }
                if after == OrderStatus::Rejected && before != OrderStatus::Rejected =>
            {
                // A refusal counts against the book's rejected entries as the
                // matcher's do: all but a post-only cross, an unaffordable or
                // zero-sized entry, and a close that could not fill.
                let uncounted = matches!(
                    reason.as_str(),
                    "post_only" | "zero_size" | "insufficient_margin" | "insufficient_capital"
                ) || (reduce_only && reason == "unfillable")
                    // An entry intent that finds the position already open is
                    // a no-op in-process.
                    || (client_id.starts_with("enter-") && reason == "position_open");
                if !uncounted {
                    self.risk.record_rejection();
                }
                events.push(EngineEvent::OrderRejected {
                    idx,
                    order_id,
                    client_id,
                    reason: intern_reason(reason),
                });
            }
            // Everything else changes the order's state, not the position;
            // the strategy hooks have nothing to hear.
            _ => {}
        }
        Ok((outcome, events))
    }

    /// Book one execution on the position ledger and the account.
    fn book_venue_fill(
        &mut self,
        idx: usize,
        ts: i64,
        order_id: u64,
        client_id: &str,
        qty: f64,
        price: f64,
    ) -> Result<Vec<EngineEvent>, VenueError> {
        let not_booked =
            |reason: &str| VenueError::FillNotBooked { order_id, reason: reason.to_string() };
        let Some(order) = self.orders.get(order_id) else {
            return Err(VenueError::UnknownOrder(order_id));
        };
        let (side, reduce_only) = (order.side, order.reduce_only);
        let (stop_attach, target_attach) = (order.stop_price, order.target_price);
        let order_direction = match side {
            OrderSide::Buy => Direction::Long,
            OrderSide::Sell => Direction::Short,
        };
        // The venue's price is the last word, so the bar around it is the
        // print itself.
        let bar = KernelBar {
            timestamp: ts,
            open: price,
            high: price,
            low: price,
            close: price,
            volume: 0.0,
        };
        // The same open-or-close rule the matcher applies (see
        // `apply_match_outcome`): netting closes against the opposite side,
        // hedging always opens.
        let hedging = self.ledger.policy() == PositionPolicy::Independent;
        let held = self.ledger.first().map(|m| m.position.direction);
        let opens = hedging || !matches!(held, Some(dir) if dir != order_direction);
        let fill = |price: f64, size: f64| EngineEvent::OrderFilled {
            idx,
            order_id,
            client_id: client_id.to_string(),
            price,
            size,
        };

        if opens {
            if reduce_only {
                return Err(not_booked("a reduce-only order filled without a position to reduce"));
            }
            let adds_to =
                self.ledger.first().filter(|m| m.entry_order_id == Some(order_id)).map(|m| m.id);
            if !hedging && self.ledger.is_in_position() && adds_to.is_none() {
                return Err(not_booked("netting holds a position and this order did not open it"));
            }
            if let Some(position_id) = adds_to {
                let Some((fill_price, size)) = self.add_at(
                    idx,
                    &bar,
                    position_id,
                    price,
                    qty,
                    stop_attach,
                    target_attach,
                    true,
                ) else {
                    return Err(not_booked("the added quantity could not be funded"));
                };
                return Ok(vec![
                    fill(fill_price, size),
                    EngineEvent::Entered {
                        idx,
                        price: fill_price,
                        size,
                        direction: order_direction,
                    },
                ]);
            }
            match self.open_at(
                idx,
                &bar,
                order_direction,
                price,
                None,
                Some(qty),
                self.last_atr,
                stop_attach,
                target_attach,
                true,
            ) {
                Some(EngineEvent::Entered { price: fill_price, size, direction, .. }) => {
                    if let Some(opened) = self.ledger.first_mut() {
                        opened.entry_order_id = Some(order_id);
                    }
                    Ok(vec![
                        fill(fill_price, size),
                        EngineEvent::Entered { idx, price: fill_price, size, direction },
                    ])
                }
                Some(EngineEvent::EntryRejected { reason, .. }) => Err(not_booked(reason.as_str())),
                _ => Err(not_booked("the position could not be opened")),
            }
        } else {
            let Some(first) = self.ledger.first() else {
                return Err(not_booked("a closing fill with no position to close"));
            };
            let (position_id, held_size) = (first.id, first.position.size);
            let slice = qty.min(held_size);
            if slice <= 0.0 {
                return Err(not_booked("nothing is held to close"));
            }
            match self.reduce_at(idx, &bar, position_id, slice, price, ExitReason::Order, true) {
                Some(EngineEvent::Exited { trade, .. }) => {
                    let (fill_price, size) = (trade.exit_price, trade.size);
                    Ok(vec![fill(fill_price, size), EngineEvent::Exited { idx, trade }])
                }
                _ => Err(not_booked("the position could not be reduced")),
            }
        }
    }
}
