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

use crate::accounts::Money;
use crate::core::types::{Direction, ExitReason, Price};
use crate::core::FillTiming;
use crate::execution::orders::{
    ExecutionReport, Order, OrderEventError, OrderKind, OrderSide, OrderStatus, QtySpec,
    ReportOutcome, TimeInForce,
};
use crate::portfolio::kernel::{EngineEvent, EngineKernel, KernelBar};
use crate::portfolio::ledger::PositionPolicy;
use crate::portfolio::risk::RejectReason;

/// What the orders a venue holds would add to an instrument if they all
/// filled; see [`EngineKernel::venue_pretrade`].
#[derive(Debug, Default)]
struct WorkingEntries {
    buy_units: f64,
    sell_units: f64,
    notional: f64,
}

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
    /// The order took the fill, the book could not book it yet. Inside the
    /// kernel only: [`EngineKernel::apply_venue_report`] keeps such a fill and
    /// reports it as [`EngineEvent::FillDeferred`] instead of failing, so a
    /// fill the venue executed is never dropped.
    #[error("the venue filled order {order_id} but the book could not book the fill ({reason})")]
    FillNotBooked { order_id: u64, reason: String },
    /// The fee for a fill could not be booked at the book's money scale.
    #[error("the fee of a fill cannot be booked: {0}")]
    FeeNotBooked(String),
    /// A correction or bust of an execution whose effect on the book cannot be
    /// taken back exactly (it closed or reduced a position, or the position
    /// has been reduced since). Nothing was changed.
    #[error("execution {exec_id} of order {order_id} cannot be restated: {reason}")]
    NotReversible { order_id: u64, exec_id: String, reason: &'static str },
}

/// What booking one execution did to the book, kept so a correction or bust of
/// the execution can take it back exactly: the cash, the margin lock, the
/// position's booked cost and entry fees, all in the units they were booked in.
#[derive(Debug, Clone)]
pub(crate) struct VenueBooking {
    pub order_id: u64,
    pub exec_id: String,
    /// The quantity and price the venue reported.
    pub qty: f64,
    pub price: f64,
    /// The quantity and price the book took, after the lot and tick rules.
    pub booked_qty: f64,
    pub booked_price: f64,
    /// The execution opened or added to a position (an exit cannot be taken back).
    pub entry: bool,
    pub position_id: u64,
    pub cash: Money,
    pub locked: f64,
    pub cost: Money,
    pub fees: Money,
    /// Levels derived from the entry shifted with the average when it was added.
    pub shift: bool,
}

/// How many executions' bookings a book keeps to restate. A correction or bust
/// of an execution older than the last `BOOKING_WINDOW` bookings is refused
/// (`NotReversible`), because what it did to the book is no longer held.
pub(crate) const BOOKING_WINDOW: usize = 8192;

/// The bookings kept to restate executions: indexed by `(order, execution)`,
/// oldest first, bounded to [`BOOKING_WINDOW`].
#[derive(Debug, Default, Clone)]
pub(crate) struct VenueBookings {
    next: u64,
    by_seq: std::collections::BTreeMap<u64, VenueBooking>,
    /// Every booking one execution made, oldest first: a fill that reversed a position
    /// reduced it and opened the other way, two bookings of one execution.
    by_exec: std::collections::HashMap<(u64, String), Vec<u64>>,
    /// The latest booking that reduced each position.
    last_reduction: std::collections::HashMap<u64, u64>,
    evicted: bool,
}

impl VenueBookings {
    pub(crate) fn push(&mut self, b: VenueBooking) {
        let seq = self.next;
        self.next += 1;
        self.by_exec.entry((b.order_id, b.exec_id.clone())).or_default().push(seq);
        if !b.entry {
            self.last_reduction.insert(b.position_id, seq);
        }
        self.by_seq.insert(seq, b);
        while self.by_seq.len() > BOOKING_WINDOW {
            let Some((&old, _)) = self.by_seq.iter().next() else { break };
            self.remove(old);
            self.evicted = true;
        }
    }

    fn remove(&mut self, seq: u64) -> Option<VenueBooking> {
        let b = self.by_seq.remove(&seq)?;
        let key = (b.order_id, b.exec_id.clone());
        if let Some(all) = self.by_exec.get_mut(&key) {
            all.retain(|s| *s != seq);
            if all.is_empty() {
                self.by_exec.remove(&key);
            }
        }
        if self.last_reduction.get(&b.position_id) == Some(&seq) {
            self.last_reduction.remove(&b.position_id);
        }
        Some(b)
    }

    /// Every booking the execution made, oldest first.
    pub(crate) fn find_all(&self, order_id: u64, exec_id: &str) -> Vec<(u64, &VenueBooking)> {
        self.by_exec
            .get(&(order_id, exec_id.to_string()))
            .into_iter()
            .flatten()
            .filter_map(|seq| self.by_seq.get(seq).map(|b| (*seq, b)))
            .collect()
    }

    /// Whether a booking after `seq` reduced `position_id`.
    pub(crate) fn reduced_since(&self, seq: u64, position_id: u64) -> bool {
        self.last_reduction.get(&position_id).is_some_and(|last| *last > seq)
    }

    /// Whether bookings older than the window were dropped, and the oldest
    /// order still held.
    pub(crate) fn window_starts_after(&self, order_id: u64) -> bool {
        self.evicted && self.by_seq.values().next().is_none_or(|first| order_id < first.order_id)
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.by_seq.len()
    }
}

/// How many fills the book keeps unbooked at once. A book that holds this many has fallen far
/// behind its venue (or its limits refuse everything the venue does): the next fill that cannot
/// be booked is an error, not one more entry in a list nobody reads.
pub const MAX_UNBOOKED_FILLS: usize = 1024;

/// A venue fill the order took and the book has not booked. Kept, in arrival
/// order, until the book can book it; see `EngineKernel::unbooked_fills`.
#[derive(Debug, Clone, PartialEq)]
pub struct UnbookedFill {
    pub idx: usize,
    pub ts: i64,
    pub order_id: u64,
    pub exec_id: String,
    pub client_id: String,
    pub qty: f64,
    pub price: f64,
    pub reason: String,
}

/// A refusal reason from a venue, as the `&'static str` the events carry.
///
/// Reasons are interned: the set a venue uses is small, and the table is
/// capped so a venue that puts an id in its reason cannot grow it without
/// bound. Past the cap a reason reads `venue_rejected`.
fn intern_reason(reason: &str) -> &'static str {
    const CAP: usize = 64;
    static TABLE: Mutex<Vec<&'static str>> = Mutex::new(Vec::new());
    // A code the host's order check gave keeps its own spelling.
    if let Some(host) =
        HOST_CODES.lock().unwrap_or_else(|e| e.into_inner()).iter().find(|r| **r == reason)
    {
        return host;
    }
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

static HOST_CODES: Mutex<Vec<&'static str>> = Mutex::new(Vec::new());

/// A refusal code from the host's pre-submit check, as the `&'static str`
/// the events carry. Unlike a venue's reasons, a host code is never replaced:
/// a code that is empty, longer than 64 bytes, or would be the 1025th distinct
/// one is an error for the caller to see.
pub(crate) fn intern_host_code(code: &str) -> std::result::Result<&'static str, String> {
    const CAP: usize = 1024;
    const MAX_LEN: usize = 64;
    if code.is_empty() || code.len() > MAX_LEN {
        return Err(format!("an order refusal code is 1..={MAX_LEN} bytes, got {}", code.len()));
    }
    let mut table = HOST_CODES.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(known) = table.iter().find(|r| **r == code) {
        return Ok(known);
    }
    if table.len() >= CAP {
        return Err(format!(
            "more than {CAP} distinct order refusal codes: codes must be a small stable set, not carry ids"
        ));
    }
    let leaked: &'static str = Box::leak(code.to_string().into_boxed_str());
    table.push(leaked);
    Ok(leaked)
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

    /// Name the position an order closes (hedging); see `OrderRequest::position_id`.
    /// Mark a market order as the strategy's answer to a fill (see [`Order::follow_up`]).
    pub fn set_order_follow_up(&mut self, order_id: u64) {
        if let Some(order) = self.orders.get_mut(order_id) {
            order.follow_up = true;
        }
    }

    /// Take the follow-up mark off an order: it goes the ordinary way (to the next bar's open
    /// under `next_bar_open`).
    pub(crate) fn clear_order_follow_up(&mut self, order_id: u64) {
        if let Some(order) = self.orders.get_mut(order_id) {
            order.follow_up = false;
        }
    }

    /// Whether market orders decided at a bar fill at the next bar's open.
    pub(crate) fn fills_at_next_open(&self) -> bool {
        self.fill_timing == FillTiming::NextBarOpen
    }

    /// Mark an order as an `enter()` intent: with a netting position held it fills nothing.
    pub fn set_order_entry_intent(&mut self, order_id: u64) {
        if let Some(order) = self.orders.get_mut(order_id) {
            order.entry_intent = true;
        }
    }

    pub fn set_order_position(&mut self, order_id: u64, position_id: Option<u64>) {
        if let Some(order) = self.orders.get_mut(order_id) {
            order.position_id = position_id;
        }
    }

    /// The book's own limits, applied before an order is sent to a venue
    /// that does not apply them for us: the kill-switch and halts, the
    /// drawdown and daily-loss latches, `max_positions` and the order-size,
    /// position and exposure limits. A simulated venue applies them itself,
    /// at the fill, exactly as the matcher does, so it skips this.
    ///
    /// An order that can only shrink a position is always allowed, as the
    /// matcher allows exits when entries are blocked. `price` is the price
    /// the order names (its limit or trigger); `mark` is the last price seen,
    /// which prices an order that names none. `own` is the order being
    /// checked, already in the book, which must not count against itself.
    ///
    /// Orders the venue is already working, or has not yet answered, count
    /// as if filled: two entries that each fit under a cap must not both be
    /// sent when together they do not. An order that needs a price (a
    /// notional or exposure limit is set) and has neither `price` nor `mark`
    /// is refused as `no_mark`, never passed unchecked.
    pub fn venue_pretrade(
        &self,
        own: Option<u64>,
        side: OrderSide,
        units: f64,
        reduce_only: bool,
        price: Option<f64>,
        mark: Option<f64>,
    ) -> Result<(), RejectReason> {
        let direction = match side {
            OrderSide::Buy => Direction::Long,
            OrderSide::Sell => Direction::Short,
        };
        let hedging = self.ledger.policy() == PositionPolicy::Independent;
        let held = self.ledger.first().map(|m| m.position.direction);
        // The matcher's rule (see `apply_match_outcome`), so a venue is never
        // sent an order the backtest would refuse: under netting an order
        // against the position closes it, which is always allowed.
        if !hedging && matches!(held, Some(dir) if dir != direction) {
            return Ok(());
        }
        if reduce_only {
            // Under netting it cannot reduce a position on its own side. With
            // nothing held it may be a leg that waits for its entry, so it
            // goes; under hedging a reduce-only order names no position to
            // reduce and the venue decides.
            return if !hedging && held.is_some() { Err(RejectReason::ReduceOnly) } else { Ok(()) };
        }
        if self.session_state.entries_blocked {
            return Err(RejectReason::Squareoff);
        }
        if self.margin.is_halted() {
            return Err(RejectReason::MarginCall);
        }
        // An order on the side of a netting position adds to it: the position-count limit
        // does not bind it.
        let counted = if !hedging && held.is_some() { 0 } else { self.gating_open_count() };
        self.risk.check_entry(counted)?;
        if !self.limits.is_active() {
            return Ok(());
        }
        let Some(at) = price.or(mark) else {
            return if self.limits.needs_price() { Err(RejectReason::NoMark) } else { Ok(()) };
        };
        let multiplier = self.multiplier();
        let working = self.working_entries(own, at);
        self.limits.check_order(units, units * at * multiplier)?;
        let before = self.net_size();
        let after = match direction {
            Direction::Long => before + working.buy_units + units,
            Direction::Short => before - working.sell_units - units,
        };
        self.limits.check_position(before, after)?;
        if self.limits.tracks_exposure() {
            let existing =
                self.gross_notional(at) + self.external_exposure.unwrap_or(0.0) + working.notional;
            self.limits.check_exposure(existing, units * at * multiplier)?;
        }
        Ok(())
    }

    /// What the orders the venue holds for this instrument would add if they
    /// all filled: the units that buy, the units that sell and their notional,
    /// the order `own` left out. Orders that only reduce are not counted, and
    /// an order is priced at its limit or trigger, else at `mark`.
    fn working_entries(&self, own: Option<u64>, mark: Price) -> WorkingEntries {
        let multiplier = self.multiplier();
        let mut out = WorkingEntries::default();
        for o in self.orders.working() {
            if Some(o.id) == own || o.reduce_only || !o.is_venue_driven() {
                continue;
            }
            let QtySpec::Units(total) = o.qty else { continue };
            let left = (total - o.filled_qty).max(0.0);
            match o.side {
                OrderSide::Buy => out.buy_units += left,
                OrderSide::Sell => out.sell_units += left,
            }
            let at = o.kind.limit_price().or(o.kind.trigger_price()).unwrap_or(mark);
            out.notional += left * at * multiplier;
        }
        out
    }

    /// The gross notional the orders the venue holds for this instrument
    /// would add if they all filled, for the other instruments' exposure check.
    pub(crate) fn working_notional(&self, mark: Option<Price>) -> f64 {
        match mark {
            Some(mark) => self.working_entries(None, mark).notional,
            None => 0.0,
        }
    }

    /// Add to the gross notional the portfolio's other instruments hold, for
    /// `max_gross_exposure`, while the account is lent to this kernel.
    pub(crate) fn add_external_exposure(&mut self, extra: f64) {
        if let Some(e) = self.external_exposure.as_mut() {
            *e += extra;
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
        // A market order is sized at the last price known: under
        // `next_bar_open` that is the decision bar's close, the same figure
        // the book sizes a deferred entry at, not the open the order fills at.
        let price = ref_price.unwrap_or_else(|| {
            if self.fill_timing == FillTiming::NextBarOpen {
                bar.close
            } else {
                self.fill_price_for(bar, direction, true)
            }
        });
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

    /// The fee this book's fee model charges for trading `size` at `price`,
    /// as a fill of order `order_id` (a brokerage charged per order is owed
    /// once over the order's fills).
    /// `direction` is the position's: the long or short side being opened
    /// (`is_entry`) or closed. Rounded to the book's money scale; an amount
    /// that cannot be booked at that scale is an error, not a raw figure.
    pub fn venue_fee(
        &self,
        order_id: u64,
        price: f64,
        size: f64,
        direction: Direction,
        is_entry: bool,
    ) -> Result<f64, VenueError> {
        let fee_price = price * self.multiplier();
        // The fill was booked with its order's earlier fills in mind: price it the same.
        let before = self.fee_model.take_slice_before(order_id);
        let fee = match self.fee_model.breakdown_after(before, fee_price, size, direction, is_entry)
        {
            Some(b) => b.total(),
            None => self.fee_model.calculate(fee_price, size, direction),
        };
        let scale = self.money_scale();
        scale
            .money(fee)
            .map(|m| scale.to_f64(m))
            .map_err(|e| VenueError::FeeNotBooked(e.to_string()))
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
    /// Corrections and busts restate an execution after the position moved:
    /// the book takes the execution back and, for a correction, books it again
    /// as restated (see [`Self::restate_execution`]).
    pub fn apply_venue_report(
        &mut self,
        idx: usize,
        ts: i64,
        order_id: u64,
        report: &ExecutionReport,
    ) -> Result<(ReportOutcome, Vec<EngineEvent>), VenueError> {
        if let Some(done) = self.restate_execution(idx, ts, order_id, report)? {
            return Ok(done);
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
            ExecutionReport::Fill { exec_id, .. } => {
                // What the execution counts as now: a correction or bust that
                // came before the fill has already restated it.
                let counted = self.orders.get(order_id).and_then(|o| o.exec_effective(exec_id));
                if let Some((qty, price, _)) = counted {
                    events.extend(
                        self.book_or_keep_fill(idx, ts, order_id, &client_id, exec_id, qty, price)?,
                    );
                }
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
                    || (client_id.starts_with("enter-") && reason == "entry_held");
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

    /// A position's booked cost, entry fees and margin lock.
    fn position_books(&self, position_id: u64) -> (Money, Money, f64) {
        match self.ledger.get(position_id) {
            Some(m) => (m.cost, m.entry_fees_money, self.margin.locked_for(position_id)),
            None => (Money::ZERO, Money::ZERO, 0.0),
        }
    }

    /// A correction or bust of an execution the book has already taken:
    /// take its booking back exactly (cash, margin lock, position cost and
    /// entry fees, in the units they were booked in) and, for a correction,
    /// book the execution again as restated. `Ok(None)` when the book never
    /// took the execution (the correction came first, or the order has no such
    /// execution), which the order's own ledger handles.
    ///
    /// Only an execution that opened or added to a position, with that
    /// position still holding at least its quantity and nothing reducing it
    /// since, can be taken back exactly. Anything else is refused with
    /// [`VenueError::NotReversible`] before the order or the book is touched:
    /// an exit has already realised a trade, and the trades after it were
    /// priced on the position it left.
    ///
    /// The strategy hears no event for a restatement (the journal records it);
    /// it sees the restated position at its next event. Protective orders the
    /// venue holds for the original quantity are not resized: reconcile.
    fn restate_execution(
        &mut self,
        idx: usize,
        ts: i64,
        order_id: u64,
        report: &ExecutionReport,
    ) -> Result<Option<(ReportOutcome, Vec<EngineEvent>)>, VenueError> {
        let exec_id = match report {
            ExecutionReport::Correction { exec_id, .. } | ExecutionReport::Bust { exec_id, .. } => {
                exec_id
            }
            _ => return Ok(None),
        };
        let found: Vec<(u64, VenueBooking)> = self
            .venue_bookings
            .find_all(order_id, exec_id)
            .into_iter()
            .map(|(at, b)| (at, b.clone()))
            .collect();
        let [(at, b)] = found.as_slice() else {
            if found.len() > 1 {
                // A fill that reversed a position reduced it and opened the other way. Taking
                // back only the opening would leave the reduction booked: cash made from a
                // fill the venue says did not happen.
                return Err(VenueError::NotReversible {
                    order_id,
                    exec_id: exec_id.clone(),
                    reason: "the execution booked more than one position change (it reversed a position)",
                });
            }
            if self.venue_bookings.window_starts_after(order_id) {
                return Err(VenueError::NotReversible {
                    order_id,
                    exec_id: exec_id.clone(),
                    reason: "the execution is older than the bookings the book keeps",
                });
            }
            return Ok(None);
        };
        let (at, b) = (*at, b.clone());
        let refuse = |reason: &'static str| VenueError::NotReversible {
            order_id,
            exec_id: exec_id.clone(),
            reason,
        };
        if !b.entry {
            return Err(refuse("the execution closed or reduced a position"));
        }
        if self.venue_bookings.reduced_since(at, b.position_id) {
            return Err(refuse("the position has been reduced since"));
        }
        match self.ledger.get(b.position_id) {
            None => return Err(refuse("the position it opened has closed")),
            Some(m) if m.position.size + b.booked_qty * 1e-9 < b.booked_qty => {
                return Err(refuse("the position is smaller than the execution"))
            }
            Some(_) => {}
        }
        // `take_back` releases the lock the execution took; it must still be held.
        if self.margin.locked_for(b.position_id) + b.locked.abs() * 1e-9 < b.locked {
            return Err(refuse("the margin the execution locked is no longer held"));
        }
        // A correction books the execution again after taking it back; if the book cannot keep
        // a fill it fails to book, that would refuse after the take-back. Refuse before anything
        // changes: the order, the position and the cash stay as they were.
        if matches!(report, ExecutionReport::Correction { .. })
            && self.unbooked_fills.len() >= MAX_UNBOOKED_FILLS
        {
            return Err(VenueError::FillNotBooked {
                order_id,
                reason: format!(
                    "the book already holds {MAX_UNBOOKED_FILLS} fills it could not book, so a \
                     correction that must be booked again is refused before it changes anything: \
                     reconcile the position with the venue"
                ),
            });
        }
        let order = self.orders.get_mut(order_id).ok_or(VenueError::UnknownOrder(order_id))?;
        let client_id = order.client_id.clone();
        let outcome = order.apply_report(report)?;
        if outcome == ReportOutcome::Duplicate {
            return Ok(Some((outcome, Vec::new())));
        }
        let counted = self.orders.get(order_id).and_then(|o| o.exec_effective(exec_id));
        if counted.is_some_and(|(q, p, _)| q == b.qty && p == b.price) {
            return Ok(Some((outcome, Vec::new())));
        }
        self.take_back(at, order_id, exec_id)?;
        let mut deferred = Vec::new();
        if let Some((qty, price, _)) = counted {
            let protect = self.protect_outbox.len();
            // A restated fill the book cannot book is kept, unbooked, like any
            // fill the book cannot take: the order already counts it, and the
            // book must not lose it.
            let mut booked = Vec::new();
            self.try_book(idx, ts, order_id, &client_id, exec_id, qty, price, &mut booked)?;
            deferred.extend(
                booked.into_iter().filter(|e| matches!(e, EngineEvent::FillDeferred { .. })),
            );
            // The position's protection was sent when it first opened.
            self.protect_outbox.truncate(protect);
        }
        // Taking the execution back (or booking it smaller) frees cash, margin and room:
        // fills the book could not take before may book now, and the strategy hears of them.
        self.rebook_unbooked(&mut deferred);
        Ok(Some((outcome, deferred)))
    }

    /// Undo one booking (checked by [`Self::restate_execution`], which found it a moment
    /// ago): a booking that is no longer held is an error, never a silent return.
    fn take_back(&mut self, at: u64, order_id: u64, exec_id: &str) -> Result<(), VenueError> {
        let Some(b) = self.venue_bookings.remove(at) else {
            return Err(VenueError::NotReversible {
                order_id,
                exec_id: exec_id.to_string(),
                reason: "the booking to take back is no longer held",
            });
        };
        self.set_cash_money(self.cash_money() - b.cash);
        let whole = self
            .ledger
            .get(b.position_id)
            .is_some_and(|m| m.position.size - b.booked_qty <= b.booked_qty * 1e-9);
        if whole {
            self.ledger.discard_position(b.position_id);
            self.margin.release(b.position_id);
        } else {
            self.ledger.unadd_from_position(
                b.position_id,
                b.booked_price,
                b.booked_qty,
                b.cost,
                b.fees,
                b.shift,
            );
            let locked = self.margin.locked_for(b.position_id);
            self.margin.set_locked(b.position_id, locked - b.locked);
        }
        Ok(())
    }

    /// Venue fills the orders took that the book has not booked, oldest
    /// first. Empty when the book agrees with the venue.
    pub fn unbooked_fills(&self) -> &[UnbookedFill] {
        &self.unbooked_fills
    }

    /// Book a fill the order has taken. A fill the book cannot book yet is
    /// kept and reported as [`EngineEvent::FillDeferred`]; every booking that
    /// succeeds then tries the kept fills again, in order, so reports that
    /// arrive across orders in another order than they happened leave the
    /// book the venue's.
    #[allow(clippy::too_many_arguments)]
    fn book_or_keep_fill(
        &mut self,
        idx: usize,
        ts: i64,
        order_id: u64,
        client_id: &str,
        exec_id: &str,
        qty: f64,
        price: f64,
    ) -> Result<Vec<EngineEvent>, VenueError> {
        let mut events = Vec::new();
        self.try_book(idx, ts, order_id, client_id, exec_id, qty, price, &mut events)?;
        if events
            .iter()
            .any(|e| matches!(e, EngineEvent::Entered { .. } | EngineEvent::Exited { .. }))
        {
            self.rebook_unbooked(&mut events);
        }
        Ok(events)
    }

    /// One attempt; on failure the fill (or what is left of it) is kept.
    #[allow(clippy::too_many_arguments)]
    fn try_book(
        &mut self,
        idx: usize,
        ts: i64,
        order_id: u64,
        client_id: &str,
        exec_id: &str,
        qty: f64,
        price: f64,
        events: &mut Vec<EngineEvent>,
    ) -> Result<(), VenueError> {
        if let Err((left, reason)) =
            self.book_venue_fill(idx, ts, order_id, client_id, exec_id, qty, price, events)
        {
            if self.unbooked_fills.len() >= MAX_UNBOOKED_FILLS {
                return Err(VenueError::FillNotBooked {
                    order_id,
                    reason: format!(
                        "{reason}; the book already holds {MAX_UNBOOKED_FILLS} fills it could not book, \
                         so it will not keep more: reconcile the position with the venue"
                    ),
                });
            }
            events.push(EngineEvent::FillDeferred {
                idx,
                order_id,
                exec_id: exec_id.to_string(),
                client_id: client_id.to_string(),
                price,
                size: left,
                reason: reason.clone(),
            });
            self.unbooked_fills.push(UnbookedFill {
                idx,
                ts,
                order_id,
                exec_id: exec_id.to_string(),
                client_id: client_id.to_string(),
                qty: left,
                price,
                reason,
            });
        }
        Ok(())
    }

    fn rebook_unbooked(&mut self, events: &mut Vec<EngineEvent>) {
        loop {
            let kept = std::mem::take(&mut self.unbooked_fills);
            let before = kept.len();
            let events_before = events.len();
            let mut again = Vec::new();
            for fill in kept {
                let mut booked = Vec::new();
                let result = self.book_venue_fill(
                    fill.idx,
                    fill.ts,
                    fill.order_id,
                    &fill.client_id,
                    &fill.exec_id,
                    fill.qty,
                    fill.price,
                    &mut booked,
                );
                events.append(&mut booked);
                if let Err((left, reason)) = result {
                    again.push(UnbookedFill { qty: left, reason, ..fill });
                }
            }
            let progressed = events.len() > events_before || again.len() < before;
            self.unbooked_fills = again;
            if !progressed || self.unbooked_fills.is_empty() {
                return;
            }
        }
    }

    /// Book one execution as the fact it is: on the position ledger and the
    /// account, at the venue's price and size exactly (see
    /// [`EngineKernel::booking_fact`]). Appends the events it caused; on `Err`
    /// returns the quantity left unbooked and why, with the events of what
    /// was booked already appended.
    #[allow(clippy::too_many_arguments)]
    fn book_venue_fill(
        &mut self,
        idx: usize,
        ts: i64,
        order_id: u64,
        client_id: &str,
        exec_id: &str,
        qty: f64,
        price: f64,
        events: &mut Vec<EngineEvent>,
    ) -> Result<(), (f64, String)> {
        self.booking_fact = true;
        self.fee_model.begin_order(order_id);
        let result = self.book_fact(idx, ts, order_id, client_id, exec_id, qty, price, events);
        let orders = &self.orders;
        self.fee_model.end_order(|id| orders.get(id).is_some_and(|o| !o.status.is_terminal()));
        self.booking_fact = false;
        for reason in std::mem::take(&mut self.fact_breaches) {
            if !events.iter().any(|e| {
                matches!(e, EngineEvent::FillBreach { order_id: o, reason: r, .. }
                    if *o == order_id && *r == reason)
            }) {
                events.push(EngineEvent::FillBreach {
                    idx,
                    reason,
                    order_id,
                    client_id: client_id.to_string(),
                });
            }
        }
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn book_fact(
        &mut self,
        idx: usize,
        ts: i64,
        order_id: u64,
        client_id: &str,
        exec_id: &str,
        qty: f64,
        price: f64,
        events: &mut Vec<EngineEvent>,
    ) -> Result<(), (f64, String)> {
        let Some(order) = self.orders.get(order_id) else {
            return Err((qty, "the order is not in this book".to_string()));
        };
        let (side, reduce_only) = (order.side, order.reduce_only);
        let target_position = order.position_id;
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
        let fill = |price: f64, size: f64| EngineEvent::OrderFilled {
            idx,
            order_id,
            client_id: client_id.to_string(),
            price,
            size,
        };
        // Under netting a venue holds one net position per instrument, so a fill
        // against the side held reduces it, oldest position first; any quantity
        // beyond what is held reverses it. Under hedging the venue keeps positions
        // apart, as the matcher does: an order that names a position
        // (`position_id`) reduces exactly that one, a reduce-only order that names
        // none takes the oldest, and any other order opens a position of its own.
        let hedging = self.ledger.policy() == PositionPolicy::Independent;
        let mut remaining = qty;
        let held: Vec<(u64, f64)> = match target_position {
            Some(position_id) => {
                let named = self
                    .ledger
                    .get(position_id)
                    .filter(|m| m.position.direction != order_direction)
                    .map(|m| (m.id, m.position.size));
                if named.is_none() {
                    return Err((
                        qty,
                        "the position the order names is not open on the other side".to_string(),
                    ));
                }
                named.into_iter().collect()
            }
            None if hedging && !reduce_only => Vec::new(),
            None => self
                .ledger
                .positions()
                .iter()
                .filter(|m| m.position.direction != order_direction)
                .map(|m| (m.id, m.position.size))
                .collect(),
        };
        if !held.is_empty() {
            for (position_id, held_size) in held {
                let slice = remaining.min(held_size);
                let cash_before = self.cash_money();
                match self.reduce_at(idx, &bar, position_id, slice, price, ExitReason::Order, true)
                {
                    Some(EngineEvent::Exited { trade, .. }) => {
                        self.venue_bookings.push(VenueBooking {
                            order_id,
                            exec_id: exec_id.to_string(),
                            qty,
                            price,
                            booked_qty: trade.size,
                            booked_price: trade.exit_price,
                            entry: false,
                            position_id,
                            cash: self.cash_money() - cash_before,
                            locked: 0.0,
                            cost: Money::ZERO,
                            fees: Money::ZERO,
                            shift: false,
                        });
                        events.push(fill(trade.exit_price, trade.size));
                        events.push(EngineEvent::Exited { idx, trade });
                    }
                    _ => return Err((remaining, "the position could not be reduced".to_string())),
                }
                remaining -= slice;
                if remaining <= qty * 1e-9 {
                    return Ok(());
                }
            }
            // A reduce-only fill larger than the position is not a
            // reversal: part of the position it reduces was filled by an
            // order whose report has not arrived. The excess is kept.
            if reduce_only || target_position.is_some() {
                return Err((
                    remaining,
                    "a reduce-only fill is larger than the position held".to_string(),
                ));
            }
            // A fill larger than the position reverses it: the venue
            // executed the whole quantity, so the rest opens the other
            // way. The matcher never does this (an exit is capped at the
            // position), so the book follows the venue.
            return self.open_fact(
                idx,
                &bar,
                order_id,
                client_id,
                exec_id,
                order_direction,
                remaining,
                price,
                (stop_attach, target_attach),
                events,
            );
        }
        if reduce_only {
            return Err((qty, "a reduce-only fill arrived with no position to reduce".to_string()));
        }
        self.open_fact(
            idx,
            &bar,
            order_id,
            client_id,
            exec_id,
            order_direction,
            remaining,
            price,
            (stop_attach, target_attach),
            events,
        )
    }

    /// Open or add `qty` in `direction` at the venue's price. Under netting a
    /// fill on the side of the position another order opened adds to it, as the
    /// matcher's does.
    #[allow(clippy::too_many_arguments)]
    fn open_fact(
        &mut self,
        idx: usize,
        bar: &KernelBar,
        order_id: u64,
        client_id: &str,
        exec_id: &str,
        direction: Direction,
        qty: f64,
        price: f64,
        (stop_attach, target_attach): (Option<f64>, Option<f64>),
        events: &mut Vec<EngineEvent>,
    ) -> Result<(), (f64, String)> {
        let hedging = self.ledger.policy() == PositionPolicy::Independent;
        let fill = |price: f64, size: f64| EngineEvent::OrderFilled {
            idx,
            order_id,
            client_id: client_id.to_string(),
            price,
            size,
        };
        let cash_before = self.cash_money();
        let shift = stop_attach.is_none() && target_attach.is_none();
        // A later part of the fill of an order adds to the position that order
        // opened; under hedging a fill of another order opens its own.
        let adds_to = self
            .ledger
            .positions()
            .iter()
            .find(|m| m.entry_order_id == Some(order_id))
            .map(|m| m.id);
        let scale_in = if hedging { None } else { self.ledger.first().map(|m| m.id) };
        if let Some(position_id) = adds_to.or(scale_in) {
            let (cost0, fees0, lock0) = self.position_books(position_id);
            let Some((fill_price, size)) =
                self.add_at(idx, bar, position_id, price, qty, stop_attach, target_attach, true)
            else {
                return Err((qty, "the added quantity could not be booked".to_string()));
            };
            let (cost1, fees1, lock1) = self.position_books(position_id);
            self.venue_bookings.push(VenueBooking {
                order_id,
                exec_id: exec_id.to_string(),
                qty,
                price,
                booked_qty: size,
                booked_price: fill_price,
                entry: true,
                position_id,
                cash: self.cash_money() - cash_before,
                locked: lock1 - lock0,
                cost: cost1 - cost0,
                fees: fees1 - fees0,
                shift,
            });
            events.push(fill(fill_price, size));
            events.push(EngineEvent::Entered { idx, price: fill_price, size, direction });
            return Ok(());
        }
        match self.open_at(
            idx,
            bar,
            direction,
            price,
            None,
            Some(qty),
            self.last_atr,
            stop_attach,
            target_attach,
            true,
        ) {
            Some(EngineEvent::Entered { price: fill_price, size, direction, .. }) => {
                // A fill at the bar's open under `next_bar_open` lives through the whole bar,
                // so its range counts toward the excursions, as for an in-process fill.
                if self.fill_timing == FillTiming::NextBarOpen {
                    if let Some((_, stepped)) = self.last_step {
                        if stepped.timestamp == bar.timestamp && stepped.open == price {
                            self.ledger.update_price(stepped.high, stepped.low);
                        }
                    }
                }
                // The position just opened is the newest one.
                let opened = self.ledger.positions().last().map(|m| m.id);
                if let Some(position_id) = opened {
                    if let Some(m) = self.ledger.get_mut(position_id) {
                        m.entry_order_id = Some(order_id);
                    }
                    let (cost1, fees1, lock1) = self.position_books(position_id);
                    self.venue_bookings.push(VenueBooking {
                        order_id,
                        exec_id: exec_id.to_string(),
                        qty,
                        price,
                        booked_qty: size,
                        booked_price: fill_price,
                        entry: true,
                        position_id,
                        cash: self.cash_money() - cash_before,
                        locked: lock1,
                        cost: cost1,
                        fees: fees1,
                        shift: false,
                    });
                }
                events.push(fill(fill_price, size));
                events.push(EngineEvent::Entered { idx, price: fill_price, size, direction });
                Ok(())
            }
            Some(EngineEvent::EntryRejected { reason, .. }) => {
                Err((qty, reason.as_str().to_string()))
            }
            _ => Err((qty, "the position could not be opened".to_string())),
        }
    }
}
