//! Order lifecycle on the kernel: submission, cancellation, modification,
//! and applying the matcher's outcomes.
//!
//! Split out of `kernel.rs` to keep that file reviewable; this is the same
//! `impl EngineKernel`, not a separate type.

use crate::core::types::{Direction, ExitReason, Price};
use crate::core::FillTiming;
use crate::execution::algos::{AlgoError, ExecAlgorithm};
use crate::execution::orders::{
    MatchOutcome, Order, OrderEngine, OrderKind, OrderSide, OrderStatus, QtySpec, TimeInForce,
};
use crate::execution::queue::QueueVerdict;
use crate::portfolio::kernel::{EngineEvent, EngineKernel, KernelBar};
use crate::portfolio::ledger::PositionPolicy;
use crate::portfolio::risk::{RejectReason, RiskGate};

impl EngineKernel {
    /// Register an execution schedule that releases slices over time.
    ///
    /// Only `QtySpec::Units` is sliceable: `CapitalFrac` resolves against
    /// equity at fill time, so each slice would size against a different
    /// account, and `FullPosition` sliced N ways would close the whole
    /// position N times. Both are refused rather than guessed at.
    #[allow(clippy::too_many_arguments)]
    pub fn submit_algo(
        &mut self,
        side: OrderSide,
        qty: QtySpec,
        kind: OrderKind,
        tif: TimeInForce,
        client_id: String,
        algo: ExecAlgorithm,
        reduce_only: bool,
        now_ns: i64,
        idx: usize,
    ) -> Result<u64, AlgoError> {
        let units = match qty {
            QtySpec::Units(units) => units,
            _ => return Err(AlgoError::InvalidUnits),
        };
        let algo_id = self.algos.submit(
            client_id.clone(),
            side,
            kind,
            tif,
            units,
            algo,
            reduce_only,
            now_ns,
        )?;
        self.pending_events.push(EngineEvent::AlgoStarted { idx, algo_id, client_id });
        Ok(algo_id)
    }

    /// Submit every slice due at this timestamp.
    ///
    /// Called from the step just before market orders sweep, so a released
    /// slice fills on the same step rather than trailing one behind.
    pub(crate) fn release_algo_slices(
        &mut self,
        idx: usize,
        now_ns: i64,
        events: &mut Vec<EngineEvent>,
    ) {
        if self.algos.is_empty() {
            return;
        }
        for slice in self.algos.release_due(now_ns) {
            if self.venue_managed {
                // The runtime sends it to the venue.
                self.slice_outbox.push(slice);
                continue;
            }
            let order_id = self.submit_order_full(
                slice.side,
                QtySpec::Units(slice.units),
                slice.kind,
                slice.tif,
                idx,
                now_ns,
                slice.client_id,
                None,
                None,
                false,
                slice.reduce_only,
                None,
            );
            self.orders.set_algo_id(order_id, Some(slice.algo_id));
        }
        events.append(&mut self.pending_events);
        for algo_id in self.algos.drain_completed() {
            events.push(EngineEvent::AlgoCompleted { idx, algo_id, client_id: String::new() });
        }
    }

    /// Stop a schedule and cancel the slices it has working.
    ///
    /// Slices that already filled stay filled: cancelling a schedule halts
    /// the remainder, it does not unwind what traded.
    pub fn cancel_algo(&mut self, algo_id: u64, idx: usize) -> bool {
        if !self.algos.cancel(algo_id) {
            return false;
        }
        for order_id in self.orders.algo_order_ids(algo_id) {
            self.cancel_order(idx, order_id);
        }
        true
    }

    /// Units still unreleased by a schedule, for diagnostics.
    pub fn algo_released(&self, algo_id: u64) -> Option<u32> {
        self.algos.get(algo_id).map(|s| s.released())
    }

    /// Submit an order from the class-based order API.
    ///
    /// `submitted_idx` is the bar the strategy was observing when it placed
    /// the order: market orders fill on that bar's step (matching the
    /// signal-entry contract), resting orders begin matching on the next
    /// bar. Returns the engine order id; acknowledgment events are
    /// delivered at the front of the next step's event list.
    #[allow(clippy::too_many_arguments)]
    pub fn submit_order(
        &mut self,
        side: OrderSide,
        qty: QtySpec,
        kind: OrderKind,
        tif: TimeInForce,
        submitted_idx: usize,
        submitted_ts: i64,
        client_id: String,
        stop_price: Option<Price>,
        target_price: Option<Price>,
    ) -> u64 {
        self.submit_order_full(
            side,
            qty,
            kind,
            tif,
            submitted_idx,
            submitted_ts,
            client_id,
            stop_price,
            target_price,
            false,
            false,
            None,
        )
    }

    /// Record an order the host refused before it was submitted: it exists
    /// for one event, `Rejected` with the host's `reason`, and is counted and
    /// delivered like any other rejection. It never works and never matches.
    #[allow(clippy::too_many_arguments)]
    pub fn submit_refused_order(
        &mut self,
        side: OrderSide,
        qty: QtySpec,
        kind: OrderKind,
        tif: TimeInForce,
        submitted_idx: usize,
        submitted_ts: i64,
        client_id: String,
        reduce_only: bool,
        reason: &'static str,
    ) -> u64 {
        let mut order = Order::plain(side, qty, kind, tif);
        order.client_id = client_id.clone();
        order.submitted_idx = submitted_idx;
        order.submitted_ts = submitted_ts;
        order.reduce_only = reduce_only;
        let id = self.orders.submit(order);
        if let Some(order) = self.orders.get_mut(id) {
            let _ = order.transition(OrderStatus::Rejected);
        }
        self.risk.record_rejection();
        self.pending_events.push(EngineEvent::OrderRejected {
            idx: submitted_idx,
            order_id: id,
            client_id,
            reason,
        });
        id
    }

    /// [`EngineKernel::submit_order`] with flags and one-triggers-other
    /// linkage. `parent_id` holds the order (unmatched, no expiry clock)
    /// until the parent fills; a dead parent cancels it.
    #[allow(clippy::too_many_arguments)]
    pub fn submit_order_full(
        &mut self,
        side: OrderSide,
        qty: QtySpec,
        kind: OrderKind,
        tif: TimeInForce,
        submitted_idx: usize,
        submitted_ts: i64,
        client_id: String,
        stop_price: Option<Price>,
        target_price: Option<Price>,
        post_only: bool,
        reduce_only: bool,
        parent_id: Option<u64>,
    ) -> u64 {
        let mut order = Order::plain(side, qty, kind, tif);
        order.client_id = client_id.clone();
        order.submitted_idx = submitted_idx;
        order.submitted_ts = submitted_ts;
        order.stop_price = stop_price;
        order.target_price = target_price;
        order.post_only = post_only;
        order.reduce_only = reduce_only;
        order.parent_id = parent_id;
        if self.fill_timing == FillTiming::NextBarOpen && matches!(qty, QtySpec::CapitalFrac(_)) {
            if self.last_step.is_some_and(|(stepped, _)| stepped == submitted_idx) {
                // Decided on a bar this kernel has already stepped (an order
                // routed from another instrument's event): that bar's close.
                order.size_ref = self.last_close;
            } else {
                // Decided on the event about to be stepped: the step stamps
                // the close of that bar.
                self.size_ref_pending = true;
            }
        }
        let id = self.orders.submit(order);

        // An expired contract takes no orders: the refusal is a rejection with
        // the stable reason `expired`, counted and delivered like any other.
        if self.expired_at(submitted_ts) {
            if let Some(order) = self.orders.get_mut(id) {
                let _ = order.transition(OrderStatus::Rejected);
            }
            self.risk.record_rejection();
            self.pending_events.push(EngineEvent::OrderRejected {
                idx: submitted_idx,
                order_id: id,
                client_id,
                reason: RejectReason::Expired.as_str(),
            });
            return id;
        }

        // A closed market takes no orders. The refusal is an ordinary
        // rejection, so it is counted and reaches the strategy as an event.
        if self.closed_at(submitted_ts) {
            if let Some(order) = self.orders.get_mut(id) {
                let _ = order.transition(OrderStatus::Rejected);
            }
            self.risk.record_rejection();
            self.pending_events.push(EngineEvent::OrderRejected {
                idx: submitted_idx,
                order_id: id,
                client_id,
                reason: RejectReason::MarketClosed.as_str(),
            });
            return id;
        }

        // Resting kinds start working immediately (held children stay
        // Submitted until their parent fills); plain market kinds are
        // acknowledged when their fill is processed in the step.
        let rests = !matches!(kind, OrderKind::Market)
            || matches!(tif, TimeInForce::AtOpen | TimeInForce::AtClose);
        if rests && parent_id.is_none() {
            if let Some(order) = self.orders.get_mut(id) {
                let _ = order.transition(OrderStatus::Accepted);
            }
            self.pending_events.push(EngineEvent::OrderAccepted {
                idx: submitted_idx,
                order_id: id,
                client_id,
            });
        }
        id
    }

    /// Put a set of working orders in one one-cancels-other group: the first
    /// fill among them cancels the rest. One-updates-other reduces to this
    /// while fills are all-or-nothing.
    pub fn link_oco(&mut self, ids: &[u64]) {
        let group = ids.iter().copied().min().unwrap_or(0);
        for id in ids {
            if let Some(order) = self.orders.get_mut(*id) {
                order.oco_group = Some(group);
            }
        }
    }

    /// Whether positions are independent (hedging) rather than netted.
    pub fn is_hedging(&self) -> bool {
        self.ledger.policy() == PositionPolicy::Independent
    }

    /// Cancel a working order. Returns `false` for unknown/finished ids.
    pub fn cancel_order(&mut self, idx: usize, id: u64) -> bool {
        let client_id = match self.orders.get(id) {
            Some(order) => order.client_id.clone(),
            None => return false,
        };
        if self.orders.cancel(id) {
            self.pending_events.push(EngineEvent::OrderCanceled { idx, order_id: id, client_id });
            true
        } else {
            false
        }
    }

    /// Cancel every working order.
    pub fn cancel_all_orders(&mut self, idx: usize) -> Vec<u64> {
        let ids = self.orders.cancel_all();
        for id in &ids {
            let client_id = self.orders.get(*id).map(|o| o.client_id.clone()).unwrap_or_default();
            self.pending_events.push(EngineEvent::OrderCanceled { idx, order_id: *id, client_id });
        }
        ids
    }

    /// Replace a working order's prices/quantity. Returns `false` when the
    /// order is unknown, finished, or the modification is not applicable.
    pub fn modify_order(
        &mut self,
        id: u64,
        qty: Option<QtySpec>,
        limit_price: Option<Price>,
        trigger_price: Option<Price>,
    ) -> bool {
        self.modify_order_at(id, qty, limit_price, trigger_price, None)
    }

    /// [`EngineKernel::modify_order`] for a change decided on a bar this kernel has not stepped
    /// yet: `decided_close` is that bar's close, which a limit moved through is compared with
    /// (the kernel's own last close is the bar before). `None` reads the kernel's own.
    pub fn modify_order_at(
        &mut self,
        id: u64,
        qty: Option<QtySpec>,
        limit_price: Option<Price>,
        trigger_price: Option<Price>,
        decided_close: Option<Price>,
    ) -> bool {
        let done = self.orders.modify(id, qty, limit_price, trigger_price);
        // A limit moved through the market (past the close the kernel last saw) crosses at the
        // venue: the next bar it meets is its arrival, so it trades there at the open, the
        // market, not at the limit it now sits behind. A limit moved to a level the market has
        // not reached keeps resting.
        if done && limit_price.is_some() {
            if let (Some(close), Some(order)) =
                (decided_close.or(self.last_close), self.orders.get_mut(id))
            {
                if let OrderKind::Limit { price } = order.kind {
                    let through = match order.side {
                        OrderSide::Buy => price > close,
                        OrderSide::Sell => price < close,
                    };
                    if through {
                        order.arrival_idx = None;
                    }
                }
            }
        }
        done
    }

    /// Shared view of an order by engine id.
    pub fn order(&self, id: u64) -> Option<&Order> {
        self.orders.get(id)
    }

    /// Instrument tick size; `0.0` without a spec.
    pub fn price_increment(&self) -> f64 {
        self.spec.as_ref().map(|s| s.price_increment).unwrap_or(0.0)
    }

    /// The lot, size step and tick this instrument's orders must sit on.
    pub fn order_grid(&self) -> crate::execution::orders::OrderGrid {
        crate::execution::orders::OrderGrid {
            lot: self.lot_size.unwrap_or(0.0).max(0.0),
            size_step: self.spec.as_ref().map_or(0.0, |s| s.size_increment).max(0.0),
            tick: self.price_increment().max(0.0),
        }
    }

    /// Match an order that arrived with the bar just stepped against the rest
    /// of that bar, and return the events it and the acknowledgements queued
    /// before it produced, in order.
    ///
    /// For a protective leg a venue is sent in answer to an entry's fill at
    /// the bar's open: a stop or target working from that price sees the
    /// remainder of the bar, as it does when the kernel itself activates the
    /// leg (`next_bar_open` entry-bar protection).
    pub fn match_arrived(&mut self, order_id: u64, arrived_ns: i64) -> Vec<EngineEvent> {
        let mut events = std::mem::take(&mut self.pending_events);
        // An entry fills at a bar's open only under `next_bar_open`; under any
        // other timing it fills at the close and the bar has nothing left.
        if self.fill_timing != FillTiming::NextBarOpen {
            return events;
        }
        // Only an order that arrives with the bar just stepped sees it.
        let Some((idx, bar)) = self.last_step.filter(|(_, b)| b.timestamp == arrived_ns) else {
            return events;
        };
        let ohlcv = bar.to_ohlcv_bar();
        let outcomes = self.orders.match_bar_among(idx, &ohlcv, &self.fill_model, &[order_id]);
        for outcome in outcomes {
            self.apply_match_outcome(idx, &bar, outcome, &mut events);
        }
        events
    }

    /// Whether this kernel already stepped the bar stamped `ts` and fills market orders decided
    /// at a bar on that bar's close: an order decided at `ts` and sent only now (by another
    /// instrument's event) has missed it, and a batch run would have met it there.
    pub fn stepped_bar_at(&self, ts: i64) -> bool {
        self.fill_timing != FillTiming::NextBarOpen
            && self.last_step.is_some_and(|(_, b)| b.timestamp == ts)
    }

    /// Fill, on the bar this kernel stepped at `ts`, a follow-up market order sent after that step
    /// (a strategy's answer to a fill it heard in the step), at the price a market order decided
    /// at that bar's observation takes: the close under `same_bar_close`, the open under
    /// `next_bar_open`. An order the kernel has not stepped `ts` for yet is left: its own step
    /// meets it (`Order::follow_up`). Returns the events the fill produced.
    pub fn fill_follow_up(&mut self, order_id: u64, ts: i64) -> Vec<EngineEvent> {
        let mut events = std::mem::take(&mut self.pending_events);
        let Some((idx, bar)) = self
            .last_step
            .filter(|(_, b)| b.timestamp == ts)
            .or(self.rest_step.filter(|(_, b)| b.timestamp == ts))
        else {
            self.pending_events = events;
            return Vec::new();
        };
        let waiting = self.orders.get(order_id).is_some_and(|o| {
            o.follow_up && matches!(o.kind, OrderKind::Market) && !o.status.is_terminal()
        });
        let open = self.schedule().is_none_or(|s| s.is_open(bar.timestamp.saturating_sub(1)));
        if !waiting || !open {
            self.pending_events = events;
            return Vec::new();
        }
        if let Some(order) =
            self.orders.get_mut(order_id).filter(|o| o.status == OrderStatus::Submitted)
        {
            let _ = order.transition(OrderStatus::Accepted);
            let client_id = order.client_id.clone();
            events.push(EngineEvent::OrderAccepted { idx, order_id, client_id });
        }
        self.apply_match_outcome(
            idx,
            &bar,
            MatchOutcome::Fill { order_id, price: f64::NAN },
            &mut events,
        );
        events
    }

    /// All non-terminal orders, in submission order.
    pub fn open_orders(&self) -> Vec<&Order> {
        self.orders.working().collect()
    }
    /// Queue verdict for a resting limit, or `None` when the queue model is
    /// off or cannot see enough to judge — the caller then falls back to
    /// `fill_prob_limit`.
    ///
    /// Only trade prints carry the size the model consumes, so bar events
    /// always fall back: a bar's volume is not volume *at* the limit price.
    fn queue_verdict(
        &mut self,
        order_id: u64,
        kind: OrderKind,
        status: OrderStatus,
        side: OrderSide,
        bar: &KernelBar,
    ) -> Option<QueueVerdict> {
        if !self.config.queue_fill_model || !self.stepping_trade {
            return None;
        }
        let limit_price = match kind {
            OrderKind::Limit { price } => price,
            OrderKind::StopLimit { price, .. }
                if matches!(status, OrderStatus::Triggered | OrderStatus::PartiallyFilled) =>
            {
                price
            }
            _ => return None,
        };
        let direction = match side {
            OrderSide::Buy => Direction::Long,
            OrderSide::Sell => Direction::Short,
        };
        let verdict = self.queue.observe_print(
            order_id,
            limit_price,
            direction,
            &self.book,
            bar.close,
            bar.volume,
        );
        (verdict != QueueVerdict::Unknown).then_some(verdict)
    }

    pub(crate) fn apply_match_outcome(
        &mut self,
        idx: usize,
        bar: &KernelBar,
        outcome: MatchOutcome,
        events: &mut Vec<EngineEvent>,
    ) {
        // The fills of one order share its per-order brokerage.
        if let MatchOutcome::Fill { order_id, .. } = &outcome {
            self.fee_model.begin_order(*order_id);
        }
        self.apply_match_outcome_inner(idx, bar, outcome, events, false);
        let orders = &self.orders;
        self.fee_model.end_order(|id| orders.get(id).is_some_and(|o| !o.status.is_terminal()));
    }

    fn apply_match_outcome_inner(
        &mut self,
        idx: usize,
        bar: &KernelBar,
        outcome: MatchOutcome,
        events: &mut Vec<EngineEvent>,
        continuation: bool,
    ) {
        // `continuation`: the rest of a fill already applied (the open leg of a flip). The
        // fill was drawn once; its stochastic draws and queue verdict are not repeated.
        // `match_bar` computed every outcome for the bar before any was
        // applied, so an earlier one can have finished an order whose own
        // outcome is still queued (a bracket leg canceled by its sibling's
        // fill). Such an outcome describes a market the order has already
        // left: drop it. Expire, Cancel and Reject outcomes are the
        // matcher's report of a terminal state it already set, so they are
        // not stale.
        if let MatchOutcome::Trigger { order_id } | MatchOutcome::Fill { order_id, .. } = outcome {
            if self.orders.get(order_id).is_none_or(|o| o.status.is_terminal()) {
                return;
            }
        }
        // What happens after the bar's open (a fill away from the open price, an expiry, a
        // cancel) is later than the open a follow-up would be priced at.
        match outcome {
            MatchOutcome::Fill { price, .. } if !price.is_nan() && price != bar.open => {
                self.off_open_at = Some(bar.timestamp);
            }
            MatchOutcome::Expire { .. } | MatchOutcome::Cancel { .. } => {
                self.off_open_at = Some(bar.timestamp);
            }
            _ => {}
        }
        let (id, matched_price) = match outcome {
            MatchOutcome::Trigger { order_id } => {
                let client_id =
                    self.orders.get(order_id).map(|o| o.client_id.clone()).unwrap_or_default();
                events.push(EngineEvent::OrderTriggered { idx, order_id, client_id });
                return;
            }
            MatchOutcome::Expire { order_id } => {
                let client_id =
                    self.orders.get(order_id).map(|o| o.client_id.clone()).unwrap_or_default();
                events.push(EngineEvent::OrderExpired { idx, order_id, client_id });
                return;
            }
            MatchOutcome::Cancel { order_id } => {
                let client_id =
                    self.orders.get(order_id).map(|o| o.client_id.clone()).unwrap_or_default();
                events.push(EngineEvent::OrderCanceled { idx, order_id, client_id });
                return;
            }
            MatchOutcome::Reject { order_id, reason } => {
                let client_id =
                    self.orders.get(order_id).map(|o| o.client_id.clone()).unwrap_or_default();
                events.push(EngineEvent::OrderRejected { idx, order_id, client_id, reason });
                return;
            }
            MatchOutcome::Fill { order_id, price } => (order_id, price),
        };

        let Some(order) = self.orders.get(id) else { return };
        let side = order.side;
        let qty = order.qty;
        let kind = order.kind;
        let status = order.status;
        let client_id = order.client_id.clone();
        let stop_attach = order.stop_price;
        let target_attach = order.target_price;
        let reduce_only = order.reduce_only;
        let entry_intent = order.entry_intent;
        let target_position = order.position_id;
        let size_ref = order.size_ref;

        // A fill the lifecycle does not allow (an order that was never
        // acknowledged, one already finished) is refused before anything is
        // booked: a rejection the strategy sees, never an abort of the run.
        if !status.can_transition_to(OrderStatus::Filled)
            && !status.can_transition_to(OrderStatus::PartiallyFilled)
        {
            if let Some(order) = self.orders.get_mut(id) {
                let _ = order.transition(OrderStatus::Rejected);
            }
            events.push(EngineEvent::OrderRejected {
                idx,
                order_id: id,
                client_id,
                reason: "illegal_transition",
            });
            return;
        }

        // Stochastic fills: a marketable resting limit may be passed over
        // (queue position, exhausted liquidity); it stays working. Stop and
        // market fills may instead slip one tick against the trader.
        // Triggered stop-limit, if-touched-limit and trailing-stop-limit
        // orders rest as limits too; `MarketToLimit` fills at the open as a
        // market order and is not one.
        let is_limit_fill = matches!(kind, OrderKind::Limit { .. })
            || (matches!(
                kind,
                OrderKind::StopLimit { .. }
                    | OrderKind::LimitIfTouched { .. }
                    | OrderKind::TrailingStopLimit { .. }
            ) && matches!(status, OrderStatus::Triggered | OrderStatus::PartiallyFilled));
        let mut queue_granted = false;
        if is_limit_fill && !continuation {
            // The queue model reads the tape; it consumes no randomness, so
            // enabling it must not shift the RNG stream for other orders.
            let verdict = self.queue_verdict(id, kind, status, side, bar);
            match verdict {
                Some(QueueVerdict::Resting) => return,
                Some(_) => {
                    // The queue model earned this fill from volume observed
                    // trading ahead of the order, so it genuinely held the
                    // price: limit slippage would double-penalize it.
                    queue_granted = true;
                }
                None => {
                    if self.config.fill_prob_limit < 1.0
                        && self.fill_rng.next_f64() >= self.config.fill_prob_limit
                    {
                        return; // untouched: still Accepted/Triggered, retries next bar
                    }
                }
            }
        }
        // The queue grants the limit price: the order rested, and volume
        // traded ahead of it is the evidence it held that price. A print that
        // went further through the limit is no price improvement for a
        // resting order.
        let matched_price = match (queue_granted, kind) {
            (true, OrderKind::Limit { price }) => price,
            (true, OrderKind::StopLimit { price, .. }) => price,
            _ => matched_price,
        };
        let matched_price = if !is_limit_fill
            && !continuation
            && !matched_price.is_nan()
            && self.config.fill_prob_slippage > 0.0
            && self.fill_rng.next_f64() < self.config.fill_prob_slippage
        {
            match &self.spec {
                Some(spec) if spec.price_increment > 0.0 => match side {
                    OrderSide::Buy => matched_price + spec.price_increment,
                    OrderSide::Sell => matched_price - spec.price_increment,
                },
                _ => matched_price,
            }
        } else {
            matched_price
        };

        // A refused order must always be observable. `reject` counts against
        // `rejected_entries` so a discarded order can never look like an
        // order that was never placed; `reject_uncounted` is the deliberate
        // exception, for refusals that are not constraint decisions (sizing
        // arithmetic, which `open_at` already reports, and failed closes,
        // which are not refused *entries*).
        fn reject_uncounted(
            orders: &mut OrderEngine,
            events: &mut Vec<EngineEvent>,
            idx: usize,
            id: u64,
            client_id: &str,
            reason: &'static str,
        ) {
            if let Some(order) = orders.get_mut(id) {
                if order.flipped {
                    // The close leg of a flip filled and the rest cannot open: the order is
                    // done at what it closed, its quantity cut to that, so no order ends
                    // rejected with fills.
                    if let QtySpec::Units(from) = order.qty {
                        let to = order.filled_qty;
                        order.qty = QtySpec::Units(to);
                        let _ = order.transition(OrderStatus::Filled);
                        events.push(EngineEvent::OrderResized {
                            idx,
                            order_id: id,
                            client_id: client_id.to_string(),
                            from,
                            to,
                        });
                        return;
                    }
                }
                let _ = order.transition(OrderStatus::Rejected);
            }
            events.push(EngineEvent::OrderRejected {
                idx,
                order_id: id,
                client_id: client_id.to_string(),
                reason,
            });
        }
        fn reject(
            orders: &mut OrderEngine,
            risk: &mut RiskGate,
            events: &mut Vec<EngineEvent>,
            idx: usize,
            id: u64,
            client_id: &str,
            reason: &'static str,
        ) {
            risk.record_rejection();
            reject_uncounted(orders, events, idx, id, client_id, reason);
        }

        // An order's side is authoritative for opening. Independent
        // (hedging) policy: every order opens in its own side's direction.
        // Net policy: an order opposing the open position closes it — that
        // is how bracket legs and take-profit orders exit — but with no
        // position the order opens in its *own* side's direction. The
        // kernel's `direction` field governs the signal path only and is
        // never consulted here, so one run can hold long and short legs and
        // a leg can flip side across rebalances once it goes flat.
        //
        // A `reduce_only` order closes only when it opposes the open
        // position; otherwise it is refused, so a protective leg left
        // working after a stop already exited can never reverse into a
        // fresh position, and one on the position's own side never closes it.
        let order_direction = match side {
            OrderSide::Buy => Direction::Long,
            OrderSide::Sell => Direction::Short,
        };
        // Partial fills: on a print, a slice is at most the print's size.
        // Only explicit unit counts (and close-all) can be sliced; a
        // capital fraction resolves once, whole, as it always did.
        let slicing = self.config.partial_fills && self.stepping_trade && bar.volume > 0.0;
        let filled_so_far = self.orders.get(id).map(|o| o.filled_qty).unwrap_or(0.0);
        let slice_of = |remaining: f64| if slicing { remaining.min(bar.volume) } else { remaining };
        let hedging = self.ledger.policy() == PositionPolicy::Independent;
        let held = self.ledger.first().map(|m| m.position.direction);
        let (opens, open_direction) = if target_position.is_some() {
            // An order that names a position (a hedging close) closes it, whatever the policy.
            (false, order_direction)
        } else if hedging {
            (true, order_direction)
        } else if reduce_only {
            // Only an order opposing the open position reduces it. Any other
            // reduce-only order, including one with no position to reduce,
            // takes the `opens` path below and is refused.
            (!matches!(held, Some(dir) if dir != order_direction), order_direction)
        } else {
            match held {
                // Opposing an open position is a close.
                Some(dir) if dir != order_direction => (false, order_direction),
                // Agreeing with one: netting adds to it (below).
                Some(_) => (true, order_direction),
                // Flat: the order's own side decides the direction.
                None => (true, order_direction),
            }
        };

        if opens {
            if reduce_only {
                // A reduce-only order must never increase exposure: it is
                // refused when it does not oppose the open position (under
                // netting), and always under hedging, where every order opens.
                reject(
                    &mut self.orders,
                    &mut self.risk,
                    events,
                    idx,
                    id,
                    &client_id,
                    "reduce_only",
                );
                return;
            }
            if self.session_state.entries_blocked {
                reject(
                    &mut self.orders,
                    &mut self.risk,
                    events,
                    idx,
                    id,
                    &client_id,
                    RejectReason::Squareoff.as_str(),
                );
                return;
            }
            // Under netting the instrument has one position, and an order on its side adds
            // to it. Under hedging only a later slice of the order that opened a position
            // adds to it; any other order opens a position of its own.
            let adds_to = if hedging {
                self.ledger.first().filter(|m| m.entry_order_id == Some(id)).map(|m| m.id)
            } else {
                self.ledger.first().map(|m| m.id)
            };
            let add_price = if matched_price.is_nan() {
                self.fill_price_for(bar, open_direction, true)
            } else {
                matched_price
            };
            // `enter()` means "be in this position": with one held it is a no-op, never an add.
            if adds_to.is_some() && !hedging && entry_intent {
                reject_uncounted(&mut self.orders, events, idx, id, &client_id, "entry_held");
                return;
            }
            if adds_to.is_some() && filled_so_far == 0.0 && !hedging {
                // The first fill of an order that adds is gated like an entry. It adds to a
                // position that already counts, so the position-count limit does not bind.
                if self.margin.is_halted() {
                    reject(
                        &mut self.orders,
                        &mut self.risk,
                        events,
                        idx,
                        id,
                        &client_id,
                        "margin_call",
                    );
                    return;
                }
                if let Err(reason) = self.risk.check_entry(0) {
                    reject(
                        &mut self.orders,
                        &mut self.risk,
                        events,
                        idx,
                        id,
                        &client_id,
                        reason.as_str(),
                    );
                    return;
                }
                if let QtySpec::Units(whole) = qty {
                    if let Err(reason) = self.check_whole_order(open_direction, whole, add_price) {
                        reject(
                            &mut self.orders,
                            &mut self.risk,
                            events,
                            idx,
                            id,
                            &client_id,
                            reason.as_str(),
                        );
                        return;
                    }
                }
            }
            if let Some(position_id) = adds_to {
                let raw_price = add_price;
                let (u, slice) = match qty {
                    QtySpec::Units(u) => {
                        let remaining = u - filled_so_far;
                        let slice = slice_of(remaining);
                        if slice <= 0.0 {
                            return;
                        }
                        if matches!(self.orders.get(id).map(|o| o.tif), Some(TimeInForce::Fok))
                            && slice < remaining
                        {
                            self.cancel_remainder(idx, id, events);
                            return;
                        }
                        (u, slice)
                    }
                    // A fraction of the capital is sized once, whole, as an opening is.
                    QtySpec::CapitalFrac(f) => {
                        let size = self.capital_units(
                            bar,
                            open_direction,
                            raw_price,
                            f,
                            size_ref.filter(|_| !is_limit_fill),
                            is_limit_fill,
                        );
                        if size <= 0.0 {
                            reject_uncounted(
                                &mut self.orders,
                                events,
                                idx,
                                id,
                                &client_id,
                                RejectReason::ZeroSize.as_str(),
                            );
                            return;
                        }
                        (size, size)
                    }
                    QtySpec::FullPosition => {
                        reject(
                            &mut self.orders,
                            &mut self.risk,
                            events,
                            idx,
                            id,
                            &client_id,
                            "invalid_qty",
                        );
                        return;
                    }
                };
                match self.add_at(
                    idx,
                    bar,
                    position_id,
                    raw_price,
                    slice,
                    stop_attach,
                    target_attach,
                    is_limit_fill,
                ) {
                    Some((price, size)) => {
                        let entered =
                            EngineEvent::Entered { idx, price, size, direction: open_direction };
                        self.record_fill(idx, id, &client_id, price, size, u, entered, events);
                    }
                    None => reject(
                        &mut self.orders,
                        &mut self.risk,
                        events,
                        idx,
                        id,
                        &client_id,
                        "unfillable",
                    ),
                }
                return;
            }
            if self.margin.is_halted() {
                reject(
                    &mut self.orders,
                    &mut self.risk,
                    events,
                    idx,
                    id,
                    &client_id,
                    "margin_call",
                );
                return;
            }
            if let Err(reason) = self.risk.check_entry(self.gating_open_count()) {
                reject(
                    &mut self.orders,
                    &mut self.risk,
                    events,
                    idx,
                    id,
                    &client_id,
                    reason.as_str(),
                );
                return;
            }
            let raw_price = if matched_price.is_nan() {
                self.fill_price_for(bar, open_direction, true)
            } else {
                matched_price
            };
            let (size_mult, explicit_units) = match qty {
                QtySpec::Units(u) => {
                    let remaining = u - filled_so_far;
                    let slice = slice_of(remaining);
                    if slice <= 0.0 {
                        return;
                    }
                    if matches!(self.orders.get(id).map(|o| o.tif), Some(TimeInForce::Fok))
                        && slice < remaining
                    {
                        // Fill-or-kill: a print too small for the whole
                        // order fills none of it.
                        self.cancel_remainder(idx, id, events);
                        return;
                    }
                    (None, Some(slice))
                }
                QtySpec::CapitalFrac(f) => (Some(f), None),
                QtySpec::FullPosition => {
                    reject(
                        &mut self.orders,
                        &mut self.risk,
                        events,
                        idx,
                        id,
                        &client_id,
                        "invalid_qty",
                    );
                    return;
                }
            };
            if let QtySpec::Units(whole) = qty {
                if let Err(reason) = self.check_whole_order(open_direction, whole, raw_price) {
                    reject(
                        &mut self.orders,
                        &mut self.risk,
                        events,
                        idx,
                        id,
                        &client_id,
                        reason.as_str(),
                    );
                    return;
                }
            }
            self.sizing_ref = size_ref.filter(|_| size_mult.is_some() && !is_limit_fill);
            match self.open_at(
                idx,
                bar,
                open_direction,
                raw_price,
                size_mult,
                explicit_units,
                // The most recent bar's ATR, so an order-path open honors an
                // ATR stop/target config exactly as a signal entry does.
                // Passing 0.0 here silently yielded no stop at all.
                self.last_atr,
                stop_attach,
                target_attach,
                is_limit_fill,
            ) {
                Some(EngineEvent::Entered { price, size, direction, .. }) => {
                    // Remember which order opened it, so a later slice of the
                    // same order adds instead of being refused.
                    if let Some(opened) = self.ledger.first_mut() {
                        opened.entry_order_id = Some(id);
                    }
                    let requested = match qty {
                        QtySpec::Units(u) => u,
                        _ => size,
                    };
                    let entered = EngineEvent::Entered { idx, price, size, direction };
                    self.record_fill(idx, id, &client_id, price, size, requested, entered, events);
                }
                Some(EngineEvent::EntryRejected {
                    reason: RejectReason::InsufficientCapital | RejectReason::InsufficientMargin,
                    ..
                }) if self.hedge_pending && open_direction == Direction::Short => {
                    // A sold option whose margin group has a buying order still
                    // to fill: instrument order puts this leg's event before its
                    // hedge's, so the hedge cannot yet fund it. The order stays
                    // working and is funded when the hedge is held, one event
                    // later, instead of being refused for a credit it is about
                    // to have.
                }
                Some(EngineEvent::EntryRejected { reason, .. }) => {
                    // `open_at` already reported this; sizing arithmetic is
                    // not a constraint refusal, and the signal path does not
                    // count it either.
                    reject_uncounted(
                        &mut self.orders,
                        events,
                        idx,
                        id,
                        &client_id,
                        reason.as_str(),
                    );
                }
                _ => reject(
                    &mut self.orders,
                    &mut self.risk,
                    events,
                    idx,
                    id,
                    &client_id,
                    "unfillable",
                ),
            }
        } else {
            let target = match target_position {
                Some(position_id) => self.ledger.get(position_id),
                None => self.ledger.first(),
            };
            let Some(first) = target else {
                // Nothing to close. Closing is only chosen when a position is
                // held, so this is a defensive refusal, counted like any other.
                reject(
                    &mut self.orders,
                    &mut self.risk,
                    events,
                    idx,
                    id,
                    &client_id,
                    "no_position",
                );
                return;
            };
            if target_position.is_some() && first.position.direction == order_direction {
                // A named position is closed by the opposite side only.
                reject(
                    &mut self.orders,
                    &mut self.risk,
                    events,
                    idx,
                    id,
                    &client_id,
                    "reduce_only",
                );
                return;
            }
            let position_id = first.id;
            let direction = first.position.direction;
            let held = first.position.size;
            let raw_price = if matched_price.is_nan() {
                self.fill_price_for(bar, direction, false)
            } else {
                matched_price
            };
            // How much this order still wants. A reduce-only or named-position exit is capped
            // at what is held: it never reverses into a new position. Any other order against
            // a netting position flips it: it closes what is held and opens the rest on its
            // own side, as a broker's net position moves.
            let flips = !hedging
                && !reduce_only
                && target_position.is_none()
                && matches!(qty, QtySpec::Units(u) if u - filled_so_far > held * (1.0 + 1e-12));
            let wanted = match qty {
                QtySpec::Units(u) if flips => u - filled_so_far,
                QtySpec::Units(u) => (u - filled_so_far).min(held),
                _ => held,
            };
            let slice = slice_of(wanted);
            if slice <= 0.0 {
                return;
            }
            if matches!(self.orders.get(id).map(|o| o.tif), Some(TimeInForce::Fok))
                && slice < wanted
            {
                self.cancel_remainder(idx, id, events);
                return;
            }
            let slice = if flips { slice.min(held) } else { slice };
            // Taking a hedge off would leave a sold option the account's model cannot margin at
            // all (a loss without bound): the close is refused, as a broker's risk system
            // refuses an exit that leaves the account short of margin, rather than booked.
            if self.close_leaves_unmarginable(position_id, slice) {
                reject(
                    &mut self.orders,
                    &mut self.risk,
                    events,
                    idx,
                    id,
                    &client_id,
                    "insufficient_margin",
                );
                return;
            }
            match self.reduce_at(
                idx,
                bar,
                position_id,
                slice,
                raw_price,
                ExitReason::Order,
                is_limit_fill,
            ) {
                Some(EngineEvent::Exited { trade, .. }) if flips => {
                    // The close leg of a flip: the order stays open for the rest, which opens
                    // the other way. Without volume slicing the rest is applied now, as the
                    // same fill; with it, on the prints that follow.
                    let (price, size) = (trade.exit_price, trade.size);
                    let exited = EngineEvent::Exited { idx, trade };
                    self.record_leg(idx, id, &client_id, price, size, exited, events);
                    if slicing {
                        if matches!(self.orders.get(id).map(|o| o.tif), Some(TimeInForce::Ioc)) {
                            self.cancel_remainder(idx, id, events);
                        }
                    } else {
                        let rest = MatchOutcome::Fill { order_id: id, price: matched_price };
                        self.apply_match_outcome_inner(idx, bar, rest, events, true);
                    }
                }
                Some(EngineEvent::Exited { trade, .. }) => {
                    // The order is done once it has what it asked for, or
                    // once nothing is left to close; the surplus of an
                    // oversized exit is canceled, never left working.
                    let requested = match qty {
                        QtySpec::Units(u) => u.min(filled_so_far + held),
                        _ => held,
                    };
                    let (price, size) = (trade.exit_price, trade.size);
                    let exited = EngineEvent::Exited { idx, trade };
                    self.record_fill(idx, id, &client_id, price, size, requested, exited, events);
                }
                // A close that could not fill is not a refused *entry*, so it
                // stays out of the rejected-entries count.
                _ => reject_uncounted(&mut self.orders, events, idx, id, &client_id, "unfillable"),
            }
        }
    }

    /// Book a fill that is not the order's last: advance `filled_qty` and leave the order
    /// partly filled, with the position event right after the fill event. The close leg of a
    /// flip is one; the open leg that follows completes the order with [`Self::record_fill`].
    #[allow(clippy::too_many_arguments)]
    fn record_leg(
        &mut self,
        idx: usize,
        id: u64,
        client_id: &str,
        price: Price,
        size: f64,
        position: EngineEvent,
        events: &mut Vec<EngineEvent>,
    ) {
        if let Some(order) = self.orders.get_mut(id) {
            order.filled_qty += size;
            order.flipped = true;
            let _ = order.transition(OrderStatus::PartiallyFilled);
        }
        events.push(EngineEvent::OrderFilled {
            idx,
            order_id: id,
            client_id: client_id.to_string(),
            price,
            size,
        });
        events.push(position);
    }

    /// Book one fill slice on an order: advance `filled_qty`, move the order
    /// to `Filled` when it has `requested`, else to `PartiallyFilled` (an
    /// IOC order gives up its remainder instead). The fill event is
    /// followed IMMEDIATELY by the position event it caused (`position`) —
    /// that adjacency is what tells a consumer the position change was
    /// order-driven, not a second fill. Contingencies (`after_fill`) come
    /// after, and only on the completing slice, so a bracket's protective
    /// sibling keeps working while its partner is partly filled.
    #[allow(clippy::too_many_arguments)]
    fn record_fill(
        &mut self,
        idx: usize,
        id: u64,
        client_id: &str,
        price: Price,
        size: f64,
        requested: f64,
        position: EngineEvent,
        events: &mut Vec<EngineEvent>,
    ) {
        let mut done = true;
        let mut ioc_remainder = false;
        if let Some(order) = self.orders.get_mut(id) {
            order.filled_qty += size;
            // Float slop: a slice sized off the same arithmetic as the
            // request is complete at the request, not one ulp past it.
            done = order.filled_qty + 1e-9 >= requested;
            if done {
                let _ = order.transition(OrderStatus::Filled);
            } else {
                let _ = order.transition(OrderStatus::PartiallyFilled);
                ioc_remainder = order.tif == TimeInForce::Ioc;
            }
        }
        events.push(EngineEvent::OrderFilled {
            idx,
            order_id: id,
            client_id: client_id.to_string(),
            price,
            size,
        });
        events.push(position);
        if done {
            self.after_fill(idx, id, events);
        } else if ioc_remainder {
            self.cancel_remainder(idx, id, events);
        } else {
            self.shrink_oco_siblings(idx, id, requested, events);
        }
    }

    /// One-cancels-other on a partial slice: the legs are alternative ways
    /// out of the same quantity, so once `id` has filled part of
    /// `requested`, no sibling may still offer more than what is left, or a
    /// later fill of the sibling would trade quantity that is already gone.
    /// Each working sibling with an explicit unit size is cut to what it has
    /// filled plus the quantity remaining on `id`; it is never raised. A
    /// sibling sized by capital fraction or whole position resolves at its
    /// own fill and is left alone. The full fill of any leg still cancels
    /// the rest (`after_fill`). Each cut is reported as an
    /// [`EngineEvent::OrderResized`], so a change to a working order is
    /// never silent.
    fn shrink_oco_siblings(
        &mut self,
        idx: usize,
        id: u64,
        requested: f64,
        events: &mut Vec<EngineEvent>,
    ) {
        let Some((group, filled)) =
            self.orders.get(id).and_then(|o| Some((o.oco_group?, o.filled_qty)))
        else {
            return;
        };
        let remaining = (requested - filled).max(0.0);
        for sibling in self.orders.live_mut() {
            if sibling.oco_group != Some(group) || sibling.id == id || sibling.status.is_terminal()
            {
                continue;
            }
            if let QtySpec::Units(q) = sibling.qty {
                let cap = sibling.filled_qty + remaining;
                if q > cap {
                    sibling.qty = QtySpec::Units(cap);
                    events.push(EngineEvent::OrderResized {
                        idx,
                        order_id: sibling.id,
                        client_id: sibling.client_id.clone(),
                        from: q,
                        to: cap,
                    });
                }
            }
        }
    }

    /// Cancel what is left of an order (IOC after a partial slice, FOK on
    /// a print too small to fill whole).
    fn cancel_remainder(&mut self, idx: usize, id: u64, events: &mut Vec<EngineEvent>) {
        let Some(order) = self.orders.get_mut(id) else { return };
        let client_id = order.client_id.clone();
        if order.transition(OrderStatus::Canceled) {
            self.queue.forget(id);
            events.push(EngineEvent::OrderCanceled { idx, order_id: id, client_id });
        }
    }

    /// Contingency consequences of a fill: activate held one-triggers-other
    /// children, then cancel one-cancels-other siblings.
    fn after_fill(&mut self, idx: usize, filled_id: u64, events: &mut Vec<EngineEvent>) {
        let children: Vec<(u64, String)> = self
            .orders
            .working()
            .filter(|o| o.parent_id == Some(filled_id) && o.status == OrderStatus::Submitted)
            .map(|o| (o.id, o.client_id.clone()))
            .collect();
        for (child_id, client_id) in children {
            if let Some(order) = self.orders.get_mut(child_id) {
                let _ = order.transition(OrderStatus::Accepted);
            }
            events.push(EngineEvent::OrderAccepted { idx, order_id: child_id, client_id });
        }

        let group = self.orders.get(filled_id).and_then(|o| o.oco_group);
        if let Some(group) = group {
            let siblings: Vec<(u64, String)> = self
                .orders
                .working()
                .filter(|o| {
                    o.oco_group == Some(group) && o.id != filled_id && !o.status.is_terminal()
                })
                .map(|o| (o.id, o.client_id.clone()))
                .collect();
            for (sibling_id, client_id) in siblings {
                if self.orders.cancel(sibling_id) {
                    events.push(EngineEvent::OrderCanceled {
                        idx,
                        order_id: sibling_id,
                        client_id,
                    });
                }
            }
        }
    }
}
