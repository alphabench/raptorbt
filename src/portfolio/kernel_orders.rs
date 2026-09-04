//! Order lifecycle on the kernel: submission, cancellation, modification,
//! and applying the matcher's outcomes.
//!
//! Split out of `kernel.rs` to keep that file reviewable; this is the same
//! `impl EngineKernel`, not a separate type.

use crate::core::types::{Direction, ExitReason, Price};
use crate::execution::algos::{AlgoError, ExecAlgorithm};
use crate::execution::orders::{
    MatchOutcome, Order, OrderEngine, OrderKind, OrderSide, OrderStatus, QtySpec, TimeInForce,
};
use crate::execution::queue::QueueVerdict;
use crate::portfolio::kernel::{EngineEvent, EngineKernel, KernelBar};
use crate::portfolio::ledger::PositionPolicy;
use crate::portfolio::risk::RiskGate;

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
        let id = self.orders.submit(order);

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
        self.orders.modify(id, qty, limit_price, trigger_price)
    }

    /// Shared view of an order by engine id.
    pub fn order(&self, id: u64) -> Option<&Order> {
        self.orders.get(id)
    }

    /// Instrument tick size; `0.0` without a spec.
    pub fn price_increment(&self) -> f64 {
        self.spec.as_ref().map(|s| s.price_increment).unwrap_or(0.0)
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

        // Stochastic fills: a marketable resting limit may be passed over
        // (queue position, exhausted liquidity); it stays working. Stop and
        // market fills may instead slip one tick against the trader.
        let is_limit_fill = matches!(kind, OrderKind::Limit { .. })
            || (matches!(kind, OrderKind::StopLimit { .. })
                && matches!(status, OrderStatus::Triggered | OrderStatus::PartiallyFilled));
        let mut queue_granted = false;
        if is_limit_fill {
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
        let matched_price = match (queue_granted, kind) {
            (true, OrderKind::Limit { price }) => price,
            (true, OrderKind::StopLimit { price, .. }) => price,
            _ => matched_price,
        };
        let matched_price = if !is_limit_fill
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
        // `reduce_only` short-circuits to the closing branch so a protective
        // leg left working after a stop already exited can never reverse
        // into a fresh position.
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
        let (opens, open_direction) = if hedging {
            (true, order_direction)
        } else if reduce_only {
            (false, order_direction)
        } else {
            match held {
                // Opposing an open position is a close.
                Some(dir) if dir != order_direction => (false, order_direction),
                // Agreeing with one: netting refuses to add (below).
                Some(_) => (true, order_direction),
                // Flat: the order's own side decides the direction.
                None => (true, order_direction),
            }
        };

        if opens {
            if reduce_only {
                // A reduce-only order must never increase exposure. Under
                // netting `reduce_only` already routed to the closing branch,
                // so this only fires for hedging, where every order opens.
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
            // A later slice of the order that opened the position adds to
            // it (partial fills); any other order is a second entry.
            let adds_to =
                self.ledger.first().filter(|m| m.entry_order_id == Some(id)).map(|m| m.id);
            if !hedging && self.ledger.is_in_position() && adds_to.is_none() {
                reject(
                    &mut self.orders,
                    &mut self.risk,
                    events,
                    idx,
                    id,
                    &client_id,
                    "position_open",
                );
                return;
            }
            if let (Some(position_id), QtySpec::Units(u)) = (adds_to, qty) {
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
                let raw_price = if matched_price.is_nan() {
                    self.fill_price_for(bar, open_direction, true)
                } else {
                    matched_price
                };
                match self.add_at(
                    idx,
                    bar,
                    position_id,
                    raw_price,
                    slice,
                    stop_attach,
                    target_attach,
                ) {
                    Some((price, size)) => {
                        self.record_fill(idx, id, &client_id, price, size, u, events);
                        events.push(EngineEvent::Entered {
                            idx,
                            price,
                            size,
                            direction: open_direction,
                        });
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
                    self.record_fill(idx, id, &client_id, price, size, requested, events);
                    events.push(EngineEvent::Entered { idx, price, size, direction });
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
            let Some(first) = self.ledger.first() else {
                // Nothing to close. A reduce-only order was routed here on
                // purpose and keeps reporting `reduce_only` (it must never
                // open, which is exactly what happened); any other order
                // wanted to open and was refused. Both are counted.
                let reason = if reduce_only { "reduce_only" } else { "no_position" };
                reject(&mut self.orders, &mut self.risk, events, idx, id, &client_id, reason);
                return;
            };
            let position_id = first.id;
            let direction = first.position.direction;
            let held = first.position.size;
            let raw_price = if matched_price.is_nan() {
                self.fill_price_for(bar, direction, false)
            } else {
                matched_price
            };
            // How much this order still wants, capped at what is held: an
            // exit larger than the position never reverses into a new one.
            let wanted = match qty {
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
            match self.reduce_at(idx, bar, position_id, slice, raw_price, ExitReason::Order) {
                Some(EngineEvent::Exited { trade, .. }) => {
                    // The order is done once it has what it asked for, or
                    // once nothing is left to close; the surplus of an
                    // oversized exit is canceled, never left working.
                    let requested = match qty {
                        QtySpec::Units(u) => u.min(filled_so_far + held),
                        _ => held,
                    };
                    self.record_fill(
                        idx,
                        id,
                        &client_id,
                        trade.exit_price,
                        trade.size,
                        requested,
                        events,
                    );
                    events.push(EngineEvent::Exited { idx, trade });
                }
                // A close that could not fill is not a refused *entry*, so it
                // stays out of the rejected-entries count.
                _ => reject_uncounted(&mut self.orders, events, idx, id, &client_id, "unfillable"),
            }
        }
    }

    /// Book one fill slice on an order: advance `filled_qty`, move the order
    /// to `Filled` when it has `requested`, else to `PartiallyFilled` (an
    /// IOC order gives up its remainder instead). Contingencies
    /// (`after_fill`) run only on the completing slice, so a bracket's
    /// protective sibling keeps working while its partner is partly filled.
    #[allow(clippy::too_many_arguments)]
    fn record_fill(
        &mut self,
        idx: usize,
        id: u64,
        client_id: &str,
        price: Price,
        size: f64,
        requested: f64,
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
        if done {
            self.after_fill(idx, id, events);
        } else if ioc_remainder {
            self.cancel_remainder(idx, id, events);
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
            .all()
            .iter()
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
                .all()
                .iter()
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
