//! Atomic order groups: legs that fill together or not at all.
//!
//! A multi-leg structure (a sold call and the long call that hedges it) is
//! submitted as ordinary orders on separate instruments, then linked with
//! [`PortfolioBook::link_group`]. From then on no leg fills alone:
//!
//! * a leg the matcher finds marketable is held back by its kernel;
//! * on the event where every leg is held back at the same instant, the group
//!   is funded as a whole (the margin model's requirement for the legs
//!   together, less what the group already locks, plus the entry costs; with
//!   no model, the legs' own funding summed) against the account's free
//!   capital;
//! * if it fits, the legs fill through the ordinary fill path, bought legs
//!   first and then by instrument index, so the order of legs in the request
//!   changes nothing and a sold leg is funded with its hedge already held;
//! * if it does not fit, every leg is rejected as `group_unfunded` and no
//!   position opens;
//! * a leg that ends without filling (cancelled, expired, rejected) takes
//!   the others with it, rejected as `group_leg_failed`.
//!
//! The legs must be sized in units. A venue-managed instrument cannot be in
//! a group: its legs are separate orders at the venue, which cannot fill
//! them atomically, and placing them hedge-first would still leave a naked
//! sold leg if the venue refuses the hedge's partner. It is refused at
//! `link_group`.

use super::PortfolioBook;
use crate::accounts::{AccountMode, LineView};
use crate::core::types::Timestamp;
use crate::execution::orders::OrderSide;
use crate::portfolio::kernel::{EngineEvent, StagedFill};
use crate::portfolio::kernel_orders::GroupLegCalc;

/// One linked group (its id is its place in the book's order): its legs as `(instrument, order id)`, by instrument.
#[derive(Debug, Clone)]
pub(super) struct AtomicGroup {
    pub legs: Vec<(usize, u64)>,
}

impl PortfolioBook {
    /// Link working orders on distinct instruments into one atomic group and
    /// return its id. Refused (nothing linked) for fewer than two legs, two
    /// legs on one instrument, an unknown or finished or reduce-only order, a
    /// venue-managed instrument, or an order not sized in units.
    pub fn link_group(&mut self, legs: &[(usize, u64)]) -> Result<u64, String> {
        if legs.len() < 2 {
            return Err("an atomic group needs at least two legs".to_string());
        }
        let mut seen = Vec::new();
        for &(k, id) in legs {
            if k >= self.kernels.len() {
                return Err(format!("unknown instrument index {k}"));
            }
            if seen.contains(&k) {
                return Err(format!(
                    "{} has two legs; give one leg the whole size",
                    self.kernels[k].symbol()
                ));
            }
            seen.push(k);
            let kernel = &self.kernels[k];
            if kernel.venue_managed {
                return Err(format!(
                    "{} is managed by a venue: its legs are separate venue orders and cannot fill atomically",
                    kernel.symbol()
                ));
            }
            let sized = kernel
                .order(id)
                .is_some_and(|o| matches!(o.qty, crate::execution::orders::QtySpec::Units(_)));
            if !sized {
                return Err(format!("order {id} on {} is not sized in units", kernel.symbol()));
            }
        }
        let group = self.atomic_next;
        let mut tagged = Vec::new();
        for &(k, id) in legs {
            if self.kernels[k].tag_group(id, group) {
                tagged.push((k, id));
            } else {
                for (tk, tid) in tagged {
                    self.kernels[tk].untag_group(tid);
                }
                return Err(format!(
                    "order {id} on {} is not a working opening order",
                    self.kernels[k].symbol()
                ));
            }
        }
        self.atomic_next += 1;
        let mut legs = legs.to_vec();
        legs.sort_by_key(|&(k, _)| k);
        self.atomic.push(super::atomic::AtomicGroup { legs });
        Ok(group)
    }

    /// Whether any group is still waiting to fill.
    pub fn has_atomic_groups(&self) -> bool {
        !self.atomic.is_empty()
    }

    /// Settle the groups `instrument`'s event at `ts` touched: refuse the
    /// ones with a dead leg, fill or refuse the ones every leg of which is
    /// held back at `ts`. Returns the events of the legs.
    pub(super) fn advance_atomic(&mut self, instrument: usize, ts: Timestamp) -> Vec<EngineEvent> {
        let mut events = Vec::new();
        let mut i = 0;
        while i < self.atomic.len() {
            if !self.atomic[i].legs.iter().any(|&(k, _)| k == instrument) {
                i += 1;
                continue;
            }
            let legs = self.atomic[i].legs.clone();
            let dead = legs
                .iter()
                .any(|&(k, id)| self.kernels[k].order(id).is_none_or(|o| o.status.is_terminal()));
            if dead {
                self.atomic.remove(i);
                events.extend(self.refuse_legs(&legs, "group_leg_failed"));
                continue;
            }
            let staged: Vec<Option<StagedFill>> =
                legs.iter().map(|&(k, id)| self.kernels[k].staged_at(id, ts)).collect();
            if staged.iter().any(Option::is_none) {
                i += 1;
                continue;
            }
            let staged: Vec<StagedFill> = staged.into_iter().flatten().collect();
            self.atomic.remove(i);
            if self.atomic_fits(&legs, &staged, ts) {
                events.extend(self.fill_legs(&legs, &staged));
            } else {
                if let Some(&(k, _)) = legs.first() {
                    self.unfunded[k] += 1;
                }
                events.extend(self.refuse_legs(&legs, "group_unfunded"));
            }
        }
        events
    }

    /// Reject every still-working leg, so no leg of a refused group lingers.
    fn refuse_legs(&mut self, legs: &[(usize, u64)], reason: &'static str) -> Vec<EngineEvent> {
        let mut events = Vec::new();
        for &(k, id) in legs {
            let idx = self.last_seen[k].map_or(0, |(idx, _)| idx);
            let e = self.kernels[k].reject_group_order(idx, id, reason);
            self.book_events(k, &e);
            events.extend(e);
        }
        events
    }

    /// Whether the legs fit the account together.
    fn atomic_fits(&self, legs: &[(usize, u64)], staged: &[StagedFill], ts: Timestamp) -> bool {
        let calcs: Option<Vec<GroupLegCalc>> =
            legs.iter().zip(staged).map(|(&(k, _), s)| self.kernels[k].group_leg_calc(s)).collect();
        let Some(calcs) = calcs else { return false };
        let fees: f64 = calcs.iter().map(|c| c.fees).sum();
        let model = self
            .config
            .margin_model
            .as_deref()
            .filter(|m| m.groups_options())
            .filter(|_| matches!(self.account.mode(), AccountMode::Margin { .. }));
        let Some(model) = model else {
            // No model that groups: each leg is funded as it would be alone.
            if calcs.iter().any(|c| !c.priceable) {
                return false;
            }
            let need: f64 = calcs.iter().map(|c| c.funded).sum::<f64>() + fees;
            return need <= self.account.free();
        };
        let Some(key) = legs.iter().find_map(|&(k, _)| self.kernels[k].option_group_key()) else {
            return false;
        };
        let across = model.groups_across_expiries();
        let mut held: Vec<(&crate::instruments::InstrumentSpec, f64, f64)> = Vec::new();
        let mut locked = 0.0;
        for kernel in &self.kernels {
            let Some((underlying, expiry)) = kernel.option_group_key() else { continue };
            if underlying != key.0 || (!across && expiry != key.1) {
                continue;
            }
            locked += kernel.locked_margin();
            if let Some((spec, lines)) = kernel.model_option_lines() {
                held.extend(lines.into_iter().map(|(_, quantity, price)| (spec, quantity, price)));
            }
        }
        let mut views: Vec<LineView<'_>> = held
            .iter()
            .map(|&(spec, quantity, price)| LineView::new(spec, quantity, price))
            .collect();
        for (&(k, _), c) in legs.iter().zip(&calcs) {
            let Some(spec) = self.kernels[k].spec.as_ref() else { return false };
            let sign = if c.direction == crate::core::types::Direction::Long { 1.0 } else { -1.0 };
            views.push(LineView::new(spec, sign * c.size, c.price));
        }
        let Ok(required) = model.group_required(&views, ts) else { return false };
        (required - locked).max(0.0) + fees <= self.account.free()
    }

    /// Fill the legs through the ordinary fill path: bought legs first.
    fn fill_legs(&mut self, legs: &[(usize, u64)], staged: &[StagedFill]) -> Vec<EngineEvent> {
        let mut order: Vec<(bool, usize, StagedFill)> = legs
            .iter()
            .zip(staged)
            .map(|(&(k, id), s)| {
                let sells = self.kernels[k].order(id).is_some_and(|o| o.side == OrderSide::Sell);
                (sells, k, *s)
            })
            .collect();
        order.sort_by_key(|&(sells, k, _)| (sells, k));
        let mut events = Vec::new();
        for (_, k, s) in order {
            let ts = s.bar.timestamp;
            let (open, exposure) = self.external_limits(k);
            let context = self.group_context_for(k, ts);
            let released = self.with_pool(k, |kernel| {
                kernel.set_external_open_count(open);
                kernel.set_external_exposure(exposure);
                kernel.set_group_context(context);
                let out = kernel.release_staged(s);
                kernel.set_external_open_count(None);
                kernel.set_external_exposure(None);
                kernel.set_group_context(None);
                out
            });
            self.book_events(k, &released);
            events.extend(released);
        }
        self.regroup_option_margin_if_alone();
        events
    }

    /// The portfolio-wide counts the kernel's limits read, as `apply` hands
    /// them over for a normal step.
    fn external_limits(&self, instrument: usize) -> (Option<usize>, Option<f64>) {
        let open = self
            .config
            .max_positions
            .map(|_| self.live_open_total() + self.grouped.map_or(0, |g| g.external_open));
        let exposure = self.kernels[instrument].tracks_exposure().then(|| {
            let own = self
                .kernels
                .iter()
                .zip(&self.last_close)
                .enumerate()
                .filter(|(i, _)| *i != instrument)
                .map(|(_, (k, close))| close.map(|c| k.gross_notional(c)).unwrap_or(0.0))
                .sum::<f64>();
            own + self.grouped.map_or(0.0, |g| g.external_exposure)
        });
        (open, exposure)
    }

    /// Keep what the book records of events a kernel produced outside the
    /// ordinary step: trades, refused entries, the order log.
    fn book_events(&mut self, k: usize, events: &[EngineEvent]) {
        for event in events {
            if let EngineEvent::Exited { trade, .. } = event {
                self.streaming.update(trade.return_pct / 100.0);
                self.trades.push((**trade).clone());
            }
            if matches!(event, EngineEvent::EntryRejected { reason, .. } if reason.is_unfunded()) {
                self.unfunded[k] += 1;
            }
            self.order_logs[k].observe(&self.kernels[k], event);
        }
    }
}
