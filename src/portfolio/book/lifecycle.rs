//! Expiry timers, physical delivery and corporate actions across a book's
//! instruments.
//!
//! The kernel settles a contract on the first bar of its own at or after the
//! expiry. What spans instruments lives here: a contract whose own bars have
//! ended settles when the book's next event notices the expiry, a physically
//! settled line delivers its underlying on the underlying's kernel, and a
//! corporate action is applied before any market data at its instant.
//!
//! Order at one timestamp `t` (see `docs/architecture/event-semantics.md`):
//! corporate actions with `ex <= t`, then dividends with `pay <= t`, then
//! contract expiries noticed at `t`, then the market event itself.

use super::PortfolioBook;
use crate::accounts::{AccountMode, Money};
use crate::core::error::{RaptorError, Result};
use crate::core::types::Timestamp;
use crate::instruments::lifecycle::{
    adjusted_dividend, AdjustmentPolicy, CorporateAction, CorporateKind, SettlementPrice,
    SettlementStyle,
};
use crate::portfolio::kernel::{EngineEvent, EngineKernel, KernelBar};

/// Whether the contract's settlement is priced from its underlying: physical
/// delivery, the `"underlying"` policy, or `"auto"` on an option (an index
/// option cash-settles at intrinsic value against the underlying's settlement
/// price, not at its own last trade).
fn wants_underlying_price(spec: &crate::instruments::InstrumentSpec) -> bool {
    spec.settlement.style == SettlementStyle::Physical
        || match spec.settlement.price {
            SettlementPrice::Underlying => true,
            SettlementPrice::Auto => {
                matches!(spec.kind, crate::instruments::InstrumentKind::Option { .. })
            }
            _ => false,
        }
}

/// Corporate actions a book was given and where it is in them.
#[derive(Debug, Default)]
pub(super) struct Lifecycle {
    pub policy: AdjustmentPolicy,
    /// This book's actions, in effective order (ties keep input order).
    pub actions: Vec<CorporateAction>,
    /// Every action given to the run, for restating dividends on the
    /// adjusted basis.
    pub all: Vec<CorporateAction>,
    pub next: usize,
    /// Dividends entitled and not yet paid: (pay instant, instrument,
    /// booked cash, per-unit amount).
    pub pending: Vec<(Timestamp, usize, Money, f64)>,
    /// Dividend cash credited so far (a short's debit is negative).
    pub dividends: Money,
}

impl PortfolioBook {
    /// Give the book the corporate actions of the run and the policy its
    /// history is on. Actions of instruments the book does not trade are
    /// ignored. Call before the first event.
    ///
    /// Under [`AdjustmentPolicy::Adjusted`] the bars are already on the
    /// final basis (see [`crate::instruments::lifecycle::adjust_bars`]): the
    /// book does not rescale positions again and restates a dividend's
    /// per-unit amount to that basis. Under [`AdjustmentPolicy::Raw`] the
    /// bars are as traded and a split or bonus rescales positions at the
    /// ex-date.
    pub fn set_corporate_actions(
        &mut self,
        actions: &[CorporateAction],
        policy: AdjustmentPolicy,
    ) -> Result<()> {
        if self.sampled > 0 {
            return Err(RaptorError::invalid_parameter(
                "corporate actions must be given before the first applied event",
            ));
        }
        if policy == AdjustmentPolicy::AsGiven && !actions.is_empty() {
            return Err(RaptorError::invalid_parameter(
                "corporate actions need an adjustment policy: \"raw\" or \"adjusted\"",
            ));
        }
        for a in actions {
            a.validate()?;
        }
        let mine: Vec<CorporateAction> = {
            let mut own: Vec<CorporateAction> = actions
                .iter()
                .filter(|a| self.kernels.iter().any(|k| k.symbol() == a.instrument))
                .cloned()
                .collect();
            own.sort_by_key(|a| a.ts);
            own
        };
        self.lifecycle =
            Lifecycle { policy, actions: mine, all: actions.to_vec(), ..Lifecycle::default() };
        Ok(())
    }

    /// Dividend cash credited so far, in booked minor units.
    pub fn dividends_money(&self) -> Money {
        self.lifecycle.dividends
    }

    /// Whether the lifecycle clock has anything to do.
    pub fn lifecycle_active(&self) -> bool {
        self.has_expiries
            || self.lifecycle.next < self.lifecycle.actions.len()
            || !self.lifecycle.pending.is_empty()
    }

    /// The adjustment policy the book's history is on.
    pub fn adjustment_policy(&self) -> AdjustmentPolicy {
        self.lifecycle.policy
    }

    /// Dividend cash credited so far, in the account's currency (a short's
    /// debit is negative).
    pub fn dividends_received(&self) -> f64 {
        self.account.scale().to_f64(self.lifecycle.dividends)
    }

    /// Advance the book's lifecycle clock to `ts` without a market event:
    /// corporate actions and dividends due, and contracts expired by `ts`
    /// settle. A live host calls it on its clock; a replay gets the same
    /// effect from the next event.
    pub fn advance_to(&mut self, ts: Timestamp) -> Vec<EngineEvent> {
        self.advance_lifecycle(ts, None)
    }

    /// The lifecycle of the event about to be handled, run before the
    /// strategy's hooks for it: a corporate action effective by `ts` has
    /// moved the position (and a dividend its cash) before the strategy sees
    /// a price on the new basis, and a contract expired by `ts` has settled.
    /// `instrument` is the event's own instrument, whose expiry the kernel
    /// settles on this event's bar. [`PortfolioBook::apply`] repeats the call,
    /// which then finds nothing left to do.
    pub fn begin_event(&mut self, ts: Timestamp, instrument: usize) -> Vec<EngineEvent> {
        if !self.lifecycle_active() {
            return Vec::new();
        }
        let events = self.advance_lifecycle(ts, Some(instrument));
        self.prepare_expiry(instrument, ts);
        events
    }

    /// Run the book's lifecycle up to `ts`, before the market event of
    /// `current` (the instrument whose own bar this is: its expiry is the
    /// kernel's to settle on that bar).
    pub(super) fn advance_lifecycle(
        &mut self,
        ts: Timestamp,
        current: Option<usize>,
    ) -> Vec<EngineEvent> {
        let mut events = Vec::new();
        // 1. Corporate actions, in effective order.
        while let Some(a) = self.lifecycle.actions.get(self.lifecycle.next).cloned() {
            if a.ts > ts {
                break;
            }
            self.lifecycle.next += 1;
            events.extend(self.apply_action(&a));
        }
        // 2. Dividends whose pay date has come.
        if !self.lifecycle.pending.is_empty() {
            self.lifecycle.pending.sort_by_key(|p| p.0);
            while self.lifecycle.pending.first().is_some_and(|p| p.0 <= ts) {
                let (_, i, cash, amount) = self.lifecycle.pending.remove(0);
                events.extend(self.credit_dividend(i, cash, amount));
            }
        }
        // 3. Contracts that expired with no later bar of their own.
        for k in 0..self.live_count() {
            let i = self.live_at(k);
            if Some(i) == current || !self.kernels[i].expired_at(ts) {
                continue;
            }
            if !self.kernels[i].has_contract_state() || self.waits_for_underlying(i, ts) {
                continue;
            }
            events.extend(self.settle_idle(i, ts));
        }
        events
    }

    /// Before the kernel of `i` steps a bar at `ts` that is at or past its
    /// expiry: say whether its owner delivers (the underlying is registered)
    /// and hand it the underlying's last price when its terms ask for one.
    pub(super) fn prepare_expiry(&mut self, i: usize, ts: Timestamp) {
        if !self.kernels[i].expired_at(ts) {
            return;
        }
        let wanted = self.kernels[i]
            .spec
            .as_ref()
            .map(|s| (wants_underlying_price(s), s.underlying().map(str::to_owned)));
        let Some((wants_underlying, underlying)) = wanted else { return };
        let u = underlying.as_deref().and_then(|name| self.index_of_symbol(name));
        let held = self.waits_for_underlying(i, ts);
        self.kernels[i].hold_settlement(held);
        self.kernels[i].set_delivers(u.is_some());
        // A held settlement takes no price yet: the underlying's last close is
        // the print BEFORE the expiry instant, and a price taken now would
        // outlive the wait and settle the contract on it (it depended on which
        // of the two instruments the run listed first).
        if wants_underlying && !held && !self.kernels[i].has_underlying_price() {
            if let Some(price) = u.and_then(|u| self.last_close[u]) {
                self.kernels[i].set_underlying_price(Some(price));
            }
        }
    }

    /// Whether contract `i`, expired at `ts`, settles on its underlying's
    /// price and that underlying has not printed at `ts` yet. The expiry is
    /// noticed once: the contract waits for the underlying's print at the
    /// instant it was first noticed ([`Self::settle_waiting`] releases it),
    /// and settles on the underlying's last price at the next later event.
    fn waits_for_underlying(&mut self, i: usize, ts: Timestamp) -> bool {
        let Some(spec) = self.kernels[i].spec.as_ref() else { return false };
        let wants = wants_underlying_price(spec);
        let Some(u) = spec.underlying().and_then(|n| self.index_of_symbol(n)) else {
            return false;
        };
        if !wants || self.last_seen[u].is_some_and(|(_, b)| b.timestamp >= ts) {
            return false;
        }
        match self.noticed[i] {
            None => {
                self.noticed[i] = Some(ts);
                true
            }
            Some(first) => first == ts,
        }
    }

    /// After instrument `u` printed at `ts`: settle the expired contracts on
    /// it that were waiting for that print, at the price just printed.
    pub(super) fn settle_waiting(&mut self, u: usize, ts: Timestamp) -> Vec<EngineEvent> {
        if !self.has_expiries {
            return Vec::new();
        }
        let symbol = self.kernels[u].symbol().to_owned();
        let mut events = Vec::new();
        for k in 0..self.live_count() {
            let i = self.live_at(k);
            let waiting = i != u
                && self.noticed[i] == Some(ts)
                && self.kernels[i].expired_at(ts)
                && self.kernels[i].has_contract_state()
                && self.kernels[i].spec.as_ref().and_then(|s| s.underlying()) == Some(&symbol);
            if waiting {
                events.extend(self.settle_idle(i, ts));
            }
        }
        events
    }

    fn index_of_symbol(&self, symbol: &str) -> Option<usize> {
        self.kernels.iter().position(|k| k.symbol() == symbol)
    }

    /// Step kernel `i` with `f` under the book's pool discipline: lend it
    /// the account's cash, take the movement back, fold it into the account.
    pub(super) fn with_pool<R>(&mut self, i: usize, f: impl FnOnce(&mut EngineKernel) -> R) -> R {
        let kernel = &mut self.kernels[i];
        let locked_before = kernel.locked_margin();
        let injected = match self.account.mode() {
            AccountMode::Cash => self.account.balance(),
            AccountMode::Margin { .. } => {
                self.account.balance() - (self.account.locked() - locked_before)
            }
        };
        let injected = kernel.lend_cash(injected);
        let out = f(kernel);
        let delta_cash = kernel.cash_money() - injected;
        let delta_locked = kernel.locked_margin() - locked_before;
        kernel.set_cash_money(Money::ZERO);
        self.settle(delta_cash, delta_locked);
        out
    }

    /// Keep what the book records of events it produced outside `apply`.
    fn record_lifecycle(&mut self, i: usize, events: &[EngineEvent]) {
        for event in events {
            if let EngineEvent::Exited { trade, .. } = event {
                self.streaming.update(trade.return_pct / 100.0);
                self.trades.push((**trade).clone());
            }
            self.order_logs[i].observe(&self.kernels[i], event);
        }
    }

    /// Settle contract `i`, whose expiry passed with no bar of its own to
    /// notice it, on the degenerate bar of its last price at `ts`.
    fn settle_idle(&mut self, i: usize, ts: Timestamp) -> Vec<EngineEvent> {
        let Some((idx, last)) = self.last_seen[i] else {
            // Never printed: there is no price to settle at, and an
            // unpriced settlement is refused, not guessed.
            return if self.kernels[i].ledger.is_in_position() {
                vec![EngineEvent::LifecycleRefused {
                    idx: 0,
                    what: "settlement",
                    reason: "no_price_seen".to_string(),
                }]
            } else {
                Vec::new()
            };
        };
        self.prepare_expiry(i, ts);
        let bar = KernelBar {
            timestamp: ts,
            open: last.close,
            high: last.close,
            low: last.close,
            close: last.close,
            volume: 0.0,
        };
        let mut events = self.with_pool(i, |k| k.settle_at(idx, &bar));
        self.record_lifecycle(i, &events);
        events.extend(self.deliver_contract_ends(i, ts));
        if !self.kernels[i].ledger.is_in_position() {
            // Settled: the contract is flat and holds nothing to mark.
            self.regroup_option_margin_if_alone();
        }
        events
    }

    /// Deliver what contract `i`'s last settlement ended, on the underlying's
    /// kernel. Returns the underlying's events.
    pub(super) fn deliver_contract_ends(&mut self, i: usize, ts: Timestamp) -> Vec<EngineEvent> {
        let ends = self.kernels[i].take_contract_ends();
        if ends.is_empty() {
            return Vec::new();
        }
        let underlying =
            self.kernels[i].spec.as_ref().and_then(|s| s.underlying()).map(str::to_owned);
        let Some(u) = underlying.as_deref().and_then(|n| self.index_of_symbol(n)) else {
            return vec![EngineEvent::LifecycleRefused {
                idx: 0,
                what: "settlement",
                reason: format!("underlying_not_registered: {}", underlying.unwrap_or_default()),
            }];
        };
        let multiplier = self.kernels[u].multiplier();
        let idx = self.last_seen[u].map_or(0, |(idx, _)| idx);
        let mut events = Vec::new();
        for end in ends {
            if end.deliver == 0.0 {
                continue;
            }
            let units = end.deliver / multiplier;
            let e = self.with_pool(u, |k| k.deliver(idx, ts, units, end.price));
            self.record_lifecycle(u, &e);
            events.extend(e);
        }
        events
    }

    pub(super) fn regroup_option_margin_if_alone(&mut self) {
        if self.grouped.is_none() {
            self.regroup_option_margin();
        }
    }

    // -- corporate actions ---------------------------------------------------

    /// Apply one action at its effective instant.
    fn apply_action(&mut self, a: &CorporateAction) -> Vec<EngineEvent> {
        let Some(i) = self.index_of_symbol(&a.instrument) else { return Vec::new() };
        let idx = self.last_seen[i].map_or(0, |(idx, _)| idx);
        match a.kind {
            CorporateKind::Split | CorporateKind::Bonus => {
                if self.lifecycle.policy == AdjustmentPolicy::Adjusted {
                    // History and trades are already on the final basis.
                    return vec![EngineEvent::CorporateAction {
                        idx,
                        kind: a.kind,
                        ratio: a.ratio,
                        amount: 0.0,
                        cash: 0.0,
                    }];
                }
                let whole = self.kernels[i].trades_whole_units();
                let mark = self.last_close[i];
                let result =
                    self.with_pool(i, |k| k.rescale_for_split(idx, a.ts, a.ratio, whole, mark));
                match result {
                    Err(reason) => vec![EngineEvent::LifecycleRefused {
                        idx,
                        what: "corporate_action",
                        reason: reason.to_string(),
                    }],
                    Ok((outcome, mut events)) => {
                        // Marks move onto the new basis with the position.
                        self.last_close[i] = self.last_close[i].map(|c| c / a.ratio);
                        if let Some((_, bar)) = self.last_seen[i].as_mut() {
                            bar.open /= a.ratio;
                            bar.high /= a.ratio;
                            bar.low /= a.ratio;
                            bar.close /= a.ratio;
                            bar.volume *= a.ratio;
                        }
                        self.record_lifecycle(i, &events);
                        events.push(EngineEvent::CorporateAction {
                            idx,
                            kind: a.kind,
                            ratio: a.ratio,
                            amount: 0.0,
                            cash: outcome.cash,
                        });
                        events
                    }
                }
            }
            CorporateKind::Dividend => {
                let amount = if self.lifecycle.policy == AdjustmentPolicy::Adjusted {
                    adjusted_dividend(&self.lifecycle.all, a)
                } else {
                    a.amount
                };
                let cash = self.kernels[i].dividend_cash(amount);
                match a.pay_ts {
                    Some(pay) if pay > a.ts => {
                        self.lifecycle.pending.push((pay, i, cash, amount));
                        Vec::new()
                    }
                    _ => self.credit_dividend(i, cash, amount),
                }
            }
        }
    }

    /// Credit a dividend's booked cash to the account.
    fn credit_dividend(&mut self, i: usize, cash: Money, amount: f64) -> Vec<EngineEvent> {
        self.with_pool(i, |k| k.credit_dividend(cash));
        self.lifecycle.dividends += cash;
        let idx = self.last_seen[i].map_or(0, |(idx, _)| idx);
        vec![EngineEvent::CorporateAction {
            idx,
            kind: CorporateKind::Dividend,
            ratio: 0.0,
            amount,
            cash: self.account.scale().to_f64(cash),
        }]
    }
}
