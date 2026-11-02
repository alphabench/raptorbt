//! What an account group needs from a book.
//!
//! A group (see `crate::runtime::groups`) owns the account its members trade
//! on and runs the account-level checks over all of them. It reaches into a
//! member only through the methods here, so the book's own bookkeeping stays
//! in `book.rs`.

use std::collections::BTreeMap;

use super::{Grouped, PortfolioBook};
use crate::accounts::{LineView, MarginModel, Money};
use crate::core::types::Timestamp;
use crate::instruments::InstrumentSpec;
use crate::portfolio::kernel::EngineEvent;
use crate::portfolio::option_groups::OptionLeg;

/// Option legs by underlying and expiry.
pub type OptionLegs = BTreeMap<(String, Option<Timestamp>), Vec<OptionLeg>>;

/// One open option position as a line of a chosen margin model.
#[derive(Debug, Clone, Copy)]
pub struct ModelLine<'a> {
    /// The kernel's index in the book (or in its group).
    pub kernel: usize,
    pub position_id: u64,
    pub spec: &'a InstrumentSpec,
    /// Signed: negative is short.
    pub quantity: f64,
    pub price: f64,
}

/// Model lines by underlying and, unless the model groups across expiries,
/// expiry.
pub type ModelLines<'a> = BTreeMap<(String, Option<Timestamp>), Vec<ModelLine<'a>>>;

/// What each line of one group locks under `model` at `now`:
/// `(kernel, position id, share)`. This is `MarginModel::group_shares`, so the
/// locks of a structure add up to what `MarginModel.required_legs` says for
/// its legs. A group the model cannot price keeps the locks its lines took
/// when they entered (each priced alone by the same model), so nothing here
/// invents a requirement.
pub fn model_shares(
    model: &MarginModel,
    lines: &[ModelLine<'_>],
    now: Timestamp,
) -> Vec<(usize, u64, f64)> {
    let views: Vec<LineView<'_>> =
        lines.iter().map(|l| LineView::new(l.spec, l.quantity, l.price)).collect();
    match model.group_shares(&views, now) {
        Ok(shares) => lines.iter().zip(shares).map(|(l, s)| (l.kernel, l.position_id, s)).collect(),
        Err(_) => Vec::new(),
    }
}

impl PortfolioBook {
    /// Put the book in a group. `share` is the part of the account's
    /// starting cash the strategy is credited with: its equity curve starts
    /// there. Call before the first event.
    pub fn join_group(&mut self, share: f64) {
        let cash =
            self.account.scale().money(share).unwrap_or_else(|e| {
                crate::accounts::money::money_fault(format!("account share: {e}"))
            });
        self.grouped =
            Some(Grouped { cash, start: share, external_open: 0, external_exposure: 0.0 });
        self.peak_equity = share;
        self.peak_before_sample = share;
    }

    /// The account's cash and locked margin, as the book last saw them.
    pub fn pool(&self) -> (crate::accounts::Money, f64) {
        (self.account.balance_money(), self.account.locked())
    }

    /// Point the book at the group's cash and locked margin.
    pub fn set_pool(&mut self, balance: crate::accounts::Money, locked: f64) {
        self.account.set_pool(balance, locked);
    }

    /// What the group's other members hold, for the account-level
    /// `max_positions` and `max_gross_exposure` the next step checks.
    pub fn set_external(&mut self, open: usize, exposure: f64) {
        if let Some(g) = &mut self.grouped {
            g.external_open = open;
            g.external_exposure = exposure;
        }
    }

    /// Positions open across the book's instruments.
    pub fn open_total(&self) -> usize {
        self.kernels.iter().map(|k| k.open_count()).sum()
    }

    /// Gross notional held at the last marks, for kernels that track it.
    pub fn exposure_total(&self) -> f64 {
        self.kernels
            .iter()
            .zip(&self.last_close)
            .map(|(k, close)| close.map(|c| k.gross_notional(c)).unwrap_or(0.0))
            .sum()
    }

    /// The marks of the open positions, which the account's cash is added to
    /// for the account's equity.
    pub fn marks_value(&self) -> f64 {
        self.positions_value()
    }

    /// Summed maintenance requirement of the open positions.
    pub fn maintenance_total(&self) -> f64 {
        self.kernels
            .iter()
            .zip(&self.last_close)
            .map(|(k, close)| close.map(|c| k.maintenance_requirement(c)).unwrap_or(0.0))
            .sum()
    }

    /// Queue an event to be returned ahead of the book's next one.
    pub fn defer(&mut self, event: EngineEvent) {
        self.deferred.push(event);
    }

    /// Latch the account's halt on this book's copy of the account.
    pub fn latch_account_halt(&mut self, idx: usize) {
        self.account.halt(idx);
    }

    /// Trip every kernel's margin kill-switch.
    pub fn halt_margin(&mut self) {
        for kernel in &mut self.kernels {
            kernel.halt_margin();
        }
    }

    /// Halt the strategy as an account halt does: new entries refused as
    /// `killed`, working orders cancelled and, with `flatten`, positions
    /// closed when the next event is applied. `idx` is what `halted_at`
    /// reports.
    pub fn halt_for_account(&mut self, reason: &str, flatten: bool, idx: usize) {
        self.halt_at(reason, flatten, idx);
    }

    /// The daily-loss day of an event on `instrument` at `ts`.
    pub fn risk_day(&self, instrument: usize, ts: Timestamp) -> Option<i64> {
        self.kernels[instrument].risk_day(ts)
    }

    pub fn any_day_blocked(&self) -> bool {
        self.kernels.iter().any(|k| k.is_day_blocked())
    }

    /// Feed the account's equity to every kernel's daily-loss gate; true if
    /// any breached.
    pub fn observe_day(&mut self, day: Option<i64>, equity: f64) -> bool {
        let mut breached = false;
        for kernel in &mut self.kernels {
            breached |= kernel.observe_day_in(day, equity);
        }
        breached
    }

    pub fn any_drawdown_halted(&self) -> bool {
        self.kernels.iter().any(|k| k.drawdown_halted())
    }

    /// Feed the account's equity and peak to every kernel's drawdown gate.
    pub fn observe_equity(&mut self, equity: f64, peak: f64) {
        for kernel in &mut self.kernels {
            kernel.observe_equity(equity, peak);
        }
    }

    /// Close everything, as a halt with `flatten` does.
    pub fn flatten(&mut self) -> Vec<EngineEvent> {
        self.cleanup_all(false, true)
    }

    /// Force-close every instrument at its last print and fold the cash into
    /// the account, as the first half of finishing does. A second call does
    /// nothing.
    pub fn close_out(&mut self) {
        self.close_out_impl();
    }

    /// Close everything at the last marks after a group margin call.
    pub fn liquidate(&mut self) -> Vec<EngineEvent> {
        self.liquidate_marked()
    }

    /// Restate the book's last sample after a change within the same event.
    pub fn restate(&mut self) {
        let equity = self.equity();
        self.restate_last_sample(equity);
    }

    /// Timestamp of the book's last sample.
    pub fn last_sample_ts(&self) -> Option<i64> {
        self.timestamps.last().copied()
    }

    /// Whether the account-level flatten-on-daily-loss is configured.
    pub fn flatten_on_daily_loss(&self) -> bool {
        self.config.flatten_on_daily_loss
    }

    pub fn liquidate_on_margin_call(&self) -> bool {
        self.config.liquidate_on_margin_call
    }

    /// Add this book's open option legs to `groups`, numbering its kernels
    /// from `base` so that legs of different books never share an index.
    pub fn collect_option_legs(&self, base: usize, groups: &mut OptionLegs) {
        if matches!(self.account.mode(), crate::accounts::AccountMode::Cash) {
            return;
        }
        for (index, kernel) in self.kernels.iter().enumerate() {
            let legs = kernel.open_option_legs(base + index);
            if legs.is_empty() {
                continue;
            }
            let Some(key) = kernel.option_group_key() else { continue };
            groups.entry(key).or_default().extend(legs);
        }
    }

    /// The book's option positions as lines of its margin model, by group.
    /// `across` groups by underlying alone (a model that credits calendars).
    pub fn collect_model_lines<'a>(
        &'a self,
        base: usize,
        across: bool,
        groups: &mut ModelLines<'a>,
    ) {
        for (index, kernel) in self.kernels.iter().enumerate() {
            let Some((spec, lines)) = kernel.model_option_lines() else { continue };
            let Some((underlying, expiry)) = kernel.option_group_key() else { continue };
            let key = (underlying, if across { None } else { expiry });
            for (position_id, quantity, price) in lines {
                groups.entry(key.clone()).or_default().push(ModelLine {
                    kernel: base + index,
                    position_id,
                    spec,
                    quantity,
                    price,
                });
            }
        }
    }

    /// The latest bar time any of the book's instruments has been stepped on.
    pub fn latest_time(&self) -> Timestamp {
        self.last_seen.iter().flatten().map(|(_, bar)| bar.timestamp).max().unwrap_or(0)
    }

    /// Re-lock the book's option lines by its chosen model's group shares.
    pub(super) fn regroup_modelled(&mut self, model: &MarginModel) {
        let now = self.latest_time();
        let mut shares = Vec::new();
        {
            let mut groups = ModelLines::new();
            self.collect_model_lines(0, model.groups_across_expiries(), &mut groups);
            for lines in groups.values() {
                shares.extend(model_shares(model, lines, now));
            }
        }
        let mut delta_locked = 0.0;
        for (kernel, position_id, share) in shares {
            delta_locked += self.kernels[kernel].set_locked_margin(position_id, share);
        }
        if delta_locked != 0.0 {
            self.account.reconcile(Money::ZERO, delta_locked);
        }
    }

    /// Set one position's locked margin; returns the change.
    pub fn set_share(&mut self, kernel: usize, position_id: u64, share: f64) -> f64 {
        self.kernels[kernel].set_locked_margin(position_id, share)
    }

    /// Number of kernels, i.e. instruments.
    pub fn kernel_count(&self) -> usize {
        self.kernels.len()
    }
}
