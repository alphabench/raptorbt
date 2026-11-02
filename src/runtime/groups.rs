//! Account groups: several strategies on one account.
//!
//! A group owns the account its members trade on: the cash, the locked
//! margin and the halt. Each member keeps its own book, so its trades,
//! orders, positions and P&L stay its own; what the group adds is
//!
//! * one pool of cash and margin. Before a member is stepped its book is
//!   pointed at the group's pool (`PortfolioBook::set_pool`) and what the
//!   step moved is taken back, so "other kernels' locks" spans every member;
//! * the account-level checks, run once over all members after each member's
//!   step: the maintenance requirement, the daily loss, the drawdown halt,
//!   and, as the book is stepped, `max_positions` and `max_gross_exposure`
//!   counted over everything the group holds;
//! * halts that fan out: a margin call, a drawdown or daily-loss halt, or
//!   [`RuntimeCore::halt_group`](crate::runtime::runner::RuntimeCore::halt_group)
//!   latches on every member.
//!
//! # Order
//!
//! Members step in registration order within an event. That decides who gets
//! scarce cash: the first member to be stepped takes it. Groups do not touch
//! each other, so their relative order changes nothing.
//!
//! The account-level checks run after each member's step, on the marks every
//! member has so far, so a member stepped later in the same event is still on
//! its previous mark when an earlier member's check runs.
//!
//! # A member's equity
//!
//! A member's equity curve is *attributed*: the member starts with an equal
//! share of the account's starting cash, and its cash moves only by its own
//! entries, exits and fees. The account's curve is [`GroupOutcome`]. A group
//! of one is the same arithmetic as an isolated book.

use crate::accounts::{AccountMode, SharedAccount};
use crate::core::types::BacktestConfig;
use crate::portfolio::book::PortfolioBook;
use crate::portfolio::kernel::EngineEvent;
use crate::portfolio::session::{ScheduleData, ScheduleEntry};

/// What an account is made of.
#[derive(Debug, Clone, Copy)]
pub struct GroupSpec {
    pub initial_capital: f64,
    pub mode: AccountMode,
    pub max_drawdown_pct: Option<f64>,
    pub max_daily_loss: Option<f64>,
    pub max_daily_loss_pct: Option<f64>,
    pub max_gross_exposure: Option<f64>,
    pub max_positions: Option<usize>,
}

impl GroupSpec {
    /// `base` with the account's capital and account-level limits in place of
    /// its own. Every other setting is the strategy's.
    pub fn config(&self, base: &BacktestConfig) -> BacktestConfig {
        BacktestConfig {
            initial_capital: self.initial_capital,
            max_drawdown_pct: self.max_drawdown_pct,
            max_daily_loss: self.max_daily_loss,
            max_daily_loss_pct: self.max_daily_loss_pct,
            max_gross_exposure: self.max_gross_exposure,
            max_positions: self.max_positions,
            ..base.clone()
        }
    }
}

/// An account's final state.
#[derive(Debug, Clone)]
pub struct GroupOutcome {
    pub initial_capital: f64,
    /// Cash at the end, with every member's positions closed.
    pub balance: f64,
    /// Account equity, once per distinct timestamp.
    pub equity_curve: Vec<f64>,
    pub drawdown_curve: Vec<f64>,
    /// Locked margin at each equity sample.
    pub locked_curve: Vec<f64>,
    pub timestamps: Vec<i64>,
    pub halted: bool,
    pub halted_at: Option<usize>,
    /// Strategy ordinals of the members, in registration order.
    pub members: Vec<usize>,
}

/// One account and the strategies on it.
#[derive(Debug)]
pub struct AccountGroup {
    account: SharedAccount,
    spec: GroupSpec,
    flatten_on_daily_loss: bool,
    liquidate_on_margin_call: bool,
    /// The run's chosen margin model, when this account margins by it.
    margin_model: Option<std::sync::Arc<crate::accounts::MarginModel>>,
    members: Vec<usize>,
    equity_curve: Vec<f64>,
    drawdown_curve: Vec<f64>,
    locked_curve: Vec<f64>,
    timestamps: Vec<i64>,
    peak_equity: f64,
    peak_before_sample: f64,
    /// Events applied to any member; the `halted_at` ordinal of a group halt.
    applied: usize,
    /// Whether the drawdown limit has tripped.
    drawdown_latched: bool,
    curve_window: Option<usize>,
}

impl AccountGroup {
    pub fn new(spec: GroupSpec, base: &BacktestConfig) -> Self {
        Self {
            account: SharedAccount::new(
                spec.mode,
                spec.initial_capital,
                crate::accounts::MoneyScale::new(base.money_decimals)
                    .unwrap_or_else(|e| panic!("invalid BacktestConfig.money_decimals: {e}")),
            ),
            spec,
            flatten_on_daily_loss: base.flatten_on_daily_loss,
            liquidate_on_margin_call: base.liquidate_on_margin_call,
            margin_model: base.margin_model.clone(),
            members: Vec::new(),
            equity_curve: Vec::new(),
            drawdown_curve: Vec::new(),
            locked_curve: Vec::new(),
            timestamps: Vec::new(),
            peak_equity: spec.initial_capital,
            peak_before_sample: spec.initial_capital,
            applied: 0,
            drawdown_latched: false,
            curve_window: None,
        }
    }

    pub fn spec(&self) -> &GroupSpec {
        &self.spec
    }

    pub fn members(&self) -> &[usize] {
        &self.members
    }

    /// Add a strategy; the cash it is credited with is an equal share of the
    /// starting cash, settled once the group is complete (see
    /// [`AccountGroup::share`]).
    pub fn add_member(&mut self, strategy: usize) {
        self.members.push(strategy);
    }

    /// The part of the starting cash each member's curve starts from.
    pub fn share(&self) -> f64 {
        self.spec.initial_capital / self.members.len().max(1) as f64
    }

    pub fn set_curve_window(&mut self, window: Option<usize>) {
        self.curve_window = window.map(|w| w.max(1));
    }

    pub fn account(&self) -> &SharedAccount {
        &self.account
    }

    /// Heap held by the sampled curves.
    pub fn curve_heap_bytes(&self) -> usize {
        (self.equity_curve.capacity()
            + self.drawdown_curve.capacity()
            + self.locked_curve.capacity()
            + self.timestamps.capacity())
            * std::mem::size_of::<f64>()
    }

    /// Account equity: the pool's cash plus every live member's marks.
    pub fn equity(&self, books: &[Option<PortfolioBook>]) -> f64 {
        let marks: f64 =
            self.members.iter().filter_map(|&m| books[m].as_ref()).map(|b| b.marks_value()).sum();
        self.account.balance() + marks
    }

    /// Run `f` on a member with the group's pool, and take the pool back.
    fn with_member<R>(
        &mut self,
        books: &mut [Option<PortfolioBook>],
        member: usize,
        f: impl FnOnce(&mut PortfolioBook) -> R,
    ) -> Option<R> {
        let book = books[member].as_mut()?;
        book.set_pool(self.account.balance_money(), self.account.locked());
        let out = f(book);
        let (balance, locked) = book.pool();
        self.account.set_pool(balance, locked);
        Some(out)
    }

    /// Point every live member at the pool.
    fn sync(&self, books: &mut [Option<PortfolioBook>]) {
        for &m in &self.members {
            if let Some(book) = books[m].as_mut() {
                book.set_pool(self.account.balance_money(), self.account.locked());
            }
        }
    }

    /// Latch the account's halt on every member.
    fn halt_all(&mut self, books: &mut [Option<PortfolioBook>], margin_call: bool) {
        self.account.halt(self.applied);
        for &m in &self.members {
            if let Some(book) = books[m].as_mut() {
                book.latch_account_halt(self.applied);
                if margin_call {
                    book.halt_margin();
                }
            }
        }
    }

    /// Halt the account by hand: every member refuses new entries as
    /// `killed`, cancels its working orders and, with `flatten`, closes its
    /// positions when its next event is applied.
    pub fn halt(&mut self, books: &mut [Option<PortfolioBook>], reason: &str, flatten: bool) {
        self.account.halt(self.applied);
        for &m in &self.members {
            if let Some(book) = books[m].as_mut() {
                book.halt_for_account(reason, flatten, self.applied);
            }
        }
    }

    /// Hand `events` to a member other than the one being stepped, to be
    /// returned ahead of its next event.
    fn defer(books: &mut [Option<PortfolioBook>], member: usize, events: Vec<EngineEvent>) {
        if let Some(book) = books[member].as_mut() {
            for event in events {
                book.defer(event);
            }
        }
    }

    /// Step `member` with an event, then run the account-level checks.
    pub fn step(
        &mut self,
        books: &mut [Option<PortfolioBook>],
        member: usize,
        entry: &ScheduleEntry,
        depth: Option<crate::data::DepthTick>,
        input: crate::portfolio::kernel::StepInput,
    ) -> Vec<EngineEvent> {
        let (mut open, mut exposure) = (0usize, 0.0f64);
        for &m in &self.members {
            if m == member {
                continue;
            }
            if let Some(book) = books[m].as_ref() {
                if self.spec.max_positions.is_some() {
                    open += book.open_total();
                }
                if self.spec.max_gross_exposure.is_some() {
                    exposure += book.exposure_total();
                }
            }
        }
        let mut events = self
            .with_member(books, member, |book| {
                book.set_external(open, exposure);
                book.apply(entry, depth, input)
            })
            .expect("a stepped member's book is live");
        self.check(books, member, entry, &mut events);
        self.sync(books);
        events
    }

    /// The member's lifecycle before an event's hooks run; see
    /// [`PortfolioBook::begin_event`].
    pub fn begin_event(
        &mut self,
        books: &mut [Option<PortfolioBook>],
        member: usize,
        ts: crate::core::types::Timestamp,
        instrument: usize,
    ) -> Vec<EngineEvent> {
        let events = self
            .with_member(books, member, |book| book.begin_event(ts, instrument))
            .expect("a stepped member's book is live");
        self.sync(books);
        events
    }

    /// The account-level work after a member's step.
    fn check(
        &mut self,
        books: &mut [Option<PortfolioBook>],
        member: usize,
        entry: &ScheduleEntry,
        events: &mut Vec<EngineEvent>,
    ) {
        self.regroup_option_margin(books);

        // A kernel-local call sees only its own slice of the account, but the
        // account is shared: escalate it so every member halts.
        if events.iter().any(|e| matches!(e, EngineEvent::MarginCall { .. })) {
            self.halt_all(books, true);
        }

        if matches!(entry.data, ScheduleData::Quote(_) | ScheduleData::Depth(_)) {
            self.applied += 1;
            return;
        }

        let mut equity = self.equity(books);
        let ts = entry.timestamp();
        if self.timestamps.last() == Some(&ts) {
            self.equity_curve.pop();
            self.drawdown_curve.pop();
            self.locked_curve.pop();
            self.timestamps.pop();
            self.peak_equity = self.peak_before_sample;
        } else {
            self.peak_before_sample = self.peak_equity;
        }
        self.equity_curve.push(equity);
        if equity > self.peak_equity {
            self.peak_equity = equity;
        }
        self.drawdown_curve.push((self.peak_equity - equity) / self.peak_equity * 100.0);
        self.locked_curve.push(self.account.locked());
        self.timestamps.push(ts);
        if let Some(window) = self.curve_window.filter(|&w| self.equity_curve.len() >= 2 * w) {
            let drop = self.equity_curve.len() - window;
            self.equity_curve.drain(..drop);
            self.drawdown_curve.drain(..drop);
            self.locked_curve.drain(..drop);
            self.timestamps.drain(..drop);
        }

        // Maintenance: the requirement is the sum over every member.
        if matches!(self.account.mode(), AccountMode::Margin { .. }) && !self.account.is_halted() {
            let required: f64 = self
                .members
                .iter()
                .filter_map(|&m| books[m].as_ref())
                .map(|b| b.maintenance_total())
                .sum();
            if required > 0.0 && equity < required {
                self.halt_all(books, true);
                events.push(EngineEvent::MarginCall { idx: entry.local_idx, equity, required });
                if self.liquidate_on_margin_call {
                    for m in self.members.clone() {
                        let Some(closed) = self.with_member(books, m, |b| b.liquidate()) else {
                            continue;
                        };
                        if m == member {
                            events.extend(closed);
                        } else {
                            Self::defer(books, m, closed);
                        }
                    }
                }
            }
        }

        // The day's loss limit, on the account's equity. Without one the
        // kernels' gates ignore the feed, so it is not worth the walk.
        let daily = self.spec.max_daily_loss.is_some() || self.spec.max_daily_loss_pct.is_some();
        let day_blocked_before = daily
            && self.members.iter().filter_map(|&m| books[m].as_ref()).any(|b| b.any_day_blocked());
        let day = books[member].as_ref().and_then(|b| b.risk_day(entry.instrument, ts));
        let mut breached = false;
        if daily {
            for &m in &self.members {
                if let Some(book) = books[m].as_mut() {
                    breached |= book.observe_day(day, equity);
                }
            }
        }
        if breached && !day_blocked_before {
            let halted = |reason: &str| EngineEvent::RiskHalted {
                idx: entry.local_idx,
                reason: reason.to_string(),
            };
            events.push(halted("daily_loss"));
            for m in self.members.clone() {
                if m != member {
                    Self::defer(books, m, vec![halted("daily_loss")]);
                }
            }
            if self.flatten_on_daily_loss {
                for m in self.members.clone() {
                    let Some(closed) = self.with_member(books, m, |b| b.flatten()) else {
                        continue;
                    };
                    if m == member {
                        events.extend(closed);
                    } else {
                        Self::defer(books, m, closed);
                    }
                }
                // The sample above was taken before the closes paid their
                // costs; restate it so the curve shows the flattened account.
                equity = self.equity(books);
                self.restate_last_sample(equity);
                for &m in &self.members {
                    if let Some(book) = books[m].as_mut() {
                        if m == member || book.last_sample_ts() == Some(ts) {
                            book.restate();
                        }
                    }
                }
            }
        }

        // Every kernel sees the same account equity, so their drawdown gates
        // latch in lockstep; record the rising edge once. Without a limit, or
        // once it has tripped, the gates ignore the feed.
        if self.spec.max_drawdown_pct.is_some() && !self.drawdown_latched {
            let peak = self.peak_equity;
            for &m in &self.members {
                if let Some(book) = books[m].as_mut() {
                    book.observe_equity(equity, peak);
                }
            }
            let tripped = self
                .members
                .iter()
                .filter_map(|&m| books[m].as_ref())
                .any(|b| b.any_drawdown_halted());
            if tripped {
                self.drawdown_latched = true;
                self.halt_all(books, false);
            }
        }
        self.applied += 1;
    }

    fn restate_last_sample(&mut self, equity: f64) {
        let Some(last) = self.equity_curve.last_mut() else { return };
        *last = equity;
        let n = self.equity_curve.len();
        if equity > self.peak_equity {
            self.peak_equity = equity;
        }
        self.drawdown_curve[n - 1] = (self.peak_equity - equity) / self.peak_equity * 100.0;
        self.locked_curve[n - 1] = self.account.locked();
    }

    /// Re-price the groups of option legs that share an underlying and
    /// expiry across every member, as a book does across its own kernels.
    fn regroup_option_margin(&mut self, books: &mut [Option<PortfolioBook>]) {
        if matches!(self.account.mode(), AccountMode::Cash) {
            return;
        }
        if let Some(model) = self.margin_model.clone() {
            if model.groups_options() {
                self.regroup_modelled(books, &model);
            }
            return;
        }
        let mut groups = crate::portfolio::book::OptionLegs::new();
        let mut bases = Vec::with_capacity(self.members.len());
        let mut base = 0;
        for &m in &self.members {
            bases.push((base, m));
            if let Some(book) = books[m].as_ref() {
                book.collect_option_legs(base, &mut groups);
                base += book.kernel_count();
            }
        }
        let mut delta_locked = 0.0;
        for legs in groups.values() {
            let Some(requirement) = crate::portfolio::option_groups::group_requirement(legs) else {
                continue;
            };
            for (kernel, position_id, share) in
                crate::portfolio::option_groups::apportion(legs, requirement.total)
            {
                // The member whose kernels start at or before `kernel`.
                let &(start, member) =
                    bases.iter().rev().find(|(start, _)| *start <= kernel).expect("a base of 0");
                if let Some(book) = books[member].as_mut() {
                    delta_locked += book.set_share(kernel - start, position_id, share);
                }
            }
        }
        if delta_locked != 0.0 {
            self.account.reconcile(crate::accounts::Money::ZERO, delta_locked);
        }
    }

    /// [`AccountGroup::regroup_option_margin`] under a chosen margin model:
    /// the option lines of every member, margined together by the model.
    fn regroup_modelled(
        &mut self,
        books: &mut [Option<PortfolioBook>],
        model: &crate::accounts::MarginModel,
    ) {
        use crate::portfolio::book::{model_shares, ModelLines};
        let mut bases = Vec::with_capacity(self.members.len());
        let mut shares = Vec::new();
        {
            let mut groups = ModelLines::new();
            let mut now = 0;
            let mut base = 0;
            for &m in &self.members {
                bases.push((base, m));
                if let Some(book) = books[m].as_ref() {
                    book.collect_model_lines(base, model.groups_across_expiries(), &mut groups);
                    now = now.max(book.latest_time());
                    base += book.kernel_count();
                }
            }
            for lines in groups.values() {
                shares.extend(model_shares(model, lines, now));
            }
        }
        let mut delta_locked = 0.0;
        for (kernel, position_id, share) in shares {
            let &(start, member) =
                bases.iter().rev().find(|(start, _)| *start <= kernel).expect("a base of 0");
            if let Some(book) = books[member].as_mut() {
                delta_locked += book.set_share(kernel - start, position_id, share);
            }
        }
        if delta_locked != 0.0 {
            self.account.reconcile(crate::accounts::Money::ZERO, delta_locked);
        }
    }

    /// Settle the group once every member is finished.
    pub fn outcome(&mut self) -> GroupOutcome {
        if let Some(last) = self.equity_curve.last_mut() {
            *last = self.account.balance();
        }
        GroupOutcome {
            initial_capital: self.spec.initial_capital,
            balance: self.account.balance(),
            equity_curve: std::mem::take(&mut self.equity_curve),
            drawdown_curve: std::mem::take(&mut self.drawdown_curve),
            locked_curve: std::mem::take(&mut self.locked_curve),
            timestamps: std::mem::take(&mut self.timestamps),
            halted: self.account.is_halted(),
            halted_at: self.account.halted_at(),
            members: self.members.clone(),
        }
    }

    /// Run `f` on a member pointed at the pool, and take the pool back; every
    /// member is pointed at the pool afterwards.
    pub fn run_on<R>(
        &mut self,
        books: &mut [Option<PortfolioBook>],
        member: usize,
        f: impl FnOnce(&mut PortfolioBook) -> R,
    ) -> R {
        let out = self.with_member(books, member, f).expect("the member's book is live");
        self.sync(books);
        out
    }

    /// Fold a finishing member's closes into the pool: `finish` is called
    /// with the member pointed at the pool, and the pool is taken back.
    pub fn close_out(&mut self, books: &mut [Option<PortfolioBook>], member: usize) {
        self.with_member(books, member, |b| b.close_out());
    }
}

#[cfg(test)]
#[path = "groups_tests.rs"]
mod tests;
