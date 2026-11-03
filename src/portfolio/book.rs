//! Per-strategy portfolio state, separate from the market tape.
//!
//! A [`PortfolioBook`] is everything one strategy owns: a kernel per
//! instrument, the shared account, the sampled equity curves, the trade and
//! order records, and the portfolio-wide halt. It holds no market data. The
//! owner (an [`EventSession`](crate::portfolio::session::EventSession), or a
//! runtime driving many books over one tape) hands it each schedule entry
//! together with the depth snapshot that entry refers to, so any number of
//! books can consume the same tape and each produces exactly what a
//! standalone session with the same intents would.
//!
//! Instruments are addressed by the tape's instrument index; a book must
//! register them in the tape's order.
//!
//! Besides the kernels and the account, the only per-instrument state a book
//! keeps is the strategy's valuation marks (`last_close`, `last_seen`): they
//! record the last print *this book has stepped*, which is what its equity
//! and its end-of-run liquidation are priced from.

use std::collections::BTreeMap;

use crate::accounts::{AccountMode, Money, SharedAccount};
use crate::core::types::{
    BacktestConfig, BacktestResult, Direction, InstrumentConfig, Timestamp, Trade,
};
use crate::data::DepthTick;
use crate::instruments::InstrumentSpec;
use crate::metrics::streaming::StreamingMetrics;
use crate::portfolio::engine::compute_backtest_metrics_with_config;
use crate::portfolio::kernel::{Adoption, EngineEvent, EngineKernel, KernelBar, StepInput};
use crate::portfolio::ledger::PositionPolicy;
use crate::portfolio::option_groups::{apportion, group_requirement, OptionLeg};
use crate::portfolio::runner::OrderLog;
use crate::portfolio::session::{InstrumentOutcome, ScheduleData, ScheduleEntry, SessionOutcome};

mod venue;

/// Why a portfolio-wide halt latched.
#[derive(Debug, Clone, Copy)]
enum HaltCause {
    /// Equity fell below the summed maintenance requirement.
    MarginCall,
    /// The drawdown kill-switch tripped on portfolio equity.
    Drawdown,
    /// The strategy was halted by hand. Every kernel's gate is already
    /// latched, so only the shared account needs marking.
    Killed,
}

/// Strategy-level totals for a strategy's `ctx.portfolio` view.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct BookTotals {
    /// Initial margin locked by open positions.
    pub locked_margin: f64,
    /// Net profit of the round trips closed so far.
    pub realized_pnl: f64,
    /// Direction-aware profit of open positions at the last marks.
    pub unrealized_pnl: f64,
    /// Sum of open positions' notional values, unsigned.
    pub gross_exposure: f64,
    /// Long notional minus short notional.
    pub net_exposure: f64,
}

/// One strategy's account, kernels and result curves.
pub struct PortfolioBook {
    config: BacktestConfig,
    kernels: Vec<EngineKernel>,
    account: SharedAccount,
    /// Last traded price this book has stepped, per instrument: the mark its
    /// equity is taken at. Per book because a book only marks what it was fed
    /// and `finish` clears it on liquidation.
    last_close: Vec<Option<f64>>,
    /// Last stepped print and its ordinal, per instrument: where this book
    /// liquidates and stamps cross-instrument orders.
    last_seen: Vec<Option<(usize, KernelBar)>>,
    equity_curve: Vec<f64>,
    drawdown_curve: Vec<f64>,
    returns: Vec<f64>,
    timestamps: Vec<i64>,
    trades: Vec<Trade>,
    streaming: StreamingMetrics,
    peak_equity: f64,
    /// Peak of the sampled curve before the current timestamp's sample, so a
    /// second event at the same timestamp can replace that sample cleanly.
    peak_before_sample: f64,
    /// One per kernel: what became of each order the instrument handled.
    order_logs: Vec<OrderLog>,
    /// Events applied to this book. The `halted_at` ordinal is this count.
    applied: usize,
    /// Samples ever taken, which outlives the window below.
    sampled: u64,
    /// Most samples of each curve to keep, for a book that runs unattended
    /// for a long time. `None` keeps every one.
    curve_window: Option<usize>,
    /// A halt asked for between events: cancelling working orders and
    /// flattening run at the top of the next `apply`. The reason, and whether
    /// to flatten.
    pending_halt: Option<(String, bool)>,
    /// Set when the book trades in an account group; see [`Grouped`].
    grouped: Option<Grouped>,
    /// Events produced for this book while another member of its group was
    /// being stepped (a group liquidation closing its positions). They are
    /// returned ahead of the next event's own, so the strategy hears of them.
    deferred: Vec<EngineEvent>,
    /// Corporate actions, dividends owed and the adjustment policy.
    lifecycle: lifecycle::Lifecycle,
    /// Whether any instrument has an expiry, so the lifecycle clock runs.
    has_expiries: bool,
    /// Per instrument, the instant its expiry was first noticed while its
    /// settlement waited for the underlying's print.
    noticed: Vec<Option<Timestamp>>,
    /// Instruments retired from a running node: flat, with no order, and skipped by the loops
    /// that run on every event. The kernel stays (its trades and orders are in the results).
    retired: Vec<bool>,
    /// How many kernels are retired; `0` is the common case and keeps the loops plain.
    dormant: usize,
    /// The kernels that are not retired, ascending: what the loops visit once any is retired.
    live: Vec<usize>,
}

/// What a book keeps when its account belongs to a group.
///
/// The cash, the locked margin and the halt of the account live with the
/// group; the book's [`SharedAccount`] is a copy the group refreshes before
/// each of the book's steps (see [`PortfolioBook::set_pool`]). What the book
/// keeps for itself is the cash its own trades have moved, so the equity it
/// samples is the strategy's, not the account's.
#[derive(Debug, Clone, Copy)]
struct Grouped {
    /// The share of the account's starting cash the strategy is credited
    /// with, plus the cash its own entries, exits and fees have moved since.
    cash: Money,
    /// The share the strategy started with.
    //
    // (`cash` is booked minor units; `start` is the f64 the curve starts from.)
    start: f64,
    /// Positions the group's other members hold, for `max_positions`.
    external_open: usize,
    /// Notional the group's other members hold, for `max_gross_exposure`.
    external_exposure: f64,
}

impl PortfolioBook {
    pub fn new(config: BacktestConfig, mode: AccountMode) -> Self {
        let pool = config.initial_capital;
        let scale = crate::accounts::MoneyScale::new(config.money_decimals)
            .unwrap_or_else(|e| panic!("invalid BacktestConfig.money_decimals: {e}"));
        Self {
            config,
            kernels: Vec::new(),
            account: SharedAccount::new(mode, pool, scale),
            last_close: Vec::new(),
            last_seen: Vec::new(),
            equity_curve: Vec::new(),
            drawdown_curve: Vec::new(),
            returns: Vec::new(),
            timestamps: Vec::new(),
            trades: Vec::new(),
            streaming: StreamingMetrics::new(),
            peak_equity: pool,
            peak_before_sample: pool,
            order_logs: Vec::new(),
            applied: 0,
            sampled: 0,
            curve_window: None,
            pending_halt: None,
            grouped: None,
            deferred: Vec::new(),
            lifecycle: lifecycle::Lifecycle::default(),
            has_expiries: false,
            noticed: Vec::new(),
            retired: Vec::new(),
            dormant: 0,
            live: Vec::new(),
        }
    }

    /// Register an instrument; returns its index.
    pub fn add_instrument(
        &mut self,
        symbol: String,
        direction: Direction,
        spec: Option<InstrumentSpec>,
        inst_config: Option<&InstrumentConfig>,
        policy: PositionPolicy,
    ) -> usize {
        let engine = crate::portfolio::engine::PortfolioEngine::new(self.config.clone());
        let mut kernel = EngineKernel::new(
            self.config.clone(),
            engine.fee_model.clone(),
            engine.slippage_model.clone(),
            engine.fill_price,
            symbol,
            direction,
            inst_config,
        )
        .with_risk_gate(self.config.risk_gate())
        .with_position_policy(policy)
        .with_account_mode(self.account.mode());
        if let Some(spec) = spec {
            self.has_expiries |= spec.expiration_ns.is_some();
            kernel.set_instrument(spec);
        }
        // The account owns all capital; kernels borrow it per step.
        kernel.set_cash_money(Money::ZERO);
        self.kernels.push(kernel);
        self.order_logs.push(OrderLog::default());
        self.noticed.push(None);
        self.retired.push(false);
        self.live.push(self.kernels.len() - 1);
        self.last_close.push(None);
        self.last_seen.push(None);
        self.kernels.len() - 1
    }

    /// Retire kernel `instrument`: it must hold no position and no working order. From now on the
    /// per-event loops skip it, so the cost of an event does not grow with instruments retired.
    pub fn retire_kernel(&mut self, instrument: usize) -> Result<(), String> {
        let kernel = &self.kernels[instrument];
        if kernel.open_count() > 0 || !kernel.open_orders().is_empty() {
            return Err(format!("{} holds a position or a working order", kernel.symbol()));
        }
        if !self.retired[instrument] {
            self.retired[instrument] = true;
            self.dormant += 1;
            self.live.retain(|&i| i != instrument);
            self.last_close[instrument] = None;
        }
        Ok(())
    }

    /// Bring a retired kernel back. It missed the risk gates' updates while it was dormant, so if
    /// the book has halted since, the kernel is killed like the others.
    pub fn revive_kernel(&mut self, instrument: usize) {
        if self.retired[instrument] {
            self.retired[instrument] = false;
            self.dormant -= 1;
            let at = self.live.partition_point(|&i| i < instrument);
            self.live.insert(at, instrument);
            if self.is_halted() {
                self.kernels[instrument].kill();
            }
        }
    }

    /// How many kernels the per-event loops visit, and the `k`-th of them.
    #[inline]
    pub(super) fn live_count(&self) -> usize {
        if self.dormant == 0 {
            self.kernels.len()
        } else {
            self.live.len()
        }
    }

    #[inline]
    pub(super) fn live_at(&self, k: usize) -> usize {
        if self.dormant == 0 {
            k
        } else {
            self.live[k]
        }
    }

    pub fn is_retired(&self, instrument: usize) -> bool {
        self.retired[instrument]
    }

    #[inline]
    fn live_any(&self, f: impl Fn(&EngineKernel) -> bool) -> bool {
        if self.dormant == 0 {
            self.kernels.iter().any(f)
        } else {
            self.live.iter().any(|&i| f(&self.kernels[i]))
        }
    }

    #[inline]
    fn live_each_mut(&mut self, mut f: impl FnMut(&mut EngineKernel)) {
        if self.dormant == 0 {
            self.kernels.iter_mut().for_each(f);
        } else {
            for &i in &self.live {
                f(&mut self.kernels[i]);
            }
        }
    }

    /// Sum `f(kernel, last mark)` over the live kernels, skipping unmarked ones.
    #[inline]
    fn live_marked_sum(&self, f: impl Fn(&EngineKernel, f64) -> f64) -> f64 {
        if self.dormant == 0 {
            let marked =
                |(k, close): (&EngineKernel, &Option<f64>)| close.map(|c| f(k, c)).unwrap_or(0.0);
            self.kernels.iter().zip(&self.last_close).map(marked).sum()
        } else {
            self.live
                .iter()
                .map(|&i| self.last_close[i].map(|c| f(&self.kernels[i], c)).unwrap_or(0.0))
                .sum()
        }
    }

    #[inline]
    fn live_open_total(&self) -> usize {
        if self.dormant == 0 {
            self.kernels.iter().map(|k| k.open_count()).sum()
        } else {
            self.live.iter().map(|&i| self.kernels[i].open_count()).sum()
        }
    }

    /// Equity sampled so far, once per distinct timestamp.
    pub fn equity_curve(&self) -> &[f64] {
        &self.equity_curve
    }

    /// Keep only the most recent samples of the equity, drawdown and returns
    /// curves, so a book that never stops does not grow with its event count.
    ///
    /// The curves, and the metrics computed from them at [`finish`], then
    /// cover the retained window only; trades and orders are always complete.
    /// Between `window` and `2 * window` samples are held. `None` (the
    /// default) keeps every sample.
    ///
    /// [`finish`]: PortfolioBook::finish
    pub fn set_curve_window(&mut self, window: Option<usize>) {
        self.curve_window = window.map(|w| w.max(1));
    }

    /// Events applied so far.
    pub fn applied(&self) -> usize {
        self.applied
    }

    /// Heap held by the sampled curves and their timestamps (capacity).
    pub fn curve_heap_bytes(&self) -> usize {
        (self.equity_curve.capacity()
            + self.drawdown_curve.capacity()
            + self.returns.capacity()
            + self.timestamps.capacity())
            * std::mem::size_of::<f64>()
    }

    /// The `submitted_idx` an order placed during the current event should
    /// carry on `instrument`'s own clock.
    ///
    /// Ordinals are per instrument, so an order routed to another
    /// instrument (a hedge leg placed from the index's print, or from a
    /// timer that fired on it) must not carry the dispatching event's
    /// ordinal: the two clocks are unrelated, and the order would match
    /// late, or never. It is stamped with the target's last stepped print
    /// instead, so it works from that instrument's next print — the first
    /// evidence of a trade after it was placed. An instrument that has not
    /// printed yet stamps 0, its first print.
    ///
    /// `current_instrument` is the instrument of the event being dispatched.
    pub fn submission_idx(
        &self,
        instrument: usize,
        event_idx: usize,
        current_instrument: Option<usize>,
    ) -> usize {
        match current_instrument {
            Some(current) if current == instrument => event_idx,
            _ => self
                .last_seen
                .get(instrument)
                .and_then(|seen| seen.map(|(idx, _)| idx))
                .unwrap_or(0),
        }
    }

    /// Kernel of an instrument, for order routing and queries.
    pub fn kernel_mut(&mut self, instrument: usize) -> &mut EngineKernel {
        &mut self.kernels[instrument]
    }

    pub fn kernel(&self, instrument: usize) -> &EngineKernel {
        &self.kernels[instrument]
    }

    /// Portfolio equity: the account balance plus each instrument's mark at
    /// its last known close.
    ///
    /// Cash mode marks positions at full value (historical model, pinned by
    /// the golden fixtures); margin mode marks direction-aware unrealized
    /// PnL, which prices winning shorts upward. The balance already includes
    /// notionally-locked margin — locks do not debit cash — so the margin arm
    /// does not double-count.
    pub fn equity(&self) -> f64 {
        self.base_cash() + self.positions_value()
    }

    /// The cash equity is taken from: the account balance, or, in a group,
    /// the cash this strategy's own trades have moved.
    fn base_cash(&self) -> f64 {
        match &self.grouped {
            Some(g) => self.account.scale().to_f64(g.cash),
            None => self.account.balance(),
        }
    }

    /// The marks of the open positions that equity adds to the cash.
    fn positions_value(&self) -> f64 {
        match self.account.mode() {
            AccountMode::Cash => self.live_marked_sum(|k, c| k.position_value(c)),
            AccountMode::Margin { .. } => self.live_marked_sum(|k, c| k.unrealized_value(c)),
        }
    }

    /// Apply a stepped kernel's cash and margin movement to the account, and
    /// to the strategy's own cash when it is in a group.
    fn settle(&mut self, delta_cash: Money, delta_locked: f64) {
        self.account.reconcile(delta_cash, delta_locked);
        if let Some(g) = &mut self.grouped {
            g.cash += delta_cash;
        }
    }

    /// Totals over the open positions and closed trades, marked at the last
    /// print each instrument was stepped on (instruments not yet marked add
    /// nothing).
    pub fn totals(&self) -> BookTotals {
        let mut totals = BookTotals {
            locked_margin: self.account.locked(),
            realized_pnl: self.trades.iter().map(|t| t.pnl).sum(),
            ..BookTotals::default()
        };
        for (kernel, mark) in self.kernels.iter().zip(&self.last_close) {
            let Some(price) = *mark else { continue };
            totals.unrealized_pnl += kernel.unrealized_value(price);
            let multiplier = kernel.ledger.contract_multiplier();
            for managed in kernel.ledger.positions() {
                let position = &managed.position;
                let notional = price * position.size * multiplier;
                totals.gross_exposure += notional;
                totals.net_exposure += notional * position.direction.multiplier();
            }
        }
        totals
    }

    /// Shared cash balance. In margin mode this includes locked initial
    /// margin; see [`PortfolioBook::free_capital`] for what can fund a new
    /// position.
    pub fn cash(&self) -> f64 {
        self.account.balance()
    }

    /// Adopt a pre-existing position on one instrument (broker-truth
    /// seeding) — see [`EngineKernel::adopt_position`], which owns the
    /// account-mode rules. Same lend/drain pool discipline as
    /// [`Self::apply`], so the cost basis comes out of the shared
    /// pool with no fees, no fill, and no trade: debited from the balance in
    /// cash mode, locked as initial margin in a fully funded margin book.
    ///
    /// Must be called before the first equity sample, and this is enforced:
    /// adopting mid-run leaves the curve flat for the pre-adoption stretch,
    /// which holds the running peak down and makes the decline that follows
    /// measure against the wrong high-water mark. Max drawdown then reads
    /// *better* than reality — 0.199% for a decline that is really 0.495%.
    /// The curve is written streaming, so this cannot be repaired later.
    ///
    /// The gate is the equity curve, not the event cursor: a quote or depth
    /// snapshot advances the cursor without sampling equity, and a live feed
    /// routinely delivers those before the first trade print. Adopting after
    /// one corrupts nothing and stays allowed.
    pub fn adopt_position(
        &mut self,
        instrument: usize,
        timestamp: Timestamp,
        price: f64,
        size: f64,
    ) -> Result<u64, String> {
        // The long-only form; see `adopt_position_with` for a signed one.
        #[allow(clippy::neg_cmp_op_on_partial_ord)]
        if !(size > 0.0) || !(price > 0.0) {
            return Err(format!(
                "adopt_position needs positive price and size (got size {size} @ {price})"
            ));
        }
        self.adopt_position_with(instrument, timestamp, price, size, &Adoption::default())
    }

    /// [`PortfolioBook::adopt_position`] with a signed size (negative is a
    /// short) and the posted margin or cost the caller states; see
    /// [`EngineKernel::adopt_position_with`].
    pub fn adopt_position_with(
        &mut self,
        instrument: usize,
        timestamp: Timestamp,
        price: f64,
        size: f64,
        extras: &Adoption,
    ) -> Result<u64, String> {
        if instrument >= self.kernels.len() {
            return Err(format!("unknown instrument index {instrument}"));
        }
        if self.sampled > 0 {
            return Err("adopt_position must be called before the first applied event".to_string());
        }
        let kernel = &mut self.kernels[instrument];
        let locked_before = kernel.locked_margin();
        // Same lend/drain discipline as `apply_current`: in margin mode the
        // kernel computes free capital from its own cash less its own locks,
        // so hand it the balance less every *other* kernel's locks.
        let injected = match self.account.mode() {
            AccountMode::Cash => self.account.balance(),
            AccountMode::Margin { .. } => {
                self.account.balance() - (self.account.locked() - locked_before)
            }
        };
        let injected = kernel.lend_cash(injected);
        let result = kernel.adopt_position_with(timestamp, price, size, extras);
        let delta_cash = kernel.cash_money() - injected;
        // Carry the locked delta too. Leaving this at 0.0 would mean the
        // shared account never learns about the adopted margin, so portfolio
        // free capital would read high by the whole cost basis — a risk
        // constraint silently weakened.
        let delta_locked = kernel.locked_margin() - locked_before;
        kernel.set_cash_money(Money::ZERO);
        self.settle(delta_cash, delta_locked);
        self.regroup_option_margin();
        result
    }

    /// Re-price every group of option legs that share an underlying and
    /// expiry, so sold legs that hedge each other lock the group's
    /// requirement rather than the sum of their naked deposits.
    ///
    /// Runs after every applied event and every adoption. In cash mode, or
    /// with no deposit-modelled option leg open, it does nothing. See
    /// [`crate::portfolio::option_groups`] for the arithmetic and the
    /// measurements it is held to.
    fn regroup_option_margin(&mut self) {
        if matches!(self.account.mode(), AccountMode::Cash) {
            return;
        }
        if let Some(model) = self.config.margin_model.clone() {
            // A chosen model margins its own option lines together.
            if model.groups_options() {
                self.regroup_modelled(&model);
            }
            return;
        }
        let mut groups: BTreeMap<(String, Option<Timestamp>), Vec<OptionLeg>> = BTreeMap::new();
        for (index, kernel) in self.kernels.iter().enumerate() {
            let legs = kernel.open_option_legs(index);
            if legs.is_empty() {
                continue;
            }
            let Some(key) = kernel.option_group_key() else { continue };
            groups.entry(key).or_default().extend(legs);
        }
        let mut delta_locked = 0.0;
        for legs in groups.values() {
            let Some(requirement) = group_requirement(legs) else { continue };
            for (kernel, position_id, share) in apportion(legs, requirement.total) {
                delta_locked += self.kernels[kernel].set_locked_margin(position_id, share);
            }
        }
        if delta_locked != 0.0 {
            self.account.reconcile(Money::ZERO, delta_locked);
        }
    }

    /// Capital available to open new positions across all instruments.
    pub fn free_capital(&self) -> f64 {
        self.account.free()
    }

    /// Whether a margin call, the drawdown kill-switch or an explicit halt
    /// has latched.
    pub fn is_halted(&self) -> bool {
        self.account.is_halted() || self.live_any(|k| k.risk_halted())
    }

    /// Halt the whole book: from now on every instrument refuses new entries
    /// as `killed`, and `halted` / `halted_at` report it like the drawdown
    /// halt. Working orders are cancelled and, when `flatten`, positions are
    /// closed at each instrument's last seen price when the next event is
    /// applied (or never, if there is none: `finish` closes everything
    /// anyway). Only the first halt's reason is kept; a later `flatten`
    /// still counts.
    pub fn halt(&mut self, reason: &str, flatten: bool) {
        self.halt_at(reason, flatten, self.applied);
    }

    /// [`PortfolioBook::halt`] with the ordinal `halted_at` reports.
    fn halt_at(&mut self, reason: &str, flatten: bool, idx: usize) {
        self.live_each_mut(|k| k.kill());
        self.halt_all(idx, HaltCause::Killed);
        match &mut self.pending_halt {
            Some((_, f)) => *f |= flatten,
            None => self.pending_halt = Some((reason.to_string(), flatten)),
        }
    }

    /// Close (and optionally cancel working orders on) every instrument at
    /// its last seen print, lending each kernel the pool as `apply` does and
    /// folding the cash back. The returned events are the closes and
    /// cancellations, in instrument order.
    fn cleanup_all(&mut self, cancel_orders: bool, flatten: bool) -> Vec<EngineEvent> {
        let mut events = Vec::new();
        for i in 0..self.kernels.len() {
            let (last_idx, last_bar) = match self.last_seen[i] {
                Some((idx, bar)) => (idx, Some(bar)),
                None => (0, None),
            };
            let kernel = &mut self.kernels[i];
            let locked_before = kernel.locked_margin();
            let injected = match self.account.mode() {
                AccountMode::Cash => self.account.balance(),
                AccountMode::Margin { .. } => {
                    self.account.balance() - (self.account.locked() - locked_before)
                }
            };
            let injected = kernel.lend_cash(injected);
            let closed = kernel.halt_cleanup(last_idx, last_bar.as_ref(), cancel_orders, flatten);
            let delta_cash = kernel.cash_money() - injected;
            let delta_locked = kernel.locked_margin() - locked_before;
            kernel.set_cash_money(Money::ZERO);
            self.settle(delta_cash, delta_locked);
            for event in &closed {
                if let EngineEvent::Exited { trade, .. } = event {
                    self.streaming.update(trade.return_pct / 100.0);
                    self.trades.push((**trade).clone());
                }
                self.order_logs[i].observe(&self.kernels[i], event);
            }
            events.extend(closed);
        }
        events
    }

    /// Step one schedule entry through its kernel.
    ///
    /// The kernel is re-pointed at the portfolio's capital, stepped, then
    /// drained: cash and locked-margin movements are folded back into the
    /// shared account. Cash mode is exactly the historical lend/drain of the
    /// whole pool.
    ///
    /// `max_positions` is counted across every instrument, so the kernel's
    /// gate refuses an entry once the *portfolio* is full — on the resting
    /// order path too, which is why the count is injected rather than
    /// pre-checked here. The count is snapshotted before the step, mirroring
    /// the array runner: an instrument that exits and re-enters on the same
    /// bar is still counted as holding its outgoing position.
    ///
    /// `depth` is the snapshot a [`ScheduleData::Depth`] entry refers to,
    /// resolved by the tape's store; the book keeps no market data.
    pub fn apply(
        &mut self,
        entry: &ScheduleEntry,
        depth: Option<DepthTick>,
        input: StepInput,
    ) -> Vec<EngineEvent> {
        // A halt asked for since the last event takes effect first: working
        // orders cancelled, positions closed at the last marks.
        let mut events = std::mem::take(&mut self.deferred);
        let Some((reason, flatten)) = self.pending_halt.take() else {
            events.extend(self.apply_entry(entry, depth, input));
            return events;
        };
        events.extend(self.cleanup_all(true, flatten));
        events.push(EngineEvent::RiskHalted { idx: entry.local_idx, reason });
        events.extend(self.apply_entry(entry, depth, input));
        events
    }

    /// The lifecycle first (corporate actions, dividends, contracts expired
    /// with no bar of their own), then the event. See `book/lifecycle.rs`.
    fn apply_entry(
        &mut self,
        entry: &ScheduleEntry,
        depth: Option<DepthTick>,
        input: StepInput,
    ) -> Vec<EngineEvent> {
        if !self.lifecycle_active() {
            return self.apply_entry_event(entry, depth, input);
        }
        let ts = entry.timestamp();
        let mut events = self.advance_lifecycle(ts, Some(entry.instrument));
        self.prepare_expiry(entry.instrument, ts);
        events.extend(self.apply_entry_event(entry, depth, input));
        // contracts waiting for this instrument's print settle on it
        events.extend(self.settle_waiting(entry.instrument, ts));
        events
    }

    fn apply_entry_event(
        &mut self,
        entry: &ScheduleEntry,
        depth: Option<DepthTick>,
        input: StepInput,
    ) -> Vec<EngineEvent> {
        let instrument = entry.instrument;

        // Portfolio-wide open count, skipped entirely when no limit is set.
        // Summing every kernel's ledger covers hedging policies, where one
        // instrument can hold several positions at once.
        let portfolio_open = self
            .config
            .max_positions
            .map(|_| self.live_open_total() + self.grouped.map_or(0, |g| g.external_open));

        // Gross notional held by the other instruments, for
        // `max_gross_exposure`; skipped entirely when no cap is set.
        let external_exposure = self.kernels[instrument].tracks_exposure().then(|| {
            let own = self
                .kernels
                .iter()
                .zip(&self.last_close)
                .enumerate()
                .filter(|(i, _)| *i != instrument)
                .map(|(_, (k, close))| close.map(|c| k.gross_notional(c)).unwrap_or(0.0))
                .sum::<f64>();
            match self.grouped {
                Some(g) => own + g.external_exposure,
                None => own,
            }
        });

        // In margin mode the kernel computes free capital as its own cash
        // minus its own locked margin, so hand it the balance less every
        // *other* kernel's locks — then its arithmetic sees the portfolio's
        // free capital.

        let group_context = self.group_context_for(instrument, entry.timestamp());
        let hedge_pending = self.hedge_pending_for(instrument);
        let kernel = &mut self.kernels[instrument];
        let locked_before = kernel.locked_margin();
        let injected = match self.account.mode() {
            AccountMode::Cash => self.account.balance(),
            AccountMode::Margin { .. } => {
                self.account.balance() - (self.account.locked() - locked_before)
            }
        };
        let injected = kernel.lend_cash(injected);
        kernel.set_external_open_count(portfolio_open);
        kernel.set_external_exposure(external_exposure);
        kernel.set_group_context(group_context);
        kernel.set_hedge_pending(hedge_pending);
        let mut events = match entry.data {
            ScheduleData::Bar(bar) => kernel.step(entry.local_idx, &bar, input),
            ScheduleData::Trade(tick) => kernel.step_trade(entry.local_idx, &tick, input),
            ScheduleData::Quote(quote) => kernel.step_quote(&quote),
            ScheduleData::Depth(_) => match depth {
                Some(snapshot) => kernel.step_depth(&snapshot),
                None => Vec::new(),
            },
        };
        let delta_cash = kernel.cash_money() - injected;
        let delta_locked = kernel.locked_margin() - locked_before;
        kernel.set_cash_money(Money::ZERO);
        kernel.set_external_open_count(None);
        kernel.set_external_exposure(None);
        kernel.set_group_context(None);
        kernel.set_hedge_pending(false);
        self.settle(delta_cash, delta_locked);
        // In a group the runtime regroups across every member, and escalates
        // a margin call to every member.
        if self.grouped.is_none() {
            self.regroup_option_margin();
        }

        // A kernel-local call sees only its own slice of the portfolio, but
        // the account is shared: escalate it so every instrument halts.
        if self.grouped.is_none()
            && events.iter().any(|e| matches!(e, EngineEvent::MarginCall { .. }))
        {
            self.halt_all(self.applied, HaltCause::MarginCall);
        }

        for event in &events {
            if let EngineEvent::Exited { trade, .. } = event {
                self.streaming.update(trade.return_pct / 100.0);
                self.trades.push((**trade).clone());
            }
            self.order_logs[instrument].observe(&self.kernels[instrument], event);
        }
        // A physically settled line the kernel just ended delivers its
        // underlying before the equity is sampled.
        if self.has_expiries {
            let delivered = self.deliver_contract_ends(instrument, entry.timestamp());
            events.extend(delivered);
        }

        // Quotes carry no traded price: they leave the mark alone, and the
        // print that follows updates it.
        if let Some(bar) = entry.as_bar() {
            self.last_close[instrument] = Some(bar.close);
            self.last_seen[instrument] = Some((entry.local_idx, bar));
        }

        // A quote does not sample equity. Marking on one would append a
        // zero return per quote, inflating the period count and distorting
        // annualized metrics purely from how chatty the feed is.
        if matches!(entry.data, ScheduleData::Quote(_) | ScheduleData::Depth(_)) {
            self.applied += 1;
            return events;
        }

        // Sample the portfolio once per distinct timestamp, after every
        // event at that timestamp: N instruments printing at the same instant
        // are one observation, not N. A later event at the same timestamp
        // replaces the sample taken after the previous one. Kernels still see
        // equity after every event, so the kill-switch reacts per event.
        let equity = self.equity();
        let ts = entry.timestamp();
        if self.timestamps.last() == Some(&ts) {
            self.equity_curve.pop();
            self.drawdown_curve.pop();
            self.returns.pop();
            self.timestamps.pop();
            self.peak_equity = self.peak_before_sample;
        } else {
            self.peak_before_sample = self.peak_equity;
        }
        let prev = self.equity_curve.last().copied();
        self.equity_curve.push(equity);
        if equity > self.peak_equity {
            self.peak_equity = equity;
        }
        self.drawdown_curve.push((self.peak_equity - equity) / self.peak_equity * 100.0);
        let ret = match prev {
            Some(p) if p != 0.0 => (equity - p) / p,
            _ => 0.0,
        };
        self.returns.push(ret);
        self.timestamps.push(ts);
        self.sampled += 1;
        if let Some(window) = self.curve_window.filter(|&w| self.equity_curve.len() >= 2 * w) {
            // Drop the oldest half in one move: amortised constant per sample.
            let drop = self.equity_curve.len() - window;
            self.equity_curve.drain(..drop);
            self.drawdown_curve.drain(..drop);
            self.returns.drain(..drop);
            self.timestamps.drain(..drop);
        }

        // The checks below are the account's. In a group they run once over
        // every member (see `crate::runtime::groups`), not once per book.
        if self.grouped.is_some() {
            self.applied += 1;
            return events;
        }

        self.account_checks(instrument, ts, entry.local_idx, equity, self.applied, &mut events);

        self.applied += 1;
        events
    }

    /// The account-level checks that follow an equity sample: margin
    /// maintenance, the day's loss limit and the drawdown kill-switch.
    /// `equity` is the sample just taken, `idx` the event's ordinal on its
    /// instrument and `halt_idx` the ordinal a halt latched here is reported
    /// at.
    fn account_checks(
        &mut self,
        instrument: usize,
        ts: Timestamp,
        idx: usize,
        mut equity: f64,
        halt_idx: usize,
        events: &mut Vec<EngineEvent>,
    ) {
        // Portfolio maintenance: the requirement is the sum of every
        // instrument's own requirement, so per-instrument `margin_maint`
        // rates apply. No single kernel can see this.
        if matches!(self.account.mode(), AccountMode::Margin { .. }) && !self.account.is_halted() {
            let required: f64 = self
                .kernels
                .iter()
                .zip(&self.last_close)
                .map(|(k, close)| close.map(|c| k.maintenance_requirement(c)).unwrap_or(0.0))
                .sum();
            if required > 0.0 && equity < required {
                self.halt_all(halt_idx, HaltCause::MarginCall);
                events.push(EngineEvent::MarginCall { idx, equity, required });
                if self.config.liquidate_on_margin_call {
                    events.extend(self.liquidate_marked());
                }
            }
        }

        // The day's loss limit, on portfolio equity. Every kernel sees the
        // same figure, so they latch together; report the rising edge once.
        let day_blocked_before = self.live_any(|k| k.is_day_blocked());
        // The book's day follows the schedule of the instrument whose event
        // this is, so every kernel shares one day.
        let day = self.kernels[instrument].risk_day(ts);
        let mut breached = false;
        self.live_each_mut(|k| breached |= k.observe_day_in(day, equity));
        if breached && !day_blocked_before {
            events.push(EngineEvent::RiskHalted { idx, reason: "daily_loss".to_string() });
            if self.config.flatten_on_daily_loss {
                let closed = self.cleanup_all(false, true);
                events.extend(closed);
                // The sample above was taken before the closes paid their
                // costs; restate it so the curve shows the flattened book.
                equity = self.equity();
                self.restate_last_sample(equity);
            }
        }

        // Kernels all see the same portfolio equity, so their drawdown gates
        // latch in lockstep; record the rising edge once.
        let peak = self.peak_equity;
        let risk_halted_before = self.live_any(|k| k.risk_halted());
        self.live_each_mut(|k| k.observe_equity(equity, peak));
        if !risk_halted_before && self.live_any(|k| k.risk_halted()) {
            self.halt_all(halt_idx, HaltCause::Drawdown);
        }
    }

    /// Close every instrument at its last mark after a margin call, folding
    /// the cash back through the account like any other close.
    fn liquidate_marked(&mut self) -> Vec<EngineEvent> {
        let mut events = Vec::new();
        for i in 0..self.kernels.len() {
            let Some((last_idx, last_bar)) = self.last_seen[i] else { continue };
            let kernel = &mut self.kernels[i];
            let locked_before = kernel.locked_margin();
            let injected = match self.account.mode() {
                AccountMode::Cash => self.account.balance(),
                AccountMode::Margin { .. } => {
                    self.account.balance() - (self.account.locked() - locked_before)
                }
            };
            let injected = kernel.lend_cash(injected);
            let closed = kernel.liquidate_all(last_idx, &last_bar);
            let delta_cash = kernel.cash_money() - injected;
            let delta_locked = kernel.locked_margin() - locked_before;
            kernel.set_cash_money(Money::ZERO);
            self.settle(delta_cash, delta_locked);
            for event in &closed {
                if let EngineEvent::Exited { trade, .. } = event {
                    self.streaming.update(trade.return_pct / 100.0);
                    self.trades.push((**trade).clone());
                }
            }
            events.extend(closed);
        }
        events
    }

    /// Overwrite the most recent equity sample (and the figures derived from
    /// it) after the book changed within the same event.
    fn restate_last_sample(&mut self, equity: f64) {
        let Some(last) = self.equity_curve.last_mut() else { return };
        *last = equity;
        let n = self.equity_curve.len();
        if equity > self.peak_equity {
            self.peak_equity = equity;
        }
        self.drawdown_curve[n - 1] = (self.peak_equity - equity) / self.peak_equity * 100.0;
        let prev = if n >= 2 { Some(self.equity_curve[n - 2]) } else { None };
        self.returns[n - 1] = match prev {
            Some(p) if p != 0.0 => (equity - p) / p,
            _ => 0.0,
        };
    }

    /// Latch a portfolio-wide halt on the shared account.
    ///
    /// The cause decides what else is needed. A margin call must trip every
    /// kernel's margin kill-switch so entries are refused with
    /// `RejectReason::MarginCall`; a drawdown halt must *not*, because each
    /// kernel's own risk gate has already latched from the portfolio equity
    /// it was fed and reports `RejectReason::DrawdownHalt`. Tripping the
    /// margin switch for a drawdown would mislabel the reason.
    fn halt_all(&mut self, idx: usize, cause: HaltCause) {
        self.account.halt(idx);
        if matches!(cause, HaltCause::MarginCall) {
            self.live_each_mut(|k| k.halt_margin());
        }
    }

    /// The force-close half of [`PortfolioBook::finish`]. Each instrument's
    /// last print is consumed, so a second call finds nothing to close.
    fn close_out_impl(&mut self) {
        for i in 0..self.kernels.len() {
            if let Some((idx, bar)) = self.last_seen[i].take() {
                let kernel = &mut self.kernels[i];
                let locked_before = kernel.locked_margin();
                let injected = match self.account.mode() {
                    AccountMode::Cash => self.account.balance(),
                    AccountMode::Margin { .. } => {
                        self.account.balance() - (self.account.locked() - locked_before)
                    }
                };
                let injected = kernel.lend_cash(injected);
                for trade in kernel.finalize_all(idx, &bar) {
                    self.streaming.update(trade.return_pct / 100.0);
                    self.trades.push(trade);
                }
                let delta_cash = kernel.cash_money() - injected;
                let delta_locked = kernel.locked_margin() - locked_before;
                kernel.set_cash_money(Money::ZERO);
                self.settle(delta_cash, delta_locked);
                self.last_close[i] = None;
            }
        }
    }

    /// Force-close every instrument at its last seen bar and compute
    /// portfolio metrics.
    pub fn finish(mut self) -> SessionOutcome {
        self.close_out_impl();
        // Positions are flat; the final mark is the balance itself.
        let final_cash = self.base_cash();
        if let Some(last) = self.equity_curve.last_mut() {
            *last = final_cash;
        }

        // In a group the strategy's curve starts from its share of the
        // account, which is not the account's capital the config carries.
        let grouped_config;
        let metrics_config = match self.grouped {
            Some(g) => {
                grouped_config = BacktestConfig { initial_capital: g.start, ..self.config.clone() };
                &grouped_config
            }
            None => &self.config,
        };
        let metrics = compute_backtest_metrics_with_config(
            &self.equity_curve,
            &self.drawdown_curve,
            &self.returns,
            &self.trades,
            &self.timestamps,
            metrics_config,
        );
        let outcomes = self
            .kernels
            .iter()
            .map(|kernel| kernel.symbol().to_string())
            .enumerate()
            .map(|(i, symbol)| InstrumentOutcome {
                symbol: symbol.clone(),
                trades: self.trades.iter().filter(|t| t.symbol == symbol).count(),
                pnl: self.trades.iter().filter(|t| t.symbol == symbol).map(|t| t.pnl).sum(),
                rejected_entries: self.kernels[i].rejected_entries(),
            })
            .collect();

        let rejected_entries: usize = self.kernels.iter().map(|k| k.rejected_entries()).sum();
        let halted = self.account.is_halted() || self.kernels.iter().any(|k| k.risk_halted());
        let halted_at = self.account.halted_at();

        // Submission order across instruments: by submission time, ties by
        // instrument registration order (the sort is stable).
        let mut order_log: Vec<_> = self
            .order_logs
            .iter()
            .zip(&self.kernels)
            .flat_map(|(log, kernel)| log.records(kernel))
            .collect();
        order_log.sort_by_key(|o| o.submitted_ts);

        let result = BacktestResult::new(
            metrics,
            self.equity_curve,
            self.drawdown_curve,
            self.trades,
            self.returns,
        )
        .with_orders(order_log);
        SessionOutcome { result, instruments: outcomes, rejected_entries, halted, halted_at }
    }
}

impl crate::runtime::memory::HeapSize for PortfolioBook {
    /// The book's own allocations, curves excluded (see
    /// [`PortfolioBook::curve_heap_bytes`]): the kernel, order-log and mark
    /// vectors at capacity plus the trade records. A kernel's internal
    /// collections (working orders, ledger, symbol text) are not walked, so a
    /// book with many open orders is undercounted.
    fn heap_bytes(&self) -> usize {
        use std::mem::size_of;
        self.kernels.capacity() * size_of::<EngineKernel>()
            + self.order_logs.capacity() * size_of::<OrderLog>()
            + self.last_close.capacity() * size_of::<Option<f64>>()
            + self.last_seen.capacity() * size_of::<Option<(usize, KernelBar)>>()
            + self.trades.capacity() * size_of::<Trade>()
    }
}

mod group;
mod lifecycle;

pub use group::{model_shares, ModelLine, ModelLines, OptionLegs};

#[cfg(test)]
#[path = "book_tests.rs"]
mod tests;
