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
    BacktestConfig, BacktestResult, Direction, ExitReason, InstrumentConfig, Timestamp, Trade,
};
use crate::data::DepthTick;
use crate::instruments::InstrumentSpec;
use crate::metrics::streaming::StreamingMetrics;
use crate::portfolio::curve::Curve;
use crate::portfolio::engine::compute_backtest_metrics_with_config;
use crate::portfolio::kernel::{Adoption, EngineEvent, EngineKernel, KernelBar, StepInput};
use crate::portfolio::ledger::PositionPolicy;
use crate::portfolio::option_groups::{apportion, group_requirement, OptionLeg};
use crate::portfolio::runner::OrderLog;
use crate::portfolio::session::{InstrumentOutcome, ScheduleData, ScheduleEntry, SessionOutcome};

mod venue;

/// How a forced flatten closes the positions.
#[derive(Clone, Copy)]
enum Flatten {
    No,
    /// Decided in the event of `instrument` stamped `ts`, before it is applied: the close
    /// is an ordinary reduce-only market order of that decision.
    Decision {
        instrument: usize,
        ts: i64,
    },
    /// Detected after the last event applied, at its close.
    AfterEvent,
}

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

/// The close of a hedge leg the engine carries out as orders: the sold legs of its group first,
/// then the hedge itself. All are ordinary reduce-only market orders, filled as the run's fill
/// timing fills an answer to an event.
struct Cover {
    /// The hedge: kernel index and position id.
    hedge: (usize, u64),
    /// The sold legs' closing orders: kernel index and order id.
    sold: Vec<(usize, u64)>,
    /// The hedge's own closing order, once the sold legs are done.
    hedge_order: Option<u64>,
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
    /// Entries each instrument could not fund (sized to nothing, or refused
    /// for capital or margin): a refusal that is not a risk limit and so is
    /// not in the risk gate's count.
    unfunded: Vec<usize>,
    /// Whether any print, quote or depth snapshot was applied: a tick run.
    tick_run: bool,
    equity_curve: Curve<f64>,
    drawdown_curve: Curve<f64>,
    returns: Curve<f64>,
    timestamps: Curve<i64>,
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
    /// Exits of hedge legs in progress: the group's sold legs are being closed with orders, the
    /// hedge's own close follows (see [`PortfolioBook::advance_covers`]).
    covers: Vec<Cover>,
    /// Serial of the engine's cover orders, for their client ids.
    cover_serial: u64,
    /// Inside [`PortfolioBook::advance_covers`], whose fills step the book again.
    advancing_covers: bool,
    /// Set once the book holds a sold option the margin model cannot bound (see
    /// [`PortfolioBook::flag_unmarginable`]).
    unmarginable: Option<String>,
    /// Judge the account's limits once per instant, in [`PortfolioBook::judge`],
    /// rather than after every event; see [`PortfolioBook::judge_per_instant`].
    judge_deferred: bool,
    /// Whether the config sets no drawdown or daily-loss limit, so the
    /// per-event feeds to those gates can change nothing and are skipped.
    limits_inert: bool,
    /// Cash account, no portfolio-wide limit, no option margin groups: the
    /// conditions (with one instrument) of the lone-bar fast path.
    simple: bool,
    /// Whether the account's model margins option lines together, so an
    /// event needs the group context; false skips building it per event.
    margin_groups: bool,
    /// Corporate actions, dividends owed and the adjustment policy.
    lifecycle: lifecycle::Lifecycle,
    /// Whether any instrument has an expiry, so the lifecycle clock runs.
    has_expiries: bool,
    /// Per instrument, the instant its expiry was first noticed while its
    /// settlement waited for the underlying's print.
    noticed: Vec<Option<Timestamp>>,
    /// Instruments that have an event in the timestamp group being applied. A
    /// contract among them settles on its own bar of the group, whichever
    /// event of the group notices its expiry first.
    in_group: Vec<usize>,
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
        let limits_inert = config.max_drawdown_pct.is_none()
            && config.max_daily_loss.is_none()
            && config.max_daily_loss_pct.is_none();
        let margin_groups = !matches!(mode, AccountMode::Cash)
            && config.margin_model.as_deref().is_some_and(|m| m.groups_options());
        Self {
            judge_deferred: false,
            limits_inert,
            simple: matches!(mode, AccountMode::Cash)
                && limits_inert
                && !margin_groups
                && config.max_positions.is_none()
                && config.max_gross_exposure.is_none(),
            margin_groups,
            config,
            kernels: Vec::new(),
            account: SharedAccount::new(mode, pool, scale),
            last_close: Vec::new(),
            last_seen: Vec::new(),
            unfunded: Vec::new(),
            tick_run: false,
            equity_curve: Curve::default(),
            drawdown_curve: Curve::default(),
            returns: Curve::default(),
            timestamps: Curve::default(),
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
            unmarginable: None,
            covers: Vec::new(),
            cover_serial: 0,
            advancing_covers: false,
            lifecycle: lifecycle::Lifecycle::default(),
            has_expiries: false,
            noticed: Vec::new(),
            in_group: Vec::new(),
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
        self.add_instrument_with_models(
            symbol,
            direction,
            spec,
            inst_config,
            policy,
            engine.fee_model,
            engine.slippage_model,
            engine.fill_price,
        )
    }

    /// [`PortfolioBook::add_instrument`] with the fee, slippage and fill
    /// models given rather than derived from the config: the single-instrument
    /// engine lets a caller swap them.
    #[allow(clippy::too_many_arguments)]
    pub fn add_instrument_with_models(
        &mut self,
        symbol: String,
        direction: Direction,
        spec: Option<InstrumentSpec>,
        inst_config: Option<&InstrumentConfig>,
        policy: PositionPolicy,
        fee_model: crate::execution::FeeModel,
        slippage_model: crate::execution::SlippageModel,
        fill_price: crate::execution::FillPrice,
    ) -> usize {
        // The kernel squares off at its own instrument's time where the run names one.
        let mut config = self.config.clone();
        config.squareoff_time_minutes = config.squareoff_minutes_for(&symbol);
        let mut kernel = EngineKernel::new(
            config,
            fee_model,
            slippage_model,
            fill_price,
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
        self.unfunded.push(0);
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

    /// Equity sampled so far, once per distinct timestamp, copied out of the
    /// curve's blocks. [`PortfolioBook::last_equity`] reads the last sample.
    pub fn equity_curve(&self) -> Vec<f64> {
        self.equity_curve.clone().into_vec()
    }

    /// The most recent equity sample.
    pub fn last_equity(&self) -> Option<f64> {
        self.equity_curve.last()
    }

    /// Room for `samples` samples in each curve, so a run that knows its
    /// length keeps them contiguous. Call before the first event.
    pub fn reserve_curves(&mut self, samples: usize) {
        self.equity_curve.reserve(samples);
        self.drawdown_curve.reserve(samples);
        self.returns.reserve(samples);
        self.timestamps.reserve(samples);
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
        self.equity_curve.heap_bytes()
            + self.drawdown_curve.heap_bytes()
            + self.returns.heap_bytes()
            + self.timestamps.heap_bytes()
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
    #[inline]
    pub fn equity(&self) -> f64 {
        self.base_cash() + self.positions_value()
    }

    /// The cash equity is taken from: the account balance, or, in a group,
    /// the cash this strategy's own trades have moved.
    #[inline]
    fn base_cash(&self) -> f64 {
        match &self.grouped {
            Some(g) => self.account.scale().to_f64(g.cash),
            None => self.account.balance(),
        }
    }

    /// The marks of the open positions that equity adds to the cash.
    #[inline]
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

    /// The exits of hedge legs still in progress, `(hedge instrument, hedge position id, sold legs
    /// sent, whether the hedge's own close is sent)`: state a checkpoint's digest must cover.
    pub fn pending_covers(&self) -> Vec<(usize, u64, usize, bool)> {
        self.covers
            .iter()
            .map(|c| (c.hedge.0, c.hedge.1, c.sold.len(), c.hedge_order.is_some()))
            .collect()
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

    /// Cancel working orders on every instrument and, with `flatten`, close its
    /// positions the way an ordinary reduce-only market order would, lending
    /// each kernel the pool as `apply` does and folding the cash back. The
    /// returned events are the closes made now and the cancellations, in
    /// instrument order.
    fn cleanup_all(&mut self, cancel_orders: bool, flatten: Flatten) -> Vec<EngineEvent> {
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
            let mut closed = kernel.halt_cleanup(last_idx, cancel_orders);
            match flatten {
                Flatten::No => {}
                // Decided in the event of `instrument` stamped `ts`: that instrument's bar
                // stamped `ts` is the first it can fill on; any other instrument's first
                // is the one after, as for any order routed to it from this event.
                Flatten::Decision { instrument, ts } => {
                    if kernel.is_in_position() {
                        let first = if i == instrument { ts } else { ts.saturating_add(1) };
                        kernel.request_flatten(ExitReason::Liquidation, first);
                    }
                }
                Flatten::AfterEvent => {
                    if let Some(bar) = last_bar.as_ref() {
                        closed.extend(kernel.flatten_after(last_idx, bar, ExitReason::Liquidation));
                    }
                }
            }
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

    /// Mark the instrument of `entry` at the price it carries, without
    /// stepping anything. The equal-timestamp rule applies every event of a
    /// timestamp group before any of them is decided or matched, so a decision
    /// at `t` reads one mark for every instrument that printed at `t`, whatever
    /// the order the group is stepped in. The step finds the same mark.
    ///
    /// The order-routing stamp (`last_seen`) stays with the step: an order for
    /// this instrument decided at `t` is stamped with the last bar it stepped,
    /// so it meets the bar of `t` still to be matched. Returns the mark the
    /// instrument had before.
    pub fn premark(&mut self, entry: &ScheduleEntry) -> Option<f64> {
        let previous = self.last_close[entry.instrument];
        if let Some(bar) = entry.as_bar() {
            self.last_close[entry.instrument] = Some(bar.close);
        }
        previous
    }

    /// Note that `instrument` has an event in the group being applied.
    pub fn group_enter(&mut self, instrument: usize) {
        self.in_group.push(instrument);
    }

    /// The group is over.
    pub fn group_leave(&mut self) {
        self.in_group.clear();
    }

    /// Set an instrument's mark outright (see [`PortfolioBook::premark`]): a
    /// decision on an event reads that event's own instrument at the mark it
    /// had before the event, as a decision alone does.
    pub fn set_mark(&mut self, instrument: usize, mark: Option<f64>) {
        self.last_close[instrument] = mark;
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
        if self.simple
            && self.kernels.len() == 1
            && self.dormant == 0
            && self.pending_halt.is_none()
            && self.deferred.is_empty()
            && !self.has_expiries
            && matches!(entry.data, ScheduleData::Bar(_))
            && !self.lifecycle_active()
        {
            return self.apply_lone_bar(entry, input);
        }
        // A halt asked for since the last event takes effect first: working
        // orders cancelled, positions closed at the last marks.
        let mut events = std::mem::take(&mut self.deferred);
        let Some((reason, flatten)) = self.pending_halt.take() else {
            events.extend(self.apply_entry(entry, depth, input));
            return events;
        };
        let flatten = if flatten {
            Flatten::Decision { instrument: entry.instrument, ts: entry.timestamp() }
        } else {
            Flatten::No
        };
        events.extend(self.cleanup_all(true, flatten));
        events.push(EngineEvent::RiskHalted { idx: entry.local_idx, reason });
        events.extend(self.apply_entry(entry, depth, input));
        events
    }

    /// Whether the open of the bar `entry` carries has something to meet ahead of the strategy
    /// ([`PortfolioBook::open_bar`]): a bar under `next_bar_open`, on a book with no halt
    /// waiting to take effect first, where an intent or a market order waits for its open.
    pub fn can_open(&self, entry: &ScheduleEntry) -> bool {
        matches!(entry.data, ScheduleData::Bar(_))
            && self.pending_halt.is_none()
            && self.kernels[entry.instrument].fills_at_next_open()
            && self.kernels[entry.instrument].has_open_work()
    }

    /// Under `next_bar_open`, meet the open of the bar `entry` carries before the strategy has
    /// seen the bar: what was decided on earlier bars fills at this open, and the events of those
    /// fills are returned for the strategy to hear knowing only what was known at that open. The
    /// book's own bookkeeping of the bar (the mark, the equity sample, the account's limits)
    /// stays with [`PortfolioBook::apply`], which carries on from here. Only for an entry
    /// [`PortfolioBook::can_open`] allows; the lifecycle of the instant has run first
    /// ([`PortfolioBook::begin_event`]).
    pub fn open_bar(&mut self, entry: &ScheduleEntry) -> Vec<EngineEvent> {
        let ScheduleData::Bar(bar) = entry.data else { return Vec::new() };
        self.lent_step(entry.instrument, entry.timestamp(), |kernel| {
            kernel.step_open(entry.local_idx, &bar)
        })
    }

    /// Whether the bar `entry` carries has resting work to meet ahead of the strategy's
    /// decision ([`PortfolioBook::rest_bar`]): a bar, on a book with no halt waiting to take
    /// effect first, whose instrument holds a position or has an order resting.
    pub fn can_rest(&self, entry: &ScheduleEntry) -> bool {
        matches!(entry.data, ScheduleData::Bar(_))
            && self.pending_halt.is_none()
            && self.kernels[entry.instrument].has_rest_work()
    }

    /// Meet what rests for the bar `entry` carries before the strategy has decided on it: the
    /// position's stop and target, the square-off, the margin check and the orders committed on
    /// earlier bars match the bar's range, as they do at a venue while the bar forms. The events
    /// are returned for the strategy to hear before `on_bar`; the book's bookkeeping of the bar
    /// (the mark, the equity sample, the account's limits) stays with [`PortfolioBook::apply`],
    /// which carries on from here with the decisions of the bar. Only for an entry
    /// [`PortfolioBook::can_rest`] allows; the lifecycle of the instant and the open have run
    /// first.
    pub fn rest_bar(&mut self, entry: &ScheduleEntry) -> Vec<EngineEvent> {
        let ScheduleData::Bar(bar) = entry.data else { return Vec::new() };
        self.lent_step(entry.instrument, entry.timestamp(), |kernel| {
            kernel.step_rest(entry.local_idx, &bar)
        })
    }

    /// [`PortfolioBook::apply`] for the commonest book: one instrument on a
    /// cash account with no portfolio-wide limit, no option margin groups, no
    /// expiry, no corporate action and no group, fed a bar. Every step the
    /// general path takes that such a book cannot need is skipped; what it
    /// takes (lend, step, drain, bookkeeping, the equity sample) is the same
    /// code, so this is the same simulation, not a second one.
    fn apply_lone_bar(&mut self, entry: &ScheduleEntry, input: StepInput) -> Vec<EngineEvent> {
        let ScheduleData::Bar(bar) = entry.data else { unreachable!("checked by the caller") };
        let kernel = &mut self.kernels[0];
        let booked = self.account.balance_money();
        kernel.set_cash_money(booked);
        let events = kernel.step(entry.local_idx, &bar, input);
        let delta_cash = kernel.cash_money() - booked;
        kernel.clear_cash();
        self.settle(delta_cash, 0.0);
        for event in &events {
            if let EngineEvent::Exited { trade, .. } = event {
                self.streaming.update(trade.return_pct / 100.0);
                self.trades.push((**trade).clone());
            }
            if is_unfunded_refusal(event) {
                self.unfunded[0] += 1;
            }
            self.order_logs[0].observe(&self.kernels[0], event);
        }
        self.last_close[0] = Some(bar.close);
        self.last_seen[0] = Some((entry.local_idx, bar));
        // No limit is configured and the account is cash, so the account-level
        // checks that follow a sample have nothing to trip.
        self.sample_equity(bar.timestamp);
        self.applied += 1;
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

    /// Run `run` on one instrument's kernel with the account's cash and the portfolio's limits
    /// lent to it, then fold what it did back into the account: cash and margin moved, a margin
    /// call escalated, trades and unfunded entries counted, the order log fed.
    fn lent_step(
        &mut self,
        instrument: usize,
        ts: Timestamp,
        run: impl FnOnce(&mut EngineKernel) -> Vec<EngineEvent>,
    ) -> Vec<EngineEvent> {
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

        let (group_context, hedge_pending) = if self.margin_groups {
            (self.group_context_for(instrument, ts), self.hedge_pending_for(instrument))
        } else {
            (None, false)
        };
        let kernel = &mut self.kernels[instrument];
        let locked_before = kernel.locked_margin();
        // Cash mode lends the booked amount itself, with no trip through f64.
        let injected = match self.account.mode() {
            AccountMode::Cash => {
                let booked = self.account.balance_money();
                kernel.set_cash_money(booked);
                booked
            }
            AccountMode::Margin { .. } => {
                kernel.lend_cash(self.account.balance() - (self.account.locked() - locked_before))
            }
        };
        kernel.set_external_open_count(portfolio_open);
        kernel.set_external_exposure(external_exposure);
        kernel.set_group_context(group_context);
        kernel.set_hedge_pending(hedge_pending);
        let mut events = run(kernel);
        let delta_cash = kernel.cash_money() - injected;
        let delta_locked = kernel.locked_margin() - locked_before;
        kernel.clear_cash();
        kernel.set_external_open_count(None);
        kernel.set_external_exposure(None);
        kernel.set_group_context(None);
        kernel.set_hedge_pending(false);
        self.settle(delta_cash, delta_locked);
        for held in self.kernels[instrument].take_held_triggers() {
            self.start_cover(instrument, held.position_id, ts);
        }
        // In a group the runtime regroups across every member, and escalates
        // a margin call to every member.
        if self.grouped.is_none() && !matches!(self.account.mode(), AccountMode::Cash) {
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
            if is_unfunded_refusal(event) {
                self.unfunded[instrument] += 1;
            }
            self.order_logs[instrument].observe(&self.kernels[instrument], event);
        }
        events.extend(self.advance_covers(ts));
        events
    }

    /// The exit of a hedge leg on `instrument` was held back (its stop or target fired, or a
    /// forced exit reached it, and closing it alone would leave a sold leg of its margin group
    /// without cover): close the sold legs left bare first, with reduce-only market orders, then
    /// the hedge. This sends the sold legs' orders; the hedge's follows once
    /// they are done ([`PortfolioBook::advance_covers`]), so no instant holds a sold option
    /// without its hedge.
    fn start_cover(&mut self, instrument: usize, position_id: u64, ts: Timestamp) {
        if self.covers.iter().any(|c| c.hedge == (instrument, position_id)) {
            return;
        }
        let mut sold = Vec::new();
        for (j, id, size) in self.sold_legs_left_bare(instrument, position_id) {
            if let Some(order) = self.send_cover(j, id, size, ts) {
                sold.push((j, order));
            }
        }
        self.covers.push(Cover { hedge: (instrument, position_id), sold, hedge_order: None });
    }

    /// Send the reduce-only market order that closes `size` of a position, as the engine's own.
    fn send_cover(
        &mut self,
        kernel: usize,
        position_id: u64,
        size: f64,
        ts: Timestamp,
    ) -> Option<u64> {
        use crate::execution::orders::{OrderKind, OrderSide, QtySpec, TimeInForce};
        let (idx, at) = match self.last_seen[kernel] {
            Some((idx, bar)) => (idx, bar.timestamp),
            None => (0, ts),
        };
        // Under `next_bar_open` an event that came after the open is answered at the next bar's
        // open. A kernel that has not stepped this instant's bar yet must not meet the order on
        // it: it works from the print after that one, whichever instrument the book steps first.
        let after_open = self.kernels[kernel].fills_at_next_open()
            && self.kernels.iter().any(|k| k.off_open_at == Some(ts));
        let idx = if after_open && at < ts { idx + 1 } else { idx };
        let direction = self.kernels[kernel]
            .position_snapshots()
            .iter()
            .find(|p| p.position_id == position_id)?
            .direction;
        let side = if direction == Direction::Long { OrderSide::Sell } else { OrderSide::Buy };
        self.cover_serial += 1;
        let k = &mut self.kernels[kernel];
        let id = k.submit_order_full(
            side,
            QtySpec::Units(size),
            OrderKind::Market,
            TimeInForce::Gtc,
            idx,
            at,
            format!("engine-cover-{}", self.cover_serial),
            None,
            None,
            false,
            true,
            None,
        );
        k.set_order_position(id, Some(position_id));
        k.set_order_follow_up(id);
        Some(id)
    }

    /// Carry the exits of hedge legs forward: meet the orders sent so far with the bar they were
    /// answering (the close under `same_bar_close`, the open under `next_bar_open`, as for any
    /// answer to an event), and once a cover's sold legs are all done send the hedge's own close.
    /// Returns the events the fills produced.
    fn advance_covers(&mut self, ts: Timestamp) -> Vec<EngineEvent> {
        let mut out = Vec::new();
        if self.advancing_covers || self.covers.is_empty() {
            return out;
        }
        self.advancing_covers = true;
        let mut at = 0;
        while at < self.covers.len() {
            let sold = self.covers[at].sold.clone();
            for (j, order) in &sold {
                out.extend(self.meet_cover_order(*j, *order, ts));
            }
            let done = sold.iter().all(|(j, o)| {
                self.kernels[*j].order(*o).is_none_or(|order| order.status.is_terminal())
            });
            if done && self.covers[at].hedge_order.is_none() {
                let (hk, position) = self.covers[at].hedge;
                let size = self.kernels[hk]
                    .position_snapshots()
                    .iter()
                    .find(|p| p.position_id == position)
                    .map(|p| p.size);
                match size.and_then(|size| self.send_cover(hk, position, size, ts)) {
                    Some(order) => self.covers[at].hedge_order = Some(order),
                    None => {
                        self.covers.remove(at);
                        continue;
                    }
                }
            }
            if let Some(order) = self.covers[at].hedge_order {
                let hk = self.covers[at].hedge.0;
                out.extend(self.meet_cover_order(hk, order, ts));
                if self.kernels[hk].order(order).is_none_or(|o| o.status.is_terminal()) {
                    self.covers.remove(at);
                    continue;
                }
            }
            at += 1;
        }
        self.advancing_covers = false;
        out
    }

    /// Fill a cover order against the bar of the event that sent it (`ts`), if its kernel has
    /// stepped that bar: at its close under `same_bar_close`, its open under `next_bar_open`. A
    /// kernel that has not stepped it yet meets the order at its own step of that bar, so the
    /// price of a close is the same whichever instrument the book steps first.
    fn meet_cover_order(&mut self, kernel: usize, order: u64, ts: Timestamp) -> Vec<EngineEvent> {
        let waiting = self.kernels[kernel].order(order).is_some_and(|o| !o.status.is_terminal());
        if !waiting {
            return Vec::new();
        }
        self.fill_follow_up(kernel, order, ts)
    }

    /// A strategy's answer to a fill it just heard, a market order marked `follow_up`, meets the
    /// bar `instrument`'s kernel stepped at `ts` (see [`EngineKernel::fill_follow_up`]). The
    /// equity sample of the instant is restated afterwards. Returns the events of the fill.
    pub fn fill_follow_up(
        &mut self,
        instrument: usize,
        order_id: u64,
        ts: Timestamp,
    ) -> Vec<EngineEvent> {
        // Under `next_bar_open` the follow-up is priced at the bar's open. An event of that bar
        // that came after the open (a limit filled inside the bar, an expiry) is not known at
        // the open, so what answers it goes the ordinary way, to the next bar.
        if self.kernels[instrument].fills_at_next_open()
            && self.kernels.iter().any(|k| k.off_open_at == Some(ts))
        {
            self.kernels[instrument].clear_order_follow_up(order_id);
            return Vec::new();
        }
        let events = self.lent_step(instrument, ts, |kernel| kernel.fill_follow_up(order_id, ts));
        if !events.is_empty() {
            self.sample_equity(ts);
        }
        events
    }

    fn apply_entry_event(
        &mut self,
        entry: &ScheduleEntry,
        depth: Option<DepthTick>,
        input: StepInput,
    ) -> Vec<EngineEvent> {
        let instrument = entry.instrument;

        let mut events = self.lent_step(instrument, entry.timestamp(), |kernel| match entry.data {
            ScheduleData::Bar(bar) => kernel.step(entry.local_idx, &bar, input),
            ScheduleData::Trade(tick) => kernel.step_trade(entry.local_idx, &tick, input),
            ScheduleData::Quote(quote) => kernel.step_quote(&quote),
            ScheduleData::Depth(_) => match depth {
                Some(snapshot) => kernel.step_depth(&snapshot),
                None => Vec::new(),
            },
        });
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
        if !matches!(entry.data, ScheduleData::Bar(_)) {
            self.tick_run = true;
        }
        if matches!(entry.data, ScheduleData::Quote(_) | ScheduleData::Depth(_)) {
            self.applied += 1;
            return events;
        }

        // Sample the portfolio once per distinct timestamp, after every
        // event at that timestamp: N instruments printing at the same instant
        // are one observation, not N. A later event at the same timestamp
        // replaces the sample taken after the previous one. Kernels still see
        // equity after every event, so the kill-switch reacts per event.
        let ts = entry.timestamp();
        let equity = self.sample_equity(ts);

        // The checks below are the account's. In a group they run once over
        // every member (see `crate::runtime::groups`), not once per book.
        if self.grouped.is_some() {
            self.applied += 1;
            return events;
        }

        if self.judge_deferred {
            self.applied += 1;
            return events;
        }
        self.account_checks(instrument, ts, entry.local_idx, equity, self.applied, &mut events);

        self.applied += 1;
        events
    }

    /// Judge the account's limits (margin maintenance, the day's loss, the
    /// drawdown kill-switch) once per instant, in [`PortfolioBook::judge`],
    /// instead of after every event. A runner that steps several instruments
    /// at one instant and wants the limits to see the instant's equity, with
    /// every instrument marked, not the equity between two of its steps,
    /// turns this on and calls `judge` after the last of them. Margin
    /// accounts keep the per-event check, which the kernels' own margin-call
    /// escalation needs.
    pub fn judge_per_instant(&mut self, on: bool) {
        self.judge_deferred = on;
    }

    /// Sample the book at `ts` and judge the account's limits on the sample.
    /// `idx` is the ordinal a halt latched here is reported at. For a bar on
    /// which the caller stepped no instrument as well as the end of an instant
    /// under [`PortfolioBook::judge_per_instant`]: the marks are the last
    /// stepped.
    pub fn judge(&mut self, ts: Timestamp, idx: usize) {
        let equity = self.sample_equity(ts);
        if self.grouped.is_none() {
            let mut events = Vec::new();
            self.account_checks(0, ts, idx, equity, idx, &mut events);
            self.deferred.extend(events);
        }
        self.applied += 1;
    }

    /// Take the equity sample of the event stamped `ts`: one sample per
    /// distinct timestamp, a later event at the same one replacing the
    /// sample the previous one took. Returns the equity sampled.
    #[inline]
    fn sample_equity(&mut self, ts: Timestamp) -> f64 {
        let equity = self.equity();
        if self.timestamps.last() == Some(ts) {
            self.equity_curve.pop();
            self.drawdown_curve.pop();
            self.returns.pop();
            self.timestamps.pop();
            self.peak_equity = self.peak_before_sample;
        } else {
            self.peak_before_sample = self.peak_equity;
        }
        let prev = self.equity_curve.last();
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
            self.equity_curve.drain_front(drop);
            self.drawdown_curve.drain_front(drop);
            self.returns.drain_front(drop);
            self.timestamps.drain_front(drop);
        }
        equity
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

        // No drawdown or daily-loss limit configured: the gates below cannot trip.
        if self.limits_inert {
            return;
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
                let closed = self.cleanup_all(false, Flatten::AfterEvent);
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
            let closed = kernel.flatten_after(last_idx, &last_bar, ExitReason::Liquidation);
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
        if equity > self.peak_equity {
            self.peak_equity = equity;
        }
        if let Some(dd) = self.drawdown_curve.last_mut() {
            *dd = (self.peak_equity - equity) / self.peak_equity * 100.0;
        }
        let prev = self.equity_curve.back(1);
        if let Some(ret) = self.returns.last_mut() {
            *ret = match prev {
                Some(p) if p != 0.0 => (equity - p) / p,
                _ => 0.0,
            };
        }
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
        if self.config.close_open_at_end {
            // the sample's drawdown and return follow it: the exit charged at the close-out
            // is in them too
            self.restate_last_sample(final_cash);
        } else if let Some(last) = self.equity_curve.last_mut() {
            *last = final_cash;
        }
        let equity_curve = std::mem::take(&mut self.equity_curve).into_vec();
        let drawdown_curve = std::mem::take(&mut self.drawdown_curve).into_vec();
        let returns = std::mem::take(&mut self.returns).into_vec();
        let timestamps = std::mem::take(&mut self.timestamps).into_vec();

        // In a group the strategy's curve starts from its share of the
        // account, which is not the account's capital the config carries.
        let mut run_config = self.config.clone();
        if let Some(g) = self.grouped {
            run_config.initial_capital = g.start;
        }
        // A tick run samples equity once per distinct event time, at whatever
        // spacing the tape happens to have, so the median spacing says nothing
        // about how many periods make a year. Without an explicit
        // `periods_per_year` it annualizes on observations per trading day
        // times 252 (documented in the metrics reference).
        if self.tick_run && run_config.periods_per_year.is_none() {
            run_config.periods_per_year = tick_periods_per_year(&timestamps, &self.config);
        }
        let metrics_config = &run_config;
        let metrics = compute_backtest_metrics_with_config(
            &equity_curve,
            &drawdown_curve,
            &returns,
            &self.trades,
            &timestamps,
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
                unfunded_entries: self.unfunded[i],
            })
            .collect();

        let rejected_entries: usize = self.kernels.iter().map(|k| k.rejected_entries()).sum();
        let unfunded_entries: usize = self.unfunded.iter().sum();
        let halted = self.account.is_halted() || self.kernels.iter().any(|k| k.risk_halted());
        let halted_at = self.account.halted_at();
        let dividends = self.dividends_received();

        // Submission order across instruments: by submission time, ties by
        // instrument registration order (the sort is stable).
        let mut order_log: Vec<_> = self
            .order_logs
            .iter()
            .zip(&self.kernels)
            .flat_map(|(log, kernel)| log.records(kernel))
            .collect();
        order_log.sort_by_key(|o| o.submitted_ts);

        let mut unverified: Vec<String> =
            self.kernels.iter().flat_map(|k| k.unverified_fee_rows()).collect();
        unverified.sort();
        unverified.dedup();
        let result =
            BacktestResult::new(metrics, equity_curve, drawdown_curve, self.trades, returns)
                .with_orders(order_log)
                .with_unverified_fee_rows(unverified)
                .with_halt(halted, halted_at)
                .with_refusals(rejected_entries, unfunded_entries);
        SessionOutcome {
            result,
            instruments: outcomes,
            rejected_entries,
            unfunded_entries,
            halted,
            halted_at,
            dividends,
        }
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

/// Periods per year of a tick run's equity series: the observations per
/// trading day times 252. `None` for a series too short to say.
fn tick_periods_per_year(timestamps: &[i64], config: &BacktestConfig) -> Option<f64> {
    if timestamps.len() < 2 {
        return None;
    }
    let offset = config.session_tz_offset_ns;
    let mut days: Vec<i64> =
        timestamps.iter().map(|&t| crate::portfolio::risk::day_of(t, offset)).collect();
    days.dedup();
    days.sort_unstable();
    days.dedup();
    let observations = (timestamps.len() - 1) as f64;
    Some(observations / days.len() as f64 * 252.0)
}

mod group;
mod lifecycle;

pub use group::{group_unbounded, model_shares, ModelLine, ModelLines, OptionLegs};

#[cfg(test)]
#[path = "book_tests.rs"]
mod tests;

/// Whether an event is a refusal for want of funds: an entry the account could not fund, or an
/// order the kernel rejected for capital or margin (the order path counts as the signal path does).
fn is_unfunded_refusal(event: &EngineEvent) -> bool {
    match event {
        EngineEvent::EntryRejected { reason, .. } => reason.is_unfunded(),
        EngineEvent::OrderRejected { reason, .. } => {
            crate::portfolio::risk::RejectReason::is_unfunded_str(reason)
        }
        _ => false,
    }
}
