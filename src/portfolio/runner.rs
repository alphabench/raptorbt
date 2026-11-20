//! Steppable single-instrument runner, and the order log every book keeps.
//!
//! The runner is a one-instrument [`PortfolioBook`]: both the batch engine and
//! per-bar drivers (e.g. the strategy session exposed to Python) loop it, and
//! it simulates nothing the book does not, so a single instrument and a
//! portfolio follow one set of rules.

use std::collections::HashMap;

use crate::accounts::AccountMode;
use crate::core::types::{BacktestConfig, BacktestResult, Direction, InstrumentConfig};
use crate::execution::orders::OrderRecord;
use crate::execution::{FeeModel, FillPrice, SlippageModel};
use crate::instruments::{AdjustmentPolicy, CorporateAction, InstrumentSpec};
use crate::portfolio::book::PortfolioBook;
use crate::portfolio::engine::PortfolioEngine;
use crate::portfolio::kernel::{EngineEvent, EngineKernel, KernelBar, StepInput};
use crate::portfolio::ledger::PositionPolicy;
use crate::portfolio::session::{ScheduleData, ScheduleEntry};

/// What became of every order one kernel handled.
///
/// A map because one order is mutated across several events (accepted, then
/// triggered, then filled in slices); the kernel's book supplies submission
/// order when the records are read out.
#[derive(Debug, Default)]
pub(crate) struct OrderLog {
    events: HashMap<u64, OrderRecord>,
}

impl OrderLog {
    /// Fold a fill or rejection event onto the order's record.
    ///
    /// Fills and rejections carry what the book cannot: the price and size of
    /// each slice, and the reason an order was refused. Everything else about
    /// the order is read from the book in [`OrderLog::records`]. Other events
    /// are ignored.
    ///
    /// The record is seeded from the book here rather than at submission, so
    /// an order is only materialised once something happened to it; the
    /// remaining orders are picked up wholesale by `records`.
    pub(crate) fn observe(&mut self, kernel: &EngineKernel, event: &EngineEvent) {
        let (order_id, fill, reason) = match event {
            EngineEvent::OrderFilled { idx, order_id, price, size, .. } => {
                (*order_id, Some((*idx, *price, *size)), None)
            }
            EngineEvent::OrderRejected { order_id, reason, .. } => (*order_id, None, Some(reason)),
            _ => return,
        };
        let Some(order) = kernel.order(order_id) else { return };
        let mut seed = OrderRecord::from_order(order, kernel.symbol());
        if fill.is_some() {
            // Zero the seed ONCE, as it is created: `record_fill` folds each
            // slice in, and the book's own `filled_qty` already holds the
            // total. Resetting on every slice instead would keep only the
            // last fill, which is exactly the partial-fill case this record
            // exists to describe.
            seed.filled_qty = 0.0;
            seed.avg_fill_price = None;
        }
        let record = self.events.entry(order_id).or_insert(seed);
        if let Some((idx, price, size)) = fill {
            record.record_fill(idx, price, size);
        }
        if let Some(reason) = reason {
            record.reject_reason = Some(reason.to_string());
        }
    }

    /// Every order the kernel handled, in submission order.
    ///
    /// Reconciled against the book, not assembled from events alone: an order
    /// that rested and expired emits no fill and no rejection, and omitting
    /// it would report "nothing happened" for exactly the case a user is
    /// asking about. Fill and reject detail is layered on where it exists.
    pub(crate) fn records(&self, kernel: &EngineKernel) -> Vec<OrderRecord> {
        kernel
            .order_book()
            .iter()
            .map(|order| match self.events.get(&order.id) {
                Some(seen) => {
                    let mut record = seen.clone();
                    // The book holds the final status; the event map holds
                    // what happened along the way.
                    record.status = order.status.as_str();
                    record
                }
                None => OrderRecord::from_order(order, kernel.symbol()),
            })
            .collect()
    }
}

/// What a single-instrument run registers in its book, kept so a builder
/// call before the first step can register it again.
#[derive(Debug, Clone)]
struct Setup {
    symbol: String,
    direction: Direction,
    inst_config: Option<InstrumentConfig>,
    fee_model: FeeModel,
    slippage_model: SlippageModel,
    fill_price: FillPrice,
    spec: Option<InstrumentSpec>,
    policy: PositionPolicy,
    account: AccountMode,
}

/// Drives one instrument bar-by-bar.
///
/// A one-instrument [`PortfolioBook`]: the account, the kernel, the sampled
/// curves, expiry and delivery, corporate actions, the day and drawdown
/// limits and the refusal counts are the book's, so a single instrument and a
/// portfolio of many are simulated by one implementation. This type only
/// turns a bar and its signals into the schedule entry the book takes.
///
/// Call [`SingleRunner::step`] once per bar in ascending index order, then
/// [`SingleRunner::finish`] to force-close any open position and compute
/// metrics.
pub struct SingleRunner {
    book: PortfolioBook,
    config: BacktestConfig,
    setup: Setup,
}

impl SingleRunner {
    /// Build a runner from engine-level models and optional per-instrument config.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        config: BacktestConfig,
        fee_model: FeeModel,
        slippage_model: SlippageModel,
        fill_price: FillPrice,
        symbol: String,
        direction: Direction,
        inst_config: Option<&InstrumentConfig>,
    ) -> Self {
        let setup = Setup {
            symbol,
            direction,
            inst_config: inst_config.cloned(),
            fee_model,
            slippage_model,
            fill_price,
            spec: None,
            policy: PositionPolicy::default(),
            account: AccountMode::default(),
        };
        let book = Self::build(&config, &setup);
        Self { book, config, setup }
    }

    /// Build a runner deriving fee/slippage/fill models from the config, the
    /// same way [`PortfolioEngine::new`] does.
    pub fn from_config(
        config: BacktestConfig,
        symbol: String,
        direction: Direction,
        inst_config: Option<&InstrumentConfig>,
    ) -> Self {
        let engine = PortfolioEngine::new(config);
        Self::new(
            engine.config.clone(),
            engine.fee_model.clone(),
            engine.slippage_model.clone(),
            engine.fill_price,
            symbol,
            direction,
            inst_config,
        )
    }

    fn build(config: &BacktestConfig, setup: &Setup) -> PortfolioBook {
        let mut book = PortfolioBook::new(config.clone(), setup.account);
        book.add_instrument_with_models(
            setup.symbol.clone(),
            setup.direction,
            setup.spec.clone(),
            setup.inst_config.as_ref(),
            setup.policy,
            setup.fee_model.clone(),
            setup.slippage_model.clone(),
            setup.fill_price,
        );
        book
    }

    /// Register the instrument again with the setup changed. Builders run
    /// before the first step, when the book holds nothing yet.
    fn rebuilt(mut self) -> Self {
        self.book = Self::build(&self.config, &self.setup);
        self
    }

    /// Attach an instrument market definition; see [`EngineKernel::set_instrument`].
    pub fn with_instrument(mut self, spec: InstrumentSpec) -> Self {
        self.setup.spec = Some(spec);
        self.rebuilt()
    }

    /// Set the position policy; see [`EngineKernel::with_position_policy`].
    pub fn with_position_policy(mut self, policy: PositionPolicy) -> Self {
        self.setup.policy = policy;
        self.rebuilt()
    }

    /// Set the account mode; see [`EngineKernel::with_account_mode`].
    pub fn with_account_mode(mut self, account: AccountMode) -> Self {
        self.setup.account = account;
        self.rebuilt()
    }

    /// Give the run the corporate actions of its instrument and the policy
    /// its bars are on; see [`PortfolioBook::set_corporate_actions`]. Call
    /// before the first step.
    pub fn set_corporate_actions(
        &mut self,
        actions: &[CorporateAction],
        policy: AdjustmentPolicy,
    ) -> crate::core::Result<()> {
        self.book.set_corporate_actions(actions, policy)
    }

    /// Dividend cash credited so far (a short's debit is negative).
    pub fn dividends_received(&self) -> f64 {
        self.book.dividends_received()
    }

    /// Advance one bar: the book's lifecycle, then the kernel's step, then
    /// the account's checks and the equity sample.
    ///
    /// The returned events are the ones the book produced; completed trades
    /// have already been recorded internally.
    pub fn step(&mut self, idx: usize, bar: &KernelBar, input: StepInput) -> Vec<EngineEvent> {
        let entry = ScheduleEntry { instrument: 0, local_idx: idx, data: ScheduleData::Bar(*bar) };
        self.book.apply(&entry, None, input)
    }

    /// Meet what rests for bar `idx` before the strategy decides on it (see
    /// [`PortfolioBook::rest_bar`]): the stops, targets and resting orders of earlier bars trade
    /// the bar's range and the events are returned for the strategy to hear. [`KernelRunner::step`]
    /// carries on from here with the bar's decisions. Nothing when nothing rests, or when the
    /// book has a lifecycle (corporate actions, expiries) to apply first, which `step` then
    /// does whole.
    pub fn rest(&mut self, idx: usize, bar: &KernelBar) -> Vec<EngineEvent> {
        let entry = ScheduleEntry { instrument: 0, local_idx: idx, data: ScheduleData::Bar(*bar) };
        if self.book.lifecycle_active() || !self.book.can_rest(&entry) {
            return Vec::new();
        }
        self.book.rest_bar(&entry)
    }

    /// Halt this strategy: new entries are refused as `killed` from now on.
    /// Working orders are cancelled, and open positions closed when
    /// `flatten`, at the start of the next step (at the last bar's close).
    /// Only the first halt's reason is kept; a later `flatten` still counts.
    pub fn halt(&mut self, reason: &str, flatten: bool) {
        self.book.halt(reason, flatten);
    }

    /// Whether a halt (explicit, the drawdown limit or a margin call) has
    /// latched.
    pub fn is_halted(&self) -> bool {
        self.book.is_halted()
    }

    /// Room for `bars` equity samples; see [`PortfolioBook::reserve_curves`].
    pub fn reserve_curves(&mut self, bars: usize) {
        self.book.reserve_curves(bars);
    }

    /// Force-close any open position and compute final metrics.
    pub fn finish(self) -> BacktestResult {
        self.book.finish().result
    }

    /// Mark-to-market equity after the most recent step, or initial capital
    /// before the first step.
    #[inline]
    pub fn equity(&self) -> f64 {
        self.book.last_equity().unwrap_or(self.config.initial_capital)
    }

    /// Current uninvested cash.
    #[inline]
    pub fn cash(&self) -> f64 {
        self.book.cash()
    }

    /// Capital free to fund a new position: cash, less the margin locked in
    /// a margin account.
    #[inline]
    pub fn free_capital(&self) -> f64 {
        self.book.free_capital()
    }

    /// Whether a position is currently open.
    #[inline]
    pub fn is_in_position(&self) -> bool {
        self.book.kernel(0).is_in_position()
    }

    /// Number of bars stepped so far.
    #[inline]
    pub fn bars_seen(&self) -> usize {
        self.book.applied()
    }

    /// Mutable access to the instrument's kernel, for callers that submit
    /// and cancel orders or adjust position state between steps. The kernel
    /// holds no cash between steps: the book lends it the account's during
    /// one, so read funds through [`SingleRunner::cash`] and
    /// [`SingleRunner::free_capital`].
    #[inline]
    pub fn kernel_mut(&mut self) -> &mut EngineKernel {
        self.book.kernel_mut(0)
    }

    /// Shared access to the instrument's kernel; see
    /// [`SingleRunner::kernel_mut`] on cash.
    #[inline]
    pub fn kernel(&self) -> &EngineKernel {
        self.book.kernel(0)
    }
}

#[cfg(test)]
mod halt_tests {
    use super::*;
    use crate::core::types::ExitReason;

    fn bar(ts: i64, close: f64) -> KernelBar {
        KernelBar {
            timestamp: ts,
            open: close,
            high: close + 1.0,
            low: close - 1.0,
            close,
            volume: 1000.0,
        }
    }

    #[test]
    fn a_flattening_halt_books_one_trade_at_the_close_of_the_bar_it_was_raised_on() {
        let config = BacktestConfig { fees: 0.0, ..BacktestConfig::default() };
        let mut runner = SingleRunner::from_config(config, "S".to_string(), Direction::Long, None);
        let enter = StepInput { entry: true, ..StepInput::default() };
        runner.step(0, &bar(0, 100.0), enter);
        runner.step(1, &bar(1, 100.0), StepInput::default());
        runner.halt("manual", true);
        let events = runner.step(2, &bar(2, 90.0), StepInput::default());
        let exits: Vec<_> = events
            .iter()
            .filter_map(|e| match e {
                EngineEvent::Exited { trade, .. } => Some(trade),
                _ => None,
            })
            .collect();
        assert_eq!(exits.len(), 1);
        assert_eq!(exits[0].exit_reason, ExitReason::Liquidation);
        // Corrected (was the previous bar's close, 100.0, on bar 1): the halt asked for
        // before bar 2 closes like an ordinary reduce-only market order, at bar 2's close.
        assert_eq!(exits[0].exit_price, 90.0);
        let result = runner.finish();
        assert_eq!(result.trades.len(), 1, "the flattened trade is booked once");
        assert_eq!(result.trades[0].exit_idx, 2, "on the bar of the close it trades at");
    }
}
