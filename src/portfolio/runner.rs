//! Steppable single-instrument runner.
//!
//! Wraps an [`EngineKernel`] together with the per-bar bookkeeping the batch
//! engine previously kept as loop locals: equity/drawdown/return curves, peak
//! tracking, streaming trade metrics, the kill-switch feed, and end-of-data
//! finalization. Both the batch engine and per-bar drivers (e.g. the strategy
//! session exposed to Python) loop this type, so the accounting has exactly one
//! implementation.

use std::collections::HashMap;

use crate::core::types::{BacktestConfig, BacktestResult, Direction, InstrumentConfig, Trade};
use crate::execution::orders::OrderRecord;
use crate::execution::{FeeModel, FillPrice, SlippageModel};
use crate::instruments::InstrumentSpec;
use crate::metrics::streaming::StreamingMetrics;
use crate::portfolio::engine::{compute_backtest_metrics_with_config, PortfolioEngine};
use crate::portfolio::kernel::{EngineEvent, EngineKernel, KernelBar, StepInput};

/// Drives one instrument bar-by-bar and accumulates result curves.
///
/// Call [`SingleRunner::step`] once per bar in ascending index order, then
/// [`SingleRunner::finish`] to force-close any open position and compute
/// metrics.
#[derive(Debug)]
pub struct SingleRunner {
    kernel: EngineKernel,
    config: BacktestConfig,
    equity_curve: Vec<f64>,
    drawdown_curve: Vec<f64>,
    returns: Vec<f64>,
    timestamps: Vec<i64>,
    trades: Vec<Trade>,
    /// What became of every order, keyed by id. A map because one order is
    /// mutated across several events (accepted, then triggered, then filled
    /// in slices); the book supplies submission order at `finish`.
    order_events: HashMap<u64, OrderRecord>,
    streaming: StreamingMetrics,
    peak_equity: f64,
    last_bar: Option<(usize, KernelBar)>,
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
        let initial_capital = config.initial_capital;
        let risk = config.risk_gate();
        let kernel = EngineKernel::new(
            config.clone(),
            fee_model,
            slippage_model,
            fill_price,
            symbol,
            direction,
            inst_config,
        )
        .with_risk_gate(risk);

        Self {
            kernel,
            config,
            equity_curve: Vec::new(),
            drawdown_curve: Vec::new(),
            returns: Vec::new(),
            timestamps: Vec::new(),
            trades: Vec::new(),
            order_events: HashMap::new(),
            streaming: StreamingMetrics::new(),
            peak_equity: initial_capital,
            last_bar: None,
        }
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

    /// Attach an instrument market definition; see [`EngineKernel::with_instrument`].
    pub fn with_instrument(mut self, spec: InstrumentSpec) -> Self {
        self.kernel.set_instrument(spec);
        self
    }

    /// Set the position policy; see [`EngineKernel::with_position_policy`].
    pub fn with_position_policy(
        mut self,
        policy: crate::portfolio::ledger::PositionPolicy,
    ) -> Self {
        self.kernel.set_position_policy(policy);
        self
    }

    /// Set the account mode; see [`EngineKernel::with_account_mode`].
    pub fn with_account_mode(mut self, account: crate::accounts::AccountMode) -> Self {
        self.kernel.set_account_mode(account);
        self
    }

    /// Advance one bar: delegate to the kernel, then account for the outcome.
    ///
    /// The returned events are the same ones the kernel produced; completed
    /// trades have already been recorded internally.
    pub fn step(&mut self, idx: usize, bar: &KernelBar, input: StepInput) -> Vec<EngineEvent> {
        let events = self.kernel.step(idx, bar, input);

        for event in &events {
            match event {
                EngineEvent::Exited { trade, .. } => {
                    self.streaming.update(trade.return_pct / 100.0);
                    self.trades.push((**trade).clone());
                }
                // Fills and rejections carry what the book cannot: the price
                // and size of each slice, and the reason an order was
                // refused. Everything else about the order is read from the
                // book at `finish`.
                EngineEvent::OrderFilled { idx, order_id, price, size, .. } => {
                    self.order_fill(*order_id, *idx, *price, *size);
                }
                EngineEvent::OrderRejected { order_id, reason, .. } => {
                    self.order_reject(*order_id, reason);
                }
                _ => {}
            }
        }

        let equity = self.kernel.equity(bar.close);
        let prev_equity = self.equity_curve.last().copied();
        self.equity_curve.push(equity);

        if equity > self.peak_equity {
            self.peak_equity = equity;
        }
        self.drawdown_curve.push((self.peak_equity - equity) / self.peak_equity * 100.0);

        // Feed the kill-switch after this bar is marked to market, so the
        // halt takes effect from the next bar's entry check onward.
        self.kernel.observe_equity(equity, self.peak_equity);

        let ret = match prev_equity {
            Some(prev) if prev != 0.0 => (equity - prev) / prev,
            _ => 0.0,
        };
        self.returns.push(ret);
        self.timestamps.push(bar.timestamp);
        self.last_bar = Some((idx, *bar));

        events
    }

    /// Force-close any open position and compute final metrics.
    /// Fold one fill slice onto the order's record.
    ///
    /// The record is seeded from the book here rather than at submission, so
    /// an order is only materialised once something happened to it; the
    /// remaining orders are picked up wholesale at `finish`.
    fn order_fill(&mut self, order_id: u64, idx: usize, price: f64, size: f64) {
        let seed = self
            .kernel
            .order_book()
            .iter()
            .find(|o| o.id == order_id)
            .map(|o| OrderRecord::from_order(o, self.kernel.symbol()));
        if let Some(mut seed) = seed {
            // Zero the seed ONCE, as it is created: `record_fill` folds each
            // slice in, and the book's own `filled_qty` already holds the
            // total. Resetting on every slice instead would keep only the
            // last fill, which is exactly the partial-fill case this record
            // exists to describe.
            seed.filled_qty = 0.0;
            seed.avg_fill_price = None;
            let record = self.order_events.entry(order_id).or_insert(seed);
            record.record_fill(idx, price, size);
        }
    }

    /// Record why an order was refused. The reason lives only in the event.
    fn order_reject(&mut self, order_id: u64, reason: &str) {
        let seed = self
            .kernel
            .order_book()
            .iter()
            .find(|o| o.id == order_id)
            .map(|o| OrderRecord::from_order(o, self.kernel.symbol()));
        if let Some(seed) = seed {
            let record = self.order_events.entry(order_id).or_insert(seed);
            record.reject_reason = Some(reason.to_string());
        }
    }

    /// Every order the run placed, in submission order.
    ///
    /// Reconciled against the book, not assembled from events alone: an order
    /// that rested and expired emits no fill and no rejection, and omitting
    /// it would report "nothing happened" for exactly the case a user is
    /// asking about. Fill and reject detail is layered on where it exists.
    fn order_log(&self) -> Vec<OrderRecord> {
        self.kernel
            .order_book()
            .iter()
            .map(|order| match self.order_events.get(&order.id) {
                Some(seen) => {
                    let mut record = seen.clone();
                    // The book holds the final status; the event map holds
                    // what happened along the way.
                    record.status = order.status.as_str();
                    record
                }
                None => OrderRecord::from_order(order, self.kernel.symbol()),
            })
            .collect()
    }

    pub fn finish(mut self) -> BacktestResult {
        if self.kernel.is_in_position() {
            if let Some((idx, bar)) = self.last_bar {
                for trade in self.kernel.finalize_all(idx, &bar) {
                    self.streaming.update(trade.return_pct / 100.0);
                    self.trades.push(trade);
                }
            }
        }

        // Built before the result takes ownership of the parts.
        let order_log = self.order_log();

        let metrics = compute_backtest_metrics_with_config(
            &self.equity_curve,
            &self.drawdown_curve,
            &self.returns,
            &self.trades,
            &self.timestamps,
            &self.config,
        );

        BacktestResult::new(
            metrics,
            self.equity_curve,
            self.drawdown_curve,
            self.trades,
            self.returns,
        )
        .with_orders(order_log)
    }

    /// Mark-to-market equity after the most recent step, or initial capital
    /// before the first step.
    #[inline]
    pub fn equity(&self) -> f64 {
        self.equity_curve.last().copied().unwrap_or(self.config.initial_capital)
    }

    /// Current uninvested cash.
    #[inline]
    pub fn cash(&self) -> f64 {
        self.kernel.cash()
    }

    /// Whether a position is currently open.
    #[inline]
    pub fn is_in_position(&self) -> bool {
        self.kernel.is_in_position()
    }

    /// Number of bars stepped so far.
    #[inline]
    pub fn bars_seen(&self) -> usize {
        self.equity_curve.len()
    }

    /// Mutable access to the underlying kernel, for callers that adjust
    /// position state between steps (e.g. programmatic stop updates).
    #[inline]
    pub fn kernel_mut(&mut self) -> &mut EngineKernel {
        &mut self.kernel
    }

    /// Shared access to the underlying kernel.
    #[inline]
    pub fn kernel(&self) -> &EngineKernel {
        &self.kernel
    }
}
