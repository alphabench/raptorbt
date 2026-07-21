//! Steppable single-instrument runner.
//!
//! Wraps an [`EngineKernel`] together with the per-bar bookkeeping the batch
//! engine previously kept as loop locals: equity/drawdown/return curves, peak
//! tracking, streaming trade metrics, the kill-switch feed, and end-of-data
//! finalization. Both the batch engine and per-bar drivers (e.g. the strategy
//! session exposed to Python) loop this type, so the accounting has exactly one
//! implementation.

use crate::core::types::{BacktestConfig, BacktestResult, Direction, InstrumentConfig, Trade};
use crate::execution::{FeeModel, FillPrice, SlippageModel};
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

    /// Advance one bar: delegate to the kernel, then account for the outcome.
    ///
    /// The returned events are the same ones the kernel produced; completed
    /// trades have already been recorded internally.
    pub fn step(&mut self, idx: usize, bar: &KernelBar, input: StepInput) -> Vec<EngineEvent> {
        let events = self.kernel.step(idx, bar, input);

        for event in &events {
            if let EngineEvent::Exited { trade, .. } = event {
                self.streaming.update(trade.return_pct / 100.0);
                self.trades.push(trade.clone());
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
    pub fn finish(mut self) -> BacktestResult {
        if self.kernel.is_in_position() {
            if let Some((idx, bar)) = self.last_bar {
                if let Some(trade) = self.kernel.finalize(idx, &bar) {
                    self.streaming.update(trade.return_pct / 100.0);
                    self.trades.push(trade);
                }
            }
        }

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
