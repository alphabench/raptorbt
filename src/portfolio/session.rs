//! Multi-instrument event session.
//!
//! Drives N instruments' bar streams as one deterministically merged event
//! schedule (via [`EventFeed`]) against per-instrument kernels sharing a
//! single cash pool — the class-contract counterpart of the array
//! portfolio runner, reusing its pool discipline: each kernel is pointed at
//! the pool before stepping and drained back after, so capital committed to
//! one instrument is unavailable to the others.
//!
//! Portfolio equity is sampled once per schedule event: pool plus every
//! instrument's position value at its last known close. Cash accounts only
//! for now — a margin account shared across kernels needs the account
//! handle planned for a later 0.5.x release.

use crate::core::types::{BacktestConfig, BacktestResult, Direction, InstrumentConfig, Trade};
use crate::data::{EventFeed, EventPayload, MarketEvent};
use crate::instruments::InstrumentSpec;
use crate::metrics::streaming::StreamingMetrics;
use crate::portfolio::engine::compute_backtest_metrics_with_config;
use crate::portfolio::kernel::{EngineEvent, EngineKernel, KernelBar, StepInput};
use crate::portfolio::ledger::PositionPolicy;
use crate::core::types::OhlcvBar;

/// One entry of the merged schedule.
#[derive(Debug, Clone, Copy)]
pub struct ScheduleEntry {
    pub instrument: usize,
    pub local_idx: usize,
    pub bar: KernelBar,
}

/// Per-instrument outcome summary.
#[derive(Debug, Clone)]
pub struct InstrumentOutcome {
    pub symbol: String,
    pub trades: usize,
    pub pnl: f64,
    pub rejected_entries: usize,
}

/// Multi-instrument session over merged bar streams.
pub struct EventSession {
    config: BacktestConfig,
    kernels: Vec<EngineKernel>,
    symbols: Vec<String>,
    bars: Vec<Vec<KernelBar>>,
    schedule: Vec<ScheduleEntry>,
    cursor: usize,
    pool: f64,
    last_close: Vec<Option<f64>>,
    last_seen: Vec<Option<(usize, KernelBar)>>,
    equity_curve: Vec<f64>,
    drawdown_curve: Vec<f64>,
    returns: Vec<f64>,
    timestamps: Vec<i64>,
    trades: Vec<Trade>,
    streaming: StreamingMetrics,
    peak_equity: f64,
    sealed: bool,
}

impl EventSession {
    pub fn new(config: BacktestConfig) -> Self {
        let pool = config.initial_capital;
        Self {
            config,
            kernels: Vec::new(),
            symbols: Vec::new(),
            bars: Vec::new(),
            schedule: Vec::new(),
            cursor: 0,
            pool,
            last_close: Vec::new(),
            last_seen: Vec::new(),
            equity_curve: Vec::new(),
            drawdown_curve: Vec::new(),
            returns: Vec::new(),
            timestamps: Vec::new(),
            trades: Vec::new(),
            streaming: StreamingMetrics::new(),
            peak_equity: pool,
            sealed: false,
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
            symbol.clone(),
            direction,
            inst_config,
        )
        .with_risk_gate(self.config.risk_gate())
        .with_position_policy(policy);
        if let Some(spec) = spec {
            kernel.set_instrument(spec);
        }
        // The pool owns all capital; kernels borrow it per step.
        kernel.set_cash(0.0);
        self.kernels.push(kernel);
        self.symbols.push(symbol);
        self.bars.push(Vec::new());
        self.last_close.push(None);
        self.last_seen.push(None);
        self.kernels.len() - 1
    }

    /// Attach an instrument's bar series (ascending timestamps).
    pub fn set_bars(&mut self, instrument: usize, bars: Vec<KernelBar>) {
        self.bars[instrument] = bars;
    }

    /// Merge all streams into the deterministic schedule. Idempotent.
    pub fn seal(&mut self) {
        if self.sealed {
            return;
        }
        let mut feed = EventFeed::new();
        for (i, bars) in self.bars.iter().enumerate() {
            let events: Vec<MarketEvent> = bars
                .iter()
                .map(|b| MarketEvent {
                    instrument: i as u32,
                    stream: i as u32,
                    payload: EventPayload::Bar(OhlcvBar {
                        timestamp: b.timestamp,
                        open: b.open,
                        high: b.high,
                        low: b.low,
                        close: b.close,
                        volume: b.volume,
                    }),
                })
                .collect();
            feed.add_stream(events);
        }
        let mut counters = vec![0usize; self.bars.len()];
        for event in feed {
            let instrument = event.instrument as usize;
            if let EventPayload::Bar(bar) = event.payload {
                let local_idx = counters[instrument];
                counters[instrument] += 1;
                self.schedule.push(ScheduleEntry {
                    instrument,
                    local_idx,
                    bar: KernelBar {
                        timestamp: bar.timestamp,
                        open: bar.open,
                        high: bar.high,
                        low: bar.low,
                        close: bar.close,
                        volume: bar.volume,
                    },
                });
            }
        }
        self.sealed = true;
    }

    /// Total scheduled events.
    pub fn len(&self) -> usize {
        self.schedule.len()
    }

    pub fn is_empty(&self) -> bool {
        self.schedule.is_empty()
    }

    /// The entry the cursor points at, if any.
    pub fn current(&self) -> Option<ScheduleEntry> {
        self.schedule.get(self.cursor).copied()
    }

    /// Kernel of an instrument, for order routing and queries.
    pub fn kernel_mut(&mut self, instrument: usize) -> &mut EngineKernel {
        &mut self.kernels[instrument]
    }

    pub fn kernel(&self, instrument: usize) -> &EngineKernel {
        &self.kernels[instrument]
    }

    /// Portfolio equity: pool plus each instrument's positions marked at
    /// its last known close.
    pub fn equity(&self) -> f64 {
        let positions: f64 = self
            .kernels
            .iter()
            .zip(&self.last_close)
            .map(|(k, close)| close.map(|c| k.position_value(c)).unwrap_or(0.0))
            .sum();
        self.pool + positions
    }

    /// Uncommitted shared cash.
    pub fn cash(&self) -> f64 {
        self.pool
    }

    /// Step the current schedule entry through its kernel and advance.
    ///
    /// The pool dance mirrors the array portfolio runner: the kernel gets
    /// the whole pool, steps, and returns what it did not commit.
    pub fn apply_current(&mut self, input: StepInput) -> Vec<EngineEvent> {
        let Some(entry) = self.current() else { return Vec::new() };
        let instrument = entry.instrument;

        let kernel = &mut self.kernels[instrument];
        kernel.set_cash(self.pool);
        let events = kernel.step(entry.local_idx, &entry.bar, input);
        self.pool = kernel.cash();
        kernel.set_cash(0.0);

        for event in &events {
            if let EngineEvent::Exited { trade, .. } = event {
                self.streaming.update(trade.return_pct / 100.0);
                self.trades.push(trade.clone());
            }
        }

        self.last_close[instrument] = Some(entry.bar.close);
        self.last_seen[instrument] = Some((entry.local_idx, entry.bar));

        // Sample the portfolio once per event; feed every kernel's
        // kill-switch so a portfolio-level drawdown halts all entries.
        let equity = self.equity();
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
        self.timestamps.push(entry.bar.timestamp);
        let peak = self.peak_equity;
        for kernel in &mut self.kernels {
            kernel.observe_equity(equity, peak);
        }

        self.cursor += 1;
        events
    }

    /// Force-close every instrument at its last seen bar and compute
    /// portfolio metrics.
    pub fn finish(mut self) -> (BacktestResult, Vec<InstrumentOutcome>) {
        for i in 0..self.kernels.len() {
            if let Some((idx, bar)) = self.last_seen[i] {
                let kernel = &mut self.kernels[i];
                kernel.set_cash(self.pool);
                for trade in kernel.finalize_all(idx, &bar) {
                    self.streaming.update(trade.return_pct / 100.0);
                    self.trades.push(trade);
                }
                self.pool = kernel.cash();
                kernel.set_cash(0.0);
                self.last_close[i] = None;
            }
        }
        // Positions are flat; the final mark is the pool itself.
        if let Some(last) = self.equity_curve.last_mut() {
            *last = self.pool;
        }

        let metrics = compute_backtest_metrics_with_config(
            &self.equity_curve,
            &self.drawdown_curve,
            &self.returns,
            &self.trades,
            &self.timestamps,
            &self.config,
        );
        let outcomes = self
            .symbols
            .iter()
            .enumerate()
            .map(|(i, symbol)| InstrumentOutcome {
                symbol: symbol.clone(),
                trades: self.trades.iter().filter(|t| &t.symbol == symbol).count(),
                pnl: self.trades.iter().filter(|t| &t.symbol == symbol).map(|t| t.pnl).sum(),
                rejected_entries: self.kernels[i].rejected_entries(),
            })
            .collect();

        let result = BacktestResult::new(
            metrics,
            self.equity_curve,
            self.drawdown_curve,
            self.trades,
            self.returns,
        );
        (result, outcomes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bars(start_ts: i64, closes: &[f64]) -> Vec<KernelBar> {
        closes
            .iter()
            .enumerate()
            .map(|(i, &c)| KernelBar {
                timestamp: start_ts + i as i64 * 10,
                open: c,
                high: c + 1.0,
                low: c - 1.0,
                close: c,
                volume: 1_000.0,
            })
            .collect()
    }

    fn session_two_instruments() -> EventSession {
        let config = BacktestConfig { fees: 0.0, ..BacktestConfig::default() };
        let mut session = EventSession::new(config);
        let a = session.add_instrument(
            "AAA".into(),
            Direction::Long,
            None,
            None,
            PositionPolicy::Net,
        );
        let b = session.add_instrument(
            "BBB".into(),
            Direction::Long,
            None,
            None,
            PositionPolicy::Net,
        );
        // Interleaved timestamps: AAA at 0,10,20..., BBB offset by 5.
        session.set_bars(a, bars(0, &[100.0, 101.0, 102.0]));
        session.set_bars(b, bars(5, &[50.0, 51.0, 52.0]));
        session.seal();
        session
    }

    #[test]
    fn schedule_interleaves_deterministically() {
        let mut session = session_two_instruments();
        let mut order = Vec::new();
        while let Some(entry) = session.current() {
            order.push((entry.instrument, entry.local_idx, entry.bar.timestamp));
            session.apply_current(StepInput::default());
        }
        assert_eq!(
            order,
            vec![(0, 0, 0), (1, 0, 5), (0, 1, 10), (1, 1, 15), (0, 2, 20), (1, 2, 25)]
        );
    }

    #[test]
    fn shared_pool_constrains_second_instrument() {
        let mut session = session_two_instruments();
        // Enter AAA with everything on its first bar.
        session.apply_current(StepInput { entry: true, ..StepInput::default() });
        assert!(session.kernel(0).is_in_position());
        let cash_after_a = session.cash();
        assert!(cash_after_a < 1.0, "pool should be nearly spent, got {cash_after_a}");

        // BBB tries to enter with an empty pool: zero-size rejection.
        let events = session.apply_current(StepInput { entry: true, ..StepInput::default() });
        assert!(events
            .iter()
            .any(|e| matches!(e, EngineEvent::EntryRejected { .. })));
        assert!(!session.kernel(1).is_in_position());
    }

    #[test]
    fn equity_marks_both_instruments() {
        let mut session = session_two_instruments();
        // Enter AAA with half the pool.
        session.apply_current(StepInput {
            entry: true,
            size_mult: Some(0.5),
            ..StepInput::default()
        });
        // BBB enters with what remains.
        session.apply_current(StepInput { entry: true, ..StepInput::default() });
        assert!(session.kernel(0).is_in_position());
        assert!(session.kernel(1).is_in_position());

        // Run out the schedule; both drift up 2 points.
        while session.current().is_some() {
            session.apply_current(StepInput::default());
        }
        let equity = session.equity();
        assert!(equity > 100_000.0, "both positions gained, equity {equity}");

        let (result, outcomes) = session.finish();
        assert_eq!(result.trades.len(), 2); // both force-closed at end
        assert_eq!(outcomes.len(), 2);
        assert!(outcomes.iter().all(|o| o.trades == 1));
        let total_pnl: f64 = outcomes.iter().map(|o| o.pnl).sum();
        assert!(total_pnl > 0.0);
    }
}
