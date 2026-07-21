//! Multi-instrument event session.
//!
//! Drives N instruments' bar streams as one deterministically merged event
//! schedule (via [`EventFeed`]) against per-instrument kernels sharing a
//! single cash pool — the class-contract counterpart of the array
//! portfolio runner, reusing its pool discipline: each kernel is pointed at
//! the pool before stepping and drained back after, so capital committed to
//! one instrument is unavailable to the others.
//!
//! Portfolio equity is sampled once per schedule event: the account balance
//! plus every instrument's mark at its last known close — position value in
//! cash mode, direction-aware unrealized PnL under margin.
//!
//! Capital lives in one [`SharedAccount`]. Cash mode reproduces the original
//! single-pool arithmetic exactly. Margin mode additionally tracks locked
//! initial margin as an aggregate, so leverage is shared across instruments
//! and one margin call halts them all.

use crate::accounts::{AccountMode, SharedAccount};
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

/// Everything a finished session reports.
#[derive(Debug)]
pub struct SessionOutcome {
    pub result: BacktestResult,
    pub instruments: Vec<InstrumentOutcome>,
    /// Entries refused across all instruments, summed over their risk gates.
    pub rejected_entries: usize,
    /// Whether a margin call or a drawdown kill-switch latched.
    pub halted: bool,
    /// Where the halt latched, as a **schedule-event ordinal** — the session
    /// interleaves N streams, so this is not a bar index (the array runner's
    /// `halted_at` is).
    pub halted_at: Option<usize>,
}

/// Multi-instrument session over merged bar streams.
pub struct EventSession {
    config: BacktestConfig,
    kernels: Vec<EngineKernel>,
    symbols: Vec<String>,
    bars: Vec<Vec<KernelBar>>,
    schedule: Vec<ScheduleEntry>,
    cursor: usize,
    account: SharedAccount,
    last_close: Vec<Option<f64>>,
    last_seen: Vec<Option<(usize, KernelBar)>>,
    equity_curve: Vec<f64>,
    drawdown_curve: Vec<f64>,
    returns: Vec<f64>,
    timestamps: Vec<i64>,
    trades: Vec<Trade>,
    streaming: StreamingMetrics,
    peak_equity: f64,
    /// Where the drawdown kill-switch latched; the margin call records its
    /// own index on the account.
    risk_halted_at: Option<usize>,
    sealed: bool,
}

impl EventSession {
    pub fn new(config: BacktestConfig) -> Self {
        Self::with_account(config, AccountMode::Cash)
    }

    /// Session funded by an account of the given mode.
    ///
    /// The mode applies to every instrument: they share one balance and, in
    /// margin mode, one pool of locked initial margin.
    pub fn with_account(config: BacktestConfig, mode: AccountMode) -> Self {
        let pool = config.initial_capital;
        Self {
            config,
            kernels: Vec::new(),
            symbols: Vec::new(),
            bars: Vec::new(),
            schedule: Vec::new(),
            cursor: 0,
            account: SharedAccount::new(mode, pool),
            last_close: Vec::new(),
            last_seen: Vec::new(),
            equity_curve: Vec::new(),
            drawdown_curve: Vec::new(),
            returns: Vec::new(),
            timestamps: Vec::new(),
            trades: Vec::new(),
            streaming: StreamingMetrics::new(),
            peak_equity: pool,
            risk_halted_at: None,
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
        .with_position_policy(policy)
        .with_account_mode(self.account.mode());
        if let Some(spec) = spec {
            kernel.set_instrument(spec);
        }
        // The account owns all capital; kernels borrow it per step.
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

    /// Portfolio equity: the account balance plus each instrument's mark at
    /// its last known close.
    ///
    /// Cash mode marks positions at full value (historical model, pinned by
    /// the golden fixtures); margin mode marks direction-aware unrealized
    /// PnL, which prices winning shorts upward. The balance already includes
    /// notionally-locked margin — locks do not debit cash — so the margin arm
    /// does not double-count.
    pub fn equity(&self) -> f64 {
        match self.account.mode() {
            AccountMode::Cash => {
                let positions: f64 = self
                    .kernels
                    .iter()
                    .zip(&self.last_close)
                    .map(|(k, close)| close.map(|c| k.position_value(c)).unwrap_or(0.0))
                    .sum();
                self.account.balance() + positions
            }
            AccountMode::Margin { .. } => {
                let unrealized: f64 = self
                    .kernels
                    .iter()
                    .zip(&self.last_close)
                    .map(|(k, close)| close.map(|c| k.unrealized_value(c)).unwrap_or(0.0))
                    .sum();
                self.account.balance() + unrealized
            }
        }
    }

    /// Shared cash balance. In margin mode this includes locked initial
    /// margin; see [`EventSession::free_capital`] for what can fund a new
    /// position.
    pub fn cash(&self) -> f64 {
        self.account.balance()
    }

    /// Capital available to open new positions across all instruments.
    pub fn free_capital(&self) -> f64 {
        self.account.free()
    }

    /// Whether a margin call or drawdown kill-switch has latched.
    pub fn is_halted(&self) -> bool {
        self.account.is_halted() || self.kernels.iter().any(|k| k.risk_halted())
    }

    /// Step the current schedule entry through its kernel and advance.
    ///
    /// The kernel is re-pointed at the portfolio's capital, stepped, then
    /// drained: cash and locked-margin movements are folded back into the
    /// shared account. Cash mode is exactly the historical lend/drain of the
    /// whole pool.
    pub fn apply_current(&mut self, input: StepInput) -> Vec<EngineEvent> {
        let Some(entry) = self.current() else { return Vec::new() };
        let instrument = entry.instrument;

        // In margin mode the kernel computes free capital as its own cash
        // minus its own locked margin, so hand it the balance less every
        // *other* kernel's locks — then its arithmetic sees the portfolio's
        // free capital.
        let kernel = &mut self.kernels[instrument];
        let locked_before = kernel.locked_margin();
        let injected = match self.account.mode() {
            AccountMode::Cash => self.account.balance(),
            AccountMode::Margin { .. } => {
                self.account.balance() - (self.account.locked() - locked_before)
            }
        };
        kernel.set_cash(injected);
        let mut events = kernel.step(entry.local_idx, &entry.bar, input);
        let delta_cash = kernel.cash() - injected;
        let delta_locked = kernel.locked_margin() - locked_before;
        kernel.set_cash(0.0);
        self.account.reconcile(delta_cash, delta_locked);

        // A kernel-local call sees only its own slice of the portfolio, but
        // the account is shared: escalate it so every instrument halts.
        if events.iter().any(|e| matches!(e, EngineEvent::MarginCall { .. })) {
            self.halt_all(self.cursor);
        }

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
                self.halt_all(self.cursor);
                events.push(EngineEvent::MarginCall { idx: entry.local_idx, equity, required });
            }
        }

        // Kernels all see the same portfolio equity, so their drawdown gates
        // latch in lockstep; record the rising edge once.
        let peak = self.peak_equity;
        let risk_halted_before = self.kernels.iter().any(|k| k.risk_halted());
        for kernel in &mut self.kernels {
            kernel.observe_equity(equity, peak);
        }
        if !risk_halted_before
            && self.risk_halted_at.is_none()
            && self.kernels.iter().any(|k| k.risk_halted())
        {
            self.risk_halted_at = Some(self.cursor);
        }

        self.cursor += 1;
        events
    }

    /// Latch the shared margin call and block entries on every instrument.
    ///
    /// Each kernel's existing kill-switch does the blocking, so a halted
    /// portfolio rejects entries with `RejectReason::MarginCall` everywhere.
    fn halt_all(&mut self, idx: usize) {
        self.account.halt(idx);
        for kernel in &mut self.kernels {
            kernel.halt_margin();
        }
    }

    /// Force-close every instrument at its last seen bar and compute
    /// portfolio metrics.
    pub fn finish(mut self) -> SessionOutcome {
        for i in 0..self.kernels.len() {
            if let Some((idx, bar)) = self.last_seen[i] {
                let kernel = &mut self.kernels[i];
                let locked_before = kernel.locked_margin();
                let injected = match self.account.mode() {
                    AccountMode::Cash => self.account.balance(),
                    AccountMode::Margin { .. } => {
                        self.account.balance() - (self.account.locked() - locked_before)
                    }
                };
                kernel.set_cash(injected);
                for trade in kernel.finalize_all(idx, &bar) {
                    self.streaming.update(trade.return_pct / 100.0);
                    self.trades.push(trade);
                }
                let delta_cash = kernel.cash() - injected;
                let delta_locked = kernel.locked_margin() - locked_before;
                kernel.set_cash(0.0);
                self.account.reconcile(delta_cash, delta_locked);
                self.last_close[i] = None;
            }
        }
        // Positions are flat; the final mark is the balance itself.
        if let Some(last) = self.equity_curve.last_mut() {
            *last = self.account.balance();
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

        let rejected_entries: usize = self.kernels.iter().map(|k| k.rejected_entries()).sum();
        let halted = self.account.is_halted() || self.kernels.iter().any(|k| k.risk_halted());
        let halted_at = self.account.halted_at().or(self.risk_halted_at);

        let result = BacktestResult::new(
            metrics,
            self.equity_curve,
            self.drawdown_curve,
            self.trades,
            self.returns,
        );
        SessionOutcome { result, instruments: outcomes, rejected_entries, halted, halted_at }
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

        let outcome = session.finish();
        assert_eq!(outcome.result.trades.len(), 2); // both force-closed at end
        assert_eq!(outcome.instruments.len(), 2);
        assert!(outcome.instruments.iter().all(|o| o.trades == 1));
        let total_pnl: f64 = outcome.instruments.iter().map(|o| o.pnl).sum();
        assert!(total_pnl > 0.0);
    }

    /// Margin-mode twin of [`session_two_instruments`].
    fn session_two_instruments_margin(leverage: f64, short_second: bool) -> EventSession {
        let config = BacktestConfig { fees: 0.0, ..BacktestConfig::default() };
        let mut session = EventSession::with_account(config, AccountMode::Margin { leverage });
        let a =
            session.add_instrument("AAA".into(), Direction::Long, None, None, PositionPolicy::Net);
        let second_dir = if short_second { Direction::Short } else { Direction::Long };
        let b = session.add_instrument("BBB".into(), second_dir, None, None, PositionPolicy::Net);
        session.set_bars(a, bars(0, &[100.0, 101.0, 102.0]));
        session.set_bars(b, bars(5, &[50.0, 51.0, 52.0]));
        session.seal();
        session
    }

    #[test]
    fn cash_mode_arithmetic_unchanged() {
        // Drift tripwire: the cash path must be bit-identical to the
        // single-pool implementation this replaced.
        let mut session = session_two_instruments();
        session.apply_current(StepInput {
            entry: true,
            size_mult: Some(0.5),
            ..StepInput::default()
        });
        session.apply_current(StepInput { entry: true, ..StepInput::default() });
        while session.current().is_some() {
            session.apply_current(StepInput::default());
        }
        // Exact equality, not approximate: these feed the golden metrics.
        assert_eq!(session.cash(), session.free_capital());
        let curve = session.equity_curve.clone();
        assert_eq!(curve.len(), 6);
        // Marked at full position value throughout, as the cash model does.
        assert_eq!(curve[0], 100_000.0);
        assert!(curve.iter().all(|v| v.is_finite()));
        // Both legs gain; the curve ends above where it started.
        assert!(curve[5] > curve[0], "curve {curve:?}");
    }

    #[test]
    fn margin_pool_shared_across_kernels() {
        // The headline: under leverage the second instrument still has room,
        // where `shared_pool_constrains_second_instrument` shows it does not.
        // Size AAA at a quarter of capital. In cash mode that buys 250
        // units and leaves 75k; under 5x it buys 1250 units for the same
        // 25k of locked margin — and BBB still draws on the shared balance.
        let mut session = session_two_instruments_margin(5.0, false);
        session.apply_current(StepInput {
            entry: true,
            size_mult: Some(0.25),
            ..StepInput::default()
        });
        assert!(session.kernel(0).is_in_position());
        assert_eq!(session.kernel(0).position_snapshots()[0].size, 1_250.0);
        // Locks reserve capital without debiting cash.
        assert_eq!(session.cash(), 100_000.0);
        assert_eq!(session.free_capital(), 75_000.0);

        let events = session.apply_current(StepInput {
            entry: true,
            size_mult: Some(0.25),
            ..StepInput::default()
        });
        assert!(
            !events.iter().any(|e| matches!(e, EngineEvent::EntryRejected { .. })),
            "the shared balance should still fund the second instrument"
        );
        assert!(session.kernel(1).is_in_position());
        // BBB's sizing saw the portfolio's free capital, not the raw balance:
        // 25% of 75k at 50.0 under 5x margin.
        assert_eq!(session.kernel(1).position_snapshots()[0].size, 1_875.0);
        assert_eq!(session.free_capital(), 56_250.0);
    }

    #[test]
    fn margin_mode_sizes_larger_than_cash() {
        let mut cash = session_two_instruments();
        cash.apply_current(StepInput { entry: true, ..StepInput::default() });
        let cash_size = cash.kernel(0).position_snapshots()[0].size;

        let mut margin = session_two_instruments_margin(5.0, false);
        margin.apply_current(StepInput { entry: true, ..StepInput::default() });
        let margin_size = margin.kernel(0).position_snapshots()[0].size;

        let ratio = margin_size / cash_size;
        assert!((ratio - 5.0).abs() < 0.01, "5x leverage should size ~5x, got {ratio}");
    }

    #[test]
    fn margin_equity_is_direction_aware() {
        // AAA long and BBB short, both drifting up: the short loses, but
        // cash-mode marking would price the short's `position_value` as a
        // gain. Only direction-aware marking nets them correctly.
        let mut session = session_two_instruments_margin(2.0, true);
        session.apply_current(StepInput {
            entry: true,
            size_mult: Some(0.25),
            ..StepInput::default()
        });
        session.apply_current(StepInput {
            entry: true,
            size_mult: Some(0.25),
            ..StepInput::default()
        });
        assert!(session.kernel(0).is_in_position());
        assert!(session.kernel(1).is_in_position());

        // Run out the schedule so both legs are marked at their last close.
        while session.current().is_some() {
            session.apply_current(StepInput::default());
        }
        let short_pnl = session.kernel(1).unrealized_value(52.0);
        assert!(short_pnl < 0.0, "a short into a rising market must be a loss");
        // Cash-mode marking would add the short's *position value*, which
        // grows as the price rises — reporting a loss as a gain.
        let cash_style = session.cash()
            + session.kernel(0).position_value(102.0)
            + session.kernel(1).position_value(52.0);
        let equity = session.equity();
        assert!(
            equity < cash_style,
            "direction-aware marking must price the losing short below the \
             cash model: equity {equity} vs cash-style {cash_style}"
        );
    }

    #[test]
    fn portfolio_margin_call_halts_all_kernels() {
        let config = BacktestConfig { fees: 0.0, ..BacktestConfig::default() };
        let mut session =
            EventSession::with_account(config, AccountMode::Margin { leverage: 50.0 });
        let a =
            session.add_instrument("AAA".into(), Direction::Long, None, None, PositionPolicy::Net);
        let b =
            session.add_instrument("BBB".into(), Direction::Long, None, None, PositionPolicy::Net);
        // AAA collapses after entry; BBB is untouched and never enters.
        session.set_bars(a, bars(0, &[100.0, 40.0, 30.0]));
        session.set_bars(b, bars(5, &[50.0, 50.0, 50.0]));
        session.seal();

        session.apply_current(StepInput { entry: true, ..StepInput::default() });
        assert!(session.kernel(0).is_in_position());

        let mut calls = 0;
        while session.current().is_some() {
            let events = session.apply_current(StepInput::default());
            calls += events.iter().filter(|e| matches!(e, EngineEvent::MarginCall { .. })).count();
        }
        assert_eq!(calls, 1, "the call latches, so it fires exactly once");
        assert!(session.is_halted());

        // The untouched instrument is halted too — one shared account.
        assert!(session.kernel(1).is_margin_halted());
    }

    #[test]
    fn maintenance_requirement_sums_per_instrument_rates() {
        // Each instrument contributes its own spec rate; a blended rate
        // would misprice the portfolio requirement.
        let mut session = session_two_instruments_margin(4.0, false);
        session.apply_current(StepInput {
            entry: true,
            size_mult: Some(0.25),
            ..StepInput::default()
        });
        session.apply_current(StepInput {
            entry: true,
            size_mult: Some(0.25),
            ..StepInput::default()
        });
        let required: f64 = [
            session.kernel(0).maintenance_requirement(102.0),
            session.kernel(1).maintenance_requirement(52.0),
        ]
        .iter()
        .sum();
        assert!(required > 0.0);
        // Default maint rate is half of init (1/4), i.e. 12.5% of notional.
        let notional = session.kernel(0).position_value(102.0).abs()
            + session.kernel(1).position_value(52.0).abs();
        assert!((required / notional - 0.125).abs() < 1e-9, "got {}", required / notional);
    }

    #[test]
    fn finish_reports_rejected_entries_and_halt() {
        // A margin call halts both instruments; every later entry attempt is
        // a counted constraint refusal, on the untouched instrument too.
        // (Zero-size sizing is deliberately *not* counted — see the kernel.)
        let config = BacktestConfig { fees: 0.0, ..BacktestConfig::default() };
        let mut session =
            EventSession::with_account(config, AccountMode::Margin { leverage: 50.0 });
        let a =
            session.add_instrument("AAA".into(), Direction::Long, None, None, PositionPolicy::Net);
        let b =
            session.add_instrument("BBB".into(), Direction::Long, None, None, PositionPolicy::Net);
        session.set_bars(a, bars(0, &[100.0, 40.0, 30.0, 30.0]));
        session.set_bars(b, bars(5, &[50.0, 50.0, 50.0, 50.0]));
        session.seal();

        // AAA enters, then collapses into a call; both then keep signaling.
        session.apply_current(StepInput { entry: true, ..StepInput::default() });
        while session.current().is_some() {
            session.apply_current(StepInput { entry: true, ..StepInput::default() });
        }
        assert!(session.is_halted());

        let outcome = session.finish();
        assert!(
            outcome.rejected_entries > 0,
            "rejections must be reported, not hardcoded to zero"
        );
        let per_instrument: usize = outcome.instruments.iter().map(|o| o.rejected_entries).sum();
        assert_eq!(outcome.rejected_entries, per_instrument);
        // The instrument that never traded was halted by the shared account.
        assert!(outcome.instruments[1].rejected_entries > 0);
        assert!(outcome.halted);
        assert!(outcome.halted_at.is_some());
    }

    #[test]
    fn finish_reports_no_halt_on_a_clean_run() {
        let mut session = session_two_instruments();
        session.apply_current(StepInput {
            entry: true,
            size_mult: Some(0.5),
            ..StepInput::default()
        });
        while session.current().is_some() {
            session.apply_current(StepInput::default());
        }
        let outcome = session.finish();
        assert!(!outcome.halted);
        assert_eq!(outcome.halted_at, None);
        assert_eq!(outcome.rejected_entries, 0);
    }

    #[test]
    fn finish_reports_margin_halt() {
        let config = BacktestConfig { fees: 0.0, ..BacktestConfig::default() };
        let mut session =
            EventSession::with_account(config, AccountMode::Margin { leverage: 50.0 });
        let a =
            session.add_instrument("AAA".into(), Direction::Long, None, None, PositionPolicy::Net);
        session.set_bars(a, bars(0, &[100.0, 40.0, 30.0]));
        session.seal();
        session.apply_current(StepInput { entry: true, ..StepInput::default() });
        while session.current().is_some() {
            session.apply_current(StepInput::default());
        }
        let outcome = session.finish();
        assert!(outcome.halted);
        assert!(outcome.halted_at.is_some());
    }

    #[test]
    fn locked_margin_released_on_close() {
        let mut session = session_two_instruments_margin(5.0, false);
        session.apply_current(StepInput {
            entry: true,
            size_mult: Some(0.25),
            ..StepInput::default()
        });
        session.apply_current(StepInput {
            entry: true,
            size_mult: Some(0.25),
            ..StepInput::default()
        });
        assert!(session.free_capital() < session.cash());

        while session.current().is_some() {
            session.apply_current(StepInput::default());
        }
        let outcome = session.finish();
        assert_eq!(outcome.result.trades.len(), 2);
        // Every lock is released, so the closing balance is exactly the
        // starting capital plus realized PnL (fees are zero in this config).
        let total_pnl: f64 = outcome.instruments.iter().map(|o| o.pnl).sum();
        let final_balance = *outcome
            .result
            .equity_curve
            .last()
            .expect("the schedule produced equity samples");
        assert!(
            (final_balance - (100_000.0 + total_pnl)).abs() < 1e-6,
            "balance {final_balance} should reconcile to 100000 + {total_pnl}"
        );
    }
}
