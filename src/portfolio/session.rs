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
//!
//! Risk gating is portfolio-wide: `max_positions` counts open positions
//! across every instrument (injected into each kernel's gate before it
//! steps, so the resting-order path is covered too), and the drawdown
//! kill-switch trips on portfolio equity and blocks entries everywhere.
//! Capital *allocation* is not: each kernel is offered the whole free
//! balance, so the strategy owns sizing via `size_frac`. The array runner's
//! `EqualWeight` budget has no counterpart here yet.

use crate::accounts::{AccountMode, SharedAccount};
use crate::core::types::{BacktestConfig, BacktestResult, Direction, InstrumentConfig, Trade};
use crate::data::{EventFeed, EventPayload, MarketEvent};
use crate::instruments::InstrumentSpec;
use crate::metrics::streaming::StreamingMetrics;
use crate::portfolio::engine::compute_backtest_metrics_with_config;
use crate::portfolio::kernel::{EngineEvent, EngineKernel, KernelBar, StepInput};
use crate::portfolio::ledger::PositionPolicy;
use crate::portfolio::risk::RejectReason;
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

/// Why a portfolio-wide halt latched.
#[derive(Debug, Clone, Copy)]
enum HaltCause {
    /// Equity fell below the summed maintenance requirement.
    MarginCall,
    /// The drawdown kill-switch tripped on portfolio equity.
    Drawdown,
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
    ///
    /// `max_positions` is counted across every instrument, so the kernel's
    /// gate refuses an entry once the *portfolio* is full — on the resting
    /// order path too, which is why the count is injected rather than
    /// pre-checked here. The count is snapshotted before the step, mirroring
    /// the array runner: an instrument that exits and re-enters on the same
    /// bar is still counted as holding its outgoing position.
    pub fn apply_current(&mut self, input: StepInput) -> Vec<EngineEvent> {
        let Some(entry) = self.current() else { return Vec::new() };
        let instrument = entry.instrument;

        // Portfolio-wide open count, skipped entirely when no limit is set.
        // Summing every kernel's ledger covers hedging policies, where one
        // instrument can hold several positions at once.
        let portfolio_open = self
            .config
            .max_positions
            .map(|_| self.kernels.iter().map(|k| k.open_count()).sum::<usize>());

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
        kernel.set_external_open_count(portfolio_open);
        let mut events = kernel.step(entry.local_idx, &entry.bar, input);
        let delta_cash = kernel.cash() - injected;
        let delta_locked = kernel.locked_margin() - locked_before;
        kernel.set_cash(0.0);
        kernel.set_external_open_count(None);
        self.account.reconcile(delta_cash, delta_locked);

        // A kernel-local call sees only its own slice of the portfolio, but
        // the account is shared: escalate it so every instrument halts.
        if events.iter().any(|e| matches!(e, EngineEvent::MarginCall { .. })) {
            self.halt_all(self.cursor, HaltCause::MarginCall);
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
                self.halt_all(self.cursor, HaltCause::MarginCall);
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
        if !risk_halted_before && self.kernels.iter().any(|k| k.risk_halted()) {
            self.halt_all(self.cursor, HaltCause::Drawdown);
        }

        self.cursor += 1;
        events
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
            for kernel in &mut self.kernels {
                kernel.halt_margin();
            }
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
        let halted_at = self.account.halted_at();

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
#[path = "session_tests.rs"]
mod tests;
