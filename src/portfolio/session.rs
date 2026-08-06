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
use crate::core::types::OhlcvBar;
use crate::core::types::TickData;
use crate::core::types::{
    BacktestConfig, BacktestResult, Direction, InstrumentConfig, Timestamp, Trade,
};
use crate::data::{
    tick_data_to_events, DepthRef, DepthTick, EventFeed, EventPayload, MarketEvent, QuoteTick,
    TradeTick,
};
use crate::instruments::InstrumentSpec;
use crate::metrics::streaming::StreamingMetrics;
use crate::portfolio::engine::compute_backtest_metrics_with_config;
use crate::portfolio::kernel::{EngineEvent, EngineKernel, KernelBar, StepInput};
use crate::portfolio::ledger::PositionPolicy;
use crate::portfolio::risk::RejectReason;

/// The market data one schedule entry carries.
#[derive(Debug, Clone, Copy)]
pub enum ScheduleData {
    Bar(KernelBar),
    Trade(TradeTick),
    Quote(QuoteTick),
    /// A depth snapshot, held in the session's store; the handle keeps
    /// schedule entries small and `Copy`.
    Depth(DepthRef),
}

/// One entry of the merged schedule.
///
/// `local_idx` is a per-instrument event ordinal, not a bar index: in a
/// tick session it advances on every event of that instrument. Order
/// matching keys off it (an order cannot rest into the event it was
/// submitted on), so it must stay monotone per instrument.
#[derive(Debug, Clone, Copy)]
pub struct ScheduleEntry {
    pub instrument: usize,
    pub local_idx: usize,
    pub data: ScheduleData,
}

impl ScheduleEntry {
    /// The bar this entry carries, or the degenerate bar of a trade print.
    /// `None` for quotes, which have no traded price.
    pub fn as_bar(&self) -> Option<KernelBar> {
        match self.data {
            ScheduleData::Bar(bar) => Some(bar),
            ScheduleData::Trade(t) => Some(KernelBar {
                timestamp: t.timestamp,
                open: t.price,
                high: t.price,
                low: t.price,
                close: t.price,
                volume: t.size,
            }),
            ScheduleData::Quote(_) | ScheduleData::Depth(_) => None,
        }
    }

    /// Event timestamp, whatever the payload.
    pub fn timestamp(&self) -> i64 {
        match self.data {
            ScheduleData::Bar(bar) => bar.timestamp,
            ScheduleData::Trade(t) => t.timestamp,
            ScheduleData::Quote(q) => q.timestamp,
            ScheduleData::Depth(d) => d.timestamp,
        }
    }
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
    ticks: Vec<Option<TickData>>,
    /// Depth snapshots, referenced by slot from the schedule.
    depth: Vec<DepthTick>,
    /// Pending per-instrument depth, merged at seal.
    depth_input: Vec<Vec<DepthTick>>,
    schedule: Vec<ScheduleEntry>,
    /// Next per-instrument event ordinal, shared by the batch merge and the
    /// streaming pushes so `local_idx` stays monotone across both.
    local_idx_next: Vec<usize>,
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
            ticks: Vec::new(),
            depth: Vec::new(),
            depth_input: Vec::new(),
            schedule: Vec::new(),
            local_idx_next: Vec::new(),
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
        self.ticks.push(None);
        self.depth_input.push(Vec::new());
        self.local_idx_next.push(0);
        self.last_close.push(None);
        self.last_seen.push(None);
        self.kernels.len() - 1
    }

    /// Attach an instrument's bar series (ascending timestamps).
    pub fn set_bars(&mut self, instrument: usize, bars: Vec<KernelBar>) {
        self.bars[instrument] = bars;
    }

    /// Attach an instrument's tick series (ascending timestamps).
    ///
    /// Trades and quotes are merged into the schedule alongside any bars,
    /// ordered by timestamp then phase — a trade precedes the quote of the
    /// same feed row, and both precede a bar closing at that timestamp.
    pub fn set_ticks(&mut self, instrument: usize, ticks: TickData) {
        self.ticks[instrument] = Some(ticks);
    }

    /// Attach an instrument's depth snapshots (ascending timestamps).
    ///
    /// Book updates are observation only: like quotes, they never fill an
    /// order or mark equity. They do inform later fills, by sizing the
    /// queue a resting limit joins.
    pub fn set_depth(&mut self, instrument: usize, depth: Vec<DepthTick>) {
        self.depth_input[instrument] = depth;
    }

    /// A depth snapshot by slot.
    pub fn depth_at(&self, slot: u32) -> Option<DepthTick> {
        self.depth.get(slot as usize).copied()
    }

    /// Merge all streams into the deterministic schedule. Idempotent.
    pub fn seal(&mut self) {
        if self.sealed {
            return;
        }
        let mut feed = EventFeed::new();
        // Stream ids must be globally unique for the merge's tiebreak, so
        // hand them out from one counter rather than reusing the instrument
        // index (a tick instrument needs three).
        let mut next_stream = 0u32;
        for (i, bars) in self.bars.iter().enumerate() {
            let stream = next_stream;
            next_stream += 1;
            let events: Vec<MarketEvent> = bars
                .iter()
                .map(|b| MarketEvent {
                    instrument: i as u32,
                    stream,
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
        for (i, ticks) in self.ticks.iter().enumerate() {
            let Some(ticks) = ticks else { continue };
            let trade_stream = next_stream;
            let quote_stream = next_stream + 1;
            next_stream += 2;
            let events = tick_data_to_events(ticks, i as u32, trade_stream, quote_stream);
            // One conversion emits both kinds interleaved; the feed needs
            // each stream monotone, so split them back apart.
            let trades: Vec<MarketEvent> = events
                .iter()
                .filter(|e| matches!(e.payload, EventPayload::Trade(_)))
                .copied()
                .collect();
            let quotes: Vec<MarketEvent> = events
                .iter()
                .filter(|e| matches!(e.payload, EventPayload::Quote(_)))
                .copied()
                .collect();
            feed.add_stream(trades);
            feed.add_stream(quotes);
        }
        for i in 0..self.depth_input.len() {
            if self.depth_input[i].is_empty() {
                continue;
            }
            let stream = next_stream;
            next_stream += 1;
            let snapshots = std::mem::take(&mut self.depth_input[i]);
            let events: Vec<MarketEvent> = snapshots
                .into_iter()
                .map(|snapshot| {
                    let slot = self.depth.len() as u32;
                    let timestamp = snapshot.timestamp;
                    self.depth.push(snapshot);
                    MarketEvent {
                        instrument: i as u32,
                        stream,
                        payload: EventPayload::Depth(DepthRef { slot, timestamp }),
                    }
                })
                .collect();
            feed.add_stream(events);
        }
        for event in feed {
            let instrument = event.instrument as usize;
            let data = match event.payload {
                EventPayload::Bar(bar) => ScheduleData::Bar(KernelBar {
                    timestamp: bar.timestamp,
                    open: bar.open,
                    high: bar.high,
                    low: bar.low,
                    close: bar.close,
                    volume: bar.volume,
                }),
                EventPayload::Trade(t) => ScheduleData::Trade(t),
                EventPayload::Quote(q) => ScheduleData::Quote(q),
                EventPayload::Depth(d) => ScheduleData::Depth(d),
            };
            let local_idx = self.local_idx_next[instrument];
            self.local_idx_next[instrument] += 1;
            self.schedule.push(ScheduleEntry { instrument, local_idx, data });
        }
        self.sealed = true;
    }

    /// Append one entry to the schedule tail with the next local ordinal.
    ///
    /// Seals first (idempotent), so any batch data attached beforehand —
    /// warmup bars, replayed history — merges ahead of everything pushed.
    /// Pushed events land in arrival order: in a live session, arrival
    /// order *is* the schedule.
    fn push_entry(&mut self, instrument: usize, data: ScheduleData) {
        self.seal();
        let local_idx = self.local_idx_next[instrument];
        self.local_idx_next[instrument] += 1;
        self.schedule.push(ScheduleEntry { instrument, local_idx, data });
    }

    /// Append a live feed row: a trade print when `ltp > 0`, then a quote
    /// when both sides of the book are present — the same split
    /// [`tick_data_to_events`] applies to batch arrays. Returns how many
    /// events were appended (0..=2).
    #[allow(clippy::too_many_arguments)]
    pub fn push_tick(
        &mut self,
        instrument: usize,
        timestamp: i64,
        ltp: f64,
        bid: f64,
        ask: f64,
        buy_qty_delta: f64,
        sell_qty_delta: f64,
    ) -> usize {
        let mut appended = 0;
        if ltp > 0.0 {
            let size = buy_qty_delta.abs() + sell_qty_delta.abs();
            let signed_size = buy_qty_delta.abs() - sell_qty_delta.abs();
            self.push_entry(
                instrument,
                ScheduleData::Trade(TradeTick { timestamp, price: ltp, size, signed_size }),
            );
            appended += 1;
        }
        if bid > 0.0 && ask > 0.0 {
            self.push_entry(instrument, ScheduleData::Quote(QuoteTick { timestamp, bid, ask }));
            appended += 1;
        }
        appended
    }

    /// Append a live bar to the schedule tail.
    pub fn push_bar(&mut self, instrument: usize, bar: KernelBar) {
        self.push_entry(instrument, ScheduleData::Bar(bar));
    }

    /// Append a live depth snapshot to the schedule tail.
    pub fn push_depth(&mut self, instrument: usize, snapshot: DepthTick) {
        let slot = self.depth.len() as u32;
        let timestamp = snapshot.timestamp;
        self.depth.push(snapshot);
        self.push_entry(instrument, ScheduleData::Depth(DepthRef { slot, timestamp }));
    }

    /// Events pushed or merged but not yet applied.
    pub fn remaining(&self) -> usize {
        self.schedule.len() - self.cursor
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

    /// Adopt a pre-existing position on one instrument (broker-truth
    /// seeding) — see [`EngineKernel::adopt_position`], which owns the
    /// account-mode rules. Same lend/drain pool discipline as
    /// [`Self::apply_current`], so the cost basis comes out of the shared
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
        if instrument >= self.kernels.len() {
            return Err(format!("unknown instrument index {instrument}"));
        }
        if !self.equity_curve.is_empty() {
            return Err(
                "adopt_position must be called before the first applied event".to_string()
            );
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
        kernel.set_cash(injected);
        let result = kernel.adopt_position(timestamp, price, size);
        let delta_cash = kernel.cash() - injected;
        // Carry the locked delta too. Leaving this at 0.0 would mean the
        // shared account never learns about the adopted margin, so portfolio
        // free capital would read high by the whole cost basis — a risk
        // constraint silently weakened.
        let delta_locked = kernel.locked_margin() - locked_before;
        kernel.set_cash(0.0);
        self.account.reconcile(delta_cash, delta_locked);
        result
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
        // Copy the snapshot out before borrowing the kernel mutably.
        let depth_snapshot = match entry.data {
            ScheduleData::Depth(handle) => self.depth_at(handle.slot),
            _ => None,
        };

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
        let mut events = match entry.data {
            ScheduleData::Bar(bar) => kernel.step(entry.local_idx, &bar, input),
            ScheduleData::Trade(tick) => kernel.step_trade(entry.local_idx, &tick, input),
            ScheduleData::Quote(quote) => kernel.step_quote(&quote),
            ScheduleData::Depth(_) => match depth_snapshot {
                Some(snapshot) => kernel.step_depth(&snapshot),
                None => Vec::new(),
            },
        };
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
            self.cursor += 1;
            return events;
        }

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
        self.timestamps.push(entry.timestamp());

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
                if self.config.liquidate_on_margin_call {
                    // Every instrument liquidates at its own last mark, and
                    // the cash it returns flows back through the shared
                    // account like any other close.
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
                        kernel.set_cash(injected);
                        let closed = kernel.liquidate_all(last_idx, &last_bar);
                        let delta_cash = kernel.cash() - injected;
                        let delta_locked = kernel.locked_margin() - locked_before;
                        kernel.set_cash(0.0);
                        self.account.reconcile(delta_cash, delta_locked);
                        for event in &closed {
                            if let EngineEvent::Exited { trade, .. } = event {
                                self.streaming.update(trade.return_pct / 100.0);
                                self.trades.push(trade.clone());
                            }
                        }
                        events.extend(closed);
                    }
                }
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
