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

use crate::accounts::AccountMode;
use crate::core::error::RaptorError;
use crate::core::types::OhlcvBar;
use crate::core::types::TickData;
use crate::core::types::{BacktestConfig, BacktestResult, Direction, InstrumentConfig, Timestamp};
use crate::core::Result as RaptorResult;
use crate::data::{
    ensure_time_sorted, tick_data_to_events, DepthRef, DepthTick, EventFeed, EventPayload,
    MarketEvent, QuoteTick, TradeTick,
};
use crate::instruments::InstrumentSpec;
use crate::portfolio::book::PortfolioBook;
use crate::portfolio::kernel::{EngineEvent, EngineKernel, KernelBar, StepInput};
use crate::portfolio::ledger::PositionPolicy;
// Used by `session_tests.rs` via `use super::*`.
#[cfg(test)]
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

fn data_timestamp(data: &ScheduleData) -> i64 {
    match data {
        ScheduleData::Bar(bar) => bar.timestamp,
        ScheduleData::Trade(t) => t.timestamp,
        ScheduleData::Quote(q) => q.timestamp,
        ScheduleData::Depth(d) => d.timestamp,
    }
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
        data_timestamp(&self.data)
    }
}

/// Per-instrument outcome summary.
#[derive(Debug, Clone)]
pub struct InstrumentOutcome {
    pub symbol: String,
    pub trades: usize,
    pub pnl: f64,
    pub rejected_entries: usize,
    /// Entries the account could not fund (see `RejectReason::is_unfunded`).
    pub unfunded_entries: usize,
}

/// Everything a finished session reports.
#[derive(Debug)]
pub struct SessionOutcome {
    pub result: BacktestResult,
    pub instruments: Vec<InstrumentOutcome>,
    /// Entries refused across all instruments, summed over their risk gates.
    pub rejected_entries: usize,
    /// Entries refused for want of capital or margin, or sized to nothing,
    /// across all instruments. Not in `rejected_entries`, which counts risk
    /// limits.
    pub unfunded_entries: usize,
    /// Whether a margin call or a drawdown kill-switch latched.
    pub halted: bool,
    /// Where the halt latched, as a **schedule-event ordinal** — the session
    /// interleaves N streams, so this is not a bar index (the array runner's
    /// `halted_at` is).
    pub halted_at: Option<usize>,
}

/// Multi-instrument session over merged bar streams: the market tape plus
/// one [`PortfolioBook`] that trades it.
pub struct EventSession {
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
    /// Latest event timestamp per instrument, so a live push can refuse one
    /// that goes backwards.
    last_ts: Vec<i64>,
    cursor: usize,
    book: PortfolioBook,
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
        Self {
            symbols: Vec::new(),
            bars: Vec::new(),
            ticks: Vec::new(),
            depth: Vec::new(),
            depth_input: Vec::new(),
            schedule: Vec::new(),
            local_idx_next: Vec::new(),
            last_ts: Vec::new(),
            cursor: 0,
            book: PortfolioBook::new(config, mode),
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
        let index = self.book.add_instrument(symbol.clone(), direction, spec, inst_config, policy);
        self.symbols.push(symbol);
        self.bars.push(Vec::new());
        self.ticks.push(None);
        self.depth_input.push(Vec::new());
        self.local_idx_next.push(0);
        self.last_ts.push(i64::MIN);
        index
    }

    /// Attach an instrument's bar series (ascending timestamps).
    ///
    /// Errors, attaching nothing, if a timestamp decreases. Equal timestamps
    /// are allowed.
    pub fn set_bars(&mut self, instrument: usize, bars: Vec<KernelBar>) -> RaptorResult<()> {
        ensure_time_sorted(
            &format!("{} bars", self.symbols[instrument]),
            bars.iter().map(|b| b.timestamp),
        )?;
        self.bars[instrument] = bars;
        Ok(())
    }

    /// Attach an instrument's tick series (ascending timestamps).
    ///
    /// Trades and quotes are merged into the schedule alongside any bars,
    /// ordered by timestamp then phase — a trade precedes the quote of the
    /// same feed row, and both precede a bar closing at that timestamp.
    ///
    /// Errors, attaching nothing, if a timestamp decreases.
    pub fn set_ticks(&mut self, instrument: usize, ticks: TickData) -> RaptorResult<()> {
        ensure_time_sorted(
            &format!("{} ticks", self.symbols[instrument]),
            ticks.timestamps.iter().copied(),
        )?;
        self.ticks[instrument] = Some(ticks);
        Ok(())
    }

    /// Attach an instrument's depth snapshots (ascending timestamps).
    ///
    /// Book updates are observation only: like quotes, they never fill an
    /// order or mark equity. They do inform later fills, by sizing the
    /// queue a resting limit joins.
    ///
    /// Errors, attaching nothing, if a timestamp decreases.
    pub fn set_depth(&mut self, instrument: usize, depth: Vec<DepthTick>) -> RaptorResult<()> {
        ensure_time_sorted(
            &format!("{} depth", self.symbols[instrument]),
            depth.iter().map(|d| d.timestamp),
        )?;
        self.depth_input[instrument] = depth;
        Ok(())
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
            self.last_ts[instrument] = self.last_ts[instrument].max(data_timestamp(&data));
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
        self.last_ts[instrument] = data_timestamp(&data);
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
        ltq: f64,
        bid_qty: f64,
        ask_qty: f64,
        oi: f64,
    ) -> RaptorResult<usize> {
        self.check_live_timestamp(instrument, timestamp)?;
        let mut appended = 0;
        if ltp > 0.0 {
            // The exchange's last traded quantity when supplied, else the
            // flow-delta proxy — the same rule as `tick_data_to_events`.
            let size = if ltq > 0.0 { ltq } else { buy_qty_delta.abs() + sell_qty_delta.abs() };
            let signed_size = buy_qty_delta.abs() - sell_qty_delta.abs();
            self.push_entry(
                instrument,
                ScheduleData::Trade(TradeTick { timestamp, price: ltp, size, signed_size, oi }),
            );
            appended += 1;
        }
        if bid > 0.0 && ask > 0.0 {
            self.push_entry(
                instrument,
                ScheduleData::Quote(QuoteTick {
                    timestamp,
                    bid,
                    ask,
                    bid_size: TickData::displayed(bid_qty),
                    ask_size: TickData::displayed(ask_qty),
                }),
            );
            appended += 1;
        }
        Ok(appended)
    }

    /// Append a live bar to the schedule tail.
    pub fn push_bar(&mut self, instrument: usize, bar: KernelBar) -> RaptorResult<()> {
        self.check_live_timestamp(instrument, bar.timestamp)?;
        self.push_entry(instrument, ScheduleData::Bar(bar));
        Ok(())
    }

    /// Append a live depth snapshot to the schedule tail.
    pub fn push_depth(&mut self, instrument: usize, snapshot: DepthTick) -> RaptorResult<()> {
        self.check_live_timestamp(instrument, snapshot.timestamp)?;
        let slot = self.depth.len() as u32;
        let timestamp = snapshot.timestamp;
        self.depth.push(snapshot);
        self.push_entry(instrument, ScheduleData::Depth(DepthRef { slot, timestamp }));
        Ok(())
    }

    /// Refuse a live event older than the instrument's latest one (batch data
    /// attached earlier counts). Equal timestamps are allowed.
    fn check_live_timestamp(&mut self, instrument: usize, timestamp: i64) -> RaptorResult<()> {
        self.seal();
        let last = self.last_ts[instrument];
        if timestamp < last {
            return Err(RaptorError::invalid_parameter(format!(
                "{} live feed: timestamps must not decrease, but event {} is {timestamp} \
                 after {last}",
                self.symbols[instrument], self.local_idx_next[instrument],
            )));
        }
        Ok(())
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

    /// The `submitted_idx` an order placed during the current event should
    /// carry on `instrument`'s own clock; see [`PortfolioBook::submission_idx`].
    pub fn submission_idx(&self, instrument: usize, event_idx: usize) -> usize {
        self.book.submission_idx(instrument, event_idx, self.current().map(|e| e.instrument))
    }

    /// Kernel of an instrument, for order routing and queries.
    pub fn kernel_mut(&mut self, instrument: usize) -> &mut EngineKernel {
        self.book.kernel_mut(instrument)
    }

    pub fn kernel(&self, instrument: usize) -> &EngineKernel {
        self.book.kernel(instrument)
    }

    /// Portfolio equity; see [`PortfolioBook::equity`].
    pub fn equity(&self) -> f64 {
        self.book.equity()
    }

    /// Shared cash balance. In margin mode this includes locked initial
    /// margin; see [`EventSession::free_capital`] for what can fund a new
    /// position.
    pub fn cash(&self) -> f64 {
        self.book.cash()
    }

    /// Adopt a pre-existing position on one instrument; see
    /// [`PortfolioBook::adopt_position`].
    pub fn adopt_position(
        &mut self,
        instrument: usize,
        timestamp: Timestamp,
        price: f64,
        size: f64,
    ) -> Result<u64, String> {
        self.book.adopt_position(instrument, timestamp, price, size)
    }

    /// Give the session the corporate actions of the run and the adjustment
    /// policy its history is on; see [`PortfolioBook::set_corporate_actions`].
    pub fn set_corporate_actions(
        &mut self,
        actions: &[crate::instruments::CorporateAction],
        policy: crate::instruments::AdjustmentPolicy,
    ) -> crate::core::Result<()> {
        self.book.set_corporate_actions(actions, policy)
    }

    /// Link working orders on distinct instruments into one atomic group:
    /// they fill together or not at all; see [`PortfolioBook::link_group`].
    pub fn link_group(&mut self, legs: &[(usize, u64)]) -> Result<u64, String> {
        self.book.link_group(legs)
    }

    /// The registered symbols, in instrument order.
    pub fn symbols(&self) -> &[String] {
        &self.symbols
    }

    /// Dividend cash credited so far.
    pub fn dividends_received(&self) -> f64 {
        self.book.dividends_received()
    }

    /// Capital available to open new positions across all instruments.
    pub fn free_capital(&self) -> f64 {
        self.book.free_capital()
    }

    /// Whether a margin call or drawdown kill-switch has latched.
    pub fn is_halted(&self) -> bool {
        self.book.is_halted()
    }

    /// Halt the book; see [`PortfolioBook::halt`].
    pub fn halt(&mut self, reason: &str, flatten: bool) {
        self.book.halt(reason, flatten);
    }

    /// Step the current schedule entry through the book and advance. See
    /// [`PortfolioBook::apply`].
    pub fn apply_current(&mut self, input: StepInput) -> Vec<EngineEvent> {
        let Some(entry) = self.current() else { return Vec::new() };
        let depth = match entry.data {
            ScheduleData::Depth(handle) => self.depth_at(handle.slot),
            _ => None,
        };
        let events = self.book.apply(&entry, depth, input);
        self.cursor += 1;
        events
    }

    /// Force-close every instrument at its last seen bar and compute
    /// portfolio metrics.
    pub fn finish(self) -> SessionOutcome {
        self.book.finish()
    }

    /// The merged schedule, applied or not.
    pub fn entries(&self) -> &[ScheduleEntry] {
        &self.schedule
    }
}

#[cfg(test)]
#[path = "session_tests.rs"]
mod tests;
